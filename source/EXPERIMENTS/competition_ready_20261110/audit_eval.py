"""Final TRAIN audit split only; never contributes supervision or gradients."""
from pathlib import Path
import json, sys, tempfile, shutil


def score_saved(expected_ids, rows, gold_loader, matches):
    if not expected_ids or len(expected_ids) != len(set(expected_ids)):
        raise ValueError('Invalid frozen audit IDs')
    if [r.get('id') for r in rows] != expected_ids:
        raise ValueError('Missing, duplicate or reordered predictions; gold remains closed')
    gold = gold_loader()
    if set(gold) != set(expected_ids):
        raise ValueError('Wrong gold split')
    verdicts = {r['id']:bool(matches(r, gold[r['id']])) for r in rows}
    correct = sum(verdicts.values())
    abstain = sum(r.get('type') == 'abstain' for r in rows)
    wrong = len(rows)-correct-abstain
    if wrong < 0:
        raise ValueError('Invalid score counts')
    return {'verdicts':verdicts,'correct':correct,'wrong':wrong,'abstain':abstain,
            'n':len(rows),'metric':(correct-wrong)/len(rows)}


def main(config_path, adapter, output):
    c = json.loads(Path(config_path).read_text())
    if c.get('mode') != 'evaluation' or c.get('split') != 'audit_exposed' or c.get('run_public') is not False:
        raise ValueError('Explicit final audit only')
    source = Path(c['source_root'])
    sys.path[:0] = [str(source), str(source/'baseline/app')]
    from relearning import sha, atomic
    from scoring import matches
    runtime = Path(c['runtime_root'])
    audit = Path(c['audit_root'])
    questions = audit/'questions.jsonl'
    labels = audit/'gold.jsonl'
    if sha(questions) != c['audit_questions_sha256'] or sha(runtime/'SPLIT_MANIFEST.json') != c['split_manifest_sha256']:
        raise ValueError('Frozen audit input changed')
    qs = [json.loads(line) for line in questions.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    split = json.loads((runtime/'SPLIT_MANIFEST.json').read_text())
    allowed = set(split['groups']['audit_exposed'])
    if allowed & set(split['groups']['learning']) or any(q['business_id'] not in allowed for q in qs) or len(qs) != 631:
        raise ValueError('Wrong audit business split')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory() as scratch:
        p = Path(scratch)
        (p/'assets').mkdir()
        (p/'assets/base').symlink_to(Path(c['base']), target_is_directory=True)
        (p/'app').symlink_to(source/'baseline/app', target_is_directory=True)
        (p/'candidate').mkdir()
        (p/'candidate/final').symlink_to(Path(adapter), target_is_directory=True)
        shutil.copy2(source/'baseline/model_lock.json', p/'model_lock.json')
        import public_infer_811
        sys.argv = ['inference','--project',str(p),'--root',str(runtime),'--questions',str(questions),
            '--variant','candidate','--output-dir',str(out/'predictions'),'--batch-size','4',
            '--cache-dir',str(p/'cache'),'--max-input-tokens','3072']
        public_infer_811.main()
    prediction_path = out/'predictions/submission.jsonl'
    rows = [json.loads(line) for line in prediction_path.read_text().splitlines() if line.strip()]
    rt = json.loads((out/'predictions/runtime.json').read_text())
    if rt['state'] != 'completed' or rt['question_logs'] != len(qs) or rt['network_attempts'] != 0:
        raise ValueError('Incomplete or non-offline inference; gold remains closed')
    def gold_loader():
        if sha(labels) != c['audit_gold_sha256']:
            raise ValueError('Audit gold fingerprint changed')
        return {r['id']:r for r in map(json.loads, labels.read_text().splitlines())}
    result = score_saved([q['id'] for q in qs], rows, gold_loader, matches)
    result.update(manifest_sha256=c['audit_manifest_sha256'],
        decoder_sha256=sha(source/'public_infer_811.py'),scorer_sha256=sha(source/'baseline/app/scoring.py'),
        prediction_sha256=sha(prediction_path),gates_passed=True,weights_updated=False,
        gold_after_predictions=True,exposure='historically exposed audit businesses; no new supervision')
    atomic(out/'EVALUATION.json', result)


if __name__ == '__main__':
    main(*sys.argv[1:])
