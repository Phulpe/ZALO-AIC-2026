"""Paired evaluation only; existing locked evaluator persists predictions first."""
import time
from relearning import verify_export, sha, atomic, seal
from quality_gate import promotion
from isolated_training import run_stage

out = Path('/kaggle/working/paired_calibration')
out.mkdir()
started = time.monotonic()
try:
    candidates = []
    for path in Path('/kaggle/input').rglob('WORKFLOW_COMPLETED.json'):
        data = json.loads(path.read_text())
        if data.get('candidate_sha256') == CONFIG['candidate_sha256']:
            candidates.append((path.parent, data))
    assert len(candidates) == 1
    pilot, receipt = candidates[0]
    pilot_manifest = verify_export(pilot)
    assert pilot_manifest == CONFIG['pilot_manifest_sha256']
    decision = json.loads((pilot/'REGRESSION_DECISION.json').read_text())
    assert decision['eligible_for_calibration'] is True
    assert decision['full']['correct_to_wrong'] == 0 and decision['old7']['correction_after_correct'] > 0
    candidate = pilot/'model_runs'/receipt['run_id']/'checkpoint'
    assert sha(candidate/'adapter_model.safetensors') == CONFIG['candidate_sha256']
    bases = list(Path('/kaggle/input').rglob('BASE_UPLOAD_MANIFEST.json'))
    runtimes = list(Path('/kaggle/input').rglob('RUNTIME_DATA_MANIFEST.json'))
    parents = list(Path('/kaggle/input').rglob('final/adapter_model.safetensors'))
    assert len(bases) == len(runtimes) == len(parents) == 1
    CONFIG.update(base=str(bases[0].parent/'base'), runtime_root=str(runtimes[0].parent))
    assert sha(runtimes[0]) == CONFIG['runtime_manifest_sha256']
    assert sha(parents[0]) == CONFIG['parent_sha256']
    for name, digest in CONFIG['base_hashes'].items():
        assert sha(Path(CONFIG['base'])/name) == digest
    runtime = runtimes[0].parent
    for file, key in [('DATA_MANIFEST.json', 'data_manifest_sha256'), ('SPLIT_MANIFEST.json', 'split_manifest_sha256'),
                      ('calibration/questions.jsonl', 'calibration_questions_sha256'), ('calibration/gold.jsonl', 'calibration_gold_sha256')]:
        assert sha(runtime/file) == CONFIG[key]
    request = out/'LOCKED_EVAL_CONFIG.json'
    atomic(request, CONFIG)
    atomic(out/'INPUT_VERIFICATION.json', {'state':'passed','pilot_manifest_sha256':pilot_manifest,
        'candidate_sha256':CONFIG['candidate_sha256'],'source_archive_sha256':CONFIG['source_archive_sha256'],
        'weights_updated':False,'public_test_enabled':False})
    if CONFIG.get('prepare_only'):
        import torch
        assert not torch.cuda.is_available()
        atomic(out/'CPU_PREFLIGHT.json', {'state':'passed','GPU_executed':False})
    else:
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES':'0'}
        for name, adapter in [('parent', parents[0].parent), ('candidate', candidate)]:
            atomic(out/'PHASE.json', {'phase':name,'elapsed_seconds':time.monotonic()-started})
            print('CALIBRATION_PHASE', name, flush=True)
            run_stage([sys.executable, str(root/'locked_eval.py'), str(request), str(adapter), str(out/name)],
                out/(name+'_process'), min(2700, int(5700-(time.monotonic()-started))), env)
        before = json.loads((out/'parent/EVALUATION.json').read_text())
        after = json.loads((out/'candidate/EVALUATION.json').read_text())
        result = promotion(before, after, CONFIG['promotion_policy'])
        summary = {'state':'completed','candidate_sha256':CONFIG['candidate_sha256'],
            'source_archive_sha256':CONFIG['source_archive_sha256'],'decision':result,
            'before':{k:before[k] for k in ['n','correct','wrong','abstain','metric']},
            'after':{k:after[k] for k in ['n','correct','wrong','abstain','metric']},
            'weights_updated':False,'public_ran':False,'BEST_MODEL_updated':False,
            'exposure':'historically exposed calibration; not untouched holdout',
            'elapsed_seconds':time.monotonic()-started}
        atomic(out/'PAIRED_CALIBRATION_RESULT.json', summary)
        print('PAIRED_CALIBRATION_RESULT', json.dumps(summary), flush=True)
except Exception as exc:
    atomic(out/'FAILURE.json', {'exception_type':type(exc).__name__,'partial_outputs_preserved':True})
    raise
finally:
    seal(out)
