"""Continue immutable inference journal, never resume training or edit its fingerprint."""
import shutil, time
from relearning import verify_export, sha, atomic, seal
from quality_gate import promotion

out = Path('/kaggle/working/calibration_recovery')
out.mkdir()
started = time.monotonic()
try:
    snapshots = [p.parent for p in Path('/kaggle/input').rglob('ARTIFACT_MANIFEST.json')
                 if sha(p) == CONFIG['partial_manifest_sha256']]
    assert len(snapshots) == 1
    prior = snapshots[0]
    assert verify_export(prior) == CONFIG['partial_manifest_sha256']
    pilot_manifests = [p for p in Path('/kaggle/input').rglob('ARTIFACT_MANIFEST.json')
                       if sha(p) == CONFIG['pilot_manifest_sha256']]
    assert len(pilot_manifests) == 1
    pilot = pilot_manifests[0].parent
    assert verify_export(pilot) == CONFIG['pilot_manifest_sha256']
    candidate = pilot/'model_runs'/CONFIG['pilot_run_id']/'checkpoint'
    assert sha(candidate/'adapter_model.safetensors') == CONFIG['candidate_sha256']
    bases = list(Path('/kaggle/input').rglob('BASE_UPLOAD_MANIFEST.json'))
    runtimes = list(Path('/kaggle/input').rglob('RUNTIME_DATA_MANIFEST.json'))
    assert len(bases) == len(runtimes) == 1
    base = bases[0].parent/'base'
    runtime = runtimes[0].parent
    assert sha(runtimes[0]) == CONFIG['runtime_manifest_sha256']
    questions = runtime/'calibration/questions.jsonl'
    assert sha(questions) == CONFIG['calibration_questions_sha256']
    for name, digest in CONFIG['base_hashes'].items():
        assert sha(base/name) == digest
    # Work only on a new copy. Never truncate/append inside the saved snapshot.
    predictions = out/'candidate/predictions'
    shutil.copytree(prior/'candidate/predictions', predictions)
    import public_infer_811
    old = public_infer_811.recover_plans(predictions/'plans.jsonl')
    qs = [json.loads(line) for line in questions.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    assert len(qs) == 552 and len(old) == CONFIG['expected_saved_plans'] == 538
    expected_ids = [q['id'] for q in qs]
    assert set(old) <= set(expected_ids)
    old_fp = (predictions/'fingerprint.json').read_bytes()
    saved = json.loads(old_fp)
    _, business_files = public_infer_811.fingerprint_inputs(runtime, {q['business_id'] for q in qs})
    actual = {
        'model':json.loads((root/'baseline/model_lock.json').read_text()), 'variant':'candidate',
        'questions_sha256':sha(questions),
        'source_sha256':{f.name:sha(f) for f in sorted((root/'baseline/app').glob('*.py'))},
        'runtime_sha256':sha(root/'public_infer_811.py'),
        'audit_source_sha256':sha(root/'audit_output.py'), 'business_files':business_files,
        'batch_size':4, 'max_new_tokens':320, 'max_input_tokens':3072,
        'thinking':False, 'sort_by_length':True,
        'adapter_sha256':sha(candidate/'adapter_model.safetensors'),
        'abstention_policy_sha256':None,
    }
    key = hashlib.sha256(json.dumps(actual,sort_keys=True).encode()).hexdigest()
    assert saved == {'key':key, **actual}, 'Fingerprint mismatch; no forced recovery'
    atomic(out/'RECOVERY_INPUT_VERIFICATION.json', {
        'state':'passed','source_archive_sha256':CONFIG['source_archive_sha256'],
        'saved_plan_records':len(old),'remaining_model_generations':len(qs)-len(old),
        'fingerprint_unchanged':True,'parent_will_not_be_regenerated':True,
        'training_resume':False,'weights_updated':False,
    })
    if CONFIG['prepare_only']:
        import torch
        assert not torch.cuda.is_available()
        atomic(out/'CPU_PREFLIGHT.json',{'state':'passed','GPU_executed':False})
    else:
        import tempfile
        from field_eval import verify_loaded
        loaded = {}
        with tempfile.TemporaryDirectory() as scratch:
            p = Path(scratch)
            (p/'assets').mkdir()
            (p/'assets/base').symlink_to(base,target_is_directory=True)
            (p/'app').symlink_to(root/'baseline/app',target_is_directory=True)
            (p/'candidate').mkdir()
            (p/'candidate/final').symlink_to(candidate,target_is_directory=True)
            shutil.copy2(root/'baseline/model_lock.json',p/'model_lock.json')
            sys.argv=['inference','--project',str(p),'--root',str(runtime),'--questions',str(questions),
                '--variant','candidate','--output-dir',str(predictions),'--batch-size','4',
                '--cache-dir','/kaggle/temp/recovery_cache','--max-input-tokens','3072']
            public_infer_811.main(after_model=lambda model,tok:loaded.update(verify_loaded(model,candidate)))
        now = public_infer_811.recover_plans(predictions/'plans.jsonl')
        assert len(now) == 552 and all(now[k] == row for k,row in old.items())
        assert (predictions/'fingerprint.json').read_bytes() == old_fp
        rt = json.loads((predictions/'runtime.json').read_text())
        assert rt['state'] == 'completed' and rt['question_logs'] == 552 and rt['network_attempts'] == 0
        rows = [json.loads(line) for line in (predictions/'submission.jsonl').read_text().splitlines() if line.strip()]
        from audit_eval import score_saved
        def gold_loader():
            gold = runtime/'calibration/gold.jsonl'
            assert sha(gold) == CONFIG['calibration_gold_sha256']
            return {r['id']:r for r in map(json.loads,gold.read_text().splitlines())}
        from scoring import matches
        after = score_saved(expected_ids, rows, gold_loader, matches)
        after.update(manifest_sha256=CONFIG['calibration_manifest_sha256'],
            decoder_sha256=sha(root/'public_infer_811.py'),scorer_sha256=sha(root/'baseline/app/scoring.py'),
            gates_passed=True,weights_updated=False,gold_after_predictions=True,
            exposure='historically exposed calibration; continuation of frozen inference journal')
        atomic(out/'candidate/EVALUATION.json',after)
        before = json.loads((prior/'parent/EVALUATION.json').read_text())
        decision = promotion(before, after, CONFIG['promotion_policy'])
        summary = {'state':'completed','candidate_sha256':CONFIG['candidate_sha256'],
            'before':{k:before[k] for k in ['n','correct','wrong','abstain','metric']},
            'after':{k:after[k] for k in ['n','correct','wrong','abstain','metric']},
            'decision':decision,'loaded':loaded,'reused_parent_evaluation':True,
            'saved_plan_records':538,'new_model_generations':len(now)-len(old),
            'original_plan_records_unchanged':True,'fingerprint_unchanged':True,
            'recovery_seconds':time.monotonic()-started,
            'full_run_timing_not_measured_by_this_continuation':True,
            'weights_updated':False,'BEST_MODEL_updated':False,'public_ran':False}
        atomic(out/'PAIRED_CALIBRATION_RESULT.json',summary)
        print('PAIRED_RECOVERY_RESULT',json.dumps(summary),flush=True)
except Exception as exc:
    atomic(out/'FAILURE.json',{'exception_type':type(exc).__name__,'partial_outputs_preserved':True})
    raise
finally:
    seal(out)
