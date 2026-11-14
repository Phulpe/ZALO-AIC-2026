"""Bounded blind learning probe before the synthetic smoke optimizer step."""
from pathlib import Path
import json
from relearning import new_run,LearningRound,mixture,atomic,seal,sha
from runtime_data import load_learning_pool,pending_coverage,reconcile_coverage

def collect(model,tokenizer,config):
    limit=config.get('learning_probe_max_questions',8)
    if type(limit)!=int or not 1<=limit<=(128 if config.get('verifier_v2_pool') else 64):raise ValueError('Learning probe budget must be 1..8')
    root=Path(config['runtime_root'])
    if sha(root/'RUNTIME_DATA_MANIFEST.json')!=config['runtime_manifest_sha256']:raise ValueError('Runtime source manifest mismatch')
    runtime=json.loads((root/'RUNTIME_DATA_MANIFEST.json').read_text())
    pool,split,gold=load_learning_pool(root,runtime['dataset_manifest_sha256'],runtime['split_manifest_sha256'])
    from baseline_bridge import canonical_total
    supported={q['id'] for q in pool if canonical_total(q) is not None} if config.get('verifier_v2_pool') else set(split['pilot_ids'])
    # Deterministic source-question-only selection, never selected using gold.
    selected=sorted([q for q in pool if q['id'] in supported],key=lambda q:q['id'])[:limit]
    if not selected:raise ValueError('No supported pilot questions')
    from baseline_bridge import BaselineBridge,canonical_correction,matches
    from planner import validate_plan
    from legacy_engine import prepare_shop
    shops={}
    for biz in sorted({q['business_id'] for q in selected}):
        directory=(root/'data'/biz).resolve()
        if not directory.is_relative_to((root/'data').resolve()):raise ValueError('Source path escape')
        for p in directory.rglob('*'):
            if p.is_file():
                relative=p.relative_to(root).as_posix()
                if sha(p)!=runtime['files'][relative]['sha256']:raise ValueError('Business source mismatch')
        shops[biz]=prepare_shop(directory,cache_dir=None)
    cfg={'mode':'learning','init_mode':'warm_start','parent_sha256':config['parent_sha256'],'learning_businesses':split['groups']['learning'],'calibration_businesses':split['groups']['calibration'],'holdout_businesses':split['groups']['audit_exposed'],'attempts_per_question':2,'data_manifest_sha256':runtime['dataset_manifest_sha256'],'split_manifest_sha256':runtime['split_manifest_sha256'],'scope':'budgeted pilot; full learning pool remains tracked separately'}
    output=new_run(Path(config['output_root']).parent/'learning_probe_runs_v2',cfg,[root])
    pending_coverage(output/'FULL_POOL_COVERAGE.json',pool,config['parent_sha256'],runtime['split_manifest_sha256'])
    bridge=BaselineBridge(model,tokenizer,shops,max_tokens=240,max_seconds=25)
    training=model.training
    try:
        metrics,replay,correction=LearningRound(output,cfg,selected,gold,bridge.predict,bridge.execute,validate_plan,bridge.semantic,matches,canonical_correction).process()
    finally:model.train(training)
    ledger=json.loads((output/'ledger.json').read_text())
    full=reconcile_coverage(output/'FULL_POOL_COVERAGE.json',ledger)
    ready=bool(replay and correction)
    if ready:
        rows=mixture(replay,correction,split['groups']['learning'])
        atomic(output/'verified_dataset.json',rows)
    summary={'run_id':output.name,'scope':'real TRAIN pilot; no weight updates in probe','pilot':metrics,'full_pool_coverage':full,'replay':len(replay),'correction':len(correction),'verified_dataset_ready':ready,'verified_dataset_sha256':sha(output/'verified_dataset.json') if ready else None,'model_parent_sha256':config['parent_sha256'],'optimizer_executed':False,'path':str(output)}
    if config.get('verifier_v2_pool'):
        from real_evidence import export
        summary['quality_gate']=export(output,selected,bridge,gold,config['supervision_gate'])
    atomic(output/'PROBE_SUMMARY.json',summary);seal(output)
    return summary
