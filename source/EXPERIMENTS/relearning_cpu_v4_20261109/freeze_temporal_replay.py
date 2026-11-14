"""Select a new temporal replay version from immutable verified TRAIN bank.

Reuses actual execution evidence only when target-verifier code/data are
unchanged. No gold read, model inference, label creation or optimizer.
"""
from pathlib import Path
import sys,json,copy,collections,datetime
exp=Path(__file__).parent;sys.path[:0]=[str(exp/'src'),str(exp/'src/baseline/app')]
from relearning import verify_export,sha,atomic,seal
from baseline_bridge import canonical_total
from quality_gate import select,digest
root=Path('D:/ZAIC/PRIVATE_DATA');prior=root/'relearning_cpu_v4_revalidation_final_r3_20261110';assert verify_export(prior)=='f598d4e8ddfc81929a7d5a786c7c012b4b0ddcece97687f5458b8508e1cc17ff'
proof=json.loads((prior/'VERIFIER_MANIFEST.json').read_text(encoding='utf-8'))
execution_files=['baseline_bridge.py','temporal_verifier.py','learning_executor.py','source_annotations.py','failure_triage.py','relearning.py','baseline/app/planner.py','baseline/app/scoring.py']
assert all(sha(exp/'src'/name)==proof[name] for name in execution_files),'Execution verifier changed; real revalidation required'
runtime=root/'relearning_v2_prepared/20261108T070936492119Z';ref=json.loads((prior/'SOURCE_SPLIT_REFERENCE.json').read_text(encoding='utf-8'));assert sha(runtime/'DATA_MANIFEST.json')==ref['data_manifest_sha256'] and sha(runtime/'SPLIT_MANIFEST.json')==ref['split_manifest_sha256'];data=json.loads((runtime/'DATA_MANIFEST.json').read_text(encoding='utf-8'));assert sha(runtime/'learning/model_pool.jsonl')==data['files']['learning/model_pool.jsonl']['sha256']
qs={q['id']:q for q in map(json.loads,(runtime/'learning/model_pool.jsonl').read_text(encoding='utf-8').splitlines())}
rows=json.loads((prior/'PRIVATE_BANK.json').read_text(encoding='utf-8'));evidence=json.loads((prior/'PRIVATE_EVIDENCE.json').read_text(encoding='utf-8'));cfg=json.loads((prior/'SUPERVISION_GATE.json').read_text(encoding='utf-8'));cfg.update(selection_policy='time_business_metric_v1',min_replay_time_buckets=5)
for row in rows:
 q=qs[row['id']];assert q['prompt']==row['prompt'] and q['business_id']==row['business_id'] and q['split']=='learning'
 evidence[row['id']]['temporal_bucket']=canonical_total(q)['time']['kind']
selected,report=select(rows,evidence,cfg);old=json.loads((root/'quality_round_v2_selected/PRIVATE_SELECTED.json').read_text(encoding='utf-8'));critical={r['id'] for r in old if r['kind']=='correction'};assert critical<={r['id'] for r in selected if r['kind']=='correction'}
protected=[r for r in rows if r['kind']=='replay'];assert len(protected)==103
assert all(sha(exp/'src'/name)==proof[name] for name in execution_files)
out=root/'relearning_cpu_v4_temporal_dataset_20261110';out.mkdir(exist_ok=False)
for name,content in [('PRIVATE_BANK.json',rows),('PRIVATE_EVIDENCE.json',evidence),('PRIVATE_SELECTED.json',selected),('PRIVATE_REGRESSION_BANK.json',protected),('SUPERVISION_GATE.json',cfg),('SELECTION_REPORT.json',report),('SOURCE_SPLIT_REFERENCE.json',ref)]:atomic(out/name,content)
atomic(out/'EVIDENCE_LINEAGE.json',{'actual_execution_source_manifest':str(prior/'ARTIFACT_MANIFEST.json'),'actual_execution_manifest_sha256':verify_export(prior),'actual_execution_verifier_sha256':cfg['verifier_sha256'],'execution_source_files_unchanged':{name:proof[name] for name in execution_files},'new_selection_policy_sha256':sha(exp/'src/quality_gate.py'),'selection_driver_sha256':sha(__file__),'gold_read_during_reselection':False,'optimizer_executed':False,'protected_old_corrections':7,'full_replay_regression_cases':103,'bank_replay':103,'bank_correction':9,'pending_learning_ids':1693,'unresolved_cases':12,'new_parent_requires_new_blind_predictions':True})
summary={'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':'frozen_cpu_verified_bank_reselected','selection':report,'old7_included':7,'all_replay_regression_cases':103,'new_gold_labels_created':0,'bank_replay':103,'bank_correction':9,'source_execution_revalidation_reused_with_exact_hashes':True,'dataset_manifest_sha256':seal(out),'private_dataset':str(out),'gold_read':False,'gpu_executed':False,'optimizer_executed':False};atomic(exp/'FROZEN_TEMPORAL_DATASET_SUMMARY.json',summary);print(json.dumps(summary,indent=2))
