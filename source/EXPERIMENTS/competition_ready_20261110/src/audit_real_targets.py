from pathlib import Path
import sys,json,contextlib,collections,time
exp=Path(__file__).parent;sys.path[:0]=[str(exp),str(exp/'baseline/app')]
from runtime_data import load_learning_pool
from baseline_bridge import canonical_total,BaselineBridge,matches
from planner import validate_plan
from legacy_engine import prepare_shop
from relearning import sha,atomic
root=Path('D:/ZAIC/PRIVATE_DATA/relearning_v2_prepared/20261108T070936492119Z')
out=Path('D:/ZAIC/PRIVATE_DATA/correction_verifier_v2_audit');out.mkdir(exist_ok=False)
runtime=json.loads((root/'RUNTIME_DATA_MANIFEST.json').read_text());pool,split,gold=load_learning_pool(root,runtime['dataset_manifest_sha256'],runtime['split_manifest_sha256'])
selected=[q for q in pool if canonical_total(q) is not None];rows=[];counts=collections.Counter();started=time.monotonic()
with (out/'PRIVATE_EXECUTOR.log').open('w',encoding='utf8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
 for biz in sorted({q['business_id'] for q in selected}):
  try:shop=prepare_shop(root/'data'/biz,cache_dir=None);bridge=BaselineBridge(None,None,{biz:shop})
  except Exception as exc:
   for q in selected:
    if q['business_id']==biz:rows.append({'id':q['id'],'state':'ingestion_error','error_type':type(exc).__name__});counts['ingestion_error']+=1
   continue
  for q in selected:
   if q['business_id']!=biz:continue
   rec={'id':q['id'],'split':'learning','source_manifest_sha256':sha(root/'RUNTIME_DATA_MANIFEST.json')}
   try:
    plan=canonical_total(q);validate_plan(plan);assert bridge.semantic(q,plan)
    answer=bridge.execute(q,plan)
    # This is an offline target audit, not a blind model attempt or correction record.
    ok=matches(answer,gold(q['id']));rec.update(state='target_audit_passed' if ok else 'gold_mismatch',plan=plan,answer=answer)
   except Exception as exc:rec.update(state='executor_error',error_type=type(exc).__name__)
   counts[rec['state']]+=1;rows.append(rec)
  atomic(out/'PRIVATE_AUDIT_ROWS.json',rows)
  atomic(out/'SUMMARY.json',{'state':'running','selected':len(selected),'audited':len(rows),'counts':dict(counts),'model_predictions':0,'training_ready':False})
atomic(out/'SUMMARY.json',{'state':'completed','pool':len(pool),'selected':len(selected),'audited':len(rows),'counts':dict(counts),'model_predictions':0,'corrections_created':0,'training_ready':False,'elapsed_seconds':time.monotonic()-started,'source_sha256':sha(exp/'baseline_bridge.py')})
print(json.dumps(json.loads((out/'SUMMARY.json').read_text())))
