"""Program-only TRAIN diagnosis. Emit aggregate enums/hashes, never question/gold."""
from pathlib import Path
import sys,json,collections,ast,datetime,importlib.metadata
exp=Path(__file__).parent;sys.path[:0]=[str(exp/'src'),str(exp/'src/baseline/app')]
from relearning import verify_export,sha,atomic,seal
from runtime_data import load_learning_pool
from field_eval import TrustedShops
from baseline_bridge import BaselineBridge,canonical_total
from planner import PlanError,validate_plan
from scoring import matches
from temporal_verifier import same_time

def main():
 root=Path('D:/ZAIC/PRIVATE_DATA');runtime=root/'relearning_v2_prepared/20261108T070936492119Z';old=root/'zaic/result_811/TIME_FOCUSED_V3_20261109/time_pilot'
 assert verify_export(old)=='d3350822ea78056abc30821ec087d83ce1d00099106480afeaa7f1d2e6502ebf'
 meta=json.loads((runtime/'RUNTIME_DATA_MANIFEST.json').read_text(encoding='utf-8'))
 pool,split,gold=load_learning_pool(runtime,meta['dataset_manifest_sha256'],meta['split_manifest_sha256']);qs={q['id']:q for q in pool}
 cache=root/'time_learning_v3_cpu/temporal_cpu/TRAIN_CACHE';shops=TrustedShops(cache,'f4f3710916ec7d305845b011b6414ad9bbf76ded189a57989c558d852f04b8e6','523db34eb8e8bab52aa52d1d046307f19dbb3668495025ca05d66abd1e4c6e70');bridge=BaselineBridge(None,None,shops)
 collection=next((old/'collections').iterdir());ledger=json.loads((collection/'ledger.json').read_text(encoding='utf-8'))
 known=set()
 tree=ast.parse((exp/'src/baseline/app/planner.py').read_text(encoding='utf-8'))
 for node in ast.walk(tree):
  if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='PlanError' and node.args and isinstance(node.args[0],ast.Constant) and isinstance(node.args[0].value,str):known.add(node.args[0].value)
 report={'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'re-execute persisted TRAIN plans only; no model generation/optimizer','environment':{k:importlib.metadata.version(k) for k in ['pandas','numpy']},'source_sha256':sha(exp/'src/baseline/app/planner.py'),'unresolved_count':0,'executor_reasons':collections.Counter(),'canonical_reasons':collections.Counter(),'model_differences':collections.Counter(),'canonical_answers_matching_gold':0}
 private=[]
 for key,row in ledger['questions'].items():
  if row['status']!='unresolved':continue
  report['unresolved_count']+=1;q=qs[key];assert q['business_id'] in split['groups']['learning'];assert row['attempts']
  rec={'id':key,'original_class':row['attempts'][-1]['error_class'],'original_state':row['attempts'][-1]['state']};plan=row['attempts'][-1].get('plan');canonical=canonical_total(q)
  if plan:
   try:answer=bridge.execute(q,json.loads(json.dumps(plan)));rec['prediction_reexecuted_correct']=matches(answer,gold(key))
   except Exception as exc:
    reason=str(exc) if str(exc) in known else 'other_'+type(exc).__name__;rec['executor_reason']=reason;report['executor_reasons'][reason]+=1
  if canonical:
   if plan:
    changed=[k for k in canonical if k!='time' and plan.get(k)!=canonical[k]]
    if not same_time(plan.get('time'),canonical['time'],q['as_of_date']):changed.append('time')
    for name in changed:report['model_differences'][name]+=1
    rec['different_fields']=changed
   try:
    answer=bridge.execute(q,canonical);correct=matches(answer,gold(key));rec['canonical_scorer_correct']=correct
    report['canonical_answers_matching_gold']+=int(correct)
    if not correct:report['canonical_reasons']['scorer_mismatch']+=1
   except Exception as exc:
    reason=str(exc) if str(exc) in known else 'other_'+type(exc).__name__;report['canonical_reasons'][reason]+=1;rec['canonical_reason']=reason
  else:report['canonical_reasons']['unsupported_grammar']+=1
  private.append(rec)
 private_dir=root/'relearning_cpu_v4_20261109';private_dir.mkdir(exist_ok=False);atomic(private_dir/'PRIVATE_REPRODUCTION.json',private);seal(private_dir)
 report['private_manifest_sha256']=sha(private_dir/'ARTIFACT_MANIFEST.json')
 atomic(exp/'DIAGNOSIS_AGGREGATE.json',report);print(json.dumps(report,ensure_ascii=True,indent=2))
if __name__=='__main__':main()
