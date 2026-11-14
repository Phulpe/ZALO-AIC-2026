"""Freeze a new dataset version from persisted blind TRAIN outputs; CPU only."""
from pathlib import Path
import sys,json,collections,datetime,copy
exp=Path(__file__).parent;sys.path[:0]=[str(exp/'src'),str(exp/'src/baseline/app')]
from relearning import sha,verify_export,atomic,seal
from runtime_data import load_learning_pool
from field_eval import TrustedShops
from baseline_bridge import BaselineBridge,canonical_total
from source_annotations import VerifiedLearningShops
from learning_executor import repair_shop
from failure_triage import classify
from planner import validate_plan
from scoring import matches
from quality_gate import select,digest
root=Path('D:/ZAIC/PRIVATE_DATA');runtime=root/'relearning_v2_prepared/20261108T070936492119Z';old=root/'zaic/result_811/TIME_FOCUSED_V3_20261109/time_pilot';assert verify_export(old)=='d3350822ea78056abc30821ec087d83ce1d00099106480afeaa7f1d2e6502ebf'
m=json.loads((runtime/'RUNTIME_DATA_MANIFEST.json').read_text(encoding='utf-8'));pool,split,gold=load_learning_pool(runtime,m['dataset_manifest_sha256'],m['split_manifest_sha256']);qs={q['id']:q for q in pool}
cache=root/'time_learning_v3_cpu/temporal_cpu/TRAIN_CACHE';base=TrustedShops(cache,'f4f3710916ec7d305845b011b6414ad9bbf76ded189a57989c558d852f04b8e6','523db34eb8e8bab52aa52d1d046307f19dbb3668495025ca05d66abd1e4c6e70');shops=VerifiedLearningShops(base,runtime,m['dataset_manifest_sha256'],m['split_manifest_sha256']);before=BaselineBridge(None,None,base);after=BaselineBridge(None,None,shops)
ledger=json.loads(next((old/'collections').glob('*/ledger.json')).read_text(encoding='utf-8'));output=root/'relearning_cpu_v4_revalidation_final_r3_20261110';output.mkdir(exist_ok=False)
report={'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'persisted E7 predictions; no new model/optimizer; local Windows CPU, not Kaggle parity','attempted':0,'before_correct':0,'after_correct':0,'correct_to_wrong':0,'repaired_executor_correct':0,'executor_exceptions_resolved':0,'remaining_unresolved':0,'verified_model_corrections':0,'triage':collections.Counter(),'numeric_discrepancy_taxonomy':collections.Counter(),'source_note_rows_removed':0};rows=[];evidence={};regression=[];private=[];seenbiz=set()
config=json.loads((root/'zaic/result_811/TIME_FOCUSED_V3_20261109/time_pilot/model_runs/20261109T125033322737Z_b786bc50/run_config.json').read_text(encoding='utf-8'))['supervision_gate'];verifier={n:sha(exp/'src'/n) for n in ['baseline_bridge.py','temporal_verifier.py','learning_executor.py','source_annotations.py','failure_triage.py','real_evidence.py','quality_gate.py','relearning.py','baseline/app/planner.py','baseline/app/scoring.py']};verifier['curation_driver.py']=sha(__file__);config['verifier_sha256']=digest(verifier)
for key,record in sorted(ledger['questions'].items(),key=lambda pair:qs[pair[0]]['business_id']):
 q=qs[key];assert q['split']=='learning' and q['business_id'] in split['groups']['learning'];attempt=record['attempts'][0];assert 'plan' in attempt or 'prediction_text' in attempt;report['attempted']+=1;plan=attempt.get('plan');canonical=canonical_total(q);assert canonical is not None
 def checked(bridge,proposed):
  try:
   validate_plan(copy.deepcopy(proposed))
   if not bridge.semantic(q,proposed):return False,'semantic_mismatch'
   answer=bridge.execute(q,copy.deepcopy(proposed));return matches(answer,gold(key)),None
  except Exception as exc:return False,type(exc).__name__
 b,be=checked(before,plan);a,ae=checked(after,plan);target_ok,target_error=checked(after,canonical)
 if be=='PlanError' and ae is None:report['executor_exceptions_resolved']+=1
 if not target_ok and target_error is None:
  import math
  actual=after.execute(q,copy.deepcopy(canonical));expected=gold(key);x=actual.get('answer');y=expected.get('answer')
  category='business_rule_ingestion_or_label_difference'
  if type(x) in {int,float} and type(y) in {int,float} and math.isfinite(x) and math.isfinite(y):
   if math.isclose(x,y,rel_tol=0,abs_tol=1e-6):category='floating_precision_suspect'
   elif x==round(y):category='rounding_contract_suspect'
   elif x==0 and y!=0:category='zero_vs_nonzero'
   elif any(math.isclose(x*factor,y,rel_tol=1e-12,abs_tol=1e-6) for factor in [.001,1000,1000000]):category='currency_scale_suspect'
  report['numeric_discrepancy_taxonomy'][category]+=1
 # Confirm original results before using the new source correction.
 assert b==record['first_correct'];report['before_correct']+=b;report['after_correct']+=a;report['correct_to_wrong']+=int(b and not a)
 classification=classify(predicted=plan,target=canonical,as_of=q['as_of_date'],source_error=target_error,canonical_correct=target_ok)
 if a and not b:
  assert classification['class']=='executor_repaired';report['repaired_executor_correct']+=1;regression.append({'id':key,'business_id':q['business_id'],'plan':plan,'role':'executor_regression_only','not_optimizer_supervision':True})
 if not a:
  report['triage'][classification['class']]+=1
  if not classification['train_eligible']:report['remaining_unresolved']+=1
 if b and a:kind='replay';target=plan;first=True
 elif not a and classification['train_eligible']:kind='correction';target=canonical;first=False;report['verified_model_corrections']+=1
 else:private.append({'id':key,'before_correct':b,'after_correct':a,'classification':classification});continue
 row={'id':key,'business_id':q['business_id'],'split':'learning','kind':kind,'prompt':q['prompt'],'response':json.dumps(target,ensure_ascii=False),'parent_sha256':config['parent_sha256'],'target_verified':True};rows.append(row)
 ev={k:config[k] for k in ['parent_sha256','source_manifest_sha256','split_manifest_sha256','verifier_sha256','scorer_sha256']};ev.update({k:True for k in ['schema','semantic','execution','scorer','blind_prediction_persisted','gold_after_prediction']});ev.update(plan_sha256=digest(target),prompt_sha256=digest(q['prompt']),first_pass_verified_correct=first);evidence[key]=ev
 private.append({'id':key,'before_correct':b,'after_correct':a,'classification':classification})
assert report['correct_to_wrong']==0 and report['attempted']==124
for business,proofs in shops.proofs.items():
 shop=shops[business];clean,proof=repair_shop(shop,proofs);report['source_note_rows_removed']+=proof['removed']
assert all(sha(Path(__file__) if name=='curation_driver.py' else exp/'src'/name)==digest0 for name,digest0 in verifier.items()),'Verifier source changed during revalidation; preserve draft and rerun new version'
selected,selection=select(rows,evidence,config)
atomic(output/'PRIVATE_BANK.json',rows);atomic(output/'PRIVATE_EVIDENCE.json',evidence);atomic(output/'PRIVATE_SELECTED.json',selected);atomic(output/'PRIVATE_EXECUTOR_REGRESSION.json',regression);atomic(output/'PRIVATE_TRIAGE.json',private);atomic(output/'SUPERVISION_GATE.json',config);atomic(output/'SELECTION_REPORT.json',selection);atomic(output/'VERIFIER_MANIFEST.json',verifier);atomic(output/'PRIVATE_SOURCE_NOTE_PROOFS.json',{business:{'approved_source_label_hashes':[[__import__('hashlib').sha256(source.encode()).hexdigest(),__import__('hashlib').sha256(label.encode()).hexdigest()] for source,label in sorted(proofs[0])],'evidence':proofs[1]} for business,proofs in shops.proofs.items()});atomic(output/'SOURCE_SPLIT_REFERENCE.json',{'data_manifest_sha256':m['dataset_manifest_sha256'],'split_manifest_sha256':m['split_manifest_sha256'],'historical_predictions_only':True,'full_learning_pool_pending':1693,'requires_new_paired_baseline_calibration':True,'no_gpu_or_optimizer_executed':True})
report.update(selection=selection,private_output=str(output),manifest_sha256=seal(output),not_a_new_model_first_pass=True)
atomic(exp/'REVALIDATION_FINAL_R3_AGGREGATE.json',report);print(json.dumps(report,ensure_ascii=True,indent=2))
