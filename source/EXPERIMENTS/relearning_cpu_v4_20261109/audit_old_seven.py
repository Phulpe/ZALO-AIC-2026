"""Saved candidate diagnostics on verified TRAIN targets, aggregate only."""
from pathlib import Path
import sys,json,collections
exp=Path(__file__).parent;sys.path[:0]=[str(exp/'src'),str(exp/'src/baseline/app')]
from baseline_bridge import canonical_total
from failure_triage import plan_difference
from relearning import atomic,sha
root=Path('D:/ZAIC/PRIVATE_DATA');run=root/'zaic/result_811/TIME_FOCUSED_V3_20261109';p=run/'evaluation_recovery/post/original14/PRIVATE_PREDICTIONS.json';predictions=json.loads(p.read_text(encoding='utf-8'));selected=json.loads((root/'quality_round_v2_selected/PRIVATE_SELECTED.json').read_text(encoding='utf-8'));correction={r['id'] for r in selected if r['kind']=='correction'}
pool={q['id']:q for q in map(json.loads,(root/'relearning_v2_prepared/20261108T070936492119Z/learning/model_pool.jsonl').read_text(encoding='utf-8').splitlines())}
summary={'scope':'7 old verified TRAIN corrections; saved D31 outputs, no new generation','cases':len(correction),'differing_fields':collections.Counter(),'predicted_time_kind':collections.Counter(),'target_time_kind':collections.Counter(),'missing_plans':0,'source_prediction_sha256':sha(p)}
for key in correction:
 q=pool[key];target=canonical_total(q);plan=predictions[key].get('plan')
 if plan is None:summary['missing_plans']+=1;continue
 summary['differing_fields'].update(plan_difference(plan,target,q['as_of_date']));kind=plan.get('time',{}).get('kind');summary['predicted_time_kind'][kind if kind in {'today','yesterday','last_month','month','day','this_month','range'} else 'other']+=1;summary['target_time_kind'][target['time']['kind']]+=1
atomic(exp/'OLD_SEVEN_AGGREGATE.json',summary);print(json.dumps(summary,ensure_ascii=True,indent=2))
