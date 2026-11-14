"""Aggregate identity/coverage invariants for unresolved TRAIN sources."""
from pathlib import Path
import sys,json,collections
exp=Path(__file__).parent;sys.path[:0]=[str(exp/'src'),str(exp/'src/baseline/app')]
from field_eval import TrustedShops
from relearning import atomic
root=Path('D:/ZAIC/PRIVATE_DATA');runtime=root/'relearning_v2_prepared/20261108T070936492119Z';qs={q['id']:q for q in map(json.loads,(runtime/'learning/model_pool.jsonl').read_text(encoding='utf-8').splitlines())};cases=json.loads((root/'relearning_cpu_v4_revalidation_final_r3_20261110/PRIVATE_TRIAGE.json').read_text(encoding='utf-8'));businesses={qs[r['id']]['source_ref'] for r in cases if r['classification']['class']=='source_scorer_or_label_discrepancy'}
shops=TrustedShops(root/'time_learning_v3_cpu/temporal_cpu/TRAIN_CACHE','f4f3710916ec7d305845b011b6414ad9bbf76ded189a57989c558d852f04b8e6','523db34eb8e8bab52aa52d1d046307f19dbb3668495025ca05d66abd1e4c6e70');counts=collections.Counter()
for business in sorted(businesses):
 shop=shops[business];f=shop.tables.get('sales');counts['businesses']+=1
 if f is None:continue
 counts['raw_sales_rows']+=len(f);counts['prepared_sales_rows']+=len(shop.lines);counts['identical_raw_rows_ignoring_source']+=int(f.drop(columns=['_source'],errors='ignore').duplicated().sum());counts['identical_raw_rows_within_source']+=int(f.duplicated().sum())
 if {'order_id','date'}<=set(f):
  valid=f.loc[f.order_id.notna()&f.date.notna()].copy();valid['month']=valid.date.dt.strftime('%Y-%m');collisions=valid.groupby('order_id').month.nunique().gt(1);counts['order_ids_with_multiple_source_months']+=int(collisions.sum());counts['businesses_with_cross_month_id_collision']+=int(collisions.any())
atomic(exp/'SOURCE_IDENTITY_AGGREGATE.json',dict(counts));print(json.dumps(counts,indent=2))
