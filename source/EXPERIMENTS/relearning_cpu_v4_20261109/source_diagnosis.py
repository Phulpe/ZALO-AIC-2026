"""Second private audit: source completeness and scorer mismatch taxonomy only."""
from pathlib import Path
import sys,json,collections,re
exp=Path(__file__).parent;sys.path[:0]=[str(exp/'src'),str(exp/'src/baseline/app')]
from relearning import sha,verify_export,atomic
from runtime_data import load_learning_pool
from field_eval import TrustedShops
from baseline_bridge import BaselineBridge,canonical_total
from legacy_engine import time_bounds,valid,norm
from scoring import matches
root=Path('D:/ZAIC/PRIVATE_DATA');runtime=root/'relearning_v2_prepared/20261108T070936492119Z';m=json.loads((runtime/'RUNTIME_DATA_MANIFEST.json').read_text(encoding='utf-8'));pool,split,gold=load_learning_pool(runtime,m['dataset_manifest_sha256'],m['split_manifest_sha256']);qs={q['id']:q for q in pool}
cache=root/'time_learning_v3_cpu/temporal_cpu/TRAIN_CACHE';shops=TrustedShops(cache,'f4f3710916ec7d305845b011b6414ad9bbf76ded189a57989c558d852f04b8e6','523db34eb8e8bab52aa52d1d046307f19dbb3668495025ca05d66abd1e4c6e70');bridge=BaselineBridge(None,None,shops)
repros=json.loads((root/'relearning_cpu_v4_20261109/PRIVATE_REPRODUCTION.json').read_text(encoding='utf-8'));report={'cases':len(repros),'case_taxonomy':collections.Counter(),'amount_missing_profiles':collections.Counter(),'rules_presence':collections.Counter(),'mismatch_type_pairs':collections.Counter(),'schema_metrics':collections.Counter()};seen=set()
for rec in repros:
 q=qs[rec['id']];plan=canonical_total(q);shop=shops[q['source_ref']];ctx=json.loads(q['prompt']);report['schema_metrics'][plan['metric']]+=1
 if rec.get('canonical_reason')=='Unparsed numeric amounts':
  start,end=time_bounds({'time':plan['time']},q['as_of_date']);f=valid(shop.lines);f=f.loc[f.date.ge(start)&f.date.lt(end+__import__('pandas').Timedelta(days=1))];bad=f.loc[f.line_total.isna()]
  report['amount_missing_profiles']['rows']+=len(bad)
  for k in ['qty','price','revenue','total','paid','cost']:
   report['amount_missing_profiles'][k+'_column_present_cases']+=int(k in f)
   if k in bad:report['amount_missing_profiles'][k+'_nonnull_bad_rows']+=int(bad[k].notna().sum())
  report['amount_missing_profiles']['missing_amount_rows_qty_price_present']+=sum(bool(__import__('pandas').notna(row.get('qty')) and __import__('pandas').notna(row.get('price'))) for row in bad.to_dict('records'))
  report['case_taxonomy']['missing_amount_guard']+=1
 elif rec.get('canonical_reason')=='Period predates observed data; coverage not proven':report['case_taxonomy']['coverage_guard']+=1
 elif not rec.get('canonical_scorer_correct',True):
  a=bridge.execute(q,plan);g=gold(q['id']);report['mismatch_type_pairs'][str(a.get('type'))+' / '+str(g.get('type'))]+=1
  # Safe coarse categories; no answer values/IDs/ratios leave the private process.
  if a.get('type')!=g.get('type'):report['case_taxonomy']['scorer_type_mismatch']+=1
  else:report['case_taxonomy']['scorer_value_mismatch']+=1
 if q['source_ref'] not in seen:
  seen.add(q['source_ref']);text=norm(shop.readme)
  for word in ['thanh tien','don gia','giam gia','vat','tong','trong','bo trong','trong la 0','cong thuc','lam tron']:
   report['rules_presence'][word]+=int(word in text)
atomic(exp/'SOURCE_DIAGNOSIS_AGGREGATE.json',report);print(json.dumps(report,ensure_ascii=True,indent=2))
