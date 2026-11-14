"""Independent frozen executor comparison and raw row evidence; aggregate only."""
exec((__import__('pathlib').Path(__file__).parent/'source_diagnosis.py').read_text(encoding='utf-8').split('for rec in repros:')[0])
import pandas as pd
from legacy_engine import answer_question
c=collections.Counter();seen=set()
for rec in repros:
 q=qs[rec['id']];shop=shops[q['source_ref']];plan=canonical_total(q)
 if rec.get('canonical_scorer_correct') is False:
  alternate=answer_question(shop,json.loads(q['prompt'])['question'],q['as_of_date'],{'intent':'revenue_total','time':plan['time'],'answer_type':'number'})
  c['legacy_executor_cases']+=1;c['legacy_executor_correct']+=int(matches(alternate,gold(q['id'])))
 if rec.get('canonical_reason')=='Unparsed numeric amounts':
  raw=shop.tables['sales'];missing=raw.loc[raw.line_total.isna()]
  for ix,row in missing.iterrows():
   key=(q['source_ref'],ix)
   if key in seen:continue
   seen.add(key);c['raw_missing_rows']+=1
   c['raw_date_missing']+=int(pd.isna(row.get('date')));c['raw_order_missing']+=int(pd.isna(row.get('order_id')) or not str(row.get('order_id')).strip())
   label=str(row.get('product') or '')
   c['explicit_annotation_marker']+=int(bool(re.match(r'^\s*(?:ghi chú|ghi chu|lưu ý|luu y|chú ý|chu y)(?:\s*[:：]|\s*$)',label,re.I)))
   c['raw_numeric_all_missing']+=int(all(pd.isna(row.get(k)) for k in ['qty','price','line_total']))
   c['raw_unit_missing']+=int(pd.isna(row.get('unit')))
atomic(exp/'INDEPENDENT_AUDIT_AGGREGATE.json',dict(c));print(json.dumps(c,ensure_ascii=True,indent=2))
