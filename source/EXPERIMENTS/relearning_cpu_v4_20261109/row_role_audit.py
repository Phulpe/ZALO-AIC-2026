"""Private source role audit; output fixed feature counts only."""
exec((__import__('pathlib').Path(__file__).parent/'source_diagnosis.py').read_text(encoding='utf-8').split('for rec in repros:')[0])
import pandas as pd
c=collections.Counter();seen=set()
for rec in repros:
 if rec.get('canonical_reason')!='Unparsed numeric amounts':continue
 q=qs[rec['id']];shop=shops[q['source_ref']];plan=canonical_total(q);start,end=time_bounds({'time':plan['time']},q['as_of_date']);f=valid(shop.lines);bad=f.loc[f.date.ge(start)&f.date.lt(end+pd.Timedelta(days=1))&f.line_total.isna()]
 for ix,row in bad.iterrows():
  key=(q['source_ref'],ix)
  if key in seen:continue
  seen.add(key);c['unique_bad_rows']+=1
  for name in ['product','product_id','customer','order_id','note','work','unit','base_unit','status']:
   c[name+'_present']+=int(pd.notna(row.get(name)) and bool(str(row.get(name)).strip()))
  label=norm(row.get('product'));c['label_empty']+=int(not label)
  for name,pat in [('total_label',r'^(tong|cong)( |$)'),('note_label',r'^(ghi chu|luu y|chu y)( |$)'),('header_label',r'^(ten hang|san pham|ma hang)( |$)')]:c[name]+=int(bool(re.search(pat,label)))
  src=str(row.get('_source',''));c['source_mapped_sales']+=int(any(a['kind']=='sales' and a['file']+':'+str(a['sheet'])==src for a in shop.audit))
atomic(exp/'ROW_ROLE_AGGREGATE.json',dict(c));print(json.dumps(c,ensure_ascii=True,indent=2))
