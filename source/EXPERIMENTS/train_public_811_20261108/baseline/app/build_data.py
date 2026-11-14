"""Freeze business-level splits and build execution-filtered TRAIN-only supervision."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,hashlib,json,random,re
import pandas as pd
import legacy_engine as legacy
from planner import context,prompt_text,answer,SYSTEM,candidates
from scoring import matches

SEED=20261107

def blank(metric,time,typ='number',group=None,k=None):
 return {'metric':metric,'time':time,'filters':[],'group_by':group,'order':'desc','limit':k,'unit':None,'answer_type':typ,'unresolved':[]}
def convert(old,q,shop):
 intent=old['intent'];qn=legacy.question_norm(q)
 metric={'revenue_total':'revenue','quantity_sold':'quantity','order_count':'orders','avg_order_value':'avg_order_value','cancelled_orders':'cancelled_orders','gross_profit':'gross_profit','purchase_total':'purchase_total','avg_import_cost':'avg_import_cost'}.get(intent,'other')
 group=None;k=old.get('k');typ=old['answer_type']
 if intent in {'argmax_entity','topk_list'}:
  group='customer' if 'khach' in qn else 'product';k=k or 1
  metric='orders' if group=='customer' and any(x in qn for x in ['nhieu don','nhieu lan']) else 'revenue' if group=='customer' or 'doanh thu' in qn or 'nhieu tien' in qn else 'quantity'
 if intent=='argmax_date':metric='revenue';group='month' if 'thang nao' in qn else 'date';k=1
 if intent=='share_ratio' and any(x in qn for x in ['huy don','don huy','bi huy']):metric='cancellation_rate'
 p=blank(metric,old['time'],typ,group,k)
 for c in candidates(shop,q):
  if c['exact']:p['filters'].append({'field':c['field'],'op':'eq','value':c['value']})
 for field in ['method','channel','group','customer_group','seller']:
  if field not in shop.lines:continue
  values=list(shop.lines[field].dropna().unique());found=[]
  for v in values:
   n=legacy.norm(v)
   aliases={'cash':['tien mat'],'transfer':['chuyen khoan'],'debt':['ghi no'],'store':['tai quay'],'phone':['dien thoai'],'zalo oa':['zalo']}.get(n,[])
   if any(re.search(r'(?<!\w)'+re.escape(x)+r'(?!\w)',qn) for x in [n,*aliases] if x):found.append(str(v))
  if len(found)==1 and not any(f['field']==field for f in p['filters']):p['filters'].append({'field':field,'op':'eq','value':found[0]})
 if metric=='quantity' and 'base_unit' in shop.lines:
  units=[str(u) for u in shop.lines.base_unit.dropna().unique()]
  asked=[u for u in units if re.search(r'(?<!\w)'+re.escape(legacy.norm(u))+r'(?!\w)',qn)]
  if len(asked)==1:p['unit']=asked[0]
 return p

def synthetic(shop,biz):
 months=sorted({str(x) for x in shop.lines.date.dropna().dt.to_period('M').unique()})
 months=[m for m in months if pd.Period(m).end_time.normalize()<=shop.end]
 for month in months[::max(1,len(months)//3)][:3]:
  y,m=map(int,month.split('-'));t={'kind':'month','value':month};date=str(pd.Period(month).end_time.date())
  samples=[(f'Doanh thu tháng {m}/{y} là bao nhiêu?',blank('revenue',t)),(f'Số đơn hợp lệ tháng {m}/{y}?',blank('orders',t)),(f'Số đơn huỷ tháng {m}/{y}?',blank('cancelled_orders',t)),(f'Tỷ lệ đơn huỷ tháng {m}/{y} là bao nhiêu phần trăm?',blank('cancellation_rate',t,'percent')),(f'Giá trị trung bình mỗi đơn tháng {m}/{y}?',blank('avg_order_value',t)),(f'Sản phẩm nào có doanh thu cao nhất tháng {m}/{y}?',blank('revenue',t,'string','product',1)),(f'Sản phẩm nào bán chạy nhất tháng {m}/{y}?',blank('quantity',t,'string','product',1)),(f'Top 3 sản phẩm theo doanh thu tháng {m}/{y}?',blank('revenue',t,'list','product',3)),(f'Ngày nào doanh thu cao nhất tháng {m}/{y}?',blank('revenue',t,'date','date',1))]
  if 'base_unit' in shop.lines:
   units=list(shop.lines.base_unit.dropna().unique())
   if len(units)==1:
    p=blank('quantity',t);p['unit']=str(units[0]);samples.append((f'Tổng số {units[0]} bán tháng {m}/{y}?',p))
  if 'product' in shop.products:
   for r in shop.products.head(3).to_dict('records'):
    if not r.get('product_id') or not r.get('product'):continue
    p=blank('revenue',t);p['filters']=[{'field':'product_id','op':'eq','value':str(r['product_id'])}]
    samples.append((f'Doanh thu {r["product"]} tháng {m}/{y}?',p))
  for q,p in samples:
   result=answer(shop,q,date,p)
   if result['type']!='abstain':yield q,date,p


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--project',required=True);parser.add_argument('--legacy-project',required=True);parser.add_argument('--train-root',required=True);a=parser.parse_args()
 project=Path(a.project);old=Path(a.legacy_project);root=Path(a.train_root)
 assert root.name=='train' and (root/'answers.jsonl').is_file(),'Only the official train split may be used here'
 qs=[json.loads(l) for l in (root/'questions.jsonl').read_text(encoding='utf-8-sig').splitlines() if l.strip()]
 gold={r['id']:r for r in map(json.loads,(root/'answers.jsonl').read_text(encoding='utf-8-sig').splitlines()) if r}
 businesses=sorted({q['business_id'] for q in qs});rng=random.Random(SEED);rng.shuffle(businesses)
 n=max(2,len(businesses)//5);split={'train':sorted(businesses[2*n:]),'calibration':sorted(businesses[:n]),'audit':sorted(businesses[n:2*n])}
 data=project/'data';data.mkdir(exist_ok=True)
 splitfile=data/'split.json'
 if splitfile.exists():assert json.loads(splitfile.read_text())==split
 else:splitfile.write_text(json.dumps(split,indent=2))
 def frozen_subset(name,limit):
  subset=[q for q in qs if q['business_id'] in split[name]]
  ordered=sorted(subset,key=lambda q:hashlib.sha256((str(SEED)+q['id']).encode()).hexdigest())
  # Round-robin businesses avoids one shop dominating a short GPU evaluation.
  by=defaultdict(list)
  for q in ordered:by[q['business_id']].append(q)
  out=[]
  while len(out)<min(limit,len(subset)):
   for biz in split[name]:
    if by[biz] and len(out)<limit:out.append(by[biz].pop())
  (data/(name+'_questions.jsonl')).write_text(''.join(json.dumps(q,ensure_ascii=False)+'\n' for q in out))
  return len(out)
 counts={'calibration':frozen_subset('calibration',24),'audit':frozen_subset('audit',48)}
 oldrows=[json.loads(l) for l in (old/'training_v1/sft_train_v1.jsonl').read_text().splitlines() if l.strip()]
 weak={r['id']:json.loads(r['response']) for r in oldrows if r.get('business_id') in split['train'] and r.get('task')=='nlu'}
 rulebook=(root/'rulebook.md').read_text(encoding='utf-8-sig');rows=[];rejected=Counter();sources=Counter();schemas={}
 for biz in split['train']:
  print('BUILD_TRAIN_SHOP:',biz,flush=True)
  try:shop=legacy.prepare_shop(root/'data'/biz,old/'cache_033')
  except Exception as exc:rejected['loader '+type(exc).__name__]+=1;continue
  schemas[biz]=hashlib.sha256((root/'data'/biz/'README.md').read_bytes()).hexdigest()
  for q in (r for r in qs if r['business_id']==biz and r['id'] in weak):
   try:
    p=convert(weak[q['id']],q['question'],shop);pred=answer(shop,q['question'],q['as_of_date'],p)
    if not matches(pred,gold[q['id']]):rejected[pred.get('reason','gold mismatch')]+=1;continue
    rows.append({'id':q['id'],'business_id':biz,'prompt':prompt_text(context(shop,q['question'],q['as_of_date'],rulebook)),'response':json.dumps(p,ensure_ascii=False,separators=(',',':')),'source':'execution_filtered_weak_plan'});sources['execution_filtered_weak_plan']+=1
   except (ValueError,KeyError,TypeError):rejected['weak plan invalid']+=1
  for i,(q,date,p) in enumerate(synthetic(shop,biz)):
   rows.append({'id':f'synthetic_{biz}_{i}','business_id':biz,'prompt':prompt_text(context(shop,q,date,rulebook)),'response':json.dumps(p,ensure_ascii=False,separators=(',',':')),'source':'constructed_plan_template'});sources['constructed_plan_template']+=1
  print('TRAIN_ROWS_SO_FAR:',len(rows),flush=True)
 assert rows,'No verified training rows'
 rng.shuffle(rows)
 # Balance sources/businesses before imposing the T4 budget; never truncate token sequences.
 buckets=defaultdict(list)
 for r in rows:buckets[(r['business_id'],r['source'])].append(r)
 selected=[]
 while len(selected)<min(768,len(rows)):
  for key in sorted(buckets):
   if buckets[key] and len(selected)<768:selected.append(buckets[key].pop())
 assert all(r['business_id'] in split['train'] for r in selected)
 path=data/'sft_train.jsonl';path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
 (data/'system.txt').write_text(SYSTEM)
 manifest={'seed':SEED,'split':split,'heldout_question_counts':counts,'available_training_rows':len(rows),'selected_rows':len(selected),'selected_by_source':dict(Counter(r['source'] for r in selected)),'selected_by_business':dict(Counter(r['business_id'] for r in selected)),'rejected_reasons':dict(rejected),'training_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'questions_sha256':hashlib.sha256((root/'questions.jsonl').read_bytes()).hexdigest(),'rulebook_sha256':hashlib.sha256((root/'rulebook.md').read_bytes()).hexdigest(),'readme_sha256':schemas,'public_test_used':False,'note':'New base model and business-disjoint splits. Prior 4B development was reused for debugging, so old 4B PA is not an unbiased comparator. Filtered weak plans may still be spurious; separate constructed templates have known plans. T4 budget caps initial candidate at 768 rows, one epoch.'}
 (data/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
 print('TRAIN_DATA_READY:',json.dumps({k:manifest[k] for k in ['selected_rows','selected_by_source','heldout_question_counts','public_test_used']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
