"""Independent submission structure audit. Does not open labels or infer answers."""
from pathlib import Path
from collections import Counter
from datetime import date
import argparse,hashlib,json,math,re
TYPES={'number','percent','string','date','boolean','list','table','unanswerable','abstain'}
def number(x):return type(x) in (int,float) and math.isfinite(x)
def scalar(x):return isinstance(x,str) or type(x) is bool or number(x)
def validate(row):
 if not isinstance(row,dict) or not isinstance(row.get('id'),str) or not row['id']:raise ValueError('invalid_id')
 typ=row.get('type')
 if typ not in TYPES:raise ValueError('invalid_type')
 expected={'id','type'} if typ in {'abstain','unanswerable'} else {'id','type','answer'}
 if set(row)!=expected:raise ValueError('invalid_keys')
 if typ in {'abstain','unanswerable'}:return
 a=row['answer']
 if typ in {'number','percent'}:
  if not number(a):raise ValueError('nonfinite_or_non_numeric')
 elif typ=='boolean':
  if type(a) is not bool:raise ValueError('invalid_boolean')
 elif typ=='string':
  if not isinstance(a,str):raise ValueError('invalid_string')
 elif typ=='date':
  if not isinstance(a,str) or not re.fullmatch(r'\d{4}-\d{2}(?:-\d{2})?',a):raise ValueError('invalid_iso_date')
  try:date.fromisoformat(a if len(a)==10 else a+'-01')
  except ValueError:raise ValueError('invalid_calendar_date')
 elif typ=='list':
  if not isinstance(a,list) or not all(scalar(x) for x in a):raise ValueError('invalid_list')
 elif typ=='table':
  if not isinstance(a,dict) or set(a)!={'columns','rows'}:raise ValueError('invalid_table')
  cols=a['columns'];rs=a['rows']
  if not isinstance(cols,list) or not cols or not all(isinstance(c,str) for c in cols) or len(set(cols))!=len(cols):raise ValueError('invalid_columns')
  if not isinstance(rs,list) or not all(isinstance(r,list) and len(r)==len(cols) and all(scalar(x) for x in r) for r in rs):raise ValueError('invalid_rows')

def audit(questions,submission):
 qids=[q['id'] for q in questions];ids=[];errors=[];types=Counter()
 if len(qids)!=len(set(qids)):raise ValueError('Duplicate question IDs')
 for i,row in enumerate(submission,1):
  try:validate(row)
  except (ValueError,TypeError) as e:errors.append({'line':i,'error':str(e)})
  if isinstance(row,dict):ids.append(row.get('id') if isinstance(row.get('id'),str) else None);types[str(row.get('type'))]+=1
 if len(ids)!=len(qids):errors.append({'error':'line_count_mismatch','expected':len(qids),'actual':len(ids)})
 if len(ids)!=len(set(ids)):errors.append({'error':'duplicate_output_ids'})
 if ids!=qids:errors.append({'error':'id_or_order_mismatch'})
 return {'passed':not errors,'expected':len(qids),'rows':len(submission),'unique_ids':len(set(ids)),'by_type':dict(types),'errors':errors}

def main():
 pa=argparse.ArgumentParser();pa.add_argument('--questions',required=True);pa.add_argument('--submission',required=True);pa.add_argument('--report',required=True);a=pa.parse_args()
 def reject(s):raise ValueError('Forbidden JSON numeric constant '+s)
 def read(path):return [json.loads(l,parse_constant=reject) for l in Path(path).read_text(encoding='utf-8-sig').splitlines() if l.strip()]
 result=audit(read(a.questions),read(a.submission));result['sha256']=hashlib.sha256(Path(a.submission).read_bytes()).hexdigest();result['bytes']=Path(a.submission).stat().st_size
 Path(a.report).write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False))
 if not result['passed']:raise SystemExit(1)
if __name__=='__main__':main()
