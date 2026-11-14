"""TRAIN-only executor diagnostics using saved model plans, never test labels."""
from pathlib import Path
from collections import Counter,defaultdict
from copy import deepcopy
import argparse,hashlib,json,sys,traceback


def rows(path):
 return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
 pa=argparse.ArgumentParser()
 for name in ['project','root','questions','plans','output-dir']:pa.add_argument('--'+name,required=True)
 pa.add_argument('--reference-submission')
 a=pa.parse_args();p=Path(a.project);root=Path(a.root).resolve();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False)
 assert root.name=='train' and (root/'answers.jsonl').is_file(),'TRAIN only'
 sys.path.insert(0,str(p/'app'))
 from planner import answer
 from legacy_engine import prepare_shop
 from scoring import matches,summarize
 qs=rows(a.questions);official={r['id']:r for r in rows(root/'questions.jsonl')};gold={r['id']:r for r in rows(root/'answers.jsonl')}
 assert len(qs)==len({r['id'] for r in qs})
 assert all(q==official[q['id']] and q['id'] in gold for q in qs),'Evaluation questions must be official TRAIN records'
 plans={r['id']:r for r in rows(a.plans)}
 assert {q['id'] for q in qs}<=set(plans),'Incomplete saved plan run'
 from infer_batched import fingerprint_inputs
 _,business_files=fingerprint_inputs(root,{q['business_id'] for q in qs})
 cache_namespaces={biz:p/'train_cache'/hashlib.sha256(json.dumps({k:v for k,v in business_files.items() if k.startswith('data/'+biz+'/')},sort_keys=True).encode()).hexdigest() for biz in {q['business_id'] for q in qs}}
 reference={r['id']:r for r in rows(a.reference_submission)} if a.reference_submission else {}
 results=[];preds=[];groups=defaultdict(Counter);errors=Counter();transitions=Counter();ratio_bins=Counter();reasons=Counter()
 for biz in sorted({q['business_id'] for q in qs}):
  shop=prepare_shop(root/'data'/biz,cache_namespaces[biz])
  for q in [q for q in qs if q['business_id']==biz]:
   rawplan=deepcopy(plans[q['id']]['plan'])
   try:body=answer(shop,q['question'],q['as_of_date'],rawplan) if rawplan is not None else {'type':'abstain','reason':'invalid_saved_plan'}
   except Exception as exc:
    errors[type(exc).__name__]+=1;body={'type':'abstain','reason':'executor_'+type(exc).__name__}
   reason=body.get('reason');reasons[reason]+=bool(reason)
   pred={'id':q['id'],**{k:v for k,v in body.items() if k in {'type','answer'}}}
   state='abstain' if pred['type']=='abstain' else 'correct' if matches(pred,gold[q['id']]) else 'wrong'
   if q['id'] in reference:
    prev=reference[q['id']];oldstate='abstain' if prev['type']=='abstain' else 'correct' if matches(prev,gold[q['id']]) else 'wrong';transitions[oldstate+' -> '+state]+=1
   pl=plans[q['id']]['plan'] or {};key='|'.join(str(pl.get(k)) for k in ['metric','group_by','answer_type'])
   groups[key][state]+=1
   if state=='wrong' and pred['type']=='number' and gold[q['id']]['type']=='number':
    x=pred['answer'];y=gold[q['id']]['answer']
    if y and x:
     ratio=abs(x/y)
     label=next((str(k)+'x' for k in [0.001,0.01,0.1,10,100,1000] if abs(ratio-k)<=abs(k)*1e-8),'other')
     ratio_bins[label]+=1
   results.append({'id':q['id'],'state':state,'pred_type':pred['type'],'gold_type':gold[q['id']]['type'],'reason':reason})
   preds.append(pred)
  print('REPLAY_PROGRESS',len(results),'/',len(qs),flush=True)
 metric=summarize(results)
 summary={'state':'completed','scope':'TRAIN-only saved-plan executor diagnostic; no new model generation','evaluation':metric,'groups':dict(groups),'transitions':dict(transitions),'uncaught_execution_errors':dict(errors),'reasons':{k:v for k,v in reasons.items() if k},'wrong_numeric_ratio_bins':dict(ratio_bins),'questions_sha256':hashlib.sha256(Path(a.questions).read_bytes()).hexdigest(),'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((p/'app').glob('*.py'))},'public_used':False}
 byid={r['id']:r for r in preds}
 (out/'submission.jsonl').write_text(''.join(json.dumps(byid[q['id']],ensure_ascii=False)+'\n' for q in qs))
 (out/'details.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results))
 (out/'evaluation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
 print('REPLAY_SUMMARY',json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
