"""Fit only on frozen TRAIN calibration. The confirmation set is not an input."""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,hashlib,json,math,sys


def wilson_lower(correct,total,z=1.6448536269514722):
 if total==0:return 0.0
 p=correct/total;zz=z*z
 return (p+zz/(2*total)-z*math.sqrt(p*(1-p)/total+zz/(4*total*total)))/(1+zz/total)


def main():
 pa=argparse.ArgumentParser()
 for key in ['project','train-root','questions','raw-output','policy-output']:pa.add_argument('--'+key,required=True)
 pa.add_argument('--min-answers',type=int,default=8);a=pa.parse_args()
 p=Path(a.project);root=Path(a.train_root).resolve();qfile=Path(a.questions);out=Path(a.raw_output)
 assert root.name=='train' and (root/'answers.jsonl').is_file()
 assert qfile.name=='calibration_questions.jsonl','Only the frozen calibration file may fit the policy'
 sys.path.insert(0,str(p/'app'))
 from abstention_policy import signature,apply_policy
 from scoring import matches,summarize
 def rows(f):return [json.loads(l) for l in Path(f).read_text().splitlines() if l.strip()]
 qs=rows(qfile);official={r['id']:r for r in rows(root/'questions.jsonl')};gold={r['id']:r for r in rows(root/'answers.jsonl')}
 assert all(q==official[q['id']] for q in qs)
 preds={r['id']:r for r in rows(out/'submission.jsonl')};plans={r['id']:r['plan'] for r in rows(out/'plans.jsonl')}
 assert set(preds)==set(plans)=={q['id'] for q in qs}
 groups=defaultdict(Counter)
 for q in qs:
  pred=preds[q['id']]
  if pred['type'] in {'abstain','unanswerable'}:continue
  g=groups[signature(plans[q['id']],pred)];g['answered']+=1;g['correct']+=int(matches(pred,gold[q['id']]))
 evidence={k:{**dict(v),'wrong':v['answered']-v['correct'],'wilson90_lower':wilson_lower(v['correct'],v['answered'])} for k,v in groups.items()}
 allowed=sorted(k for k,v in evidence.items() if v['answered']>=a.min_answers and v['wilson90_lower']>0.5)
 policy={'version':'calibrated-abstention-v1','allowed_signatures':allowed,'min_answers':a.min_answers,'selection_rule':'Wilson 90% lower bound > 0.5 on concrete answered predictions; n>=min_answers','verified_unanswerable_kept':True,'calibration_questions_sha256':hashlib.sha256(qfile.read_bytes()).hexdigest(),'calibration_evidence':evidence,'public_used':False,'confirmation_used':False}
 path=Path(a.policy_output);assert not path.exists(),'Do not overwrite a frozen policy';path.write_text(json.dumps(policy,ensure_ascii=False,indent=2))
 filtered=[];states=[]
 for q in qs:
  pred=preds[q['id']];body=apply_policy(plans[q['id']] or {},{k:v for k,v in pred.items() if k!='id'},policy);new={'id':q['id'],**{k:v for k,v in body.items() if k in {'type','answer'}}};filtered.append(new)
  states.append({'state':'abstain' if new['type']=='abstain' else 'correct' if matches(new,gold[q['id']]) else 'wrong'})
 result={'scope':'policy fitted and scored on calibration only; confirmation still unopened','evaluation':summarize(states),'allowed_signatures':allowed,'evidence':evidence}
 (path.parent/'policy_calibration_evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 (path.parent/'policy_calibration_submission.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in filtered))
 print('POLICY_FITTED',json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
