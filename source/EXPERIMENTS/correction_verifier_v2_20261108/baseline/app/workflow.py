"""Train, compare on calibration only, then open the frozen audit once."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,subprocess,sys,time

def stage(app,p,label,cmd):
 with (p/(label+'.log')).open('w') as log:
  proc=subprocess.Popen(cmd,cwd=str(app),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env={**os.environ,'PYTORCH_ALLOC_CONF':'expandable_segments:True'})
  for line in proc.stdout:log.write(line);log.flush();print(label+': '+line,end='',flush=True)
  rc=proc.wait()
 if rc:raise RuntimeError(label+' failed; inspect '+label+'.log')

def main():
 pa=argparse.ArgumentParser();pa.add_argument('--project',required=True);pa.add_argument('--train-root',required=True);a=pa.parse_args();p=Path(a.project);app=Path(__file__).resolve().parent;root=Path(a.train_root)
 assert (p/'data/manifest.json').is_file()
 if not (p/'smoke_model.json').is_file():stage(app,p,'MODEL_SMOKE_RETRY',[sys.executable,'-u',str(app/'smoke_model.py'),'--project',str(p)])
 if not (p/'training_complete.json').is_file():stage(app,p,'TRAIN_9B',[sys.executable,'-u',str(app/'train.py'),'--project',str(p)])
 for variant in ['base','candidate']:
  out=p/('calibration_'+variant)
  if not (out/'evaluation.json').exists():
   stage(app,p,'CALIBRATION_'+variant,[sys.executable,'-u',str(app/'infer.py'),'--project',str(p),'--root',str(root),'--questions',str(p/'data/calibration_questions.jsonl'),'--variant',variant,'--output-dir',str(out)])
   stage(app,p,'SCORE_'+variant,[sys.executable,'-u',str(app/'evaluate_saved.py'),'--root',str(root),'--questions',str(p/'data/calibration_questions.jsonl'),'--output-dir',str(out)])
 comparisons={v:json.loads((p/('calibration_'+v)/'evaluation.json').read_text()) for v in ['base','candidate']}
 # Tie goes to the unmodified instruction model, avoiding unsupported claims
 # that extra training always improves generalization.
 winner=max(comparisons,key=lambda v:(comparisons[v]['PA'],v=='base'))
 selection={'model':json.loads((p/'model_lock.json').read_text()),'variant':winner,'calibration':comparisons,'selected_using':'calibration PA only; audit not yet used','promoted_over_4b':False,'public_score_4b':-.207,'public_score_9b':None,'network_api_allowed':False}
 (p/'selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2))
 out=p/'audit_selected'
 if not (out/'evaluation.json').exists():
  stage(app,p,'AUDIT_SELECTED',[sys.executable,'-u',str(app/'infer.py'),'--project',str(p),'--root',str(root),'--questions',str(p/'data/audit_questions.jsonl'),'--variant',winner,'--output-dir',str(out)])
  stage(app,p,'SCORE_AUDIT',[sys.executable,'-u',str(app/'evaluate_saved.py'),'--root',str(root),'--questions',str(p/'data/audit_questions.jsonl'),'--output-dir',str(out)])
 audit=json.loads((out/'evaluation.json').read_text());complete={'state':'completed','selection':selection,'audit':audit,'training':json.loads((p/'training_complete.json').read_text()),'model_smoke':json.loads((p/'smoke_model.json').read_text()),'public_test_used_for_training':False,'public_test_rerun':False,'docker_tested':False,'note':'9B initial candidate trained and evaluated on small frozen train-derived sets; no claim of public-test improvement until a separate public run and official scoring.'}
 (p/'UPGRADE_COMPLETED.json').write_text(json.dumps(complete,ensure_ascii=False,indent=2));print('UPGRADE_9B_COMPLETED:',json.dumps(complete,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
