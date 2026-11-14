"""Frozen TRAIN calibration -> fresh confirmation 2 -> audited public inference.
No training, public-label access, manual answer edits or organizer submission.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,subprocess,sys,time


def main():
 pa=argparse.ArgumentParser();pa.add_argument('--project',required=True);pa.add_argument('--home',required=True);pa.add_argument('--run-public',action='store_true');a=pa.parse_args()
 p=Path(a.project);h=Path(a.home);train=Path('/content/drive/MyDrive/zaic/train');start=time.monotonic()
 def read(f):return json.loads(Path(f).read_text())
 def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
 def write(f,v):
  f=Path(f);tmp=f.with_name(f.name+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2));os.replace(tmp,f)
 def status(stage,**extra):
  v={'stage':stage,'utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-start,**extra};write(h/'REPAIR_STATUS.json',v);print('STATUS',json.dumps(v),flush=True)
 def verify_source():
  frozen=read(h/'V21_SOURCE_FROZEN.json')
  for folder in ['app','tools']:
   assert {f.name:sha(f) for f in sorted((p/folder).glob('*.py'))}==frozen[folder+'_sha256'],'Source changed during run'
  assert not (p/'abstention_policy.json').exists(),'Raw variant frozen'
 def run(label,cmd):
  status(label)
  with (h/(label+'.log')).open('w') as log:
   rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTORCH_ALLOC_CONF':'expandable_segments:True'}).returncode
  if rc:raise RuntimeError(label+' exited '+str(rc))
 def infer(stage):
  out=h/('candidate_'+stage);q=h/(stage+'_questions.jsonl')
  run(stage,[sys.executable,'-u',str(p/'tools/infer_batched.py'),'--project',str(p),'--root',str(train),'--questions',str(q),'--variant','candidate','--output-dir',str(out),'--batch-size','4','--max-input-tokens','3072','--cache-dir',str(p/'train_cache')])
  run(stage+'_score',[sys.executable,str(p/'app/evaluate_saved.py'),'--root',str(train),'--questions',str(q),'--output-dir',str(out)])
  verify_source();m=read(out/'evaluation.json');r=read(out/'runtime.json')
  assert not r['loader_errors'] and r['network_attempts']==0 and r['oom_retries']==0,'Runtime quality gate failed'
  return m,r
 try:
  verify_source();assert read(h/'LOADER_VERIFIED.json')['loaded']
  freeze=read(h/'EVALUATION_FREEZE.json');assert sha(h/'calibration_questions.jsonl')==freeze['stages']['calibration']['questions_sha256']
  c2=read(h/'CONFIRMATION2_FREEZE.json')
  # Frozen before the ingestion repair. Match its recorded hash without opening labels.
  digest=sha(h/'confirmation2_questions.jsonl')
  assert digest in json.dumps(c2),'Confirmation 2 digest does not match the pre-repair freeze'
  calibration,cr=infer('calibration');assert calibration['n']==200
  selected={'variant':'raw_v21','calibration':calibration,'selection_rule':'PA >= 0.30 baseline, no loader/network/OOM errors; no policy fitted','confirmed_before_confirmation2':True,'public_used':False}
  write(h/'V21_SELECTION.json',selected)
  if calibration['PA']<0.30:status('completed_not_promoted',reason='Calibration below baseline');return
  confirmation,rr=infer('confirmation2');assert confirmation['n']==100
  gate=confirmation['PA']>0 and confirmation['correct']+confirmation['wrong']>=10
  write(h/'V21_CONFIRMATION_GATE.json',{'passed':gate,'evaluation':confirmation,'loader_errors':rr['loader_errors'],'min_answered':10,'fresh_question_hash':digest,'public_used':False,'selection_frozen_before_confirmation':True,'note':'TRAIN development only; not a public score. Historical business exposure disclosed.'})
  if not gate:status('completed_not_promoted',reason='Fresh confirmation gate failed');return
  if not a.run_public:status('completed_ready_for_public');return
  public=Path('/content/drive/MyDrive/zaic/public_test');out=h/'public_test_v21_frozen'
  run('public_v21',[sys.executable,'-u',str(p/'tools/infer_batched.py'),'--project',str(p),'--root',str(public),'--questions',str(public/'questions.jsonl'),'--variant','candidate','--output-dir',str(out),'--batch-size','4','--max-input-tokens','3072','--cache-dir',str(p/'public_cache')])
  run('public_audit',[sys.executable,str(p/'tools/audit_output.py'),'--questions',str(public/'questions.jsonl'),'--submission',str(out/'submission.jsonl'),'--report',str(out/'format_audit.json')])
  runtime=read(out/'runtime.json');audit=read(out/'format_audit.json');verify_source()
  assert runtime['n']==1000 and runtime['question_logs']==1000 and audit['passed'] and not runtime['loader_errors'] and runtime['network_attempts']==0 and runtime['oom_retries']==0
  result={'state':'completed','calibration':calibration,'fresh_confirmation':confirmation,'public_runtime':runtime,'format_audit':audit,'public_PA':None,'submitted_to_organizer':False,'docker_tested':False,'training_repeated':False}
  write(h/'REPAIR_V21_COMPLETED.json',result);status('completed',output=str(out),public_PA=None)
 except BaseException as exc:
  status('failed',error_type=type(exc).__name__,error=str(exc));raise


if __name__=='__main__':main()
