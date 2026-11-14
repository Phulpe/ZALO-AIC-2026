"""Resume the frozen v2.1 public journal; no training or organizer submission.
Restore source/links and exact versions first. Run outside app/tools.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,fcntl,hashlib,json,os,subprocess,sys,time


def main():
 pa=argparse.ArgumentParser();pa.add_argument('--home',default='/content/drive/MyDrive/ZAIC_9B_Repair_v21_20261107');a=pa.parse_args()
 p=Path('/content/zaic_9b_v21_20261107');h=Path(a.home);public=Path('/content/drive/MyDrive/zaic/public_test');out=h/'public_test_v21_frozen'
 read=lambda f:json.loads(Path(f).read_text())
 sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
 def write(f,v):
  f=Path(f);tmp=f.with_name(f.name+'.tmp');tmp.write_text(json.dumps(v,indent=2));os.replace(tmp,f)
 def status(stage,**extra):
  v={'stage':stage,'utc':datetime.now(timezone.utc).isoformat(),**extra};write(h/'PUBLIC_RESUME_STATUS.json',v);print('RESUME_STATUS',json.dumps(v),flush=True)
 def verify():
  frozen=read(h/'V21_SOURCE_FROZEN.json')
  for folder in ['app','tools']:assert {f.name:sha(f) for f in (p/folder).glob('*.py')}==frozen[folder+'_sha256'],'Frozen source changed'
  fp=read(out/'fingerprint.json')
  assert sha(p/'candidate/final/adapter_model.safetensors')==fp['adapter_sha256']
  assert read(p/'model_lock.json')==fp['model']
  assert not (p/'abstention_policy.json').exists() and fp['abstention_policy_sha256'] is None
  assert fp['batch_size']==4 and fp['max_input_tokens']==3072 and fp['max_new_tokens']==320 and not fp['thinking']
 try:
  lock=(h/'PUBLIC_RESUME.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  verify();assert read(h/'V21_CONFIRMATION_GATE.json')['passed']
  import torch
  from importlib.metadata import version
  assert torch.cuda.is_available(),'GPU unavailable; retain checkpoint'
  for package,v in read(p/'versions.json').items():assert version(package)==v,'Dependency version mismatch: '+package
  # Runtime validates the full input/source fingerprint and recovers only complete journal records.
  initial=[json.loads(line) for line in (out/'plans.jsonl').read_bytes().splitlines(keepends=True) if line.endswith(b'\n')]
  status('public_resuming',initial_complete_plan_records=len({r['id'] for r in initial}),prior_observed_runtime_seconds=3095.17)
  cmd=[sys.executable,'-u',str(p/'tools/infer_batched.py'),'--project',str(p),'--root',str(public),'--questions',str(public/'questions.jsonl'),'--variant','candidate','--output-dir',str(out),'--batch-size','4','--max-input-tokens','3072','--cache-dir',str(p/'public_cache')]
  with (h/'public_resume_v21.log').open('a') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTORCH_ALLOC_CONF':'expandable_segments:True'})
  status('auditing')
  subprocess.run([sys.executable,str(p/'tools/audit_output.py'),'--questions',str(public/'questions.jsonl'),'--submission',str(out/'submission.jsonl'),'--report',str(out/'format_audit.json')],check=True)
  verify();r=read(out/'runtime.json');audit=read(out/'format_audit.json')
  assert r['n']==1000 and r['question_logs']==1000 and audit['passed'] and not r['loader_errors'] and r['network_attempts']==0 and r['oom_retries']==0
  # Independently require 1,000 unique complete plans and logs, no unfinished placeholders.
  plans=[json.loads(line) for line in (out/'plans.jsonl').read_bytes().splitlines(keepends=True) if line.endswith(b'\n')]
  logs=[json.loads(line) for line in (out/'logs/questions.jsonl').read_text().splitlines() if line.strip()]
  assert len(plans)==len({x['id'] for x in plans})==len(logs)==len({x['id'] for x in logs})==1000
  result={'state':'completed_after_resume','calibration':read(h/'candidate_calibration/evaluation.json'),'fresh_confirmation':read(h/'candidate_confirmation2/evaluation.json'),'public_runtime':r,'format_audit':audit,'plan_records':1000,'prior_interrupted_runtime_seconds_observed_at_least':3095.17,'public_PA':None,'submitted_to_organizer':False,'training_repeated':False,'docker_tested':False}
  write(h/'REPAIR_V21_RESUMED_COMPLETED.json',result);status('completed',sha256=audit['sha256'],public_PA=None)
 except BaseException as exc:
  status('failed',error_type=type(exc).__name__,error=str(exc));raise


if __name__=='__main__':main()
