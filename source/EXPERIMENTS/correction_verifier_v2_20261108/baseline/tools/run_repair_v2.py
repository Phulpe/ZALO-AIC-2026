"""TRAIN calibration -> frozen policy choice -> confirmation -> optional public.
Never trains, changes answer values, or submits to the organizer.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,shutil,subprocess,sys,time


def main():
 pa=argparse.ArgumentParser();pa.add_argument('--project',required=True);pa.add_argument('--home',required=True);pa.add_argument('--run-public',action='store_true');a=pa.parse_args()
 p=Path(a.project);h=Path(a.home);train=Path('/content/drive/MyDrive/zaic/train');ev=Path('/content/zaic_v2_eval');began=time.monotonic()
 def read(f):return json.loads(Path(f).read_text())
 def write(f,value):
  f=Path(f);tmp=f.with_name(f.name+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));os.replace(tmp,f)
 def status(stage,**extra):
  value={'stage':stage,'utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-began,**extra};write(h/'REPAIR_STATUS.json',value);print('REPAIR_STATUS',json.dumps(value,ensure_ascii=False),flush=True)
 def run(label,cmd):
  status(label)
  with (h/(label+'.log')).open('w') as log:
   proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTORCH_ALLOC_CONF':'expandable_segments:True'})
   for line in proc.stdout:log.write(line);log.flush();print(label+': '+line,end='',flush=True)
   rc=proc.wait()
  if rc:raise RuntimeError(label+' exited '+str(rc))
 def infer(label,stage,out):
  run(label,[sys.executable,'-u',str(p/'tools/infer_batched.py'),'--project',str(p),'--root',str(train),'--questions',str(ev/(stage+'_questions.jsonl')),'--variant','candidate','--output-dir',str(out),'--batch-size','4','--max-input-tokens','3072','--cache-dir',str(p/'train_cache')])
  run(label+'_score',[sys.executable,str(p/'app/evaluate_saved.py'),'--root',str(train),'--questions',str(ev/(stage+'_questions.jsonl')),'--output-dir',str(out)])
 def verify_source():
  actual={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((p/'app').glob('*.py'))}
  assert actual==read(h/'V2_SOURCE_FROZEN.json')['app_sha256'],'App changed during experiment'
 try:
  assert train.name=='train' and (train/'answers.jsonl').is_file()
  frozen=read(h/'EVALUATION_FREEZE.json')
  for stage in ['calibration','confirmation']:
   assert hashlib.sha256((ev/(stage+'_questions.jsonl')).read_bytes()).hexdigest()==frozen['stages'][stage]['questions_sha256']
  verify_source();assert not (p/'abstention_policy.json').exists()
  status('waiting_for_frozen_baseline')
  while not (h/'BASELINE_DONE.json').exists():
   if (h/'BASELINE_FAILED.json').exists():raise RuntimeError('Frozen baseline failed')
   time.sleep(10)
  baseline=read(h/'baseline_calibration_200/evaluation.json');assert baseline['n']==200
  run('executor_replay_200',[sys.executable,'-u',str(p/'tools/replay_train.py'),'--project',str(p),'--root',str(train),'--questions',str(ev/'calibration_questions.jsonl'),'--plans',str(h/'baseline_calibration_200/plans.jsonl'),'--reference-submission',str(h/'baseline_calibration_200/submission.jsonl'),'--output-dir',str(h/'executor_replay_v2_200')])
  raw=h/'candidate_calibration_200_raw';infer('candidate_calibration_200','calibration',raw)
  verify_source();raw_metric=read(raw/'evaluation.json');assert raw_metric['n']==200
  policy_file=h/'learned_abstention_policy.json'
  run('fit_calibration_policy',[sys.executable,str(p/'tools/fit_abstention.py'),'--project',str(p),'--train-root',str(train),'--questions',str(ev/'calibration_questions.jsonl'),'--raw-output',str(raw),'--policy-output',str(policy_file)])
  filtered=read(h/'policy_calibration_evaluation.json')['evaluation']
  use_policy=filtered['PA']>raw_metric['PA']
  selected=filtered if use_policy else raw_metric
  if use_policy:shutil.copy2(policy_file,p/'abstention_policy.json')
  selection={'selected_using':'Frozen TRAIN calibration PA only; confirmation still unopened','variant':'policy' if use_policy else 'raw','baseline':baseline,'raw':raw_metric,'policy':filtered,'selected':selected,'policy_sha256':hashlib.sha256(policy_file.read_bytes()).hexdigest() if use_policy else None,'public_used':False}
  write(h/'V2_SELECTION.json',selection)
  if selected['PA']<=0 or selected['PA']<baseline['PA']:
   status('completed_not_promoted',reason='No positive calibration improvement',selection=selection);return
  confirmation=h/'candidate_confirmation_100';infer('candidate_confirmation_100','confirmation',confirmation)
  verify_source();confirm=read(confirmation/'evaluation.json');assert confirm['n']==100
  answered=confirm['correct']+confirm['wrong']
  gate=confirm['PA']>0 and answered>=10 and confirm['correct']>confirm['wrong']
  write(h/'V2_CONFIRMATION_GATE.json',{'passed':gate,'evaluation':confirm,'min_answered':10,'selection_frozen_before_confirmation':True,'public_used':False,'note':'Small confirmation sample, historical development exposure disclosed; this does not guarantee public PA.'})
  if not gate:
   status('completed_not_promoted',reason='Independent confirmation gate failed',confirmation=confirm);return
  if not a.run_public:
   status('completed_ready_for_public',confirmation=confirm);return
  public=Path('/content/drive/MyDrive/zaic/public_test');out=h/'public_test_v2_frozen'
  run('public_v2',[sys.executable,'-u',str(p/'tools/infer_batched.py'),'--project',str(p),'--root',str(public),'--questions',str(public/'questions.jsonl'),'--variant','candidate','--output-dir',str(out),'--batch-size','4','--max-input-tokens','3072','--cache-dir',str(p/'public_cache')])
  run('audit_public_v2',[sys.executable,str(p/'tools/audit_output.py'),'--questions',str(public/'questions.jsonl'),'--submission',str(out/'submission.jsonl'),'--report',str(out/'format_audit.json')])
  audit=read(out/'format_audit.json');runtime=read(out/'runtime.json');verify_source()
  assert audit['passed'] and runtime['n']==1000 and not runtime['loader_errors'] and runtime['network_attempts']==0
  result={'state':'completed','selection':selection,'confirmation':confirm,'public_runtime':runtime,'format_audit':audit,'public_PA':None,'submitted_to_organizer':False,'docker_tested':False,'training_repeated':False}
  write(h/'REPAIR_V2_COMPLETED.json',result);status('completed',public_output=str(out),public_PA=None)
 except BaseException as exc:
  status('failed',error_type=type(exc).__name__,error=str(exc));raise


if __name__=='__main__':main()
