"""Bounded train/evaluate/public stages. Parent and original artifacts are immutable."""
from pathlib import Path
import json,sys,os,time,subprocess
from relearning import atomic,sha,seal,verify_export
from quality_gate import promotion

def main(path):
 c=json.loads(Path(path).read_text());out=Path(c['session_output']);out.mkdir(parents=True,exist_ok=False);started=time.monotonic();here=Path(__file__).parent
 def stage(name,args,limit):
  remaining=c['total_remaining_seconds']-(time.monotonic()-started)-60
  if remaining<=0:raise TimeoutError('Global GPU budget exhausted')
  atomic(out/'PHASE.json',{'phase':name,'state':'running','elapsed':time.monotonic()-started})
  with (out/(name+'_PRIVATE.log')).open('w',encoding='utf8') as log:
   result=subprocess.run([sys.executable,*args],cwd=here,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','HF_DEACTIVATE_ASYNC_LOAD':'1'},stdout=log,stderr=subprocess.STDOUT,timeout=min(limit,remaining))
  if result.returncode:raise RuntimeError(name+' failed; retain private log')
 try:
  source=Path(c['verified_collection']);assert verify_export(source)==c['collection_manifest_sha256'];assert json.loads((source/'QUALITY_GATE_RESULT.json').read_text())['state']=='passed'
  stage('parent_calibration',[str(here/'locked_eval.py'),str(path),c['adapter'],str(out/'parent_calibration')],3300)
  stage('training',[str(here/'train_selected.py'),str(path)],1500)
  receipt=json.loads((out/'CANDIDATE_RECEIPT.json').read_text());candidate=Path(receipt['run_path'])/'checkpoint'
  stage('candidate_calibration',[str(here/'locked_eval.py'),str(path),str(candidate),str(out/'candidate_calibration')],3300)
  before=json.loads((out/'parent_calibration/EVALUATION.json').read_text());after=json.loads((out/'candidate_calibration/EVALUATION.json').read_text());decision=promotion(before,after,c['promotion_policy']);atomic(out/'MODEL_DECISION.json',decision)
  selected=str(candidate) if decision['promote'] else c['adapter'];atomic(out/'PUBLIC_MODEL_SELECTION.json',{'adapter':selected,'candidate_selected':decision['promote'],'parent_preserved':True})
  stage('public_test',[str(here/'locked_eval.py'),str(path),selected,str(out/'public_test'),'public'],5400)
  assert verify_export(Path(receipt['run_path']))==receipt['manifest_sha256']
  atomic(out/'WORKFLOW_COMPLETED.json',{'state':'completed','decision':decision,'gpu_seconds':time.monotonic()-started,'candidate_sha256':receipt['adapter_sha256'],'public_complete':True,'best_model_sync_pending':True})
 except Exception as exc:
  atomic(out/'FAILURE.json',{'exception_type':type(exc).__name__,'elapsed':time.monotonic()-started});raise
 finally:seal(out)
if __name__=='__main__':main(sys.argv[1])
