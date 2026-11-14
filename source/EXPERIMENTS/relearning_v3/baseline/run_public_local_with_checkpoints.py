"""Run frozen offline inference locally; a separate parent saves immutable Drive checkpoints.
Requires the user's existing Colab authentication. Does not share files or submit.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,io,json,os,signal,subprocess,sys,time
import google.auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

HOME=Path('/content/zaic_v21_final_run')
PARENT='1bPHpPmC29_qar42kEIPeXSGgbg_PxNkh'

def main():
 creds,_=google.auth.default();api=build('drive','v3',credentials=creds,cache_discovery=False)
 assert api.about().get(fields='user(emailAddress)').execute()['user']['emailAddress']=='n26dcat045@student.ptithcm.edu.vn','Wrong account'
 local=HOME/'public_test_v21_frozen';manifest=[];last_count=-1;worker=None
 def status(stage,**extra):
  v={'stage':stage,'utc':datetime.now(timezone.utc).isoformat(),**extra};tmp=HOME/'LOCAL_PUBLIC_STATUS.tmp';tmp.write_text(json.dumps(v,indent=2));os.replace(tmp,HOME/'LOCAL_PUBLIC_STATUS.json');print('PUBLIC_LOCAL_STATUS',json.dumps(v),flush=True)
 def upload(name,data,parent=PARENT,mime='application/octet-stream'):
  md5=hashlib.md5(data).hexdigest()
  q="'"+parent+"' in parents and name='"+name+"' and trashed=false"
  for attempt in range(3):
   found=api.files().list(q=q,fields='files(id,name,md5Checksum,size)',pageSize=100).execute()['files']
   assert len(found)<=1,'Duplicate immutable output names'
   if found:
    result=found[0];assert result.get('md5Checksum')==md5,'Existing checkpoint checksum differs';return result
   try:
    result=api.files().create(body={'name':name,'parents':[parent]},media_body=MediaIoBaseUpload(io.BytesIO(data),mimetype=mime,resumable=False),fields='id,name,md5Checksum,size').execute()
    assert result['md5Checksum']==md5 and int(result['size'])==len(data),'Uploaded bytes differ'
    return result
   except Exception:
    if attempt==2:raise
    time.sleep(2)
 def snapshot(force=False):
  nonlocal last_count
  data=(local/'plans.jsonl').read_bytes();complete=b''.join(line for line in data.splitlines(keepends=True) if line.endswith(b'\n'))
  rows=[json.loads(line) for line in complete.splitlines()];n=len({r['id'] for r in rows});assert n==len(rows) and n>=679,'Journal lost records'
  if n==last_count or (not force and n-last_count<16):return
  sha=hashlib.sha256(complete).hexdigest();result=upload(f'v21_plans_{n:04d}_{sha[:12]}.jsonl',complete)
  manifest.append({'plans':n,'sha256':sha,'md5':result['md5Checksum'],'drive_file_id':result['id'],'utc':datetime.now(timezone.utc).isoformat()});last_count=n
  (HOME/'DRIVE_CHECKPOINT_MANIFEST.json').write_text(json.dumps(manifest,indent=2));status('running',saved_plans=n,drive_checkpoint_verified=True)
 try:
  snapshot(True)
  worker=subprocess.Popen([sys.executable,'-u','/content/resume_public_v21_local.py','--home',str(HOME)],stdout=(HOME/'resume_controller.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTORCH_ALLOC_CONF':'expandable_segments:True'})
  (HOME/'WORKER_PID.json').write_text(json.dumps({'controller':os.getpid(),'worker_group':worker.pid}))
  while worker.poll() is None:snapshot();time.sleep(15)
  assert worker.returncode==0,'Inference controller failed; inspect local logs'
  snapshot(True)
  completed=json.loads((HOME/'REPAIR_V21_RESUMED_COMPLETED.json').read_text());assert completed['format_audit']['passed'] and completed['plan_records']==1000
  status('uploading_verified_result',saved_plans=1000)
  folder=api.files().create(body={'name':'public_test_v21_verified_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'),'mimeType':'application/vnd.google-apps.folder','parents':[PARENT]},fields='id').execute()['id']
  uploads={}
  for name in ['submission.jsonl','plans.jsonl','runtime.json','format_audit.json','fingerprint.json']:
   uploads[name]=upload(name,(local/name).read_bytes(),folder)
  uploads['question_logs.jsonl']=upload('question_logs.jsonl',(local/'logs/questions.jsonl').read_bytes(),folder)
  for name in ['REPAIR_V21_RESUMED_COMPLETED.json','DRIVE_CHECKPOINT_MANIFEST.json']:
   uploads[name]=upload(name,(HOME/name).read_bytes(),folder)
  uploads['JOURNAL_RECOVERY_VERIFIED.json']=upload('JOURNAL_RECOVERY_VERIFIED.json',(local/'JOURNAL_RECOVERY_VERIFIED.json').read_bytes(),folder)
  result={'state':'completed_and_uploaded','folder_id':folder,'folder_url':'https://drive.google.com/drive/folders/'+folder,'submission_file_id':uploads['submission.jsonl']['id'],'submission_url':'https://drive.google.com/file/d/'+uploads['submission.jsonl']['id']+'/view?authuser=1','uploads':uploads,'public_PA':None,'submitted_to_organizer':False}
  (HOME/'DRIVE_VERIFIED_RESULT.json').write_text(json.dumps(result,indent=2));status('completed',saved_plans=1000,submission_sha256=completed['format_audit']['sha256'],folder_id=folder,public_PA=None)
 except BaseException as exc:
  if worker and worker.poll() is None:
   try:os.killpg(worker.pid,signal.SIGTERM)
   except ProcessLookupError:pass
  status('failed',error_type=type(exc).__name__,error=str(exc));raise

if __name__=='__main__':main()
