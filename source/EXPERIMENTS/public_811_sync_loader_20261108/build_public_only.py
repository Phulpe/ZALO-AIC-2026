from pathlib import Path
import json,hashlib,base64,zipfile,io,ast
root=Path(__file__).parent
config=json.loads(Path('D:/ZAIC/EXPERIMENTS/train_public_811_20261108/RUN_CONFIG_811.json').read_text(encoding='utf-8'))
config.update(action='preflight',gpu_confirmed=False,session_output='/kaggle/working/result_811_public',prior_gpu_seconds=1500,max_public_seconds=1800)
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(root.rglob('*')):
  if p.is_file() and (p.suffix=='.py' or p.name in {'SOURCE_SHA256.json','model_lock.json'}) and '__pycache__' not in p.parts:z.write(p,p.relative_to(root).as_posix())
payload=buf.getvalue();digest=hashlib.sha256(payload).hexdigest();(root/'PUBLIC_CODE_811.zip').write_bytes(payload)
old=json.loads(Path('D:/ZAIC/EXPERIMENTS/train_public_811_20261108/ZAIC_TRAIN_PUBLIC_811.ipynb').read_text(encoding='utf-8'))
launch=''.join(old['cells'][2]['source']);start=launch.index('payload=base64.b64decode(');end=launch.index('\n',start)
launch=launch[:start]+'payload=base64.b64decode('+repr(base64.b64encode(payload).decode())+')'+launch[end:]
line=next(x for x in launch.splitlines() if x.startswith('assert hashlib.sha256(payload)'))
launch=launch.replace(line,'assert hashlib.sha256(payload).hexdigest()=='+repr(digest)).replace('run811_source_','public811_source_')
verify="""sys.path[:0]=[str(workspace),str(workspace/'baseline/app'),str(workspace/'baseline/tools')]
from relearning import verify_export,sha
from public_only_811 import validate_candidate,MANIFEST,CANDIDATE
bases=list(Path('/kaggle/input').rglob('BASE_UPLOAD_MANIFEST.json'))
receipts=list(Path('/kaggle/input').rglob('TRAIN_RECEIPT.json'))
bundles=list(Path('/kaggle/input').rglob('TRAIN_PROVENANCE.json'))
runtimes=list(Path('/kaggle/input').rglob('RUNTIME_DATA_MANIFEST.json'))
assert len(bases)==len(receipts)==len(bundles)==len(runtimes)==1
CONFIG.update(base=str(bases[0].parent/'base'),saved_train_receipt=str(receipts[0]),bundle=str(bundles[0].parent),runtime_root=str(runtimes[0].parent))
receipt=json.loads(receipts[0].read_text());validate_candidate(receipt)
candidate=receipts[0].parent/'model_runs'/receipt['run_id']
assert verify_export(candidate)==MANIFEST
assert sha(candidate/'checkpoint/adapter_model.safetensors')==CANDIDATE
assert verify_export(bundles[0].parent)==CONFIG['bundle_manifest_sha256']
for name,expected in CONFIG['base_hashes'].items():assert sha(Path(CONFIG['base'])/name)==expected
assert (workspace/'audit_output.py').is_file()
ready={'state':'passed','inference_only':True,'training_repeated':False,'candidate_sha256':CANDIDATE,'checkpoint_manifest_sha256':MANIFEST,'public_questions':1000,'base_files_verified':15,'source_zip_sha256':SOURCE_HASH}
Path('/kaggle/working/PUBLIC_PREFLIGHT_811.json').write_text(json.dumps(ready,indent=2))
print(json.dumps(ready))
""".replace('SOURCE_HASH',repr(digest))
bootstrap=(root/'OFFLINE_BNB_BOOTSTRAP.py').read_text(encoding='utf-8')
gpu="""import os,time,signal
if CONFIG['action']=='public_inference':
 assert CONFIG['gpu_confirmed'] and CONFIG['budget_approved']
 exec(BOOTSTRAP)
 sys.path[:0]=[str(workspace),str(workspace/'baseline/app'),str(workspace/'baseline/tools')]
 from runtime_gpu import single_gpu_env
 bases=list(Path('/kaggle/input').rglob('BASE_UPLOAD_MANIFEST.json'))
 receipts=list(Path('/kaggle/input').rglob('TRAIN_RECEIPT.json'))
 bundles=list(Path('/kaggle/input').rglob('TRAIN_PROVENANCE.json'))
 runtimes=list(Path('/kaggle/input').rglob('RUNTIME_DATA_MANIFEST.json'))
 assert len(bases)==len(receipts)==len(bundles)==len(runtimes)==1
 CONFIG.update(base=str(bases[0].parent/'base'),saved_train_receipt=str(receipts[0]),bundle=str(bundles[0].parent),runtime_root=str(runtimes[0].parent))
 request=workspace/'PUBLIC_REQUEST_811.json';request.write_text(json.dumps(CONFIG,indent=2))
 log=(workspace/'PUBLIC_ONLY_PRIVATE.log').open('w',encoding='utf-8');started=time.monotonic()
 job=subprocess.Popen([sys.executable,str(workspace/'public_only_811.py'),str(request)],cwd=workspace,env=single_gpu_env(os.environ),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try:
  last=None
  load_stage_seen=None
  while job.poll() is None:
   progress=Path(CONFIG['session_output'])/'public_test/PROGRESS.json'
   status=json.loads(progress.read_text()) if progress.exists() else {'phase':'loading_candidate','training_repeated':False}
   if status.get('stage')=='loading_base':
    if load_stage_seen is None:load_stage_seen=time.monotonic()
    if time.monotonic()-load_stage_seen>240:raise TimeoutError('Base loading stalled for four minutes; preserve stack dump')
   else:load_stage_seen=None
   text=json.dumps(status)
   if text!=last:print('PUBLIC811_PROGRESS',text,flush=True);last=text
   if time.monotonic()-started>CONFIG['max_public_seconds']:raise TimeoutError('Remaining GPU budget reached')
   time.sleep(15)
  assert job.returncode==0,'Public inference failed; preserve partial outputs and stop GPU'
  for name in ['PUBLIC_SUMMARY.json','REGRESSION_SUMMARY.json','WORKFLOW_RESULT.json']:print((Path(CONFIG['session_output'])/name).read_text())
 finally:
  if job.poll() is None:
   os.killpg(job.pid,signal.SIGTERM)
   try:job.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(job.pid,signal.SIGKILL);job.wait()
  log.close();print('Save results, stop GPU now, Accelerator=None. Download after shutdown.',flush=True)
elif CONFIG['action']!='preflight':raise ValueError('Inference-only notebook: no training route')
""".replace('BOOTSTRAP',repr(bootstrap))
def cell(kind,text):
 c={'cell_type':kind,'metadata':{},'source':text.splitlines(keepends=True)}
 if kind=='code':ast.parse(text);c.update(outputs=[],execution_count=None)
 return c
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3'},'kaggle':{'accelerator':'none','isInternetEnabled':False}},'cells':[cell('markdown','# Public result_811 — inference only from completed four-step candidate\n\nTraining is already complete and preserved in version 366886259. This notebook never retrains. Candidate hash b3063abe404bc394a943360f61f60f2d1f400d35340813c0e303f742c4260e85 is pinned. Packaging fix adds the unchanged audit_output.py companion. Fixed decoder/guards; no public gold or weight updates. Earlier GPU sessions used about 25 minutes; this inference session is limited to 30 minutes plus shutdown margin within the approved total 60 minutes. Default preflight and Accelerator=None.'),cell('code','CONFIG='+repr(config)),cell('code',launch),cell('code',verify),cell('code',gpu)]}
(root/'ZAIC_PUBLIC_ONLY_811.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8')
(root/'PUBLIC_BUILD_RECEIPT.json').write_text(json.dumps({'source_zip_sha256':digest,'candidate_sha256':'b3063abe404bc394a943360f61f60f2d1f400d35340813c0e303f742c4260e85','training_repeated':False,'public_gpu_limit_seconds':1800,'code_ast_valid':True},indent=2))
print('Inference-only notebook built',digest)
