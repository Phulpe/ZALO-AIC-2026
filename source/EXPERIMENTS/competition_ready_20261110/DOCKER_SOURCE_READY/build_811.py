from pathlib import Path
import json,ast,zipfile,io,hashlib,base64
root=Path(__file__).parent
source=Path('D:/ZAIC/EXPERIMENTS/relearning_v3/ZAIC_RELEARNING_V3_READY.ipynb')
old=json.loads(source.read_text(encoding='utf-8'))
config=ast.literal_eval(''.join(old['cells'][1]['source']).split('=',1)[1])
package=json.loads((root/'INPUT_PACKAGE.json').read_text())
provenance=json.loads(Path(package['bundle']).joinpath('TRAIN_PROVENANCE.json').read_text())
config.update(action='preflight',budget_approved=True,gpu_confirmed=False,training_objective='verified_replay_refresh',max_steps=4,approved_max_steps=4,max_seconds=900,approved_max_seconds=900,total_gpu_seconds=3600,gradient_accumulation=4,learning_businesses=provenance['learning_businesses'],output_root='/kaggle/working/result_811/model_runs',session_output='/kaggle/working/result_811',bundle_manifest_sha256=package['bundle_manifest_sha256'],data_manifest_sha256=provenance['data_manifest_sha256'],split_manifest_sha256=provenance['split_manifest_sha256'],source_train_questions_sha256=provenance['raw_train_questions_sha256'],public_questions_sha256=package['public_questions_sha256'])
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(root.rglob('*')):
        if p.is_file() and (p.suffix=='.py' or p.name in {'SOURCE_SHA256.json','model_lock.json'}) and '__pycache__' not in p.parts:z.write(p,p.relative_to(root).as_posix())
payload=buf.getvalue();digest=hashlib.sha256(payload).hexdigest();(root/'CODE_811.zip').write_bytes(payload)
launch=''.join(old['cells'][2]['source'])
# Reuse the audited extraction/check pattern with the new immutable source archive.
launch=launch[launch.index('from pathlib import Path'):]
start=launch.index('payload=base64.b64decode(');end=launch.index('\n',start)
launch=launch[:start]+'payload=base64.b64decode('+repr(base64.b64encode(payload).decode())+')'+launch[end:]
line=next(x for x in launch.splitlines() if x.startswith('assert hashlib.sha256(payload)'))
launch=launch.replace(line,'assert hashlib.sha256(payload).hexdigest()=='+repr(digest)).replace('relearning_v3_','run811_source_')
verify="""sys.path[:0]=[str(workspace),str(workspace/'baseline/app'),str(workspace/'baseline/tools')]
from relearning import verify_export,verify_parent
from workflow_811 import validate_request
bases=list(Path('/kaggle/input').rglob('BASE_UPLOAD_MANIFEST.json'))
adapters=list(Path('/kaggle/input').rglob('final/adapter_model.safetensors'))
bundles=list(Path('/kaggle/input').rglob('TRAIN_PROVENANCE.json'))
runtimes=list(Path('/kaggle/input').rglob('RUNTIME_DATA_MANIFEST.json'))
assert len(bases)==len(adapters)==len(bundles)==len(runtimes)==1
CONFIG['base']=str(bases[0].parent/'base');CONFIG['adapter']=str(adapters[0].parent)
CONFIG['bundle']=str(bundles[0].parent);CONFIG['runtime_root']=str(runtimes[0].parent)
assert verify_export(Path(CONFIG['bundle']))==CONFIG['bundle_manifest_sha256']
verify_parent(CONFIG['adapter'],CONFIG['parent_sha256'])
for name,expected in CONFIG['base_hashes'].items():
    p=Path(CONFIG['base'])/name;h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    assert h.hexdigest()==expected
assert hashlib.sha256(runtimes[0].read_bytes()).hexdigest()=='523db34eb8e8bab52aa52d1d046307f19dbb3668495025ca05d66abd1e4c6e70'
rows=json.loads((Path(CONFIG['bundle'])/'TRAIN_VERIFIED_REPLAY.json').read_text())
provenance=json.loads(bundles[0].read_text());validate_request(CONFIG,rows,provenance)
sys.path[:0]=[str(workspace),str(workspace/'baseline/app'),str(workspace/'baseline/tools')]
from transformers import AutoTokenizer
from train import CompletionDataset
tokenizer=AutoTokenizer.from_pretrained(CONFIG['base'],local_files_only=True,trust_remote_code=False)
dataset=CompletionDataset(rows,tokenizer,2048)
assert dataset.skipped==0 and len(dataset)==len(rows)==10
receipt={'state':'passed','base_files_verified':len(CONFIG['base_hashes']),'train_rows':10,'overlength_targets':0,'bundle_manifest_sha256':CONFIG['bundle_manifest_sha256'],'parent_sha256':CONFIG['parent_sha256'],'public_questions':1000,'public_gold_included':False,'source_zip_sha256':SOURCE_HASH}
Path('/kaggle/working/PREFLIGHT_811.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
""".replace('SOURCE_HASH',repr(digest))
bootstrap=(root/'OFFLINE_BNB_BOOTSTRAP.py').read_text(encoding='utf-8')
gpu="""if CONFIG['action']=='train_public':
    assert CONFIG['budget_approved'] and CONFIG['gpu_confirmed']
    exec(BOOTSTRAP)
    sys.path[:0]=[str(workspace),str(workspace/'baseline/app'),str(workspace/'baseline/tools')]
    from runtime_gpu import single_gpu_env
    bases=list(Path('/kaggle/input').rglob('BASE_UPLOAD_MANIFEST.json'))
    adapters=list(Path('/kaggle/input').rglob('final/adapter_model.safetensors'))
    bundles=list(Path('/kaggle/input').rglob('TRAIN_PROVENANCE.json'))
    runtimes=list(Path('/kaggle/input').rglob('RUNTIME_DATA_MANIFEST.json'))
    assert len(bases)==len(adapters)==len(bundles)==len(runtimes)==1
    CONFIG.update(base=str(bases[0].parent/'base'),adapter=str(adapters[0].parent),bundle=str(bundles[0].parent),runtime_root=str(runtimes[0].parent))
    request=workspace/'RUN_811_REQUEST.json';request.write_text(json.dumps(CONFIG,indent=2))
    log=(workspace/'WORKFLOW_811_PRIVATE.log').open('w',encoding='utf-8')
    began=time.monotonic()
    job=subprocess.Popen([sys.executable,str(workspace/'workflow_811.py'),'workflow',str(request)],cwd=workspace,env=single_gpu_env(os.environ),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        last=None
        while job.poll() is None:
            status=Path(CONFIG['session_output'])/'PHASE_STATUS.json'
            progress=Path(CONFIG['session_output'])/'public_test/PROGRESS.json'
            phase=json.loads(status.read_text()) if status.exists() else {'phase':'initializing'}
            if progress.exists():phase.update(json.loads(progress.read_text()))
            text=json.dumps(phase)
            if text!=last:print('RUN811_PROGRESS',text,flush=True);last=text
            if time.monotonic()-began>3330:raise TimeoutError('Global GPU budget stop')
            time.sleep(15)
        assert job.returncode==0,'Workflow failed; preserve outputs and stop GPU'
        print((Path(CONFIG['session_output'])/'TRAIN_RECEIPT.json').read_text())
        print((Path(CONFIG['session_output'])/'PUBLIC_SUMMARY.json').read_text())
        print((Path(CONFIG['session_output'])/'REGRESSION_SUMMARY.json').read_text())
        print((Path(CONFIG['session_output'])/'WORKFLOW_RESULT.json').read_text())
    finally:
        if job.poll() is None:
            import signal
            os.killpg(job.pid,signal.SIGTERM)
            try:job.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(job.pid,signal.SIGKILL);job.wait()
        log.close()
        print('Save output now; stop GPU immediately; Accelerator=None. Download after shutdown.',flush=True)
elif CONFIG['action']!='preflight':raise ValueError('Unknown action; no fallback')
""".replace('BOOTSTRAP',repr(bootstrap))
gpu='import time,os\n'+gpu
def cell(kind,text):
    obj={'cell_type':kind,'metadata':{},'source':text.splitlines(keepends=True)}
    if kind=='code':ast.parse(text);obj.update(outputs=[],execution_count=None)
    return obj
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3'},'kaggle':{'accelerator':'none','isInternetEnabled':False}},'cells':[cell('markdown','# ZAIC result_811 — verified TRAIN pilot then public_test\n\nUser approved 4 optimizer steps and at most 60 GPU minutes for the whole task. Warm-start original 9B adapter; 10 verified TRAIN replay targets, accumulation=4. No new corrections, no synthetic training, no full-pool completion claim. Candidate public inference: 1000 questions, no public gold or weight updates, fixed decoder/guards. Preserve BEST_MODEL. Default CPU preflight. Save outputs then stop GPU immediately.'),cell('code','CONFIG='+repr(config)),cell('code',launch),cell('code',verify),cell('code',gpu)]}
(root/'ZAIC_TRAIN_PUBLIC_811.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8')
(root/'RUN_CONFIG_811.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
(root/'BUILD_RECEIPT.json').write_text(json.dumps({'code_zip_sha256':digest,'input_manifest_sha256':package['bundle_manifest_sha256'],'notebook_sha256':hashlib.sha256((root/'ZAIC_TRAIN_PUBLIC_811.ipynb').read_bytes()).hexdigest(),'code_cells_ast_valid':True},indent=2))
print('811 code/notebook built',digest)
