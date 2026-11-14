from pathlib import Path
import json,hashlib,base64,zipfile,io,ast
root=Path(__file__).parent
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(root.rglob('*')):
        if p.is_file() and (p.suffix in {'.py','.pyw'} or p.name in {'model_lock.json','SOURCE_SHA256.json'}) and not any(x in p.parts for x in ['__pycache__','synthetic_runs']):
            z.write(p,p.relative_to(root).as_posix())
payload=buf.getvalue();(root/'RELEARNING_CODE_ONLY.zip').write_bytes(payload)
digest=hashlib.sha256(payload).hexdigest()
launch='''from pathlib import Path
import base64,hashlib,json,zipfile,io,subprocess,sys,datetime
payload=base64.b64decode(PAYLOAD)
assert hashlib.sha256(payload).hexdigest()==EXPECTED
workspace=Path('/kaggle/working/relearning_v2_')/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
workspace.mkdir(parents=True,exist_ok=False)
with zipfile.ZipFile(io.BytesIO(payload)) as z:
    assert sum(i.file_size for i in z.infolist())<10*1024*1024
    for i in z.infolist():
        assert (workspace/i.filename).resolve().is_relative_to(workspace.resolve())
    z.extractall(workspace)
result=subprocess.run([sys.executable,'-X','utf8','run_checks.py'],cwd=workspace,capture_output=True,text=True)
print(result.stdout)
assert result.returncode==0,'CPU tests failed: see private TEST_DETAILS.txt; no GPU/training started'
demo=subprocess.run([sys.executable,'synthetic_demo.py'],cwd=workspace,capture_output=True,text=True)
print(demo.stdout)
assert demo.returncode==0,'Synthetic loop failed'
'''.replace('PAYLOAD',repr(base64.b64encode(payload).decode())).replace('EXPECTED',repr(digest))
config={'action':'preflight','init_mode':'warm_start','mode':'learning','gpu_confirmed':False,'competition_permission_evidence':{'status':'unverified'},'max_steps':1,'approved_max_steps':1,'max_seconds':180,'approved_max_seconds':180,'lr':5e-6,'gradient_accumulation':1,'learning_businesses':['synthetic'],'revision':'c202236235762e1c871ad0ccb60c8ee5ba337b9a','parent_sha256':'e7dfa635beccce128c99f9d3fd993d33dbb4c88e73e3ddeccbc2bc915eec4a4d','user_authorized':True}
config['competition_permission_evidence']=json.loads((root/'PERMISSION_DECISION.json').read_text(encoding='utf-8'))
manifest_path=Path('D:/ZAIC/KAGGLE_STAGING/20261107T191003Z/01_BASE_QWEN35_9B/BASE_UPLOAD_MANIFEST.json')
config['base_hashes']={k:v['sha256'] for k,v in json.loads(manifest_path.read_text(encoding='utf-8'))['files'].items()}
gpu='''# This block stays disabled during CPU preflight. Never Run All with training enabled.
if CONFIG['action']=='gpu_smoke':
    assert CONFIG['gpu_confirmed'], 'Review smoke GPU budget first'
    from permission import permitted
    assert permitted(CONFIG['competition_permission_evidence']), 'Permission not accepted for this run'
    sys.path.insert(0,str(workspace))
    from train_warm import train
    from relearning import sha
    base_manifests=list(Path('/kaggle/input').rglob('BASE_UPLOAD_MANIFEST.json'))
    adapters=list(Path('/kaggle/input').rglob('final/adapter_model.safetensors'))
    assert len(base_manifests)==len(adapters)==1
    CONFIG['base']=str(base_manifests[0].parent/'base')
    assert CONFIG['base_hashes']=={k:v['sha256'] for k,v in json.loads(base_manifests[0].read_text())['files'].items()}, 'Base source manifest mismatch'
    CONFIG['adapter']=str(adapters[0].parent)
    CONFIG['output_root']='/kaggle/working/relearning_model_runs'
    from smoke_targets import targets
    result=train(CONFIG,targets())
    print(json.dumps({'gpu_smoke_run':result.name,'gpu_session_must_stop_now':True}))
elif CONFIG['action']!='preflight':
    raise ValueError('Long training is not enabled by this notebook')
'''
def cell(kind,text):
    obj={'cell_type':kind,'metadata':{},'source':text.splitlines(keepends=True)}
    if kind=='code':obj.update(execution_count=None,outputs=[]);ast.parse(text)
    return obj
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3'},'kaggle':{'accelerator':'none','isInternetEnabled':False}},'cells':[cell('markdown','# ZAIC Relearning v1 — synthetic CPU preflight\n\nNew experiment, baseline unchanged. Default: 99 baseline tests + new tests + synthetic correction/replay round. No contest data, no GPU. Warm-start GPU block stays gated by permission evidence and explicit smoke configuration. Stop session and set Accelerator=None immediately after GPU work. Full resume is separate and disabled.'),cell('code','CONFIG='+repr(config)),cell('code',launch),cell('code',gpu)]}
(root/'ZAIC_RELEARNING_V2.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8')
(root/'PACKAGE_MANIFEST.json').write_text(json.dumps({'zip_sha256':digest,'files':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.py') if '__pycache__' not in p.parts and 'synthetic_runs' not in p.parts}},indent=2),encoding='utf-8')
print('Notebook and code-only archive ready:',digest)
