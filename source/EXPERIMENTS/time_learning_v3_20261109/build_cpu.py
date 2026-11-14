from pathlib import Path
import json,hashlib,base64,zipfile,io
root=Path(__file__).parent;src=root/'src';folder=root/'KAGGLE_CPU';folder.mkdir(exist_ok=True)
b=io.BytesIO()
with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
 for f in sorted(src.rglob('*')):
  if f.is_file() and f.suffix in {'.py','.json'} and '__pycache__' not in f.parts and f.name not in {'TEST_RESULTS.json'}:z.write(f,f.relative_to(src).as_posix())
payload=b.getvalue();digest=hashlib.sha256(payload).hexdigest();config=json.loads(Path('D:/ZAIC/EXPERIMENTS/quality_round_v2_20261109/FINISH_GPU_REQUEST_LOCAL.json').read_text(encoding='utf-8'));config.update(mode='preflight',gpu_confirmed=False,cpu_output='/kaggle/working/temporal_cpu',source_sha256=digest,trusted_cache_manifest_sha256='f88577a80b62263bf0676fc1ef7f7bd732c92a27033adedd532dfabde5109297')
code='''from pathlib import Path
import base64,io,zipfile,hashlib,sys,subprocess,json,os,stat
'''+ 'payload=base64.b64decode('+repr(base64.b64encode(payload).decode())+')\n'+f"assert hashlib.sha256(payload).hexdigest()=={digest!r}\n"+'''
root=Path('/kaggle/working/source');root.mkdir()
with zipfile.ZipFile(io.BytesIO(payload)) as z:
 for f in z.infolist():assert (root/f.filename).resolve().is_relative_to(root.resolve()) and not stat.S_ISLNK(f.external_attr>>16)
 z.extractall(root)
p=subprocess.run([sys.executable,'-X','utf8','run_checks.py'],cwd=root,capture_output=True,text=True);print(p.stdout);assert p.returncode==0
'''+ 'CONFIG='+repr(config)+'\n'+'''
bases=list(Path('/kaggle/input').rglob('BASE_UPLOAD_MANIFEST.json'));runtimes=list(Path('/kaggle/input').rglob('RUNTIME_DATA_MANIFEST.json'));collections=[p.parent for p in Path('/kaggle/input').rglob('QUALITY_GATE_RESULT.json') if p.parent.name==CONFIG['verified_collection_run_id']];caches=list(Path('/kaggle/input').rglob('CACHE_PROVENANCE.json'))
assert len(bases)==len(runtimes)==len(collections)==len(caches)==1
CONFIG.update(base=str(bases[0].parent/'base'),runtime_root=str(runtimes[0].parent),verified_collection=str(collections[0]),trusted_cache_root=str(caches[0].parent))
request=root/'CPU_REQUEST.json';request.write_text(json.dumps(CONFIG,indent=2))
p=subprocess.run([sys.executable,str(root/'cpu_prepare.py'),str(request)],cwd=root,timeout=1500);assert p.returncode==0,'CPU preflight failed; inspect private log locally'
'''
(folder/'cpu.py').write_text(code,encoding='utf-8');meta={'id':'lnghongphc/zaic-time-v3-cpu-preflight-20261109','title':'ZAIC Time v3 CPU Preflight 20261109','code_file':'cpu.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':False,'enable_internet':False,'dataset_sources':['lnghongphc/zaic-base-qwen35-9b-frozen-c2022362','lnghongphc/zaic-train-runtime-v2-20261108'],'kernel_sources':['lnghongphc/zaic-quality-round-v2-20261109/3','lnghongphc/zaic-trusted-train-cache-20261109/1'],'competition_sources':[]};(folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');(root/'CPU_SOURCE_SHA256.txt').write_text(digest+'\n');print(json.dumps({'source_sha256':digest,'source_bytes':len(payload),'gpu_enabled':False}))
