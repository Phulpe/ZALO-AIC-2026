from pathlib import Path
import json,hashlib,base64,zipfile,io
root=Path(__file__).parent;src=root/'src';buffer=io.BytesIO()
with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
 for f in sorted(src.rglob('*')):
  if f.is_file() and f.suffix in {'.py','.json'} and '__pycache__' not in f.parts and f.name!='TEST_RESULTS.json':z.write(f,f.relative_to(src).as_posix())
payload=buffer.getvalue();digest=hashlib.sha256(payload).hexdigest();(root/'CODE_V3.zip').write_bytes(payload)
config=json.loads(Path('D:/ZAIC/EXPERIMENTS/quality_round_v2_20261109/FINISH_GPU_REQUEST_LOCAL.json').read_text(encoding='utf-8'));prepared=Path('D:/ZAIC/PRIVATE_DATA/time_learning_v3_cpu/temporal_cpu');config.update(action='temporal_pilot',mode='learning',init_mode='warm_start',gpu_confirmed=True,source_sha256=digest,approved_gpu_seconds=7200,total_gpu_seconds=7200,soft_limit_seconds=6600,calibration_min_remaining_seconds=3100,run_public=False,loss_objective='temporal_payload_completion_v1',time_token_weight=4.,lr=1e-5,gradient_accumulation=4,max_rounds=1,max_epochs_equivalent=2,max_steps=32,approved_max_steps=32,max_seconds=900,approved_max_seconds=900,cpu_manifest_sha256=hashlib.sha256((prepared/'ARTIFACT_MANIFEST.json').read_bytes()).hexdigest(),cache_manifest_sha256=hashlib.sha256((prepared/'TRAIN_CACHE/ARTIFACT_MANIFEST.json').read_bytes()).hexdigest(),prior_workflow_manifest_sha256='67ef3491e7c05274cadf34d53c31901018fee3d7963d32e3365c29c700061462',prior_candidate_sha256='28e745628fc1509018f93ea8b5299473d93f73ac907b63fefe9dfb48b3deca99',session_output='/kaggle/working/time_pilot',budget_authorization=json.loads((root/'GPU_AUTHORIZATION.json').read_text(encoding='utf-8')))
config['supervision_gate']['budget']={'approved_gpu_seconds':7200,'rounds':1,'scope':'new temporal TRAIN pilot and evaluation; no public rerun'}
config['supervision_gate']['optimizer_ready']=True
config.update(parent_run_id='9B_stable_48',optimizer_init='new',scheduler_init='new',source_checkpoint=None,training_objective='verified_temporal_correction_replay',test_evidence={'kernel':'lnghongphc/zaic-time-v3-final-cpu-check-20261109/2','baseline_tests':99,'new_tests':131,'pool_rows':124,'max_proposed_tokens':1455},promotion_requires=['positive verified correction gain','zero replay regressions','complete frozen calibration','delta at least0.01','zero calibration correct-to-wrong'])
bootstrap='''from pathlib import Path
import base64,io,zipfile,hashlib,sys,subprocess,json,os,stat,time
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
os.environ['HF_DEACTIVATE_ASYNC_LOAD']='1'
'''+ 'payload=base64.b64decode('+repr(base64.b64encode(payload).decode())+')\n'+f'assert hashlib.sha256(payload).hexdigest()=={digest!r}\n'+'''
root=Path('/kaggle/working/source');root.mkdir()
with zipfile.ZipFile(io.BytesIO(payload)) as z:
 for f in z.infolist():assert (root/f.filename).resolve().is_relative_to(root.resolve()) and not stat.S_ISLNK(f.external_attr>>16)
 z.extractall(root)
p=subprocess.run([sys.executable,'-X','utf8','run_checks.py'],cwd=root,capture_output=True,text=True);print(p.stdout);assert p.returncode==0
'''
common='CONFIG='+repr(config)+'\n'+'''
bases=list(Path('/kaggle/input').rglob('BASE_UPLOAD_MANIFEST.json'));runtimes=list(Path('/kaggle/input').rglob('RUNTIME_DATA_MANIFEST.json'));collections=[p.parent for p in Path('/kaggle/input').rglob('QUALITY_GATE_RESULT.json') if p.parent.name==CONFIG['verified_collection_run_id']];prepareds=list(Path('/kaggle/input').rglob('CPU_PREPARATION_RESULT.json'))
assert len(bases)==len(runtimes)==len(collections)==len(prepareds)==1
CONFIG.update(base=str(bases[0].parent/'base'),runtime_root=str(runtimes[0].parent),verified_collection=str(collections[0]),cpu_prepared=str(prepareds[0].parent))
'''
cpu=root/'FINAL_CPU';cpu.mkdir(exist_ok=True);(cpu/'cpu.py').write_text(bootstrap+common+'''
request=root/'FINAL_CPU_REQUEST.json';request.write_text(json.dumps(CONFIG,indent=2))
p=subprocess.run([sys.executable,str(root/'final_cpu_probe.py'),str(request)],cwd=root);assert p.returncode==0
''',encoding='utf-8')
metadata={'id':'lnghongphc/zaic-time-v3-final-cpu-check-20261109','title':'ZAIC Time v3 Final CPU Check 20261109','code_file':'cpu.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':False,'enable_internet':False,'dataset_sources':['lnghongphc/zaic-base-qwen35-9b-frozen-c2022362','lnghongphc/zaic-train-runtime-v2-20261108'],'kernel_sources':['lnghongphc/zaic-time-v3-cpu-preflight-20261109/1','lnghongphc/zaic-quality-round-v2-20261109/3'],'competition_sources':[]};(cpu/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
gpu=root/'KAGGLE_GPU';gpu.mkdir(exist_ok=True);bnb=(src/'OFFLINE_BNB_BOOTSTRAP.py').read_text(encoding='utf-8');tail='''
refs=list(Path('/kaggle/input').rglob('WORKFLOW_COMPLETED.json'));assert len(refs)==1
adapters=[p for p in Path('/kaggle/input').rglob('final/adapter_model.safetensors') if 'checkpoint' not in p.parent.parts];assert len(adapters)==1
CONFIG.update(adapter=str(adapters[0].parent),prior_workflow=str(refs[0].parent))
request=root/'TIME_PILOT_REQUEST.json';request.write_text(json.dumps(CONFIG,indent=2))
p=subprocess.run([sys.executable,str(root/'time_pilot.py'),str(request)],cwd=root,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'},timeout=6750);assert p.returncode==0,'Pilot failed; preserve outputs and inspect filtered private evidence'
''';(gpu/'gpu.py').write_text(bootstrap+'\n'+bnb+'\n'+common+tail,encoding='utf-8');gpu_meta={**metadata,'id':'lnghongphc/zaic-time-focused-learning-v3-20261109','title':'ZAIC Time Focused Learning v3 20261109','code_file':'gpu.py','enable_gpu':True,'dataset_sources':metadata['dataset_sources']+['lnghongphc/zaic-trained-9b-final-checkpoint48-20261108','lnghongphc/zaic-official-bnb-0502-wheel-20261108'],'kernel_sources':metadata['kernel_sources']+['lnghongphc/zaic-quality-train-eval-public-20261109/1']};(gpu/'kernel-metadata.json').write_text(json.dumps(gpu_meta,indent=2),encoding='utf-8')
(root/'FROZEN_RUN_CONFIG.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8');(root/'FINAL_SOURCE_SHA256.txt').write_text(digest+'\n');print(json.dumps({'source_sha256':digest,'source_bytes':len(payload),'cpu_only_stage_prepared':True,'gpu_started':False}))
