"""Stage approved TRAIN supervision and build CPU-only real-input preflight."""
from pathlib import Path
import json,sys,shutil,zipfile,hashlib
exp=Path(__file__).parent;sys.path.insert(0,str(exp/'src'))
from relearning import sha,verify_export,atomic,seal
root=Path('D:/ZAIC/PRIVATE_DATA');proposal=json.loads((exp/'NEXT_RUN_PROPOSAL.json').read_text(encoding='utf-8'));bank=Path(proposal['frozen_dataset_root']);assert verify_export(bank)==proposal['frozen_dataset_manifest_sha256']
stage=root/'v4_kaggle_input_20261110';stage.mkdir(exist_ok=False);bundle=stage/'bundle';bundle.mkdir();shutil.copytree(bank,bundle/'BANK')
old=root/'quality_round_v2_selected/PRIVATE_SELECTED.json';old_rows=json.loads(old.read_text(encoding='utf-8'));critical=sorted(r['id'] for r in old_rows if r['kind']=='correction');assert len(critical)==7
atomic(bundle/'CRITICAL_IDS.json',{'ids':critical,'reference_file_sha256':sha(old),'scope':'previous verified TRAIN corrections; fixed before new model run'})
run_config=json.loads((root/'zaic/result_811/TIME_FOCUSED_V3_20261109/time_pilot/model_runs/20261109T125033322737Z_b786bc50/run_config.json').read_text(encoding='utf-8'))
control={'bank_manifest_sha256':verify_export(bank),'selected_dataset_sha256':proposal['selected_dataset_sha256'],'source_archive_sha256':proposal['source_archive_sha256'],'base_hashes':run_config['base_hashes'],'parent_sha256':proposal['parent_adapter_sha256'],'revision':proposal['base_revision'],'runtime_manifest_sha256':run_config['runtime_manifest_sha256'],'data_manifest_sha256':run_config['data_manifest_sha256'],'split_manifest_sha256':run_config['split_manifest_sha256'],'selection_counts':{'correction':9,'replay':9,'full_regression':103},'competition_permission_reference':run_config['competition_permission_evidence'],'gpu_authorized':False,'optimizer_authorized':False}
atomic(bundle/'CONTROL.json',control);bundle_hash=seal(bundle)
upload=stage/'upload';upload.mkdir()
with zipfile.ZipFile(upload/'FROZEN_V4.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in bundle.rglob('*'):
  if p.is_file():z.write(p,p.relative_to(bundle).as_posix())
receipt={'bundle_manifest_sha256':bundle_hash,'bank_manifest_sha256':verify_export(bank),'archive_sha256':sha(upload/'FROZEN_V4.zip'),'scope':'Private TRAIN verified SFT and regression only; no raw transaction tables, gold answers, public/holdout rows, unresolved journals, credentials or model/base shards','origin':'Existing authorized private TRAIN runtime and E7 predictions saved on Kaggle367303123; local CPU-verified bank','files':json.loads((bundle/'ARTIFACT_MANIFEST.json').read_text(encoding='utf-8'))['files']};atomic(upload/'TRANSFER_RECEIPT.json',receipt)
atomic(upload/'dataset-metadata.json',{'id':'lnghongphc/zaic-v4-frozen-correction-replay-20261110','title':'ZAIC v4 Frozen Correction Replay 20261110','licenses':[{'name':'other'}]})
deploy=exp/'KAGGLE_REAL_PREFLIGHT';deploy.mkdir(exist_ok=False);body=(exp/'KAGGLE_CPU_TESTS_R5/cpu_tests.py').read_text(encoding='utf-8')
extra='''
from relearning import verify_export,sha,atomic,seal
import shutil,struct
assert not __import__('torch').cuda.is_available(),'CPU preflight must not allocate GPU'
markers=list(Path('/kaggle/input').rglob('TRANSFER_RECEIPT.json'));assert len(markers)==1
receipt=json.loads(markers[0].read_text());assert receipt['bundle_manifest_sha256']==BUNDLE_HASH
archive=markers[0].parent/'FROZEN_V4.zip'
if archive.is_file():
 assert sha(archive)==receipt['archive_sha256']
 bundle=Path('/kaggle/temp/zaic_v4_input');bundle.mkdir()
 with zipfile.ZipFile(archive) as z:
  for f in z.infolist():assert (bundle/f.filename).resolve().is_relative_to(bundle.resolve()) and not stat.S_ISLNK(f.external_attr>>16)
  z.extractall(bundle)
else:
 candidates=[p.parent for p in markers[0].parent.rglob('ARTIFACT_MANIFEST.json') if sha(p)==BUNDLE_HASH];assert len(candidates)==1;bundle=candidates[0]
assert verify_export(bundle)==BUNDLE_HASH
bank=bundle/'BANK';c=json.loads((bundle/'CONTROL.json').read_text());assert verify_export(bank)==c['bank_manifest_sha256']
assert c['gpu_authorized'] is False and c['optimizer_authorized'] is False and c['source_archive_sha256']==SOURCE_HASH
bases=list(Path('/kaggle/input').rglob('BASE_UPLOAD_MANIFEST.json'));runtimes=list(Path('/kaggle/input').rglob('RUNTIME_DATA_MANIFEST.json'));adapters=list(Path('/kaggle/input').rglob('final/adapter_model.safetensors'));assert len(bases)==len(runtimes)==len(adapters)==1
base=bases[0].parent/'base';runtime=runtimes[0].parent;adapter=adapters[0].parent
for name,digest in c['base_hashes'].items():assert sha(base/name)==digest
index=json.loads((base/'model.safetensors.index.json').read_text());shards=set(index['weight_map'].values());assert len(shards)==4 and all((base/p).is_file() for p in shards)
assert sha(adapter/'adapter_model.safetensors')==c['parent_sha256']
ac=json.loads((adapter/'adapter_config.json').read_text());assert ac['r']==8
with (adapter/'adapter_model.safetensors').open('rb') as f:
 n=struct.unpack('<Q',f.read(8))[0];assert 0<n<16*1024**2;header=json.loads(f.read(n));assert len(header)>1
assert sha(runtime/'RUNTIME_DATA_MANIFEST.json')==c['runtime_manifest_sha256'];assert sha(runtime/'DATA_MANIFEST.json')==c['data_manifest_sha256'];assert sha(runtime/'SPLIT_MANIFEST.json')==c['split_manifest_sha256']
from runtime_data import load_learning_pool
pool,split,gold=load_learning_pool(runtime,c['data_manifest_sha256'],c['split_manifest_sha256']);assert gold.rows is None;qs={q['id']:q for q in pool}
rows=json.loads((bank/'PRIVATE_BANK.json').read_text());ev=json.loads((bank/'PRIVATE_EVIDENCE.json').read_text());gate=json.loads((bank/'SUPERVISION_GATE.json').read_text())
from quality_gate import select,digest
selected,selection=select(rows,ev,gate);assert selection['dataset_sha256']==c['selected_dataset_sha256'];assert digest(selected)==digest(json.loads((bank/'PRIVATE_SELECTED.json').read_text()))
critical=set(json.loads((bundle/'CRITICAL_IDS.json').read_text())['ids']);assert len(critical)==7 and critical<={r['id'] for r in selected if r['kind']=='correction'}
regression=json.loads((bank/'PRIVATE_REGRESSION_BANK.json').read_text());assert len(regression)==103 and len({r['id'] for r in regression})==103
for r in rows:
 q=qs[r['id']];assert r['split']=='learning' and q['business_id']==r['business_id'] and q['prompt']==r['prompt'] and r['parent_sha256']==c['parent_sha256']
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(base,local_files_only=True,trust_remote_code=False);tok.padding_side='right'
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
from train import CompletionDataset
from time_objective import training_classes
dataset,_,_=training_classes(4.);old=CompletionDataset(selected,tok);new=dataset(selected,tok);assert old.skipped==0 and len(old)==len(new)==18
for a,b in zip(old.items,new.items):
 assert a['input_ids']==b['input_ids'] and a['labels']==b['labels'];n=sum(v!=-100 for v in b['labels']);assert b['labels'][-n:]==b['input_ids'][-n:] and b['labels'][-1]==tok.eos_token_id and not b['time_focus'][-1] and not any(b['time_focus'][:-n])
assert gold.rows is None
out={'state':'passed','gpu_executed':False,'optimizer_executed':False,'base_model_loaded':False,'gold_opened':False,'source_archive_sha256':SOURCE_HASH,'bundle_manifest_sha256':BUNDLE_HASH,'bank_manifest_sha256':c['bank_manifest_sha256'],'base_files_verified':len(c['base_hashes']),'base_shards_verified':len(shards),'parent_adapter_sha256':c['parent_sha256'],'adapter_rank':ac['r'],'selected':selection,'full_replay_regression':103,'critical_old7_present':True,'selected_rows_tokenized':18,'max_tokens':max(len(b['input_ids']) for b in new.items),'completion_mask_shift_eos_valid':True,'resolved_paths':{'base':str(base),'adapter':str(adapter),'runtime':str(runtime),'frozen_regression_root':str(bank)}}
atomic(Path('/kaggle/working/REAL_PREFLIGHT_RESULT.json'),out);print('REAL_PREFLIGHT_AGGREGATE',json.dumps(out));seal(Path('/kaggle/working/source'))
'''
body+='\nBUNDLE_HASH='+repr(bundle_hash)+'\n'+extra;(deploy/'preflight.py').write_text(body,encoding='utf-8');compile(body,'preflight.py','exec')
atomic(deploy/'kernel-metadata.json',{'id':'lnghongphc/zaic-v4-real-input-preflight-20261110','title':'ZAIC v4 Real Input Preflight 20261110','code_file':'preflight.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':False,'enable_internet':False,'dataset_sources':['lnghongphc/zaic-base-qwen35-9b-frozen-c2022362','lnghongphc/zaic-train-runtime-v2-20261108','lnghongphc/zaic-trained-9b-final-checkpoint48-20261108','lnghongphc/zaic-v4-frozen-correction-replay-20261110'],'kernel_sources':[],'competition_sources':[]})
atomic(exp/'REAL_PREFLIGHT_STAGING.json',{'state':'staged','bundle_manifest_sha256':bundle_hash,'archive_sha256':receipt['archive_sha256'],'archive_bytes':(upload/'FROZEN_V4.zip').stat().st_size,'file_inventory':receipt['files'],'GPU_authorized':False,'optimizer_authorized':False});print('STAGED_PRIVATE_TRAIN_ONLY',bundle_hash,(upload/'FROZEN_V4.zip').stat().st_size)
