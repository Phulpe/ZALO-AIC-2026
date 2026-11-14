from pathlib import Path
import json
exp=Path(__file__).parent
folder=exp/'CPU_PRESERVATION';folder.mkdir(exist_ok=False)
code='''from pathlib import Path
import json,hashlib,zipfile,stat,shutil,datetime
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
marks=list(Path('/kaggle/input').rglob('STATE_UPLOAD_INVENTORY.json'));assert len(marks)==1
meta=json.loads(marks[0].read_text());assert meta['training_manifest_sha256']=='c51d0b0f422ebb93f9a6fd927f9c7a3a6cb8d3eefffdf5987e89dca423863aaf'
out=Path('/kaggle/working/preserved_training/20261109T125033322737Z_b786bc50');out.mkdir(parents=True)
archive=marks[0].parent/'PRESERVED_STATE.zip'
if archive.is_file():
 assert sha(archive)==meta['archive_sha256']
 with zipfile.ZipFile(archive) as z:
  assert {p.filename for p in z.infolist()}==set(meta['files'])
  for item in z.infolist():assert (out/item.filename).resolve().is_relative_to(out.resolve()) and not stat.S_ISLNK(item.external_attr>>16)
  z.extractall(out)
else:
 matches=[p for p in marks[0].parent.rglob('ARTIFACT_MANIFEST.json') if sha(p)==meta['training_manifest_sha256']];assert len(matches)==1;origin=matches[0].parent.resolve()
 for name,record in meta['files'].items():
  p=(origin/name).resolve();q=(out/name).resolve();assert p.is_relative_to(origin) and q.is_relative_to(out.resolve()) and not p.is_symlink();assert sha(p)==record['sha256'] and p.stat().st_size==record['bytes'];q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
assert {p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file()}==set(meta['files'])
for name,record in meta['files'].items():assert sha(out/name)==record['sha256'] and (out/name).stat().st_size==record['bytes']
assert sha(out/'ARTIFACT_MANIFEST.json')==meta['training_manifest_sha256']
manifest=json.loads((out/'ARTIFACT_MANIFEST.json').read_text())
for name,record in manifest['files'].items():assert sha(out/name)==record['sha256'] and (out/name).stat().st_size==record['bytes']
result={'state':'completed','observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'run_id':out.name,'training_manifest_sha256':meta['training_manifest_sha256'],'verified_files':len(meta['files']),'verified_bytes':sum(r['bytes'] for r in meta['files'].values()),'checkpoint_sets':len(list(out.rglob('adapter_model.safetensors'))),'gpu_executed':False,'optimizer_executed':False,'pickle_loaded':False,'no_per_question_payload':True,'source_gpu_version':3,'source_gpu_wrapper_status':'ERROR_after_training_completed','training_state':json.loads((out/'training_result.json').read_text())['state']}
Path('/kaggle/working/PRESERVATION_RECEIPT.json').write_text(json.dumps(result,indent=2));print('CHECKPOINT_PRESERVATION_VERIFIED',json.dumps(result))
'''
(folder/'preserve.py').write_text(code,encoding='utf-8')
meta={'id':'lnghongphc/zaic-time-v3-checkpoint-preservation-20261109','title':'ZAIC Time v3 Checkpoint Preservation 20261109','code_file':'preserve.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':False,'enable_internet':False,'dataset_sources':['lnghongphc/zaic-time-v3-checkpoint-states-20261109'],'kernel_sources':[],'competition_sources':[]}
(folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');compile(code,'preserve.py','exec');print('CPU_PRESERVATION_READY')
