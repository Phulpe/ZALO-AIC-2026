"""Harden expanded-ZIP handling and verify each TRAIN source file on CPU."""
from pathlib import Path
import json,hashlib
exp=Path(__file__).parent;deploy=exp/'KAGGLE_REAL_PREFLIGHT'
source=(deploy/'preflight.py').read_text(encoding='utf-8')
old="candidates=[p.parent for p in markers[0].parent.rglob('ARTIFACT_MANIFEST.json') if sha(p)==BUNDLE_HASH];assert len(candidates)==1;bundle=candidates[0]"
new="""candidates=[p for p in markers[0].parent.rglob('ARTIFACT_MANIFEST.json') if sha(p)==BUNDLE_HASH];assert len(candidates)==1
 manifest=candidates[0];origin=manifest.parent.resolve();bundle=Path('/kaggle/temp/zaic_v4_input');bundle.mkdir()
 for name,item in json.loads(manifest.read_text())['files'].items():
  original=(origin/name).resolve();target=(bundle/name).resolve();assert original.is_relative_to(origin) and target.is_relative_to(bundle.resolve()) and not original.is_symlink()
  assert original.stat().st_size==item['bytes'] and sha(original)==item['sha256'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original,target)
 shutil.copy2(manifest,bundle/'ARTIFACT_MANIFEST.json')"""
assert old in source;source=source.replace(old,new)
source+='''
data_manifest=json.loads((runtime/'DATA_MANIFEST.json').read_text());verified_bytes=0
for name,item in data_manifest['business_source_files'].items():
 p=(runtime/name).resolve();assert p.is_relative_to(runtime.resolve()) and p.stat().st_size==item['bytes'] and sha(p)==item['sha256'];verified_bytes+=item['bytes']
test_result=json.loads(Path('/kaggle/working/TEST_RESULTS.json').read_text());test_result.update(competition_data_mounted=True,tests_use_synthetic_data_only=True);atomic(Path('/kaggle/working/TEST_RESULTS.json'),test_result)
out.update(raw_TRAIN_source_files_verified=len(data_manifest['business_source_files']),raw_TRAIN_source_bytes_verified=verified_bytes,raw_source_parsed=False,preflight_version=2);atomic(Path('/kaggle/working/REAL_PREFLIGHT_RESULT.json'),out)
files={name:{'bytes':Path('/kaggle/working',name).stat().st_size,'sha256':sha(Path('/kaggle/working',name))} for name in ['TEST_RESULTS.json','TEST_DETAILS.txt','REAL_PREFLIGHT_RESULT.json']}
atomic(Path('/kaggle/working/PREFLIGHT_ARTIFACT_MANIFEST.json'),{'state':'passed','files':files,'GPU_executed':False,'optimizer_executed':False});print('PREFLIGHT_V2_COMPLETE',json.dumps({'state':'passed','raw_source_files':out['raw_TRAIN_source_files_verified'],'source_sha256':SOURCE_HASH,'bank_manifest_sha256':c['bank_manifest_sha256']}))
'''
compile(source,'preflight_v2.py','exec');(deploy/'preflight_v2.py').write_text(source,encoding='utf-8');meta=json.loads((deploy/'kernel-metadata.json').read_text());(deploy/'kernel-metadata.v1.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');meta['code_file']='preflight_v2.py';(deploy/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');(exp/'REAL_PREFLIGHT_WRAPPER_V2.json').write_text(json.dumps({'wrapper_sha256':hashlib.sha256(source.encode()).hexdigest(),'reason':'manifest-bound safe reconstruction when Kaggle expandsZIP; verify88originalTRAINfiles; correct mounted-data scope metadata','gpu':False},indent=2),encoding='utf-8');print('CPU_PREFLIGHT_V2_READY')
