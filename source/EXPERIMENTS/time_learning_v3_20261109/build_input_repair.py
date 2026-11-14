"""Keep failed wrappers and source unchanged; verify corrected inputs on CPU."""
from pathlib import Path
import json,hashlib,ast
root=Path(__file__).parent;folder=root/'REPAIR_01';folder.mkdir(exist_ok=True)
c=json.loads((root/'FROZEN_RUN_CONFIG.json').read_text(encoding='utf-8'));ref=json.loads((root/'REFERENCE_PACKAGING_RECEIPT.json').read_text());spent=json.loads((root/'QUOTA_AFTER_SETUP_ERROR.json').read_text())['gpu_used_seconds']-json.loads((root/'QUOTA_BEFORE.json').read_text())['gpu_used_seconds']
c.update(prior_workflow_manifest_sha256=ref['bundle_manifest_sha256'],original_prior_workflow_manifest_sha256=ref['original_workflow_manifest_sha256'],reference_archive_sha256=ref['archive_sha256'],soft_limit_seconds=6300,prior_setup_gpu_seconds=spent)
assert spent+6750+180<=7200
cpu=(root/'FINAL_CPU/cpu.py').read_text(encoding='utf-8');prefix=cpu.split('CONFIG=')[0];common=cpu.split('CONFIG=',1)[1].split("request=root/'FINAL_CPU_REQUEST.json'",1)[0];common=common[common.index('\n')+1:]
resolver='''
sys.path.insert(0,str(root))
from relearning import verify_export,sha,verify_parent
markers=list(Path('/kaggle/input').rglob('REFERENCE_UPLOAD_MANIFEST.json'));assert len(markers)==1,'Need immutable reference dataset'
ref_upload=json.loads(markers[0].read_text());archive=markers[0].parent/'PRIOR_REFERENCE.zip'
assert CONFIG['reference_archive_sha256']==ref_upload['archive_sha256']
reference=Path('/kaggle/temp/temporal_prior_reference');reference.mkdir(parents=True,exist_ok=False)
if archive.is_file():
 assert sha(archive)==CONFIG['reference_archive_sha256']
 with zipfile.ZipFile(archive) as z:
  for f in z.infolist():assert (reference/f.filename).resolve().is_relative_to(reference.resolve()) and not stat.S_ISLNK(f.external_attr>>16)
  z.extractall(reference)
else:
 # Kaggle may expand uploaded ZIPs. Reconstruct only manifest-bound files.
 import shutil
 manifests=[p for p in markers[0].parent.rglob('ARTIFACT_MANIFEST.json') if sha(p)==CONFIG['prior_workflow_manifest_sha256']];assert len(manifests)==1
 manifest=manifests[0];files=json.loads(manifest.read_text())['files'];origin=manifest.parent.resolve()
 for name,item in files.items():
  source=(origin/name).resolve();target=(reference/name).resolve();assert source.is_relative_to(origin) and target.is_relative_to(reference.resolve()) and not source.is_symlink()
  assert source.stat().st_size==item['bytes'] and sha(source)==item['sha256'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
 shutil.copy2(manifest,reference/'ARTIFACT_MANIFEST.json')
assert verify_export(reference)==CONFIG['prior_workflow_manifest_sha256']==ref_upload['bundle_manifest_sha256']
receipt=json.loads((reference/'CANDIDATE_RECEIPT.json').read_text());prior=reference/'model_runs'/Path(receipt['run_path']).name/'checkpoint'
assert sha(prior/'adapter_model.safetensors')==CONFIG['prior_candidate_sha256']
adapters=list(Path('/kaggle/input').rglob('final/adapter_model.safetensors'));assert len(adapters)==1
CONFIG.update(adapter=str(adapters[0].parent),prior_workflow=str(reference));verify_parent(CONFIG['adapter'],CONFIG['parent_sha256'])
assert verify_export(Path(CONFIG['cpu_prepared']))==CONFIG['cpu_manifest_sha256']
'''
cpu_tail='''
# This superset CPU mount diagnoses the previously missing notebook marker.
legacy=list(Path('/kaggle/input').rglob('WORKFLOW_COMPLETED.json'))
for name,digest in CONFIG['base_hashes'].items():assert sha(Path(CONFIG['base'])/name)==digest
request=root/'FINAL_CPU_REQUEST.json';request.write_text(json.dumps(CONFIG,indent=2))
p=subprocess.run([sys.executable,str(root/'final_cpu_probe.py'),str(request)],cwd=root);assert p.returncode==0
Path('/kaggle/working/INPUT_REPAIR_RESULT.json').write_text(json.dumps({'state':'passed','legacy_workflow_marker_count':len(legacy),'reference_manifest_sha256':CONFIG['prior_workflow_manifest_sha256'],'prior_adapter_sha256':sha(prior/'adapter_model.safetensors'),'parent_adapter_sha256':sha(Path(CONFIG['adapter'])/'adapter_model.safetensors'),'base_files_verified':len(CONFIG['base_hashes']),'gpu_executed':False},indent=2))
print('ALL_GPU_INPUTS_VERIFIED_ON_CPU')
'''
gpu_tail='''
request=root/'TIME_PILOT_REQUEST.json';request.write_text(json.dumps(CONFIG,indent=2))
p=subprocess.run([sys.executable,str(root/'time_pilot.py'),str(request)],cwd=root,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'},timeout=6500);assert p.returncode==0,'Pilot failed; preserve private outputs'
'''
common_code='CONFIG='+repr(c)+'\n'+common+resolver
(folder/'cpu.py').write_text(prefix+common_code+cpu_tail,encoding='utf-8');bnb=(root/'src/OFFLINE_BNB_BOOTSTRAP.py').read_text(encoding='utf-8')
# CPU tests already verify this exact archive; do not burn GPU quota rerunning them.
gpu_prefix=prefix[:prefix.index('p=subprocess.run(')]+"print('FROZEN_SOURCE_VERIFIED_CPU_TESTS_ALREADY_PASSED',flush=True)\n"
(folder/'gpu.py').write_text(gpu_prefix+'\n'+bnb+'\n'+common_code+gpu_tail,encoding='utf-8')
gpu_meta=json.loads((root/'KAGGLE_GPU/kernel-metadata.gpu-v1.saved.json').read_text());gpu_meta['kernel_sources']=[s for s in gpu_meta['kernel_sources'] if 'zaic-quality-train-eval-public-' not in s];gpu_meta['dataset_sources'].append('lnghongphc/zaic-temporal-v3-prior-reference-20261109');gpu_meta.update(code_file='gpu.py',enable_gpu=True)
(folder/'kernel-metadata.gpu.json').write_text(json.dumps(gpu_meta,indent=2));cpu_meta={**gpu_meta,'id':'lnghongphc/zaic-time-v3-input-repair-20261109','title':'ZAIC Time v3 Input Repair 20261109','code_file':'cpu.py','enable_gpu':False,'kernel_sources':gpu_meta['kernel_sources']+['lnghongphc/zaic-quality-train-eval-public-20261109/1']};(folder/'kernel-metadata.json').write_text(json.dumps(cpu_meta,indent=2));(folder/'RUN_CONFIG.json').write_text(json.dumps(c,ensure_ascii=False,indent=2),encoding='utf-8')
for name in ['cpu.py','gpu.py']:ast.parse((folder/name).read_text(encoding='utf-8'))
(folder/'REPAIR_EVIDENCE.json').write_text(json.dumps({'source_archive_unchanged':c['source_sha256'],'prior_setup_gpu_seconds':spent,'next_hard_timeout_seconds':6750,'startup_cleanup_reserve_seconds':180,'maximum_combined_gpu_seconds':spent+6750+180,'approved_seconds':7200,'optimizer_started_in_failed_version':False,'resolver_wrapper_sha256':hashlib.sha256((folder/'gpu.py').read_text(encoding='utf-8').encode()).hexdigest()},indent=2));print('REPAIR_WRAPPERS_READY_SOURCE_UNCHANGED',c['source_sha256'])
