"""Package only saved candidate and TRAIN selections; never create an optimizer."""
from pathlib import Path
import json,sys,shutil,zipfile,hashlib,ast
exp=Path(__file__).parent;sys.path.insert(0,str(exp/'src'))
from relearning import sha,seal,atomic
local=Path('D:/ZAIC/PRIVATE_DATA/zaic/result_811/TIME_FOCUSED_V3_20261109/time_pilot')
receipt=json.loads((local/'CANDIDATE_RECEIPT.json').read_text());run=local/'model_runs'/Path(receipt['run_path']).name
assert sha(run/'checkpoint/adapter_model.safetensors')==receipt['adapter_sha256']
folder=exp/'EVALUATION_RECOVERY';folder.mkdir(exist_ok=False)
bundle=Path('D:/ZAIC/PRIVATE_DATA/time_v3_evaluation_candidate');bundle.mkdir(exist_ok=False)
stage=bundle/'bundle';stage.mkdir()
for name in ['adapter_model.safetensors','adapter_config.json']:
 target=stage/'candidate'/name;target.parent.mkdir(exist_ok=True);shutil.copy2(run/'checkpoint'/name,target)
shutil.copy2(local/'PRIVATE_SELECTED.json',stage/'PRIVATE_SELECTED.json');atomic(stage/'CANDIDATE_RECEIPT.json',receipt)
manifest=seal(stage);upload=bundle/'upload';upload.mkdir()
with zipfile.ZipFile(upload/'CANDIDATE.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in stage.rglob('*'):
  if p.is_file():z.write(p,p.relative_to(stage).as_posix())
atomic(upload/'EVALUATION_UPLOAD.json',{'manifest_sha256':manifest,'archive_sha256':sha(upload/'CANDIDATE.zip'),'adapter_sha256':receipt['adapter_sha256'],'training_manifest_sha256':receipt['manifest_sha256']})
atomic(upload/'dataset-metadata.json',{'title':'ZAIC Temporal v3 Saved Candidate Evaluation 20261109','id':'lnghongphc/zaic-temporal-v3-saved-evaluation-20261109','licenses':[{'name':'other'}]})
child='''from pathlib import Path
import sys,json,tempfile,os
sys.path[:0]=[str(Path(__file__).parent),str(Path(__file__).parent/'baseline/app')]
from relearning import atomic,verify_export,sha,seal
def main(path):
 c=json.loads(Path(path).read_text());out=Path(c['evaluation_output']);out.mkdir(exist_ok=False)
 assert c['mode']=='evaluation' and c['run_public'] is False and c['weights_updated'] is False
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 assert torch.cuda.device_count()==1
 from runtime_data import load_learning_pool
 runtime=Path(c['runtime_root']);manifest=json.loads((runtime/'RUNTIME_DATA_MANIFEST.json').read_text());pool,split,gold=load_learning_pool(runtime,manifest['dataset_manifest_sha256'],manifest['split_manifest_sha256'])
 from quality_gate import select,digest
 original=Path(c['verified_collection']);bank=json.loads((original/'verified_bank.json').read_text());evidence=json.loads((original/'PRIVATE_TARGET_EVIDENCE.json').read_text());old,report=select(bank['correction']+bank['replay'],evidence,c['supervision_gate']);assert report['dataset_sha256']==c['expected_training_dataset_sha256']
 selected=json.loads(Path(c['selected_file']).read_text());assert all(q['business_id'] in split['groups']['learning'] for q in selected+old)
 assert len({q['id'] for q in selected})==18 and len(old)==14
 from field_eval import TrustedShops,teacher_stats,verify_loaded,evaluate_rows
 shops=TrustedShops(Path(c['cpu_prepared'])/'TRAIN_CACHE',c['cache_manifest_sha256'],c['runtime_manifest_sha256'])
 from modeling import load_base
 from peft import PeftModel
 from offline_guard import install
 install()
 with tempfile.TemporaryDirectory() as scratch:
  p=Path(scratch);(p/'assets').mkdir();(p/'assets/base').symlink_to(Path(c['base']),target_is_directory=True)
  model,tok=load_base(p);model=PeftModel.from_pretrained(model,c['candidate'],is_trainable=False,local_files_only=True);model.eval()
  assert not any(v.requires_grad for v in model.parameters())
  loaded=verify_loaded(model,c['candidate']);assert loaded['adapter_sha256']==c['candidate_sha256']
  atomic(out/'NEW_CANDIDATE_FIELD_PROBE.json',{'loaded':loaded,'teacher':teacher_stats(model,tok,old)})
  a=evaluate_rows(model,tok,old,pool,gold,shops,out/'original14');b=evaluate_rows(model,tok,selected,pool,gold,shops,out/'selected_regression')
  from pilot_gate import learning_gate
  atomic(out/'POST_RESULT.json',{'original14_gate':learning_gate(a),'selected_gate':learning_gate(b),'optimizer_executed':False,'weights_updated':False,'candidate_sha256':loaded['adapter_sha256']})
 seal(out)
if __name__=='__main__':main(sys.argv[1])
'''
(folder/'evaluation_child.py').write_text(child,encoding='utf-8')
# Exact verified training source remains intact. The recovery adds only an inference child.
wrapper=(exp/'REPAIR_01/GPU_DEPLOY/gpu.py').read_text(encoding='utf-8').split("request=root/'TIME_PILOT_REQUEST.json'")[0]
extra='''
CONFIG.update(mode='evaluation',weights_updated=False,run_public=False)
marks=list(Path('/kaggle/input').rglob('EVALUATION_UPLOAD.json'));assert len(marks)==1
meta=json.loads(marks[0].read_text());assert meta['manifest_sha256']==EXPECTED_MANIFEST
incoming=Path('/kaggle/temp/saved_candidate');incoming.mkdir()
zpath=marks[0].parent/'CANDIDATE.zip'
if zpath.is_file():
 assert sha(zpath)==meta['archive_sha256']
 with zipfile.ZipFile(zpath) as z:
  for f in z.infolist():assert (incoming/f.filename).resolve().is_relative_to(incoming.resolve()) and not stat.S_ISLNK(f.external_attr>>16)
  z.extractall(incoming)
else:
 manifests=[p for p in marks[0].parent.rglob('ARTIFACT_MANIFEST.json') if sha(p)==EXPECTED_MANIFEST];assert len(manifests)==1
 manifest=manifests[0];origin=manifest.parent.resolve()
 import shutil
 for name,item in json.loads(manifest.read_text())['files'].items():
  source=(origin/name).resolve();target=(incoming/name).resolve();assert source.is_relative_to(origin) and target.is_relative_to(incoming.resolve()) and not source.is_symlink()
  assert sha(source)==item['sha256'] and source.stat().st_size==item['bytes'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
 shutil.copy2(manifest,incoming/'ARTIFACT_MANIFEST.json')
assert verify_export(incoming)==EXPECTED_MANIFEST
CONFIG.update(candidate=str(incoming/'candidate'),selected_file=str(incoming/'PRIVATE_SELECTED.json'),candidate_sha256=meta['adapter_sha256'],evaluation_output='/kaggle/working/evaluation_recovery/post')
assert sha(Path(CONFIG['candidate'])/'adapter_model.safetensors')==EXPECTED_ADAPTER==meta['adapter_sha256']
import importlib.metadata,torch
actual_env={'versions':{n:importlib.metadata.version(n) for n in ['transformers','peft','bitsandbytes','torch','accelerate']},'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda}
saved_env=json.loads((prior.parent/'environment.json').read_text());assert actual_env==saved_env
from relearning import atomic,seal
output=Path('/kaggle/working/evaluation_recovery');output.mkdir();atomic(output/'environment.json',actual_env)
request=output/'EVALUATION_REQUEST.json';atomic(request,CONFIG)
child=root/'evaluation_child.py';child.write_text(CHILD_CODE)
started=time.monotonic()
try:
 with (output/'POST_PRIVATE.log').open('w') as log:
  r=subprocess.run([sys.executable,str(child),str(request)],cwd=root,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'},stdout=log,stderr=subprocess.STDOUT,timeout=1300)
 assert r.returncode==0,'Saved-candidate post test failed; preserve private log'
 post=json.loads((output/'post/POST_RESULT.json').read_text());eligible=post['original14_gate']['passed'] and post['selected_gate']['passed'];result={'state':'completed','optimizer_executed':False,'candidate_sha256':EXPECTED_ADAPTER,'post':post,'calibration_ran':False,'promoted':False,'public_ran':False}
 if eligible and time.monotonic()-started<1050:
  # The inference child exited, so CUDA model memory is released by process teardown.
  before=json.loads((reference/'parent_calibration/EVALUATION.json').read_text());assert before['manifest_sha256']==CONFIG['calibration_manifest_sha256'] and before['decoder_sha256']==sha(root/'public_infer_811.py') and before['scorer_sha256']==sha(root/'baseline/app/scoring.py')
  with (output/'CALIBRATION_PRIVATE.log').open('w') as log:
   r=subprocess.run([sys.executable,str(root/'locked_eval.py'),str(request),CONFIG['candidate'],str(output/'candidate_calibration')],cwd=root,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'},stdout=log,stderr=subprocess.STDOUT,timeout=3000)
  assert r.returncode==0,'Calibration failed; preserve candidate'
  from quality_gate import promotion
  after=json.loads((output/'candidate_calibration/EVALUATION.json').read_text());decision=promotion(before,after,CONFIG['promotion_policy']);result.update(calibration_ran=True,promoted=decision['promote'],calibration_decision=decision)
 atomic(output/'RECOVERY_COMPLETED.json',result);print('EVALUATION_RECOVERY_COMPLETED',json.dumps(result),flush=True)
finally:seal(output)
'''
wrapper+='\nEXPECTED_MANIFEST='+repr(manifest)+'\nEXPECTED_ADAPTER='+repr(receipt['adapter_sha256'])+'\nCHILD_CODE='+repr(child)+'\n'+extra
(folder/'gpu.py').write_text(wrapper,encoding='utf-8')
meta=json.loads((exp/'REPAIR_01/GPU_DEPLOY/kernel-metadata.gpu-v3.json').read_text());meta.update(id='lnghongphc/zaic-time-v3-saved-candidate-eval-20261109',title='ZAIC Time v3 Saved Candidate Evaluation 20261109',code_file='gpu.py',enable_gpu=True);meta['dataset_sources'].append('lnghongphc/zaic-temporal-v3-saved-evaluation-20261109');atomic(folder/'kernel-metadata.json',meta)
tree=ast.parse(child)
assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in {'train','backward','step'} for n in ast.walk(tree))
assert 'is_trainable=False' in child and 'requires_grad' in child and 'weights_updated=False' in extra
compile(wrapper,'gpu.py','exec');compile(child,'evaluation_child.py','exec')
atomic(folder/'RECOVERY_SAFETY_CHECK.json',{'state':'passed','inference_only':True,'optimizer_rounds_added':0,'candidate_sha256':receipt['adapter_sha256'],'bundle_manifest_sha256':manifest,'gpu_before_seconds':25624.623,'gpu_start_before_seconds':23987.469,'gpu_used_this_round_seconds':1637.154,'hard_timeout_seconds':4500,'startup_reserve_seconds':180,'combined_bound_seconds':6317.154,'authorized_seconds':7200,'post_test_before_calibration':True,'source_archive_unchanged':sha(exp/'CODE_V3.zip'),'wrapper_sha256':sha(folder/'gpu.py')})
print('READY_INFERENCE_ONLY',manifest,sha(folder/'gpu.py'))
