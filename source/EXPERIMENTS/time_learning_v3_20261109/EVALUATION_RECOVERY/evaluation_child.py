from pathlib import Path
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
