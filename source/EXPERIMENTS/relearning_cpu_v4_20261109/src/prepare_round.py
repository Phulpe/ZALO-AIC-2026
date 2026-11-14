"""Bounded blind TRAIN collection. No optimizer or public-test access."""
from pathlib import Path
import os,sys,json,tempfile,time
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
os.environ['HF_DEACTIVATE_ASYNC_LOAD']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
sys.path.insert(0,str(Path(__file__).parent/'baseline/app'))
from relearning import verify_parent,sha,atomic,seal

def main(path):
 config=json.loads(Path(path).read_text());out=Path(config['session_output']);out.mkdir(parents=True,exist_ok=False)
 verify_parent(config['adapter'],config['parent_sha256'])
 for name,expected in config['base_hashes'].items():assert sha(Path(config['base'])/name)==expected
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 from modeling import load_base
 from peft import PeftModel
 from offline_guard import install
 install()
 with tempfile.TemporaryDirectory() as scratch:
  p=Path(scratch);(p/'assets').mkdir();(p/'assets/base').symlink_to(Path(config['base']),target_is_directory=True)
  model,tok=load_base(p)
  model=PeftModel.from_pretrained(model,config['adapter'],is_trainable=False,local_files_only=True)
  model.eval();model.config.use_cache=True
  from learning_probe import collect
  config.update(verifier_v2_pool=True,learning_probe_max_questions=128)
  result=collect(model,tok,config)
 atomic(out/'BLIND_COLLECTION_RESULT.json',result)
 seal(out)
 print(json.dumps({k:v for k,v in result.items() if k not in ('path',)}))
if __name__=='__main__':main(sys.argv[1])
