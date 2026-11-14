from pathlib import Path
import sys,json,os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['HF_DEACTIVATE_ASYNC_LOAD']='1'
sys.path.insert(0,str(Path(__file__).parent))
from relearning import atomic,verify_export,sha
from quality_train import train_verified
if __name__=='__main__':
 config=json.loads(Path(sys.argv[1]).read_text());source=Path(config['verified_collection']);assert verify_export(source)==config['collection_manifest_sha256']
 evidence=json.loads((source/'PRIVATE_TARGET_EVIDENCE.json').read_text());bank=json.loads((source/'verified_bank.json').read_text())
 rows=[row for row in bank['correction']+bank['replay'] if row['id'] in evidence]
 from quality_gate import select
 selected,_=select(rows,evidence,config['supervision_gate'])
 config.update(mode='learning',init_mode='warm_start',gpu_confirmed=True,collect_learning_probe=False,gradient_accumulation=4,max_epochs_equivalent=2,max_steps=min(32,len(selected)*2//4),approved_max_steps=32,max_seconds=1200,approved_max_seconds=1200)
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 run=train_verified(config,rows,evidence)
 from smoke_validation import tensor_delta,safe_adapter_tensors
 delta=tensor_delta(safe_adapter_tensors(Path(config['adapter'])/'adapter_model.safetensors'),safe_adapter_tensors(Path(run)/'checkpoint/adapter_model.safetensors'))
 atomic(Path(config['session_output'])/'CANDIDATE_RECEIPT.json',{'adapter_delta':delta,'run_path':str(run),'manifest_sha256':verify_export(run),'adapter_sha256':sha(Path(run)/'checkpoint/adapter_model.safetensors')})
