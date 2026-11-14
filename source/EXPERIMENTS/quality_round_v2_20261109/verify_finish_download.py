"""Read-only verification; emits aggregate evidence, never per-question content."""
from pathlib import Path
import json,sys
from relearning import verify_export,sha,atomic

def main(root):
 root=Path(root); out=root/'round_finish'
 outer=verify_export(out)
 receipt=json.loads((out/'CANDIDATE_RECEIPT.json').read_text(encoding='utf-8'))
 run=out/'model_runs'/Path(receipt['run_path']).name
 manifest=verify_export(run)
 assert manifest==receipt['manifest_sha256']
 assert sha(run/'checkpoint/adapter_model.safetensors')==receipt['adapter_sha256']
 config=json.loads((run/'run_config.json').read_text(encoding='utf-8'))
 training=json.loads((run/'training_result.json').read_text(encoding='utf-8'))
 assert config['init_mode']=='warm_start' and config['mode']=='learning'
 assert config['parent_sha256']=='e7dfa635beccce128c99f9d3fd993d33dbb4c88e73e3ddeccbc2bc915eec4a4d'
 assert training['state']=='completed' and 0<training['steps']<=config['max_steps']
 required=['adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json']
 checkpoints=[]
 for p in sorted(run.rglob('trainer_state.json')):
  folder=p.parent
  checkpoints.append({'path':str(folder),'present':{n:(folder/n).is_file() for n in required+['scaler.pt']},'bytes':sum(f.stat().st_size for f in folder.rglob('*') if f.is_file())})
 metrics={}
 full_metrics={}
 for stage in ['parent_calibration','candidate_calibration']:
  data=json.loads((out/stage/'EVALUATION.json').read_text(encoding='utf-8'))
  full_metrics[stage]=data
  metrics[stage]={k:v for k,v in data.items() if k!='verdicts'}
 public=json.loads((out/'public_test/PUBLIC_VALIDATION.json').read_text(encoding='utf-8'))
 submission=out/'public_test/predictions/submission.jsonl'
 assert sha(submission)==public['submission_sha256']
 ids=[json.loads(line)['id'] for line in submission.read_text(encoding='utf-8').splitlines() if line.strip()]
 assert len(ids)==len(set(ids))==1000
 from quality_gate import promotion
 decision=json.loads((out/'MODEL_DECISION.json').read_text())
 assert promotion(full_metrics['parent_calibration'],full_metrics['candidate_calibration'],config['promotion_policy'])==decision
 result={'state':'verified','outer_manifest_sha256':outer,'run_manifest_sha256':manifest,'candidate_adapter_sha256':receipt['adapter_sha256'],'candidate_delta':receipt['adapter_delta'],'run_path':str(run),'training':training,'planned_steps':config['max_steps'],'checkpoints':checkpoints,'calibration':metrics,'model_decision':decision,'public_model_selection':json.loads((out/'PUBLIC_MODEL_SELECTION.json').read_text()),'public_validation':public,'workflow':json.loads((out/'WORKFLOW_COMPLETED.json').read_text()),'resume_execution_verified':False,'scope':'Downloaded saved Kaggle version; all manifest files checked; no pickle deserialization'}
 atomic(root/'LOCAL_VERIFICATION.json',result)
 print(json.dumps(result,ensure_ascii=True,indent=2))

if __name__=='__main__':main(sys.argv[1])
