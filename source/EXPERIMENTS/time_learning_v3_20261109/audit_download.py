"""Audit saved v3 artifacts and print aggregate metrics only; never load pickle."""
from pathlib import Path
import json,sys,struct,math,datetime
sys.path.insert(0,str(Path(__file__).parent/'src'))
from relearning import verify_export,sha,atomic

def audit(folder):
 folder=Path(folder);out=folder/'time_pilot';outer=verify_export(out)
 summary={'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'outer_manifest_sha256':outer,'artifact_files_verified':len(json.loads((out/'ARTIFACT_MANIFEST.json').read_text())['files']),'pickle_loaded':False}
 for name in ['COLLECTION_SUMMARY.json','SELECTION_REPORT.json','PREVIOUS_ROUND_FIELD_PROBE.json','NEW_CANDIDATE_FIELD_PROBE.json','PILOT_RESULT.json','WORKFLOW_COMPLETED.json','FAILURE.json','BASELINE_CALIBRATION_REUSE.json']:
  if (out/name).exists():summary[name]=json.loads((out/name).read_text(encoding='utf-8'))
 if (out/'CANDIDATE_RECEIPT.json').exists():
  receipt=json.loads((out/'CANDIDATE_RECEIPT.json').read_text());run=out/'model_runs'/Path(receipt['run_path']).name
  assert verify_export(run)==receipt['manifest_sha256'];assert sha(run/'checkpoint/adapter_model.safetensors')==receipt['adapter_sha256']
  cfg=json.loads((run/'run_config.json').read_text(encoding='utf-8'));result=json.loads((run/'training_result.json').read_text());assert cfg['mode']=='learning' and cfg['init_mode']=='warm_start' and cfg['loss_objective']=='temporal_payload_completion_v1' and cfg['parent_sha256']=='e7dfa635beccce128c99f9d3fd993d33dbb4c88e73e3ddeccbc2bc915eec4a4d'
  summary.update(run_path=str(run),run_id=run.name,training=result,candidate_sha256=receipt['adapter_sha256'],training_manifest_sha256=receipt['manifest_sha256'],adapter_delta=receipt['adapter_delta'])
  checkpoints=[];manifest=json.loads((run/'ARTIFACT_MANIFEST.json').read_text())['files']
  for weight in sorted(run.rglob('adapter_model.safetensors')):
   with weight.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];assert 0<n<16*1024**2;header=json.loads(f.read(n))
   spans=[]
   for k,t in header.items():
    if k=='__metadata__':continue
    a,b=t['data_offsets'];assert 0<=a<=b<=weight.stat().st_size-8-n;assert b-a==math.prod(t['shape'])*{'F32':4,'BF16':2,'F16':2}[t['dtype']];spans.append((a,b))
   spans.sort();assert spans[0][0]==0 and spans[-1][1]==weight.stat().st_size-8-n and all(a[1]==b[0] for a,b in zip(spans,spans[1:]))
   required=['adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt','scaler.pt','rng_state.pth','trainer_state.json'];states={name:(weight.parent/name).is_file() for name in required}
   state=json.loads((weight.parent/'trainer_state.json').read_text()) if states['trainer_state.json'] else {}
   checkpoints.append({'path':str(weight.parent),'step':state.get('global_step'),'states':states,'adapter_sha256':manifest[weight.relative_to(run).as_posix()]['sha256'],'bytes':weight.stat().st_size,'header_valid':True,'full_resume_executed':False})
  summary['checkpoints']=checkpoints;summary['unique_adapter_hashes']=len({c['adapter_sha256'] for c in checkpoints})
  for name in ['original14','selected_regression']:
   path=out/name
   if not (path/'REGRESSION_SUMMARY.json').exists():continue
   summary[name]=json.loads((path/'REGRESSION_SUMMARY.json').read_text());verify_export(path)
   from pilot_gate import learning_gate
   summary[name+'_gate_recomputed']=learning_gate(summary[name])
 if (out/'candidate_calibration/EVALUATION.json').exists():
  data=json.loads((out/'candidate_calibration/EVALUATION.json').read_text());summary['candidate_calibration']={k:v for k,v in data.items() if k!='verdicts'}
 def finite(value):
  if isinstance(value,float):assert math.isfinite(value)
  elif isinstance(value,dict):
   for v in value.values():finite(v)
  elif isinstance(value,list):
   for v in value:finite(v)
 finite(summary)
 atomic(folder/'LOCAL_VERIFICATION.json',summary)
 print(json.dumps({k:v for k,v in summary.items() if k!='checkpoints'},ensure_ascii=True,indent=2));print('CHECKPOINT_SETS_VERIFIED',len(summary.get('checkpoints',[])))
 return summary

if __name__=='__main__':audit(sys.argv[1])
