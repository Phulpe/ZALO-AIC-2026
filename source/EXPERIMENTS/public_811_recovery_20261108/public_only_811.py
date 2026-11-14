"""Inference-only recovery from the verified four-step checkpoint. Never trains."""
from pathlib import Path
import json,sys,time
sys.path[:0]=[str(Path(__file__).parent/'baseline/app'),str(Path(__file__).parent/'baseline/tools')]
from relearning import verify_export,sha,atomic,seal
from workflow_811 import public_child

CANDIDATE='b3063abe404bc394a943360f61f60f2d1f400d35340813c0e303f742c4260e85'
MANIFEST='ace8f09230d5b8d64d2c81c43062d5f1abb875fb551d669747c1d3f38a4bee1b'

def validate_candidate(receipt):
    if receipt.get('steps')!=4 or receipt.get('synthetic_training') is not False or receipt.get('adapter_sha256')!=CANDIDATE or receipt.get('manifest_sha256')!=MANIFEST:raise ValueError('Only the verified four-step real TRAIN checkpoint is permitted')

def main(config_path):
    config=json.loads(Path(config_path).read_text());started=time.monotonic();out=Path(config['session_output']);out.mkdir(parents=True,exist_ok=False)
    try:
        receipt_file=Path(config['saved_train_receipt']);receipt=json.loads(receipt_file.read_text());validate_candidate(receipt)
        source=receipt_file.parent/'model_runs'/receipt['run_id']
        assert verify_export(source)==MANIFEST and sha(source/'checkpoint/adapter_model.safetensors')==CANDIDATE
        assert verify_export(Path(config['bundle']))==config['bundle_manifest_sha256']
        atomic(out/'SOURCE_TRAIN_RECEIPT.json',receipt)
        resolved={**receipt,'run_path':str(source),'original_run_path':receipt['run_path'],'original_receipt_sha256':sha(receipt_file),'input_path_resolved_without_modifying_source':True}
        atomic(out/'TRAIN_RECEIPT.json',resolved)
        atomic(out/'RUN_REQUEST.json',config)
        atomic(out/'PHASE_STATUS.json',{'phase':'public_only','state':'running','training_repeated':False})
        public_child(config_path)
        atomic(out/'WORKFLOW_RESULT.json',{'state':'completed','training_repeated':False,'source_training_steps':4,'public_questions':1000,'candidate_sha256':CANDIDATE,'best_model_promoted':False,'elapsed_seconds':time.monotonic()-started,'gpu_shutdown_required':True})
        atomic(out/'PHASE_STATUS.json',{'phase':'completed','state':'completed','training_repeated':False})
    except Exception as exc:
        atomic(out/'FAILURE.json',{'state':'failed','exception_type':type(exc).__name__,'training_repeated':False,'elapsed_seconds':time.monotonic()-started,'gpu_shutdown_required':True})
        raise
    finally:seal(out)

if __name__=='__main__':main(sys.argv[1])
