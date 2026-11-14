"""Four-step verified TRAIN replay pilot, then frozen public inference; aggregate stdout."""
from pathlib import Path
import json,sys,os,time,subprocess,tempfile,shutil
sys.path[:0]=[str(Path(__file__).parent/'baseline/app'),str(Path(__file__).parent/'baseline/tools')]
from relearning import atomic,sha,seal,verify_export,PARENT_SHA
from runtime_gpu import single_gpu_env

def validate_request(config,rows,provenance):
    if config.get('training_objective')!='verified_replay_refresh':raise ValueError('Explicit replay-refresh objective required; correction/replay has a different gate')
    if config.get('budget_approved') is not True or config.get('max_steps')!=4 or config.get('total_gpu_seconds')!=3600:raise ValueError('Reviewed four-step/60-minute budget required')
    if config.get('parent_sha256')!=PARENT_SHA or provenance['parent_sha256']!=PARENT_SHA:raise ValueError('Wrong parent; smoke candidate is forbidden')
    if len(rows)!=10 or len({r['id'] for r in rows})!=10:raise ValueError('Unexpected verified target set')
    if any(r.get('split')!='learning' or r.get('target_verified') is not True or r.get('kind')!='replay' or r.get('parent_sha256')!=PARENT_SHA or r['business_id'] not in provenance['learning_businesses'] for r in rows):raise ValueError('Invalid TRAIN target/split')

def validate_public_result(runtime,questions,predictions):
    from audit_output import validate
    if len(predictions)!=len(questions) or len(questions)!=1000:raise ValueError('Expected 1000 public records')
    if [q['id'] for q in questions]!=[r['id'] for r in predictions]:raise ValueError('Public IDs/order mismatch')
    if runtime.get('state')!='completed' or runtime.get('n')!=1000 or runtime.get('question_logs')!=1000 or runtime.get('network_attempts')!=0:raise ValueError('Incomplete public inference cannot pass as completed')
    for row in predictions:validate(row)

def regression(model,tok,config,rows,out):
    from baseline_bridge import BaselineBridge,matches
    from legacy_engine import prepare_shop
    root=Path(config['runtime_root']);shops={};verdicts={};attempts=out/'regression_attempts.jsonl'
    selected=[r for r in rows if r.get('origin')!='verified_historical_parent_training_sft']
    if len(selected)!=4:raise ValueError('Frozen parent regression set changed')
    for row in selected:
        biz=row['business_id']
        if biz not in shops:shops[biz]=prepare_shop(root/'data'/biz,cache_dir=None)
        bridge=BaselineBridge(model,tok,shops,max_tokens=240,max_seconds=25)
        ctx=json.loads(row['prompt'])
        q={'id':row['id'],'prompt':row['prompt'],'source_ref':biz,'as_of_date':ctx['as_of_date']}
        result={'id':row['id'],'state':'attempting','model_sha256':config['candidate_sha256']}
        try:
            plan=bridge.predict(q,1);result.update(state='predicted',plan=plan)
            with attempts.open('a',encoding='utf-8') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
            # Reference plan/answer is opened only after the blind candidate plan is persisted.
            reference=json.loads(row['response'])
            verdict=bridge.semantic(q,plan) and matches(bridge.execute(q,plan),bridge.execute(q,reference))
        except Exception as exc:
            verdict=False;result.update(state='error',exception_type=type(exc).__name__)
            with attempts.open('a',encoding='utf-8') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
        verdicts[row['id']]=bool(verdict)
    report={'mode':'evaluation','split':'learning_replay_regression','expected':4,'attempted':len(verdicts),'parent_correct':4,'candidate_correct':sum(verdicts.values()),'correct_to_wrong':sum(not v for v in verdicts.values()),'gold_used_to_change_predictions':False,'weight_updates':False,'full_calibration_completed':False,'best_model_promoted':False}
    atomic(out/'REGRESSION_SUMMARY.json',report)
    return report

def train_child(config_path):
    from train_warm import train
    config=json.loads(Path(config_path).read_text());bundle=Path(config['bundle'])
    rows=json.loads((bundle/'TRAIN_VERIFIED_REPLAY.json').read_text())
    provenance=json.loads((bundle/'TRAIN_PROVENANCE.json').read_text())
    validate_request(config,rows,provenance)
    run=train(config,rows)
    result=json.loads((run/'training_result.json').read_text())
    if result['steps']!=4:raise ValueError('Did not complete four optimizer steps')
    from smoke_validation import tensor_delta,safe_adapter_tensors
    delta=tensor_delta(safe_adapter_tensors(Path(config['adapter'])/'adapter_model.safetensors'),safe_adapter_tensors(run/'checkpoint/adapter_model.safetensors'))
    receipt={'state':'completed','run_path':str(run),'run_id':run.name,'steps':4,'training_objective':'verified_replay_refresh','real_train_rows':10,'synthetic_training':False,'correction_count':0,'parent_sha256':PARENT_SHA,'adapter_sha256':sha(run/'checkpoint/adapter_model.safetensors'),'manifest_sha256':verify_export(run),'adapter_delta':delta,'metrics':result['metrics'],'best_model_promoted':False}
    atomic(Path(config['session_output'])/'TRAIN_RECEIPT.json',receipt)

def public_child(config_path):
    config=json.loads(Path(config_path).read_text());out=Path(config['session_output']);receipt=json.loads((out/'TRAIN_RECEIPT.json').read_text())
    run=Path(receipt['run_path']);assert verify_export(run)==receipt['manifest_sha256']
    before=sha(run/'checkpoint/adapter_model.safetensors');config['candidate_sha256']=before
    rows=json.loads((Path(config['bundle'])/'TRAIN_VERIFIED_REPLAY.json').read_text())
    public=Path(config['bundle'])/'PUBLIC'
    assert not (public/'answers.jsonl').exists()
    with tempfile.TemporaryDirectory(prefix='zaic_811_public_') as scratch:
        p=Path(scratch);(p/'assets').mkdir();(p/'assets/base').symlink_to(Path(config['base']).resolve(),target_is_directory=True)
        (p/'app').symlink_to(Path(__file__).parent/'baseline/app',target_is_directory=True)
        (p/'candidate').mkdir();(p/'candidate/final').symlink_to(run/'checkpoint',target_is_directory=True)
        shutil.copy2(Path(__file__).parent/'baseline/model_lock.json',p/'model_lock.json')
        import public_infer_811
        sys.argv=['public_infer_811','--project',str(p),'--root',str(public),'--questions',str(public/'questions.jsonl'),'--variant','candidate','--output-dir',str(out/'public_test'),'--batch-size','4','--cache-dir',str(p/'cache'),'--max-input-tokens','3072']
        public_infer_811.main(after_model=lambda model,tok:regression(model,tok,config,rows,out))
    questions=[json.loads(l) for l in (public/'questions.jsonl').read_text(encoding='utf-8-sig').splitlines() if l.strip()]
    predictions=[json.loads(l) for l in (out/'public_test/submission.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    runtime=json.loads((out/'public_test/runtime.json').read_text());validate_public_result(runtime,questions,predictions)
    assert before==sha(run/'checkpoint/adapter_model.safetensors')
    summary={k:v for k,v in runtime.items() if k not in {'batch_timings','log_path'}}
    summary.update(public_gold_accessed=False,weight_updates=False,adapter_sha256=before,submission_validated=True,official_score_available=False)
    atomic(out/'PUBLIC_SUMMARY.json',summary)

def main(config_path):
    config=json.loads(Path(config_path).read_text());started=time.monotonic();out=Path(config['session_output']);out.mkdir(parents=True,exist_ok=False)
    atomic(out/'RUN_REQUEST.json',config)
    bundle=Path(config['bundle']);assert verify_export(bundle)==config['bundle_manifest_sha256']
    rows=json.loads((bundle/'TRAIN_VERIFIED_REPLAY.json').read_text());provenance=json.loads((bundle/'TRAIN_PROVENANCE.json').read_text());validate_request(config,rows,provenance)
    try:
        for phase,limit in [('train',900),('public',3300)]:
            remaining=min(limit,3300-(time.monotonic()-started));assert remaining>0
            atomic(out/'PHASE_STATUS.json',{'phase':phase,'state':'running','elapsed_seconds':round(time.monotonic()-started,1)})
            with (out/(phase.upper()+'_PRIVATE.log')).open('w',encoding='utf-8') as log:
                job=subprocess.run([sys.executable,str(Path(__file__).resolve()),phase,str(config_path)],env=single_gpu_env(os.environ),stdout=log,stderr=subprocess.STDOUT,timeout=remaining)
            if job.returncode:raise RuntimeError(phase+' process failed; preserve private logs')
        atomic(out/'WORKFLOW_RESULT.json',{'state':'completed','train_steps':4,'real_train_rows':10,'public_questions':1000,'best_model_promoted':False,'total_seconds':time.monotonic()-started,'gpu_shutdown_required':True})
        atomic(out/'PHASE_STATUS.json',{'phase':'completed','state':'completed','elapsed_seconds':round(time.monotonic()-started,1)})
    except Exception as exc:
        atomic(out/'FAILURE.json',{'state':'failed','exception_type':type(exc).__name__,'elapsed_seconds':time.monotonic()-started,'preserve_partial_outputs':True,'gpu_shutdown_required':True})
        raise
    finally:seal(out)

if __name__=='__main__':
    phase,path=sys.argv[1:]
    if phase=='train':train_child(path)
    elif phase=='public':public_child(path)
    elif phase=='workflow':main(path)
    else:raise ValueError('Unknown phase')
