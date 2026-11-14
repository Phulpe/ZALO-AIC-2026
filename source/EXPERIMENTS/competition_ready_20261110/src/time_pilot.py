"""One bounded real-model temporal pilot, all GPU phases in one Kaggle job."""
from pathlib import Path
import json,sys,os,time,gc,tempfile,collections,subprocess,importlib.metadata
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
os.environ['HF_DEACTIVATE_ASYNC_LOAD']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
sys.path[:0]=[str(Path(__file__).parent),str(Path(__file__).parent/'baseline/app')]
from relearning import atomic,sha,seal,verify_export,verify_parent,new_run,LearningRound

def main(path):
 c=json.loads(Path(path).read_text());out=Path(c['session_output']);out.mkdir(exist_ok=False);started=time.monotonic();deadline=started+c['soft_limit_seconds'];here=Path(__file__).parent;result={}
 def remaining():return deadline-time.monotonic()
 def phase(name,state='running'):
  r={'phase':name,'state':state,'elapsed_seconds':time.monotonic()-started,'remaining_soft_seconds':max(0,remaining())};atomic(out/'PHASE.json',r);print('TEMPORAL_PHASE',json.dumps(r),file=sys.__stdout__,flush=True)
 def cleanup():
  gc.collect();torch.cuda.empty_cache()
  if torch.cuda.memory_allocated()>512*1024**2:raise RuntimeError('Prior model remains allocated; preserve output instead of OOM/fallback')
 try:
  if c['mode']!='learning' or c['init_mode']!='warm_start' or c['run_public'] is not False:raise ValueError('Explicit bounded learning pilot required')
  phase('input_verification');runtime=Path(c['runtime_root']);cpu=Path(c['cpu_prepared']);assert verify_export(cpu)==c['cpu_manifest_sha256'];assert sha(runtime/'RUNTIME_DATA_MANIFEST.json')==c['runtime_manifest_sha256']
  verify_parent(c['adapter'],c['parent_sha256'])
  for name,digest in c['base_hashes'].items():assert sha(Path(c['base'])/name)==digest
  original=Path(c['verified_collection']);assert verify_export(original)==c['collection_manifest_sha256']
  if c.get('frozen_regression_root'):
   assert verify_export(Path(c['frozen_regression_root']))==c['frozen_regression_manifest_sha256']
  reference=Path(c['prior_workflow']);assert verify_export(reference)==c['prior_workflow_manifest_sha256']
  receipt=json.loads((reference/'CANDIDATE_RECEIPT.json').read_text());prior=reference/'model_runs'/Path(receipt['run_path']).name/'checkpoint';assert sha(prior/'adapter_model.safetensors')==c['prior_candidate_sha256']
  old_env=json.loads((prior.parent/'environment.json').read_text())
  import torch
  torch.set_num_threads(1);torch.set_num_interop_threads(1)
  if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Exactly one visible CUDA device required')
  env={'versions':{n:importlib.metadata.version(n) for n in old_env['versions']},'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda}
  atomic(out/'environment.json',env)
  if env!=old_env:raise ValueError('Runtime changed; CPU/batch parity required before optimizer')
  from runtime_data import load_learning_pool,pending_coverage,reconcile_coverage
  manifest=json.loads((runtime/'RUNTIME_DATA_MANIFEST.json').read_text());pool,split,gold=load_learning_pool(runtime,manifest['dataset_manifest_sha256'],manifest['split_manifest_sha256'])
  model_pool=json.loads((cpu/'PRIVATE_MODEL_POOL.json').read_text());pool_manifest=json.loads((cpu/'POOL_MANIFEST.json').read_text());assert sha(cpu/'PRIVATE_MODEL_POOL.json')==pool_manifest['sha256']
  q_by_id={q['id']:q for q in pool};assert all(q_by_id.get(q['id'])==q for q in model_pool)
  from quality_gate import select,digest,promotion
  bank=json.loads((original/'verified_bank.json').read_text());evidence=json.loads((original/'PRIVATE_TARGET_EVIDENCE.json').read_text());old_rows,selection=select(bank['correction']+bank['replay'],evidence,c.get('historical_reference_gate',c['supervision_gate']));assert selection['dataset_sha256']==c['expected_training_dataset_sha256']
  from field_eval import TrustedShops,teacher_stats,verify_loaded,evaluate_rows
  shops=TrustedShops(cpu/'TRAIN_CACHE',c['cache_manifest_sha256'],c['runtime_manifest_sha256'])
  from modeling import load_base
  from peft import PeftModel
  from offline_guard import install
  install()
  with tempfile.TemporaryDirectory() as scratch:
   p=Path(scratch);(p/'assets').mkdir();(p/'assets/base').symlink_to(Path(c['base']),target_is_directory=True)
   phase('teacher_probe_parent_and_previous_candidate');model,tok=load_base(p);model=PeftModel.from_pretrained(model,c['adapter'],is_trainable=False,local_files_only=True);model.eval()
   parent_loaded=verify_loaded(model,c['adapter']);parent_stats=teacher_stats(model,tok,old_rows)
   model.load_adapter(str(prior),adapter_name='prior',is_trainable=False,local_files_only=True);model.set_adapter('prior');model.eval();prior_loaded=verify_loaded(model,prior,'prior');prior_stats=teacher_stats(model,tok,old_rows)
   atomic(out/'PREVIOUS_ROUND_FIELD_PROBE.json',{'parent':parent_stats,'prior_candidate':prior_stats,'parent_load':parent_loaded,'prior_load':prior_loaded,'weights_updated':False})
   model.set_adapter('default');model.eval();assert verify_loaded(model,c['adapter'])['adapter_sha256']==c['parent_sha256']
   phase('blind_train_collection');from baseline_bridge import BaselineBridge,canonical_correction
   from planner import validate_plan
   from scoring import matches
   cfg={'mode':'learning','init_mode':'warm_start','parent_sha256':c['parent_sha256'],'learning_businesses':split['groups']['learning'],'calibration_businesses':split['groups']['calibration'],'holdout_businesses':split['groups']['audit_exposed'],'attempts_per_question':1,'data_manifest_sha256':manifest['dataset_manifest_sha256'],'split_manifest_sha256':manifest['split_manifest_sha256'],'scope':'frozen supported TRAIN pilot; full pool coverage tracked separately'}
   collection=new_run(out/'collections',cfg,[runtime]);pending_coverage(collection/'FULL_POOL_COVERAGE.json',pool,c['parent_sha256'],manifest['split_manifest_sha256'])
   bridge=BaselineBridge(model,tok,shops,max_tokens=240,max_seconds=25);engine=LearningRound(collection,cfg,sorted(model_pool,key=lambda q:(q['business_id'],q['id'])),gold,bridge.predict,bridge.execute,validate_plan,bridge.semantic,matches,canonical_correction)
   metrics,replay,correction=engine.process(stop_requested=lambda:remaining()<3000);coverage=reconcile_coverage(collection/'FULL_POOL_COVERAGE.json',engine.ledger)
   gate=dict(c['supervision_gate']);gate.update(verifier_sha256=digest(pool_manifest['verifier_manifest']))
   from real_evidence import export
   quality=export(collection,pool,bridge,gold,gate);seal(collection)
   atomic(out/'COLLECTION_SUMMARY.json',{'metrics':metrics,'full_pool':coverage,'quality_gate':quality})
   if metrics['attempt_coverage']!=1 or quality['state']!='passed':raise ValueError('Incomplete attempts or insufficient verified supervision; optimizer blocked')
   fresh_bank=json.loads((collection/'verified_bank.json').read_text());fresh_ev=json.loads((collection/'PRIVATE_TARGET_EVIDENCE.json').read_text());selected,selection=select([r for r in fresh_bank['correction']+fresh_bank['replay'] if r['id'] in fresh_ev],fresh_ev,gate)
   atomic(out/'PRIVATE_SELECTED.json',selected);atomic(out/'SELECTION_REPORT.json',selection)
   del bridge,engine,model
  cleanup()
  phase('temporal_warm_start_training')
  train_config={**c,'mode':'learning','init_mode':'warm_start','gpu_confirmed':True,'collect_learning_probe':False,'supervision_gate':gate,'loss_objective':'temporal_payload_completion_v1','time_token_weight':4.,'gradient_accumulation':4,'lr':1e-5,'max_epochs_equivalent':2,'max_steps':min(32,len(selected)*2//4),'approved_max_steps':32,'max_seconds':min(900,int(remaining()-300)),'approved_max_seconds':900,'output_root':str(out/'model_runs')}
  if train_config['max_seconds']<120:raise TimeoutError('Insufficient optimizer and checkpoint budget')
  train_config['child_timeout_seconds']=min(int(remaining()-120),train_config['max_seconds']+300)
  from isolated_training import train_verified_isolated
  run=Path(train_verified_isolated(train_config,fresh_bank['correction']+fresh_bank['replay'],fresh_ev));candidate=run/'checkpoint'
  from smoke_validation import safe_adapter_tensors,tensor_delta
  delta=tensor_delta(safe_adapter_tensors(Path(c['adapter'])/'adapter_model.safetensors'),safe_adapter_tensors(candidate/'adapter_model.safetensors'))
  atomic(out/'CANDIDATE_RECEIPT.json',{'run_path':str(run),'manifest_sha256':verify_export(run),'adapter_sha256':sha(candidate/'adapter_model.safetensors'),'adapter_delta':delta})
  cleanup()
  with tempfile.TemporaryDirectory() as scratch:
   p=Path(scratch);(p/'assets').mkdir();(p/'assets/base').symlink_to(Path(c['base']),target_is_directory=True)
   phase('reload_and_mandatory_post_tests');model,tok=load_base(p);model=PeftModel.from_pretrained(model,str(candidate),is_trainable=False,local_files_only=True);model.eval();loaded=verify_loaded(model,candidate)
   after_stats=teacher_stats(model,tok,old_rows);atomic(out/'NEW_CANDIDATE_FIELD_PROBE.json',{'loaded':loaded,'teacher':after_stats})
   from pilot_gate import after_training,learning_gate
   if c.get('frozen_regression_root'):
    from protected_regression import evaluate_protected
    post,legacy=evaluate_protected(model,tok,c['frozen_regression_root'],c['frozen_regression_manifest_sha256'],c['parent_sha256'],split['groups']['learning'],pool,gold,shops,out/'protected_regression',old_rows,c['expected_training_dataset_sha256'],stop_requested=lambda:remaining()<90)
   else:
    legacy=evaluate_rows(model,tok,old_rows,pool,gold,shops,out/'original14',lambda:remaining()<90)
    post=evaluate_rows(model,tok,selected,pool,gold,shops,out/'selected_regression',lambda:remaining()<90)
   legacy_gate=learning_gate(legacy)
   decision=after_training({'run_full_calibration':False,'run_public':False},lambda:post);decision['original14_gate']=legacy_gate;decision['eligible_for_full_calibration']=decision['learning_gate']['passed'] and legacy_gate['passed'];result.update(decision)
   del model
  cleanup();atomic(out/'PILOT_RESULT.json',result)
  if result['eligible_for_full_calibration'] and remaining()>c['calibration_min_remaining_seconds']:
   phase('full_calibration_candidate')
   before=json.loads((reference/'parent_calibration/EVALUATION.json').read_text());assert before['manifest_sha256']==c['calibration_manifest_sha256'] and before['decoder_sha256']==sha(here/'public_infer_811.py') and before['scorer_sha256']==sha(here/'baseline/app/scoring.py')
   atomic(out/'BASELINE_CALIBRATION_REUSE.json',{'source':str(reference/'parent_calibration/EVALUATION.json'),'source_sha256':sha(reference/'parent_calibration/EVALUATION.json'),'model_sha256':c['parent_sha256'],'runtime_equal':True,'decoder_and_scorer_equal':True,'calibration_manifest_equal':True})
   eval_config=out/'CALIBRATION_CONFIG.json';atomic(eval_config,c)
   with (out/'CALIBRATION_PRIVATE.log').open('w') as log:r=subprocess.run([sys.executable,str(here/'locked_eval.py'),str(eval_config),str(candidate),str(out/'candidate_calibration')],cwd=here,stdout=log,stderr=subprocess.STDOUT,timeout=min(3000,int(remaining()-90)))
   if r.returncode:raise RuntimeError('Candidate full calibration failed; preserve candidate')
   after=json.loads((out/'candidate_calibration/EVALUATION.json').read_text());cal=promotion(before,after,c['promotion_policy']);result.update(calibration_ran=True,promoted=cal['promote'],calibration_decision=cal)
  else:result.update(calibration_ran=False,promoted=False,calibration_skip_reason='learning gate failed or remaining budget below frozen threshold')
  result.update(state='completed',public_ran=False,elapsed_seconds=time.monotonic()-started,candidate_sha256=sha(candidate/'adapter_model.safetensors'),run_id=run.name);atomic(out/'WORKFLOW_COMPLETED.json',result);phase('completed','completed')
 except Exception as exc:
  atomic(out/'FAILURE.json',{'exception_type':type(exc).__name__,'elapsed_seconds':time.monotonic()-started,'partial_artifacts_preserved':True});phase('failed','failed');raise
 finally:seal(out)
 return result

if __name__=='__main__':
 import contextlib
 with Path('/kaggle/working/TIME_PILOT_PRIVATE.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):r=main(sys.argv[1])
 print('TIME_PILOT_RESULT',json.dumps(r),flush=True)
