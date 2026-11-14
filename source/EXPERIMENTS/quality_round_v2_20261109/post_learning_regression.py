"""One-shot evaluation of trained correction/replay; no learning or retries."""
from pathlib import Path
import os,sys,json,tempfile
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
os.environ['HF_DEACTIVATE_ASYNC_LOAD']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
sys.path[:0]=[str(Path(__file__).parent),str(Path(__file__).parent/'baseline/app')]
from relearning import atomic,sha,seal,verify_export

def main(path):
 c=json.loads(Path(path).read_text());assert c['mode']=='evaluation';out=Path(c['regression_output']);out.mkdir(parents=True,exist_ok=False);source=Path(c['verified_collection']);assert verify_export(source)==c['collection_manifest_sha256']
 from quality_gate import select
 bank=json.loads((source/'verified_bank.json').read_text());evidence=json.loads((source/'PRIVATE_TARGET_EVIDENCE.json').read_text());rows=[row for row in bank['correction']+bank['replay'] if row['id'] in evidence];selected,report=select(rows,evidence,c['supervision_gate']);assert report['dataset_sha256']==c['expected_training_dataset_sha256']
 assert sha(Path(c['candidate_adapter'])/'adapter_model.safetensors')==c['candidate_sha256']
 from runtime_data import load_learning_pool
 root=Path(c['runtime_root']);assert sha(root/'RUNTIME_DATA_MANIFEST.json')==c['runtime_manifest_sha256'];manifest=json.loads((root/'RUNTIME_DATA_MANIFEST.json').read_text());pool,split,gold=load_learning_pool(root,manifest['dataset_manifest_sha256'],manifest['split_manifest_sha256']);qs={q['id']:q for q in pool}
 from offline_guard import install
 install()
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 from modeling import load_base
 from peft import PeftModel
 from baseline_bridge import BaselineBridge,matches
 from legacy_engine import prepare_shop
 from prediction_errors import BlindPredictionError
 trusted_cache=None
 if c.get('trusted_cache_root'):
  import importlib.metadata
  trusted_cache=Path(c['trusted_cache_root'])
  assert verify_export(trusted_cache)==c['trusted_cache_manifest_sha256']
  provenance=json.loads((trusted_cache/'CACHE_PROVENANCE.json').read_text())
  assert provenance['parser_sha256']==sha(Path(__file__).parent/'baseline/app/legacy_engine.py')
  assert provenance['runtime_manifest_sha256']==sha(root/'RUNTIME_DATA_MANIFEST.json')
  assert provenance['dataset_sha256']==report['dataset_sha256']
  assert provenance['pandas_version']==importlib.metadata.version('pandas')
 verdicts={};ledger={};kinds={r['id']:r['kind'] for r in selected}
 with tempfile.TemporaryDirectory() as scratch:
  p=Path(scratch);(p/'assets').mkdir();(p/'assets/base').symlink_to(Path(c['base']),target_is_directory=True);model,tok=load_base(p);model=PeftModel.from_pretrained(model,c['candidate_adapter'],is_trainable=False,local_files_only=True);model.eval();model.config.use_cache=True
  for biz in sorted({row['business_id'] for row in selected}):
   if trusted_cache is not None:
    # Only this task's pinned, hashed CPU-produced cache is allowed here.
    import pickle
    meta=provenance['files'][biz];cache_file=(trusted_cache/meta['path']).resolve();assert cache_file.is_relative_to(trusted_cache.resolve()) and sha(cache_file)==meta['sha256']
    with cache_file.open('rb') as stream:shop=pickle.load(stream)
   else:shop=prepare_shop(root/'data'/biz,cache_dir=None)
   bridge=BaselineBridge(model,tok,{biz:shop},max_tokens=240,max_seconds=25)
   for row in selected:
    if row['business_id']!=biz:continue
    q=qs[row['id']];rec={'state':'attempting'};ledger[q['id']]=rec;atomic(out/'PRIVATE_PREDICTIONS.json',ledger)
    try:
     plan=bridge.predict(q,1);rec.update(state='predicted',plan=plan);atomic(out/'PRIVATE_PREDICTIONS.json',ledger)
     # Evaluation labels open only after the blind plan is durable.
     verdict=bridge.semantic(q,plan) and matches(bridge.execute(q,plan),gold(q['id']))
    except Exception as exc:
     verdict=False;rec.update(state='error',exception_type=type(exc).__name__)
     if isinstance(exc,BlindPredictionError):rec['prediction_text']=exc.prediction_text
    verdicts[q['id']]=bool(verdict);atomic(out/'PRIVATE_PREDICTIONS.json',ledger)
 correction=[i for i in verdicts if kinds[i]=='correction'];replay=[i for i in verdicts if kinds[i]=='replay'];assert len(verdicts)==len(selected)
 atomic(out/'REGRESSION_SUMMARY.json',{'mode':'evaluation','scope':'trained learning targets, not calibration','expected':len(selected),'attempted':len(verdicts),'real_output_ids':sum('plan' in a or 'prediction_text' in a for a in ledger.values()),'correction_before_correct':0,'correction_after_correct':sum(verdicts[i] for i in correction),'correction_total':len(correction),'replay_before_correct':len(replay),'replay_after_correct':sum(verdicts[i] for i in replay),'correct_to_wrong':sum(not verdicts[i] for i in replay),'weights_updated':False,'candidate_sha256':c['candidate_sha256']});seal(out)
if __name__=='__main__':main(sys.argv[1])
