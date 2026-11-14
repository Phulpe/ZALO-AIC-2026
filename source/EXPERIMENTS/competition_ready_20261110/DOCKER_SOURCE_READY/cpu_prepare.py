"""Tokenize audited TRAIN rows, revalidate targets, cache trusted data on CPU."""
from pathlib import Path
import json,sys,contextlib,pickle,importlib.metadata,collections
sys.path[:0]=[str(Path(__file__).parent),str(Path(__file__).parent/'baseline/app')]
from relearning import sha,verify_export,atomic,seal

def main(path):
 c=json.loads(Path(path).read_text());out=Path(c['cpu_output']);out.mkdir(exist_ok=False)
 runtime=Path(c['runtime_root']);assert sha(runtime/'RUNTIME_DATA_MANIFEST.json')==c['runtime_manifest_sha256']
 m=json.loads((runtime/'RUNTIME_DATA_MANIFEST.json').read_text())
 from runtime_data import load_learning_pool
 pool,split,gold=load_learning_pool(runtime,m['dataset_manifest_sha256'],m['split_manifest_sha256']);qs={q['id']:q for q in pool}
 from baseline_bridge import canonical_total,canonical_total_v2_legacy,BaselineBridge,matches
 supported=[q for q in pool if canonical_total(q) is not None]
 assert not gold.rows
 atomic(out/'PRIVATE_MODEL_POOL.json',supported)
 atomic(out/'POOL_MANIFEST.json',{'count':len(supported),'sha256':sha(out/'PRIVATE_MODEL_POOL.json'),'runtime_manifest_sha256':c['runtime_manifest_sha256'],'learning_businesses':split['groups']['learning'],'gold_opened':False,'full_learning_expected':len(pool),'verifier_manifest':{n:sha(Path(__file__).parent/n) for n in ['baseline_bridge.py','temporal_verifier.py']}})
 source=Path(c['verified_collection']);assert verify_export(source)==c['collection_manifest_sha256']
 from quality_gate import select
 bank=json.loads((source/'verified_bank.json').read_text());evidence=json.loads((source/'PRIVATE_TARGET_EVIDENCE.json').read_text());selected,selection=select(bank['correction']+bank['replay'],evidence,c['supervision_gate']);assert selection['dataset_sha256']==c['expected_training_dataset_sha256']
 # Tokenizer only: no base model, no CUDA allocation or optimizer.
 from offline_guard import install
 install()
 from transformers import AutoTokenizer
 base=Path(c['base'])
 for name in ['tokenizer.json','tokenizer_config.json','chat_template.jinja','vocab.json','merges.txt','config.json']:assert sha(base/name)==c['base_hashes'][name]
 tok=AutoTokenizer.from_pretrained(base,local_files_only=True,trust_remote_code=False);tok.padding_side='right'
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 from train import CompletionDataset
 from time_objective import training_classes
 from planner import SYSTEM
 temporal,collate,_=training_classes(4.);old=CompletionDataset(selected,tok);new=temporal(selected,tok)
 stats=collections.defaultdict(lambda:collections.Counter());assert len(old)==len(new)==len(selected)
 for row,a,b in zip(selected,old.items,new.items):
  assert a['input_ids']==b['input_ids'] and a['labels']==b['labels']
  prefix=tok(tok.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':row['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)['input_ids']
  n=sum(v!=-100 for v in b['labels']);assert b['input_ids'][:-n]==prefix and b['labels'][:-n]==[-100]*len(prefix)
  assert b['labels'][-n:]==b['input_ids'][-n:] and b['labels'][-1]==tok.eos_token_id and not b['time_focus'][-1]
  assert all(w==0 for w in b['loss_weights'][:-n]) and all(not v for v in b['time_focus'][:-n])
  count=stats[row['kind']];count.update(rows=1,completion_tokens=n,time_tokens=sum(b['time_focus']),prefix_tokens=len(prefix))
 token_report={'rows':len(selected),'prefix_train_inference_equal':True,'original_ids_labels_unchanged':True,'completion_mask_shift_and_eos_valid':True,'groups':dict(stats),'time_weight':4.,'model_loaded':False,'gpu_executed':False,'tokenizer_version':importlib.metadata.version('tokenizers')}
 atomic(out/'TOKEN_AUDIT.json',token_report)
 # Reuse only our pinned, hashed cache after parser/runtime/library checks.
 oldcache=Path(c['trusted_cache_root']);assert verify_export(oldcache)==c['trusted_cache_manifest_sha256'];oldmeta=json.loads((oldcache/'CACHE_PROVENANCE.json').read_text());parser=sha(Path(__file__).parent/'baseline/app/legacy_engine.py')
 assert oldmeta['parser_sha256']==parser and oldmeta['runtime_manifest_sha256']==c['runtime_manifest_sha256'] and oldmeta['pandas_version']==importlib.metadata.version('pandas')
 cache=out/'TRAIN_CACHE';cache.mkdir();files={};verdicts=collections.Counter();bridge=BaselineBridge(None,None,{})
 from legacy_engine import prepare_shop
 ledger=json.loads((source/'ledger.json').read_text())['questions'];rows_by_biz=collections.defaultdict(list)
 for row in bank['correction']+bank['replay']:rows_by_biz[row['business_id']].append(row)
 for biz in sorted(split['groups']['learning']):
  folder=runtime/'data'/biz
  for f in folder.rglob('*'):
   if f.is_file():assert sha(f)==m['files'][f.relative_to(runtime).as_posix()]['sha256']
  if biz in oldmeta['files']:
   f=(oldcache/oldmeta['files'][biz]['path']).resolve();assert f.is_relative_to(oldcache.resolve())
   with f.open('rb') as stream:shop=pickle.load(stream)
  else:shop=prepare_shop(folder,cache_dir=None)
  target=cache/(biz+'.pkl')
  with target.open('wb') as stream:pickle.dump(shop,stream,pickle.HIGHEST_PROTOCOL)
  files[biz]={'path':target.name,'sha256':sha(target),'bytes':target.stat().st_size};bridge.shops={biz:shop}
  for row in rows_by_biz[biz]:
   rec=ledger[row['id']];assert any('plan' in a or 'prediction_text' in a for a in rec['attempts'])
   plan=json.loads(row['response']);q=qs[row['id']]
   # Previously persisted blind attempt precedes this TRAIN-only revalidation.
   if bridge.semantic(q,plan) and matches(bridge.execute(q,plan),gold(row['id'])):verdicts[row['kind']]+=1
   else:raise ValueError('Previously verified target failed new gate')
  del shop
 atomic(cache/'CACHE_PROVENANCE.json',{'producer':'this CPU job; trusted generated cache','scope':'all learning businesses, no labels','parser_sha256':parser,'runtime_manifest_sha256':c['runtime_manifest_sha256'],'pandas_version':importlib.metadata.version('pandas'),'learning_businesses':split['groups']['learning'],'files':files});seal(cache)
 atomic(out/'CPU_PREPARATION_RESULT.json',{'state':'passed','supported_pool':len(supported),'previous_supported':sum(canonical_total_v2_legacy(q) is not None for q in pool),'new_pending_supported':sum(q['id'] not in ledger for q in supported),'cache_businesses':len(files),'cache_manifest_sha256':sha(cache/'ARTIFACT_MANIFEST.json'),'revalidated_existing_bank':dict(verdicts),'token_audit':token_report,'gpu_executed':False});seal(out)
 return json.loads((out/'CPU_PREPARATION_RESULT.json').read_text())

if __name__=='__main__':
 with Path('/kaggle/working/CPU_PREP_PRIVATE.log').open('w') as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):r=main(sys.argv[1])
 print(json.dumps(r))
