"""Actual tokenizer checks for the final payload objective and expanded pool."""
from pathlib import Path
import json,sys,collections
sys.path[:0]=[str(Path(__file__).parent),str(Path(__file__).parent/'baseline/app')]
from relearning import sha,verify_export,atomic

def main(c):
 prepared=Path(c['cpu_prepared']);assert verify_export(prepared)==c['cpu_manifest_sha256'];runtime=Path(c['runtime_root']);assert sha(runtime/'RUNTIME_DATA_MANIFEST.json')==c['runtime_manifest_sha256']
 from offline_guard import install
 install()
 from transformers import AutoTokenizer
 base=Path(c['base'])
 for name in ['tokenizer.json','tokenizer_config.json','chat_template.jinja','vocab.json','merges.txt','config.json']:assert sha(base/name)==c['base_hashes'][name]
 tok=AutoTokenizer.from_pretrained(base,local_files_only=True,trust_remote_code=False)
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 from baseline_bridge import canonical_total
 from modeling import chat
 from planner import SYSTEM
 pool=json.loads((prepared/'PRIVATE_MODEL_POOL.json').read_text());model_manifest=json.loads((prepared/'POOL_MANIFEST.json').read_text());assert sha(prepared/'PRIVATE_MODEL_POOL.json')==model_manifest['sha256']
 lengths=[]
 for q in pool:
  response=json.dumps(canonical_total(q),ensure_ascii=False);prefix=tok(chat(tok,SYSTEM,q['prompt']),add_special_tokens=False)['input_ids'];target=tok(response+tok.eos_token,add_special_tokens=False)['input_ids'];lengths.append(len(prefix)+len(target))
 assert len(pool)==124 and max(lengths)<=2048
 source=Path(c['verified_collection']);assert verify_export(source)==c['collection_manifest_sha256']
 from quality_gate import select
 bank=json.loads((source/'verified_bank.json').read_text());evidence=json.loads((source/'PRIVATE_TARGET_EVIDENCE.json').read_text());rows,report=select(bank['correction']+bank['replay'],evidence,c['supervision_gate']);assert report['dataset_sha256']==c['expected_training_dataset_sha256']
 from time_objective import training_classes,time_span,token_focus
 from train import CompletionDataset
 dataset,_,_=training_classes(4.);old=CompletionDataset(rows,tok);new=dataset(rows,tok);counts=collections.defaultdict(collections.Counter)
 for row,a,b in zip(rows,old.items,new.items):
  assert a['input_ids']==b['input_ids'] and a['labels']==b['labels'];n=sum(v!=-100 for v in b['labels']);assert b['labels'][-n:]==b['input_ids'][-n:];assert not any(b['time_focus'][:-n]);assert not b['time_focus'][-1] and b['labels'][-1]==tok.eos_token_id
  enc=tok(row['response']+tok.eos_token,add_special_tokens=False,return_offsets_mapping=True);obj=token_focus(enc['offset_mapping'],time_span(row['response']),len(row['response']));stats=counts[row['kind']];stats.update(rows=1,completion_tokens=n,payload_tokens=sum(b['time_focus']),whole_time_object_tokens=sum(obj))
 r={'state':'passed','pool_rows':len(pool),'pool_max_proposed_length':max(lengths),'all_pool_proposed_targets_within_context':True,'proposed_targets_unverified_no_new_gold_read':True,'original14_prefix_and_labels_unchanged':True,'original14_eos_and_focus_mask_valid':True,'groups':dict(counts),'objective':'temporal_payload_completion_v1','time_weight':4.,'gpu_executed':False,'base_model_loaded':False};atomic(Path('/kaggle/working/FINAL_TOKEN_AUDIT.json'),r);return r

if __name__=='__main__':print(json.dumps(main(json.loads(Path(sys.argv[1]).read_text()))))
