"""Offline runtime experiment: length-aware batches, recoverable plans, atomic outputs.
Deploy outside frozen app. Calibrate batch parity before public/Docker use.
"""
from pathlib import Path
from collections import Counter
import argparse,gc,hashlib,json,os,time,traceback

def atomic_output(path,questions,predictions):
 from audit_output import validate
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+'.tmp')
 with tmp.open('w',encoding='utf-8') as f:
  for q in questions:
   row=predictions.get(q['id'],{'id':q['id'],'type':'abstain'})
   validate(row);f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
  f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)

def recover_plans(path):
 path=Path(path)
 if not path.exists():return {}
 data=path.read_bytes();lines=data.splitlines(keepends=True);records={};valid=0
 for i,line in enumerate(lines):
  if not line.endswith(b'\n'):
   if i!=len(lines)-1:raise ValueError('Truncated plan journal before last line')
   break
  r=json.loads(line);assert isinstance(r['id'],str)
  if r['id'] in records:assert records[r['id']]==r,'Conflicting plan records'
  records[r['id']]=r;valid+=len(line)
 if valid<len(data):
  with path.open('r+b') as f:f.truncate(valid)
 return records

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def fingerprint_inputs(root,businesses):
 root=Path(root).resolve();files={};data=(root/'data').resolve()
 for biz in sorted(businesses):
  folder=(data/biz).resolve();assert folder.is_relative_to(data)
  for f in sorted(folder.rglob('*')):
   if f.is_file():files[f.relative_to(root).as_posix()]={'bytes':f.stat().st_size,'sha256':sha(f)}
 for f in sorted(root.iterdir()):
  if f.is_file() and f.name not in {'answers.jsonl','questions.jsonl'}:files[f.name]={'bytes':f.stat().st_size,'sha256':sha(f)}
 return root,files

def thinking_span(raw,eos=''):
 if eos and raw.endswith(eos):raw=raw[:-len(eos)]
 start=raw.find('<think>')
 if start<0:return None
 start+=len('<think>');end=raw.rfind('</think>')
 return raw[start:end] if end>=start else raw[start:]

def atomic_logs(path,questions,logs):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
 with tmp.open('w',encoding='utf-8') as f:
  for q in questions:
   if q['id'] not in logs:continue
   row=logs[q['id']]
   assert row['id']==q['id'] and type(row['elapsed_ms']) is int and row['elapsed_ms']>=0
   assert {'id','answer','elapsed_ms','thinking','code','inputs','output'}<=set(row)
   f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
  f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)

def main():
 started=time.monotonic()
 pa=argparse.ArgumentParser();pa.add_argument('--project',required=True);pa.add_argument('--root',required=True);pa.add_argument('--questions',required=True);pa.add_argument('--variant',choices=['base','candidate'],required=True);pa.add_argument('--output-dir',required=True);pa.add_argument('--batch-size',type=int,default=4);pa.add_argument('--cache-dir',required=True);pa.add_argument('--max-input-tokens',type=int,default=3072);a=pa.parse_args()
 assert 1<=a.batch_size<=32
 assert 256<=a.max_input_tokens<=8192
 p=Path(a.project);root=Path(a.root).resolve();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 import sys
 sys.path.insert(0,str(p/'app'))
 os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',DO_NOT_TRACK='1')
 from offline_guard import install,ATTEMPTS
 install()
 import torch
 from peft import PeftModel
 from modeling import load_base,chat
 from planner import SYSTEM,context,prompt_text,parse_plan,answer,VERSION
 from abstention_policy import load_policy,apply_policy
 from legacy_engine import prepare_shop
 policy_path=p/'abstention_policy.json';policy=load_policy(policy_path)
 qs=[json.loads(l) for l in Path(a.questions).read_text(encoding='utf-8-sig').splitlines() if l.strip()]
 assert len({q['id'] for q in qs})==len(qs)
 root,business_files=fingerprint_inputs(root,{q['business_id'] for q in qs})
 config={'model':json.loads((p/'model_lock.json').read_text()),'variant':a.variant,'questions_sha256':sha(a.questions),'source_sha256':{f.name:sha(f) for f in sorted((p/'app').glob('*.py'))},'runtime_sha256':sha(__file__),'audit_source_sha256':sha(Path(__file__).with_name('audit_output.py')),'business_files':business_files,'batch_size':a.batch_size,'max_new_tokens':320,'max_input_tokens':a.max_input_tokens,'thinking':False,'sort_by_length':True}
 if a.variant=='candidate':config['adapter_sha256']=sha(p/'candidate/final/adapter_model.safetensors')
 config['abstention_policy_sha256']=sha(policy_path) if policy_path.exists() else None
 key=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest();fp=out/'fingerprint.json'
 if fp.exists():assert json.loads(fp.read_text())['key']==key,'Incompatible run checkpoint; use a new output directory'
 else:fp.write_text(json.dumps({'key':key,**config},ensure_ascii=False,indent=2))
 journal=out/'plans.jsonl';done=recover_plans(journal);assert set(done)<={q['id'] for q in qs}
 cache_namespaces={biz:Path(a.cache_dir)/hashlib.sha256(json.dumps({k:v for k,v in business_files.items() if k.startswith('data/'+biz+'/')},sort_keys=True).encode()).hexdigest() for biz in {q['business_id'] for q in qs}}
 preds={};traces={};reasons=Counter();errors={};timings=[];oom_retries=0
 generated_now=set()
 logs_path=out/'logs/questions.jsonl'
 if logs_path.exists():traces={r['id']:r for r in map(json.loads,logs_path.read_text().splitlines())}
 previous={r['id']:r for r in map(json.loads,(out/'submission.jsonl').read_text().splitlines())} if (out/'submission.jsonl').exists() else {}
 assert set(previous)<={q['id'] for q in qs}
 atomic_output(out/'submission.jsonl',qs,previous)
 model,tok=load_base(p)
 if a.variant=='candidate':model=PeftModel.from_pretrained(model,str(p/'candidate/final'),local_files_only=True,is_trainable=False)
 model.eval();model.config.use_cache=True;tok.padding_side='left';torch.manual_seed(20261107)
 rules=(root/'rulebook.md').read_text(encoding='utf-8-sig');model_ready=time.monotonic()
 def save_many(records):
  with journal.open('a',encoding='utf-8') as f:
   for record in records:
    done[record['id']]=record;generated_now.add(record['id']);f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
   f.flush();os.fsync(f.fileno())
 def save(record):save_many([record])
 def generate(batch):
  nonlocal oom_retries
  ids=None;generated=None;oom=False
  try:
   ids=tok([r['chat'] for r in batch],return_tensors='pt',padding=True,add_special_tokens=False).to('cuda');width=ids['input_ids'].shape[1]
   stamp=time.monotonic()
   with torch.inference_mode():generated=model.generate(**ids,do_sample=False,num_beams=1,max_new_tokens=320,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
   records=[]
   for r,seq in zip(batch,generated[:,width:].tolist()):
    if tok.eos_token_id in seq:seq=seq[:seq.index(tok.eos_token_id)+1]
    text=tok.decode(seq,skip_special_tokens=True);raw=tok.decode(seq,skip_special_tokens=False)
    records.append({'id':r['q']['id'],'plan':parse_plan(text),'input_tokens':r['length'],'prompt_sha256':r['prompt_sha256'],'output_tokens':len(seq),'thinking':thinking_span(raw,tok.eos_token),'model_output':raw,'max_new_tokens_hit':len(seq)==320 and tok.eos_token_id not in seq})
   timings.append({'n':len(batch),'seconds':time.monotonic()-stamp,'padded_input_tokens':width,'generated_width':generated.shape[1]-width})
   save_many(records)
  except torch.OutOfMemoryError:oom=True
  if oom:
   oom_retries+=1;ids=None;generated=None;gc.collect();torch.cuda.empty_cache()
   if len(batch)>1:
    mid=len(batch)//2;generate(batch[:mid]);generate(batch[mid:])
   else:save({'id':batch[0]['q']['id'],'plan':None,'runtime_error':'single_item_oom'})
 for biz in sorted({q['business_id'] for q in qs}):
  business_entered=time.monotonic()
  bqs=[q for q in qs if q['business_id']==biz]
  try:shop=prepare_shop(root/'data'/biz,cache_namespaces[biz])
  except Exception as exc:shop=None;errors[biz]=type(exc).__name__
  pending=[]
  for q in bqs:
   if q['id'] in done:continue
   if shop is None:save({'id':q['id'],'plan':None,'runtime_error':'loader_unavailable'});continue
   prompt=prompt_text(context(shop,q['question'],q['as_of_date'],rules));rendered=chat(tok,SYSTEM,prompt);n=len(tok(rendered,add_special_tokens=False)['input_ids'])
   if n>a.max_input_tokens:save({'id':q['id'],'plan':None,'runtime_error':'context_overlength'})
   else:pending.append({'q':q,'chat':rendered,'length':n,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()})
  def execute_ready():
   for q in bqs:
    if q['id'] in preds or q['id'] not in done:continue
    plan=done[q['id']]['plan']
    exception_info=None
    try:body=answer(shop,q['question'],q['as_of_date'],plan) if shop is not None and plan is not None else {'type':'abstain','reason':done[q['id']].get('runtime_error','invalid_plan')}
    except Exception as exc:
     body={'type':'abstain','reason':'executor_'+type(exc).__name__}
     exception_info={'type':type(exc).__name__,'frames':[{'file':Path(f.filename).name,'function':f.name,'line':f.lineno} for f in traceback.extract_tb(exc.__traceback__)[-4:]]}
    pre_policy_output=dict(body)
    body=apply_policy(plan or {},body,policy)
    computed_output=dict(body)
    if body.get('reason'):reasons[body['reason']]+=1
    if body['type'] in {'abstain','unanswerable'}:body={'type':body['type']}
    preds[q['id']]={'id':q['id'],**body}
    traces[q['id']]={'id':q['id'],'answer':body,'elapsed_ms':max(0,int((time.monotonic()-business_entered)*1000)),'thinking':done[q['id']].get('thinking'),'code':str(Path(__file__).resolve())+'::main.execute_ready','inputs':{'question':q['question'],'as_of':q['as_of_date'],'plan':plan,'shop_preparation':{'source':str((p/'app/legacy_engine.py').resolve())+'::prepare_shop','folder':str(root/'data'/biz),'cache_dir':str(cache_namespaces[biz])}},'output':computed_output,'model_output':done[q['id']].get('model_output'),'model_calls_this_run':int(q['id'] in generated_now and 'model_output' in done[q['id']]),'model_generation_source':('current_run' if q['id'] in generated_now else 'journal_replay') if 'model_output' in done[q['id']] else 'no_model_call','timing_scope':'business batch entry before data reads; GPU microbatches'}
    if exception_info is not None:traces[q['id']]['executor_exception']=exception_info
    if pre_policy_output!=body:traces[q['id']]['pre_policy_output']=pre_policy_output
   atomic_logs(logs_path,qs,traces)
   atomic_output(out/'submission.jsonl',qs,{**previous,**preds})
   print('BATCH_PROGRESS:',len(preds),'/',len(qs),'SECONDS',round(time.monotonic()-started,2),flush=True)
  execute_ready();pending.sort(key=lambda r:(r['length'],r['q']['id']))
  for i in range(0,len(pending),a.batch_size):generate(pending[i:i+a.batch_size]);execute_ready()
  del shop,pending;gc.collect()
 assert len(preds)==len(qs) and len(traces)==len(qs) and not ATTEMPTS
 result={'state':'completed','variant':a.variant,'n':len(qs),'by_type':dict(Counter(r['type'] for r in preds.values())),'loader_errors':errors,'invalid_plans':sum(r['plan'] is None for r in done.values()),'network_attempts':len(ATTEMPTS),'seconds_including_load_and_fingerprint':time.monotonic()-started,'startup_seconds':model_ready-started,'generation_seconds':sum(t['seconds'] for t in timings),'batch_size_requested':a.batch_size,'batch_timings':timings,'oom_retries':oom_retries,'peak_gpu_GB':torch.cuda.max_memory_allocated()/2**30,'reasons':dict(reasons),'engine':VERSION,'runtime':'batch-0.2.0','question_logs':len(traces),'log_path':str(logs_path),'timing_scope':'elapsed_ms starts at business-batch entry before data reads; shared initialization reported separately','output_sha256':sha(out/'submission.jsonl')}
 (out/'runtime.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print('BATCH_INFERENCE_COMPLETED:',json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
