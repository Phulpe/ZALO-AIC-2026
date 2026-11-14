"""Offline inference only: this module never opens reference answers."""
import os
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',DO_NOT_TRACK='1')
from offline_guard import install,ATTEMPTS
install()
from pathlib import Path
from collections import Counter
import argparse,gc,hashlib,json,time
import torch
from peft import PeftModel
from modeling import load_base,chat
from planner import SYSTEM,context,prompt_text,parse_plan,answer,VERSION
from legacy_engine import prepare_shop

def main():
 pa=argparse.ArgumentParser();pa.add_argument('--project',required=True);pa.add_argument('--root',required=True);pa.add_argument('--questions',required=True);pa.add_argument('--variant',choices=['base','candidate'],required=True);pa.add_argument('--output-dir',required=True);a=pa.parse_args()
 p=Path(a.project);root=Path(a.root);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 questions=[json.loads(l) for l in Path(a.questions).read_text(encoding='utf-8-sig').splitlines() if l.strip()];assert len({q['id'] for q in questions})==len(questions)
 fingerprint={'model':json.loads((p/'model_lock.json').read_text()),'variant':a.variant,'questions_sha256':hashlib.sha256(Path(a.questions).read_bytes()).hexdigest(),'code_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')},'max_new_tokens':320,'max_input_tokens':2048,'thinking':False}
 if a.variant=='candidate':fingerprint['adapter_sha256']=hashlib.sha256((p/'candidate/final/adapter_model.safetensors').read_bytes()).hexdigest()
 key=hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest();metadata=out/'fingerprint.json'
 if metadata.exists():assert json.loads(metadata.read_text())['key']==key,'Incompatible checkpoint'
 else:metadata.write_text(json.dumps({'key':key,**fingerprint},indent=2))
 saved=out/'plans.jsonl';done={r['id']:r for r in map(json.loads,saved.read_text().splitlines())} if saved.exists() else {}
 model,tok=load_base(p)
 if a.variant=='candidate':model=PeftModel.from_pretrained(model,str(p/'candidate/final'),local_files_only=True,is_trainable=False)
 model.eval();model.config.use_cache=True;tok.padding_side='left';torch.manual_seed(20261107)
 began=time.monotonic();rulebook=(root/'rulebook.md').read_text(encoding='utf-8-sig');preds={};reasons=Counter();loader_errors={};seconds_generate=0
 for biz in sorted({q['business_id'] for q in questions}):
  folder=(root/'data'/biz).resolve();assert folder.is_relative_to((root/'data').resolve())
  try:shop=prepare_shop(folder,p.parent/'cache_033')
  except Exception as exc:shop=None;loader_errors[biz]=type(exc).__name__
  for q in [r for r in questions if r['business_id']==biz]:
   if q['id'] in done:record=done[q['id']];plan=record['plan']
   elif shop is None:plan=None
   else:
    prompt=prompt_text(context(shop,q['question'],q['as_of_date'],rulebook))
    inputs=tok(chat(tok,SYSTEM,prompt),return_tensors='pt',add_special_tokens=False).to('cuda')
    if inputs['input_ids'].shape[1]>2048:plan=None;reasons['context_overlength']+=1
    else:
     stamp=time.monotonic()
     with torch.inference_mode():generated=model.generate(**inputs,do_sample=False,num_beams=1,max_new_tokens=320,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
     text=tok.decode(generated[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True);plan=parse_plan(text);seconds_generate+=time.monotonic()-stamp
     record={'id':q['id'],'plan':plan,'input_tokens':inputs['input_ids'].shape[1],'output_tokens':generated.shape[1]-inputs['input_ids'].shape[1],'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()}
     done[q['id']]=record
     with saved.open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush()
   body=answer(shop,q['question'],q['as_of_date'],plan) if shop is not None and plan is not None else {'type':'abstain','reason':'Missing/invalid/overlength plan or loader unavailable'}
   if body.get('reason'):reasons[body['reason']]+=1
   if body['type'] in {'abstain','unanswerable'}:body={'type':body['type']}
   preds[q['id']]={'id':q['id'],**body}
   print('INFER_PROGRESS:',a.variant,len(preds),'/',len(questions),'SECONDS',round(time.monotonic()-began,2),flush=True)
  del shop;gc.collect()
 output=out/'submission.jsonl';output.write_text(''.join(json.dumps(preds[q['id']],ensure_ascii=False,allow_nan=False)+'\n' for q in questions))
 assert not ATTEMPTS,'Outbound network attempt during inference'
 summary={'variant':a.variant,'n':len(questions),'by_type':dict(Counter(v['type'] for v in preds.values())),'invalid_plans':sum(r['plan'] is None for r in done.values()),'loader_errors':loader_errors,'network_attempts':len(ATTEMPTS),'seconds':time.monotonic()-began,'generation_seconds':seconds_generate,'reasons':dict(reasons),'engine':VERSION,'output_sha256':hashlib.sha256(output.read_bytes()).hexdigest()}
 (out/'runtime.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print('OFFLINE_INFERENCE_COMPLETED:',json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
