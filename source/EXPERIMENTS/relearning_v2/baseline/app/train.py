"""Fresh QLoRA adapter on the new 9B base, completion-only loss, frequent checkpoints."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,inspect,json,math,random,time
import torch
import torch.nn.functional as F
from transformers import Trainer,TrainingArguments,TrainerCallback
from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
from modeling import load_base,chat,prepare_qlora
from planner import SYSTEM,validate_plan

class CompletionDataset(torch.utils.data.Dataset):
 def __init__(self,rows,tok,maxlen=2048):
  self.items=[];self.skipped=0
  for r in rows:
   try:validate_plan(json.loads(r['response']))
   except ValueError:self.skipped+=1;continue
   prefix=tok(chat(tok,SYSTEM,r['prompt']),add_special_tokens=False)['input_ids']
   target=tok(r['response']+tok.eos_token,add_special_tokens=False)['input_ids']
   if len(prefix)+len(target)>maxlen:self.skipped+=1;continue
   self.items.append({'input_ids':prefix+target,'labels':[-100]*len(prefix)+target})
 def __len__(self):return len(self.items)
 def __getitem__(self,i):return self.items[i]

def collate(items):
 assert len(items)==1,'Memory-saving loss requires batch=1'
 ids=torch.tensor([items[0]['input_ids']],dtype=torch.long)
 return {'input_ids':ids,'attention_mask':torch.ones_like(ids),'labels':torch.tensor([items[0]['labels']],dtype=torch.long)}

class CompletionTrainer(Trainer):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  self.model_accepts_loss_kwargs=False
  if self.accelerator.scaler is not None:
   self.accelerator.scaler=torch.amp.GradScaler('cuda',init_scale=16.0,growth_interval=2000)
 def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
  labels=inputs.pop('labels');n=int(labels.ne(-100).sum().item())
  # Only materialize the logits needed to predict completion tokens. The full
  # prompt is still attended to; no input truncation and no label shifting shortcut.
  outputs=model(**inputs,logits_to_keep=n+1,use_cache=False)
  logits=outputs.logits[:,:-1,:].contiguous();targets=labels[:,-n:].contiguous()
  loss=F.cross_entropy(logits.float().view(-1,logits.shape[-1]),targets.view(-1))
  return (loss,outputs) if return_outputs else loss

class Progress(TrainerCallback):
 def __init__(self,path):self.path=Path(path);self.bad_gradients=0
 def on_log(self,args,state,control,logs=None,**kw):
  record={'step':state.global_step,'max_steps':state.max_steps,'epoch':state.epoch,**(logs or {})}
  scaler=getattr(kw.get('optimizer'),'scaler',None)
  if scaler is not None:record['loss_scale']=scaler.get_scale()
  if 'grad_norm' in record:
   finite=math.isfinite(float(record['grad_norm']))
   record['gradient_finite']=finite
   self.bad_gradients=0 if finite else self.bad_gradients+1
   if not finite:record['grad_norm']=None
  print('TRAIN_PROGRESS:',json.dumps(record),flush=True)
  self.path.write_text(json.dumps(record,indent=2))
  if self.bad_gradients>=3:raise FloatingPointError('Three nonfinite gradients; do not mark training complete')

def main():
 pa=argparse.ArgumentParser();pa.add_argument('--project',required=True);a=pa.parse_args();p=Path(a.project)
 rows=[json.loads(l) for l in (p/'data/sft_train.jsonl').read_text().splitlines() if l.strip()]
 manifest=json.loads((p/'data/manifest.json').read_text());split=manifest['split']
 assert all(r['business_id'] in split['train'] for r in rows)
 assert not set(split['train'])&(set(split['calibration'])|set(split['audit']))
 model,tok=load_base(p,True)
 assert 'logits_to_keep' in inspect.signature(model.forward).parameters,'Completion-only logits unsupported by installed loader'
 model=prepare_qlora(model)
 # Frozen embeddings/head are huge at this vocabulary size. Keep them fp16;
 # small norms and trainable LoRA parameters stay fp32.
 for module in [model.get_input_embeddings(),model.get_output_embeddings()]:
  for param in module.parameters():
   assert not param.requires_grad;param.data=param.data.to(torch.float16)
 config=LoraConfig(task_type='CAUSAL_LM',r=8,lora_alpha=16,lora_dropout=.05,target_modules='all-linear',bias='none')
 model=get_peft_model(model,config);model.config.use_cache=False
 trainable,total=model.get_nb_trainable_parameters();assert total<=15_000_000_000
 ds=CompletionDataset(rows,tok)
 assert len(ds)>=128,f'Only {len(ds)} usable training samples after token-length checks'
 out=p/'candidate';out.mkdir(exist_ok=True)
 meta={'model':json.loads((p/'model_lock.json').read_text()),'trainable_parameters':trainable,'total_parameters':total,'train_rows':len(ds),'skipped_overlength':ds.skipped,'max_length':2048,'epochs':1,'learning_rate':2e-5,'rank':8,'batch_size':1,'gradient_accumulation':16,'initial_loss_scale':16.0,'old_adapter_reused':False,'public_test_used':False,'data_sha256':manifest['training_sha256']}
 (out/'metadata.json').write_text(json.dumps(meta,indent=2))
 print('TRAIN_9B_CONFIG:',json.dumps(meta),flush=True)
 args=TrainingArguments(output_dir=str(out/'checkpoints'),num_train_epochs=1,learning_rate=2e-5,per_device_train_batch_size=1,gradient_accumulation_steps=16,warmup_steps=max(1,math.ceil(math.ceil(len(ds)/16)*.05)),lr_scheduler_type='cosine',fp16=True,bf16=False,optim='paged_adamw_8bit',gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},save_strategy='steps',save_steps=5,save_total_limit=3,logging_steps=1,eval_strategy='no',report_to='none',remove_unused_columns=False,seed=20261107,dataloader_num_workers=0,max_grad_norm=.3)
 trainer=CompletionTrainer(model=model,args=args,train_dataset=ds,data_collator=collate,processing_class=tok,callbacks=[Progress(p/'training_progress.json')])
 ckpts=sorted((out/'checkpoints').glob('checkpoint-*'),key=lambda x:int(x.name.split('-')[-1])) if (out/'checkpoints').exists() else []
 resume=str(ckpts[-1]) if ckpts else None
 begin=time.monotonic();result=trainer.train(resume_from_checkpoint=resume)
 trainer.save_model(str(out/'final'));tok.save_pretrained(str(out/'final'));trainer.save_metrics('train',result.metrics)
 final={**meta,'state':'completed','steps':trainer.state.global_step,'epoch':trainer.state.epoch,'seconds':time.monotonic()-begin,'adapter':str(out/'final'),'peak_gpu_GB':torch.cuda.max_memory_allocated()/2**30}
 (p/'training_complete.json').write_text(json.dumps(final,indent=2));print('TRAIN_9B_COMPLETED:',json.dumps(final),flush=True)
if __name__=='__main__':main()
