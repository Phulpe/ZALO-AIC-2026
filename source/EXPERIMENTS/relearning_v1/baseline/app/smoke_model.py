from pathlib import Path
import argparse,gc,json,time
import torch
import torch.nn.functional as F
from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
from modeling import load_base,chat,prepare_qlora

def main():
 pa=argparse.ArgumentParser();pa.add_argument('--project',required=True);a=pa.parse_args();p=Path(a.project)
 model,tok=load_base(p,True)
 model=prepare_qlora(model)
 for m in [model.get_input_embeddings(),model.get_output_embeddings()]:
  for param in m.parameters():param.data=param.data.to(torch.float16)
 model=get_peft_model(model,LoraConfig(task_type='CAUSAL_LM',r=8,lora_alpha=16,lora_dropout=.05,target_modules='all-linear',bias='none'))
 model.config.use_cache=False
 prefix=tok(chat(tok,'Return JSON only.','Calculate 1+1.'),add_special_tokens=False)['input_ids'];target=tok('{"answer":2}'+tok.eos_token,add_special_tokens=False)['input_ids']
 ids=torch.tensor([prefix+target],device='cuda');n=len(target)
 model.eval()
 with torch.no_grad():
  full=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits
  loss_full=F.cross_entropy(full[:,len(prefix)-1:-1,:].float().reshape(-1,full.shape[-1]),ids[:,-n:].reshape(-1))
  short=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=n+1).logits
  loss_short=F.cross_entropy(short[:,:-1,:].float().reshape(-1,short.shape[-1]),ids[:,-n:].reshape(-1))
  delta=abs(float(loss_full)-float(loss_short));assert delta<.02,'Completion logit slicing changed loss'
 del full,short;gc.collect();torch.cuda.empty_cache()
 model.train()
 # A longer synthetic prefix checks activation memory without competition content.
 long=tok(chat(tok,'Return JSON only.','Schema: date, amount, product. '*180+'Calculate 1+1.'),add_special_tokens=False)['input_ids'][-1200:]
 ids=torch.tensor([long+target],device='cuda')
 out=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=n+1)
 loss=F.cross_entropy(out.logits[:,:-1,:].float().reshape(-1,out.logits.shape[-1]),ids[:,-n:].reshape(-1));assert torch.isfinite(loss)
 loss.backward();grads=[v.grad for v in model.parameters() if v.requires_grad and v.grad is not None];assert grads and all(torch.isfinite(g).all() for g in grads)
 result={'state':'passed','loss':float(loss.detach()),'completion_loss_delta':delta,'sequence_tokens':ids.shape[1],'trainable_and_total':model.get_nb_trainable_parameters(),'peak_gpu_GB':torch.cuda.max_memory_allocated()/2**30,'adapter_saved':False,'competition_data_used':False}
 (p/'smoke_model.json').write_text(json.dumps(result,indent=2));print('MODEL9B_SMOKE_PASSED:',json.dumps(result),flush=True)
if __name__=='__main__':main()
