"""Qwen3.5 text-only NF4 loading; inference never downloads or calls APIs."""
from pathlib import Path
import json
import torch
from transformers import AutoTokenizer,Qwen3_5ForCausalLM,BitsAndBytesConfig

def load_base(project,training=False):
 p=Path(project);base=p/'assets/base'
 assert (base/'config.json').is_file(),'Prepare model assets before offline loading'
 if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required')
 tok=AutoTokenizer.from_pretrained(str(base),local_files_only=True,trust_remote_code=False)
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 tok.padding_side='right' if training else 'left'
 quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
 model=Qwen3_5ForCausalLM.from_pretrained(str(base),local_files_only=True,trust_remote_code=False,quantization_config=quant,device_map={'':0},dtype=torch.float16,attn_implementation='sdpa')
 count=model.num_parameters()
 assert count<=15_000_000_000,'Parameter ceiling exceeded'
 print('TEXT_MODEL_LOADED:',json.dumps({'parameters':count,'gpu_memory_GB':torch.cuda.memory_allocated()/2**30,'model_class':type(model).__name__}),flush=True)
 return model,tok

def chat(tok,system,prompt):
 return tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)

def prepare_qlora(model):
 # Equivalent frozen-base/checkpoint setup without PEFT's blanket fp32
 # upcast of the two 1B-parameter vocabulary matrices, which OOMs on T4.
 for param in model.parameters():param.requires_grad=False
 for name,module in model.named_modules():
  if 'norm' in name.rsplit('.',1)[-1].lower():
   for param in module.parameters(recurse=False):
    if param.is_floating_point():param.data=param.data.to(torch.float32)
 model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
 model.enable_input_require_grads();model.config.use_cache=False
 return model
