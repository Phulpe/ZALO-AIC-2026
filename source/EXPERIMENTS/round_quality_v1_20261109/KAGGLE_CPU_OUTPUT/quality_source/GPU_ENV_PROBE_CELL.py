import json,importlib.metadata,importlib.util
import torch
probe={'cuda_available':torch.cuda.is_available(),'torch':torch.__version__,'torch_cuda':torch.version.cuda,'gpu_count':torch.cuda.device_count(),'bitsandbytes_present':importlib.util.find_spec('bitsandbytes') is not None}
if probe['cuda_available']:
    probe['gpu0']=torch.cuda.get_device_name(0)
    probe['gpu0_gib']=torch.cuda.get_device_properties(0).total_memory/2**30
if probe['cuda_available'] and probe['bitsandbytes_present']:
    import bitsandbytes as bnb
    x=torch.randn(64,64,device='cuda:0',dtype=torch.float16)
    packed,state=bnb.functional.quantize_4bit(x,quant_type='nf4')
    restored=bnb.functional.dequantize_4bit(packed,state)
    probe['nf4_kernel_finite']=bool(torch.isfinite(restored).all().item())
    p=torch.nn.Parameter(torch.randn(128,128,device='cuda:0',dtype=torch.float32))
    optimizer=bnb.optim.PagedAdamW8bit([p],lr=1e-3,min_8bit_size=0)
    loss=p.square().mean();loss.backward();optimizer.step()
    probe['bnb_optimizer_finite']=bool(torch.isfinite(p).all().item())
    del x,packed,state,restored,p,optimizer,loss
    torch.cuda.empty_cache()
Path('/kaggle/working/GPU_ENV_PROBE_V2.json').write_text(json.dumps(probe,indent=2))
print(json.dumps(probe,indent=2))
