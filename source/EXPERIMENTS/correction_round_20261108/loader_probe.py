"""Bounded loader diagnostics; aggregate metadata only, no private rows."""
from pathlib import Path
import json,time,hashlib,inspect

def probe(config_path):
    config=json.loads(Path(config_path).read_text())
    target=Path(config_path).parent/'LOADER_DIAGNOSTIC.json'
    report={'state':'starting','phases':{}}
    def save():target.write_text(json.dumps(report,indent=2))
    save()
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    report.update(torch=torch.__version__,threads=torch.get_num_threads(),cuda_available=torch.cuda.is_available())
    save()
    import transformers.core_model_loading as loading
    code=Path(inspect.getfile(loading));report['loader_source_sha256']=hashlib.sha256(code.read_bytes()).hexdigest()
    lines=code.read_text().splitlines()
    report['loader_source_lines_1260_1280']=lines[1259:1280]
    save()
    started=time.monotonic()
    tensor=torch.zeros((1024,1024),dtype=torch.float16)
    gpu=tensor.to('cuda:0');torch.cuda.synchronize()
    report['phases']['small_cuda_copy_seconds']=time.monotonic()-started
    del gpu,tensor
    torch.cuda.empty_cache()
    save()
    from safetensors import safe_open
    base=Path(config['base'])
    shard=next(iter(sorted(base.glob('*.safetensors'))))
    started=time.monotonic()
    with safe_open(shard,framework='pt',device='cpu') as f:
        key=next(iter(f.keys()))
        value=f.get_slice(key)
        report['first_tensor_shape']=list(value.get_shape())
    report['phases']['first_shard_header_seconds']=time.monotonic()-started
    report['state']='probe_passed';save()
