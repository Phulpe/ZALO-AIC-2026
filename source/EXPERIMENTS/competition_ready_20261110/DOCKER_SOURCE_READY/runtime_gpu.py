def single_gpu_env(environment):
    """A fresh child sees only physical GPU 0; prevents Trainer DataParallel batch=2."""
    return {**environment,'CUDA_VISIBLE_DEVICES':'0'}

def require_single_gpu(count):
    if count!=1:raise RuntimeError('Start a fresh child with CUDA_VISIBLE_DEVICES=0; loss requires batch=1')
