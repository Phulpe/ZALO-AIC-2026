"""Post-training checks. Does not load GPU models or pickle state files."""
import math

def tensor_delta(parent,candidate):
    import numpy as np
    if set(parent)!=set(candidate) or not parent:raise ValueError('Adapter keys mismatch')
    changed=0;delta_sq=0.;count=0
    for name in parent:
        a=np.asarray(parent[name],dtype=np.float64);b=np.asarray(candidate[name],dtype=np.float64)
        if a.shape!=b.shape:raise ValueError('Adapter tensor shape mismatch')
        if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite adapter weights')
        d=b-a;delta_sq+=float(np.sum(d*d));changed+=int(np.count_nonzero(d));count+=a.size
    if not changed or not math.isfinite(delta_sq):raise ValueError('No finite learned weight update')
    return {'all_weights_finite':True,'changed_elements':changed,'total_elements':count,'delta_l2':math.sqrt(delta_sq)}

def safe_adapter_tensors(path):
    from safetensors.torch import load_file
    # Safetensors only. Never torch.load optimizer/RNG files for this check.
    return {k:v.float().cpu().numpy() for k,v in load_file(str(path),device='cpu').items()}

def validate_after_training(parent,checkpoint,checkpoint_verifier,reload_forward):
    """reload_forward must actually reload the saved adapter and return evidence.

    This function does not fabricate reload results, alter a run manifest, or
    update BEST_MODEL/LATEST_SYNCED. Write its receipt outside the sealed run.
    """
    manifest_sha=checkpoint_verifier(checkpoint.parent)
    delta=tensor_delta(safe_adapter_tensors(parent/'adapter_model.safetensors'),safe_adapter_tensors(checkpoint/'adapter_model.safetensors'))
    forward=reload_forward(checkpoint)
    if forward.get('reloaded_from_saved_adapter') is not True or not math.isfinite(forward.get('loss',float('nan'))) or forward.get('mask_shift_verified') is not True:
        raise ValueError('Checkpoint reload/forward/mask evidence failed')
    return {'checkpoint_manifest_sha256':manifest_sha,'adapter_delta':delta,'reload':forward,'status':'passed','checkpoint_full_resume_verified':False}

def reload_forward_checkpoint(checkpoint,base,rows):
    """Actual offline PEFT reload + native/masked loss parity on a tiny fixture.

    Call only after permissions, CUDA/dependencies and GPU budget gates pass.
    Caller must release the training model and stop the GPU session afterwards.
    """
    import gc,tempfile
    from pathlib import Path
    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from modeling import load_base
    from train import CompletionDataset,collate
    gc.collect();torch.cuda.empty_cache()
    with tempfile.TemporaryDirectory(prefix='zaic_reload_') as scratch:
        project=Path(scratch);(project/'assets').mkdir();(project/'assets/base').symlink_to(Path(base).resolve(),target_is_directory=True)
        model,tok=load_base(project,False)
        model=PeftModel.from_pretrained(model,str(checkpoint),is_trainable=False,local_files_only=True)
        model.eval();model.config.use_cache=False
        ds=CompletionDataset(rows,tok,2048)
        if len(ds)!=len(rows) or ds.skipped or not len(ds):raise ValueError('Reload fixture invalid')
        batch={k:v.to('cuda:0') for k,v in collate([ds[0]]).items()};labels=batch.pop('labels')
        n=int(labels.ne(-100).sum().item())
        if n<=0 or not labels[:,:-n].eq(-100).all() or not labels[:,-n:].ge(0).all():raise ValueError('Completion mask invalid')
        with torch.inference_mode():
            reference=model(**batch,labels=labels,use_cache=False)
            native_loss=float(reference.loss.item());del reference
            compact=model(**batch,logits_to_keep=n+1,use_cache=False)
            logits=compact.logits[:,:-1,:].float()
            loss=float(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels[:,-n:].reshape(-1)).item())
        if not math.isfinite(loss) or not math.isfinite(native_loss) or abs(loss-native_loss)>.02:raise ValueError('Native/compact loss parity failed')
        result={'reloaded_from_saved_adapter':True,'loss':loss,'native_loss':native_loss,'loss_delta':abs(loss-native_loss),'mask_shift_verified':True,'input_tokens':int(batch['input_ids'].shape[1]),'completion_tokens':n}
        del model,batch,labels,compact,logits
        gc.collect();torch.cuda.empty_cache()
        return result
