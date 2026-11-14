"""New warm-start runner. Baseline imports are from the isolated copied source."""
from pathlib import Path
import json, sys, math, time, random, importlib.metadata, shutil
from relearning import new_run, warm_start, atomic, sha, seal, track, REVISION
from permission import permitted

def train(config,rows):
    state={}
    try:
        return _train(config,rows,state)
    except Exception as exc:
        if 'run' in state:atomic(state['run']/'FAILURE.json',{'state':'failed','error_type':type(exc).__name__,'partial_checkpoints_preserved':True})
        raise

def _train(config,rows,state):
    if config.get('mode')!='learning' or config.get('init_mode')!='warm_start':raise ValueError('Wrong mode')
    if not config.get('gpu_confirmed'):raise ValueError('GPU execution not reviewed')
    if not permitted(config.get('competition_permission_evidence')):raise ValueError('Competition permission evidence not accepted for this run')
    if not config.get('user_authorized'):raise ValueError('User authorization missing')
    if type(config.get('max_steps'))!=int or not 1<=config['max_steps']<=config['approved_max_steps']:raise ValueError('Step budget required')
    if not 0<config.get('max_seconds',0)<=config['approved_max_seconds']:raise ValueError('Time budget required')
    here=Path(__file__).parent
    sys.path.insert(0,str(here/'baseline/app'))
    from modeling import load_base,prepare_qlora
    from train import CompletionDataset,collate,CompletionTrainer
    from planner import validate_plan
    import torch
    from transformers import TrainingArguments,TrainerCallback
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    base=Path(config['base']);adapter=Path(config['adapter'])
    hashes=config['base_hashes']
    if not hashes or set(hashes)!={p.name for p in base.iterdir() if p.is_file()}:raise ValueError('Base manifest incomplete')
    for name,h in hashes.items():
        if sha(base/name)!=h:raise ValueError('Base source mismatch')
    if config['revision']!=REVISION:raise ValueError('Wrong revision')
    disk_path=Path(config['output_root']).resolve()
    while not disk_path.exists():disk_path=disk_path.parent
    if shutil.disk_usage(disk_path).free<4*1024**3:raise ValueError('Insufficient checkpoint disk space')
    if not rows or any(r.get('target_verified') is not True or r.get('split')!='learning' or r['business_id'] not in config['learning_businesses'] for r in rows):raise ValueError('Unverified targets/split')
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate targets')
    for r in rows:validate_plan(json.loads(r['response']))
    run=new_run(config['output_root'],config,[base,adapter,here/'baseline'])
    state['run']=run
    atomic(run/'data_manifest.json',{'count':len(rows),'sha256':__import__('relearning').fingerprint(rows),'ids':[r['id'] for r in rows]})
    # Scratch assets are symlinked; never export base or raw training data.
    import tempfile
    with tempfile.TemporaryDirectory(prefix='zaic_warm_') as scratch:
        p=Path(scratch);(p/'assets').mkdir();(p/'assets/base').symlink_to(base.resolve(),target_is_directory=True)
        model,tok=load_base(p,True);model=prepare_qlora(model)
        model,parent=warm_start(model,adapter,config)
        atomic(run/'parent.json',parent)
        if config.get('collect_learning_probe'):
            from learning_probe import collect
            probe=collect(model,tok,config)
            atomic(run/'LEARNING_PROBE_REFERENCE.json',probe)
        ds=CompletionDataset(rows,tok,config.get('max_length',2048))
        if ds.skipped or len(ds)!=len(rows):raise ValueError('Targets invalid or overlength; no silent dropping')
        versions={}
        for name in ['torch','transformers','peft','accelerate','bitsandbytes']:
            versions[name]=importlib.metadata.version(name)
        atomic(run/'environment.json',{'versions':versions,'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda})
        start=time.monotonic()
        class Safety(TrainerCallback):
            def on_pre_optimizer_step(self,args,state,control,model=None,**kw):
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad):raise FloatingPointError('Nonfinite gradients')
            def on_step_end(self,args,state,control,**kw):
                if time.monotonic()-start>config['max_seconds']:control.should_training_stop=True
            def on_log(self,args,state,control,logs=None,**kw):
                if any(isinstance(v,(float,int)) and not math.isfinite(v) for v in (logs or {}).values()):raise FloatingPointError('Nonfinite metrics')
        args=TrainingArguments(output_dir=str(run/'intermediate'),max_steps=config['max_steps'],learning_rate=config.get('lr',5e-6),per_device_train_batch_size=1,gradient_accumulation_steps=config.get('gradient_accumulation',16),fp16=True,bf16=False,optim='paged_adamw_8bit',gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},save_strategy='steps',save_steps=1,save_total_limit=None,logging_steps=1,report_to='none',remove_unused_columns=False,seed=config.get('seed',20261108),max_grad_norm=.3)
        trainer=CompletionTrainer(model=model,args=args,train_dataset=ds,data_collator=collate,processing_class=tok,callbacks=[Safety()])
        # New Trainer => new optimizer/scheduler; never infer a resume directory.
        if trainer.optimizer is not None or trainer.lr_scheduler is not None:raise ValueError('Unexpected optimizer state')
        result=trainer.train(resume_from_checkpoint=False)
        ck=run/'checkpoint';ck.mkdir(exist_ok=False)
        trainer.save_model(str(ck));tok.save_pretrained(str(ck))
        torch.save(trainer.optimizer.state_dict(),ck/'optimizer.pt')
        torch.save(trainer.lr_scheduler.state_dict(),ck/'scheduler.pt')
        import numpy as np
        torch.save({'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()},ck/'rng_state.pth')
        scaler=trainer.accelerator.scaler
        if scaler is not None:torch.save(scaler.state_dict(),ck/'scaler.pt')
        trainer.state.save_to_json(str(ck/'trainer_state.json'))
        atomic(run/'training_result.json',{'state':'completed','steps':trainer.state.global_step,'metrics':result.metrics,'init_mode':'warm_start','resume_verified':False})
        seal(run);track(Path(config['output_root'])/'STATE.json',run,'checkpoint',{})
        return run
