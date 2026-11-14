"""Reviewed correction/replay pilot gate; never launches from the default config."""
import json,math
from pathlib import Path

PARENT='e7dfa635beccce128c99f9d3fd993d33dbb4c88e73e3ddeccbc2bc915eec4a4d'
REVISION='c202236235762e1c871ad0ccb60c8ee5ba337b9a'

def validate_config(config):
    if config.get('budget_approved') is not True or config.get('gpu_confirmed') is not True:
        raise ValueError('Reviewed GPU step/time budget required; preflight only')
    if config.get('mode')!='learning' or config.get('init_mode')!='warm_start':raise ValueError('Explicit learning warm_start required')
    if config.get('parent_sha256')!=PARENT or config.get('revision')!=REVISION:raise ValueError('Use the verified original 9B parent, never the synthetic smoke candidate')
    if type(config.get('max_steps'))!=int or not 1<=config['max_steps']<=4:raise ValueError('Pilot step limit is 1..4')
    if type(config.get('max_seconds')) not in (int,float) or not 0<config['max_seconds']<=900:raise ValueError('Pilot time limit is 900 seconds')
    if not math.isfinite(config.get('lr',math.nan)) or not 0<config['lr']<=5e-6:raise ValueError('Pilot learning rate limit')

def validate_rows(config,rows):
    if not rows or not {'replay','correction'}<={r.get('kind') for r in rows}:raise ValueError('Need verified correction AND replay; do not train on environment failures')
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate target IDs across banks')
    if any(r.get('target_verified') is not True or r.get('split')!='learning' or r.get('business_id') not in config['learning_businesses'] or r.get('parent_sha256')!=PARENT for r in rows):raise ValueError('Unverified target, wrong split or wrong parent')

def run_pilot(config,learner=None):
    validate_config(config)
    if not config.get('verified_dataset_directory') or not config.get('verified_dataset_manifest_sha256'):raise ValueError('Verified correction dataset is not ready')
    from relearning import verify_export
    root=Path(config['verified_dataset_directory'])
    if verify_export(root)!=config['verified_dataset_manifest_sha256']:raise ValueError('Dataset artifact changed')
    rows=json.loads((root/'verified_dataset.json').read_text(encoding='utf-8'))
    validate_rows(config,rows)
    if learner is None:
        from train_warm import train
        learner=train
    # train() performs base/parent hashes, plan validation and protected-output checks.
    return learner(config,rows)
