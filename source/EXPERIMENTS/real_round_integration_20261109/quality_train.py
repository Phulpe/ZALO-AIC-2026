"""Only explicit verified correction/replay can reach the optimizer."""
from pathlib import Path
import json
from quality_gate import select
from relearning import atomic

def train_verified(config,rows,evidence):
    selected,report=select(rows,evidence,config['supervision_gate'])
    if config['parent_sha256']!=config['supervision_gate']['parent_sha256']:raise ValueError('Parent drift')
    if config['init_mode']!='warm_start' or config['mode']!='learning':raise ValueError('Explicit warm-start learning required')
    if config['max_steps']>config['max_epochs_equivalent']*len(selected)/config['gradient_accumulation']:raise ValueError('Oversampling budget exceeded')
    from train_warm import train
    run=train(config,selected)
    # A new evidence artifact requires resealing the new experiment run.
    atomic(Path(run)/'SUPERVISION_SELECTION.json',report)
    from relearning import seal
    seal(run)
    from relearning import track
    track(Path(config['output_root'])/'STATE.json',run,'checkpoint',{})
    return run
