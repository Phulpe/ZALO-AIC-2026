"""Train once from an immutable verified bank; no recollection or promotion.

The CLI defaults to CPU validation. GPU execution needs explicit authorization
and a separately persisted successful smoke receipt for these exact inputs.
"""
from pathlib import Path
import json, math, sys
from relearning import verify_export, verify_parent, sha, atomic
from quality_gate import select


def validate_policy(c, execute=False):
    if c.get('mode') != 'learning' or c.get('init_mode') != 'warm_start':
        raise ValueError('Explicit warm-start learning required')
    if c.get('optimizer_init') != 'new' or c.get('scheduler_init') != 'new':
        raise ValueError('New optimizer and scheduler required')
    if c.get('collect_learning_probe') is not False or c.get('run_public') is not False:
        raise ValueError('No recollection or public test in this pilot')
    if c.get('max_steps') != 18 or c.get('gradient_accumulation') != 4 or c.get('max_epochs_equivalent') != 4:
        raise ValueError('Reviewed 18-step policy required')
    for key in ('max_seconds', 'child_timeout_seconds'):
        value = c.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('Finite time budget required')
    if not c['max_seconds'] < c['child_timeout_seconds'] <= c['max_seconds'] + 300:
        raise ValueError('Bounded child lifetime required')
    if execute and any(c.get(k) is not True for k in ('user_authorized', 'budget_approved', 'gpu_confirmed')):
        raise ValueError('New GPU authorization required')


def prepare(c):
    validate_policy(c)
    bank = Path(c['frozen_regression_root'])
    if verify_export(bank) != c['frozen_regression_manifest_sha256']:
        raise ValueError('Frozen bank changed')
    rows = json.loads((bank/'PRIVATE_BANK.json').read_text(encoding='utf-8'))
    evidence = json.loads((bank/'PRIVATE_EVIDENCE.json').read_text(encoding='utf-8'))
    selected, report = select(rows, evidence, c['supervision_gate'])
    if report['dataset_sha256'] != c['selected_dataset_sha256'] or len(selected) != 18:
        raise ValueError('Selected dataset changed')
    if sum(r['kind'] == 'correction' for r in selected) != 9:
        raise ValueError('Correction count changed')
    verify_parent(c['adapter'], c['parent_sha256'])
    base = Path(c['base'])
    if set(c['base_hashes']) != {p.name for p in base.iterdir() if p.is_file()}:
        raise ValueError('Incomplete base inventory')
    if any(sha(base/name) != digest for name, digest in c['base_hashes'].items()):
        raise ValueError('Base hash mismatch')
    output = Path(c['output_root']).resolve()
    for protected in (bank.resolve(), base.resolve(), Path(c['adapter']).resolve(), Path(__file__).parent.resolve()):
        if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError('Output overlaps preserved input')
    return rows, evidence, report


def execute(c, trainer=None):
    validate_policy(c, execute=True)
    rows, evidence, selection = prepare(c)
    smoke = json.loads(Path(c['smoke_receipt']).read_text(encoding='utf-8'))
    expected = {k: c[k] for k in ('parent_sha256', 'selected_dataset_sha256', 'source_archive_sha256')}
    if smoke.get('state') != 'passed' or smoke.get('fingerprints') != expected or smoke.get('GPU_executed') is not True:
        raise ValueError('Matching real GPU smoke required')
    if trainer is None:
        from isolated_training import train_verified_isolated
        trainer = train_verified_isolated
    run = Path(trainer(c, rows, evidence))
    atomic(run.parent/'FROZEN_PILOT_RECEIPT.json', {
        'state': 'trained_pending_regression', 'run_path': str(run),
        'manifest_sha256': verify_export(run), 'selection': selection,
        'promoted': False, 'public_ran': False,
        'required_next': 'old7 + correction9 + full103 replay, then paired calibration',
    })
    return run


if __name__ == '__main__':
    config = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    if len(sys.argv) == 3 and sys.argv[2] == '--execute':
        execute(config)
    elif len(sys.argv) == 2:
        _, _, report = prepare(config)
        print(json.dumps({'state': 'CPU_validated', 'selection': report, 'GPU_executed': False}))
    else:
        raise ValueError('Usage: frozen_pilot.py CONFIG [--execute]')
