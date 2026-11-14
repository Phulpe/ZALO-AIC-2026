"""GUI notebook backend. No Kaggle CLI, tokens or Google Drive dependency.

Runs frozen source in a fresh working directory. Baseline is never overwritten.
Only synthetic CPU tests are run during preflight. No public inference here.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys

SFT_SHA = '5057658ac10c19579c675c9f926cc13f1e17641b88eb312318e48be75e6bb170'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def verify_frozen(project):
    project = Path(project)
    manifest = json.loads((project / 'SOURCE_SHA256.json').read_text())
    for name, expected in manifest.items():
        target = (project / name).resolve()
        if not target.is_relative_to(project.resolve()) or sha(target) != expected:
            raise ValueError('Frozen source checksum mismatch')
    return len(manifest)

def synthetic_tests(project):
    """Run in a subprocess with captured output, returning only aggregate counts."""
    program = '''import sys,unittest,json,io
from pathlib import Path
p=Path(sys.argv[1]);sys.path[:0]=[str(p/'app'),str(p/'tools')]
names=['test_core','test_v2_regression','test_audit_output','test_batch_io','test_policy']
suite=unittest.TestLoader().loadTestsFromNames(names)
result=unittest.TextTestRunner(stream=io.StringIO(),verbosity=0).run(suite)
print(json.dumps({'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful()}))
'''
    proc = subprocess.run([sys.executable, '-c', program, str(project)], capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError('Synthetic tests could not start; inspect installed pandas/numpy/openpyxl versions locally.')
    summary = json.loads(proc.stdout.strip().splitlines()[-1])
    if summary['tests'] != 99 or not summary['passed']:
        raise RuntimeError('Synthetic tests failed: ' + json.dumps(summary))
    return summary

def environment_report():
    versions = {}
    for name in ['torch', 'transformers', 'peft', 'accelerate', 'bitsandbytes', 'pandas', 'numpy', 'openpyxl', 'safetensors', 'huggingface-hub']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'python': sys.version.split()[0], 'packages': versions}

def run(config, source_root):
    action = config.get('ACTION', 'preflight')
    if action not in {'preflight', 'smoke', 'train'}:
        raise ValueError('ACTION must be preflight, smoke or train')
    source = Path(source_root) / 'frozen_v21'
    output_root = Path('/kaggle/working/zaic_runs')
    output_root.mkdir(parents=True, exist_ok=True)
    report = {'state': 'preflight_started', 'frozen_python_files': verify_frozen(source),
              'environment': environment_report(), 'synthetic_tests': synthetic_tests(source),
              'kaggle_gpu_execution_verified_by_this_package': False}
    preflight = output_root / 'GUI_PREFLIGHT.json'
    preflight.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if action == 'preflight':
        report['state'] = 'cpu_preflight_passed'
        preflight.write_text(json.dumps(report, indent=2))
        print('CPU preflight passed. No model loaded, no training started.')
        return report
    if not config.get('GPU_RUN_CONFIRMED', False):
        raise ValueError('Review GPU/time and set GPU_RUN_CONFIRMED=True to run smoke/train.')
    import torch
    from transformers import Qwen3_5ForCausalLM
    import peft, bitsandbytes, accelerate
    if not torch.cuda.is_available():
        raise RuntimeError('Enable a CUDA GPU in Kaggle Session options.')
    report['cuda'] = {'torch_cuda': torch.version.cuda, 'gpu_count': torch.cuda.device_count(),
                      'gpu0': torch.cuda.get_device_name(0),
                      'gpu0_total_gib': torch.cuda.get_device_properties(0).total_memory / 2**30}
    # Existing loader uses cuda:0. A second T4 is not pooled by this implementation.
    base = Path(config['BASE_INPUT']).resolve()
    if not str(base).startswith('/kaggle/input/'):
        raise ValueError('BASE_INPUT must point into a GUI-attached /kaggle/input resource.')
    from importlib.machinery import SourceFileLoader
    from importlib.util import spec_from_loader, module_from_spec
    loader = SourceFileLoader('zaic_local_check', str(Path(source_root)/'gui/local_check.pyw'))
    spec = spec_from_loader(loader.name, loader)
    checker = module_from_spec(spec)
    loader.exec_module(checker)
    base_check = checker.verify_base(base)
    if not base_check['structural_check_passed']:
        raise ValueError('Base model incomplete: ' + json.dumps(base_check['errors']))
    report['base_structural_check'] = base_check
    if shutil.disk_usage('/kaggle/working').free < 4 * 2**30:
        raise RuntimeError('Less than 4 GiB free; review checkpoint/output space before training.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    project = output_root / (stamp + '_' + action)
    project.mkdir(exist_ok=False)
    shutil.copytree(source / 'app', project / 'app', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(source / 'model_lock.json', project / 'model_lock.json')
    (project / 'assets').mkdir()
    (project / 'assets/base').symlink_to(base, target_is_directory=True)
    if action == 'train':
        if not config.get('COMPETITION_CLOUD_USE_ALLOWED', False):
            raise ValueError('Confirm competition permission for Kaggle processing before attaching/using SFT.')
        if not config.get('TRAIN_FRESH_ADAPTER', False):
            raise ValueError('This notebook trains a fresh adapter. Set TRAIN_FRESH_ADAPTER=True after reviewing.')
        sft = Path(config['SFT_INPUT']).resolve()
        if not str(sft).startswith('/kaggle/input/'):
            raise ValueError('SFT_INPUT must be an approved GUI-attached /kaggle/input directory.')
        training_sha = sha(sft / 'sft_train.jsonl')
        manifest = json.loads((sft / 'manifest.json').read_text())
        if training_sha != SFT_SHA or manifest['training_sha256'] != training_sha:
            raise ValueError('SFT differs from the frozen baseline. Create/review a separate new-data experiment.')
        (project / 'data').mkdir()
        for name in ['sft_train.jsonl', 'manifest.json']:
            shutil.copy2(sft / name, project / 'data' / name)
        report['sft_sha256'] = training_sha
    env = {**os.environ, 'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1',
           'HF_DATASETS_OFFLINE':'1', 'HF_HUB_DISABLE_TELEMETRY':'1',
           'DO_NOT_TRACK':'1', 'TOKENIZERS_PARALLELISM':'false',
           'PYTORCH_ALLOC_CONF':'expandable_segments:True'}
    report['state'] = action + '_starting'
    report['fresh_run'] = str(project)
    report['baseline_adapter_overwritten'] = False
    (project / 'GUI_RUN_METADATA.json').write_text(json.dumps(report, indent=2))
    stages = ['smoke_model.py'] + (['train.py'] if action == 'train' else [])
    try:
        for script in stages:
            log_path = project / (script.replace('.py', '') + '.log')
            with log_path.open('w', encoding='utf-8') as log:
                process = subprocess.Popen([sys.executable, '-u', str(project/'app'/script), '--project', str(project)],
                    cwd=project/'app', env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in process.stdout:
                    log.write(line); log.flush()
                    print(line, end='')
                code = process.wait()
            if code:
                raise RuntimeError(script + ' failed; download the local log, redact before sharing.')
    finally:
        # Never export input base weights with notebook outputs, even if an
        # output archiver follows symlinks. Hard interruption still needs GUI review.
        (project / 'assets/base').unlink(missing_ok=True)
    report['state'] = action + '_completed'
    report['kaggle_gpu_execution_verified_by_this_package'] = True
    if action == 'train':
        adapter = project/'candidate/final/adapter_model.safetensors'
        report['new_adapter_sha256'] = sha(adapter)
        report['training_result'] = json.loads((project/'training_complete.json').read_text())
    (project/'GUI_RUN_METADATA.json').write_text(json.dumps(report, indent=2))
    print('Complete. Save Version without rerunning, then download outputs in the Kaggle GUI.')
    return report
