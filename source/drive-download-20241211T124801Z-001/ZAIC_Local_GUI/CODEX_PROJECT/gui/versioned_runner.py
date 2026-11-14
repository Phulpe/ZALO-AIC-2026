"""Reviewed working runner. Frozen source stays byte-identical. Preflight by default."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import uuid
from run_sync import seal, digest, write_json
from kaggle_runner import verify_frozen, synthetic_tests, environment_report, SFT_SHA


def run(config, source_root):
    source_root = Path(source_root)
    source = source_root / 'frozen_v21'
    action = config.get('ACTION', 'preflight')
    if action not in {'preflight', 'smoke', 'train'}:
        raise ValueError('Unknown action')
    now = datetime.now(timezone.utc)
    run_id = now.strftime('%Y%m%dT%H%M%S%fZ') + '_' + action + '_' + uuid.uuid4().hex[:8]
    root = Path('/kaggle/working/zaic_runs') / run_id
    root.mkdir(parents=True, exist_ok=False)
    meta = {'schema': 'zaic-run-1', 'run_id': run_id, 'action': action,
            'created_utc': now.isoformat(), 'state': 'started', 'config': config,
            'environment': environment_report(), 'metrics': {},
            'versions': {'source_manifest_sha256': digest(source / 'SOURCE_SHA256.json'),
                         'runner_sha256': digest(Path(__file__)),
                         'base': json.loads((source / 'model_lock.json').read_text()),
                         'data_sha256': None}, 'resume_supported_by_this_runner': False}
    write_json(root / 'RUN_METADATA.json', meta)
    try:
        meta['source_files_verified'] = verify_frozen(source)
        meta['synthetic_tests'] = synthetic_tests(source)
        if action != 'preflight':
            if not config.get('GPU_RUN_CONFIRMED'):
                raise ValueError('GPU smoke requires reviewed configuration')
            if action == 'train' and not (config.get('COMPETITION_CLOUD_USE_ALLOWED') and
                config.get('CLOUD_PERMISSION_EVIDENCE') and config.get('TRAIN_FRESH_ADAPTER')):
                raise ValueError('Training blocked: cloud permission evidence and fresh-run review required')
            import torch
            from transformers import Qwen3_5ForCausalLM
            import peft, bitsandbytes, accelerate
            if not torch.cuda.is_available():
                raise ValueError('CUDA GPU unavailable')
            meta['gpu'] = {'name': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
                           'count': torch.cuda.device_count()}
            base = Path(config['BASE_INPUT']).resolve()
            if not base.is_relative_to(Path('/kaggle/input')):
                raise ValueError('Base must be attached input')
            from importlib.machinery import SourceFileLoader
            loader = SourceFileLoader('local_check', str(source_root / 'gui/local_check.pyw'))
            spec = importlib.util.spec_from_loader(loader.name, loader)
            checker = importlib.util.module_from_spec(spec); loader.exec_module(checker)
            base_check = checker.verify_base(base, hash_shards=True)
            if not base_check['structural_check_passed']:
                raise ValueError('Base incomplete')
            # A local hash alone is insufficient: require a reviewed source manifest.
            expected = config.get('BASE_FILE_HASHES', {})
            required = [p for p in base.rglob('*') if p.is_file()]
            if not expected or any(expected.get(p.relative_to(base).as_posix()) != digest(p) for p in required):
                raise ValueError('Base hash/source manifest not verified')
            meta['versions']['base_file_hashes'] = expected
            if shutil.disk_usage('/kaggle/working').free < 4 * 2**30:
                raise ValueError('Insufficient checkpoint space')
            # Runtime scratch is outside saved notebook outputs. Never copy base/SFT into exports.
            scratch = Path('/kaggle/temp/zaic') / run_id
            scratch.mkdir(parents=True, exist_ok=False)
            shutil.copytree(source / 'app', scratch / 'app', ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copy2(source / 'model_lock.json', scratch / 'model_lock.json')
            (scratch / 'assets').mkdir(); (scratch / 'assets/base').symlink_to(base, target_is_directory=True)
            if action == 'train':
                sft = Path(config['SFT_INPUT']).resolve()
                if not sft.is_relative_to(Path('/kaggle/input')) or digest(sft / 'sft_train.jsonl') != SFT_SHA:
                    raise ValueError('Frozen SFT hash mismatch')
                manifest = json.loads((sft / 'manifest.json').read_text())
                if manifest['training_sha256'] != SFT_SHA:
                    raise ValueError('SFT manifest mismatch')
                (scratch / 'data').symlink_to(sft, target_is_directory=True)
                meta['versions']['data_sha256'] = SFT_SHA
                meta['versions']['data_manifest_sha256'] = digest(sft / 'manifest.json')
                # Save the complete final-step optimizer/scheduler/RNG state as well as adapter.
                train = scratch / 'app/train.py'
                text = train.read_text()
                needle = "trainer.save_model(str(out/'final'))"
                if text.count(needle) != 1:
                    raise ValueError('Training source changed; review final checkpoint patch')
                text = text.replace(needle, "trainer._save_checkpoint(trainer.model, trial=None);trainer.save_state();" + needle)
                train.write_text(text)
                meta['versions']['working_train_sha256'] = digest(train)
            env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                   'HF_DATASETS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1', 'DO_NOT_TRACK': '1',
                   'TOKENIZERS_PARALLELISM': 'false', 'PYTORCH_ALLOC_CONF': 'expandable_segments:True'}
            for script in ['smoke_model.py'] + (['train.py'] if action == 'train' else []):
                # Capture locally; never put possible dataset-dependent tracebacks in notebook output.
                with (scratch / (script + '.log')).open('w', encoding='utf-8') as log:
                    result = subprocess.run([sys.executable, str(scratch / 'app' / script), '--project', str(scratch)],
                        cwd=scratch / 'app', env=env, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError('GPU stage failed; inspect/redact runtime log locally')
            meta['smoke'] = json.loads((scratch / 'smoke_model.json').read_text())
            shutil.copy2(scratch / 'smoke_model.json', root / 'smoke_model.json')
            if action == 'train':
                shutil.copytree(scratch / 'candidate', root / 'candidate')
                for name in ['training_complete.json', 'training_progress.json']:
                    shutil.copy2(scratch / name, root / name)
                meta['metrics'] = json.loads((scratch / 'training_complete.json').read_text())
                meta['adapter_sha256'] = digest(root / 'candidate/final/adapter_model.safetensors')
                checkpoints = sorted((root / 'candidate/checkpoints').glob('checkpoint-*'),
                                     key=lambda p: int(p.name.split('-')[-1]))
                if not checkpoints:
                    raise ValueError('No resumable checkpoint exported')
                latest = checkpoints[-1]
                for name in ['optimizer.pt', 'scheduler.pt', 'trainer_state.json', 'rng_state.pth',
                             'adapter_model.safetensors', 'adapter_config.json']:
                    if not (latest / name).is_file():
                        raise ValueError('Incomplete optimizer checkpoint')
                if int(latest.name.split('-')[-1]) != meta['metrics']['steps']:
                    raise ValueError('Final checkpoint step mismatch')
                meta['checkpoint'] = {'path': latest.relative_to(root).as_posix(), 'full_state_verified': True}
        meta['state'] = 'completed'
        manifest_sha = seal(root, meta)
        archive = shutil.make_archive(str(root), 'zip', root_dir=root)
        print(json.dumps({'run_id': run_id, 'state': 'completed', 'action': action,
            'source_files_verified': meta['source_files_verified'], 'synthetic_tests': meta['synthetic_tests'],
            'artifact_manifest_sha256': manifest_sha, 'archive': archive}, indent=2))
        return meta
    except Exception as exc:
        meta['state'] = 'failed'; meta['error_type'] = type(exc).__name__
        write_json(root / 'RUN_METADATA.json', meta)
        print(json.dumps({'run_id': run_id, 'state': 'failed', 'error_type': type(exc).__name__}))
        raise RuntimeError('Run failed. Only aggregate diagnostics may be shared.') from None
