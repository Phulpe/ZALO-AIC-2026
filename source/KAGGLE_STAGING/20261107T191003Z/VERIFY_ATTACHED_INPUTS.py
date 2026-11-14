"""Run on Kaggle CPU after attaching the four reviewed inputs.

Only aggregate integrity results are printed. No pickle, model loading or GPU.
"""
from pathlib import Path
import hashlib
import json

ROOT = Path('/kaggle/input')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def verify_manifest(name, child):
    matches = list(ROOT.rglob(name))
    assert len(matches) == 1, 'Expected exactly one reviewed input version'
    manifest_path = matches[0]
    files = json.loads(manifest_path.read_text())['files']
    directory = manifest_path.parent / child
    for relative, record in files.items():
        path = (directory / relative).resolve()
        assert path.is_relative_to(directory.resolve())
        assert path.is_file() and path.stat().st_size == record['bytes']
        assert sha(path) == record['sha256'], 'Input checksum mismatch'
    return directory, len(files)

base, base_count = verify_manifest('BASE_UPLOAD_MANIFEST.json', 'base')
sft, sft_count = verify_manifest('SFT_UPLOAD_MANIFEST.json', 'data')
index = json.loads((base / 'model.safetensors.index.json').read_text())
shards = set(index['weight_map'].values())
assert len(shards) == 4 and all((base / name).is_file() for name in shards)
adapters = list(ROOT.rglob('final/adapter_model.safetensors'))
assert len(adapters) == 1, 'Expected one trained 9B input'
adapter = adapters[0]
assert adapter.stat().st_size == 86623432
assert sha(adapter) == 'e7dfa635beccce128c99f9d3fd993d33dbb4c88e73e3ddeccbc2bc915eec4a4d'
config = json.loads((adapter.parent / 'adapter_config.json').read_text())
assert config['base_model_name_or_path'] == '/content/drive/MyDrive/zaic/datamate_v1/qwen35_9b/assets/base'
assert config['r'] == 8
EXPECTED_TRAINED = {'final/adapter_config.json': {'bytes': 1302, 'sha256': '81a8f3b6bb0fea3694645faecbd639b546d852178d9b5392bf5fb9cc08380abc'}, 'final/adapter_model.safetensors': {'bytes': 86623432, 'sha256': 'e7dfa635beccce128c99f9d3fd993d33dbb4c88e73e3ddeccbc2bc915eec4a4d'}, 'final/chat_template.jinja': {'bytes': 7756, 'sha256': 'a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715'}, 'final/README.md': {'bytes': 5276, 'sha256': '40cf28822b7f9e697cd8b498aa396a84c6146e6d7f4a894937286413e4232ba5'}, 'final/tokenizer.json': {'bytes': 19989339, 'sha256': 'a5cd9732badce41de57e6efce8302930ded1c1188c5f81feb2bd6c24c4a1941f'}, 'final/tokenizer_config.json': {'bytes': 1464, 'sha256': '6fc30221b0ec773811dfa2a04ebf0eec1a7e36a3bb1698888e28b38eabc1126c'}, 'final/training_args.bin': {'bytes': 5265, 'sha256': 'ec304bce42943a583fdf95233a19e69e9c5e0fef14e0d40c05930fabc800447d'}, 'checkpoints/checkpoint-48/adapter_config.json': {'bytes': 1302, 'sha256': '81a8f3b6bb0fea3694645faecbd639b546d852178d9b5392bf5fb9cc08380abc'}, 'checkpoints/checkpoint-48/adapter_model.safetensors': {'bytes': 86623432, 'sha256': 'e7dfa635beccce128c99f9d3fd993d33dbb4c88e73e3ddeccbc2bc915eec4a4d'}, 'checkpoints/checkpoint-48/chat_template.jinja': {'bytes': 7756, 'sha256': 'a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715'}, 'checkpoints/checkpoint-48/optimizer.pt': {'bytes': 44586181, 'sha256': '2cd52041406af338fb8c0331e9facda39d1efb093f068331f326611af0e6e03b'}, 'checkpoints/checkpoint-48/README.md': {'bytes': 5276, 'sha256': '40cf28822b7f9e697cd8b498aa396a84c6146e6d7f4a894937286413e4232ba5'}, 'checkpoints/checkpoint-48/rng_state.pth': {'bytes': 14645, 'sha256': '3f4a6e311653bed0ebcbed5bc7b638bd0e12c196803494f9098d770fb2fd7bce'}, 'checkpoints/checkpoint-48/scaler.pt': {'bytes': 1383, 'sha256': 'ef4a091e18e7e3127686face18428e35b3700c95f401ba63fb35b5aa6ba828ca'}, 'checkpoints/checkpoint-48/scheduler.pt': {'bytes': 1465, 'sha256': '8f8c15637bdf0a16f4962ec44b7231f03100e3eae1261ae580198c04915f691c'}, 'checkpoints/checkpoint-48/tokenizer.json': {'bytes': 19989339, 'sha256': 'a5cd9732badce41de57e6efce8302930ded1c1188c5f81feb2bd6c24c4a1941f'}, 'checkpoints/checkpoint-48/tokenizer_config.json': {'bytes': 1464, 'sha256': '6fc30221b0ec773811dfa2a04ebf0eec1a7e36a3bb1698888e28b38eabc1126c'}, 'checkpoints/checkpoint-48/trainer_state.json': {'bytes': 9479, 'sha256': '2b23b0e119734a1eec535a70ddee2b2b5a047a45a2ebdbf91b5a8066f235549f'}, 'checkpoints/checkpoint-48/training_args.bin': {'bytes': 5265, 'sha256': 'ec304bce42943a583fdf95233a19e69e9c5e0fef14e0d40c05930fabc800447d'}, 'model_lock.json': {'bytes': 184, 'sha256': '6c7616ebe581cbf90a3b0d78f95631f4c907c1e65953f7c796375271447b2a21'}, 'training_complete.json': {'bytes': 835, 'sha256': 'f2f598be99e53dd9084b9916c5f91e99bfbe150342c17d82a5b31ba390fb38b2'}, 'versions.json': {'bytes': 229, 'sha256': '3ed74b4e44117a3c573f18dd8123dcc66b86b45251203eb6779bbd51fdec5eff'}}
for relative, record in EXPECTED_TRAINED.items():
    target = adapter.parent.parent / relative
    assert target.is_file() and target.stat().st_size == record['bytes']
    assert sha(target) == record['sha256'], 'Trained input checksum mismatch'
lock = json.loads((adapter.parent.parent / 'model_lock.json').read_text())
assert lock['model'] == 'Qwen/Qwen3.5-9B'
assert lock['revision'] == 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'
checkpoint = adapter.parent.parent / 'checkpoints/checkpoint-48'
required = ['adapter_model.safetensors', 'adapter_config.json', 'optimizer.pt',
            'scheduler.pt', 'scaler.pt', 'rng_state.pth', 'trainer_state.json']
assert all((checkpoint / name).is_file() and (checkpoint / name).stat().st_size > 0
           for name in required)
print(json.dumps({'input_integrity': 'passed', 'base_files': base_count,
                  'base_shards': len(shards), 'sft_files': sft_count,
                  'final_9b_hash': 'matched', 'checkpoint_48_required_files': 'present',
                  'checkpoint_state_hashes': 'matched',
                  'training_started': False}, indent=2))
