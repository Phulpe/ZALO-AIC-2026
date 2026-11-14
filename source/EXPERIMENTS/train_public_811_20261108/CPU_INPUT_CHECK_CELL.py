from pathlib import Path
import json,hashlib,importlib.metadata
from permission import permitted
from relearning import sha,verify_parent
assert permitted(CONFIG['competition_permission_evidence'])
bases=list(Path('/kaggle/input').rglob('BASE_UPLOAD_MANIFEST.json'))
adapters=list(Path('/kaggle/input').rglob('final/adapter_model.safetensors'))
assert len(bases)==len(adapters)==1
base=bases[0].parent/'base'
for name,digest in CONFIG['base_hashes'].items():
    assert sha(base/name)==digest,'Base hash mismatch'
assert len(CONFIG['base_hashes'])==15
verify_parent(adapters[0].parent,CONFIG['parent_sha256'])
packages={}
for name in ['torch','transformers','peft','accelerate','bitsandbytes','safetensors']:
    try: packages[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: packages[name]=None
receipt={'base_files_sha256_matched':15,'parent_adapter_sha256_matched':True,'permission_status':CONFIG['competition_permission_evidence']['status'],'independently_verified':False,'packages':packages,'gpu_used':False}
Path('/kaggle/working/CPU_INPUT_RECEIPT_V2.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt,indent=2))
