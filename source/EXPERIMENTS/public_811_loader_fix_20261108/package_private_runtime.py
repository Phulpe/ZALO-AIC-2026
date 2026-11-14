"""Private TRAIN runtime snapshot. Never emit raw records into tool output."""
from pathlib import Path
import json,shutil,zipfile
from relearning import sha,atomic
here=Path(__file__).parent
prep=json.loads((here/'PRIVATE_POOL_PREPARATION.json').read_text())
root=Path(prep['path']);source=Path('D:/ZAIC/PRIVATE_DATA/zaic/train')
split=json.loads((root/'SPLIT_MANIFEST.json').read_text())
target=root/'data';target.mkdir(exist_ok=False)
for business in split['groups']['learning']+split['groups']['calibration']:
    folder=source/'data'/business
    assert folder.resolve().is_relative_to((source/'data').resolve())
    shutil.copytree(folder,target/business)
files={}
for p in sorted(root.rglob('*')):
    if not p.is_file() or p.name=='PRIVATE_PREPARE.log' or 'audit_exposed' in p.relative_to(root).parts:continue
    assert not p.is_symlink()
    files[p.relative_to(root).as_posix()]={'bytes':p.stat().st_size,'sha256':sha(p)}
runtime={'scope':'TRAIN learning/calibration only; no public/private test; audit-exposed data excluded from cloud bundle','user_permission_status':'user_attested','independently_verified':False,'files':files,'learning_pool_count':prep['learning_pool'],'calibration_count':prep['calibration'],'pilot_supported_count':prep['pilot_supported_questions'],'dataset_manifest_sha256':prep['data_manifest_sha256'],'split_manifest_sha256':prep['split_manifest_sha256'],'corrected_targets_verified':False}
atomic(root/'RUNTIME_DATA_MANIFEST.json',runtime)
archive=root.parent/(root.name+'_TRAIN_RUNTIME.zip')
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
    for name in files:z.write(root/name,'TRAIN_RUNTIME/'+name)
    z.write(root/'RUNTIME_DATA_MANIFEST.json','TRAIN_RUNTIME/RUNTIME_DATA_MANIFEST.json')
summary={'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'runtime_manifest_sha256':sha(root/'RUNTIME_DATA_MANIFEST.json'),'files':len(files)+1,'learning':prep['learning_pool'],'calibration':prep['calibration'],'audit_gold_excluded_from_upload':True,'correction_replay_dataset_ready':False}
atomic(here/'PRIVATE_RUNTIME_PACKAGE.json',summary)
print(json.dumps(summary,indent=2))
