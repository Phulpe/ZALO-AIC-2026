from pathlib import Path
import sys, shutil, json

root=Path('D:/ZAIC')
project=root/'drive-download-20241211T124801Z-001/ZAIC_Local_GUI/CODEX_PROJECT'
sys.path.insert(0,str(project/'gui'))
from run_sync import import_saved_run, digest

archive=root/'DOWNLOADS_ARCHIVE/KAGGLE_PREFLIGHT_366774242_results.zip'
if not archive.exists():
    shutil.copy2(Path('E:/results.zip'),archive)
receipt={
    'saved_version_confirmed':True,
    'version_url':'https://www.kaggle.com/code/lnghongphc/zaic-9b-training-ready-20261108?scriptVersionId=366774242',
    'remote_manifest_sha256':'271810b616742ff27add152d99cbf9146e54dd722c287d2884c0642abeeb4380',
    'observation':'Version #1 Successful; saved output includes original run metadata and manifest',
    'download_archive_sha256':digest(archive),
    'training_started':False,
}
record=import_saved_run(archive,root/'PRIVATE_DATA/runs',receipt)
print(json.dumps(record,indent=2))
