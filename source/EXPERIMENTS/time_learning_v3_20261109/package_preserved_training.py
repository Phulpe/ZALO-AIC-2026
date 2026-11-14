"""Create immutable full-state preservation input, not a model merge."""
from pathlib import Path
import json,sys,zipfile
exp=Path(__file__).parent;sys.path.insert(0,str(exp/'src'))
from relearning import verify_export,sha,atomic
source=Path('D:/ZAIC/PRIVATE_DATA/zaic/result_811/TIME_FOCUSED_V3_20261109/time_pilot')
manifest=verify_export(source)
stage=Path('D:/ZAIC/PRIVATE_DATA/time_v3_complete_preservation');stage.mkdir(exist_ok=False)
with zipfile.ZipFile(stage/'PRESERVED_TRAINING.zip','w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
 for p in source.rglob('*'):
  if p.is_file():z.write(p,p.relative_to(source).as_posix())
atomic(stage/'PRESERVATION_UPLOAD.json',{'outer_manifest_sha256':manifest,'archive_sha256':sha(stage/'PRESERVED_TRAINING.zip'),'bytes':(stage/'PRESERVED_TRAINING.zip').stat().st_size,'source_version':3,'script_version_id':367303123,'training_completed':True,'wrapper_failed_after_training':True,'all_checkpoints_preserved':True})
atomic(stage/'dataset-metadata.json',{'title':'ZAIC Time v3 Complete Run Preservation 20261109','id':'lnghongphc/zaic-time-v3-complete-preservation-20261109','licenses':[{'name':'other'}]})
print('PRESERVATION_READY',manifest,(stage/'PRESERVED_TRAINING.zip').stat().st_size)
