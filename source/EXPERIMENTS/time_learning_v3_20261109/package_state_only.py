"""Safer preservation payload: full training states, no per-question artifacts."""
from pathlib import Path
import json,sys,zipfile,collections
exp=Path(__file__).parent;sys.path.insert(0,str(exp/'src'))
from relearning import verify_export,sha,atomic
source=Path('D:/ZAIC/PRIVATE_DATA/zaic/result_811/TIME_FOCUSED_V3_20261109/time_pilot/model_runs/20261109T125033322737Z_b786bc50')
manifest=verify_export(source)
allowed={'ARTIFACT_MANIFEST.json','README.md','SUPERVISION_SELECTION.json','adapter_config.json','adapter_model.safetensors','chat_template.jinja','data_manifest.json','environment.json','optimizer.pt','parent.json','rng_state.pth','run_config.json','scaler.pt','scheduler.pt','source_manifest.json','tokenizer.json','tokenizer_config.json','trainer_state.json','training_args.bin','training_result.json'}
files=[p for p in source.rglob('*') if p.is_file()];assert {p.name for p in files}==allowed
# Metadata are inspected programmatically, without exposing dataset contents.
for p in files:
 if p.suffix=='.json' and p.name not in {'tokenizer.json'}:
  data=json.loads(p.read_text(encoding='utf-8'))
  def no_raw(v):
   if isinstance(v,dict):
    assert not set(v)&{'question','question_text','gold','messages','prompt','completion','transactions','prediction_text','target'}
    for item in v.values():no_raw(item)
   elif isinstance(v,list):
    for item in v:no_raw(item)
  no_raw(data)
stage=Path('D:/ZAIC/PRIVATE_DATA/time_v3_state_only_preservation');stage.mkdir(exist_ok=False)
with zipfile.ZipFile(stage/'PRESERVED_STATE.zip','w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
 for p in files:z.write(p,p.relative_to(source).as_posix())
inventory={'scope':'Weights, complete optimizer/scheduler/scaler/RNG/trainer states, tokenizer and aggregate metadata only. No question/gold/SFT rows, predictions, journal or private log.','origin':'https://www.kaggle.com/code/lnghongphc/zaic-time-focused-learning-v3-20261109?scriptVersionId=367303123','origin_download_checksum_verified':True,'training_manifest_sha256':manifest,'files':{p.relative_to(source).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files},'training_completed':True,'wrapper_failed_after_training':True,'pickle_loaded':False,'allowed_basenames':sorted(allowed),'archive_sha256':sha(stage/'PRESERVED_STATE.zip')}
atomic(stage/'STATE_UPLOAD_INVENTORY.json',inventory)
atomic(stage/'dataset-metadata.json',{'title':'ZAIC Time v3 Checkpoint States 20261109','id':'lnghongphc/zaic-time-v3-checkpoint-states-20261109','licenses':[{'name':'other'}]})
print('STATE_ONLY_READY',len(files),manifest,(stage/'PRESERVED_STATE.zip').stat().st_size)
