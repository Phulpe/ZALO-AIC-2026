"""Update only checkpoint/sync pointers after successful CPU saved-version proof."""
from pathlib import Path
import json,sys,shutil,datetime
exp=Path(__file__).parent;sys.path.insert(0,str(exp/'src'))
from relearning import verify_export,sha,track,atomic
dest=Path('D:/ZAIC/PRIVATE_DATA/zaic/result_811/TIME_FOCUSED_V3_20261109');v=json.loads((dest/'LOCAL_VERIFICATION.json').read_text());run=Path(v['run_path']).resolve();assert verify_export(run)==v['training_manifest_sha256'];receipt=json.loads((dest/'PRESERVATION_RECEIPT.json').read_text());assert receipt['state']=='completed' and receipt['training_manifest_sha256']==v['training_manifest_sha256'] and receipt['checkpoint_sets']==10 and not receipt['gpu_executed']
remote=dest/'preserved_training'/run.name/'ARTIFACT_MANIFEST.json';assert sha(remote)==v['training_manifest_sha256']
tracker=dest.parent/'RUN_STATE.json';before=json.loads((exp/'RUN_STATE_BEFORE.json').read_text());assert json.loads(tracker.read_text())==before
shutil.copy2(tracker,dest/'RUN_STATE_BEFORE.json');track(tracker,run,'checkpoint',{});url='https://www.kaggle.com/code/lnghongphc/zaic-time-v3-checkpoint-preservation-20261109?scriptVersionId=367316919';track(tracker,run,'sync',{'saved_version_completed':True,'saved_version_url':url,'remote_manifest_sha256':v['training_manifest_sha256']});after=json.loads(tracker.read_text());assert after['BEST_MODEL']==before['BEST_MODEL'];atomic(dest/'RUN_STATE_AFTER_PRESERVATION.json',after)
atomic(dest/'PRESERVATION_SYNC_EVIDENCE.json',{'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'preservation_status':'COMPLETE','preservation_url':url,'script_version_id':367316919,'training_manifest_sha256':v['training_manifest_sha256'],'verified_bytes':receipt['verified_bytes'],'checkpoint_sets':10,'BEST_MODEL_unchanged':True,'evaluation_still_pending':True})
atomic(dest/'WEIGHTS_INVENTORY_RUN.json',{'run_id':run.name,'checkpoints':v['checkpoints'],'manifest_sha256':v['training_manifest_sha256'],'pickle_deserialized':False})
print('LATEST_CHECKPOINT_AND_LATEST_SYNCED_VERIFIED_BEST_UNCHANGED')
