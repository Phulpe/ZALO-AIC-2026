"""Evidence-backed audit. Reads artifacts without executing notebooks or pickle."""
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import ast, csv, hashlib, io, json, math, re

ROOT=Path('D:/ZAIC');PRIVATE=ROOT/'PRIVATE_DATA';ZAIC=PRIVATE/'zaic';DM=ZAIC/'datamate_v1';Q=DM/'qwen35_9b'
PROJECT=ROOT/'drive-download-20241211T124801Z-001/ZAIC_Local_GUI/CODEX_PROJECT'
E=ROOT/'TRAINING_AUDIT_EVIDENCE'
inv=json.loads((ROOT/'WEIGHTS_INVENTORY_AFTER_RENAME.json').read_text(encoding='utf-8'))
drive=json.loads((E/'DRIVE_METADATA.json').read_text(encoding='utf-8'))
probe=json.loads((E/'LOCAL_TRAINING_PROBE.json').read_text(encoding='utf-8'))
extra=json.loads((E/'EXTRA_PROBE.json').read_text(encoding='utf-8'))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def clean(x):
 if isinstance(x,float) and not math.isfinite(x):return None
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,list):return [clean(v) for v in x]
 return x
def metric(d):
 keys={'n','correct','wrong','abstain','PA','answered_accuracy','coverage','epoch','train_loss','train_runtime',
       'training_steps_this_run','all_exact','json_object','schema_ok','intent','answer_type','time','k'}
 if not isinstance(d,dict):return {}
 out={k:clean(v) for k,v in d.items() if k in keys and isinstance(v,(int,float))}
 for k in ['counts','accuracy','baseline','candidate','calibration','audit','metrics','evaluation']:
  if isinstance(d.get(k),dict):out[k]=metric(d[k])
 return out

# Recompute hashes for model-learning artifacts and full checkpoint states in this audit.
# Base multi-GB hashes are referenced to the prior complete post-rename verification,
# not silently represented as newly rehashed by this script.
fresh={};artifact_paths=set()
for a in inv['adapters']:
 folder=Path(a['path'])
 for p in folder.iterdir():
  if p.is_file():artifact_paths.add(p)
for p in sorted(artifact_paths):
 before=p.stat();h=sha(p);after=p.stat()
 fresh[str(p)]={'bytes':after.st_size,'sha256':h,'stable_during_hash':before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns}
print('ARTIFACT_HASHES_COMPLETE',len(fresh),flush=True)

def artifact(folder):
 folder=Path(folder);w=folder/'adapter_model.safetensors';cfg=folder/'adapter_config.json'
 if not w.is_file():return {'path':str(folder),'weight_found':False}
 state=read(folder/'trainer_state.json') if (folder/'trainer_state.json').is_file() else {}
 config=read(cfg) if cfg.is_file() else {}
 needed=['optimizer.pt','scheduler.pt','scaler.pt','rng_state.pth','trainer_state.json']
 result={'path':str(folder),'weight_found':True,'weight':fresh[str(w)],'adapter_config_present':cfg.is_file(),
  'config_base':config.get('base_model_name_or_path'),'config_revision':config.get('revision'),
  'rank':config.get('r'),'step':state.get('global_step'),
  'missing_resume_files':[n for n in needed if not (folder/n).is_file() or (folder/n).stat().st_size==0],
  'state_files':{n:fresh.get(str(folder/n)) for n in needed},'pickle_or_model_loaded':False}
 previous=next((r for r in inv['adapters'] if r['path']==str(folder)),{})
 result['matches_prior_inventory']=result['weight']['sha256']==previous.get('adapter_sha256')
 result['source_integrity']=previous.get('source_integrity','unknown')
 return result

remote=[];folders={}
for node in drive['folders']:
 folders[node['folder_path']]=node
 for f in node['files']:
  remote.append({**f,'relative_path':node['folder_path']+'/'+f['title']})

def remote_matches(local_folder):
 prefix=Path(local_folder).relative_to(PRIVATE).as_posix()
 rows=[]
 for f in remote:
  if f.get('file_or_folder')=='folder' or not (f['relative_path']==prefix or f['relative_path'].startswith(prefix+'/')):continue
  local=PRIVATE/f['relative_path']
  if re.search(r'adapter_model\.safetensors$|adapter_config\.json$|optimizer\.pt$|scheduler\.pt$|scaler\.pt$|rng_state\.pth$|trainer_state\.json$',f['relative_path']):
   rows.append({'local_path':str(local),'drive_url':f['url'],'drive_id':f['id'],'remote_bytes':int(f['size']) if f.get('size') else None,
     'local_exists':local.is_file(),'size_matches':local.is_file() and f.get('size') is not None and local.stat().st_size==int(f['size']),
     'checksum_status':'remote checksum unavailable in connector; size-only comparison',
     'created_utc':f.get('created_time'),'modified_utc':f.get('modified_time')})
 return rows

def evidence_files(paths):
 result=[]
 for p in paths:
  p=Path(p)
  if p.is_file():result.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
 return result
def dataset(paths):
 result=[]
 for p in paths:
  p=Path(p)
  if p.is_file():
   record={'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
   if p.suffix=='.jsonl':
    # No row content is returned to the AI or written to this audit.
    with p.open(encoding='utf-8-sig') as f:record['rows']=sum(bool(l.strip()) for l in f)
   result.append(record)
 return result

nb=ROOT/'ZAIC_QLoRA_intent_v0.ipynb'
R=[]
def run(run_id,path,base,revision,start,initialization,confidence,status,checkpoints,code,data,logs,evals,lesson):
 path=Path(path)
 finals=[path/'final'] if (path/'final').is_dir() else [path]
 adapters=[artifact(p) for p in finals]+[artifact(path/'checkpoints'/('checkpoint-'+str(s))) for s in checkpoints]
 adapters=[a for a in adapters if a.get('weight_found') or a['path']==str(finals[0])]
 evaluations=[]
 for p in evals:
  p=Path(p)
  if p.is_file():
   try:evaluations.append({'path':str(p),'sha256':sha(p),'aggregate':metric(read(p))})
   except Exception:evaluations.append({'path':str(p),'status':'aggregate parser could not read'})
 source=evidence_files(code)
 row={'run_id':run_id,'kind':'training_run_or_attempt','local_root':str(path),'base_model':base,
   'base_revision':revision,'time_and_basis':start,'initialization':initialization,'lineage_confidence':confidence,
   'status':status,'artifacts':adapters,'code_and_config':source,'data':dataset(data),'logs':evidence_files(logs),
   'evaluations':evaluations,'lesson':lesson,'remote':remote_matches(path),'kaggle_copy':'No saved weight artifact observed in lnghongphc account',
   'environment':{},'reproduction_status':'partial_not_reproduced','inference_status':'weights_available_base/runtime_checks_still_required' if any(a.get('weight_found') for a in adapters) else 'chưa tìm thấy trọng số',
   'resume_status':'checkpoint state file set present; execution unverified' if any(a.get('weight_found') and a.get('step') is not None and not a['missing_resume_files'] for a in adapters) else 'no full resumable checkpoint found'}
 R.append(row);return row

b4='Qwen/Qwen3-4B-Instruct-2507';b9='Qwen/Qwen3.5-9B';rev9='c202236235762e1c871ad0ccb60c8ee5ba337b9a'
rev4='Run metadata/adapter config does not pin revision; local base_model_lock: cdbee75f17c01a7cc42f958dc650907174af0554 (not proof of historical load)'
sft4=[ZAIC/'sft_data/sft_train.jsonl',ZAIC/'sft_data/sft_val.jsonl']
old20=run('4B_20steps_20261106_043640',ZAIC/'adapter_intent_v0_20steps_20261106_043640',b4,rev4,
 '20-step output in notebook cell 8; backup folder timestamp 20261106_043640 has no timezone; Drive folder creation UTC available separately',
 'Base model -> fresh LoRA according to notebook initialization; cell 8 trainer.train(); cell 20 saves trainer.model to 20-step backup. Notebook execution_count is null throughout: historical ordering remains imperfect.',
 'medium: notebook + saved adapter; not a full ordered execution trace','20-step output observed; adapter preserved',[],[nb],sft4,[],[],
 'Benchmark training (save_strategy=no); saved adapter backup exists but no optimizer/scaler/RNG checkpoint for the exact 20-step state.')
run1=run('4B_run1_20261106_044117',ZAIC/'adapter_intent_v0_run1_20261106_044117',b4,rev4,
 'Folder stamp 20261106_044117 has no timezone; evaluation summary proves 150 training_steps_this_run; checkpoint states 125/150',
 'Notebook cell 22 creates Trainer(model=trainer.model, args=replace(trainer.args,...)) and calls trainer_full.train() without resume_from_checkpoint. In-memory weights continue from prior trainer; likely 20-step weights, but null execution counters prevent a perfect proof. This is not optimizer resume.',
 'medium-high: explicit in-memory model reuse + 150-step artifacts; exact pre-run bytes unavailable','completed: 150 steps this run', [125,150],[nb],sft4,[],[ZAIC/'adapter_intent_v0_run1_20261106_044117/evaluation/summary.json',ZAIC/'adapter_intent_v0_run1_20261106_044117/checkpoints/train_results.json'],
 '20 + 150 is a potential cumulative weight-update count, not trainer_state.global_step=170; the new Trainer resets its own step/optimizer. Exact cumulative lineage not independently snapshotted.')
cand=run('4B_candidate_20261106_090435',DM/'candidate_20261106_090435',b4,rev4,
 'Folder stamp 20261106_090435 has no timezone; checkpoint states 80/83; Drive final modification provides a storage timestamp',
 'run_metadata.json explicitly previous_adapter=adapter_intent_v0_run1_20261106_044117/final. train_candidate.py loads PeftModel.from_pretrained(base, resume_checkpoint or baseline_adapter). Initial parent strongly evidenced; automatic checkpoint-resume use during this run not proven by a launch receipt.',
 'high for run1 weight parent; actual resume checkpoint unknown','completed: 83 steps', [80,83],[DM/'train_candidate.py',DM/'candidate_20261106_090435/run_metadata.json'],[DM/'training_v1/sft_train_v1.jsonl',DM/'training_v1/manifest.json'],[],[DM/'candidate_20261106_090435/checkpoints/train_results.json',DM/'model_selection.json'],
 'Existing 4B adapter is fine-tuned on a new 2642-row supervision dataset; candidate is not a fresh 9B model. Selection/evaluation must be read separately from train loss.')
initial=run('9B_initial_workflow_attempt',Q/'candidate',b9,rev9,
 'WORKFLOW.log Drive created 2026-11-06T14:17:36.414Z; exact launch time not recorded',
 'TRAIN_9B_CONFIG old_adapter_reused=false + source loads base and creates LoRA. No TRAIN_PROGRESS marker in WORKFLOW.log; do not infer successful optimizer updates.',
 'medium for initialization; unknown number of weight updates','failed workflow, no separately identifiable weight artifact',[],[Q/'app/train.py'],[Q/'data/sft_train.jsonl',Q/'data/manifest.json'],[Q/'WORKFLOW.log'],[],
 'RuntimeError in initial workflow. Later candidate/final is from stable run and must not be assigned to this failed attempt merely because the working path was reused.')
# The reused path currently contains the later stable adapter. It is NOT this attempt's artifact.
initial['artifacts']=[];initial['remote']=[];initial['inference_status']='chưa tìm thấy trọng số riêng cho lượt lỗi';initial['resume_status']='chưa tìm thấy checkpoint riêng';initial['path_reused_by']='9B_stable_48'
failed=run('9B_scale65536_failed_20261106_144528',Q/'candidate_scale65536_failed_20261106_144528',b9,rev9,
 'Archive suffix 20261106_144528 reflects archive/rename time; workflow_resume.log Drive created 2026-11-06T14:36:11.029Z',
 'Metadata old_adapter_reused=false; workflow_resume.log reports step 1..4 and nonfinite gradients. Filename resume is not evidence of optimizer checkpoint resume.',
 'high that 4-step attempt ran; parent checkpoint not documented','failed: step 4 of 48; nonfinite gradients',[],[Q/'candidate_scale65536_failed_20261106_144528/metadata.json'],[Q/'data/sft_train.jsonl',Q/'data/manifest.json'],[Q/'workflow_resume.log',Q/'progress_scale65536_failed_20261106_144528.json'],[],
 'Initial FP16 loss scale 65536 produced NaN grad_norm; archived metadata/logs remain. No adapter/checkpoint found locally or in listed Drive failure folder. Save interval 5 and stopping at 4 may explain no scheduled checkpoint; this is not proof of lost data.')
failed['inference_status']='chưa tìm thấy trọng số riêng';failed['resume_status']='no checkpoint found (Drive checkpoints listing empty)'
stable=run('9B_stable_48',Q/'candidate',b9,rev9,
 'Drive candidate folder created 2026-11-06T14:47:06.137Z; final adapter modified 2026-11-06T15:48:16.086Z; exact training start not asserted',
 'training_complete.json, metadata.json and TRAIN_9B_CONFIG: old_adapter_reused=false. Base Qwen3.5-9B + fresh rank-8 LoRA; no inherited 4B adapter. Source can auto-resume checkpoints but actual launch receipt specifying resume is absent.',
 'high for fresh 9B adapter / no 4B weight reuse; checkpoint resume within run unproven','completed: 48 steps', [40,45,48],[Q/'app/train.py',Q/'app/modeling.py',Q/'app/build_data.py',Q/'candidate/metadata.json',Q/'model_lock.json'],[Q/'data/sft_train.jsonl',Q/'data/manifest.json',Q/'data/split.json'],[Q/'TRAIN_9B.log',Q/'workflow_stable.log',Q/'training_complete.json'],[Q/'candidate/checkpoints/train_results.json',Q/'calibration_candidate/evaluation.json',Q/'audit_selected/evaluation.json'],
 'Stable configuration records initial_loss_scale=16 and finite gradients; completed 48 steps. Runtime v2/v2.1 later changed ingestion/contracts/execution, not these adapter weights.')
stable['environment']=read(Q/'versions.json')
stable['base_revision_evidence']='training_complete.json model lock + model_lock.json + base assets manifest; not guessed from folder name'
for row in [initial,failed]:row['environment']={'status':'later versions.json snapshot exists; exact failed-attempt library environment unproven'}
for row in [old20,run1,cand]:row['environment']={'status':'notebook output lists transformers 5.22.0 / peft 0.21.2 / bnb 0.50.2; execution counters null and historical attribution uncertain; no run-specific complete pip lock found'}

# Explicit backups: compare payload hashes, do not count copies as additional training.
aliases=[]
alias_targets=[(Q/'training_code/checkpoints/checkpoint-48',Q/'candidate/checkpoints/checkpoint-48','9B_stable_48'),
 (Q/'training_code/teacher_4b_adapter/final',DM/'candidate_20261106_090435/final','4B_candidate_20261106_090435'),
 (DM/'offline_export/assets/adapter',DM/'candidate_20261106_090435/final','4B_candidate_20261106_090435')]
for backup,original,rid in alias_targets:
 a=artifact(backup);b=artifact(original)
 same=a.get('weight',{}).get('sha256')==b.get('weight',{}).get('sha256')
 matching_states={n:fresh.get(str(backup/n),{}).get('sha256')==fresh.get(str(original/n),{}).get('sha256') for n in ['optimizer.pt','scheduler.pt','scaler.pt','rng_state.pth','trainer_state.json'] if (backup/n).is_file() and (original/n).is_file()}
 aliases.append({'run_id':rid,'backup':str(backup),'original':str(original),'adapter_sha256_equal':same,'state_hash_equal':matching_states,'artifact':a,'remote':remote_matches(backup)})

# Metadata/log evidence beyond training: retain lessons without inventing trained models.
nontraining=[
 {'event_id':'9B_smoke_initial_OOM','kind':'synthetic_model_smoke','evidence':evidence_files([Q/'MODEL_SMOKE.log']),'status':'OutOfMemoryError recorded; no saved adapter expected','lesson':'Avoid blanket fp32 upcast of huge frozen embedding/head; see repaired smoke code/history.'},
 {'event_id':'9B_smoke_repaired_passed','kind':'synthetic_model_smoke','evidence':evidence_files([Q/'MODEL_SMOKE_REPAIRED.log',Q/'smoke_model.json']),'status':'synthetic loss/backward smoke passed in historical runtime; adapter_saved=false','lesson':'Model smoke success does not establish new model training or Kaggle compatibility.'},
 {'event_id':'runtime_v2_v21','kind':'runtime_repair_evaluation','evidence':evidence_files([PRIVATE/'ZAIC_9B_Repair_v21_20261107/candidate_calibration/evaluation.json',PRIVATE/'ZAIC_9B_Repair_v21_20261107/candidate_confirmation2/evaluation.json']),'status':'same 9B weights; code repair/evaluation, not a new training run','lesson':'Calibration200 PA .35; confirmation2 PA .30. Opened confirmation2 cannot be called untouched holdout.'},
 {'event_id':'Kaggle_preparation','kind':'preflight_or_draft','status':'3 notebooks listed as drafts; no dataset/model listed; no saved training-weight artifact found','evidence':['KAGGLE_OBSERVATION.json'],'lesson':'Historical notebook outputs do not prove a currently running or saved Kaggle training run.'}
]

lineage=[
 {'parent':'Qwen3-4B','child':old20['run_id'],'relation':'fresh LoRA','confidence':'medium; notebook initialization not immutable execution trace'},
 {'parent':old20['run_id'],'child':run1['run_id'],'relation':'same in-memory trainer.model; new Trainer, not full optimizer resume','confidence':'medium-high; notebook cell 22, null execution counters'},
 {'parent':run1['run_id'],'child':cand['run_id'],'relation':'adapter weight initialization','confidence':'high; previous_adapter metadata + PeftModel loader'},
 {'parent':'Qwen3.5-9B','child':initial['run_id'],'relation':'fresh adapter attempt','confidence':'medium'},
 {'parent':'Qwen3.5-9B','child':failed['run_id'],'relation':'fresh adapter attempt; checkpoint parent not documented','confidence':'medium-high'},
 {'parent':'Qwen3.5-9B','child':stable['run_id'],'relation':'fresh rank-8 adapter, not 4B adapter inheritance','confidence':'high'},
 {'parent':stable['run_id'],'child':'runtime_v2_v21','relation':'same adapter; runtime repair only','confidence':'high; frozen hash / handoff'}]

remote_size_checks=[f for row in R for f in row['remote']]+[f for a in aliases for f in a['remote']]
remote_diffs=[f for f in remote_size_checks if not f['size_matches']]
local_evidence=[]
for p in PRIVATE.rglob('*'):
 if p.is_file() and p.suffix in ['.log','.py','.md'] and not any(x in p.parts for x in ['train','public_test','data','reference']):
  local_evidence.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})

graph='''flowchart LR
 B4["Qwen3-4B base"] -->|"fresh LoRA: notebook evidence"| S20["20-step adapter"]
 S20 -.->|"in-memory model reuse; medium-high confidence"| R150["run1: 150 new steps"]
 R150 -->|"previous_adapter metadata + loader"| C83["candidate: 83 steps"]
 B9["Qwen3.5-9B base"] --> A0["initial workflow failure: updates unknown"]
 B9 --> F4["scale65536 failure: step 4"]
 B9 -->|"fresh rank-8 adapter"| S48["stable 9B: 48 steps"]
 S48 -->|"same weights, runtime changes only"| V21["repair v2 / v2.1"]
 C83 -->|"equal SHA256 copy"| CP4["offline_export / teacher_4b_adapter"]
 S48 -->|"checkpoint-48 state hashes equal"| CP9["training_code checkpoint backup"]
'''

audit={'schema':'zaic-training-history-audit-1','checked_utc':datetime.now(timezone.utc).isoformat(),
 'scope':{'local':str(ROOT),'drive_folder_snapshots':len(drive['folders']),'drive_search_extra_results':'empty results; not proof no other historical folders',
 'kaggle_account':'lnghongphc','limits':['No complete organizer/team source inventory of every past run','Drive resume_own and known verified folder listing empty through this connector; live remote contents unverified','Remote Drive hashes not exposed; remote weight copies size-compared only','No notebook or pickle executed; no runtime resume/reproduction performed']},
 'runs':R,'backup_aliases':aliases,'lineage':lineage,'nontraining_events':nontraining,
 'current_artifact_hashes':fresh,'code_log_note_inventory':local_evidence,'remote_size_comparisons':remote_size_checks,
 'remote_size_mismatches':remote_diffs,'legacy_output_slot':{'path':str(ZAIC/'adapter_intent_v0'),'files':sum(1 for p in (ZAIC/'adapter_intent_v0').rglob('*') if p.is_file()),
 'status':'empty local and Drive listing; notebook names this as original output slot. No evidence it is a distinct additional preserved training run; do not infer loss or absorption.'},
 'base_prior_verification':{'evidence':str(ROOT/'WEIGHTS_INVENTORY_AFTER_RENAME.json'),'note':'Prior full file SHA256 plus post-rename full SHA256 comparison to manifests. Not recomputed for large base shards in this audit. 4B total_size metadata inconsistency persists.'},
 'conclusions':{
 'use_learned_results':'Four successful adapter stages are locally preserved; 9B final matches the known reference hash; actual inference in target runtime not run.',
 'continue_training':'Seven canonical checkpoints plus one checkpoint-48 backup have full nonempty state files and verified local hashes; runtime optimizer resume and exact parent lineage not execution-tested. 20-step exact state lacks full checkpoint.',
 'reproduce':'Substantial code/data hashes/config/log evidence retained, strongest for stable 9B; exact reproduction not certified for any run. Old 4B revision and run-specific environment unpinned; old failed code states not fully archived.',
 'missing_or_unverified':'Distinct weights for initial failed workflow and step-4 nonfinite attempt not found; source-wide full history unknown; live Drive checksum equality unavailable; no saved Kaggle model observed.'}}
dump(E/'TRAINING_HISTORY_DETAILS.json',clean(audit))
dump(E/'CHECKSUM_MANIFEST.json',{'checked_utc':audit['checked_utc'],'files':fresh,'note':'Full bytes SHA256, no pickle deserialization'})
(E/'TRAINING_LINEAGE.mmd').write_text(graph,encoding='utf-8')

fields=['run_id','kind','time_and_basis','base_model','base_revision','initialization','lineage_confidence','status','final_adapter','all_checkpoints','resume_state',
 'code_configuration','data_manifest_hash','library_environment','logs','metrics_evaluation','lessons','local_copies','drive_copies','drive_checksum_status','kaggle_copy','use_results','continue_training','reproduce','missing_unverified']
csvrows=[]
for r in R:
 final=next((a for a in r['artifacts'] if a.get('step') is None),{})
 ck=[a for a in r['artifacts'] if a.get('step') is not None]
 backups=[a for a in aliases if a['run_id']==r['run_id']]
 csvrows.append(dict(zip(fields,[r['run_id'],r['kind'],r['time_and_basis'],r['base_model'],r['base_revision'],r['initialization'],r['lineage_confidence'],r['status'],final,ck,r['resume_status'],r['code_and_config'],r['data'],r['environment'],r['logs'],r['evaluations'],r['lesson'],[r['local_root']]+[a['backup'] for a in backups],r['remote'],'size-only; no live remote checksum exposed',r['kaggle_copy'],r['inference_status'],r['resume_status'],r['reproduction_status'],'See per-run details and audit limits'])))
buf=io.StringIO(newline='');writer=csv.DictWriter(buf,fieldnames=fields);writer.writeheader()
for row in csvrows:writer.writerow({k:json.dumps(clean(v),ensure_ascii=False,allow_nan=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})

lines=['# Kiểm toán lịch sử huấn luyện ZAIC — 08/11/2026','',
 'Đã kiểm tra cục bộ D:\\ZAIC; đọc metadata 54 thư mục Drive; kiểm tra tài khoản Kaggle lnghongphc bằng giao diện. Không chạy notebook, train, upload, hợp nhất adapter hoặc sửa artifact gốc.',
 '', '## Kết luận theo bốn mục','',
 '1. **Sử dụng thành quả đã học:** tìm thấy 4 giai đoạn adapter thành công: bản 20 steps, run1 150 steps, candidate 4B 83 steps và stable 9B 48 steps. Hash trọng số/tệp checkpoint được tính lại trong lần kiểm toán này. Adapter 9B cuối vẫn khớp hash chuẩn. Chưa thực thi suy luận để chứng nhận runtime.',
 '2. **Tiếp tục huấn luyện:** 7 checkpoint chính (9B 40/45/48, run1 125/150, candidate 80/83) và bản sao 9B checkpoint-48 có đủ optimizer/scheduler/scaler/RNG/trainer state không rỗng. Đây là đủ bộ tệp để chuẩn bị resume; chưa nạp pickle hoặc chạy resume. Bản 20 steps chỉ có adapter, không đủ khôi phục chính xác optimizer/RNG của bước đó.',
 '3. **Tái lập:** chưa chứng nhận tái lập hoàn toàn bất kỳ run nào. Stable 9B có hồ sơ đầy nhất; run 4B chưa pin revision trong metadata và không có environment lock riêng đáng tin theo từng run; notebook đã sửa qua nhiều lượt, execution_count đều null. Mã hiện tại không chứng minh mọi phiên bản mã lỗi trước đó còn đủ.',
 '4. **Thiếu/chưa xác minh:** chưa tìm thấy trọng số riêng của workflow lỗi ban đầu và lượt nonfinite step 4. Không kết luận đã mất hay được model cuối kế thừa. Chưa có danh mục nguồn toàn bộ các run ngoài phạm vi; checksum Drive live chưa kiểm được; chưa thấy bản trọng số đã lưu trên Kaggle.',
 '', '## Bảng tổng quan run','',
 '| Run | Bắt đầu từ đâu và độ chắc chắn | Artifact giữ được | Tiếp tục train | Tái lập |','|---|---|---|---|---|']
for r in R:
 finals=[a for a in r['artifacts'] if a.get('weight_found') and a.get('step') is None]
 steps=[str(a['step']) for a in r['artifacts'] if a.get('weight_found') and a.get('step') is not None]
 lines.append(f"| {r['run_id']} | {r['initialization']} ({r['lineage_confidence']}) | {'adapter cuối' if finals else 'chưa tìm thấy adapter riêng'}; checkpoints {', '.join(steps) or 'không'} | {r['resume_status']} | chưa chứng nhận |")
lines+=['','## Quan hệ kế thừa có bằng chứng','', '```mermaid',graph.rstrip(),'```','',
 'Mũi tên đứt từ 20 steps đến run1 thể hiện bất định của nhật ký thực thi: source cell 22 dùng trainer.model đang ở RAM, nhưng notebook không còn execution_count để chứng minh hoàn hảo model nào đang trong biến lúc đó. Không có cạnh kế thừa trọng số từ 4B sang 9B. Bản teacher_4b_adapter là bản sao được so hash, không phải bằng chứng merge adapter.',
 '', 'Không gọi run1 là checkpoint-170: 150 là bước của Trainer mới; khả năng đã kế thừa 20 cập nhật trước đó là quan hệ trọng số, không phải resume optimizer/global_step.',
 '', '## Hồ sơ chi tiết từng run','']
for r in R:
 lines += ['### '+r['run_id'],'', '- Thời điểm và căn cứ: '+r['time_and_basis'],
 '- Base: '+r['base_model']+'; revision: '+r['base_revision'],
 '- Khởi tạo: '+r['initialization'], '- Trạng thái: '+r['status'], '- Độ chắc chắn: '+r['lineage_confidence'],
 '- Môi trường: '+json.dumps(r['environment'],ensure_ascii=False), '- Bài học: '+r['lesson'],
 '', '| Artifact | Bytes | SHA256 | Step | Thiếu bộ resume |','|---|---:|---|---:|---|']
 for a in r['artifacts']:
  lines.append(f"| {a['path']} | {a.get('weight',{}).get('bytes','—')} | {a.get('weight',{}).get('sha256','chưa tìm thấy')} | {a.get('step','—')} | {', '.join(a.get('missing_resume_files',[])) or 'không thiếu tệp'} |")
 lines += ['', '- Mã/cấu hình (hash chi tiết trong CSV/JSON): '+('; '.join(f['path'] for f in r['code_and_config']) or 'chưa tìm thấy'),
 '- Dữ liệu/manifest (chỉ hash/count, không nội dung): '+json.dumps(r['data'],ensure_ascii=False),
 '- Log: '+('; '.join(f['path'] for f in r['logs']) or 'không có log riêng; đối chiếu notebook và trainer_state'),
 '- Đánh giá tổng hợp: '+json.dumps(r['evaluations'],ensure_ascii=False),
 '- Bản Drive: '+('; '.join(sorted(set(x['drive_url'] for x in r['remote'] if x['local_path'].endswith('adapter_model.safetensors')))) or 'chưa đối chứng trọng số riêng'),
 '- Kaggle: '+r['kaggle_copy'],'']
lines += ['## Bản sao và checkpoint đầy đủ','', '| Bản sao | Run nguồn | Adapter hash giống | Trạng thái checkpoint so hash |','|---|---|---|---|']
for a in aliases:lines.append(f"| {a['backup']} | {a['run_id']} | {a['adapter_sha256_equal']} | {a['state_hash_equal']} |")
lines += ['', 'Mọi bản trùng được giữ nguyên. Hash giống chỉ chứng minh artifact giống bytes tại thời điểm kiểm tra; không chứng minh có một lượt train mới hay một bản sao lưu độc lập trên ổ vật lý khác.',
 '', '## Log lỗi và các sự kiện không phải huấn luyện','']
for x in nontraining:lines += ['- **'+x['event_id']+'**: '+x['status']+' — '+x['lesson']]
lines += ['', 'Danh sách log/mã/ghi chú và hash để truy lại nằm trong TRAINING_AUDIT_EVIDENCE/TRAINING_HISTORY_DETAILS.json. Không loại các log lỗi. Các lần public inference, journal replay, repair v2/v2.1 và preflight được phân biệt với lượt cập nhật trọng số.',
 '', '## Đối chiếu máy – Drive – Kaggle','',
 f"- Đã so {len(remote_size_checks)} tệp trọng số/config/state theo đường dẫn và kích thước Drive; {len(remote_diffs)} mục không khớp kích thước hoặc chưa có bản máy trong tập này. Danh sách từng file có URL trong JSON chi tiết.",
 '- Connector không trả md5Checksum/sha256Checksum dù yêu cầu fields. Vì vậy các bản Drive này chỉ được đánh dấu đã thấy/khớp size, không đánh dấu đã đồng bộ checksum. Hash máy khớp manifest bàn giao là bằng chứng riêng, không phải kiểm mới hash live Drive.',
 '- List nguồn resume_own và verified bằng connector trả rỗng; không suy ra Drive gốc đã trống hoặc dữ liệu mất. Bộ verified vẫn có bản cục bộ đã được kiểm ở lần trước.',
 '- Kaggle Your Work/Code hiện có 3 mục draft. Notebook preflight được xem trực tiếp: 0 versions, session off, No input attached. Your Datasets và Your Models: No results found. Không thấy bản trọng số đã lưu trên tài khoản đang truy cập; không kết luận về tài khoản khác hoặc vùng draft disk chưa đọc.',
 '- Tìm thư mục Drive bằng search ZAIC/adapter/candidate không có kết quả bổ sung; direct folder listing hoạt động. Search rỗng không phải chứng minh không còn các run ngoài danh sách.',
 '', '## Cần bổ sung để đóng kiểm toán','',
 '1. Biên bản nguồn toàn bộ run/phiên Colab và mọi thư mục Drive ngoài zaic, Repair_v21, Resume_Own; xác định tài khoản sở hữu các thư mục connector chưa đọc được.',
 '2. Hash Drive live cho adapter/checkpoint/state (hoặc tải một bản đối chiếu riêng có kiểm hash); chưa cần thay thế/xóa bất kỳ bản hiện có.',
 '3. Mã/cấu hình đúng thời điểm chạy từng thất bại; transcript có thứ tự thực thi đáng tin hoặc launch receipt chỉ rõ checkpoint resume. Không dùng tên workflow_resume.log làm bằng chứng đã resume.',
 '4. Base revision thực sự được nạp ở 4B và environment lock riêng từng run. Lock hiện có là ứng viên tái lập; không tự coi là lịch sử đã xác minh.',
 '5. Kiểm thử inference/resume/reproduction riêng sau khi được giao; lần kiểm toán này chỉ kiểm artifact và metadata.',
 '6. Giải thích bất nhất metadata total_size của base 4B (lệch 655.360 bytes) trước chứng nhận runtime; toàn bộ shard đã khớp manifest nguồn và không bị sửa.',
 '', 'RUN_REGISTRY.csv chứa từng run với mã/cấu hình, dataset hash, môi trường, metrics, bản sao và trạng thái checksum. Checkpoint state hash đầy đủ nằm trong TRAINING_AUDIT_EVIDENCE/CHECKSUM_MANIFEST.json. Không có raw question/answer/SFT trong các báo cáo này.']
text='\n'.join(lines)+'\n'
for folder in [ROOT,PROJECT]:
 for name,content in [('TRAINING_HISTORY_AUDIT.md',text),('RUN_REGISTRY.csv',buf.getvalue())]:
  target=folder/name
  # Preserve any earlier audit under a new timestamp; never overwrite a previous report.
  if target.exists():target=folder/(target.stem+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+target.suffix)
  with target.open('x',encoding='utf-8-sig' if target.suffix=='.csv' else 'utf-8',newline='') as f:f.write(content)
print(json.dumps({'runs':len(R),'successful_weight_stages':4,'failed_attempts_without_separate_weight':2,
 'adapter_files_rehashed':sum(p.endswith('adapter_model.safetensors') for p in fresh),'state_and_support_files_rehashed':len(fresh),
 'drive_size_comparisons':len(remote_size_checks),'drive_size_mismatches':len(remote_diffs),'alias_checks':aliases,
 'reports_written':True},ensure_ascii=False,indent=2),flush=True)
