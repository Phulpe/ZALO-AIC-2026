"""Local CPU readiness audit. Never prints question, answer or SFT content."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib,json,shutil,uuid

ROOT=Path('D:/ZAIC');EXP=ROOT/'EXPERIMENTS/relearning_v1'
STAGE=ROOT/'KAGGLE_STAGING/20261107T191003Z'
now=datetime.now(timezone(timedelta(hours=7)))
OUT=ROOT/'EXPERIMENTS'/('relearning_preparation_'+now.strftime('%Y%m%dT%H%M%S'))
OUT.mkdir(exist_ok=False)

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()

def write(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

permission={'user_authorized':True,'user_attestation':'Ban tổ chức cho phép các thí sinh thực hiện huấn luyện model trên bất kỳ nền tảng thứ 3 nào','source':'User statement in current chat','observed_at':now.isoformat(),'competition_permission_evidence':{'status':'unverified','independent_source':None},'account_authorization_reconfirmation_required':False}
write('PERMISSION_EVIDENCE.json',permission)
parent=ROOT/'PRIVATE_DATA/zaic/datamate_v1/qwen35_9b/candidate/final'
parent_sha=sha(parent/'adapter_model.safetensors')
assert parent_sha=='e7dfa635beccce128c99f9d3fd993d33dbb4c88e73e3ddeccbc2bc915eec4a4d'
source=json.loads((EXP/'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
for name,digest in source['files'].items():
    assert sha(EXP/name)==digest,'Packaged source changed'
assert sha(EXP/'RELEARNING_CODE_ONLY.zip')==source['zip_sha256']
tests=json.loads((EXP/'TEST_RESULTS_KAGGLE_V2.json').read_text())
assert tests['baseline']['passed'] and tests['baseline']['tests']==99
assert tests['new']['passed'] and tests['new']['tests']==40
manifest=json.loads((STAGE/'01_BASE_QWEN35_9B/BASE_UPLOAD_MANIFEST.json').read_text())
base=STAGE/'01_BASE_QWEN35_9B/base'
checks=[]
for name,item in manifest['files'].items():
    p=base/name
    ok=p.is_file() and p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
    checks.append({'name':name,'bytes':item['bytes'],'sha256':item['sha256'],'passed':ok})
    print('BASE_FILE_CHECKED',len(checks),len(manifest['files']),flush=True)
assert all(c['passed'] for c in checks)
index=json.loads((base/'model.safetensors.index.json').read_text())
assert len(set(index['weight_map'].values()))==4
write('INPUT_CHECKS.json',{'observed_at':now.isoformat(),'scope':'local sources and parent for next warm-start; no remote runtime checksum claim','base':checks,'parent':{'path':str(parent),'bytes':(parent/'adapter_model.safetensors').stat().st_size,'sha256':parent_sha},'source_zip_sha256':source['zip_sha256'],'source_files_verified':len(source['files']),'historical_kaggle_cpu_tests':tests})
common={'proposal_id':now.strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8],'mode':'learning','init_mode':'warm_start','parent_sha256':parent_sha,'parent_local_path':str(parent),'revision':manifest['revision'],'base_hashes':{n:i['sha256'] for n,i in manifest['files'].items()},'user_authorized':True,'competition_permission_evidence':permission['competition_permission_evidence'],'gpu_confirmed':False,'lr':5e-6,'seed':20261108,'max_length':2048,'batch_size':1,'source_zip_sha256':source['zip_sha256'],'stop_on':['nonfinite_loss','nonfinite_gradient','OOM','input_or_split_mismatch','checkpoint_write_failure','budget_expired'],'preserve_parent_and_checkpoint48':True,'long_training_enabled':False}
write('SMOKE_CONFIG_REVIEW.json',{**common,'phase':'gpu_smoke_only','targets':'smoke_targets.targets() - verified synthetic fixture only','max_steps':1,'approved_max_steps':1,'gradient_accumulation':1,'max_seconds':180,'approved_max_seconds':180,'proposed_total_gpu_session_seconds':900,'post_checks':['adapter_hash_changed','all_adapter_tensors_finite','checkpoint_adapter_reload','loss_mask_shift','checkpoint_manifest_verified'],'gpu_session_shutdown':'Agent must stop session and verify Accelerator=None after test; no unattended shutdown guarantee'})
write('PILOT_CONFIG_REVIEW.json',{**common,'phase':'small_correction_replay_pilot','max_steps':4,'approved_max_steps':None,'gradient_accumulation':16,'proposed_max_gpu_hours':0.25,'max_seconds':600,'approved_max_seconds':None,'dataset_path':None,'data_manifest_sha256':None,'split_manifest_sha256':None,'calibration_manifest_sha256':None,'minimum_improvement':0,'max_regressions':0,'attempts_per_question':3,'training_authorized':False,'promotion_requires':'locked comparable calibration and regression evidence; no promotion on loss alone'})
shutil.copy2(EXP/'ZAIC_RELEARNING_V1.ipynb',OUT/'ZAIC_RELEARNING_PREFLIGHT.ipynb')
shutil.copy2(EXP/'RELEARNING_CODE_ONLY.zip',OUT/'RELEARNING_CODE_ONLY.zip')
blocks=['independent_competition_permission_evidence_unverified','warm_start_9b_gpu_smoke_and_checkpoint_reload_not_run','real_verified_correction_replay_dataset_and_frozen_evaluation_not_prepared','pilot_budget_not_yet_accepted']
write('READINESS.json',{'observed_at':now.isoformat(),'ready_for_model_training':False,'ready_for_cpu_preparation':True,'gpu_must_remain_off':True,'passed':['source_package_hash','parent_adapter_hash','all_15_local_base_files_hash','four_shards_resolve','99_baseline_and_40_new_tests_previous_Kaggle_CPU_evidence'],'blockers':blocks,'scope_limit':'No GPU test or fresh remote file rehash performed in this preparation','output_dir':str(OUT)})
report='''# Chuẩn bị lượt warm-start — chưa đủ điều kiện train

Đã đối chiếu lại nguồn gói code, adapter cha 9B và đủ 15 file base bằng SHA256 trên máy. Bốn shard được index tham chiếu đầy đủ. Báo cáo 99 baseline + 40 test mới trên Kaggle CPU đã tồn tại và khớp gói code; lần này không gọi đó là kiểm thử GPU mới.

Notebook preflight và ZIP chỉ có mã đã được sao chép vào thư mục chuẩn bị này. Không sửa notebook/adapter/checkpoint gốc. Hai cấu hình đã chuẩn bị để xem xét: smoke 1 optimizer step trên fixture; pilot warm-start tối đa 4 steps, LR 5e-6, batch 1/accum 16, đề xuất tối đa 0,25 giờ GPU. Đây không phải lệnh train hoặc cam kết dùng hết ngân sách.

## Điều kiện còn thiếu

1. Bằng chứng thể lệ độc lập: lời xác nhận mới của người dùng đã lưu nguyên văn, quyền tài khoản đã có; chưa đổi thành verified. Không hỏi lại cùng quyền tài khoản. Theo mục 11 AGENTS.md, vẫn chưa mở xử lý tài sản thi trên Kaggle khi thiếu nguồn này.
2. Smoke warm-start thật: finite loss/gradient, completion-mask/shift, adapter thay đổi hữu hạn, checkpoint reload và checksum. Code v1 hiện đã có train/save; các phép xác minh sau train này chưa có bằng chứng thực thi GPU. Không coi unit test mock là đã load 9B.
3. Correction/replay thật cùng split/evaluation đã freeze: hiện mới có demo giả lập. Bộ SFT baseline 768 không tự trở thành correction/replay mới. Bộ sửa tự động hiện chỉ hỗ trợ nhóm câu tổng tháng; câu ngoài phạm vi vẫn phải ở ledger.
4. Chốt phần ngân sách pilot còn thiếu trước train. Không cần hỏi lại init_mode: đã chọn warm_start, optimizer/scheduler mới; checkpoint-48 chỉ bảo toàn/đối chứng.

## Thứ tự tiếp tục

- Xác minh nguồn cho phép → giữ GPU off để kiểm source/input/split → kiểm môi trường GPU và chạy smoke 1 step, có giới hạn, chỉ khi đủ gates.
- Lưu output bền vững rồi tắt session/Accelerator=None ngay; sau đó mới tải về máy và đối chiếu manifest. Không chờ tải file trong khi GPU chạy rỗi.
- Khi smoke đạt, tạo learning pool/ledger và correction/replay được xác minh trong môi trường được phép, freeze evaluation, hoàn thiện config pilot và kiểm input trước run.
- Chỉ báo sẵn sàng cho pilot khi READINESS đã có bằng chứng đạt các điều kiện tương ứng; không tự bật train dài, không tự promote BEST_MODEL.

Hai notebook được quan sát session off trong lượt chuẩn bị này. Không bật GPU hoặc chạy train. Bằng chứng session: D:\\ZAIC\\EXPERIMENTS\\relearning_v1\\READINESS_SESSION_OFF.jpg.
'''
(OUT/'READINESS_VI.md').write_text(report,encoding='utf-8')
(ROOT/'RELEARNING_READINESS_LATEST.json').write_text(json.dumps({'directory':str(OUT),'ready_for_model_training':False},indent=2),encoding='utf-8')
print('PREPARATION_COMPLETE',str(OUT),flush=True)
