"""Publish aggregate CPU readiness evidence. No training/model pointer changes."""
from pathlib import Path
import sys,json,hashlib,zipfile,datetime,shutil,importlib.metadata
from zoneinfo import ZoneInfo
exp=Path(__file__).parent;root=exp.parent.parent;sys.path.insert(0,str(exp/'src'))
from relearning import sha,verify_export,atomic
tests=json.loads((exp/'KAGGLE_CPU_RESULT_V5/TEST_RESULTS.json').read_text(encoding='utf-8'));archive=exp/'CODE_V4_R5.zip';assert sha(archive)==tests['source_archive_sha256'];assert sum(v['tests'] for v in tests['results'].values())==282 and all(v['passed'] and v['skipped']==0 for v in tests['results'].values());assert not tests['gpu_executed'] and not tests['competition_data_mounted']
with zipfile.ZipFile(archive) as z:
 for name in z.namelist():assert hashlib.sha256(z.read(name)).hexdigest()==sha(exp/'src'/name)
source_receipt=json.loads((exp/'SOURCE_COPY_RECEIPT.json').read_text(encoding='utf-8'))
baseline={n:h for n,h in source_receipt['files'].items() if n.startswith('baseline/')};assert all(sha(exp/'src'/n)==h for n,h in baseline.items())
data=json.loads((exp/'FROZEN_TEMPORAL_DATASET_SUMMARY.json').read_text(encoding='utf-8'));bank=Path(data['private_dataset']);assert verify_export(bank)==data['dataset_manifest_sha256'];assert len(data['selection']['replay_time_buckets'])==5 and data['old7_included']==7 and data['all_replay_regression_cases']==103
real=json.loads((exp/'REVALIDATION_FINAL_R3_AGGREGATE.json').read_text(encoding='utf-8'));assert real['before_correct']==real['after_correct']==103 and real['correct_to_wrong']==0 and real['executor_exceptions_resolved']==3
old=root/'EXPERIMENTS/time_learning_v3_20261109';snapshot=json.loads((old/'PRE_RUN_SNAPSHOT.json').read_text(encoding='utf-8'))
for item in snapshot['protected_files']:assert sha(item['path'])==item['sha256']
model_root=root/'PRIVATE_DATA/zaic/result_811';state=json.loads((model_root/'RUN_STATE.json').read_text(encoding='utf-8'));assert state==json.loads((model_root/'TIME_FOCUSED_V3_20261109/RUN_STATE_AFTER_PRESERVATION.json').read_text(encoding='utf-8'));assert verify_export(state['LATEST_SYNCED']['run'])==state['LATEST_SYNCED']['manifest_sha256']
now=datetime.datetime.now(datetime.timezone.utc);observed=now.astimezone(ZoneInfo('Asia/Saigon')).isoformat()
atomic(exp/'ORIGINALS_PRESERVED.json',{'observed_at':observed,'protected_files':snapshot['protected_files'],'D31_run_manifest_sha256':state['LATEST_SYNCED']['manifest_sha256'],'model_tracker_unchanged':True,'training_registry_sha256':sha(root/'RUN_REGISTRY.csv')})
manifest=json.loads((bank/'ARTIFACT_MANIFEST.json').read_text(encoding='utf-8'));atomic(exp/'DATASET_INVENTORY.json',{'dataset':str(bank),'manifest_sha256':data['dataset_manifest_sha256'],'parent_adapter_sha256':'e7dfa635beccce128c99f9d3fd993d33dbb4c88e73e3ddeccbc2bc915eec4a4d','selected_roles':{'correction':9,'replay':9},'full_replay_regression':103,'files':manifest['files'],'no_new_labels_created':True,'copied_or_merged_models':False})
controllers=exp/'FROZEN_CONTROLLERS';controllers.mkdir(exist_ok=False)
for name in ['revalidate_private.py','freeze_temporal_replay.py','diagnose_private.py','source_diagnosis.py','audit_old_seven.py','audit_source_identity.py','audit_raw_projection.py','audit_source_rules.py']:shutil.copy2(exp/name,controllers/name)
atomic(controllers/'SOURCE_MANIFEST.json',{p.name:sha(p) for p in controllers.iterdir() if p.is_file()})
versions={p:importlib.metadata.version(p) for p in ['pandas','numpy','openpyxl','python-dateutil','pytz','tzdata','et-xmlfile']};atomic(exp/'ENVIRONMENT_CPU.json',{'python':sys.version,'platform':sys.platform,'packages':versions,'Kaggle_CPU_environment':tests['environment'],'scope':'real source replay local Windows CPU; all synthetic tests Kaggle Linux CPU; not new GPU parity'})
(exp/'requirements_cpu.lock').write_text(''.join(f'{name}=={version}\n' for name,version in sorted(versions.items())),encoding='utf-8')
proposal={'status':'proposal_only_not_authorized_to_execute','observed_at':observed,'run_id':None,'mode':'learning','init_mode':'warm_start','parent_adapter_sha256':'e7dfa635beccce128c99f9d3fd993d33dbb4c88e73e3ddeccbc2bc915eec4a4d','parent_checkpoint_resume':None,'base_model':'Qwen/Qwen3.5-9B','base_revision':'c202236235762e1c871ad0ccb60c8ee5ba337b9a','optimizer_init':'new','scheduler_init':'new','source_archive_sha256':sha(archive),'source_archive_path':str(archive),'frozen_dataset_root':str(bank),'frozen_dataset_manifest_sha256':data['dataset_manifest_sha256'],'selected_dataset_sha256':data['selection']['dataset_sha256'],'selection_policy':'time_business_metric_v1','protected_replay_count':103,'critical_old_corrections':7,'proposed_max_steps':18,'proposed_max_epochs_equivalent':4,'proposed_gradient_accumulation':4,'proposed_lr':1e-5,'proposed_time_token_weight':4.0,'proposal_rationale':'Prior9steps~2epochs still0/7 old time errors; propose controlled4epochs with replay5time buckets and all103 replay regression. This is a hypothesis, not demonstrated optimal.','budget_approved':False,'gpu_confirmed':False,'proposed_gpu_minutes_cap':120,'proposed_optimizer_rounds':1,'current_next_round_gpu_authorization':False,'default_accelerator':'None','default_action':'preflight','public_test_enabled':False,'promotion_requirements':{'old7_correction_gain_positive':True,'all9_correction_gain_positive':True,'max_correct_to_wrong_on103_replay':0,'paired_frozen_calibration_required_before_BEST':True,'minimum_calibration_metric_gain':.01,'max_calibration_correct_to_wrong':0,'calibration_is_untouched_holdout':False},'required_before_start':['Resolve real Kaggle input paths and verify source/base/adapter/data/split hashes onCPU','Approve budget for a new optimizer round; previous one-round budget is closed','Run new process-isolation GPU smoke before treating the new runner as GPU-validated','Never use12 unresolved source/scorer cases as supervised targets','Do not reuse old calibration blindly after executor/source policy changes; freeze comparable evaluation first'],'stop_conditions':['wrong base/adapter/hash/split','nonfinite loss or gradient','checkpoint write failure or insufficient disk','approved time or steps exhausted','failed old7 or full replay gate: preserve candidate, do not promote or run public']}
atomic(exp/'NEXT_RUN_PROPOSAL.json',proposal)
status={'observed_at':observed,'state':'CPU_code_data_tests_complete_GPU_validation_pending','source_archive_sha256':sha(archive),'baseline_tests':99,'prior_new_tests':131,'added_tests':52,'total_tests':282,'test_skips':0,'Kaggle_CPU_ref':'lnghongphc/zaic-relearning-v4-cpu-tests-20261110/5','dataset_manifest_sha256':data['dataset_manifest_sha256'],'selected_data_sha256':data['selection']['dataset_sha256'],'selection':data['selection'],'critical_old7_preserved':7,'full_replay_regression':103,'source_revalidation':real,'new_model_predictions_generated':0,'optimizer_executed':False,'GPU_executed':False,'long_training_ready':False,'requires_new_budget':True,'BEST_MODEL_changed':False,'LATEST_CHECKPOINT_changed':False,'LATEST_SYNCED_changed':False,'web_used':False,'capabilities':{'source_annotation_repair':'code + syntheticCPUtests + realTRAINCPUreexecution','temporal_replay_selection':'code + tests + frozen real verified bank','counterfactual_correction_rejection':'code + tests','isolated_optimizer_child':'code + actual CPU subprocess tests; not new GPU execution','protected103_and_old7_regression':'code + mocked-model end-to-end CPU tests; not new GPU execution','full_resume':'not implemented'}}
atomic(exp/'IMPLEMENTATION_STATUS.json',status)
report=f'''# CPU v4 — chuẩn bị vòng học tiếp theo

Thời điểm: {observed} (giờ Việt Nam). Thư mục giữ tên theo lúc tạo ngày09/11; báo cáo hoàn tất ngày10/11/2026.

## Đã thực hiện

- Làm hoàn toàn bằng công cụ nội bộ và Kaggle CLI/SDK. Không mở web, không chạy model/GPU/optimizer mới.
- Bản nguồn mới: `{exp/'src'}`. Baseline/app/tools giữ nguyên byte/hash; không sửa frozen_v21, model, adapter, checkpoint hay tracker.
- Đã tạo venv CPU riêng bằng gói PyPI chính thức: pandas2.3.3 để đọc cache cùng phiên bản, numpy2.2.6 và openpyxl3.1.5. Không thay môi trường cũ, không thay thiết lập symlink/bảo mật Windows.

## Kiểm tra thực tế nguồn và lỗi

- Tái thực thi124 dự đoán E7 đã lưu, xác minh manifest trước đọc. Kết quả103/124 đúng trước và sau sửa; correct→wrong=0. Đây không phải dự đoán mới của candidateD31, không phải holdout/public và không cộng lại thành coverage mới. Toàn learning pool còn1693 pending.
- Ba trường hợp từng báo Unparsed numeric amounts có dòng ghi chú bị kéo metadata giao dịch xuống. Bản sửa chỉ loại dòng được đối chiếu với XLSX gốc đúng checksum: marker ghi chú rõ, ô gốc ngày/ID/số lượng/đơn giá/thành tiền đều trống, nhãn nguồn và cached row không mơ hồ. Đọc công thức bằng data_only=False; không coi công thức chưa tính là ô trống. Không điền tiền hay sinh số0 để vượt guard.
- Bản sửa xử lý23 dòng ghi chú đã chứng minh trong các nguồn liên quan; ba trường hợp thoát exception nhưng đáp án vẫn không khớp gold. Vì thế không báo tăng accuracy và không đưa chúng làm target.
- Còn12 unresolved:1 chưa có bằng chứng phủ thời gian,11 canonical plan chưa khớp số với gold. Chưa xác định nguyên nhân cuối giữa ingestion, quy tắc nghiệp vụ, scorer và nhãn; không tự gán tất cả thành lỗi model hay sửa gold. Sai khác không thuộc nhóm lỗi làm tròn nhỏ hoặc hệ số tiền tệ đơn giản trong phép kiểm đã chạy.
- Đã kiểm348.618 dòng nguồn XLSX/SQLite:2.222 dòng trùng hoàn toàn bằng số dòng trùng sau chiếu cột; không thấy mất phân biệt dòng do phép chiếu trong phạm vi này. Không đổi cơ chế dedup dựa trên giả thuyết chưa được chứng minh. Có48 ID xuất hiện ở nhiều tháng trong1 doanh nghiệp; chưa có bằng chứng độc lập để đổi namespace ID.
- Bảy correction cũ đều chỉ sai time so với target yesterday. Cả7 đã có trong9 correction; lỗi không phải do bỏ chúng khỏi dataset. Giữ log và dữ liệu chẩn đoán chi tiết trong PRIVATE_DATA, chỉ xuất thống kê/hashes ra báo cáo.

## Correction/replay đã đóng băng

- Bank có9 correction và103 replay đã được xác minh bằng nguồn thực thi/scorer. Không tạo nhãn mới; giữ riêng12 unresolved. Evidence cũ chỉ được tái dùng sau khi đối chiếu hash bộ kiểm thực thi, dữ liệu, prompt và target; chính sách chọn mẫu mới có hash riêng.
- Chọn9 correction+9 replay,12 doanh nghiệp. Replay phủ5 nhóm: last_week2,month2,this_week2,today2,yesterday1; trước đây thiếu this_week. Không nhân bản mẫu để lấp quota.
- Giữ cả103 replay làm tập hồi quy TRAIN cố định. Mẫu sửa chỉ do executor, có plan không đổi, bị loại khỏi supervision correction; không bắt model học để che bug code.
- Dataset: `{bank}`. Manifest `{data['dataset_manifest_sha256']}`. Hash18 mẫu `{data['selection']['dataset_sha256']}`. Các file PRIVATE chứa nội dung thật, không đưa vào chat hay repo công khai. Đây là bản local đã freeze, chưa khẳng định đã upload input mới lên Kaggle.

## Runner và cổng đánh giá

- `src/isolated_training.py`: optimizer chạy đúng một child, parent đợi child thoát rồi mới đánh giá. Có timeout hữu hạn, log kín, giữ artifact khi lỗi, kiểm manifest/parent, chặn ghi vào vùng model/source được bảo vệ; không fallback fresh/resume.
- `src/time_pilot.py`: đã gọi child train; hỗ trợ tham chiếu lịch sử riêng và bank hồi quy đóng băng.
- `src/protected_regression.py`: sinh một lần trên9 correction+103 replay, rồi tính nhóm7 lỗi cũ từ dự đoán đã khóa bằng CPU; không sinh lại sau xem gold. Ca kiểm thử toàn luồng xác nhận: tăng correct ở mẫu mới nhưng nhóm cũ fail vẫn bị chặn, hoặc có bất kỳ correct→wrong nào thì không promote.
- Có code và tests CPU cho pipeline mới; chưa gọi là đã chứng minh giải phóng VRAM trên GPU. Full resume vẫn chưa triển khai/xác minh.

## Kiểm thử và bằng chứng

- Kaggle Linux CPU:75 app+24 tools+183 tests thí nghiệm=**282/282 đạt**,0 skip. Bao gồm230 tests trước và52 tests bổ sung. Kết quả: `KAGGLE_CPU_RESULT_V5/TEST_RESULTS.json`, chi tiết `TEST_DETAILS.txt`.
- Source archive `{sha(archive)}`. CLI ref `{status['Kaggle_CPU_ref']}`. Các version kiểm thử trước được giữ; version5 là bằng chứng cuối.
- Windows không cấp quyền tạo symlink cho một số test baseline và khác biểu diễn path; không sửa baseline hoặc bật quyền hệ thống để né. Kiểm tra toàn bộ được xác nhận trên Linux. Kiểm tra nguồn thật chạy Windows CPU; chưa coi đó là GPU/batch parity mới.
- Các bản nháp bị đổi source trong lúc đối chiếu được giữ riêng; guard hash đã chặn việc công nhận bản nháp đó. Chỉ dataset R3 đã seal và bản chọn temporal mới ở trên là evidence được dùng.
- BaselineE7, B306, adapter28, submission cũ và toàn runD31 được kiểm lại hash, còn nguyên. RUN_REGISTRY không thêm một lượt train giả cho công việc CPU; LATEST_CHECKPOINT/LATEST_SYNCED/BEST_MODEL giữ nguyên.

## Cấu hình đề xuất và phần còn thiếu

`NEXT_RUN_PROPOSAL.json` là đề xuất để xem xét, không phải lệnh đã được cấp quyền chạy. Đề xuất một pilot warm-start E7, optimizer/scheduler mới, tối đa18 steps (~4 epoch tương đương),LR1e-5,time-weight4,9+9 mẫu,checkpoint mỗi bước; ngân sách đề xuất tối đa120 phút gồm kiểm tra và calibration có điều kiện. Lý do: pilot9steps trước giảm NLL nhưng vẫn0/7 lỗi cũ; thêm bước chỉ là giả thuyết phải đo, không cam kết tối ưu.

Chưa mở vòng GPU mới. Trước chạy cần ngân sách mới, CPU preflight đường dẫn/input Kaggle thật, smoke runner mới và đánh giá so sánh đã khóa. Chỉ promote sau cổng7 lỗi cũ,9 correction,0 regression/103 replay và calibration đạt; không dùng gold public. Calibration552 đã tiếp xúc lịch sử, không gọi untouched holdout. Không đưa12 trường hợp chưa xác minh vào SFT và không hứa PA tăng.

Phạm vi đã hoàn thành: code, kiểm thử CPU, chẩn đoán nguồn và dataset local đóng băng. Phạm vi chưa chứng minh: GPU cho runner v4, model mới tốt hơn, full resume và tái lập bit-for-bit. Tham khảo IMPLEMENTATION_STATUS.json, DATASET_INVENTORY.json, ORIGINALS_PRESERVED.json và CLI_FINAL_STATUS.json.
'''
(exp/'CPU_READINESS_REPORT_VI.md').write_text(report,encoding='utf-8')
training=root/'training.md';shutil.copy2(training,exp/'training_BEFORE_CPU_APPEND.md')
with training.open('a',encoding='utf-8') as f:f.write(f'\n\n## CPU v4 — {observed}\n\n282/282 tests LinuxCPU đạt,0skip; baseline byte giữ nguyên. Sửa source-proven annotation, không impute tiền;124 dự đoán E7 đã lưu vẫn103 đúng,0regression,3exception được xử lý nhưng chưa tăngaccuracy.12unresolved gồm1coverage và11canonical/gold lệch, không ép làm supervision. Dataset9correction+9replay phủ5nhóm thời gian,12biz; bank103replay được giữ làm regression và7lỗi cũ nằm đủ trongcorrection. Manifest {data["dataset_manifest_sha256"]}. Runner train child+protected regression có code/testsCPU, chưaGPUv4; fullresume chưa triển khai. Không optimizer/model mới hoặc cập nhậtBEST/tracker. Chi tiết:{exp/"CPU_READINESS_REPORT_VI.md"}.\n')
atomic(exp/'CURRENT_WORK.json',{'state':'CPU_deliverables_complete','observed_at':observed,'report':str(exp/'CPU_READINESS_REPORT_VI.md'),'source_sha256':sha(archive),'dataset_manifest_sha256':data['dataset_manifest_sha256'],'no_active_local_worker':True,'no_optimizer_executed':True,'next_action':'Review budget/config then GPU smoke; do not rerun finalizer or claim new accuracy'})
print(json.dumps({'state':status['state'],'tests':282,'dataset_manifest_sha256':data['dataset_manifest_sha256'],'report':str(exp/'CPU_READINESS_REPORT_VI.md')},ensure_ascii=True))
