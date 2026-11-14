"""Record completed training without inventing unavailable GPU evaluation."""
from pathlib import Path
import json,sys,csv,shutil,datetime,zipfile,stat
exp=Path(__file__).parent;sys.path.insert(0,str(exp/'src'))
from relearning import sha,verify_export,atomic,seal
dest=Path('D:/ZAIC/PRIVATE_DATA/zaic/result_811/TIME_FOCUSED_V3_20261109')
v=json.loads((dest/'LOCAL_VERIFICATION.json').read_text(encoding='utf-8'));run=Path(v['run_path']).resolve();assert verify_export(run)==v['training_manifest_sha256'];assert verify_export(dest/'time_pilot')==v['outer_manifest_sha256']
off=json.loads((dest/'GPU_SHUTDOWN_RECEIPT.json').read_text(encoding='utf-8'));blocked=json.loads((dest/'EVALUATION_CAPACITY_BLOCK.json').read_text(encoding='utf-8'));assert off['gpu_reserved_seconds']==0 and off['active_events']==0 and off['accelerator']=='None' and off['total_this_round_seconds']<=7200
assert blocked['queued_minutes']>=60 and blocked['evaluation_model_started'] is False and blocked['queued_run_cancelled'] is True
assert json.loads((dest.parent/'RUN_STATE.json').read_text(encoding='utf-8'))==json.loads((dest/'RUN_STATE_AFTER_PRESERVATION.json').read_text(encoding='utf-8'));assert v['training']['state']=='completed' and v['training']['steps']==9
for item in json.loads((exp/'PRE_RUN_SNAPSHOT.json').read_text(encoding='utf-8'))['protected_files']:assert sha(item['path'])==item['sha256']
cfg=json.loads((run/'run_config.json').read_text(encoding='utf-8'));assert sha(exp/'CODE_V3.zip')==cfg['source_sha256'];tests=json.loads((exp/'CPU_DEFAULT_V4/source/TEST_RESULTS.json').read_text(encoding='utf-8'));assert tests['baseline']['passed'] and tests['new']['passed']
code=dest/'DEPLOYED_CODE';code.mkdir(exist_ok=False);source=code/'source_snapshot';source.mkdir()
with zipfile.ZipFile(exp/'CODE_V3.zip') as z:
 for item in z.infolist():assert (source/item.filename).resolve().is_relative_to(source.resolve()) and not stat.S_ISLNK(item.external_attr>>16)
 z.extractall(source)
for name,digest in json.loads((run/'source_manifest.json').read_text(encoding='utf-8')).items():assert sha(source/name)==digest
for p,name in [(exp/'REPAIR_01/GPU_DEPLOY/gpu.py','training_gpu_version3.py'),(exp/'EVALUATION_RECOVERY/gpu.py','evaluation_gpu_version1.py'),(exp/'EVALUATION_RECOVERY/evaluation_child.py','evaluation_child.py'),(exp/'CPU_PRESERVATION/preserve.py','preservation_cpu.py')]:shutil.copy2(p,code/name)
seal(code)
now=datetime.datetime.now(datetime.timezone.utc).isoformat();col=v['COLLECTION_SUMMARY.json'];unresolved=json.loads((dest/'UNRESOLVED_AGGREGATE.json').read_text(encoding='utf-8'));preservation=json.loads((dest/'PRESERVATION_RECEIPT.json').read_text(encoding='utf-8'));sync=json.loads((dest/'PRESERVATION_SYNC_EVIDENCE.json').read_text(encoding='utf-8'));assert preservation['training_manifest_sha256']==v['training_manifest_sha256']
result={'observed_at':now,'state':'training_completed_evaluation_blocked_by_Kaggle_queue','run_id':run.name,'training':v['training'],'candidate_sha256':v['candidate_sha256'],'training_manifest_sha256':v['training_manifest_sha256'],'collection':col,'selection':v['SELECTION_REPORT.json'],'unresolved':unresolved,'checkpoint_sets':10,'unique_adapter_hashes':9,'LATEST_CHECKPOINT_updated':True,'LATEST_SYNCED_updated':True,'BEST_MODEL_unchanged':True,'candidate_reloaded_for_inference':False,'post_evaluation_completed':False,'calibration_ran':False,'public_ran':False,'full_resume_executed':False,'gpu_shutdown':off,'capacity_block':blocked,'remaining_gpu_seconds':7200-off['total_this_round_seconds'],'preservation':preservation,'source_gpu_wrapper_failed_after_checkpoint':True}
atomic(dest/'FINAL_STATUS.json',result);atomic(exp/'FINAL_STATUS.json',result)
report=f'''# Vòng học thời gian v3 — train đã xong, đánh giá bị chặn

Quan sát UTC:{now}. Run `{run.name}`. Giới hạn120 phút GPU, một vòng optimizer.

## Kết quả thực tế

- CPU99 baseline+131 tests mới=230 đạt. Source archive `{cfg['source_sha256']}`. Baseline source giữ nguyên.
- TRAIN124/124 câu được thử thật;103 đúng ngay.21 cần sửa,9 correction được xác minh,12 unresolved gồm8 lỗi model và4 executor. Gold mở sau prediction được lưu; không lấy gold public làm supervision.
- Dataset9 correction+9 replay,11 doanh nghiệp. Còn1693/1817 câu learning pool chưa thử ở vòng này; không gọi coverage toàn pool100%.
- Warm-start E7 đã xác minh, baseQwen3.5-9B revisionc202236235762e1c871ad0ccb60c8ee5ba337b9a, optimizer/scheduler mới. Không resume checkpoint48 hoặc fallback fresh.
- GPU đã train9 steps,epoch1.889,LR1e-5,rank8,batch1/accum4. Loss trung bình {v['training']['metrics']['train_loss']:.6f}; gradient/loss hữu hạn theo guard. Objective giá trị time trọng số4, giữ mask/shift/EOS. Loss không chứng minh PA tăng.
- Adapter mới86.623.432 byte; SHA256 `{v['candidate_sha256']}`. Delta trọng số hữu hạn và khác E7. Header safetensors kiểm an toàn; chưa reload candidate để suy luận trên GPU.
- 10 bộ checkpoint gồmstep1–9 vàfinal,9 hash adapter khác nhau; giữ cả final trùng checkpoint9. Đủ optimizer/scheduler/scaler/RNG/trainer state theo file; chưa thực thi full resume.
- 148 artifact từ GPUv3 đã tải về và khớp checksum. Bản bảo toàn CPU hoàn tất và xác minh128 file state/tokenizer/metadata,1.512.433.451 byte. Manifest `{v['training_manifest_sha256']}`. LATEST_CHECKPOINT vàLATEST_SYNCED đã cập nhật bằng phiên CPU{sync['script_version_id']}. BEST_MODEL giữ nguyên.
- E7,B306,adapter28 vàsubmission cũ khớp hash trước/sau. Không xóa/đổi tên/gộp bản gốc; không chạy public hoặc nộp bài vòng này.

## Hai vấn đề tách riêng

1. GPUtrainv3 đã hoàn tất optimizer và lưu đầy đủ checkpoint nhưng wrapper lỗi sau train: guard thấy hơn512MiB bộ nhớ vẫn còn allocated. Giữ nguyên FAILURE và log. Không chạy optimizer lần thứ hai.
2. Đã chuẩn bị và submit wrapper inference-only trong process riêng, strict hash/tensor load, đánh giá14 mẫu cũ+18 mẫu mới, sau đó calibration có điều kiện. Kaggle giữQUEUED ít nhất{blocked['queued_minutes']:.1f} phút, chưa khởi động model. Đã hủy hàng đợi để không tự bật GPU ngoài giám sát; không có kết quả post-test/calibration để báo.

Gói đầy đủ có journal bị auto-review chặn upload. Đã thay bằng whitelist128 file model/state/metadata, loại toàn bộ câu hỏi/gold/SFT/prediction/journal; bản Private này đã được CPU đối chiếu hoàn tất. Journal gốc vẫn được giữ trong bản run cũ trên Kaggle và máy.

## Mức triển khai

| Hạng mục | Trạng thái và phạm vi |
|---|---|
| Warm-start | Đã viết code, qua tests, chạy GPU9 bước từ E7 với optimizer/scheduler mới |
| Learning/correction/replay | Qua tests; thu predictionGPU124 câu;9+9 supervision đã kiểm, thực sự train |
| Objective time | Qua tests mask/shift/token alignment và9 bướcGPU, chưa chứng minh cải thiện trả lời mù |
| Checkpoint/sync | Đã chạy thực tế;10 bộ state, manifest máy–Kaggle khớp |
| Đánh giá candidate sau train | Code đã chuẩn bị/compile; chưa chạy GPU vìQUEUED; không gọi đã đạt |
| Pipeline một process | GPUintegration lỗi cleanup sau train; vòng đời model cần tách process ở runner sau |
| Full resume | Chưa triển khai/xác minh; có file không đồng nghĩa đã resume được |

## Bốn kết luận và phần còn lại

1. **Dùng thành quả đã học:** đã có adapter mới đầy đủ, hash/header/delta hợp lệ; chưa thử reload/inference candidate. BaselineE7 vẫn nguyên và dùng được như trước.
2. **Tiếp tục train:** có đủ file state để khảo sát resume; fullresume chưa thực thi. Warm-start đã chạy thật. Không mặc định model mới tốt hơn.
3. **Tái lập:** có source/config/dataset/split manifests và environment cùng checksum. Chưa chứng minh tái lập bit-for-bit.
4. **Còn thiếu:** post14+18, cổng correct→wrong, calibration552 nếu đủ điều kiện;12 unresolved và1693 pending. Calibration552 đã tiếp xúc lịch sử, không phải untouched holdout.

GPU đã dừng, Accelerator=None, ActiveEvents0, quota reserved0. Đã dùng {off['total_this_round_seconds']/60:.2f}/120 phút; còn tối đa {(7200-off['total_this_round_seconds'])/60:.2f} phút theo cấp ngân sách này, chưa cấp optimizer vòng mới. Cấu hình recovery chỉ inference, hard4500giây, có thể chạy khi Kaggle cấpGPU. Không cần người dùng nhậpCLI. Không hứa tự theo dõi/tắt GPU khi agent ngắt.

Bằng chứng: LOCAL_VERIFICATION.json, GPU_SHUTDOWN_RECEIPT.json, EVALUATION_CAPACITY_BLOCK.json, PRESERVATION_RECEIPT.json, PRESERVATION_SYNC_EVIDENCE.json và WEIGHTS_INVENTORY_RUN.json trong thư mục này. Source/evaluation wrapper lưu trong DEPLOYED_CODE. Sự cho phép xử lý trên Kaggle được ghi theo xác nhận người dùng trong run config; không tự nâng thành nguồn thể lệ độc lập.
'''
(dest/'RESULT_REPORT_VI.md').write_text(report,encoding='utf-8');(exp/'RESULT_REPORT_VI.md').write_text(report,encoding='utf-8')
registry=Path('D:/ZAIC/RUN_REGISTRY.csv');shutil.copy2(registry,dest/'RUN_REGISTRY_BEFORE.csv')
with registry.open(encoding='utf-8-sig',newline='') as f:r=csv.DictReader(f);fields=r.fieldnames;rows=list(r)
assert not any(r['run_id']==run.name for r in rows)
row={k:'' for k in fields};row.update(run_id=run.name,kind='warm_start_temporal_correction_replay',time_and_basis=now,base_model='Qwen/Qwen3.5-9B',base_revision=cfg['revision'],initialization='warm_start E7;new optimizer/scheduler;9 steps',lineage_confidence='verified parent/source/data hashes',status='train completed;post evaluation not started;KaggleQUEUED then cancelled',final_adapter=str(run/'checkpoint'),all_checkpoints='step1-9+final;10 sets;9 unique hashes',resume_state='optimizer/scheduler/scaler/RNG/trainer present;fullresume unverified',code_configuration=str(code),data_manifest_hash=cfg['data_manifest_sha256'],library_environment=str(run/'environment.json'),logs='private training cleanup failure preserved;evaluation queue evidence',metrics_evaluation='TRAIN124:103correct;9correction+9replay;12unresolved;loss0.140494;post/cal/public not run',lessons='training saved before cleanup guard failure;separate process evaluation prepared but Kaggle did not allocate GPU',local_copies=str(dest),drive_copies='not checked this round',drive_checksum_status='unverified',kaggle_copy=sync['preservation_url'],use_results='artifact integrity verified;candidate inference not executed;BEST unchanged',continue_training='warmstart proven;resume only file presence',reproduce='source/data/split/env hashes;no bit-for-bit proof',missing_unverified='post14+18;conditionalcalibration;12unresolved;1693pending;fullresume')
with registry.open('a',encoding='utf-8',newline='') as f:csv.DictWriter(f,fields).writerow(row)
training=Path('D:/ZAIC/training.md');shutil.copy2(training,dest/'training_BEFORE.md')
with training.open('a',encoding='utf-8') as f:f.write(f'\n\n## Bằng chứng vòng v3 — {run.name}\n\nWarm-start E7,9steps,9 correction+9replay,CPU230tests đạt. TRAIN124/124 đã thử;9 correctionverified,12unresolved,1693pending toànpool. Traincompleted và checkpoint đủ trước khi wrapper lỗi guard allocatedmemory. Không train lại; inference-only recovery bịKaggleQUEUED hơn60phút,chưa chạy model,đã hủy để bảo vệquota. Candidate {v["candidate_sha256"]};10 bộ state khớp manifest máy–Kaggle CPU savedversion. LATEST_CHECKPOINT/LATEST_SYNCED cập nhật,BEST giữ nguyên. Chưa cóposttest/calibration/public của candidate;fullresume chưa chứng minh. Báo cáo:{dest/"RESULT_REPORT_VI.md"}.\n')
atomic(dest/'CAPABILITY_STATUS.json',{'warm_start':'GPU9steps_completed','learning':'GPU124_attempts_completed','correction_replay':'verified9+9_and_trained','temporal_loss':'GPU9steps_completed_quality_unverified','checkpoint_sync':'10sets_verified','post_evaluation':'not_started_Kaggle_queue','single_process_integration':'failed_after_training_cleanup','full_resume':'not_implemented'})
print(json.dumps({'state':result['state'],'report':str(dest/'RESULT_REPORT_VI.md'),'gpu_minutes':off['total_this_round_seconds']/60}))
