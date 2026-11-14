"""Finalize only from verified outputs and observed GPU shutdown evidence."""
from pathlib import Path
import json,csv,shutil,datetime,sys,hashlib
from relearning import verify_export,atomic,sha

def main():
 exp=Path(__file__).parent;root=Path('D:/ZAIC');dest=root/'PRIVATE_DATA/zaic/result_811/QUALITY_ROUND_20261109'
 v=json.loads((dest/'LOCAL_VERIFICATION.json').read_text(encoding='utf-8'));sync=json.loads((dest/'SYNC_RECEIPT.json').read_text(encoding='utf-8'));run=Path(v['run_path'])
 assert verify_export(run)==v['run_manifest_sha256']
 post=dest/'POST_REGRESSION';post_dir=post/'regression_output';post_manifest=verify_export(post_dir);r=json.loads((post_dir/'REGRESSION_SUMMARY.json').read_text(encoding='utf-8'))
 assert r['expected']==r['attempted']==r['real_output_ids']==14 and r['candidate_sha256']==v['candidate_adapter_sha256'] and not r['weights_updated']
 off=json.loads((dest/'GPU_SHUTDOWN_RECEIPT.json').read_text(encoding='utf-8'));assert off['active_events']==0 and off['interactive_draft_off'] and off['interactive_accelerator']=='None'
 tests=json.loads((exp/'FINAL_POST_CPU_RECEIPT/source/POST_CPU_TEST_RESULT.json').read_text());assert tests['tests']==2 and tests['failures']==tests['errors']==0 and not tests['gpu_executed']
 state=json.loads((dest.parent/'RUN_STATE.json').read_text(encoding='utf-8'));before=json.loads((dest/'RUN_STATE_BEFORE.json').read_text());assert state['BEST_MODEL']==before['BEST_MODEL']
 assert state['LATEST_SYNCED']['manifest_sha256']==v['run_manifest_sha256'] and state['LATEST_CHECKPOINT']['manifest_sha256']==v['run_manifest_sha256']
 observed=datetime.datetime.now(datetime.timezone.utc).isoformat();before_m=v['calibration']['parent_calibration'];after=v['calibration']['candidate_calibration'];inventory=json.loads((dest/'WEIGHTS_INVENTORY_RUN.json').read_text(encoding='utf-8'))
 status={'observed_at':observed,'scope':'One bounded TRAIN pilot and one optimizer round, not the full learning pool','run_id':run.name,'mode':'learning','init_mode':'warm_start','training_steps':v['training']['steps'],'source_adapter_sha256':'e7dfa635beccce128c99f9d3fd993d33dbb4c88e73e3ddeccbc2bc915eec4a4d','candidate_adapter_sha256':v['candidate_adapter_sha256'],'calibration':v['calibration'],'post_regression':r,'post_manifest_sha256':post_manifest,'public_model':'original E7 because candidate failed promotion threshold','public_rows':1000,'public_candidate_evaluated':False,'promoted':False,'checkpoint_sets_verified':len(inventory['checkpoints']),'artifact_files_verified':135,'source_files_verified':85,'linux_tests':{'baseline':99,'new':105,'failures':0,'errors':0},'full_pool':{'expected':1817,'attempted':97,'pending':1720,'unresolved_in_attempted':11},'gpu_shutdown':off,'full_resume_executed':False,'sync_receipt':str(dest/'SYNC_RECEIPT.json')}
 atomic(dest/'ROUND_FINAL_STATUS.json',status);atomic(exp/'ROUND_FINAL_STATUS.json',status)
 text=f'''# Kết quả vòng học ZAIC ngày 09/11/2026

Quan sát UTC: {observed}. Run `{run.name}`. Phạm vi: pilot TRAIN, một vòng optimizer; không phải toàn bộ learning pool.

## Kết quả thực tế

- Warm-start từ adapter E7 đã xác minh; optimizer/scheduler mới, không resume checkpoint-48, không fallback fresh.
- 97/97 câu pilot có dự đoán model thật và ledger; toàn pool mới đạt 97/1817 (5,34%), còn 1720 pending.
- 79 replay đúng lần đầu, 7 correction đã xác minh, 11 unresolved vẫn được giữ. Correction coverage 7/18 = 38,89%; không bịa target.
- Chọn 7 correction + 7 replay thuộc 11 doanh nghiệp, không oversample. Dataset digest `06ce1359691931ba45ec2eab440016d5be5c245df574de3c3bfd40e8028b5372`.
- Train {v['training']['steps']} bước, khoảng {v['training']['metrics']['train_runtime']:.1f} giây optimizer, loss {v['training']['metrics']['train_loss']:.6f}; khoảng 1,857 epoch. Các trọng số hữu hạn, delta L2 {v['candidate_delta']['delta_l2']:.8f}. Loss không phải bằng chứng tăng PA.

## Đánh giá trước/sau

Calibration 552 câu đã từng lộ, không phải untouched holdout. Cùng ID, decoder, scorer, manifest; chốt dự đoán trước khi đọc gold.

| Chỉ số | Parent E7 | Candidate |
|---|---:|---:|
| Đúng | {before_m['correct']} | {after['correct']} |
| Sai | {before_m['wrong']} | {after['wrong']} |
| Abstain | {before_m['abstain']} | {after['abstain']} |
| (đúng − sai)/552 | {before_m['metric']:.8f} | {after['metric']:.8f} |

Delta +0,00362319, đúng→sai trên calibration = 0. Chưa đạt ngưỡng đã khóa +0,01: **không promote**, BEST_MODEL giữ nguyên.

Đánh giá riêng 14 mẫu đã train, không cập nhật weights: correction đúng sau train {r['correction_after_correct']}/{r['correction_total']} (trước 0); replay giữ đúng {r['replay_after_correct']}/{r['replay_before_correct']}; đúng→sai {r['correct_to_wrong']}. Đây là kiểm tra trên mẫu đã học, không chứng minh tổng quát hóa.

**Hiệu quả correction chưa đạt ở vòng này.** Cả 7 plan correction sau train trùng plan lần đầu của parent và khác target đã xác minh ở trường time. Không có lỗi parse/hạ tầng trong 7 mẫu này. Xem CORRECTION_OUTCOME_DIAGNOSIS.json; không sửa target hay hạ ngưỡng để báo thành công. Chưa xác minh nguyên nhân học chưa hiệu quả; nên đo riêng token/loss của trường time và mở rộng dữ liệu TRAIN đã kiểm chứng trước lượt optimizer tiếp theo.

## Public_test và giới hạn kết luận

Workflow chọn **adapter gốc E7** khi candidate không đạt cổng. Vì vậy public_test này không đo candidate mới. Đã hoàn tất 1000/1000: 235 answered, 765 abstain; loader errors 0, OOM retries 0, network attempts 0. Không truy cập public gold, không nộp bài.

SHA256 submission `e0815c96524e9db0badf987e45e13227eec8e57c7f3195f5aa93a580a0508780`, trùng baseline lịch sử. Bản B306 4-step và submission cũ vẫn nguyên hash, không bị thay thế.

## Artifact và đồng bộ

- Adapter mới: `{v['candidate_adapter_sha256']}`.
- 8 bộ adapter/state: checkpoint-1 đến checkpoint-7 và checkpoint cuối. Đủ file optimizer, scheduler, scaler, RNG, trainer state theo inventory; **chưa thử full resume**.
- 8 file adapter có 7 hash khác nhau: final trùng trọng số checkpoint-7, vẫn giữ cả hai bộ. 7/7 log loss và grad_norm hữu hạn; xem TRAINING_NUMERIC_CHECK.json.
- Đã đối chiếu SHA256/kích thước 135 file output; 103 file trong manifest training run; 85 file Python khớp source snapshot. Header safetensors của cả 8 adapter hợp lệ, không deserialize pickle để kiểm kê.
- Run manifest `{v['run_manifest_sha256']}`; outer manifest `{v['outer_manifest_sha256']}`.
- LATEST_CHECKPOINT và LATEST_SYNCED đã cập nhật sau kiểm hash. BEST_MODEL không đổi; không dùng latest làm best.
- PHASE.json của wrapper cũ giữ nhãn trước khi chạy pha public. Kết luận hoàn tất dựa WORKFLOW_COMPLETED.json, runtime completed, trạng thái Kaggle và biên nhận tắt GPU; không sửa file đã seal để đổi nhãn.
- Adapter config còn giữ đường dẫn Colab lịch sử. Không sửa file đã seal. Base thực tế được ràng buộc bằng config/run manifest và 15 hash input: Qwen/Qwen3.5-9B revision c202236235762e1c871ad0ccb60c8ee5ba337b9a. Loader nạp base đã pin một cách tường minh.

## Trạng thái chức năng

| Chức năng | Code/tests | Chạy thực tế |
|---|---|---|
| Warm-start, optimizer/scheduler mới | Đã triển khai, qua tests | GPU train 7 bước |
| Blind learning và ledger/retry hữu hạn | Đã triển khai, qua tests | GPU 97 câu; 1720 pending |
| Semantic/executor/scorer correction gate | Đã triển khai, qua tests | 7 correction được chấp nhận; 11 unresolved loại khỏi train |
| Replay và phát hiện đúng→sai | Đã triển khai, qua tests | GPU 14 mẫu sau train và calibration 552 |
| Checkpoint/checksum/tracker | Đã triển khai, qua tests | 8 bộ state, tải và khớp hai nơi |
| Full resume | Hạng mục riêng, chưa triển khai/xác minh | Chưa chạy |

99 baseline + 103 tests mới của source đã freeze đều đạt Linux; thêm 2 tests post-regression đạt Linux CPU (tổng 204 tests khác nhau). Source frozen baseline không đổi. GPU smoke được chứng minh trong train/finite gradient, tensor delta và reload inference thực tế, không suy từ mock.

## Bốn kết luận

1. **Dùng thành quả đã học:** candidate đã reload và chạy inference thật; lưu riêng để nghiên cứu. Chưa được chọn làm BEST_MODEL.
2. **Tiếp tục huấn luyện:** có adapter để warm-start và đầy đủ file state đã kiểm hash; không gọi là đã chứng minh full resume. RNG checkpoint cuối do wrapper lưu; cần kiểm luồng resume tương thích riêng.
3. **Tái lập:** giữ source snapshot, entrypoint, metadata, run config, base/data/split hash, môi trường ghi nhận và checkpoint. Chưa rerun tái lập; chưa có đầy đủ lock toàn môi trường/container/driver để cam kết tái lập bit-for-bit.
4. **Còn thiếu:** 1720 learning IDs chưa thử; 11 unresolved; candidate chưa đạt cổng, chưa chạy public_test bằng candidate; full resume chưa kiểm. Chưa có PA chính thức mới. Quyền cuộc thi theo xác nhận người dùng, chưa xác minh nguồn độc lập.

## GPU và vận hành tiếp

Đã xác nhận Active Events = 0, quota API GPU reserved = 0; hai notebook thử nghiệm trở về CPU. Phiên tương tác cũ được quan sát off và Accelerator=None. Xem GPU_SHUTDOWN_RECEIPT.json và GPU_STOPPED_UI_EVIDENCE.md (bản ghi trạng thái giao diện). Job GPU tự kết thúc theo timeout; không hứa tự theo dõi khi agent đã ngắt.

Ngân sách thêm tối đa 4 giờ GPU được ghi riêng. GPU_BUDGET_RECONCILIATION.json dùng toàn bộ thời gian trước main COMPLETE làm cận trên (kể cả CPU), cộng hard timeout 1100 giây và reserve 180 giây của post: dưới 14400 giây. Thời gian xếp hàng không cộng thành GPU đã dùng.

Lần sau ưu tiên mở rộng correction đã xác minh từ TRAIN và xử lý lỗi ingestion/executor trước tăng steps; không học từ gold public. Không lặp train nhỏ để tiêu hết quota. Đánh giá mọi checkpoint trung gian là việc chưa làm; hiện chỉ candidate cuối có calibration.

## Đường dẫn bằng chứng

- Model run: `{run}`.
- Source thực thi: `{dest/'DEPLOYED_CODE'}`; archive ec85b0ba610eb454dbbb57c3f7a04dbc48cb62c5c07074570ffa55e306c862ab.
- LOCAL_VERIFICATION.json, SYNC_RECEIPT.json, WEIGHTS_INVENTORY_RUN.json trong thư mục báo cáo này.
- Post: `{post}`; manifest {post_manifest}.
- Learning ledger: D:/ZAIC/PRIVATE_DATA/KAGGLE_RUNS/quality_round_v2_collection/round/learning_probe_runs_v2/20261109T054019836418Z_72f0b303.
- Kaggle training v1: https://www.kaggle.com/code/lnghongphc/zaic-quality-train-eval-public-20261109?scriptVersionId=367190892
- Kaggle post v1: https://www.kaggle.com/code/lnghongphc/zaic-quality-post-regression-20261109?scriptVersionId=367236935

Mở notebook mặc định CPU để xem preflight; không Run All notebook Colab lịch sử. Mỗi lượt optimizer tiếp theo cần run/config/output mới và ngân sách cho lượt đó. Không mặc định ngân sách 4 giờ còn nguyên.
'''
 for folder in [dest,exp]: (folder/'ROUND_COMPLETION_REPORT_VI.md').write_text(text,encoding='utf-8')
 registry=root/'RUN_REGISTRY.csv';csv.field_size_limit(10_000_000)
 with registry.open(encoding='utf-8-sig',newline='') as f: reader=csv.DictReader(f);fields=reader.fieldnames;old=list(reader)
 assert not any(row['run_id']==run.name for row in old),'Registry already contains this run; do not duplicate'
 shutil.copy2(registry,dest/'RUN_REGISTRY_BEFORE.csv')
 original_registry_bytes=registry.read_bytes()
 if original_registry_bytes and not original_registry_bytes.endswith(b'\n'):
  with registry.open('ab') as f:f.write(b'\r\n')
 def js(value):return json.dumps(value,ensure_ascii=False)
 row=dict.fromkeys(fields,'');row.update(run_id=run.name,kind='training_run',time_and_basis=f'{observed}; sealed Kaggle v1 and local SHA256 verification',base_model='Qwen/Qwen3.5-9B',base_revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a',initialization='warm_start E7 original adapter; new optimizer/scheduler; no checkpoint resume',lineage_confidence='high: parent.json, pinned source SHA256, actual code and training artifacts',status='7 steps completed; synced; not promoted',final_adapter=js({'path':str(run/'checkpoint'),'sha256':v['candidate_adapter_sha256']}),all_checkpoints=js([{'path':i['path'],'step':i['step'],'adapter_sha256':i['files']['adapter_model.safetensors']['sha256']} for i in inventory['checkpoints']]),resume_state='8 sets have optimizer/scheduler/scaler/RNG/trainer state, hashes verified; full resume not implemented or tested',code_configuration=js({'source_archive':'ec85b0ba610eb454dbbb57c3f7a04dbc48cb62c5c07074570ffa55e306c862ab','snapshot':str(dest/'DEPLOYED_CODE'),'config':str(run/'run_config.json')}),data_manifest_hash=js({'runtime':'523db34eb8e8bab52aa52d1d046307f19dbb3668495025ca05d66abd1e4c6e70','selected':'06ce1359691931ba45ec2eab440016d5be5c245df574de3c3bfd40e8028b5372'}),library_environment=js(json.loads((run/'environment.json').read_text(encoding='utf-8'))),logs=js([str(dest/'round_finish'),str(post)]),metrics_evaluation=js({'calibration':v['calibration'],'decision':v['model_decision'],'post_regression':r,'public_adapter':'original E7; new candidate public not run'}),lessons='One calibration error corrected, delta below threshold; preserve BEST. Expand verified TRAIN corrections rather than overtrain 14 rows. Public baseline rerun reproduced old hash.',local_copies=js([str(dest)]),drive_copies='[]',drive_checksum_status='not updated or checked for this run',kaggle_copy='https://www.kaggle.com/code/lnghongphc/zaic-quality-train-eval-public-20261109?scriptVersionId=367190892',use_results='Saved candidate reload/inference proven; not promoted',continue_training='Warm-start possible; full resume unverified',reproduce='Source/config/input hashes/state preserved; no rerun or bit-for-bit guarantee',missing_unverified='1720 learning IDs pending;11 unresolved;full resume;new candidate public;official PA')
 with registry.open('a',encoding='utf-8',newline='') as f:csv.DictWriter(f,fieldnames=fields).writerow(row)
 assert registry.read_bytes().startswith(original_registry_bytes)
 with (dest/'RUN_REGISTRY_THIS_RUN.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerow(row)
 decision=f'\n\n## Cập nhật có bằng chứng 09/11/2026 — {run.name}\n\nWarm-start E7, 7 steps, 7 correction +7 replay đã xác minh. Calibration 552 câu đã lộ: 228→229 đúng, 53→52 sai, 0 correct→wrong, delta +0.00362319 <0.01: không promote. Post correction {r["correction_after_correct"]}/7, replay {r["replay_after_correct"]}/7, correct→wrong {r["correct_to_wrong"]}. Public1000 dùng E7 theo cổng; không phải kết quả candidate. 8 bộ checkpoint/state đã khớp hash Kaggle–máy; LATEST_CHECKPOINT/LATEST_SYNCED cập nhật, BEST giữ nguyên. GPU đã tắt. Full resume chưa triển khai/xác minh; còn1720 pending/11 unresolved. Bằng chứng: {dest/"ROUND_COMPLETION_REPORT_VI.md"}. Các mục trạng thái cũ trong tài liệu là lịch sử; xác nhận quyền hiện tại theo AGENTS.md/user attestation trong run config, không nâng thành bằng chứng thể lệ độc lập.\n'
 shutil.copy2(root/'training.md',dest/'training_BEFORE.md')
 with (root/'training.md').open('a',encoding='utf-8') as f:f.write(decision)
 atomic(exp/'ACTIVE_HANDOFF.json',{**status,'state':'completed_bounded_round','report':str(dest/'ROUND_COMPLETION_REPORT_VI.md'),'next_gpu_job':None,'next_round_requires_new_config_and_budget':True})
 print(json.dumps({'state':'completed','run_id':run.name,'report':str(dest/'ROUND_COMPLETION_REPORT_VI.md'),'registry_appended':True,'best_unchanged':True,'post_regression':r},ensure_ascii=True))

if __name__=='__main__':main()
