# Kết quả vòng học ZAIC ngày 09/11/2026

Quan sát UTC: 2026-11-09T10:05:54.041731+00:00. Run `20261109T065047772787Z_d5826809`. Phạm vi: pilot TRAIN, một vòng optimizer; không phải toàn bộ learning pool.

## Kết quả thực tế

- Warm-start từ adapter E7 đã xác minh; optimizer/scheduler mới, không resume checkpoint-48, không fallback fresh.
- 97/97 câu pilot có dự đoán model thật và ledger; toàn pool mới đạt 97/1817 (5,34%), còn 1720 pending.
- 79 replay đúng lần đầu, 7 correction đã xác minh, 11 unresolved vẫn được giữ. Correction coverage 7/18 = 38,89%; không bịa target.
- Chọn 7 correction + 7 replay thuộc 11 doanh nghiệp, không oversample. Dataset digest `06ce1359691931ba45ec2eab440016d5be5c245df574de3c3bfd40e8028b5372`.
- Train 7 bước, khoảng 140.9 giây optimizer, loss 0.128258; khoảng 1,857 epoch. Các trọng số hữu hạn, delta L2 0.07910480. Loss không phải bằng chứng tăng PA.

## Đánh giá trước/sau

Calibration 552 câu đã từng lộ, không phải untouched holdout. Cùng ID, decoder, scorer, manifest; chốt dự đoán trước khi đọc gold.

| Chỉ số | Parent E7 | Candidate |
|---|---:|---:|
| Đúng | 228 | 229 |
| Sai | 53 | 52 |
| Abstain | 271 | 271 |
| (đúng − sai)/552 | 0.31702899 | 0.32065217 |

Delta +0,00362319, đúng→sai trên calibration = 0. Chưa đạt ngưỡng đã khóa +0,01: **không promote**, BEST_MODEL giữ nguyên.

Đánh giá riêng 14 mẫu đã train, không cập nhật weights: correction đúng sau train 0/7 (trước 0); replay giữ đúng 7/7; đúng→sai 0. Đây là kiểm tra trên mẫu đã học, không chứng minh tổng quát hóa.

**Hiệu quả correction chưa đạt ở vòng này.** Cả 7 plan correction sau train trùng plan lần đầu của parent và khác target đã xác minh ở trường time. Không có lỗi parse/hạ tầng trong 7 mẫu này. Xem CORRECTION_OUTCOME_DIAGNOSIS.json; không sửa target hay hạ ngưỡng để báo thành công. Chưa xác minh nguyên nhân học chưa hiệu quả; nên đo riêng token/loss của trường time và mở rộng dữ liệu TRAIN đã kiểm chứng trước lượt optimizer tiếp theo.

## Public_test và giới hạn kết luận

Workflow chọn **adapter gốc E7** khi candidate không đạt cổng. Vì vậy public_test này không đo candidate mới. Đã hoàn tất 1000/1000: 235 answered, 765 abstain; loader errors 0, OOM retries 0, network attempts 0. Không truy cập public gold, không nộp bài.

SHA256 submission `e0815c96524e9db0badf987e45e13227eec8e57c7f3195f5aa93a580a0508780`, trùng baseline lịch sử. Bản B306 4-step và submission cũ vẫn nguyên hash, không bị thay thế.

## Artifact và đồng bộ

- Adapter mới: `28e745628fc1509018f93ea8b5299473d93f73ac907b63fefe9dfb48b3deca99`.
- 8 bộ adapter/state: checkpoint-1 đến checkpoint-7 và checkpoint cuối. Đủ file optimizer, scheduler, scaler, RNG, trainer state theo inventory; **chưa thử full resume**.
- 8 file adapter có 7 hash khác nhau: final trùng trọng số checkpoint-7, vẫn giữ cả hai bộ. 7/7 log loss và grad_norm hữu hạn; xem TRAINING_NUMERIC_CHECK.json.
- Đã đối chiếu SHA256/kích thước 135 file output; 103 file trong manifest training run; 85 file Python khớp source snapshot. Header safetensors của cả 8 adapter hợp lệ, không deserialize pickle để kiểm kê.
- Run manifest `1be04939b87cb36e6266e0a5e597a8fdf134e46c87e85322cc3e4f8673adeb02`; outer manifest `67ef3491e7c05274cadf34d53c31901018fee3d7963d32e3365c29c700061462`.
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

- Model run: `D:\ZAIC\PRIVATE_DATA\zaic\result_811\QUALITY_ROUND_20261109\round_finish\model_runs\20261109T065047772787Z_d5826809`.
- Source thực thi: `D:\ZAIC\PRIVATE_DATA\zaic\result_811\QUALITY_ROUND_20261109\DEPLOYED_CODE`; archive ec85b0ba610eb454dbbb57c3f7a04dbc48cb62c5c07074570ffa55e306c862ab.
- LOCAL_VERIFICATION.json, SYNC_RECEIPT.json, WEIGHTS_INVENTORY_RUN.json trong thư mục báo cáo này.
- Post: `D:\ZAIC\PRIVATE_DATA\zaic\result_811\QUALITY_ROUND_20261109\POST_REGRESSION`; manifest bd2ddc60382ffaa2812a673dc6176a112edb9ffa3dfd8530743e81bd1657abe6.
- Learning ledger: D:/ZAIC/PRIVATE_DATA/KAGGLE_RUNS/quality_round_v2_collection/round/learning_probe_runs_v2/20261109T054019836418Z_72f0b303.
- Kaggle training v1: https://www.kaggle.com/code/lnghongphc/zaic-quality-train-eval-public-20261109?scriptVersionId=367190892
- Kaggle post v1: https://www.kaggle.com/code/lnghongphc/zaic-quality-post-regression-20261109?scriptVersionId=367236935

Mở notebook mặc định CPU để xem preflight; không Run All notebook Colab lịch sử. Mỗi lượt optimizer tiếp theo cần run/config/output mới và ngân sách cho lượt đó. Không mặc định ngân sách 4 giờ còn nguyên.
