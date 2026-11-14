# Kết quả vòng học thời gian v3

Quan sát UTC: 2026-11-09T16:46:50.522428+00:00. Run `20261109T125033322737Z_b786bc50`. Một vòng optimizer, giới hạn 120 phút GPU.

## Thực thi và bảo toàn

- Warm-start từ E7 đã xác minh, base Qwen3.5-9B revision c202236235762e1c871ad0ccb60c8ee5ba337b9a; optimizer/scheduler mới. Không resume checkpoint48, không fallback fresh.
- 99 baseline +131 tests mới =230 đạt trên Kaggle Linux CPU; baseline source không đổi. Input, split, tokenizer/mask/shift/EOS được kiểm trước GPU.
- Pool TRAIN 124/124 được thử thật; 103 đúng ngay; 21 cần sửa, 9 correction được xác minh, 12 unresolved còn ledger. Toàn learning pool1817 còn1693 chưa thử ở vòng này; không tuyên bố coverage toàn bộ100%.
- Dataset:9 correction+9 replay,11 doanh nghiệp. Prediction được lưu trước đọc gold; target correction qua kiểm ngữ nghĩa/thực thi/scorer. Không dùng gold public.
- Train9 bước,1.889 epoch, LR1e-5, rank8, batch1/accum4. Loss trung bình 0.140494. Objective chỉ tăng trọng số4 cho giá trị trường time, không tăng trọng số dấu JSON cố định.
- Adapter mới `d3104b03f4e5a48b5b65fc5525a23d814eb025bc71b080be214ebc2e9e8ff11b`; 86.623.432 byte. 10 bộ checkpoint gồm step1–9 và final,9 hash adapter khác nhau. Optimizer/scheduler/scaler/RNG/trainer state đủ file, checksum khớp; chưa thực thi full resume.
- 148 artifact đã khớp manifest tải từ GPUv3. Notebook GPUv3 ERROR sau train vì guard phát hiện hơn512MiB bộ nhớ chưa giải phóng; training_result đã completed và checkpoint đã lưu. Giữ nguyên log lỗi. Không train lại; đánh giá bằng child inference riêng để giải phóng model theo vòng đời process.
- Bản bảo toàn CPU hoàn tất,128 file chỉ gồm model/state/tokenizer/metadata; không journal/câu hỏi/gold/SFT. Manifest run `c51d0b0f422ebb93f9a6fd927f9c7a3a6cb8d3eefffdf5987e89dca423863aaf` khớp bản máy. Bản zip đầy đủ có journal bị auto-review chặn và không upload; phương án whitelist state-only đã thay thế.
- E7, B306, adapter28 và submission cũ giữ nguyên hash. Không ghi đè kết quả public cũ, không nộp bài, không chạy public lần này.

## Đánh giá thật sau train

| Tập | Correction trước | Correction sau | Replay sau | Correct→wrong |
|---|---:|---:|---:|---:|
| 14 mẫu cũ | 0/7 | 0/7 | 7/7 | 0 |
| 18 mẫu vòng mới | 0/9 | 2/9 | 9/9 | 0 |

Cả hai tập là TRAIN đã tiếp xúc, không phải holdout. Đánh giá không cập nhật trọng số, nạp adapter strict và so đủ496 tensor với file nguồn. Chỉ kết quả calibration đủ điều kiện mới cho phép đổi BEST_MODEL.

| Adapter | NLL giá trị time trên7 correction cũ | NLL time replay cũ |
|---|---:|---:|
| E7 | 1.725643 | 0.000970 |
| 28 cũ | 1.431382 | 0.001042 |
| Candidate mới | 0.707811 | 0.001324 |

Teacher-forcing là chẩn đoán token, không phải độ chính xác trả lời mù.

Không chạy calibration đầy đủ: cổng 7 lỗi TRAIN cũ chưa đạt (0/7 sửa đúng), dù tập mới sửa được2/9 và giữ replay9/9. Không chọn model từ loss.

BEST_MODEL giữ nguyên. LATEST_CHECKPOINT và LATEST_SYNCED trỏ tới run9 bước được bảo toàn trên phiên CPU hoàn tất và khớp hash.

## Giới hạn và bước tiếp theo

- Đủ artifact cho inference; đã load/test trên GPU. Chưa chứng minh model tốt hơn trên dữ liệu độc lập nếu calibration không đạt.
- Đủ file trạng thái để khảo sát resume; chưa triển khai/xác minh full resume.
- Có source/config/data/split/environment/hash cho kiểm toán; chưa chứng minh tái lập bit-for-bit.
- Còn12 unresolved trong pool124 và1693 pending toàn pool. Không tự tạo target hoặc retry vô hạn để lấp số lượng.
- GPU dùng tổng 35.74/120 phút, gồm setup lỗi và đánh giá. GPU đã dừng, quota reserved0, Active Events0, Accelerator=None. Không bảo đảm điều khiển khi agent ngắt.
- Dữ liệu calibration552 đã tiếp xúc lịch sử; không gọi untouched holdout. Không có PA public mới của candidate này.

Bằng chứng: LOCAL_VERIFICATION.json, FINAL_STATUS.json, GPU_SHUTDOWN_RECEIPT.json, SAVED_VERSION_EVIDENCE.json, PRESERVATION_RECEIPT.json, time_pilot và evaluation_recovery trong thư mục này.

## Bổ sung vận hành sau khi nối lại

Trong lúc chat bị ngắt, phiên đánh giá đã tự được Kaggle cấp GPU và hoàn tất (436,6 giây). Agent không thể thực hiện mốc hủy hàng đợi khi đã ngắt. Không có optimizer bổ sung. Phiên CPU xác nhận mặc định bị xếp hàng đã được dừng trước CreateSession; Accelerator=None được lưu và kiểm bằng giao diện.

Đã phát hiện thêm phiên GPU tương tác public cũ đang nhàn rỗi, dừng ngay và đưa về None. Khoảng 70,54 giây của phiên này được cộng bảo thủ vào tổng 35,74 phút. Không có cell public được chạy trong thao tác này. Ảnh GPU_OFF_VERIFIED.png ghi nhận Draft Session off, Accelerator=None và cảnh báo sự cố kết nối notebook của Kaggle.

12 unresolved gồm 8 lỗi model và 4 lỗi executor. Trước vòng train tiếp theo, ưu tiên sửa lỗi executor, mở rộng correction được xác minh và tách process train/eval. Không tăng số bước hoặc dùng gold public để ép đạt cổng.

Chỉ đạo mới: ưu tiên Kaggle CLI/SDK; chỉ dùng web khi CLI/API thiếu chức năng cần thiết. Lần kiểm tra tiếp nối được lưu trong CLI_RECHECK_20261109.json. SDK có cancel_kernel_session yêu cầu session ID đã xác minh; chưa kiểm chứng gọi dừng thực tế bằng API trong lượt này.
