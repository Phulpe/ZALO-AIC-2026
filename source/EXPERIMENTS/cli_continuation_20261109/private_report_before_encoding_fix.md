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


## B? sung v?n h?nh sau khi n?i l?i

Trong l?c chat b? ng?t, phi?n ??nh gi? ?? t? ???c Kaggle c?p GPU v? ho?n t?t (436,6 gi?y); agent kh?ng th? th?c hi?n m?c h?y h?ng ??i khi ?? ng?t. Kh?ng c? optimizer b? sung. Phi?n CPU x?c nh?n m?c ??nh b? x?p h?ng, ?? d?ng tr??c CreateSession; c?u h?nh Accelerator=None v?n ???c l?u v? ki?m b?ng giao di?n. ?? ph?t hi?n th?m phi?n GPU t??ng t?c public c? ?ang nh?n r?i, d?ng ngay v? ??a v?None; kho?ng70,54 gi?y c?a phi?n n?y c?ng ???c c?ng b?o th? v?o t?ng35,74 ph?t. Kh?ng c? cell public ???c ch?y trong thao t?c n?y.

Giao di?n Kaggle c? c?nh b?o n?n t?ng g?p s? c? kh?i ??ng/duy tr? k?t n?i notebook; ?nh GPU_OFF_VERIFIED.png ghi l?i th?ng b?o n?y c?ng Draft Session off v?Accelerator=None.

12 unresolved g?m8 l?i model v?4 l?i executor. Tr??c v?ng train ti?p theo, ?u ti?n s?a4 l?i executor, m? r?ng correction ???c x?c minh cho c?c l?i ch?a gi?i quy?t v? chuy?n runner train/eval sang process t?ch bi?t. Kh?ng t?ng s? b??c ho?c d?ng gold public ?? ?p ??t c?ng.
