# Kết quả pilot chuẩn bị thi — 10/11/2026

Run: `20261109T193243329820Z_4be82a43` (ID theo UTC, thực thi ngày 10/11 giờ Việt Nam). Kaggle: `lnghongphc/zaic-competition-pilot-20261110/3`.

## Đã chạy thực tế

- Bản mã riêng; E7 và mọi candidate/checkpoint cũ giữ nguyên. 288/288 tests Kaggle Linux CPU đạt, không skip, gồm 99 baseline. Thêm 3 kiểm tra hồi quy bootstrap tại máy.
- Preflight input thật xác minh source, base, parent, bank và cache; wrapper chỉ có một CONFIG có hiệu lực. Lần bootstrap lỗi ban đầu bị chặn trước optimizer, giữ log riêng.
- GPU smoke trên mẫu dài nhất 1.319 token: loss 0,0822248, 496 tensor gradient hữu hạn, mask/shift/EOS đúng; VRAM đỉnh 9.141.243.904 byte. Không optimizer trong smoke.
- Warm-start adapter E7, optimizer/scheduler mới; không resume hoặc fallback fresh. Train đủ 18 bước, khoảng 342 giây optimizer, epoch thực ghi 3,667, time weight4. Không gọi đây là 4 epoch đầy đủ.
- Candidate `94a5bcc879cec0f00ba725e42a6df0db83532fcfed8b85e21e63d2780b0247a5`; reload đối chiếu 496 tensor khớp file.

| Phạm vi TRAIN đã tiếp xúc | Trước | Sau |
|---|---:|---:|
| Correction đã xác minh | 0/9 | 8/9 |
| Nhóm 7 lỗi cũ, nằm trong 9 correction | 0/7 | 6/7 |
| Replay bảo vệ | 103/103 | 103/103 |

Đánh giá có 112/112 dự đoán thật, không cập nhật trọng số, 0 đúng→sai. Nhóm cũ lấy từ cùng dự đoán, không generate lại sau đọc gold. Chỉ còn một correction khác target ở trường time; 8 plan trùng target, không có lỗi sinh/parse. Không dùng các số này làm ước lượng accuracy thi hoặc holdout.

## Bảo toàn

259 file đã đối chiếu bytes/SHA256 khớp manifest phiên Kaggle v3, gồm 19 bộ adapter/checkpoint (step1–18 và final). Không deserialize optimizer/RNG để kiểm kê. File trạng thái có đủ theo inventory nhưng chưa triển khai/thực thi full resume.

Bản máy: `D:\ZAIC\PRIVATE_DATA\zaic\result_811\COMPETITION_PILOT_20261110\KAGGLE_V3\competition_pilot`.

Biên nhận: `SYNC_VERIFICATION.json`. Tracker riêng của thí nghiệm: `MODEL_STATE.json`; LATEST_CHECKPOINT và LATEST_SYNCED trỏ run mới sau kiểm hash, BEST_MODEL giữ nguyên. Không sửa STATE.json nằm trong output đã seal. Root RUN_REGISTRY.csv đã thêm run, giữ schema cũ và có bản sao trước cập nhật.

## Trạng thái tiếp theo

Candidate vượt cổng TRAIN để vào calibration, **chưa được chọn BEST_MODEL**. Preflight calibration cặp đã passed; job `lnghongphc/zaic-paired-calibration-20261110/1` đã khởi chạy. So E7 và candidate trên cùng 552 câu, cùng source/decoder/scorer; gold chỉ mở sau lưu dự đoán. Tập này có exposure lịch sử, không phải untouched holdout. Chưa có kết quả calibration lúc viết báo cáo này.

Ngân sách mới 240 phút GPU không bao gồm public_test. Sau pilot và lần bootstrap lỗi, API ghi khoảng 35,016 phút đã dùng, còn 204,984 phút trước calibration. Đây là số quan sát, cần đối chiếu lại sau job. Notebook pilot đã Quick Save v4 với enable_gpu=False; quota reserved0 tại lúc xác nhận. Không tạo phiên tương tác trong nhiệm vụ CLI này.

Chưa public_test, chưa nộp bài, chưa có điểm thi chính thức mới. Chưa chạy toàn learning pool1817; không tuyên bố hoàn thành coverage toàn pool. Không suy ra cải thiện tổng quát chỉ từ loss hoặc 8/9 correction.

## Kiểm tra bổ sung trong lúc calibration chạy

- 15/15 file base khớp metadata chính thức tại revision đã pin: shard/tokenizer LFS so SHA256 và bytes; file Git nhỏ so Git blob SHA1 và SHA256 đã khóa. Không tải lại model. Xem BASE_OFFICIAL_REVISION_VERIFICATION.json.
- API model chính thức ghi revision hiện dùng có từ tháng3/2026, Apache-2.0. Header adapter và metadata base cho tổng tham số suy luận tối đa9.674.743.536, dưới15tỷ; không tính parent E7 làm model inference vì chỉ đóng gói làm evidence train, không được dùng ensemble/fallback tự động.
- Đã đọc Evaluation/Rule/Submission Guidance và Docker guideline công khai; thêm14tests đặc tả chấm đạt CPU. Không sửa scorer đang dùng trong calibration, không chứng nhận mã chấm riêng của ban tổ chức trùng byte với scorer nội bộ.
- Docker draft, entrypoint và manifest đã viết; 4tests cổng release đạt. Có base PyTorch2.11/CUDA12.8 pin digest, nguồn clean đúng archive đã qua288tests. Chưa build/chưa chạy GPU Docker; runtime CUDA khác Kaggle cần smoke/batch parity khi build thật.
- Người dùng chọn **chỉ chuẩn bị file Docker**, không bật daemon hoặc đổi kho lưu. Người dùng chọn **chưa chạy public_test**. Hai quyết định này thay mọi giả định sẽ chạy các bước đó ngay trong lượt hiện tại.
- Audit evaluator ngoài learning pool có5tests chống rò gold đạt CPU; chưa chạy GPU hoặc dùng nhãn audit để học. Chỉ xem xét audit nếu calibration đạt và ngân sách còn đủ.

## Calibration bị ngắt và tiếp tục suy luận

Phiên paired-calibration v1 ERROR vì child candidate chạm2700giây. E7 đã hoàn thành552câu:228đúng,53sai,271abstain, PA nội bộ0,3170289855. Candidate có538plan/dự đoán thật và14pending; file552dòng có abstain tạm không thay thế attempt. Không chấm/phán đạt cho candidate thiếu14câu.

Đã đưa notebook về CPU bằng QuickSavev2, API xác nhận enable_gpu=false, machine_shape=None và quota reserved0. Snapshot lỗi20file tải về khớp manifest233b46741d905ddbf52ff85720ad6719ecd006639be0e29f9d616294e76458b2. Tổng GPU chốt sau phiên này123,548phút, còn116,452phút trong240phút.

Kaggle không cho attach phiên ERROR làm input. Upload seed dẫn xuất TRAIN ban đầu bị automatic approval review từ chối vì cần phê duyệt payload cụ thể. Người dùng đã trực tiếp cho phép đúng ZIP53.444byte, SHA2568115b30d91731d8d0c6e2e30fdecee5419e48a3167420ec188bc08414e205848 vào dataset Private. Gói chỉ có538recordplan, fingerprint, verdictE7 và manifest nguồn; không có filegold/câu hỏi/giao dịch/loginputs/token, nhưng plan có thể chứa tên thực thểTRAIN. Không chuyển gói bằng đường khác để vượt từ chối. Dataset sau phê duyệt: `lnghongphc/zaic-calibration-journal-seed-20261110/1`, ready.

Preflight CPU recoveryv2 đã passed, xác nhận fingerprint nguyên bản,538record cũ và14 lượt sinh còn thiếu. Phiên `lnghongphc/zaic-calibration-recovery-20261110/1` đang RUNNING với cap1800giây. Đây là inference journal continuation, không full-resume training, không optimizer, không tính lại E7. Cần xác nhận14record mới và538record cũ không đổi sau khi job hoàn tất. Thời gian recovery không được dùng làm tốc độ chạy mới toàn552câu.
