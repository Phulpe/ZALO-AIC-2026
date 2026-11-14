# Kế hoạch cải thiện điểm bằng quota Kaggle 30 giờ — 14/11/2026

Nguyên tắc (theo người dùng): **Kaggle chỉ dùng để cải thiện**, không chạy public trên Kaggle. Public chạy trên GPU thuê khi đã có cấu hình khóa. Chỉ dùng TRAIN, không dùng public gold. Mốc tốt nhất: v2b = 0,881.

## Điểm có thể giành (từ public v2b)
- Phần LLM (176 câu): khoảng 140 đúng / 21 sai / 15 bỏ. Đây là đòn bẩy chính.
- Phần hint (824 câu): còn khoảng 30 câu sai. Kiểm chéo bằng LLM chỉ đúng khi hint có dấu hiệu bất ổn (v2c 0,875 cho thấy điều đó). Phần này để làm trên CPU, không tốn GPU.

## Vấn đề đo lường cần sửa trước
Tập 38 câu quá nhỏ: chênh 2–3 câu là nhiễu, dễ chọn nhầm cấu hình. Cần **tập proxy lớn hơn** chỉ lấy từ TRAIN:
- 38 câu no-hint TRAIN/cal (giữ nguyên làm mốc lịch sử);
- khoảng 120 câu calibration/audit **có hint nhưng chạy giấu hint**, ưu tiên khuôn ít gặp (giống câu LLM trên public). Các câu này không nằm trong ngân hàng few-shot (chỉ lấy từ learning).
- Chia 2 nửa: **dev** để chọn cấu hình, **holdout** (khóa, chỉ mở ở bước xác nhận).

## Các giai đoạn

| GĐ | Việc | Nơi chạy | GPU |
|---|---|---|---|
| 0 | Tạo tập proxy dev/holdout (manifest + sha); phân tích lỗi các câu sai/bỏ đã có (38 câu × 3 lượt, CPU); viết plan kernel `improve`; preflight CPU trên Kaggle | máy local + Kaggle CPU | 0 |
| 1 | **Một lượt thu thập lớn** trên dev: cấu hình hiện tại nhưng **5 mẫu, ghi đủ mọi mẫu** (`--fixed-samples`). Từ log, mô phỏng offline trên CPU: 3 mẫu / 5 mẫu, min-votes 2/3, ngưỡng bỏ trống theo số phiếu, k few-shot → một lượt GPU cho nhiều cấu hình | Kaggle 2×T4 | ~8 h |
| 2 | Chế độ thinking **chỉ cho câu khó** ở GĐ1 (phiếu chia rẽ hoặc bỏ trống), max-tokens lớn hơn; đo xem có chuyển bỏ trống → đúng mà không thêm sai | Kaggle | ~6 h |
| 3 | Sửa code/prompt theo phân tích lỗi (ví dụ quy đơn vị, làm tròn, định dạng danh sách, kỳ thời gian); kiểm trước bằng test CPU, rồi đo trên dev | local + Kaggle | ~6 h |
| 4 | **Xác nhận** cấu hình tốt nhất (ghép GĐ1–3) so với mốc trên **holdout** chưa mở. Khóa cấu hình nếu thắng rõ và không thêm câu sai | Kaggle | ~5 h |
| — | Dự phòng | | ~5 h |

Tiêu chí giữ một thay đổi (khóa trước khi chạy): trên dev, net tăng ≥ 3 và số sai không tăng; trên holdout, không kém mốc. PA phạt câu sai, nên ưu tiên giảm sai hơn tăng số câu trả lời.

## Song song, không tốn GPU
- Rà các luật hint có khuôn chưa gặp trên TRAIN (support 0: 87 câu public) bằng đọc code và test TRAIN tổng quát, không theo ID public.
- Chuẩn bị gói GPU thuê v7 chứa cấu hình đã khóa, để chạy public một lần khi bạn thuê máy.

## Kỳ vọng (ước lượng, chưa đo)
Phần LLM: giảm sai 21 → khoảng 12–15 và trả lời thêm vài câu đang bỏ trống. Tổng khoảng +0,005 đến +0,02, tức public khoảng 0,886–0,90. Không cam kết.
