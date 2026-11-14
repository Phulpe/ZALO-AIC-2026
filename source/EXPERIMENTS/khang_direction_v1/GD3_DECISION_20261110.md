# GĐ3 — quyết định trước khi chạy (10/11/2026)

Nguồn ủy quyền: người dùng duyệt "phương án đầy đủ" (780 phút GPU, dừng hỏi nếu dự kiến > 936 phút) và ngày 10/11 yêu cầu làm liên tục, tự quyết nếu không phản hồi trong 15 phút.

- run: `khang_dir_v1_gd3_*`, mode=learning (pha A–C), evaluation (pha D)
- **init_mode = fresh**: LoRA mới từ base Qwen/Qwen3.5-9B rev c2022362… (bản đã pin, kiểm 11 file nhỏ theo hash). Không warm-start từ adapter 9B cũ (e7dfa635…) hay E7. Lý do: adapter cũ học định dạng plan JSON của hướng cũ, khác định dạng hội thoại viết code Python của Khang; nạp vào sẽ kéo model về định dạng cũ. Checkpoint-48 và adapter cũ giữ nguyên, không đụng tới.
- Dữ liệu: learning pool TRAIN (18 doanh nghiệp, 1817 câu) — runtime-v2 manifest 523db34e…; gold learning chỉ dùng ở pha B để lọc quỹ đạo đúng (rejection sampling). Runner không nhận gold. Calibration/audit không dùng để học.
- Đánh giá: 120 ID calibration đã đóng băng (f337f7f8…), hints-off, không gold khi chạy; chấm cục bộ sau khi khóa dự đoán.
- LoRA: r=8, alpha=16, all-linear, lr 1e-4 cosine, 1 epoch, grad-accum 16, maxlen 6144, tối đa 2 quỹ đạo/câu; loss chỉ trên lượt assistant cuối (code cho ra đáp án đúng).
- Dừng: loss/gradient không hữu hạn 3 lần, SFT < 64 mẫu, hết giờ từng pha. Không fallback sang chế độ khác.
- Thăng hạng (đề xuất, chưa có xác nhận người dùng): PA trên 120 câu tăng ≥ +0.02 so với base GĐ2 và số câu đúng→sai ≤ 1/3 số câu sai→đúng.
- Tham số thời gian từng pha sẽ chỉnh theo số giây/câu đo ở GĐ2 (v5) trước khi đẩy job.
