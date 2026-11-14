# Kết quả no-hint base 9B (gd3 kernel v2, 11/11/2026)

- run_id khang_dir_v1_nohint_base_20261111, init_mode=fresh (base, không adapter), mode=evaluation, hints ON, 3 mẫu/vote≥2.
- Output: PRIVATE_DATA/khang_direction_v1/gd3_v2 (STATUS.json, ARTIFACT_MANIFEST.json). GPU 11017 s (~184 phút); quota sau chạy 13.55/30 h.
- 38 câu no-hint (26 learning + 12 calibration): trả lời 5 (đúng 4, sai 1), bỏ trống 33. Net +3. Chấm cục bộ sau khi khóa dự đoán, scorer khang answers.score.
- Tốc độ: 89 lượt gọi LLM / ~10700 s ≈ 120 s/lượt (prompt 20–22k token, chunked prefill, batch 1, peak ~9.8 GB/T4).
- Kết luận: RFT/LoRA trên learning pool (1817 câu) không khả thi trên T4 với prompt 18k+ token (ước tính nhiều ngày GPU). Giá trị no-hint thấp (+3 net/38 câu với 3 giờ GPU).
- Quyết định (tự quyết theo quy tắc 15 phút): dừng tiêu GPU cho GĐ3; giữ chính sách Tier1 (hint Khang) + abstain câu no-hint; chuyển GĐ4 đóng băng/tài liệu. Không train adapter mới.

## Cập nhật 11/11 — bước nâng PA (CPU)
- Môi trường chấm: 2×RTX 5090 32GB, vLLM, 6 giờ/~500 câu (~43 s/câu), ≤15B tham số. Tốc độ T4 không đại diện.
- Public_test (không gold): hint 824/1000 (82.4%), no-hint 175, hint lỗi 1. Train là 98.6% → public thiếu hint nhiều hơn hẳn.
- Lý do thiếu hint public (đếm): thiếu bộ lọc/tham số 84, không nhận số đo 62, nhiều kỳ 16, guard chặn 10, khác 3.
- Độ bền biến thể (300 câu train): không dấu/viết thường 100% giữ; teencode mất hint 1; "Q03/2025", "quý 03" bị đọc thành THÁNG 3 (lỗi im lặng).
- Sửa ở bản sao khang_v2 (khang gốc đóng băng giữ nguyên): regex quý chấp nhận số 0 đầu; thêm tests/test_quarter_zero_pad.py; 9/9 file test Khang đạt; Tier1 train/cal/audit không đổi đáp án nào so với v1.
- Không viết luật từ nội dung câu public (thể lệ cấm phát triển trên test; AGENTS cấm đưa câu hỏi vào ngữ cảnh).
