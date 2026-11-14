# Trạng thái tạm dừng — 14/11/2026 ~00:35 (giờ Việt Nam)

Thay cho RESUME_STATE_20261112.md. Chi tiết: training.md mục "Đánh giá LoRA, public v2 và kiểm chéo hint", RUN_REGISTRY (3 dòng ngày 13/11).

- Public v2 (base_fewshot) = **0,879** (nộp 13/11 23:08). v2b (kiểm chéo) = **0,881** (nộp 14/11 00:34) — tốt nhất hiện tại. File:
  - D:/ZAIC/result/public_v2b_crosscheck_20261113/submission.jsonl (sha a1f785bd…): khuyến nghị theo quy tắc đã khóa. Giống bản A, chỉ đổi 2 câu hint thành abstain.
  - D:/ZAIC/result/public_v2_base_fewshot_20261113/submission.jsonl (sha 77408cbf…): bản A, base + few-shot k3 + hint v4.
- LoRA khang_lora_vast4080_20261112: ck40 net +13, ck127 net +14, mốc +28 → không dùng.
- Máy Vast 2×4080S (<REDACTED_IP>:40453): vLLM đã dừng, GPU 0%. Người dùng cần Destroy instance trên trang Vast.
- Bài học: không `pip -U huggingface_hub` trong image vLLM 0.30.0 (2.4 làm hỏng transformers; dùng <2.0); `pkill -f` phải dùng mẫu kiểu "[x]yz" để không tự kill phiên ssh.
- v2c (kiểm chéo mở rộng, 6 câu hint thêm bỏ trống): result/public_v2c_crosscheck_rest_20261114, chưa nộp. Tổng kết: TONG_KET_PA_20261114.md. Sao lưu máy thuê: vast4080s_20261113/remote_backup. Máy Vast: vLLM đã dừng; người dùng Destroy.
- v2c nộp = 0,875 (kém v2b): kiểm chéo khuôn phổ biến sai 6/6 → bỏ hướng mở rộng. BEST = v2b 0,881.
- Việc tiếp theo: cải thiện phần LLM (21 sai, 15 bỏ) trên 38 câu TRAIN bằng quota Kaggle 30 giờ đã reset.
- 14/11 18:09: GĐ0 xong (training.md mục "GĐ0 kế hoạch cải thiện Kaggle"). Sẵn sàng GĐ1: đổi improve_kernel.py preflight_cpu=False, kernel-metadata enable_gpu=true, push (plan improve_collect, ~9,5 h GPU, 75 câu dev × 5 mẫu, max-tokens 4096, ẩn hint). Chờ người dùng duyệt.
