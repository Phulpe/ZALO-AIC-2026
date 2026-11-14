# Trạng thái tạm dừng — 12/11/2026 (giờ Việt Nam), người dùng tắt máy

## Đã xong (có bằng chứng)
- Few-shot eval v7 (kernel zaic-khang-lora-teacher-20261112 v7): 38 câu no-hint TRAIN/cal 31/3/4, net +28 (base v5 +22). Output: PRIVATE_DATA/khang_direction_v1/fewshot_eval_v7; ERROR_ANALYSIS_FEWSHOT_V7.jsonl.
- Xác nhận v8 (16 workers + fixed-samples + few-shot): 20/2/16, net +18; 13 abstain do deadline (16 workers chậm hơn trên T4). Output: fewshot_confirm_v8; ERROR_ANALYSIS_FEWSHOT_V8.jsonl.
- Quyết định: dùng `--fewshot --fewshot-k 3`; T4 dùng 8 workers / kernel-workers 2. Đã ghi RUN_REGISTRY.csv, GPU_BUDGET.json, training.md (bản trước: PRIVATE_DATA/khang_direction_v1/*.before_20261112.*).
- kaggle/push_v2/v2_kernel.py đã bật few-shot + 8 workers + dataset bank (bản cũ v2_kernel.before_fewshot_20261112.py). CHƯA push, public_test=False.
- Docker 5090 bản nháp mới: docker_qwen35_v4/ (fixed-samples, few-shot, progress; Dockerfile kiểm sha bank). Chỉ qua `bash -n`; chưa build/chạy.
- Phân tích tốc độ (log v5/v7/v8): runner đã dừng sớm theo phiếu; few-shot gần như không thêm chi phí (prompt trung vị 43.3k vs 42.9k); nút thắt là decode T4.
- GPU: quota 29.10/30 h, reset 2026-11-14 00:00 UTC. Không có kernel nào đang chạy (v8 COMPLETE).

## Đang dở — mục 1: độ bền hint (CPU, khang_v4 = bản sao khang_v3 chưa sửa)
- Script: hint_robustness_v3.py (19 biến thể; chẩn đoán giai đoạn mất hint). Kết quả lần 1: PRIVATE_DATA/khang_direction_v1/HINT_ROBUSTNESS_V3_base.{jsonl,summary.json}.
  - Mất hint nhỏ: no_punct 10(+1 error), pct_word 8, filler 4, greet_tail 3, khác 1–2; đa số lý do measure_no_code.
  - period_front: 51 lost + 7 changed — NGHI do bộ sinh biến thể sai (tách "tháng 6" khỏi "năm 2025"). Đã sửa period_front trong script (chỉ đổi khi đúng một cụm kỳ đầy đủ, không mốc thời gian khác); lần chạy lại bị ngắt, chưa có kết quả hợp lệ (file HINT_ROBUSTNESS_V3_pf.* nếu có là không đầy đủ).
- Bước tiếp:
  1. Chạy lại period_front (WSL ~/zaic_khang_venv, lệnh như trong RESUME: `python hint_robustness_v3.py <khang_v4> <khang_v4/data/train> PRIVATE_DATA/khang_direction_v1/TIER1_V2_PER_ID.jsonl <out.jsonl> 600 period_front`).
  2. Nếu vẫn có "changed" -> đây là lỗi thật của hint (rủi ro sai trên public/private), ưu tiên sửa.
  3. Sửa _norm/measure trong khang_v4/src/datamate/hints.py cho no_punct / pct_word / filler / greet_tail; chạy lại toàn bộ 19 biến thể + Tier1 TRAIN (độ chính xác hint không giảm, 0 changed) + bộ test khang (run_khang_v3_tests.sh trỏ sang v4).
  4. Upload code dataset mới (khang_v4) + cập nhật v2_kernel khang_src; ghi RUN_REGISTRY/training.md.
- Sau đó (cần người dùng duyệt): public v2 sau reset quota 14/11; build Docker 5090; upload audit. Không tự nộp.

## Cập nhật 12/11/2026 (cuối ngày) — mục "Đang dở" ở trên ĐÃ XONG
- period_front chạy lại với bộ sinh đã sửa; sửa khang_v4 hints.py (accent rule 1 từ, "phan tram"->%); 13/13 test khang_v4 pass (TEST_KHANG_V4_WSL.txt); Tier1 TRAIN v4 = v2 trên 3000/3000 (TIER1_V4_*); kết quả HINT_ROBUSTNESS_V3_{base,pf,sub,v4fix}.
- code_upload_v4 (manifest be86ea5c…, 167 file) = dataset zaic-khang-direction-code-v1 v4.0, tải về kiểm 0 bad. v2_kernel.py khang_src=khang_v4/src. Vẫn CHƯA push, plan=v2_bench, public_test=False.
- Bộ máy thuê 5090: remote_5090/{build_pkg.py,setup.sh,run.sh}; gói PRIVATE_DATA/remote_5090_pkg_v1 (132 file, manifest c1d69318…, không gold, không public). Chỉ qua bash -n.
- Còn lại không cần GPU: build thử Docker v4 trên máy local (Docker Desktop có, chưa bật; cần chép model 19 GB vào khang_v4/models); người dùng kiểm thể lệ chuyển dữ liệu cho bên thứ ba.
- Cần GPU: 5090 (chờ SSH + thể lệ) hoặc Kaggle sau reset 14/11 00:00 UTC; public v2 cần duyệt. Không tự nộp.
- 12/11 tối: người dùng chọn thuê RTX 4090. run.sh tự chọn profile: 5090 bf16 / 2x4090 bf16 TP2 / 1x4090 fp8 (khác số học, chỉ tham khảo). Gói mới: PRIVATE_DATA/remote_gpu_pkg_v2 (manifest cf1f1514…); gói v1 giữ nguyên.
- 12/11 tối: người dùng thuê vast.ai 4x RTX 5060 Ti (host 244103, VN) và xác nhận thể lệ cho phép GPU thuê (PRIVATE_DATA/COMPETITION_PERMISSION_ATTESTATION_20261112.md, user attestation). SSH key: PRIVATE_DATA/SSH_VAST/vast_ed25519 (public key người dùng dán vào Vast). run.sh thêm profile bf16_tp2x2 (2 server TP2, :8000/:8001). Gói: PRIVATE_DATA/remote_gpu_pkg_v3.
- 12/11 ~21:45 VN: vast.ai instance 60365444 (2x RTX 4000 Ada, Romania): setup.sh OK, 4 shard Qwen3.5-9B sha256 KHOP WEIGHTS_INVENTORY (tai tu HF revision pin), vLLM 0.30.0 torch 2.13 cu130 sm_89. Chua chay luot nao (khong co output). Nguoi dung dung de doi GPU re hon. Lan sau: dung lai remote_gpu_pkg_v3 + setup.sh (+ kiem shard nhu tren).

- 2026-11-12T20:05Z: run khang_lora_vast4080_20261112 COMPLETED 127/127, final + 26 checkpoint ở PRIVATE_DATA/khang_direction_v1/lora_vast4080_20261112 (sha khớp). Tiếp theo: Destroy Vast instance; đánh giá (merge + vLLM 38 câu) sau khi người dùng cho phép.
