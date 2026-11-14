# Trạng thái tạm dừng — 11/11/2026 khoảng 23h30 giờ Việt Nam (16:30 UTC)

Người dùng yêu cầu tạm dừng và sao lưu để tiếp tục sau. File này ghi đủ thông tin để tiếp tục mà không phải dò lại.

## Đã hoàn tất, có bằng chứng
| Hạng mục | Bằng chứng |
|---|---|
| Public v1: 0,833. Bản chỉ hint: 0,760 (hint đúng 792, sai 32; vLLM đúng 116, sai 43) | VLLM_V5_RESULT_20261111.md; RUN_REGISTRY.csv |
| Kế hoạch v2 | PLAN_SPEED_HINT_PA_20261111.md (có mục "Cập nhật thực hiện") |
| Runner khang_v3: log PROG/SUMMARY, PROGRESS.tsv, --fixed-samples, --parallel-first, --question-timeout | khang_v3/src/datamate/runner.py; tests/test_runner_v3_progress.py |
| Bộ test khang_v3 đạt 10/10 | TEST_KHANG_V3_WSL.txt |
| Kiểm độ bền hint: 600 câu × 10 biến thể, 0 đáp án sai | PRIVATE_DATA/khang_direction_v1/HINT_ROBUSTNESS_V2.jsonl; hint_robustness_v2.py |
| Gói code v2 (115 file, manifest sha 11b9fc874caeff436c365fc8f62f1281a6c78e08b10466a4ba49356efbb13a09), đã upload thành bản mới của dataset lnghongphc/zaic-khang-direction-code-v1 (ready) | PRIVATE_DATA/khang_direction_v1/code_upload_v2 |
| Kernel v2 (sha d60a790f…c4ef) | kaggle/push_v2/v2_kernel.py, kernel-metadata.json |
| Ủy quyền GPU v2: thêm 330 phút; hỏi lại nếu dự kiến vượt 400 phút | GPU_BUDGET.json → v2_authorization |

## Đang chạy lúc tạm dừng
- Kernel **lnghongphc/zaic-khang-v2-20261111 version 1**, plan v2_bench, run_id khang_dir_v2_bench_20261111.
- Đẩy lên lúc 14:58 UTC. Giới hạn cứng 7200 s, tự dừng muộn nhất khoảng 17:00 UTC (00h00–01h00 giờ Việt Nam).
- Đây là batch run. Khi chạy xong, GPU tự giải phóng; không cần máy cục bộ bật.
- Tác vụ theo dõi cục bộ đã dừng. Lần sau cần tự kiểm trạng thái.

## Bước tiếp theo khi tiếp tục
1. Kiểm trạng thái kernel: `kernels status lnghongphc/zaic-khang-v2-20261111`, rồi tải output về **PRIVATE_DATA/khang_direction_v1/v2_bench** (`kernels output … -p …`). Kiểm ARTIFACT_MANIFEST và xác nhận COMPLETE (GPU đã dừng).
2. Ghi GPU thực dùng vào GPU_BUDGET.json (runs) và kiểm quota.
3. Đọc STATUS.json, các bước cần xem:
   - vllm_install, vllm_args_check (có cờ speculative không);
   - server (vllm_mtp ready? nếu không: server_fail_tail rồi vllm);
   - runner bench_A và bench_B;
   - bench_throughput (gen_tok_per_s, spec accepted/drafted).
4. Chấm điểm (chỉ trên CPU, gold TRAIN/calibration, sau khi đã khóa dự đoán):
   - `python EXPERIMENTS/khang_direction_v1/vllm_v5 style score`: dùng scratchpad score_sub.py hoặc analysis/error_analysis.py với logs của run/bench_A/logs và run/bench_B/logs.
   - So với v5 (26 đúng / 4 sai / 8 bỏ qua).
   - Tính độ ổn định A và B: chênh lệch PA trên tập con, số ID có đáp án khác nhau.
5. Ước tính thời gian public: 176 câu × (token/câu của bench) ÷ gen_tok_per_s ở tải cao. Mục tiêu tổng ≤ 10.800 s.
   - Đạt: đặt CONFIG plan="v2_public", run_id mới, public_test=True, max_gpu_seconds=10800, phase public khoảng 8400. Kiểm lại hàm build_dm/public. Đẩy kernel. Theo dõi tab Logs.
   - Không đạt: chỉnh cấu hình (bỏ MTP nếu chậm hơn, giảm workers, cân nhắc AWQ) và báo người dùng trước.
6. Đóng băng v2 (manifest + hash) trước khi chạy public. Cập nhật RUN_REGISTRY.csv và training.md.
7. Không tự nộp. Người dùng tự nộp submission_public.jsonl.

## Chưa làm hoặc đã hoãn (có lý do)
- C1, LLM kiểm chéo hint: bỏ (lý do trong PLAN, mục Cập nhật).
- C3, mở rộng độ phủ hint: chưa làm, rủi ro tạo câu sai.
- Audit: cần người dùng duyệt upload.
- Thinking mode: chỉ cân nhắc trên 5090.
- docker_qwen35: bản nháp, chưa thử trên 5090, chưa cập nhật sang khang_v3.
