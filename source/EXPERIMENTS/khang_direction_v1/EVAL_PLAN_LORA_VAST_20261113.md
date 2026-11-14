# Kế hoạch đánh giá LoRA `khang_lora_vast4080_20261112` — 13/11/2026 (giờ Việt Nam)

Trạng thái: **KẾ HOẠCH, chưa thực thi**. Chưa có code mới, chưa chạy GPU. Mode = `evaluation` (không cập nhật trọng số, gold chỉ dùng để chấm sau khi khóa dự đoán).

## 1. Đối tượng và mốc so sánh

- Adapter: `PRIVATE_DATA/khang_direction_v1/lora_vast4080_20261112` (final sha `3bcbd545…` = checkpoint-127; 26 checkpoint bước 5…125 và 127).
- Định dạng huấn luyện: prompt runner **không có dòng "Code gợi ý" và không có few-shot** (build_teacher_sft.py). Vì vậy đánh giá chính dùng LoRA **không few-shot**.
- Tập đánh giá: 38 câu no-hint TRAIN/calibration, `nohint_ids.txt` sha `854ceceb…`. Đã kiểm không trùng SFT (RUN_CONFIG: "no eval38"; kernel plan `lora` có `assert not (sft_q & nh)`).
- Mốc so sánh cùng 38 câu, cùng Kaggle 2×T4 fp16, cùng `eval_v5_args` (samples 3, min-votes 2, no-thinking, temp 0.6, max-tokens 2048, 8 workers, kernel-workers 2):
  - base no-hint v5: 26 đúng / 4 sai / 8 bỏ (net +22)
  - **base + few-shot k3 v7: 31 đúng / 3 sai / 4 bỏ (net +28)** ← mốc phải vượt (cấu hình đang dùng cho public v2)

## 2. Chọn checkpoint (không mặc định lấy final)

Đường loss (train.log): khoảng 42 bước/epoch; loss xuống ~0,02 ở bước ~35 và ~0,001 cuối epoch 1. Ở epoch 2,33 (khoảng bước 98) có một đỉnh loss 4,74, sau đó trở về ~0,0003. Từ epoch 1 trở đi gần như học thuộc 337 mẫu, nên nguy cơ overfit cao.

| Đợt | Checkpoint | Lý do |
|---|---|---|
| 1 | **checkpoint-40** (~1 epoch) | điểm loss vừa chạm sàn, ít overfit nhất mà vẫn học đủ định dạng |
| 1 | **checkpoint-127 (final)** | để đo xem học thêm 2 epoch có giúp hay hại |
| 2 (có điều kiện) | checkpoint-20 hoặc checkpoint-85 | chỉ chạy nếu đợt 1 cho thấy xu hướng: ck40 > final thì đo ck20; final > ck40 thì đo ck85 |

Chỉ dùng `adapter_model.safetensors` + `adapter_config.json`. Không nạp optimizer.pt/rng_state.pth/training_args.bin (pickle, không cần cho inference).

## 3. Cách chạy (Kaggle, sau reset quota 14/11 00:00 UTC = 07:00 VN)

**Tại sao Kaggle:** mốc 31/3/4 đo trên T4 fp16. Chạy cùng môi trường thì so sánh mới sạch, và không tốn tiền thuê máy. vLLM serve LoRA trực tiếp trên các module linear-attention (`in_proj_qkv/a/b/z`) của Qwen3.5 chưa được chứng minh, nên dùng **merge** đã có test (`bridge_v3/merge_lora.py` + `test_merge_lora.py`).

Các bước (CPU trước, GPU sau cùng):

1. **CPU, máy local:** đóng gói dataset Kaggle private mới `zaic-khang-lora-vast4080-adapters-20261112`, gồm ck40 và ck127, mỗi cái 2 file (~173 MB/cái), kèm `ADAPTERS_MANIFEST.json` (sha256 từng file, đối chiếu `checkpoint-N.FILES_SHA256.txt`). Upload bằng CLI, tải lại để kiểm hash.
2. **CPU, code:** tạo `kaggle/push_lora_eval/` (bản sao `lora_kernel.py`, không sửa bản cũ) và thêm plan `lora_eval`:
   - Kiểm sha base, nohint_ids, manifest adapter, SFT (để assert không leak).
   - Lặp qua từng candidate: merge vào `/kaggle/temp/merged_<ck>` → vLLM (`start_server_v2`, không spec) → `run_live("eval38_<ck>", …, eval_v5_args)` → tắt server → **xóa thư mục merged** trước candidate sau (đĩa temp không đủ chứa 2×19 GB).
   - Ghi `PROGRESS.tsv` và log tiến độ từng câu ra stdout. Mỗi candidate có hạn giờ riêng; hết giờ thì ghi trạng thái và chuyển sang candidate kế.
   - Test giả lập: thêm `test_lora_eval_plan.py` (manifest sai thì dừng, leak thì dừng, thứ tự xóa merged) và chạy cùng bộ test khang hiện có.
3. **CPU Kaggle (Accelerator=None):** plan `lora_preflight` mở rộng: kiểm input mount, sha adapter, chạy test merge giả lập. Không tốn quota GPU.
4. **GPU Kaggle (2×T4):** một kernel `lora_eval` cho đợt 1. Bật GPU ngay trước khi push. Kernel tự kết thúc. Sau đó kiểm trạng thái bằng CLI và xác nhận không còn phiên GPU nào.
5. **CPU local:** tải output về, kiểm hash, chấm bằng `analysis/error_analysis.py` (gold TRAIN/calibration, sau khi khóa dự đoán). Ghi `ERROR_ANALYSIS_LORA_<ck>.jsonl` và so từng câu với v7.

## 4. Ngân sách GPU (ước lượng, chưa đo)

- Mỗi candidate: merge ~10–15 phút + nạp vLLM ~5 phút + eval38 ~90 phút (v7 mất 5276 s) ≈ **1 giờ 50**.
- Đợt 1 (2 candidate) ≈ **3,7 giờ**, đặt giới hạn kernel 4,5 giờ. Đợt 2 thêm ≈ 1,9 giờ.
- Quota tuần mới 30 giờ. Lượt public v2 lần trước tốn ~5,4 giờ (19528 s). Đợt 1 + đợt 2 + public v2 ≈ 11 giờ, còn dư nhiều.
- Dừng và hỏi lại nếu dự kiến vượt 6 giờ cho phần đánh giá này.

**Phương án thay thế (Vast, có trả tiền):** thuê lại 2×4080 hoặc 2×4000 Ada, merge rồi chạy vLLM bf16 TP2. Nhanh hơn nhiều, đánh giá được 4 checkpoint trong khoảng 2 giờ. Nhưng phải **chạy lại mốc base+few-shot trên cùng máy**, vì bf16 trên máy khác không so trực tiếp được với 31/3/4 trên T4. Chỉ chọn khi bạn muốn có kết quả trước 07:00 ngày 14/11.

## 5. Tiêu chí quyết định (khóa trước khi chạy)

Một checkpoint LoRA chỉ được xem là **ứng viên thay đường LLM** khi đạt cả ba điều:
1. net (đúng − sai) **≥ +31**, tức hơn mốc v7 ít nhất 3. Với 38 câu, chênh 1–2 câu nằm trong nhiễu.
2. số sai **≤ 3** (PA phạt sai, độ chính xác khi trả lời phải giữ ≥ ~90%).
3. số câu đúng ở v7 mà LoRA làm sai **≤ 1**, và không có lỗi hệ thống mới (timeout/OOM/plan lỗi tăng).

Kết quả:
- Đạt: chạy **xác nhận lần 2** (khác seed) trên cùng 38 câu cho checkpoint đó. Có thể thử thêm biến thể LoRA + few-shot. Chỉ sau đó mới đề xuất đưa vào public v2, và việc này cần bạn duyệt.
- Không đạt: giữ base + few-shot k3 cho public v2. Adapter vẫn lưu làm tài sản, ghi RUN_REGISTRY là "evaluated, not promoted".
- BEST_MODEL không đổi chỉ vì có kết quả trên 38 câu. LATEST_CHECKPOINT = ck127 (đã ghi).

## 6. Ghi chép sau khi chạy

RUN_REGISTRY gồm: run_id `khang_lora_eval_20261114`, parent `khang_lora_vast4080_20261112`, sha từng adapter và từng merged index, số đúng/sai/bỏ/thời gian từng candidate, quota trước/sau. Cập nhật thêm training.md (mục mới, giữ lịch sử), GPU_BUDGET.json và RESUME_STATE.

## 7. Phần cần bạn quyết định

- Chọn Kaggle (miễn phí, chờ đến 07:00 14/11) hay Vast (trả tiền, chạy ngay).
- Có cho phép tôi làm ngay các bước CPU 1–3 (upload adapter lên dataset Kaggle private, viết code, test, preflight CPU) không. Các bước này không tốn GPU.
