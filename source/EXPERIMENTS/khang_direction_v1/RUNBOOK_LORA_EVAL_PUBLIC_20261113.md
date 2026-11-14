# Runbook: đánh giá LoRA rồi chạy public 1000 câu trên GPU thuê — 13/11/2026

Đây là bản thực thi trên GPU thuê của [EVAL_PLAN_LORA_VAST_20261113.md](EVAL_PLAN_LORA_VAST_20261113.md). Tiêu chí quyết định ở mục 5 của file đó **giữ nguyên** và đã khóa trước khi chạy. Mode: `evaluation` cho bước 2–4 và `submission`-dry cho bước 6 (không tự nộp).

## Gói gửi lên máy thuê

- `PRIVATE_DATA/remote_gpu_pkg_v6`: 241 file, 865 MB, PKG_MANIFEST sha `6a17440d…`. Không chứa gold.
  - `code/`: khang_v4 + run.sh (thêm `FEWSHOT`, `BUDGET`, chặn trường hợp vLLM phục vụ nhầm model) + eval_lora.sh (mới) + merge_lora.py.
  - `data_nohint38/`: 38 câu đánh giá TRAIN/calibration.
  - `data_public/`: public chính thức, questions sha `b33ae585…` (giống public v1). Bản `khang_v4/data/public_test` có sha khác (`50e154a2…`) nên **không dùng**.
  - `adapters/checkpoint-{20,40,85,127}/`: mỗi checkpoint chỉ 2 file inference, sha khớp FILES_SHA256. Không mang optimizer, rng hay pickle.
- Kiểm trên CPU (13/11): `bash -n` cả 4 script; khớp khóa adapter → base index pinned (`26d3539b…`) cho 4 checkpoint: 248/248 module, 0 không khớp. Test merge giả lập cần torch, chạy ở bước 1 trên máy thuê.
- Bản cũ: `PRIVATE_DATA/khang_direction_v1/run.sh.before_lora_eval_20261113`, `build_pkg.before_lora_eval_20261113.py`.

## Bước hint v4 trên public (CPU, 13/11, đã chạy)

- `PRIVATE_DATA/khang_direction_v1/public_hint_v4_cpu_20261113` (server giả, không có LLM): 1000/1000 câu trong 222 s.
- Hint v4 trả lời **824 câu, giống hệt hint v1 từng ID và từng đáp án** (0 câu đổi, 0 thêm, 0 mất). Phần hint trên public vẫn là +760 (792 đúng / 32 sai, tính từ điểm hint-only 0,760). Phần sửa v4 không ảnh hưởng public.
- Vì vậy mọi thay đổi điểm chỉ đến từ **176 câu đi đường LLM** (v1: 159 trả lời, ước tính 116 đúng / 43 sai, net +73).

## Yêu cầu máy

- Tổng VRAM ≥ 32 GB (1×≥32 GB, hoặc 2×16–24 GB chạy TP2). Đĩa trống ≥ 80 GB (base 19 GB + bản gộp 19 GB + dư). RAM ≥ 48 GB để gộp.
- Image `vllm/vllm-openai:v0.30.0` (đã chạy được trên Vast ngày 12/11).
- Người dùng cung cấp: host, port SSH. Key dùng `PRIVATE_DATA/SSH_VAST/vast_ed25519`.

## Thứ tự chạy (agent tự làm, trừ nộp bài)

| # | Lệnh trên máy thuê | Ước lượng | Ghi chú |
|---|---|---|---|
| 0 | scp gói v6 → `/workspace/pkg` | 5–15 phút | |
| 1 | `bash code/setup.sh` + test merge giả lập | 15–30 phút | tải base từ HF theo revision pin, kiểm hash 4 shard |
| 2 | `bash code/run.sh base_fs38 data_nohint38` | ~20–40 phút | **mốc trên cùng máy** (base + few-shot), vì bf16 khác T4 fp16 |
| 3 | `bash code/eval_lora.sh checkpoint-40 lora40_38` | ~30–50 phút | gộp + vLLM + 38 câu, không few-shot |
| 4 | `bash code/eval_lora.sh checkpoint-127 lora127_38` | ~30–50 phút | |
| 5 | (local) tải tgz, kiểm sha, chấm bằng `analysis/error_analysis.py` | 5 phút | gold chỉ ở máy local |
| 5b | (có điều kiện) checkpoint-20 hoặc checkpoint-85; xác nhận seed 2 cho ứng viên đạt | 30–90 phút | chỉ khi đợt 1 cho thấy xu hướng hoặc có ứng viên đạt |
| 6 | public: base → `bash code/run.sh public1000_v2 data_public`; nếu LoRA đạt → `MODEL_DIR=… FEWSHOT=0 bash code/run.sh public1000_lora data_public` | 1,5–3 giờ | hint v4 tất định + LLM cho câu còn lại |
| 7 | (local) tải, kiểm sha, kiểm 1000 dòng / không trùng ID, lập file nộp | 10 phút | **không tự nộp**; người dùng nộp |
| 8 | dừng vLLM; báo người dùng Destroy instance | — | agent không giữ GPU chạy rỗi |

Tổng thời gian ước tính: **4–6 giờ máy thuê**. Chưa đo trên loại GPU sẽ thuê; con số thật lấy từ PROGRESS.tsv của bước 2.

## Chọn bản public

- Mặc định: **base + few-shot k3 + hint v4**.
- Chỉ dùng LoRA khi LoRA đạt cả ba tiêu chí (so với mốc của bước 2 trên **cùng máy**) và qua xác nhận seed 2.
- Nếu public chạy trên máy thuê bf16 thay vì Kaggle T4: kết quả vẫn hợp lệ cho public, nhưng ghi rõ môi trường trong RUN_REGISTRY. Môi trường chấm private của BTC phải tái lập theo Docker riêng.
