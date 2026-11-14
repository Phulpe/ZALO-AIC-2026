# ZAIC 2026 DataMate — bàn giao để huấn luyện/cải thiện tiếp (14/11/2026)

Base: **Qwen/Qwen3.5-9B**, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Hướng hiện tại là "hướng Khang": luật hint tất định, LLM tự viết code và biểu quyết nhiều mẫu.

> Repo này **chỉ chứa mã và tài liệu**. Không có dữ liệu thi, đáp án, trọng số hay token. Hãy để repo ở chế độ **Private** và chỉ mời thành viên trong đội. Dữ liệu lấy từ challenge.zalo.ai và phải tuân thủ thể lệ cuộc thi.

## 1. Hiện trạng điểm

| Bản | Nội dung | Public |
|---|---|---|
| hint-only | chỉ luật hint | 0,760 |
| v1 | hint + Qwen3.5-9B base (Kaggle T4) | 0,833 |
| v2 | hint v4 + few-shot k3 + 3 mẫu biểu quyết (min-votes 2) | 0,879 |
| **v2b — tốt nhất** | v2 + kiểm chéo hint (2 câu hint bị LLM phản đối mạnh thì bỏ trống) | **0,881** |
| v2c | kiểm chéo mở rộng sang các khuôn phổ biến | 0,875 (kém hơn, đã bỏ) |

PA = (đúng − sai)/1000.
- Phần hint (824 câu) cố định ở khoảng +760.
- Đòn bẩy còn lại là phần LLM (176 câu): khoảng 140 đúng, 21 sai, 15 bỏ trống.

Đọc chi tiết theo thứ tự:
1. `source/EXPERIMENTS/khang_direction_v1/TONG_KET_PA_20261114.md`: việc có ích, việc thất bại và bài học.
2. `source/EXPERIMENTS/khang_direction_v1/PLAN_CAI_THIEN_KAGGLE_20261114.md`: kế hoạch GĐ0–4 đang làm.
3. `source/training.md`: đặc tả và nhật ký quyết định huấn luyện.
4. `source/AGENTS.md`: quy tắc bắt buộc về bảo toàn trọng số, ranh giới dữ liệu và quota GPU.
5. `source/RUN_REGISTRY.csv`: mọi lượt chạy, có hash và cấu hình.

## 2. Mã chính

| Đường dẫn (`source/EXPERIMENTS/khang_direction_v1/…`) | Vai trò |
|---|---|
| `khang_v5/` | **Pipeline hiện tại.** `src/datamate`: runner, hints, fewshot, kernel, llm, answers. v5 = v4 + cờ `--all-samples` (ghi đủ mọi mẫu để mô phỏng offline). |
| `khang_v4/` | Bản đã tạo submission v2/v2b, cùng `scripts/` (build image, eval, check_*). |
| `analysis/` | `simulate_votes.py` (mô phỏng số mẫu/min-votes/max-tokens từ log), `build_proxy.py` (tập dev/holdout từ TRAIN), `error_analysis.py`, `public_hint_risk.py`, `build_crosscheck_sets.py`, `apply_crosscheck.py`, `score_eval38.py`. |
| `bridge_v3/` | `build_fewshot_bank.py` (ngân hàng few-shot k3 từ TRAIN learning), `train_lora_v4.py`, `merge_lora.py`. |
| `kaggle/push_improve/` | Kernel Kaggle GĐ1 (`improve_kernel.py`, plan `improve_collect`), đang chạy ngày 14/11. |
| `kaggle/push_v2/`, `push_lora/`, `push_public/` … | Các kernel Kaggle trước đó. |
| `remote_5090/` | Chạy trên GPU thuê (Vast): `setup.sh`, `run.sh`, `eval_lora.sh`, `run_hintsoff.py`, `build_pkg.py`. |
| `docker_qwen35_v4/`, `khang_v5/docker/` | Docker nộp bài (vLLM 0.30.0). |
| `RUNBOOK_LORA_EVAL_PUBLIC_20261113.md` | Runbook chạy trên GPU thuê, từng bước, có ước lượng thời gian. |

Repo gốc của Khang (v4/v5 ban đầu): https://github.com/Vuduykhang2306/zaic2026-datamate

Các thư mục khác trong `source/` là lịch sử (relearning, correction, time_learning, gói Codex). Giữ lại để truy vết, không phải entrypoint.

## 3. Tài sản không có trong repo và cách lấy

| Tài sản | Lấy ở đâu |
|---|---|
| Dữ liệu train/public (questions, answers, data cửa hàng, calendar, rulebook) | Tải từ challenge.zalo.ai, đặt vào `khang_v5/data/train` và `khang_v5/data/public_test`. Public chính thức có questions sha `b33ae585…`. |
| Base Qwen3.5-9B | HuggingFace `Qwen/Qwen3.5-9B` @ `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. `remote_5090/setup.sh` tải về và kiểm hash 4 shard. |
| Ngân hàng few-shot k3 (1791 ví dụ, sha `f74eb15e…`) | Sinh lại bằng `bridge_v3/build_fewshot_bank.py` từ TRAIN learning, hoặc xin chia sẻ Kaggle dataset `zaic-fewshot-bank-v1-20261112`. |
| Tập proxy dev/holdout (75/73 ID TRAIN) | Sinh lại bằng `analysis/build_proxy.py` (seed cố định 20261114, `PROXY_SPLITS=train,calibration`). Dev sha `05ad9be0…`. |
| Kaggle datasets private (code, runtime, base, wheels vLLM/bnb) | Chủ tài khoản Kaggle bấm **Share** cho tài khoản đồng đội. Danh sách nằm trong `kaggle/push_improve/kernel-metadata.json`. |
| Adapter LoRA cũ (9B, 4B) | Chỉ ở máy chủ dự án. **LoRA SFT 337 mẫu bị overfit, đã loại.** Không cần để làm tiếp hướng hiện tại. |
| Submission v2/v2b/v2c | Chỉ ở máy chủ dự án (có đáp án public dự đoán, không đưa lên git). |

## 4. Chạy thử nhanh

Yêu cầu: Python 3.12, `pip install pandas==2.2.3 openpyxl==3.1.5`. Dữ liệu train phải nằm ở `khang_v5/data/train`.

```bash
cd source/EXPERIMENTS/khang_direction_v1/khang_v5
PYTHONPATH=src python tests/test_runner_all_samples.py   # test runner bằng server LLM giả
PYTHONPATH=src python -m datamate.runner --help
```

Một số test cần dữ liệu thật. File `test_kernel_api.py` bị loại khỏi repo vì chứa danh sách ID câu hỏi.

Chạy LLM thật: dựng vLLM 0.30.0 phục vụ Qwen3.5-9B (xem `remote_5090/run.sh`), rồi gọi runner với `--base-urls http://127.0.0.1:8000 --samples 3 --min-votes 2 --fixed-samples` cùng few-shot. Cấu hình đầy đủ nằm trong `remote_5090/run.sh` và CONFIG của `kaggle/push_improve/improve_kernel.py`.

## 5. Việc tiếp theo (theo kế hoạch đã thống nhất)

- **GĐ1** (đang chạy trên Kaggle, 14/11): 75 câu dev × 5 mẫu, max-tokens 4096, ghi đủ mẫu và trajectory. Khi xong, dùng `analysis/simulate_votes.py` mô phỏng 3/5 mẫu, min-votes 2/3 và cap 2048 so với 4096.
- **GĐ2**: bật thinking chỉ cho câu khó (phiếu chia rẽ hoặc bỏ trống).
- **GĐ3**: sửa code/prompt theo phân tích lỗi (làm tròn, đơn vị, câu không trả lời được).
- **GĐ4**: xác nhận trên holdout. Tiêu chí: trên dev, net +3 trở lên và số sai không tăng; trên holdout, không kém mốc.
- Public 1000 câu chỉ chạy trên GPU thuê, sau khi cấu hình đã khóa. Người nộp bài là trưởng nhóm.

## 6. Quy tắc bắt buộc (tóm tắt `source/AGENTS.md`)

- Chỉ dùng TRAIN để học và chọn cấu hình. Không dùng gold public/private, không sửa đáp án theo từng ID.
- Không đưa câu hỏi, gold hay dữ liệu cửa hàng lên git, chat hoặc dịch vụ LLM bên ngoài.
- Mỗi lượt chạy có `run_id` riêng và được ghi vào RUN_REGISTRY (hash code/data/model, cấu hình, kết quả).
- Kaggle GPU: chỉ bật khi code/input đã sẵn sàng, tắt ngay khi chạy xong.
- Không xóa hay ghi đè checkpoint/trọng số. Mọi thử nghiệm làm trên bản sao hoặc nhánh mới.

## 7. Về bản xuất này

- Tạo bằng `tools/build_export.py`, chỉ đọc từ thư mục gốc `D:/ZAIC`. Đường dẫn gốc được giữ nguyên dưới `source/`.
- Đã che:
  - văn bản câu hỏi thật dài từ 25 ký tự trở lên, thay bằng `<REDACTED_COMPETITION_QUESTION>` (405 chỗ trong 60 file);
  - 1 địa chỉ IP máy thuê.
- Đã loại:
  - file nhị phân, log, dữ liệu, trọng số và file trên 1,5 MB;
  - 9 file chứa từ 10 ID câu hỏi trở lên;
  - outputs của notebook.
- `EXPORT_MANIFEST.json` ghi sha gốc và sha bản xuất cho từng file. Kiểm bằng `python tools/verify_export.py [thư_mục_ZAIC_gốc]`.
- Một số test/tài liệu có câu hỏi bị che nên không chạy nguyên trạng. Cần dữ liệu thật đặt vào `data/`.
