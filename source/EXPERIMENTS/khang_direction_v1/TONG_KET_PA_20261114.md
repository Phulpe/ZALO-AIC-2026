# Tổng kết: những việc có ích cho việc nâng PA — đến 14/11/2026 ~01:30 (giờ Việt Nam)

Mọi dữ liệu nằm trong D:\ZAIC. Bằng chứng chi tiết ở RUN_REGISTRY.csv, training.md và PRIVATE_DATA/khang_direction_v1/vast4080s_20261113.

## Điểm public chính thức (người dùng nộp)

| Bản | Nội dung | Public |
|---|---|---|
| hint-only (11/11) | chỉ luật hint Khang | 0,760 |
| v1 (11/11) | hint + Qwen3.5-9B base trên Kaggle T4 (hầu hết câu chỉ chạy 1 lần) | 0,833 |
| v2 (13/11) | hint v4 + base + **few-shot k3 + 3 mẫu biểu quyết** (Vast 2×4080S) | 0,879 |
| **v2b (14/11)** | v2 + **kiểm chéo hint** (2 câu hint bị LLM phản đối mạnh → bỏ trống) | **0,881** |
| v2c (14/11) | v2b + kiểm chéo thêm 208 câu hint: 184 phiếu mạnh, 6 câu khác hint → bỏ trống (result/public_v2c_crosscheck_rest_20261114, sha 5916c459…) | 0,875 — 6 câu bỏ trống đều là hint đúng → không dùng |

PA = (đúng − sai)/1000. Phần hint (824 câu) cố định khoảng +760 (792/32). Mức tăng đến từ phần LLM (176 câu): +73 → +119.

## Việc có ích (đã chứng minh bằng số liệu)

1. **Few-shot k3 từ ngân hàng TRAIN** (1791 ví dụ có code đúng gold TRAIN, không dùng calibration/public). Trên 38 câu đánh giá: net +22 → +28. Trên public: phần LLM đúng ~73% → ~87%.
2. **Chạy đủ 3 mẫu và biểu quyết (min-votes 2)** trên GPU đủ mạnh. Trên Kaggle T4 không làm được do quá chậm. Đây là nguyên nhân lớn thứ hai của +0,046.
3. **Kiểm chéo hint bằng LLM** (giấu dòng Code gợi ý, LLM tự giải; chỉ bỏ trống khi mọi lần chạy đồng ý và khác hint). Hiệu chỉnh trên TRAIN: 0/37 câu phản đối nhầm. Trên public: đúng 2/2 (v2b +0,002). Đây là cách hợp lệ duy nhất đã thấy để giảm số câu hint sai mà không cần gold public.
4. **Hạ tầng GPU thuê tái lập được**: gói remote_gpu_pkg_v6 (manifest + sha), setup.sh kiểm hash base 4 shard, run.sh tự chọn profile, eval_lora.sh, run_hintsoff.py. Một lượt public 1000 câu mất 38 phút trên 2×4080S (Kaggle T4 mất ~5,4 giờ).
5. **Phân tích rủi ro hint không cần gold** (public_hint_risk.py): đo độ bền theo các biến thể giữ nguyên nghĩa, và độ hiếm của khuôn code trên TRAIN. Dùng để xếp thứ tự kiểm chéo.

## Việc đã thử nhưng không giúp (giữ làm bài học)

- **LoRA SFT 337 mẫu** (khang_lora_vast4080_20261112): ck40 net +13, ck127 net +14, so với mốc +28. Overfit (loss ~0 từ epoch 1). **Không nên huấn luyện tiếp từ bộ SFT này.** Muốn LoRA có ích cần nhiều dữ liệu và đa dạng hơn, có tập kiểm tra riêng, và dừng sớm theo đánh giá chứ không theo loss.
- **Hint v4** (sửa độ bền gõ không dấu và "phần trăm"): trên TRAIN không đổi; trên public cho đáp án giống hệt v1. Vô hại nhưng không tăng điểm.

## Còn lại để nâng PA (theo thứ tự hiệu quả dự kiến)

1. **Kiểm chéo nốt các câu hint chưa kiểm** (~500 câu khuôn phổ biến; khoảng 1 giờ 40 trên 2×4080S). Ước tính còn ~25–30 câu hint sai.
2. **Chạy lại các câu rủi ro chưa có phiếu đồng thuận** (76/114), có thể với nhiều mẫu hơn.
3. **Phần LLM**: 21 câu sai, 15 câu bỏ trống. Thử thêm mẫu (5 thay vì 3) hoặc chế độ thinking cho các câu bỏ trống, đo trên 38 câu TRAIN trước.
4. Không chọn bản nộp bằng cách chạy lại nhiều lần rồi lấy điểm public cao nhất (khớp theo leaderboard, không phải cải thiện thật).

## Bài học vận hành

- Không `pip -U huggingface_hub` trong image vLLM 0.30.0: bản 2.4 làm hỏng transformers 5.17 (dùng <2.0).
- `pkill -f` qua ssh phải dùng mẫu "[x]yz", nếu không sẽ kill luôn chính phiên ssh.
- V100 (Volta) không chạy được image vLLM CUDA 13; cần GPU từ Ampere trở lên.
- Phép kiểm chéo dùng chung server với lượt khác thì chậm khoảng 3–4 lần; nên chạy riêng.

## Cập nhật 14/11 17:30: v2c = 0,875 (kém v2b 6 câu)
- LLM phản đối mạnh ở khuôn phổ biến (gặp 3–60 lần trên TRAIN) thì sai 6/6. Hai lần đúng ở v2b đều có tín hiệu phía hint (khuôn chưa gặp trên TRAIN, hoặc đáp án đổi khi viết lại câu).
- Quy tắc sửa: chỉ bỏ trống hint khi LLM phản đối mạnh **và** hint có dấu hiệu bất ổn. Không mở rộng kiểm chéo sang khuôn phổ biến. Bản tốt nhất vẫn là v2b 0,881.
