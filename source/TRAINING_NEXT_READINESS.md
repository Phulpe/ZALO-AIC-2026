# Kiểm tra sẵn sàng lượt huấn luyện kế tiếp — 08/11/2026

## Kết luận

**Hạ tầng warm-start 9B đã qua kiểm thử GPU thực tế và đồng bộ checkpoint smoke. Chưa mở huấn luyện correction/replay thật.** Cổng dữ liệu chưa đạt: probe thực tế mới nhất có bốn câu đúng, bốn replay và **không có correction được xác minh**. Không tạo mẫu sai giả để mở train. Ngân sách pilot là đề xuất cho lượt sau, chưa phải quyền chạy train dài.

Không cần người dùng nhập CLI, cài Kaggle CLI hoặc gửi token. Không cần hỏi lại quyền thao tác tài khoản hoặc lời xác nhận đã cung cấp. Bước tiếp theo có ích là thu thập lỗi model thực sự trên TRAIN và xác minh corrected plan; lỗi môi trường/executor phải sửa ở đúng thành phần trước khi cân nhắc supervision.

## Quy tắc và phạm vi

- Đã đọc lại `D:\ZAIC\AGENTS.md` do người dùng sửa. SHA256 hiện hành: `40530bc05e78ae3bb17bb9f7c407eb5e6744c1b738fa9ca70585870bb9d047f8`.
- Đã dùng skill `D:\ZAIC\.agents\skills\caveman\SKILL.md`. Bản mã mới nằm riêng trong `D:\ZAIC\EXPERIMENTS\relearning_v3`; bản v1/v2 và mọi log lỗi được giữ lại.
- Quyền xử lý tài sản trên Kaggle được chấp nhận theo xác nhận của người dùng, ghi `user_attested`, không đổi thành bằng chứng độc lập của ban tổ chức. Xem `D:\ZAIC\EXPERIMENTS\relearning_v3\PERMISSION_DECISION.json`.
- Không đưa câu hỏi, gold, giao dịch hoặc nội dung SFT vào ngữ cảnh AI. Các kiểm tra dữ liệu chạy bằng chương trình, chỉ trả schema, số lượng, hash và tổng hợp.
- Không huấn luyện dài, không nộp bài, không gộp adapter, không sửa/xóa bản gốc. Full resume chưa triển khai/xác minh; điều đó không ngăn warm-start đã hoạt động.

## Những việc đã hoàn thành

| Hạng mục | Bằng chứng hiện tại | Giới hạn |
|---|---|---|
| Bảo toàn parent | Adapter gốc vẫn 86.623.432 byte; SHA256 `e7dfa635beccce128c99f9d3fd993d33dbb4c88e73e3ddeccbc2bc915eec4a4d` | Không phải kiểm toán lại toàn bộ lịch sử trọng số trong lượt chuẩn bị này |
| Base | Qwen/Qwen3.5-9B, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`; 15 tài sản gồm bốn shard đã đối chiếu | Base giữ riêng, không xuất lại trong checkpoint |
| TRAIN runtime | Kaggle CPU kiểm đủ 120 tệp, 139.114.932 byte theo manifest SHA256 `523db34eb8e8bab52aa52d1d046307f19dbb3668495025ca05d66abd1e4c6e70` | Learning/calibration; không chứa audit gold hoặc public/private test |
| Code v3 | ZIP SHA256 `ed36e2c29ef0628ce3506ba1729254bd9d2371330feed260cb982fa6f5bf1399` | Mã baseline không sửa |
| Kiểm thử Kaggle CPU | **99/99 baseline + 67/67 mới đạt**, không failures/errors | Bằng chứng CPU tách khỏi GPU |
| Dependency | bitsandbytes 0.50.2 từ PyPI, wheel SHA256 `55348a9a4a21bfd99cf8c7b32fe67b4030ae5c2a05738e03c1747f65fa6ec283` | Bootstrap offline có hash pin; không thay PyTorch |
| GPU | T4, CUDA 13.0; NF4 và optimizer 8-bit trên tensor nhỏ đạt | Kaggle cấp hai T4 nhưng tiến trình model chỉ thấy GPU 0 |
| Warm-start thật | Nạp đúng adapter gốc trainable, optimizer/scheduler mới; **một optimizer step trên hai mẫu giả lập** | Không phải resume checkpoint-48 hoặc train correction thật |
| Checkpoint | Lưu adapter, optimizer, scheduler, scaler, RNG, trainer state, tokenizer, config/manifests; nạp lại adapter và forward đạt | Có state không có nghĩa full resume đã chứng minh |
| Đồng bộ | Version 6 tải về máy, kiểm toàn bộ manifest; 126 tệp output có inventory/hash | Tracker chỉ trong phạm vi smoke; không thay BEST_MODEL lịch sử |
| Pilot wrapper | 8/8 tests mới trên máy; notebook AST hợp lệ, mặc định CPU/preflight | Wrapper này chưa chạy GPU; dùng runner v3 đã có GPU smoke |
| Kết thúc | Accelerator=None, hai notebook liên quan đều off; Active Events báo **0 Active Events** | Không hứa tự tắt/sync sau khi agent ngắt |

Môi trường GPU quan sát thực tế: torch `2.11.0+cu130`, transformers `5.20.2`, peft `0.21.2`, accelerate `1.15.0`, bitsandbytes `0.50.2`.

## Kết quả GPU và checkpoint đã bảo toàn

- Run smoke: `20261108T081309663414Z_9aa8d57e`.
- Biên nhận: `D:\ZAIC\EXPERIMENTS\relearning_v3\GPU_SMOKE_RECEIPT.json`.
- Manifest checkpoint: `e0b7ae9510d2acfd6c9b4de9d4f0146a4a80d228dc00077751fba1e3e87323b7`.
- Adapter smoke: SHA256 `e71bca7cadce97638cb6f317cc87f15d5cbced580c8f9a4be3e764320c03895d`.
- Có 21.628.885/21.639.168 phần tử adapter thay đổi; mọi trọng số hữu hạn. Đây là kiểm chứng update kỹ thuật, không phải cải thiện PA.
- Forward sau reload: loss và native loss cùng `0.09091801196336746`, chênh lệch `0.0`; kiểm mask/shift đạt. Loss của fixture không đo chất lượng cuộc thi.
- Bản trên máy: `D:\ZAIC\PRIVATE_DATA\KAGGLE_RUNS\relearning_v3_smoke_366867761`.
- Checkpoint thực nằm trong `output\relearning_model_runs_v2\20261108T081309663414Z_9aa8d57e\checkpoint`. Hậu tố thư mục `v2` được runner kế thừa; **mã chạy là v3**, định danh bằng source hash và run ID, không suy từ tên thư mục.
- `D:\ZAIC\EXPERIMENTS\relearning_v3\SMOKE_TRACKER.json` tách LATEST_CHECKPOINT, LATEST_SYNCED và BEST_MODEL. Hai con trỏ đầu trỏ run smoke đã khớp; BEST_MODEL trong tracker smoke là null. Tracker/model tốt nhất lịch sử giữ nguyên.

## Correction/replay và giới hạn hiện tại

Probe v3 đã chạy bốn câu TRAIN: cả bốn đúng ngay lần đầu và được xác minh thành replay; không có correction. Ledger toàn learning pool ghi 4/1.817 attempted, 1.813 pending; không báo coverage toàn pool 100%.

Ngoài ra chương trình CPU đã kiểm lại sáu target replay từ SFT của parent, đúng learning split, khớp ngữ nghĩa, nguồn, executor và scorer. Hash SFT `5057658ac10c19579c675c9f926cc13f1e17641b88eb312318e48be75e6bb170` khớp cả metadata huấn luyện và training_complete. Sáu ID này không trùng bốn replay GPU: **10 replay duy nhất, 0 correction**. Các target lịch sử không được gọi là dự đoán đúng của model hiện tại.

Bằng chứng replay CPU: `D:\ZAIC\EXPERIMENTS\replay_seed_preparation_20261108\RESULT.json`, `PARENT_SFT_PROVENANCE.json`; target chi tiết chỉ ở vùng PRIVATE_DATA được ghi trong RESULT.json.

Provider correction hiện chỉ xác minh chặt chẽ một số dạng tổng theo tháng; nguồn TRAIN có 20 câu thuộc phạm vi đó. Chưa xác nhận provider đủ cho toàn bộ 1.817 câu. Các câu ngoài phạm vi phải giữ unresolved/chưa xác minh; không suy ra model sai chỉ vì verifier chưa hỗ trợ. Nhánh correction/replay đã qua tests giả lập; **chưa có GPU train trên correction thật**.

Calibration có 552 câu đã có exposure lịch sử, không phải untouched holdout. Chưa chạy so sánh parent/candidate trên toàn bộ calibration trong đợt smoke; không thăng BEST_MODEL. Wrapper orchestration có cổng đánh giá nhưng kết nối/kiểm chứng lượt đánh giá thực tế phải hoàn tất trước promotion.

## Lần thử lỗi và bài học được giữ lại

Version 4 và gói local `D:\ZAIC\PRIVATE_DATA\KAGGLE_RUNS\relearning_v2_failure_366862797` giữ 91 tệp output, log, ledger và FAILURE.json. Probe cũ có 16 lỗi sinh/parse qua tám câu, chưa có corrected plan; không biến chúng thành target học. Bốn câu được kiểm lại với cấu hình v3 đều đúng. Bằng chứng này không chứng minh model cần học lại tám câu.

Nguyên nhân smoke v2 dừng trước optimizer là Trainer tự dùng hai GPU và tạo batch=2 trong khi hàm loss yêu cầu batch=1. V3 đặt `CUDA_VISIBLE_DEVICES=0` cho child mới và chặn nếu còn nhiều GPU. V3 cũng bật inference cache, giữ raw prediction lỗi trong ledger Private trước khi đọc gold và không in nội dung vào exception. Có tests mới chống hồi quy cho các thay đổi này.

## Notebook và cấu hình cho lượt sau

- Notebook đang mở, bản mặc định version 7: https://www.kaggle.com/code/lnghongphc/zaic-relearning-training-readiness-20261108/edit . Accelerator=None, action=preflight, gpu_confirmed=False. Version 7 lưu cấu hình an toàn; lấy checkpoint ở version 6.
- GPU smoke/output version 6: https://www.kaggle.com/code/lnghongphc/zaic-relearning-training-readiness-20261108?scriptVersionId=366867761 . Quick Save successful chứng minh lưu output, không tự thay thế biên nhận GPU bên trong.
- Notebook local khớp bản chuẩn bị: `D:\ZAIC\EXPERIMENTS\relearning_v3\ZAIC_RELEARNING_V3_READY.ipynb`.
- Notebook pilot correction/replay có khóa: `D:\ZAIC\EXPERIMENTS\next_pilot_preparation\ZAIC_NEXT_CORRECTION_REPLAY_PILOT.ipynb`.
- Cấu hình: `D:\ZAIC\EXPERIMENTS\next_pilot_preparation\PILOT_REQUEST.json`. Mặc định `budget_approved=false`, `gpu_confirmed=false`; đường dẫn/hash dataset verified đang null vì chưa đủ correction. Không điền giả để bỏ cổng.
- Entry thực thi: `D:\ZAIC\EXPERIMENTS\next_pilot_preparation\pilot_entry.py`; 8 gate tests ở `test_pilot_entry.py`. Chặn sai parent, fresh/resume, quá ngân sách, replay-only, sai split, chưa verified, trùng ID và thiếu dataset trước khi import model.

Đề xuất pilot là warm-start lại **adapter 9B gốc**, optimizer/scheduler mới, tối đa 4 bước, 900 giây cho pha train/test, LR 5e-6, batch=1, gradient accumulation=1. Không dùng adapter smoke làm parent. Ngân sách này không bao gồm thu thập toàn pool hoặc đánh giá 552 câu. Thu thập lỗi/kiểm chứng corrected plan và đánh giá phải có giới hạn riêng trước khi mở các pha đó.

Lượt sau chỉ mở pilot sau khi có cả correction và replay đã verified, source/data/split manifests đã freeze, input khớp hash và người dùng chấp nhận ngân sách còn thiếu. Khi xong/lỗi: lưu output bền vững, tắt GPU ngay, rồi tải về và kiểm checksum mới đánh dấu synced. Không cần người dùng thao tác terminal.

Ảnh trạng thái cuối: `D:\ZAIC\EXPERIMENTS\relearning_v3\GPU_OFF_FINAL.jpg`.
