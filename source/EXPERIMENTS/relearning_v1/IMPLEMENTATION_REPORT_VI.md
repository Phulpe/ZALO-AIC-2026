# Triển khai relearning_v1 — 08/11/2026

Mã mới tại `D:\ZAIC\EXPERIMENTS\relearning_v1`. Không sửa frozen_v21 hoặc adapter/checkpoint gốc. Chưa chạy optimizer trên model thật, chưa dùng GPU, chưa xử lý câu hỏi/SFT thật trong lượt này.

## Trạng thái từng phần

| Phần | Trạng thái và bằng chứng | Chưa xác minh |
|---|---|---|
| Warm-start | Đã viết `relearning.warm_start` và `train_warm.train`: kiểm parent hash, base/revision, nạp PEFT trainable; Trainer mới, optimizer/scheduler mới; không fallback fresh/resume. Test mock kiểm trainable, lỗi nạp, sai parent, mode và optimizer metadata đạt. | Chưa load adapter 9B bằng PEFT thật, chưa GPU/optimizer step |
| Learning/ledger | Đã qua tests và chạy CPU với dữ liệu giả lập: prediction được ghi trước gold; allowlist câu hỏi không chứa gold; split tách biệt; retry 1–3; mọi lỗi có record; unresolved giữ lại. | Chưa chạy learning pool thật; timeout generate là giới hạn best-effort của thư viện, không phải process hard-kill |
| Scoring/correction | `baseline_bridge.py` nối scorer/executor baseline; correction được validate, kiểm ngữ nghĩa độc lập, execute và score lại. Test tích hợp executor thật trên fixture tổng tháng đạt. | Bộ correction/ngữ nghĩa tự động hiện chỉ hỗ trợ các mẫu câu tổng tháng được khớp chặt (doanh thu/số lượng/số đơn). Các dạng khác giữ unresolved; cần mở rộng có tests, không tự coi tất cả plan là verified |
| Replay/regression | Bank verified, mixture cân bằng khi cả hai nhóm có mẫu, không oversample; giữ mẫu chưa dùng. `cycle.run_cycle` nối learning→dataset→learner→evaluation→regression→promotion. Test toàn vòng với learner giả lập chứng minh candidate có regression không thay BEST_MODEL. | Chưa chứng minh hạn chế quên sau gradient update thật; evaluator thực tế cho learning/calibration cần được chuẩn bị trong môi trường được phép |
| Checkpoint | `train_warm.py` viết adapter/tokenizer, optimizer/scheduler, scaler nếu có, Python/NumPy/Torch/CUDA RNG, trainer state, config, source/environment/data manifests và hash. Các checkpoint trung gian được Trainer giữ; lỗi ghi FAILURE, không xóa bằng chứng. | Serialization/reload của checkpoint model thật chưa chạy; full resume riêng, chưa triển khai |
| Ba trạng thái | `track()` tách LATEST_CHECKPOINT/BEST_MODEL/LATEST_SYNCED; test thiếu file, sai checksum, regression và sync thiếu saved-version/hash đạt. | Không cập nhật tracker model đang dùng vì chưa có run model mới. Báo cáo CPU đã lấy về không đồng nghĩa checkpoint model đã đồng bộ |
| Notebook | Đã tạo và chạy notebook mới trên Kaggle CPU, version 2 Successful/Private. Mặc định preflight, không input thi. Cổng GPU smoke 1 step dùng hai target fixture được executor/scorer xác minh; gate quyền và gpu_confirmed vẫn tắt. | GPU smoke 9B chưa chạy; không bật long train trong notebook này |

## Kết quả thực thi

- Windows `.venv` có sẵn của project: **40/40 test mới đạt**. Không cài thêm dependency.
- Kaggle Linux CPU: **99/99 baseline + 40/40 mới đạt**, 0 failures/errors. Xem `TEST_RESULTS_KAGGLE_V2.json`, `KAGGLE_CPU_OBSERVED.txt` và ảnh `KAGGLE_CPU_TEST_PROOF.jpg`.
- Gói code cuối SHA256: `2889f2f5a8979253ba6a4f8ecce63f3fc18edda08226d449ca34b4d8358afcb1`.
- Demo CPU cuối: run `20261108T061853536563Z_78e98a79`; 4 expected/4 attempted, attempt coverage 100%; hai first-pass answers khớp scorer nhưng một plan chỉ đúng số tình cờ nên không được replay. Ba trường hợp cần correction/ngữ nghĩa; một correction verified (1/3), hai unresolved. Dataset có một replay + một correction. **optimizer_executed=false**.
- Bộ 99 test nguyên bản trên Windows còn hai lỗi tương thích có sẵn: đường dẫn dấu `\` và quyền tạo symlink. Không thay baseline hoặc thiết lập bảo mật để che lỗi. Báo cáo Windows tại `TEST_RESULTS.json`/`TEST_DETAILS.txt`; Linux đã chạy nguyên bộ đạt.
- `PRESERVATION_CHECK.json`: 27 hash nguồn gốc và 27 hash bản sao đạt; adapter gốc 86.623.432 byte, SHA256 đúng `e7dfa635beccce128c99f9d3fd993d33dbb4c88e73e3ddeccbc2bc915eec4a4d`. Không thao tác ghi trên các checkpoint cũ.

## Notebook và cách dùng

[Notebook thí nghiệm](https://www.kaggle.com/code/lnghongphc/zaic-relearning-v1-synthetic-preflight-20261108/edit)

[Version 2 đã lưu](https://www.kaggle.com/code/lnghongphc/zaic-relearning-v1-synthetic-preflight-20261108?scriptVersionId=366835817)

`ZAIC_RELEARNING_V1.ipynb` là bản local tương ứng, nhúng ZIP chỉ có mã và fixture giả lập. Cell preflight bung vào workspace mới, kiểm SHA gói, chạy cả hai bộ test và demo. `run_checks.py` không import torch để chạy tests. `train_warm.py` chỉ import thư viện GPU khi gọi train sau gates.

Để dùng dữ liệu thật, chương trình trong môi trường được phép phải tạo pool allowlist và gold_reader riêng, nối BaselineBridge + semantic verifier/correction provider vào `cycle.run_cycle`; learner quản lý giải phóng model suy luận trước khi gọi trainer, evaluator dùng mode evaluation và split/hash cố định. Không cung cấp raw pool/gold vào chat. Đây là giao diện chương trình đã viết, chưa phải tác vụ đã chạy trên dữ liệu cuộc thi.

Ledger hoàn tất có thể được đọc lại với fingerprint khớp; ledger có attempt dở dang bị từ chối phục hồi tự động để tránh tính sai coverage. Giữ file đó để kiểm toán/phục hồi riêng; không xóa hoặc giả nhận đã chạy xong.

## Phần bị chặn và bước còn lại

Theo mục 11 `D:\ZAIC\AGENTS.md`: khi `competition_permission_evidence.status` chưa verified, chỉ tiếp tục mã/môi trường/test giả lập, chưa xử lý tài sản thi trên Kaggle. Quyền thao tác tài khoản và lời xác nhận của người dùng đã được giữ riêng; chưa có bằng chứng thể lệ độc lập đủ phạm vi để mở cổng. Đã tìm nguồn chính thức DataMate, nhưng nội dung được công cụ đọc không cung cấp xác nhận cloud cụ thể; không tự suy diễn hoặc đổi trạng thái thành verified.

Vì vậy chưa gắn adapter/SFT thi vào notebook mới và chưa bật GPU. Notebook 9B cũ giữ nguyên; session của nó quan sát lúc tiếp quản là off. Session CPU của notebook mới cũng đã được dừng sau khi lưu thành công; xem `KAGGLE_SESSION_OFF_PROOF.jpg`. Không có phiên GPU được bật trong nhiệm vụ này.

Sau khi có bằng chứng cho phép phù hợp: kiểm môi trường CUDA/Transformers/PEFT/bnb thực tế, attach đúng base/parent đã pin, chạy smoke 1 step trên fixture; kiểm loss/gradient, adapter thay đổi hữu hạn, checkpoint reload. Tắt GPU ngay sau test. Chưa chạy vòng train dài; agent phải đề xuất ngân sách và chỉ hỏi phần còn thiếu trước thực thi.

Full resume checkpoint-48 chưa triển khai theo phạm vi người dùng yêu cầu. Không dùng nó để chặn việc phát triển warm-start hoặc vòng learning.

API đã đối chiếu với [PEFT from_pretrained](https://huggingface.co/docs/peft/package_reference/peft_model) và [Trainer callbacks](https://huggingface.co/docs/transformers/main_classes/callback); việc đọc tài liệu API không thay thế GPU smoke.
