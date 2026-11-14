# Vận hành lượt học lại kế tiếp

Phạm vi chuẩn bị ngày 08/11/2026: bản thí nghiệm mới `relearning_v2`. Không sửa `frozen_v21`, adapter 9B gốc, checkpoint-40/45/48, kết quả cuối hay notebook gốc. Quy tắc hiện hành là `D:\ZAIC\AGENTS.md`; `training.md` là đặc tả, không phải bằng chứng đã thực thi.

## Tài sản và quyền

- Base cố định: Qwen/Qwen3.5-9B, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Kiểm đủ 15 tài sản, gồm config/tokenizer/index và bốn shard, theo BASE_UPLOAD_MANIFEST.json.
- Adapter nguồn: `D:\ZAIC\PRIVATE_DATA\zaic\datamate_v1\qwen35_9b\candidate\final`. Trọng số 86.623.432 byte, SHA256 `e7dfa635beccce128c99f9d3fd993d33dbb4c88e73e3ddeccbc2bc915eec4a4d`.
- Quyền thao tác tài khoản đã được người dùng cho phép. Phạm vi xử lý tài sản cuộc thi được ghi là `user_attested`, không ghi thành xác nhận độc lập của ban tổ chức. Xem PERMISSION_DECISION.json và hash AGENTS.md kèm theo.
- TRAIN runtime Private chứa learning và calibration; không chứa audit gold hoặc public/private test. Không mở nội dung câu hỏi/gold/giao dịch trong giao diện AI. Chương trình chỉ xuất báo cáo tổng hợp.
- Wheel bitsandbytes 0.50.2 lấy từ PyPI, SHA256 `55348a9a4a21bfd99cf8c7b32fe67b4030ae5c2a05738e03c1747f65fa6ec283`. Hướng dẫn chính thức: https://huggingface.co/docs/bitsandbytes/installation. Bootstrap offline không thay PyTorch, không chạy mã từ nguồn không xác định.

## Phân biệt các lượt thực thi

1. Preflight mặc định dùng CPU: 99 baseline tests và 60 tests mới, kiểm input bằng checksum, demo giả lập. `action=preflight`, `gpu_confirmed=False`, Accelerator=None.
2. Probe GPU môi trường: nhận diện CUDA, GPU; kiểm NF4 và optimizer bằng tensor giả nhỏ. Probe này không nạp adapter 9B và không chứng minh model đã học.
3. Smoke 9B: warm-start đúng adapter nguồn, probe tối đa tám câu TRAIN thuộc phạm vi hỗ trợ, sau đó một optimizer step trên hai mẫu giả lập. Lưu và nạp lại checkpoint, kiểm trọng số thay đổi hữu hạn và forward loss hữu hạn. Candidate giả lập không được chọn làm parent cho lượt học thật hoặc BEST_MODEL.
4. Lượt correction/replay thật kế tiếp: dùng lại adapter 9B gốc làm parent, optimizer/scheduler mới. Chỉ dùng dataset có correction và replay đã được xác minh. NEXT_PILOT_REVIEW.json là đề xuất cấu hình, chưa phải bằng chứng chạy hoặc quyền dùng hết quota.

Không gọi warm-start là resume checkpoint-48. Full resume là hạng mục riêng, chưa xác nhận hoạt động. Không fallback sang fresh khi nạp parent lỗi.

## Thứ tự thao tác của agent

1. Đọc báo cáo mới nhất và đối chiếu input của lượt định chạy. Kiểm code ZIP, base revision/shards, adapter hash, manifest/split và wheel. Thực hiện chuẩn bị, gắn input và checksum trên CPU.
2. Chỉ bật T4 khi cell test đã sẵn sàng. Nếu đổi accelerator làm mất workspace, chạy lại CONFIG và cell giải nén/preflight; không dùng Run All khi action huấn luyện đã bật. Bootstrap offline có hash pin trước khi import.
3. Chạy probe môi trường trước. Thiếu CUDA/NF4/optimizer, OOM hoặc lỗi dependency thì dừng GPU, đưa None và xử lý tiếp trên CPU. Không để GPU chạy trong lúc phân tích hoặc chờ người dùng.
4. Khi probe đạt, bật đúng `action=gpu_smoke`, `gpu_confirmed=True`, chạy riêng cell GPU. Không mở train dài. Theo dõi giới hạn subprocess 900 giây; giữ log lỗi và checkpoint một phần nếu có.
5. Chương trình lưu dự đoán trước khi mở gold qua LazyGoldReader, chấm bằng scorer, phân loại lỗi, retry hữu hạn. Correction phải qua kiểm ngữ nghĩa, executor và scorer; không dùng dự đoán sai làm target. Những câu chưa giải được vẫn ở ledger.
6. Probe chỉ thử tối đa tám câu, không phải vòng học đầy đủ. Learning pool có 1.817 câu; ledger toàn pool phải ghi pending cho phần chưa thử. Không báo attempt coverage 100% từ tám câu. Provider correction hiện chỉ hỗ trợ chặt chẽ một số câu tổng theo tháng; các loại khác giữ unresolved.
7. Trước học correction/replay thật, freeze dataset, evaluation IDs/split/scorer và cấu hình. Calibration 552 câu đã có exposure lịch sử, không gọi là untouched holdout. So parent/candidate trên cùng tập, kiểm correct→wrong. Không đổi BEST_MODEL nếu thiếu bằng chứng hoặc không đạt tiêu chí.
8. Khi test/train xong hoặc lỗi: lưu output cần thiết vào phiên bản bền vững, dừng phiên GPU và chọn None ngay. Kiểm cả Interactive Session và phiên nền trong Active Events. Đóng tab không tương đương dừng GPU.
9. Chỉ sau khi GPU tắt mới tải output về thư mục run riêng trên máy. Kiểm mọi file theo manifest; giữ bản trước nếu thiếu hoặc xung đột. Không tải lại base cố định. Không đổi tên/xóa/gộp tài sản gốc.

## Trạng thái tách riêng

- LATEST_CHECKPOINT: checkpoint mới nhất có bằng chứng tạo thành công và manifest, không tự thay BEST_MODEL.
- BEST_MODEL: bản đạt đánh giá đã freeze và tiêu chí promotion; giữ nguyên nếu thiếu đánh giá hoặc candidate không đạt.
- LATEST_SYNCED: run đã tải về máy và khớp toàn bộ manifest hai nơi. Có bản trên Kaggle hoặc có tên file trên máy chưa đủ xác nhận synced.

Giữ trạng thái thử nghiệm trong thư mục v2. Không ghi đè tracker lịch sử. Chỉ hứa đồng bộ sau những run được agent thực sự theo dõi; không hứa tự đồng bộ hoặc tắt GPU khi agent/trình duyệt đã ngắt.
