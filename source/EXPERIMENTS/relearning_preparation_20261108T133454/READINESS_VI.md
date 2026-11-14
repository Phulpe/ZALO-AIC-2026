# Chuẩn bị lượt warm-start — chưa đủ điều kiện train

Đã đối chiếu lại nguồn gói code, adapter cha 9B và đủ 15 file base bằng SHA256 trên máy. Bốn shard được index tham chiếu đầy đủ. Báo cáo 99 baseline + 40 test mới trên Kaggle CPU đã tồn tại và khớp gói code; lần này không gọi đó là kiểm thử GPU mới.

Notebook preflight và ZIP chỉ có mã đã được sao chép vào thư mục chuẩn bị này. Không sửa notebook/adapter/checkpoint gốc. Hai cấu hình đã chuẩn bị để xem xét: smoke 1 optimizer step trên fixture; pilot warm-start tối đa 4 steps, LR 5e-6, batch 1/accum 16, đề xuất tối đa 0,25 giờ GPU. Đây không phải lệnh train hoặc cam kết dùng hết ngân sách.

## Điều kiện còn thiếu

1. Bằng chứng thể lệ độc lập: lời xác nhận mới của người dùng đã lưu nguyên văn, quyền tài khoản đã có; chưa đổi thành verified. Không hỏi lại cùng quyền tài khoản. Theo mục 11 AGENTS.md, vẫn chưa mở xử lý tài sản thi trên Kaggle khi thiếu nguồn này.
2. Smoke warm-start thật: finite loss/gradient, completion-mask/shift, adapter thay đổi hữu hạn, checkpoint reload và checksum. Code v1 hiện đã có train/save; các phép xác minh sau train này chưa có bằng chứng thực thi GPU. Không coi unit test mock là đã load 9B.
3. Correction/replay thật cùng split/evaluation đã freeze: hiện mới có demo giả lập. Bộ SFT baseline 768 không tự trở thành correction/replay mới. Bộ sửa tự động hiện chỉ hỗ trợ nhóm câu tổng tháng; câu ngoài phạm vi vẫn phải ở ledger.
4. Chốt phần ngân sách pilot còn thiếu trước train. Không cần hỏi lại init_mode: đã chọn warm_start, optimizer/scheduler mới; checkpoint-48 chỉ bảo toàn/đối chứng.

## Thứ tự tiếp tục

- Xác minh nguồn cho phép → giữ GPU off để kiểm source/input/split → kiểm môi trường GPU và chạy smoke 1 step, có giới hạn, chỉ khi đủ gates.
- Lưu output bền vững rồi tắt session/Accelerator=None ngay; sau đó mới tải về máy và đối chiếu manifest. Không chờ tải file trong khi GPU chạy rỗi.
- Khi smoke đạt, tạo learning pool/ledger và correction/replay được xác minh trong môi trường được phép, freeze evaluation, hoàn thiện config pilot và kiểm input trước run.
- Chỉ báo sẵn sàng cho pilot khi READINESS đã có bằng chứng đạt các điều kiện tương ứng; không tự bật train dài, không tự promote BEST_MODEL.

Hai notebook được quan sát session off trong lượt chuẩn bị này. Không bật GPU hoặc chạy train. Bằng chứng session: D:\ZAIC\EXPERIMENTS\relearning_v1\READINESS_SESSION_OFF.jpg.

## Bổ sung kiểm tra cuối đã chuẩn bị thực tế

- `smoke_validation.py`: kiểm tensor hữu hạn và thực sự có thay đổi so với parent; từ chối sai keys/shape, NaN/Inf hoặc không có update. Xác minh manifest trước reload; không nạp pickle optimizer/RNG.
- `reload_forward_checkpoint`: đã viết luồng load lại adapter đã lưu bằng PEFT, kiểm mask completion và loss parity giữa loss native với logits compact (sai khác tối đa 0,02). Đây là code chưa chạy GPU, không phải chứng nhận reload thành công.
- 8/8 test CPU trong `test_smoke_validation.py` đạt trên `.venv` sẵn có. Một số test dùng callback giả lập cho reload: không được tính là GPU test. 99 baseline + 40 test trong gói trước là bằng chứng Kaggle Linux riêng, không cộng thành 147 test đã chạy trên Kaggle.
- `ZAIC_FINAL_SMOKE_REVIEW.ipynb`: giữ mặc định preflight, nhúng gói code đã pin cùng supplement có hash riêng. GPU stage chỉ cho 1 step, child process bị giới hạn 900 giây kể cả load/train/reload. Timeout/exception luôn yêu cầu agent dừng session/đưa Accelerator=None; không hứa tự tắt Kaggle khi agent ngắt. Chưa upload/chạy revision bổ sung này trên Kaggle.
- Script tạo notebook đã kiểm cú pháp các cell; SHA256 notebook/supplement nằm trong `SUPPLEMENT_MANIFEST.json`. Các điều kiện GPU và quyền vẫn tắt. Không thay đổi gói v1 đã kiểm thử hoặc notebook 9B cũ.
