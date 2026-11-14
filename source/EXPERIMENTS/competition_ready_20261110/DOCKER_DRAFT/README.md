# ZAIC DataMate — bản nháp đóng gói

**Chưa build image, chưa kiểm GPU Docker, chưa chốt model thi.** `ready_to_infer=false` là cổng bắt buộc, không phải lỗi cần bỏ qua. Chỉ bật sau khi chốt model và kiểm thử image.

Base PyTorch chính thức đã pin digest, CUDA12.8 đáp ứng mức tối thiểu công bố cho RTX5090. Runtime khác CUDA13.0 của Kaggle; cần smoke/batch parity trên image thật trước gọi sẵn sàng.

`predict.sh` đọc `/data/questions.jsonl`, giữ nguồn `/data` read-only, dùng cache `/tmp`, ghi `/result/submission.jsonl` và log từng câu. Không gọi dịch vụ mạng trong inference; không tải model tại runtime; không dùng gold/public IDs hoặc training artifacts để sửa dự đoán.

Chỉ dùng một Qwen3.5-9B với một adapter được chọn khi inference. Parent E7 và evidence được đóng gói để tái hiện train, không được tự dùng làm ensemble/fallback. Không gộp adapter. Giữ nguyên trọng số gốc trên máy.

Named contexts lấy tài sản đang có trên D:, tránh sao chép một staging base19GB khác. Agent sẽ build, kiểm kích thước tar dưới100GB, kiểm hash và thử đúng lệnh offline khi Docker daemon/storage đủ điều kiện. Không tự đẩy registry, chia sẻ Drive công khai hoặc nộp bài.

Nguồn guideline: https://dl-challenge.zalo.ai/2026/datamate_dkewxftdzo/zac2026_DataMate_docker_guideline_final.pdf
