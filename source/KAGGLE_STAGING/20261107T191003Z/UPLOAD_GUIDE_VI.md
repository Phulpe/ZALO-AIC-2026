# Sẵn sàng upload Kaggle — chỉ dành cho 9B

1. Notebook mới: File → Import Notebook → chọn 03_CODE_AND_NOTEBOOK/ZAIC_9B_READY_PREFLIGHT.ipynb.
2. Add Input / Upload: tạo dataset mã Private chỉ chứa ZAIC_CODE_ONLY.zip. Chạy mặc định preflight (CPU): checksum nguồn + 99 tests. Không gắn notebook Colab cũ.
3. Dataset base Private: upload toàn bộ NỘI DUNG thư mục 01_BASE_QWEN35_9B, giữ base/ và BASE_UPLOAD_MANIFEST.json cùng cấp. Giữ cố định phiên bản này cho mọi run.
4. Dataset TRAIN SFT Private: upload toàn bộ NỘI DUNG 02_APPROVED_TRAIN_SFT, giữ data/ và SFT_UPLOAD_MANIFEST.json cùng cấp. Chỉ bộ SFT baseline; không có public/private test, câu hỏi đánh giá, transaction hay journal.
5. Gắn đúng một version mỗi dataset. Notebook tự xác định đường dẫn thực tế từ manifest, không phụ thuộc tên slug Kaggle.
6. Bật GPU, xem báo cáo phiên bản thư viện, chạy smoke 9B giả lập. Chưa train cho tới khi đạt cổng smoke và cấu hình được xem xét. Không cài đè stack Colab theo thói quen.
7. REVIEW_FIRST_TRAIN_CONFIG.json ghi cấu hình train fresh adapter. Không tiếp tục checkpoint cũ bằng notebook này. Adapter/checkpoint cũ giữ ở PRIVATE_DATA.
8. Sau run, Save Version kèm output của phiên hiện tại, tránh chạy lại train. Đợi version thành công, tải ZIP về runs/<run_id>, so manifest SHA256 trước khi đánh dấu đã đồng bộ. Giữ checkpoint đầy đủ để chuẩn bị resume riêng.

Base trong staging là hardlink tới file chuẩn trên cùng ổ D: không tốn thêm 19 GB nhưng cũng KHÔNG phải bản sao lưu độc lập. Chỉ dùng để upload, không sửa nội dung các file staging.

Quyền cloud: lưu lời xác nhận của người dùng trong CLOUD_PERMISSION_RECORD.json; chưa có URL ngoại lệ chính thức để đối chứng độc lập. Không thay đổi chia sẻ thành Public.

Bộ 4B không nằm trong gói upload 9B. Đã sửa tên/vị trí theo index nhưng vẫn giữ nguyên bất nhất metadata total_size, chưa chứng nhận runtime.

Notebook hỗ trợ cả ZIP mã nguyên vẹn và CODEX_PROJECT do Kaggle tự giải nén; cả hai đều được kiểm hash từng file trước import.
