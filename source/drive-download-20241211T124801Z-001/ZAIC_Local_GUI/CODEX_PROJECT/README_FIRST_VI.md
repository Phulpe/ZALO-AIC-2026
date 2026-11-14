# ZAIC — bắt đầu bằng giao diện, không cần CLI

Gói bàn giao: 07/11/2026. Đọc `docs/HANDOFF_CODEX_VI.md` để hiểu đầy đủ dự án.

**Gói ZIP này chứa mã và tài liệu, không chứa dữ liệu cuộc thi hay trọng số.** Chưa thể coi toàn bộ dự án đã về máy chỉ bằng việc giải nén gói này. Bộ trọng số base khoảng 19 GB; dung lượng toàn bộ lịch sử chưa được kiểm kê.

## Làm ngay trên Windows

1. Giải nén ZIP bàn giao. Đặt **nội dung thư mục CODEX_PROJECT** tại `D:\ZAIC\CODEX_PROJECT` (có thể chọn ổ khác).
2. Tạo `D:\ZAIC\PRIVATE_DATA` và `D:\ZAIC\DOWNLOADS_ARCHIVE` bằng File Explorer. Dữ liệu gốc và trọng số nằm ngoài thư mục Codex đọc mã.
3. Mở `docs/DOWNLOAD_AND_GUI_VI.md`: tải đủ các thư mục Drive được liệt kê, giải nén bằng giao diện và giữ bản gốc. Drive có thể chia thành nhiều ZIP độc lập; cần lấy đủ tất cả.
4. Trong ứng dụng Codex giống ảnh: **Choose project → chọn D:\ZAIC\CODEX_PROJECT → This computer**. Dán nội dung trong `docs/START_CODEX_PROMPT_VI.txt`.
5. Kiểm tra trên máy: nếu đã có Python 3.10+ với tkinter và liên kết tệp `.pyw`, nhấp đúp `gui/local_check.pyw`. Chọn thư mục `zaic` đã giải nén, rồi chọn thư mục kết quả cuối ở lượt kiểm tra thứ hai. Không cần nhập lệnh. Nếu Windows chưa chạy được `.pyw`, để Codex kiểm tra/cài Python từ nguồn chính thức qua GUI; không đổi đuôi giả thành `.exe`.
6. Với Kaggle, nhập `notebooks/ZAIC_Kaggle_GUI.ipynb` bằng menu File của một notebook mới. Lượt đầu chỉ chạy `ACTION='preflight'`; chưa tải model, chưa huấn luyện.

## Những thứ có sẵn

- `AGENTS.md`: quy tắc tiếp tục công việc, GUI trước, giới hạn dữ liệu và bảo toàn bản đã chạy.
- `docs/HANDOFF_CODEX_VI.md`: báo cáo chi tiết về kiến trúc, huấn luyện, lỗi, kết quả, checkpoint, lộ trình.
- `docs/DOWNLOAD_AND_GUI_VI.md`: bảng tải toàn bộ nguồn, cách thao tác Codex/Kaggle và xử lý lỗi thường gặp.
- `docs/TRANSFER_MANIFEST.json`: danh mục nguồn và trạng thái tải, không phải biên bản đã tải đủ.
- `frozen_v21/`: bản mã v2.1 nguyên trạng; 27 file app/tools và 2 script khôi phục Colab, có SHA256.
- `reports/`: báo cáo v2.1 và biên bản kiểm tra gói bàn giao.
- `gui/`: kiểm tra cục bộ và backend notebook mới; không gọi Kaggle CLI.
- `notebooks/`: notebook mới, không chứa output/câu hỏi/trọng số.

## Vai trò của máy và Kaggle

Máy cá nhân giữ bản gốc. Codex giúp sửa mã, tạo notebook và đọc báo cáo tổng hợp. Kaggle cung cấp GPU và cần bản sao input trên hạ tầng Kaggle; đường dẫn `D:\...` không tự xuất hiện ở đó. Chỉ gắn dữ liệu cuộc thi sau khi xác nhận thể lệ cho phép xử lý trên Kaggle. Đặt Private không tự giải quyết điều kiện này.

Notebook được chuẩn bị để chạy kiểm tra, smoke test rồi **huấn luyện adapter mới từ base** với cấu hình cũ. Nó chưa được chạy GPU trên Kaggle và không phải quy trình tiếp tục optimizer/checkpoint cũ. Mỗi lượt GPU tạo thư mục mới; không ghi đè adapter đã có. Nếu chỉ muốn dùng thành quả cũ để suy luận thì không cần huấn luyện lại.

Không có bảo đảm PA = 1. V2.1 public đã xuất đủ 1.000 dòng nhưng chưa có điểm chấm mới; 235 dòng trả lời không đồng nghĩa 235 dòng đúng.
