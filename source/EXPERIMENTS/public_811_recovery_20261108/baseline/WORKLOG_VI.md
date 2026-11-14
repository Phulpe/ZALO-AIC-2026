# ZAIC DataMate — bản sửa v2.1 và kết quả public_test, 07/11/2026

## Trạng thái hiện tại

**Đã hoàn tất 1.000 câu public_test, kiểm tra cấu trúc/ID/nhật ký thành công và lưu bộ kết quả cuối lên Drive của tài khoản chạy GPU. Chưa nộp lên ban tổ chức; chưa có PA public mới. Mục tiêu PA = 1 chưa đạt.**

File kết quả: https://drive.google.com/file/d/17n5Cv2zIJdz7Kf1tyXeBKhZdI-vr1TVN/view?authuser=1
Thư mục kết quả đã xác minh: https://drive.google.com/drive/folders/1TupBKE0PL5DFsYGoBVq5qqmok-4Vn4WV
SHA256 submission: `e0815c96524e9db0badf987e45e13227eec8e57c7f3195f5aa93a580a0508780`
Kích thước: 85573 byte.

Chỉ dùng file trong thư mục kết quả đã xác minh này. Thư mục cũ chứa các file của lượt chạy gián đoạn và các mảnh checkpoint được bảo toàn.

## Kết quả đã đo

| Tập / phiên bản | Đúng | Sai | Bỏ qua | PA |
|---|---:|---:|---:|---:|
| TRAIN hiệu chỉnh 200 câu, bộ cũ | 89 | 29 | 82 | 0,30 |
| TRAIN hiệu chỉnh 200 câu, v2/v2.1 sinh mới | 91 | 21 | 88 | 0,35 |
| TRAIN xác nhận mới 100 câu, v2.1 | 36 | 6 | 58 | 0,30 |
| Public_test 1.000 câu, v2.1 | Chưa biết | Chưa biết | 765 | Chưa chấm |

Public có **235 câu trả lời**, không được gọi là 235 câu đúng. Cấu trúc: 174 number, 22 string, 16 list, 11 percent, 8 table, 4 date; 765 abstain. Không có unanswerable trong file cuối.

PA đang được mã chấm nội bộ tính bằng (đúng − sai)/tổng. Tỷ lệ đúng trong phần đã trả lời của tập xác nhận là 36/42 = 85,71%, khác PA toàn tập 0,30. Độ phủ public hiện tại còn thấp, nên không thể xem bản này đã đạt PA = 1.

Tập hiệu chỉnh và tập xác nhận lấy từ TRAIN, tách doanh nghiệp với nhau và với dữ liệu SFT 9B. Các doanh nghiệp đã từng xuất hiện trong lịch sử phát triển 4B/audit được công khai là có tiếp xúc trước; không gọi là hoàn toàn chưa thấy trong toàn bộ lịch sử dự án. Tập xác nhận 2 được cố định trước khi sửa loader, loại mọi ID đánh giá trước đó:
`fa7250d85bf5c162c7934c909fae1fa21cc34c0e9a8559b88d3e6b6d881cfc8c`.
Nhãn public/private không được dùng để huấn luyện, chọn chính sách hoặc sửa đáp án.

## Những lỗi đã sửa

1. File `BanHang_HoaDon.xlsx` bị nhận nhầm thành danh mục hàng hóa vì chuỗi chuẩn hóa `banhanghoadon` chứa `hanghoa` qua ranh giới hai từ. Ưu tiên nhận diện chi tiết giao dịch và hóa đơn trước tên danh mục chung; sheet chưa biết quay về tên file, sheet có vai trò rõ vẫn ưu tiên.
2. Bổ sung cách viết mã hóa đơn/mã giao dịch để nối đầu hóa đơn với dòng hàng. Kiểm tra trên TRAIN khôi phục 95.845 dòng chi tiết, 51.044 đơn; toàn bộ 95.845 dòng có mã đơn và ngày bán, khoảng 01/01/2025–25/05/2026. Không thêm nhánh theo mã doanh nghiệp/câu hỏi.
3. Giữ các sửa v2: thời gian lồng nhau được kiểm tra; không đoán năm từ đồng hồ; không gọi month_bounds sai số tham số; dùng quy tắc README/rulebook liên quan; chặn mâu thuẫn rõ về chỉ tiêu, thời gian và điều kiện; xử lý phủ dữ liệu; sửa xếp hạng khách thiếu tên và danh tính theo mã; bảo toàn lượng thập phân; không dùng nhầm tổng phải trả làm doanh thu từng dòng; ghi vị trí lỗi và xuất JSONL có kiểm tra.
4. Cổng phát hành yêu cầu không lỗi loader/network/OOM. Chính sách bỏ qua học từ hiệu chỉnh bị loại do PA thấp hơn bản raw. Không huấn luyện lại adapter.

75 kiểm thử ứng dụng và 24 kiểm thử công cụ đều thành công, tổng 99, tại máy cục bộ và phiên Colab đã khôi phục. Mã ứng dụng/công cụ được cố định trước các lượt chạy.

## Gián đoạn và khôi phục checkpoint

Lượt public đầu lưu được 611 kế hoạch, sau đó Colab ngắt kết nối và báo hạn mức GPU. Người dùng cung cấp phiên T4 mới trên tài khoản sinh viên. Đã gắn Drive, thêm lối tắt dữ liệu và khôi phục đúng phiên bản thư viện. Người dùng đăng nhập lại tài khoản chứa bản lưu và cập nhật quyền truy cập để phiên mới đọc được.

Lượt chạy tiếp trên thư mục dùng chung sinh thêm 68 kế hoạch, nhưng lớp gắn Drive tạo nhiều file trùng tên thay vì cập nhật journal như dự kiến. Đã dừng nhóm tiến trình đó. Các file kế hoạch được đổi tên riêng để bảo toàn, thư mục cũ đổi thành `public_test_v21_preserved_fragments`; không xóa các phần kế hoạch.

Sau khi thông tin đăng nhập Colab được xác thực, đã đọc 17 mảnh qua Drive API vào **ổ cục bộ của Colab**, đối chiếu MD5 từng file, hợp nhất theo ID với điều kiện bản ghi trùng phải giống hệt. Kết quả: **679 kế hoạch, 0 xung đột, không sửa giá trị kế hoạch**. Fingerprint gốc được giữ nguyên byte; key:
`96945a4b5bc607922f1039219d7ce659c751ace5fccd2ce73b8d57a7771ba5a7`.

Lượt cuối chạy hoàn toàn trên ổ cục bộ, dùng lại 679 kế hoạch và sinh mới 321 kế hoạch. Tiến trình riêng bên ngoài suy luận lưu checkpoint bất biến, tên chứa số bản ghi và SHA256; mọi bản tải lên được đối chiếu MD5/kích thước. Kết quả cuối và 9 file liên quan được lưu trong thư mục Drive mới, có ID riêng. Không thay đổi quyền chia sẻ hay gửi thông báo cho người khác bằng công cụ.

## Kiểm tra kết quả cuối

- 1.000 dòng JSONL; 1.000 ID duy nhất, đúng thứ tự câu hỏi, không thiếu/trùng.
- 1.000 kế hoạch và 1.000 nhật ký câu hỏi.
- Nhật ký ghi đúng nguồn: 679 journal_replay, 321 current_run.
- Audit passed; 0 lỗi cấu trúc.
- Loader errors: {}; OOM retries: 0; network attempts trong tiến trình suy luận: 0.
- Không có lý do lỗi dạng executor_exception trong thống kê lý do bỏ qua cuối.
- Lượt chạy cuối: 1.491,58 giây, gồm 95,67 giây khởi tạo, 1.303,40 giây sinh kế hoạch; VRAM cực đại 8,79 GiB.
- Thời gian lượt cuối không bao gồm hai lượt trước, sửa mã, xác thực, khôi phục và sao lưu.
- Các phần đã tải lên Drive được kiểm tra MD5/kích thước. SHA256 của file submission trùng giá trị trong runtime/audit.

## Hạn chế và bước phát triển tiếp

765 câu bị bỏ qua. Các nhóm lớn trong thống kê là: ngày nguồn chưa được xác nhận/phân tích đầy đủ (133), kế hoạch không hợp lệ (106), kế hoạch mâu thuẫn thời gian hỏi (83), phép tổng hợp/so sánh/ngưỡng chưa hỗ trợ (60), số tiền nguồn chưa được phân tích đầy đủ (59). Đây là lý do từ bộ thực thi, không phải nhãn đúng/sai public.

Nếu phát triển v2.2, ưu tiên cải thiện đọc ngày/số và hợp đồng kế hoạch trên TRAIN; mở rộng phép so sánh, đếm thực thể, tồn kho/công nợ với kiểm thử nghiệp vụ. Cố định tập xác nhận mới trước sửa tiếp; tập xác nhận 2 nay đã mở. Không ép trả lời hay gán unanswerable chỉ để giảm số abstain.

Không thể hứa PA = 1 hoặc suy ra điểm public từ điểm TRAIN. Chưa xây/chạy kiểm thử Docker network-none hay RTX 5090. Gói mã là mã nguồn và bằng chứng tổng hợp, không phải image Docker hay gói trọng số đạt chứng nhận nộp giải pháp.

## Mô hình và môi trường

Model lock: Qwen/Qwen3.5-9B, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
Adapter giữ nguyên:
`e7dfa635beccce128c99f9d3fd993d33dbb4c88e73e3ddeccbc2bc915eec4a4d`.

Python 3.13.16; Torch 2.11.0+cu130; Transformers 5.22.0; PEFT 0.21.2; accelerate 1.15.0; bitsandbytes 0.50.2; pandas 2.2.3; numpy 2.4.6; safetensors 0.8.0; huggingface-hub 1.33.0; openpyxl 3.1.5. Batch 4, đầu vào tối đa 3.072 token, đầu ra 320 token, greedy, thinking tắt. Engine datamate-qwen35-9b-0.2.1, runtime batch-0.2.0.

Drive API chỉ được dùng cho xác thực/khôi phục/sao lưu bên ngoài tiến trình suy luận. Không có API LLM bên ngoài trong suy luận. Các script checkpoint Colab không phải entrypoint Docker network-none.

## Vị trí và tiếp tục công việc

- Source đang chạy: `/content/zaic_9b_v21_20261107`.
- Kết quả cục bộ cuối: `/content/zaic_v21_final_run/public_test_v21_frozen`.
- Marker hoàn tất: `/content/zaic_v21_final_run/DRIVE_VERIFIED_RESULT.json`, `REPAIR_V21_RESUMED_COMPLETED.json`.
- Thư mục checkpoint Drive của tài khoản chạy GPU: `ZAIC_9B_Resume_Own_20261107`, ID `1bPHpPmC29_qar42kEIPeXSGgbg_PxNkh`.
- Thư mục kết quả cuối: ID `1TupBKE0PL5DFsYGoBVq5qqmok-4Vn4WV`.
- Tài khoản đang sở hữu kết quả cuối: tài khoản sinh viên của người dùng.
- Không chạy lại toàn bộ notebook hoặc controller khi đã hoàn tất. Thay đổi mã/config/trọng số cần thí nghiệm và thư mục output mới; không sửa fingerprint để ép tiếp tục.
- Hai script ở gốc gói mã (`resume_public_v21.py`, `run_public_local_with_checkpoints.py`) mô tả quy trình khôi phục đã dùng; đường dẫn/ID Drive là riêng cho thí nghiệm này.
- Bộ mã gồm 27 file Python cố định trong app/tools và 2 script hạ tầng khôi phục. Không chứa câu hỏi, nhãn, đáp án từng câu, dữ liệu giao dịch hay trọng số mô hình.

Nguồn dữ liệu cuộc thi: challenge.zalo.ai. Tất cả phép xử lý dữ liệu chi tiết ở lại trong môi trường Colab/Drive của người dùng; báo cáo và gói mã chỉ xuất số liệu tổng hợp.
