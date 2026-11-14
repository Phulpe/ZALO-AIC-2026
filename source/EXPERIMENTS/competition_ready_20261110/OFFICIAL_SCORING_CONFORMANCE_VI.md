# Đối chiếu đặc tả chấm công khai

Đọc ngày 10/11/2026 từ tab Evaluation của https://challenge.zalo.ai/portal/datamate/. Dùng endpoint GET công khai mà SPA chính thức gọi; không đăng nhập hoặc truy cập dữ liệu riêng. Phản hồi thành công được lưu riêng cùng hai ảnh bảng quy tắc/công thức.

Đặc tả xác nhận PA = (đúng − sai)/N; abstain nhận 0. Scorer baseline đang dùng khớp các quy tắc chính đã đối chiếu về type/value, số chính xác, làm tròn percent, chuẩn hóa chuỗi có giữ dấu, thứ tự list và hàng/cột table.

14 tests giả lập mới tại `test_official_contract.py` đã đạt trên CPU máy cá nhân. Không sử dụng câu hỏi, gold hoặc SFT cuộc thi trong fixtures. Không sửa scorer/source đã freeze đang chạy calibration. Đây là kiểm tra theo đặc tả công khai, không phải chứng nhận tương đương toàn bộ mã chấm riêng của ban tổ chức hoặc bằng chứng về điểm leaderboard.

Bản phản hồi/ảnh nguồn: `D:\ZAIC\PRIVATE_DATA\competition_ready_20261110\official_evaluation_response_valid2.json` và `official_scoring_contract_0.png`, `official_scoring_contract_1.png`.
