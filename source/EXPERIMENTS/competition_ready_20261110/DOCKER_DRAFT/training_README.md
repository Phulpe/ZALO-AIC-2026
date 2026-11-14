# Evidence huấn luyện trong image

Pilot ngày10/11 warm-start E7 (không resume), optimizer/scheduler mới, source CODE_READY_R1, bank correction/replay đã freeze,18steps. Evidence chứa đầy đủ checkpoint/state và run config. Runtime TRAIN và bank được giữ riêng theo vai trò; calibration/audit không đưa vào optimizer hoặc dùng sửa câu trả lời.

Entrypoint tái hiện pilot: source `frozen_pilot.py`, cấu hình tham khảo trong evidence; phải remap các đường dẫn sang vị trí trong image, giữ hash và tạo output mới. Không dùng nguyên đường dẫn `/kaggle/working` lịch sử. Chưa chạy thử tái hiện hoặc full resume, chưa bảo đảm bit-for-bit.

E7 là adapter riêng đã huấn luyện từ base; pilot này không chứng minh tái hiện đầy đủ giai đoạn E7 từ đầu. Cần bổ sung code/data/config lịch sử stage1 đã kiểm kê trước khi gọi hồ sơ toàn chuỗi là đầy đủ. Không xóa giới hạn này hoặc thay bằng việc có sẵn trọng số cuối.
