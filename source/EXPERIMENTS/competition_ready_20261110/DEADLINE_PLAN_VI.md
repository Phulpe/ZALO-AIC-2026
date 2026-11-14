# Chốt bản thi trước cuối ngày 11/11/2026

## Quyết định mới thay thế phạm vi ngân sách cũ

Người dùng đã làm rõ: **240 phút GPU KHÔNG bao gồm public_test**. Phần ngân sách và dự trữ public_test ở kế hoạch bên dưới là lịch sử, không còn áp dụng. Ngân sách này dành cho smoke, train và đánh giá nội bộ. Chưa có giới hạn GPU riêng được chốt cho public_test; chuẩn bị mã và input trước, chưa khởi chạy public_test.

Tiến độ mới: 288/288 tests Linux CPU đã đạt, source `af590010e0277770820b1c8ab13d55d35a851a8cc45375351c2f715bc24f02d1`. GPU wrapper v1 bị chặn trước optimizer vì cấu hình lịch sử ghi đè CONFIG mới; đã lưu log, đưa notebook về CPU bằng Quick Save v2. Wrapper sửa giữ duy nhất một CONFIG; đang chạy chuẩn bị đầy đủ trên CPU với input thật trước khi cấp GPU lại. Không suy diễn lỗi wrapper thành model đã train thất bại.

Múi giờ: Asia/Saigon. Ngân sách mới do người dùng cấp: **240 phút GPU tổng cộng**, bao gồm smoke, train, đánh giá và public_test. Không cộng ngân sách cũ đã đóng. CPU không được bật GPU. Ghi nhận tiêu thụ thực tế và thời gian chưa xác minh trước mỗi phiên mới.

## Thứ tự ưu tiên

1. Hoàn thiện driver dùng bank frozen, tránh thu lại dự đoán đã có bằng chứng. Warm-start E7, optimizer/scheduler mới. Pilot đề xuất 18 bước, LR 1e-5, batch 1, accumulation 4, time weight 4. Không tự chuyển fresh/resume. Đây là giả thuyết cần đo, không phải cấu hình tối ưu đã được chứng minh.
2. Kiểm thử CPU toàn bộ source mới và kiểm hash input. Bản v4 trước có 282 tests Linux CPU đạt; thêm 6 tests driver đang chạy trên Kaggle. Bằng chứng cũ không được tính là lần chạy source mới.
3. Chạy GPU smoke theo đúng fingerprint source/parent/dataset, loss và gradient hữu hạn, mask/shift đúng. Dùng subprocess tách riêng và chờ kết thúc trước train. Không được tạo receipt giả từ kết quả CPU.
4. Một pilot từ bank 9 correction + 9 replay đã freeze. Lưu từng checkpoint và trạng thái đầy đủ; bảo toàn E7 và mọi candidate cũ. Driver mới không tự promote hoặc chạy public.
5. Đánh giá mù nhóm 7 lỗi cũ, cả 9 correction và toàn bộ 103 replay. Nếu không đạt cổng: giữ candidate, dừng nhánh đó. Chỉ chạy thêm optimizer nếu có chẩn đoán thực chứng, không lặp cùng cấu hình để tiêu hết quota.
6. Khi vượt regression, chạy calibration cặp trước/sau cùng source/executor/config. Không tái dùng baseline calibration khác chính sách executor. Bộ calibration có exposure lịch sử phải được ghi rõ. Chỉ chọn BEST_MODEL theo tiêu chí đã freeze.
7. Dành ít nhất 60 phút ngân sách cho public_test và xác minh kết quả. Chỉ chạy một model đã chọn, không học từ public gold. 1.000 ID phải đủ/duy nhất, schema hợp lệ; nếu quá hạn thì lưu trạng thái thật, không điền đáp án giả.
8. Đóng băng mã/model/config, tải artifact và kiểm checksum. Dừng GPU ngay sau tác vụ; kiểm phiên nền, phiên tương tác theo phạm vi CLI/API. Không tự nộp bài.

## Mốc dừng theo hạn

- Ngày 10/11: ưu tiên hoàn tất runner, smoke và pilot đầu; tiến độ phụ thuộc hàng đợi Kaggle và các cổng thực tế.
- Ngày 11/11: ưu tiên đánh giá và chốt model; ngừng thử nghiệm mới trước 18:00 giờ Việt Nam để giữ thời gian đóng gói.
- Nếu candidate không đạt: dùng baseline bảo toàn làm phương án dự phòng. Không gọi model đó có điểm cạnh tranh cao khi chưa có bằng chứng.
- Mục tiêu bàn giao trước cuối ngày 11/11: bản dự thi chạy được, tái nạp được, có evidence và checksum. Không bảo đảm điểm cao nhất hoặc thời gian hoàn tất hàng đợi bên ngoài.

## Hiện trạng lúc viết

- Đã tạo bản source thí nghiệm riêng; không sửa source v4 đã freeze.
- Đã viết `src/frozen_pilot.py`: mặc định CPU, cấm recollection/public, khóa 18 bước, chặn thực thi khi thiếu budget/GPU smoke khớp fingerprint.
- 6 tests driver mới đạt trên CPU máy cá nhân. Job CPU Linux: `lnghongphc/zaic-final-readiness-cpu-20261110/1`, kỳ vọng 288 tests; chưa gọi đạt trước khi tải kết quả.
- Chưa chạy GPU cho vòng mới; chưa cải thiện trọng số hoặc cập nhật BEST_MODEL.
- Còn tích hợp smoke thật và driver đánh giá/calibration với runner mới; `frozen_pilot.py` hiện dừng sau lưu checkpoint, không được coi là toàn workflow đã hoàn thiện.

Ngân sách chi tiết: `GPU_BUDGET.json`. Cấu hình chuẩn bị CPU: `LOCAL_PILOT_CONFIG.json`, không dùng nguyên đường dẫn Windows trên Kaggle.
