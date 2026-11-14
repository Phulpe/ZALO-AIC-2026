# CPU v4 — chuẩn bị vòng học tiếp theo

Thời điểm: 2026-11-10T01:15:51.941534+07:00 (giờ Việt Nam). Thư mục giữ tên theo lúc tạo ngày09/11; báo cáo hoàn tất ngày10/11/2026.

## Đã thực hiện

- Làm hoàn toàn bằng công cụ nội bộ và Kaggle CLI/SDK. Không mở web, không chạy model/GPU/optimizer mới.
- Bản nguồn mới: `D:\ZAIC\EXPERIMENTS\relearning_cpu_v4_20261109\src`. Baseline/app/tools giữ nguyên byte/hash; không sửa frozen_v21, model, adapter, checkpoint hay tracker.
- Đã tạo venv CPU riêng bằng gói PyPI chính thức: pandas2.3.3 để đọc cache cùng phiên bản, numpy2.2.6 và openpyxl3.1.5. Không thay môi trường cũ, không thay thiết lập symlink/bảo mật Windows.

## Kiểm tra thực tế nguồn và lỗi

- Tái thực thi124 dự đoán E7 đã lưu, xác minh manifest trước đọc. Kết quả103/124 đúng trước và sau sửa; correct→wrong=0. Đây không phải dự đoán mới của candidateD31, không phải holdout/public và không cộng lại thành coverage mới. Toàn learning pool còn1693 pending.
- Ba trường hợp từng báo Unparsed numeric amounts có dòng ghi chú bị kéo metadata giao dịch xuống. Bản sửa chỉ loại dòng được đối chiếu với XLSX gốc đúng checksum: marker ghi chú rõ, ô gốc ngày/ID/số lượng/đơn giá/thành tiền đều trống, nhãn nguồn và cached row không mơ hồ. Đọc công thức bằng data_only=False; không coi công thức chưa tính là ô trống. Không điền tiền hay sinh số0 để vượt guard.
- Bản sửa xử lý23 dòng ghi chú đã chứng minh trong các nguồn liên quan; ba trường hợp thoát exception nhưng đáp án vẫn không khớp gold. Vì thế không báo tăng accuracy và không đưa chúng làm target.
- Còn12 unresolved:1 chưa có bằng chứng phủ thời gian,11 canonical plan chưa khớp số với gold. Chưa xác định nguyên nhân cuối giữa ingestion, quy tắc nghiệp vụ, scorer và nhãn; không tự gán tất cả thành lỗi model hay sửa gold. Sai khác không thuộc nhóm lỗi làm tròn nhỏ hoặc hệ số tiền tệ đơn giản trong phép kiểm đã chạy.
- Đã kiểm348.618 dòng nguồn XLSX/SQLite:2.222 dòng trùng hoàn toàn bằng số dòng trùng sau chiếu cột; không thấy mất phân biệt dòng do phép chiếu trong phạm vi này. Không đổi cơ chế dedup dựa trên giả thuyết chưa được chứng minh. Có48 ID xuất hiện ở nhiều tháng trong1 doanh nghiệp; chưa có bằng chứng độc lập để đổi namespace ID.
- Bảy correction cũ đều chỉ sai time so với target yesterday. Cả7 đã có trong9 correction; lỗi không phải do bỏ chúng khỏi dataset. Giữ log và dữ liệu chẩn đoán chi tiết trong PRIVATE_DATA, chỉ xuất thống kê/hashes ra báo cáo.

## Correction/replay đã đóng băng

- Bank có9 correction và103 replay đã được xác minh bằng nguồn thực thi/scorer. Không tạo nhãn mới; giữ riêng12 unresolved. Evidence cũ chỉ được tái dùng sau khi đối chiếu hash bộ kiểm thực thi, dữ liệu, prompt và target; chính sách chọn mẫu mới có hash riêng.
- Chọn9 correction+9 replay,12 doanh nghiệp. Replay phủ5 nhóm: last_week2,month2,this_week2,today2,yesterday1; trước đây thiếu this_week. Không nhân bản mẫu để lấp quota.
- Giữ cả103 replay làm tập hồi quy TRAIN cố định. Mẫu sửa chỉ do executor, có plan không đổi, bị loại khỏi supervision correction; không bắt model học để che bug code.
- Dataset: `D:\ZAIC\PRIVATE_DATA\relearning_cpu_v4_temporal_dataset_20261110`. Manifest `a0e11f70c0c8e2b29b983e265a77f4e30ed0436ef6714fef4e4c2cfe377720c3`. Hash18 mẫu `8e25ca89eabb57766046136caafa34b60a13a21edbdcc4de009504fe358e671f`. Các file PRIVATE chứa nội dung thật, không đưa vào chat hay repo công khai. Đây là bản local đã freeze, chưa khẳng định đã upload input mới lên Kaggle.

## Runner và cổng đánh giá

- `src/isolated_training.py`: optimizer chạy đúng một child, parent đợi child thoát rồi mới đánh giá. Có timeout hữu hạn, log kín, giữ artifact khi lỗi, kiểm manifest/parent, chặn ghi vào vùng model/source được bảo vệ; không fallback fresh/resume.
- `src/time_pilot.py`: đã gọi child train; hỗ trợ tham chiếu lịch sử riêng và bank hồi quy đóng băng.
- `src/protected_regression.py`: sinh một lần trên9 correction+103 replay, rồi tính nhóm7 lỗi cũ từ dự đoán đã khóa bằng CPU; không sinh lại sau xem gold. Ca kiểm thử toàn luồng xác nhận: tăng correct ở mẫu mới nhưng nhóm cũ fail vẫn bị chặn, hoặc có bất kỳ correct→wrong nào thì không promote.
- Có code và tests CPU cho pipeline mới; chưa gọi là đã chứng minh giải phóng VRAM trên GPU. Full resume vẫn chưa triển khai/xác minh.

## Kiểm thử và bằng chứng

- Kaggle Linux CPU:75 app+24 tools+183 tests thí nghiệm=**282/282 đạt**,0 skip. Bao gồm230 tests trước và52 tests bổ sung. Kết quả: `KAGGLE_CPU_RESULT_V5/TEST_RESULTS.json`, chi tiết `TEST_DETAILS.txt`.
- Source archive `edaa634f1e7ea7e392400c672956cfa3d88ae36013db124b45c0e992d2a510d5`. CLI ref `lnghongphc/zaic-relearning-v4-cpu-tests-20261110/5`. Các version kiểm thử trước được giữ; version5 là bằng chứng cuối.
- Windows không cấp quyền tạo symlink cho một số test baseline và khác biểu diễn path; không sửa baseline hoặc bật quyền hệ thống để né. Kiểm tra toàn bộ được xác nhận trên Linux. Kiểm tra nguồn thật chạy Windows CPU; chưa coi đó là GPU/batch parity mới.
- Các bản nháp bị đổi source trong lúc đối chiếu được giữ riêng; guard hash đã chặn việc công nhận bản nháp đó. Chỉ dataset R3 đã seal và bản chọn temporal mới ở trên là evidence được dùng.
- BaselineE7, B306, adapter28, submission cũ và toàn runD31 được kiểm lại hash, còn nguyên. RUN_REGISTRY không thêm một lượt train giả cho công việc CPU; LATEST_CHECKPOINT/LATEST_SYNCED/BEST_MODEL giữ nguyên.

## Cấu hình đề xuất và phần còn thiếu

`NEXT_RUN_PROPOSAL.json` là đề xuất để xem xét, không phải lệnh đã được cấp quyền chạy. Đề xuất một pilot warm-start E7, optimizer/scheduler mới, tối đa18 steps (~4 epoch tương đương),LR1e-5,time-weight4,9+9 mẫu,checkpoint mỗi bước; ngân sách đề xuất tối đa120 phút gồm kiểm tra và calibration có điều kiện. Lý do: pilot9steps trước giảm NLL nhưng vẫn0/7 lỗi cũ; thêm bước chỉ là giả thuyết phải đo, không cam kết tối ưu.

Chưa mở vòng GPU mới. Trước chạy cần ngân sách mới, CPU preflight đường dẫn/input Kaggle thật, smoke runner mới và đánh giá so sánh đã khóa. Chỉ promote sau cổng7 lỗi cũ,9 correction,0 regression/103 replay và calibration đạt; không dùng gold public. Calibration552 đã tiếp xúc lịch sử, không gọi untouched holdout. Không đưa12 trường hợp chưa xác minh vào SFT và không hứa PA tăng.

Phạm vi đã hoàn thành: code, kiểm thử CPU, chẩn đoán nguồn và dataset local đóng băng. Phạm vi chưa chứng minh: GPU cho runner v4, model mới tốt hơn, full resume và tái lập bit-for-bit. Tham khảo IMPLEMENTATION_STATUS.json, DATASET_INVENTORY.json, ORIGINALS_PRESERVED.json và CLI_FINAL_STATUS.json.
