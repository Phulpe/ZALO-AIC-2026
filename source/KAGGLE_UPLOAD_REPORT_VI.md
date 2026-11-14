# Upload ZAIC lên Kaggle — 08/11/2026

Tài khoản quan sát: **Lương Hoàng Phúc / lnghongphc**. Chỉ chọn tệp nguồn trong `D:\ZAIC`. Không chạy huấn luyện, không nộp bài, không thay đổi chia sẻ thành Public.

| Input | Trạng thái đã quan sát | Đối chiếu |
|---|---|---|
| [Mã nguồn v21](https://www.kaggle.com/datasets/lnghongphc/zaic-code-v21-verified-20261108) | Private; tạo thành công; gắn vào notebook mới | 33 file mã/wrapper được kiểm hash trước import; 27 file frozen; 99/99 tests đạt trên CPU Kaggle |
| [SFT baseline 768](https://www.kaggle.com/datasets/lnghongphc/zaic-train-sft-768-frozen-20261108) | Private; tạo thành công; gắn vào notebook mới | SHA256 của SFT, manifest, split trên Kaggle khớp bản máy; không đọc nội dung vào hội thoại |
| [Adapter 9B cuối và checkpoint-48](https://www.kaggle.com/datasets/lnghongphc/zaic-trained-9b-final-checkpoint48-20261108) | Private; tạo thành công; Kaggle báo 22 file; gắn vào notebook mới | Adapter cuối 86.623.432 byte, SHA256 e7dfa635beccce128c99f9d3fd993d33dbb4c88e73e3ddeccbc2bc915eec4a4d; 12 file checkpoint-48 và model_lock khớp hash trên Kaggle |
| [Base Qwen3.5-9B](https://www.kaggle.com/datasets/lnghongphc/zaic-base-qwen35-9b-frozen-c2022362) | Private; tạo thành công; 16 file gồm manifest và 15 assets; gắn vào notebook và sẵn sàng sử dụng | SHA256 15/15 assets trên Kaggle khớp máy; bốn shard đúng index; header, dtype/shape/data_offsets và tập tensor đều hợp lệ. Tổng tensor 19.306.216.416 byte đúng metadata index. ZIP local 19.329.398.314 byte, SHA256 c8dba1174f41c1ded7a3155a6249954175f11bc5ac2dfaba21296aede996fac3 |

Notebook mới: [ZAIC 9B Training Ready 20261108](https://www.kaggle.com/code/lnghongphc/zaic-9b-training-ready-20261108/edit). Notebook Colab lịch sử không chạy Run All và không ghi đè mã.

**Upload hoàn tất.** [Version 2, scriptVersionId=366781944](https://www.kaggle.com/code/lnghongphc/zaic-9b-training-ready-20261108?scriptVersionId=366781944) đã báo Successful, Private; Quick Save kèm output, đủ bốn dataset đã gắn. Đây là phiên bản bổ sung kiểm tra input, không phải lượt train mới. Mặc định notebook vẫn là preflight.

Biên nhận tổng hợp và base từ output đã lưu ở version 2 đã được lấy qua giao diện về staging: `INPUT_INTEGRITY_RECEIPT_KAGGLE_V2.json`, `BASE_INTEGRITY_RECEIPT_KAGGLE_V2.json`. Bản base 276 byte trên máy khớp hash do CPU Kaggle tính: `5c4f4597fc74a68173997d0db69c23020340e4026ea0469bf1aa58b8e599afad`. Bằng chứng ảnh: `FINAL_KAGGLE_VERIFIED_PROOF.jpg`. `LATEST_SYNCED` dưới đây vẫn chỉ mô tả run preflight đã nhập từ version 1; không dùng nó để tuyên bố một lượt train mới đã đồng bộ.

## Phiên bản preflight đã lưu và đồng bộ

- [Version 1, scriptVersionId=366774242](https://www.kaggle.com/code/lnghongphc/zaic-9b-training-ready-20261108?scriptVersionId=366774242): Successful, Private, Quick Save kèm output hiện tại.
- Run `20261108T012323415198Z_preflight_5916a455`: 99 tests, 0 failures, 0 errors.
- Hash ARTIFACT_MANIFEST: `271810b616742ff27add152d99cbf9146e54dd722c287d2884c0642abeeb4380`.
- ZIP output do GUI tải về đã được sao chép nguyên vẹn vào `D:\ZAIC\DOWNLOADS_ARCHIVE\KAGGLE_PREFLIGHT_366774242_results.zip`; Chrome dùng vị trí tải mặc định `E:\results.zip`. Không xóa hoặc ghi đè tệp này.
- Bản run đã kiểm từng file tại `D:\ZAIC\PRIVATE_DATA\runs\20261108T012323415198Z_preflight_5916a455`.
- `D:\ZAIC\PRIVATE_DATA\runs\SYNC_STATE.json`: LATEST_SYNCED là run preflight; BEST_MODEL chưa chọn. Đây không phải một lượt model mới đã học.

## Cổng còn lại trước huấn luyện

1. Base đã hoàn tất upload, gắn vào notebook và đạt kiểm tra hash/header trên CPU. SHA256 biên nhận `BASE_INTEGRITY_RECEIPT.json` được CPU Kaggle tính: `5c4f4597fc74a68173997d0db69c23020340e4026ea0469bf1aa58b8e599afad`.
2. Kaggle đang yêu cầu chủ tài khoản xác minh số điện thoại để dùng GPU/Internet. Chưa thực hiện thao tác xác minh thay người dùng.
3. Kiểm tra thư viện và GPU smoke bằng dữ liệu giả lập trước khi train. Preflight CPU không chứng minh GPU tương thích hay chất lượng mô hình.
   Môi trường CPU đã ghi nhận: Python 3.13.18, torch 2.11.0+cpu, transformers 5.20.2, peft 0.21.2, accelerate 1.15.0; bitsandbytes chưa có. Không cài stack GPU vào phiên CPU; kiểm tra lại image khi chọn accelerator.
4. Notebook mặc định ACTION=preflight, GPU_RUN_CONFIRMED=False, TRAIN_FRESH_ADAPTER=False. Luồng train hiện tại tạo adapter mới từ base; **chưa triển khai resume checkpoint-48**. Input checkpoint được giữ để chuẩn bị riêng luồng resume đã xem xét, không tự nạp optimizer pickle trong kiểm kê.
5. Không tải lại base cố định cho mỗi run. Sau lượt train đã lưu, tải đủ checkpoint/adapter về thư mục run và đối chiếu manifest rồi mới cập nhật LATEST_SYNCED. BEST_MODEL cần kết quả đánh giá tương ứng.

Quyền dùng cloud dựa trên xác nhận trực tiếp của người dùng rằng ban tổ chức cho phép nền tảng thứ ba để huấn luyện; chưa có URL ngoại lệ chính thức để kiểm chứng độc lập. Các input giữ Private. Không đưa dữ liệu thi hoặc nội dung SFT vào ngữ cảnh AI.

Nguồn máy và các checkpoint khác vẫn giữ nguyên. Upload này chỉ chọn hệ 9B: không trộn adapter 4B; không đại diện cho bản sao cloud toàn bộ lịch sử train. Manifest ZIP và trạng thái chi tiết tại `D:\ZAIC\KAGGLE_STAGING\20261107T191003Z`.
