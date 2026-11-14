# Kiểm kê bảo toàn trọng số ZAIC — 08/11/2026

Phạm vi: D:\ZAIC, gồm PRIVATE_DATA/weights_pending và project mã đã chuyển vào thư mục giải nén. Không sửa trọng số, không unpickle, không GPU, không upload.

SHA256 tính trên toàn bộ bytes; header safetensors kiểm length, JSON, shape/dtype, offsets, không chồng lấn/lỗ hổng và kích thước payload. Không chứng nhận giá trị tensor hay tương thích runtime.

## Adapter và checkpoint

| Thư mục | Base từ config/bằng chứng | Step | Bytes adapter | Cấu trúc adapter | Bộ trạng thái resume |
|---|---|---:|---:|---|---|
| D:\ZAIC\PRIVATE_DATA\zaic\adapter_intent_v0_20steps_20261106_043640 | Qwen/Qwen3-4B-Instruct-2507 | None | 132187888 | đạt | thiếu: optimizer.pt, scheduler.pt, scaler.pt, rng_state.pth, trainer_state.json |
| D:\ZAIC\PRIVATE_DATA\zaic\adapter_intent_v0_run1_20261106_044117\final | Qwen/Qwen3-4B-Instruct-2507 | None | 132187888 | đạt | thiếu: optimizer.pt, scheduler.pt, scaler.pt, rng_state.pth, trainer_state.json |
| D:\ZAIC\PRIVATE_DATA\zaic\adapter_intent_v0_run1_20261106_044117\checkpoints\checkpoint-125 | Qwen/Qwen3-4B-Instruct-2507 | 125 | 132187888 | đạt | đủ tệp, chưa chạy xác minh |
| D:\ZAIC\PRIVATE_DATA\zaic\adapter_intent_v0_run1_20261106_044117\checkpoints\checkpoint-150 | Qwen/Qwen3-4B-Instruct-2507 | 150 | 132187888 | đạt | đủ tệp, chưa chạy xác minh |
| D:\ZAIC\PRIVATE_DATA\zaic\datamate_v1\candidate_20261106_090435\final | Qwen/Qwen3-4B-Instruct-2507 | None | 132187888 | đạt | thiếu: optimizer.pt, scheduler.pt, scaler.pt, rng_state.pth, trainer_state.json |
| D:\ZAIC\PRIVATE_DATA\zaic\datamate_v1\candidate_20261106_090435\checkpoints\checkpoint-80 | Qwen/Qwen3-4B-Instruct-2507 | 80 | 132187888 | đạt | đủ tệp, chưa chạy xác minh |
| D:\ZAIC\PRIVATE_DATA\zaic\datamate_v1\candidate_20261106_090435\checkpoints\checkpoint-83 | Qwen/Qwen3-4B-Instruct-2507 | 83 | 132187888 | đạt | đủ tệp, chưa chạy xác minh |
| D:\ZAIC\PRIVATE_DATA\zaic\datamate_v1\offline_export\assets\adapter | Qwen/Qwen3-4B-Instruct-2507 | None | 132187888 | đạt | thiếu: optimizer.pt, scheduler.pt, scaler.pt, rng_state.pth, trainer_state.json |
| D:\ZAIC\PRIVATE_DATA\zaic\datamate_v1\qwen35_9b\candidate\final | Qwen3.5-9B | None | 86623432 | đạt | thiếu: optimizer.pt, scheduler.pt, scaler.pt, rng_state.pth, trainer_state.json |
| D:\ZAIC\PRIVATE_DATA\zaic\datamate_v1\qwen35_9b\candidate\checkpoints\checkpoint-40 | Qwen3.5-9B | 40 | 86623432 | đạt | đủ tệp, chưa chạy xác minh |
| D:\ZAIC\PRIVATE_DATA\zaic\datamate_v1\qwen35_9b\candidate\checkpoints\checkpoint-45 | Qwen3.5-9B | 45 | 86623432 | đạt | đủ tệp, chưa chạy xác minh |
| D:\ZAIC\PRIVATE_DATA\zaic\datamate_v1\qwen35_9b\candidate\checkpoints\checkpoint-48 | Qwen3.5-9B | 48 | 86623432 | đạt | đủ tệp, chưa chạy xác minh |
| D:\ZAIC\PRIVATE_DATA\zaic\datamate_v1\qwen35_9b\training_code\checkpoints\checkpoint-48 | Qwen3.5-9B | 48 | 86623432 | đạt | đủ tệp, chưa chạy xác minh |
| D:\ZAIC\PRIVATE_DATA\zaic\datamate_v1\qwen35_9b\training_code\teacher_4b_adapter\final | Qwen/Qwen3-4B-Instruct-2507 | None | 132187888 | đạt | thiếu: optimizer.pt, scheduler.pt, scaler.pt, rng_state.pth, trainer_state.json |

## Bộ base và weights_pending

### D:\ZAIC\PRIVATE_DATA\zaic\datamate_v1\offline_export\assets\base

Config: {'model_type': 'qwen3', 'architectures': ['Qwen3ForCausalLM'], 'text_hidden_size': 2560, 'text_vocab_size': 151936}. Đủ nội dung shard: True; khớp tổng bytes tensor index: False; nạp được theo đường dẫn hiện tại: False.

| Tên index yêu cầu | Tệp khớp header | SHA256 đối chứng nguồn | Trạng thái |
|---|---|---|---|
| model-00001-of-00003.safetensors | D:\ZAIC\PRIVATE_DATA\weights_pending\model-00001-of-00003-012.safetensors | True | downloaded_but_not_at_index_path |
| model-00002-of-00003.safetensors | D:\ZAIC\PRIVATE_DATA\weights_pending\model-00002-of-00003-013.safetensors | True | downloaded_but_not_at_index_path |
| model-00003-of-00003.safetensors | D:\ZAIC\PRIVATE_DATA\zaic\datamate_v1\offline_export\assets\base\model-00003-of-00003.safetensors | True | present_at_expected_path |
### D:\ZAIC\PRIVATE_DATA\zaic\datamate_v1\qwen35_9b\assets\base

Config: {'model_type': 'qwen3_5', 'architectures': ['Qwen3_5ForConditionalGeneration'], 'text_hidden_size': 4096, 'text_vocab_size': 248320}. Đủ nội dung shard: True; khớp tổng bytes tensor index: True; nạp được theo đường dẫn hiện tại: False.

| Tên index yêu cầu | Tệp khớp header | SHA256 đối chứng nguồn | Trạng thái |
|---|---|---|---|
| model.safetensors-00001-of-00004.safetensors | D:\ZAIC\PRIVATE_DATA\weights_pending\model.safetensors-00001-of-00004-017.safetensors | True | downloaded_but_not_at_index_path |
| model.safetensors-00002-of-00004.safetensors | D:\ZAIC\PRIVATE_DATA\weights_pending\model.safetensors-00002-of-00004-016.safetensors | True | downloaded_but_not_at_index_path |
| model.safetensors-00003-of-00004.safetensors | D:\ZAIC\PRIVATE_DATA\weights_pending\model.safetensors-00003-of-00004-015.safetensors | True | downloaded_but_not_at_index_path |
| model.safetensors-00004-of-00004.safetensors | D:\ZAIC\PRIVATE_DATA\weights_pending\model.safetensors-00004-of-00004-014.safetensors | True | downloaded_but_not_at_index_path |

## Giới hạn và bổ sung

Adapter 9B cuối: {"path": "D:\\ZAIC\\PRIVATE_DATA\\zaic\\datamate_v1\\qwen35_9b\\candidate\\final", "model": "Qwen3.5-9B", "evidence": {"config_base": "/content/drive/MyDrive/zaic/datamate_v1/qwen35_9b/assets/base", "config_revision": null, "resolved_base_path": "D:\\ZAIC\\PRIVATE_DATA\\zaic\\datamate_v1\\qwen35_9b\\assets\\base", "resolved_model_type": "qwen3_5", "rank": 8, "alpha": 16}, "checkpoint_step": null, "checkpoint_name_step": null, "adapter_bytes": 86623432, "adapter_sha256": "e7dfa635beccce128c99f9d3fd993d33dbb4c88e73e3ddeccbc2bc915eec4a4d", "safetensors_valid": true, "adapter_files_structurally_complete": true, "missing_resume_files": ["optimizer.pt", "scheduler.pt", "scaler.pt", "rng_state.pth", "trainer_state.json"], "resume_files_present": false, "resume_execution_verified": false, "inference_execution_verified": false, "tokenizer_files": ["tokenizer.json", "tokenizer_config.json"], "source_integrity": "known_9b_hash_match"}

Không kết luận đã bảo toàn đầy đủ: chưa có danh sách nguồn đầy đủ mọi lần huấn luyện. Các hash không có đối chứng nguồn chỉ là dấu vân tay cục bộ.
Base 9B nếu đủ nội dung nhưng sai tên/vị trí: chưa cần tải lại các shard đã khớp hash; chỉ đề xuất sắp xếp sau phê duyệt, chưa thực hiện.
Base ba shard: xem dòng missing_or_unidentified để biết file cần tìm/tải bổ sung. Không trộn với bộ 9B.
Đủ tệp resume không chứng minh optimizer/scaler/RNG đọc được hoặc resume đúng; chỉ kiểm bytes/hash, không thực thi pickle.

## File chưa xác định

## ZIP và file tải dở

Không có ZIP hay file tải dở mang đuôi .crdownload/.part/.partial/.tmp trong phạm vi. Có 16 file .lock rỗng, ghi riêng trong lock_files của JSON; không tự coi là tải dở.

## Phân loại bổ sung từng tệp weights_pending

| File hiện tại | Base | Tên index yêu cầu | Bytes | SHA256 |
|---|---|---|---:|---|
| model-00001-of-00003-012.safetensors | Qwen/Qwen3-4B-Instruct-2507 | model-00001-of-00003.safetensors | 3957900840 | 75311d91bb08cf0b882913da464a1e722a31fb44db35208663487efb7a3d8ed6 |
| model-00002-of-00003-013.safetensors | Qwen/Qwen3-4B-Instruct-2507 | model-00002-of-00003.safetensors | 3987450520 | 0b48adbb1f60e901153d91907ba11ce63bd4b8b584482e730f48808d055dfba1 |
| model.safetensors-00001-of-00004-017.safetensors | Qwen/Qwen3.5-9B | model.safetensors-00001-of-00004.safetensors | 5276436216 | db6f444b43d318c92f360a13a25561a6a65b10c0631b8ed305a426dbaa6c380e |
| model.safetensors-00002-of-00004-016.safetensors | Qwen/Qwen3.5-9B | model.safetensors-00002-of-00004.safetensors | 5335161512 | 31c7d7e2dd5d207840b31cc59083c8f4c4718959149e0358c0364052bb9a0330 |
| model.safetensors-00003-of-00004-015.safetensors | Qwen/Qwen3.5-9B | model.safetensors-00003-of-00004.safetensors | 5368717440 | 7ec36ba3a4176a44c3c0876ad80c56a2f70c84bf008d82e9501df642f17dadec |
| model.safetensors-00004-of-00004-014.safetensors | Qwen/Qwen3.5-9B | model.safetensors-00004-of-00004.safetensors | 3325995712 | b62b0c4cd7e44edee103ee8f4fe225f246d5e768e07bfd5f25b63a8aa1fdd0c6 |

Đã tìm thấy 11/11 vị trí adapter/checkpoint đã nêu; các bản bổ sung ghi riêng trong JSON.

Bộ 4B được nhận diện bởi config Qwen3ForCausalLM, cấu trúc tensor theo index và model lock Qwen/Qwen3-4B-Instruct-2507. Đã tìm thấy offline_export/manifest.json và xác nhận SHA256/kích thước cả 3 shard đều khớp; xem bất nhất metadata dưới đây.

Không cần tải bổ sung 4 shard 9B đã khớp manifest nguồn. Không có shard nào thiếu nội dung trong hai bộ index đã kiểm; tên/vị trí hiện tại vẫn chưa đáp ứng loader.

Cần bổ sung bằng chứng: hash nguồn cho checkpoint/optimizer/RNG và các adapter chưa có đối chứng trong JSON; danh sách đầy đủ mọi run trên Drive để xác nhận toàn bộ lịch sử. Không cần tải lại các shard đã khớp manifest.

Mọi bản cùng tên khác hash được giữ riêng. Không chọn BEST_MODEL từ thời gian mới nhất. Không thử nạp file pt/bin/pickle.

Phạm vi không bao gồm các ổ/thư mục ngoài D:\ZAIC; chưa chứng nhận mọi tài sản trên máy/Drive đã được sao lưu đầy đủ.

## Kết quả đối chiếu cuối cùng với manifest 4B

- Cả 3 shard Qwen3-4B-Instruct-2507 khớp kích thước và SHA256 trong offline_export/manifest.json; tên tensor khớp index và mọi header/payload boundary đều hợp lệ.
- Tuy nhiên index ghi total_size=8.045.591.552, còn tổng payload tensor thực tế=8.044.936.192 bytes: lệch **655.360 bytes**. Đây là bất nhất metadata đã ghi nhận; chưa xác minh nguyên nhân. Không có bằng chứng từ việc này rằng file tải bị cắt, vì cả 3 hash khớp manifest và header không bị cắt. Chưa chứng nhận loadable/runtime, chưa sửa index.
- Base 9B: cả 4 shard khớp manifest nguồn; 775 tên tensor khớp index, tổng payload 19.306.216.416 bytes khớp index. Config cũng khớp manifest nguồn. Khóa model ghi revision c202236235762e1c871ad0ccb60c8ee5ba337b9a; adapter_config tự nó không ghi revision, nên định danh revision dựa thêm vào model_lock và manifest.
- Không có safetensors nào chưa xác định trong phạm vi quét. Hai file model-... trong weights_pending thuộc bộ 4B, không phải 9B.
- Không tìm thấy ZIP hay .crdownload/.part/.partial/.tmp trong D:\ZAIC tại lúc kiểm. Có 16 tệp .lock rỗng: đây là khóa phối hợp/cache, không tự chứng minh download đang dở và không phải shard rỗng.
- Đủ tệp trạng thái ở cả 8 thư mục checkpoint (gồm bản sao checkpoint-48 trong training_code); trainer_state.global_step khớp tên 40/45/48/80/83/125/150. Chưa deserialize hoặc chạy resume.

**Kết luận:** các vị trí adapter/checkpoint đã nêu đều tìm thấy; trọng số base hai bộ đã có và khớp manifest bàn giao. Chưa sẵn sàng suy luận ngay vì 6 shard còn ở weights_pending với tên khác index; bộ 4B còn bất nhất metadata total_size. Chưa kết luận bảo toàn toàn bộ lịch sử ngoài danh sách nguồn đã biết.


## Cập nhật sau đổi tên được người dùng cho phép

Đã đổi tên/chuyển 6 shard từ weights_pending vào đúng base 9B/4B; SHA256 sau chuyển đều khớp. Xem WEIGHTS_INVENTORY_AFTER_RENAME.json để lấy đường dẫn hiện hành. Các nhận xét về sai tên/vị trí phía trên là trạng thái trước chuyển; bất nhất metadata total_size 4B vẫn còn. Adapter/checkpoint cũ không thay đổi. Gói upload 9B: D:\ZAIC\KAGGLE_STAGING\20261107T191003Z
