# Kiểm kê tài sản trên máy

Có 247 tệp / 199,818,238 bytes. SHA256 đã tính cho từng tệp hiện có.

Hash mới là kiểm kê cục bộ; chỉ hash có đối chứng nguồn mới xác nhận truyền nguyên vẹn.

| Nhóm | Tệp hiện có | Bytes | Trạng thái |
|---|---:|---:|---|
| TRAIN | 16 | 26260166 | chưa xác minh đầy đủ |
| public | 7 | 7663133 | chưa xác minh đầy đủ |
| SFT/manifest | 1 | 1462 | chưa xác minh đầy đủ |
| base/tokenizer | 4 | 6722959 | chưa xác minh đầy đủ |
| adapter 9B | 0 | 0 | còn thiếu |
| checkpoints | 3 | 19992187 | chưa xác minh đầy đủ |
| repair_v21 | 118 | 18297096 | chưa xác minh đầy đủ |
| resume_own | 35 | 13475765 | chưa xác minh đầy đủ |

Bộ verified: đạt kiểm tra 9 tên tệp, 1000 ID duy nhất và hash submission

Thiếu: TRAIN/public questions + rulebook, TRAIN answers; base config/tokenizer_config/index và toàn bộ shards; adapter 9B + config; frozen SFT + manifest; notebook gốc.

Checkpoint chỉ hiện diện một phần: chưa có bằng chứng đủ optimizer/scheduler/RNG/adapter ở cùng step.
Repair/resume và lịch sử ngoài zaic: cần kiểm kê đệ quy từ nguồn Drive để đối chiếu đầy đủ.
Không tải lại bộ verified nếu đối chứng nguồn còn khớp. Không coi .metadata/.lock là file đích.
