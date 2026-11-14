# Kết quả vLLM no-hint v5 — 11/11/2026

- observed_at: 2026-11-11 06:40 UTC. Kernel: lnghongphc/zaic-khang-direction-gd3-20261110, version 5, trạng thái COMPLETE.
- run_id khang_dir_v1_vllm_nohint_20261111b; init_mode fresh (base Qwen3.5-9B, không dùng adapter); code manifest ee6e9e98…7b33.
- Môi trường: 2×T4, vLLM 0.30.0 (torch 2.13 cu130), fp16, tensor parallel 2, prefix cache bật.
- Output: PRIVATE_DATA/khang_direction_v1/gd3_v5/out. 47/48 file khớp ARTIFACT_MANIFEST. STATUS.json lệch hash vì được ghi thêm bước server_stopped sau khi lập manifest; nội dung đã đọc trực tiếp.

## Thời gian (giây)
| Bước | Thời gian |
|---|---|
| pip install vLLM (online) | 244 |
| Nạp model và khởi tạo server | 720 |
| Probe 4 câu | 501 |
| 34 câu còn lại | 4865 |
| Tổng phiên GPU | 6474 |

- Thông lượng sinh khoảng 18 tok/s; prefix cache trúng 60%.
- Không có lỗi sample. 194 request: 190 kết thúc bình thường, 4 bị cắt do max-tokens.
- Độ trễ mỗi câu: trung vị 834 s, tối đa 2947 s (8 worker chạy song song).

## Độ chính xác trên 38 câu no-hint (TRAIN và calibration)
Gold được dùng sau khi đã khóa dự đoán.

| Cấu hình | Đúng | Sai | Bỏ qua | Điểm ròng |
|---|---|---|---|---|
| vLLM v5, runner min-votes 2 | 26 | 4 | 8 | +22 |
| HF bnb gd3_v2, tham chiếu | 4 | 1 | 33 | +3 |

Theo split:
- train: 17 đúng, 4 sai, 5 bỏ qua.
- calibration: 9 đúng, 0 sai, 3 bỏ qua.

## PA ghép (Tier1 hint, rồi vLLM cho câu không có hint)
| Split | Chỉ Tier1 | Tier1 + vLLM |
|---|---|---|
| train (1817 câu) | 0.9857 | 0.9928 |
| calibration (552 câu) | 0.9783 | 0.9946 |

- Audit (631 câu) chưa chạy phần LLM, vì upload split audit cần người dùng duyệt.
- Tính quyết định trên train: phần LLM chỉ chiếm 26/1817 câu (1,4%), nên chênh lệch PA giữa hai lần chạy tối đa khoảng 0,029, dưới ngưỡng 0,03. Đây là chặn trên tính bằng lý thuyết, chưa chạy kiểm hai lần thật.

## Quyết định
- Giữ cấu hình: no-thinking, 3 sample, min-votes 2, max-tokens 2048, temperature 0.6.
- Không thử chế độ thinking: tỉ lệ chính xác hiện tại đã là 87% (26/30) và thinking chạy chậm hơn nhiều.
- Gói wheel offline (198 file, 3,9 GB, Python 3.13) đã tải xong bằng kernel CPU lnghongphc/zaic-vllm-0300-wheels-20261111, chưa thử cài.

## Phân tích lỗi và quyết định trước public (11/11, CPU)
Script: analysis/error_analysis.py. Chi tiết từng ID lưu ở PRIVATE_DATA/khang_direction_v1/ERROR_ANALYSIS_V5.jsonl; dưới đây chỉ là số tổng hợp.

**8 câu bỏ qua:**
- 5 câu có ít nhất một lần trả lời đúng, nhưng các lần trả lời không trùng nhau (6/8 câu có top_votes=1).
- 7/8 câu có lỗi khi chạy code.

**4 câu sai:**
- 2 câu lệch gold dưới 5% (gần đúng).
- 3 câu có 2 lần trả lời trùng nhau nhưng cùng sai.

**Ngưỡng nộp:**
- Câu có đúng 2 phiếu trùng: 15 đúng, 3 sai. Nâng ngưỡng lên 3/3 sẽ mất nhiều câu đúng hơn số câu sai loại được. Quyết định: giữ min-votes 2 (mục 3, không đổi).
- Câu chỉ có 1 phiếu (chủ yếu loại unanswerable, được chấp nhận ngay): 11 đúng, 1 sai. Giữ nguyên.

**Mục 2 (mở rộng hint):**
- Train chỉ có 18 câu có số đo nhưng chưa sinh được code hint: 7 thiếu bộ lọc, 7 còn tham số "...", 1 thiếu kỳ.
- Mẫu quá nhỏ để viết luật mà không học vẹt.
- Câu có hint được nộp thẳng không qua LLM, nên luật mới có thể tạo câu sai (−1) trên public mà không kiểm chứng được.
- Hoãn.

**Mục 4 (thinking) và mục 5 (audit):**
- Hoãn vì hạn chót public là 22h giờ Việt Nam.
- Audit còn cần người dùng duyệt upload.

**Public kernel v3:**
- Bước 1 chỉ chạy code hint, dùng server LLM giả trả lỗi 400, để mọi câu có hint chắc chắn có đáp án trước khi hết giờ.
- Bước 2 dùng vLLM cho các câu còn bỏ qua, rồi gộp kết quả.
- Đã thử bước 1 trên máy với 30 câu public: chạy hết trong 44 giây, không treo.

## Kết quả public_test (kernel public v3, 11/11/2026)
- Kernel hoàn tất lúc 13:57 UTC, trạng thái vllm_public_done. Phiên GPU 19528 s; đây là batch run, đã COMPLETE nên GPU đã dừng.
- Ngân sách GPU đã dùng 776,7/780 phút. Quota Kaggle đã dùng 22,56/30 giờ.
- 1185/1186 file khớp manifest. STATUS.json lệch hash vì được ghi thêm sau khi lập manifest.
- File submission_public.jsonl: 1000 dòng, ID không trùng; sha256 511ac8b5…2f582.
- Kết quả trả lời:
  - 983 câu có đáp án: 824 câu từ hint, 159 câu từ vLLM.
  - 17 câu bỏ qua.
- Trong 159 câu vLLM trả lời, 144 câu chỉ có 1 lần trả lời: runner tự giảm số lần trả lời mỗi câu khi dự kiến không kịp giờ.
  - Độ chính xác của một lần trả lời đơn lẻ trên train/calibration là 0,73–0,78.
  - Câu có ≥2 lần trả lời trùng nhau đạt khoảng 0,87.
- PA ước tính, chưa có số chính thức: khoảng 0,81–0,91.
  - Phụ thuộc vào độ chính xác của câu có hint trên public, giả định 97–100%.
  - Và độ chính xác của vLLM trên câu public, có thể khó hơn câu train.
- Bản dự phòng chỉ dùng hint: run/public_hint/submission.jsonl, 824 câu trả lời.

## Điểm public chính thức (người dùng tự nộp 11/11/2026 21:29 giờ Việt Nam, mô tả "v1")
- Public Score 0.833. Nguồn: ảnh chụp leaderboard người dùng gửi trong chat.
- Người dùng chưa xác nhận đã nộp file nào. Nếu là submission_public.jsonl (sha 511ac8b5…, 983 câu có đáp án): đúng 908, sai 75, độ chính xác 92,4%.
- Phân rã theo nguồn chưa biết:
  - Nếu hint đúng 100%, vLLM đúng 84/159, độ chính xác khoảng 53%, gần mức hòa điểm.
  - Nếu hint có sai, vLLM tốt hơn mức đó.
- Nộp thêm bản chỉ có hint sẽ tách được hai phần.

## Tách nguồn lỗi bằng bản chỉ có hint (người dùng tự nộp 21:32, mô tả "hint")
- Bản chỉ có hint (sha 7002ab73…, 824 câu có đáp án) được Public Score 0.760, tức đúng 792, sai 32. Độ chính xác của hint trên public là 96,1%; trên train/calibration/audit là 100%, nên các luật hint có dấu hiệu khớp quá sát với train.
- Suy ra phần vLLM trong bản v1: đúng 116, sai 43 trên 159 câu, độ chính xác 73%, cộng ròng 0,073. Con số này khớp với độ chính xác của một lần trả lời đơn lẻ trên train (0,73–0,78).
- Điểm đang mất so với 1,0, tổng 0,167:
  - Hint sai: 32 câu, mất 0,064.
  - vLLM sai: 43 câu, mất 0,086.
  - Câu bỏ qua: 17 câu, mất 0,017.
