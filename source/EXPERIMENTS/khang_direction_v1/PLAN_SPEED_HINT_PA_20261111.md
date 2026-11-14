# Kế hoạch v2: tăng tốc câu không hint, nâng hint, nâng PA (lập 11/11/2026)

Mốc hiện tại, public chính thức do người dùng tự nộp:

| Bản | PA | Có đáp án | Đúng | Sai | Bỏ qua |
|---|---|---|---|---|---|
| v1 đầy đủ | 0,833 | 983 | 908 | 75 | 17 |
| Chỉ hint | 0,760 | 824 | 792 | 32 | – |
| Phần vLLM suy ra | +0,073 | 159 | 116 | 43 | 17 |

Bằng chứng: VLLM_V5_RESULT_20261111.md.

**Điều chỉnh một hiểu lầm:** lần chạy public v1 không bị ngắt. Runner đã chạy đủ 176/176 câu, kết thúc ở giây 17.826. Nguyên nhân mất điểm là runner thấy không kịp giờ nên **tự giảm số lần trả lời xuống 1** cho 144/159 câu. Không có bỏ phiếu, độ chính xác chỉ khoảng 73%; có bỏ phiếu trên train là khoảng 87%. Muốn nâng điểm thì phải tăng tốc để giữ đủ 3 lần trả lời trong ngân sách thời gian.

## A. Đưa phần câu không hint xuống ≤ 3 giờ trên Kaggle 2×T4

Tổng thời gian v1 là 19.528 s (5,4 giờ):

| Thành phần | v1 (s) | Mục tiêu v2 (s) | Cách làm |
|---|---|---|---|
| Chép model và cài vLLM offline | 600 | 600 | Giữ nguyên |
| Chạy câu có hint (CPU) | 446 | 0 (chạy chồng) | Chạy song song khi vLLM đang nạp model |
| Nạp vLLM | 790 | ≤ 600 | Chạy chồng như trên; thử cache torch.compile |
| Câu không hint (176 câu) | 17.826, gần như chỉ 1 lần trả lời/câu | **≤ 8.500, đủ bỏ phiếu** | A1–A5 |
| **Tổng** | 19.528 | **≤ 10.800 (3 giờ)** | |

Các cách tăng tốc, đều đo trước bằng benchmark trên 38 câu no-hint của train/calibration:

- **A1. Tăng số request chạy song song.**
  - v1 mới dùng khoảng 25% KV cache. Khi chạy một luồng, tốc độ sinh chỉ 18 tok/s; T4 bị giới hạn bởi băng thông bộ nhớ, nên chạy batch lớn hơn tăng thông lượng gần tuyến tính.
  - Thay đổi: max-num-seqs 16 → 48, workers 16 → 32.
  - Kỳ vọng nhanh hơn 1,5–2,5 lần.
- **A2. Speculative decoding bằng đầu MTP có sẵn trong Qwen3.5-9B.**
  - Không thêm model, không tính thêm vào giới hạn 15B tham số.
  - Kỳ vọng decode nhanh hơn 1,4–2 lần.
  - Cần kiểm vLLM 0.30 có hỗ trợ trên T4 fp16 không. Nếu không hỗ trợ thì bỏ.
- **A3. Bỏ phiếu theo kiểu chạy song song.**
  - Chạy 2 lần trả lời đồng thời. Chỉ chạy lần thứ 3 khi 2 lần đầu khác nhau (trên train, khoảng 60% câu có 2 lần đầu trùng nhau).
  - Giảm độ trễ mỗi câu. Tổng số token không tăng.
- **A4. Ngân sách cố định thay cho giảm số lần trả lời.**
  - Runner không tự hạ xuống 1 lần trả lời nữa.
  - Câu dài vượt mốc thời gian (ví dụ > 900 s) thì dừng và bỏ qua (0 điểm), thay vì làm chậm cả lô.
  - Có hàng ưu tiên: câu ngắn chạy trước.
- **A5. Giảm token thừa.**
  - max-steps 6 → 5. Chỉ áp dụng nếu train cho thấy bước 6 hầu như không tạo đáp án đúng.
  - Giữ max-tokens 2048.

Ước tính sau A1–A3 nếu cả ba đạt: 17.826 s ở chế độ 1 lần trả lời tương đương khoảng 35.000 s cho 2,2 lần trả lời trung bình mỗi câu. Chia cho mức tăng tốc tổng 4–5 lần, còn khoảng 7.000–8.800 s. **Mục tiêu 3 giờ có khả năng đạt được nhưng chưa được chứng minh**; benchmark sẽ quyết định. Nếu chưa đạt, phương án lùi theo thứ tự:
1. Lượng tử hóa AWQ 4-bit. Phải đo lại độ chính xác.
2. Bỏ phiếu 2/2 rồi dừng.

**Đề private (khoảng 500 câu) trên máy chấm 2×RTX 5090:**
- Ước tính khoảng 88 câu không hint, chạy bf16.
- Mỗi card 32 GB chứa trọn model, nên chạy **2 server độc lập**, mỗi card một bản (không chia model qua 2 card).
- Ước tính dưới 1 giờ, giới hạn BTC là 6 giờ.
- Vẫn bỏ phiếu đầy đủ. Xếp hạng khi hòa điểm theo thời gian chạy ngắn hơn, nên vẫn nên tăng tốc.
- predict.sh cấu hình riêng cho 5090; chưa chạy thử trên 5090.

## B. Bảng log tiến độ theo dõi trực tiếp trên tab Logs của Kaggle

- Mỗi câu hoàn tất in một dòng ra stdout, nên hiện ngay trong tab Logs khi job đang chạy. Đồng thời ghi vào `out/PROGRESS.tsv`.
- Dòng mẫu:

  ```
  PROG | 0123/0176 | q_ab12cd34ef56 | stage=llm | status=answered | type=number | votes=2/2 | samples=2 | 214s | elapsed=01:42:10 | eta=00:58:00
  ```

- **Không in nội dung câu hỏi hay giá trị đáp án**, theo quy định dữ liệu. Chỉ in ID, loại đáp án, số phiếu, thời gian.
- Cứ 10 phút in thêm một dòng tổng kết: số câu đã xong, đã trả lời, bỏ qua, lỗi, tốc độ token/giây, ETA.
- Bước hint cũng in tiến độ theo lô, 50 câu một dòng.

## C. Câu có hint: hai chỉ số khác nhau

- **Độ chính xác hint** hiện 96,1% (sai 32/824). Mỗi câu sai bị −1, nên đây là ưu tiên số 1.
- **Độ phủ hint** hiện 82,4%.

**Ràng buộc:** thể lệ mục 6 cấm phát triển luật trên dữ liệu test; AGENTS cấm đưa nội dung câu hỏi vào ngữ cảnh AI. Vì vậy:
- **Không sửa theo từng câu public.**
- **Không dò leaderboard theo từng ID.**
- Chỉ dùng TRAIN và các biến thể tổng hợp sinh từ TRAIN.

Các bước:
- **C1. Kiểm chéo hint bằng LLM** (tùy chọn có sẵn `--llm-on-code-hint`).
  - Câu có hint vẫn chạy thêm 1 lần LLM có gợi ý.
  - LLM trùng với hint thì nộp. Khác thì chạy thêm 1 lần nữa: 2 lần LLM trùng nhau thì nộp đáp án đó, còn không thì bỏ qua.
  - Chi phí: khoảng 824 lần gọi ngắn, cần đo thời gian.
  - Chỉ bật cho các loại hint có rủi ro, xác định theo C2.
  - Kỳ vọng: biến phần lớn trong 32 câu sai thành câu đúng hoặc bỏ qua, tức +0,02 đến +0,05.
- **C2. Tìm loại hint dễ vỡ trên train.**
  - Sinh tự động các biến thể diễn đạt của câu train: đảo thứ tự từ, từ đồng nghĩa (doanh thu/tiền bán), cách viết kỳ (T3/tháng 3/03-2025), viết tắt, không dấu, thêm hoặc bớt chữ đệm.
  - Loại hint nào đổi đáp án khi đổi cách diễn đạt là loại dễ vỡ: bật C1 cho loại đó và sửa luật cho tổng quát hơn.
  - Đã từng phát hiện lỗi im lặng ở "Q03/quý 03" bằng đúng cách này.
- **C3. Mở rộng độ phủ một cách tổng quát.**
  - Train còn 18 câu có số đo nhưng chưa sinh được code: 7 thiếu bộ lọc, 7 còn tham số "...", 1 thiếu kỳ.
  - Sửa trình điền bộ lọc và tham số để chạy được với mọi tên thực thể khớp trong dữ liệu shop.
  - Kiểm bằng biến thể tổng hợp. Mọi luật mới phải giữ đúng 100% trên train, calibration và audit.
  - Câu có code mới nhưng chưa chắc thì cho qua kiểm chéo C1, không nộp thẳng.
- **C4. Mục tiêu thực tế:**
  - Độ chính xác hint ≥ 99%: sai thì chuyển thành bỏ qua.
  - Độ phủ 85–88%.
  - **Đạt 100% độ phủ cùng 100% độ chính xác trên public không thể cam kết.** Đề có câu thuộc dạng chưa từng gặp trong train.

## D. Lộ trình nâng PA

| Giai đoạn | Nội dung | PA public kỳ vọng | GPU |
|---|---|---|---|
| 0, đã có | v1 | 0,833 | đã dùng |
| 1 | A1–A5 và log B: benchmark trên 38 câu train/calibration | (đo tốc độ) | ~1,5 giờ |
| 2 | C2 và C1 trên CPU, kiểm chéo đo trên train/calibration | (đo trên train) | ~1 giờ |
| 3 | Đóng băng v2, chạy public một lần, ≤ 3 giờ | 0,86–0,90 | ~3 giờ |
| 4 | C3 mở rộng độ phủ; audit một lần (cần duyệt upload) | +0,01–0,03 | ~1 giờ |
| 5 | Thinking cho câu khó, chỉ khi chạy trên 5090; SFT LoRA từ quỹ đạo đúng trên TRAIN (cần ngân sách lớn) | +? | lớn |

**Không hứa PA = 1.** Mức trần khả thi gần nhất là khoảng 0,90. Mỗi bước đều phải có số đo trên train/calibration trước khi chạy public. Không dùng public để chọn luật.

**Ngân sách GPU:**
- Quota Kaggle còn 7,44 giờ đến 14/11 00:00 UTC, sau đó làm mới 30 giờ.
- Giai đoạn 1–3 cần khoảng 5,5 giờ. Ngân sách đã duyệt (780 phút) đã dùng hết, nên **cần người dùng duyệt ngân sách mới**.

## Cập nhật thực hiện 11/11/2026, khoảng 22h giờ Việt Nam
- Runner khang_v3:
  - Log tiến độ PROG/SUMMARY và file PROGRESS.tsv.
  - --fixed-samples, --parallel-first 2, --question-timeout.
  - Bộ test 10/10 đạt, gồm test mới test_runner_v3_progress.py. Kết quả: TEST_KHANG_V3_WSL.txt.
- Dataset code v2 (thêm khang_v3): manifest 11b9fc87…3a09, đã upload lên Kaggle (trạng thái ready).
- Kernel v2 (push_v2/v2_kernel.py):
  - Chạy câu có hint trong lúc vLLM đang nạp model.
  - In log tiến độ trực tiếp ra tab Logs.
  - Thử MTP; lỗi thì chạy lại không MTP.
  - max-num-seqs 48, 32 worker.
  - Benchmark: 2 lượt A/B chạy đồng thời trên 38 câu no-hint. Đang chạy.
- C1 (LLM kiểm chéo hint): **bỏ**. Khi chạy kèm hint, LLM thấy sẵn kết quả của hint nên gần như luôn đồng ý (587/587). Khi tắt hint, LLM chỉ đúng khoảng 73%, sẽ bất đồng với nhiều câu hint vốn đúng. Cả hai cách đều không có lợi.
- C2 (độ bền hint, 600 câu train, 10 kiểu biến thể; HINT_ROBUSTNESS_V2.jsonl):
  - **0 trường hợp ra đáp án sai.**
  - Mất hint: bỏ dấu câu 10, phần trăm ↔ % 8, thêm lời chào 4, teencode 1, cách ghi tháng 1, T3/Q1 1.
  - Kết luận: 32 câu hint sai trên public không đến từ khác biệt cách viết, mà từ cách hiểu nghĩa câu hỏi. Không chẩn đoán được nếu không đọc câu public, mà thể lệ mục 6 cấm phát triển luật trên dữ liệu test.
  - Không sửa luật hint lần này: phần mất hint chỉ làm câu chuyển sang LLM, không gây sai; mỗi sửa đổi phải kiểm lại toàn bộ train.

## Kết quả benchmark v2 (kernel zaic-khang-v2-20261111 v1, COMPLETE 11/11 khoảng 17:00 UTC)
- Phiên GPU 7236 s (121 phút). Output: PRIVATE_DATA/khang_direction_v1/v2_bench/out. 85/86 file khớp hash; STATUS.json lệch vì ghi sau khi lập manifest.
- Cài vLLM offline 181 s. MTP được chấp nhận: tỉ lệ nhận token nháp 34711/47706 = 73%. Nạp server mất 850 s.
- **Thất bại về tốc độ:**
  - Server chỉ chạy được 2 request cùng lúc, KV cache dùng 76–92%, khoảng 110–126 request phải chờ. Thông lượng sinh trung bình 10,2 tok/s, có 55 lần preemption.
  - Client hết thời gian chờ 900 s: 99 lỗi timeout, 14 lỗi hết hạn câu.
  - Lượt A: 0/38 câu có đáp án. Lượt B: 8/38.
  - Không đo được độ chính xác và độ ổn định giữa 2 lần chạy.
- Nguyên nhân: model lai (linear attention) cộng MTP cộng max-num-seqs 48 trên T4 16 GB làm giảm mạnh số request chạy được song song. Hai runner nạp dữ liệu cùng lúc cũng tốn 740 s.
- **Bài học:**
  - Trên T4 giữ cấu hình đã chứng minh ở v5: không MTP, max-num-seqs 16, 8–16 worker.
  - MTP chỉ thử lại trên GPU nhiều bộ nhớ (5090).
  - Kernel cần in số request đang chạy/đang chờ của vLLM ra log, để phát hiện sớm.
- **Ước tính lại:**
  - v5 sinh 131k token cho 38 câu với tốc độ tổng khoảng 24 tok/s. Bỏ phiếu đầy đủ cho 176 câu public cần khoảng 600k token, tức khoảng 7 giờ trên T4.
  - **Mục tiêu ≤ 3 giờ có bỏ phiếu đầy đủ không đạt được trên T4.**
  - Trên 2×5090 (băng thông khoảng 11 lần, bf16, mỗi card một server) vẫn dự kiến dưới 1 giờ cho private.
