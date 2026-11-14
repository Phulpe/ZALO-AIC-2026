# training.md — Học lại từ lỗi sai và bảo toàn năng lực ZAIC

Cập nhật: 08/11/2026, giờ Việt Nam. Thay thế phương hướng trong trainning.md cũ. Đây là đặc tả để agent triển khai, chưa phải tính năng đã được cài hoặc trọng số mới đã được huấn luyện. Đọc D:\ZAIC\AGENTS.md và báo cáo kiểm toán thực tế trước khi chạy.

## 1. Yêu cầu mới của người dùng

Trong chế độ học trên TRAIN, xử lý tất cả câu hỏi: luôn cố sinh một dự đoán, kể cả chưa chắc chắn hoặc dự đoán sai. Không loại một câu khỏi vòng học chỉ vì model muốn abstain, plan lỗi, thiếu khả năng executor hoặc không đủ token. Câu sai phải có chẩn đoán và được đưa vào hàng đợi khắc phục; câu đúng phải được lưu bằng chứng và được kiểm tra lại để hạn chế quên.

Mục tiêu là học từ phản hồi có kiểm chứng và tăng khả năng làm đúng khi thi thật. Không đồng nhất “trả lời mọi câu” với “mọi câu đều đúng” hoặc cam kết PA=1.

**Không tồn tại một bộ trọng số riêng tương ứng với từng câu đúng trong LoRA hiện tại.** Sinh câu trả lời và chấm đúng không tự cập nhật trọng số. Chỉ optimizer step trong huấn luyện mới thay đổi adapter; một bước có thể ảnh hưởng nhiều năng lực. Lưu checkpoint bảo toàn trạng thái model tại một thời điểm, không chứng minh đã giữ vĩnh viễn kiến thức của một câu. Muốn giữ năng lực phải kết hợp checkpoint bất biến, replay mẫu đúng và kiểm thử hồi quy sau mỗi vòng.

## 2. Hai chế độ chạy độc lập

| Chế độ | Hành vi |
|---|---|
| LEARNING_FORCE_ATTEMPT | Mọi câu trong learning pool của TRAIN đều được thử trả lời; chấp nhận sai để chẩn đoán; không xuất làm bài nộp |
| EVALUATION_OR_SUBMISSION | Model/config đã khóa, không học từ gold; chính sách trả lời/abstain chọn trước bằng calibration và thể lệ |

Yêu cầu thử mọi câu thay thế hướng dẫn cũ về abstain trong chế độ học. Không xóa guard nghiệp vụ dùng chung để thực hiện yêu cầu này. Thêm đường chạy thí nghiệm riêng có cờ learning_mode, giữ output chẩn đoán tách với output thi.

Mỗi câu học phải có attempt record, kể cả exception, timeout, OOM hoặc dữ liệu không đủ. Trạng thái lỗi phải được ghi trung thực, không giả một câu trả lời thành công. Retry có giới hạn; không lặp vô hạn cho đến khi tình cờ khớp gold.

Nếu nguồn không đủ để suy ra đáp án, vẫn ghi dự đoán tốt nhất hoặc nhận định không đủ dữ liệu, uncertainty và nguyên nhân. Không bịa số rồi coi là kiến thức đã xác minh. Nếu gold đúng là unanswerable, dự đoán unanswerable có căn cứ là một lần trả lời hợp lệ, không phải bỏ câu.

## 3. Baseline phải giữ nguyên

- Base: Qwen/Qwen3.5-9B, revision c202236235762e1c871ad0ccb60c8ee5ba337b9a.
- Adapter9B cuối: 86.623.432 byte; SHA256 e7dfa635beccce128c99f9d3fd993d33dbb4c88e73e3ddeccbc2bc915eec4a4d.
- Baseline lịch sử: QLoRA NF4, rank8/alpha16/dropout0.05, all-linear, max_length2048, lr2e-5, batch1/accum16,1epoch/48steps,768rows; completion-only loss.
- Checkpoint9B đã thấy:40/45/48; không ghi đè. Model9B khởi tạo adapter mới từ base, không kế thừa trọng số4B. Không merge4B vào9B để “gom kinh nghiệm”.
- V2/v2.1 sửa loader/contracts/executor, không train adapter mới. Public v2.1 có235answered/765abstain trong1000câu, chưa có PA chính thức mới được xác minh trong bàn giao.
- Kết quả TRAIN v2.1: calibration200 có91đúng/21sai/88abstain,PA0.35; confirmation100 có36đúng/6sai/58abstain,PA0.30. Các tập này đã mở, không gọi là untouched holdout.

Kiểm audit/registry mới nhất để cập nhật trạng thái; dữ liệu trên không thay thế kiểm tra hash thực tế. Lưu bản sao baseline và environment/source/data manifests trước thay đổi.

## 4. Dữ liệu được phép học và cách tách phần đánh giá

“Học mọi câu” áp dụng cho toàn bộ learning pool thuộc TRAIN đã công bố trước vòng học. Không kéo public/private/holdout vào learning pool để đạt đủ100%.

- Split theo doanh nghiệp; freeze learning/calibration/holdout và hash trước tối ưu. Dedupe ID, nội dung gần trùng và template rò giữa split.
- Chỉ learning pool được dùng để xem gold, tạo correction, replay và cập nhật gradient.
- Calibration có thể chấm mọi câu để chọn cấu hình nhưng không đưa lỗi/gold/plan của nó vào correction training. Nếu quyết định dùng nó để học, đổi vai trò chính thức và không báo lại như validation độc lập.
- Holdout chỉ mở ở mốc đã định; không đưa phản hồi từng câu holdout vào vòng học. Nếu đã mở nhiều lần, ghi exposure và thay đánh giá hoặc dùng cross-validation theo doanh nghiệp phù hợp.
- Các doanh nghiệp từng dùng trong lịch sử4B có exposure; công khai điều đó. Split9B lịch sử: train18biz102,103,104,105,107,108,110,113,115,116,118,119,120,122,123,126,129,130;cal6biz101,109,112,114,121,128;audit6biz106,111,117,124,125,127.
- Nhãn TRAIN được dùng hợp lệ theo thể lệ. Không nhờ người gán nhãn test hoặc dùng đáp án người trên validation bị hạn chế. Agent kiểm thể lệ hiện hành trước thu thập phản hồi ngoài.

Raw/gold/SFT/journal chi tiết giữ trong môi trường được phép; không đưa vào chat hoặc dịch vụ LLM ngoài. Trước đưa dữ liệu/adapter liên quan lên Kaggle, kiểm quyền theo thể lệ; Private không tự chứng minh được phép.

## 5. Vòng học lại bắt buộc

Mỗi vòng có round_id, parent_adapter_hash, dữ liệu/split/schema/source hash, ngân sách thử lại và số optimizer steps tối đa. Đánh giá trước cập nhật, sau đó train theo batch đã kiểm, không tự sửa trọng số ngay sau từng câu.

### Bước A — suy luận mù gold

1. Đọc từng câu learning pool cùng nguồn/rulebook/as_of_date hợp lệ.
2. Sinh plan và predicted answer trước khi truy cập gold; khóa/lưu attempt đầu với model/config hash.
3. Nếu đường chuẩn abstain hoặc plan không hợp lệ, đường learning ghi nhận lý do và thử sửa plan/generate ứng viên khác với ngân sách hữu hạn.
4. Dùng tối đa3 attempts/câu/vòng làm cấu hình khởi đầu, có thể giảm theo ngân sách. Ghi cả attempt đầu và cuối; không chỉ báo attempt may mắn nhất.
5. Không cho model thấy gold trong prompt suy luận đánh giá. Gold chỉ mở ở bộ chấm/chẩn đoán của learning split sau dự đoán.

Không yêu cầu lưu chuỗi suy nghĩ nội tâm đầy đủ. Lưu plan có cấu trúc, phép toán/code, nguồn dữ liệu liên quan, kết quả thực thi và giải thích lỗi ngắn có thể kiểm chứng.

### Bước B — chấm và phân loại

Dùng scorer chính thức nếu có và phù hợp; nếu dùng scorer nội bộ, ghi rõ giới hạn và kiểm quy tắc số/percent/list/table/date. Mỗi câu có status: correct, wrong, invalid_plan, runtime_error, unresolved_data hoặc label_suspect. Các trạng thái chưa chấm được không được tính correct.

Hai metrics khác nhau:
- First-pass accuracy: chất lượng trước mọi phản hồi sửa lỗi.
- Post-feedback solved rate: đã giải được sau thử/chữa trên learning pool; không phải độ tổng quát.

Không bỏ lỗi hạ tầng khỏi mẫu số mà không báo. Luôn reconcile N câu và N bản ghi trạng thái.

### Bước C — câu đúng: bảo toàn có kiểm tra

- Lưu attempt, adapter_hash đang tạo ra đáp án, plan/executor/source/config hashes và scorer verdict vào solved bank.
- Đúng về kết quả chưa đủ làm target plan: kiểm plan thực thi đúng ngữ nghĩa, metric/kỳ/filter/unit, không khớp số tình cờ. Chỉ mẫu đã kiểm chứng được vào replay/SFT.
- Lưu checkpoint theo mốc/vòng có manifest thay vì tạo cả bản model cho từng câu. Có thể gắn mỗi câu với checkpoint đã dùng.
- Đưa mẫu đúng vào replay bank; cân bằng doanh nghiệp/operation, không để câu dễ lấn át. Không khóa cứng weights “của câu đúng” vì không có ánh xạ độc lập đó.
- Đánh giá lại solved bank sau cập nhật; ghi correct→wrong là regression, không xóa bằng chứng đúng trước đó.

### Bước D — câu sai: tìm nguyên nhân trước khi học

Phân loại ít nhất:
1. Ingestion/file role/join/ngày/số/đơn vị/VAT/cancel/return sai.
2. Hiểu metric, temporal/as_of_date, entity/filter/comparator/group_by/output type sai.
3. Plan schema/parse/truncation/prompt thiếu ngữ cảnh.
4. Executor chưa hỗ trợ hoặc công thức nghiệp vụ sai.
5. Source coverage không đủ, nhãn mơ hồ hoặc scorer không tương thích.
6. Runtime/OOM/lỗi lưu, không phải lỗi kiến thức.

Sửa parser/executor bằng test giả lập khi nguyên nhân nằm ở code; không train model ghi nhớ một đáp án để che bug. Ngày nguồn chưa parse, số tiền chưa parse, composite/threshold và temporal mismatch là ưu tiên từ evidence cũ.

Tạo corrected plan từ dữ liệu TRAIN/rulebook, thực thi và chấm lại. Một gold answer không tự cung cấp gold plan. Nếu chưa tìm được plan có nghĩa và khớp gold, giữ unresolved/label_suspect trong hàng đợi; không chế plan hoặc sửa gold để qua kiểm tra. Không dùng dự đoán sai của model làm nhãn đúng.

### Bước E — xây correction dataset và học lại

- Chỉ dùng corrected plans đã được kiểm chứng và replay plans đúng có nguồn gốc rõ.
- Mặc định thử mixture50%correction/50%replay theo batch khi đủ mẫu, cân bằng shop/operation; đây là giả thuyết khởi đầu phải đo, không phải tỷ lệ tối ưu đã biết. Không nhân bản thiếu kiểm soát khi một nhóm quá ít.
- Giữ prompt suy luận không chứa gold. Target là plan/đáp án theo thiết kế thật của pipeline; không đổi sang học thuộc ID→answer. Với pipeline hiện tại, ưu tiên completion plan rồi executor tính answer.
- Tạo dataset version mới, lưu manifest counts/hash/split/round/provenance, dedupe và kiểm token-length. Các câu chưa tạo được target vẫn nằm trong attempt/error ledger, không âm thầm biến mất.
- Warm-start từ adapter9B đúng hash trong chế độ trainable; optimizer/schedule mới là run mới. Full resume chỉ khi đủ states và data/order/config tương thích. Notebook GUI cũ chỉ hỗ trợ fresh adapter, cần sửa wrapper mới trước khi dùng cơ chế này.
- Giữ rank8/batch1/accum16/max_length2048 khi phù hợp. Có thể thử LR5e-6 hoặc1e-5 cho warm-start sau smoke; chọn trước ngân sách nhỏ và không coi là cấu hình đã được chứng minh tốt nhất.
- Optimizer step mới tạo trọng số mới. Lưu trước/sau, số step, trainable params và kiểm cập nhật hữu hạn; train loss giảm không chứng minh làm test tốt hơn.

### Bước F — kiểm hồi quy và quyết định giữ model

Chạy lại learning first-pass, solved replay/regression bank và calibration độc lập với decoder cố định. Báo đúng/sai/unresolved/attempt coverage theo doanh nghiệp/operation, không chỉ tổng loss.

Lưu mọi checkpoint thử nghiệm có giá trị để truy vết; chỉ promote BEST_MODEL khi đạt tiêu chí đã định trước trên calibration, không regression vượt ngưỡng và các gates hạ tầng đạt. Candidate không đạt vẫn giữ evidence nhưng không thay bản tốt nhất. Tách BEST_MODEL, LATEST_CHECKPOINT và LATEST_SYNCED.

Không đảm bảo giữ mọi câu đúng tuyệt đối sau gradient update. Nếu có regression, giảm bước/LR, tăng replay hoặc rollback sang parent/best đã lưu và thiết kế vòng mới. Không tự xóa candidate hoặc cập nhật vô hạn cho đến khi khớp TRAIN.

## 6. Đặc tả “không bỏ qua câu nào” có thể kiểm toán

Agent phải đảm bảo:
- Mỗi round: expected_ids khớp attempt_ids, không thiếu/trùng;100%attempt coverage trên learning pool.
- Ghi kết quả first-pass kể cả khi guard đường chuẩn abstain; best-effort candidate nằm trong trường riêng, không giả thành output được xác minh.
- Mọi câu sai/lỗi có error_class, next_action và trạng thái correction. Không có trạng thái “skip và quên”.
- Retry hết ngân sách thì unresolved, giữ lại cho vòng sau hoặc sửa hạ tầng; không gán bừa “đã giải”.
- N correct+wrong+unscorable/error phải bằng N đã xử lý theo quy ước loại trừ rõ ràng.
- Không đòi phải có gradient update cho mọi câu nếu chưa có target đáng tin; giữ câu trong hàng đợi là bắt buộc, tạo nhãn sai là không được phép.

Nếu model hiện chỉ sinh plan và executor từ chối, agent cần bổ sung learning fallback có schema chẩn đoán và candidate_answer, tách khỏi hàm answer chuẩn. Không đơn giản thay tất cả abstain bằng0, chuỗi rỗng hoặc unanswerable.

## 7. Ghi chép và checkpoint

Các tệp chi tiết giữ riêng ở runtime/private storage được phép:
- attempts.jsonl: round_id, question_id, split, adapter/config hashes, attempt_no, predicted_plan, predicted_answer, execution_status, reason, latency; không cần lưu suy nghĩ nội tâm.
- feedback.jsonl: verdict, scorer_version, error_class, corrected_plan đã kiểm hoặc unresolved_reason, provenance và target_verified.
- replay_manifest.json: các sample refs đã kiểm, nhóm nghiệp vụ, nguồn và exposure; không coi đáp án đúng tình cờ là target.
- round_metrics.json: first-pass, sau feedback, coverage, regression correct→wrong, calibration, compute budget.
- run_config.json/environment.json/source/data/schema hashes/parent/init_mode.
- adapter, checkpoint với optimizer/scheduler/scaler/RNG/trainer state khi cần resume, checksums và lesson log.

Không ghi đè baseline. Ghi journal/checkpoint local disk bằng atomic write; backup snapshot bất biến. Không append journal trực tiếp trên Drive/FUSE dùng chung vì đã có sự cố file trùng/trạng thái cũ.

Sau khi lưu Kaggle output/version thành công, tải về máy và so hash rồi cập nhật LATEST_SYNCED. Không gọi “đã lưu tri thức của câu X” chỉ vì có dòng log; phải có checkpoint được xác minh và bằng chứng kiểm lại câu đó.

## 8. Kiểm thử trước triển khai cơ chế học lại

Giữ99 tests baseline (75app+24tools) và thêm tests cho:
1. N câu tạo N attempt records, cả plan lỗi/exception/timeout.
2. Gold không vào prompt inference; split holdout không vào correction/replay.
3. Mẫu wrong không tự được đưa thành target; corrected plan phải qua validation+execution+scoring.
4. Câu đúng khớp số nhưng sai metric/filter bị loại khỏi supervised replay.
5. Regression đúng→sai được ghi và có thể rollback; không overwrite BEST_MODEL.
6. Retry có giới hạn; unresolved còn trong ledger.
7. Warm-start load đúng parent và trainable params; full resume không bị gọi nhầm khi reset optimizer.
8. Khôi phục journal chỉ với fingerprint tương thích; không thiếuID sau gián đoạn.

Smoke GPU trên bản sao: load đúng adapter, loss mask/shift đúng, forward/backward/optimizer hữu hạn và checkpoint reload được. Không nạp pickle states không rõ nguồn. Không chạy dài khi chưa đủ dữ liệu được phép và nguồn target xác minh.

## 9. Ngân sách, tiêu chí dừng và triển khai thi

Khởi đầu bằng một vòng nhỏ trên learning pool có kiểm chứng; vẫn tạo ledger cho toàn bộ pool và đánh dấu pending cho phần chưa chạy. Không báo100%đã xử lý khi mới smoke subset. Sau khi hạ tầng đạt, hoàn tất vòng qua mọi câu learning pool trong ngân sách cho phép.

Đặt trước max_rounds, steps/round, attempts/question, max_GPU_hours, ngưỡng regression và tiêu chí promote. Nếu người dùng chưa cấp ngân sách train dài, chuẩn bị config/smoke/review trước. Dừng có kiểm soát khi NaN/Inf, hash/split sai, checkpoint hỏng, OOM lặp, hết thời gian/dung lượng hoặc calibration không cải thiện theo patience định trước. Không coi unresolved>0 là lý do đốt GPU vô hạn.

Chế độ thi: khóa source/schema/weights/config; không mở gold, không train online từ test. Chọn chính sách abstain trên calibration theo scoring chính thức; bật forced-answer ở TRAIN không bắt buộc forced-answer khi thi. Đo PA/coverage/answered accuracy và thời gian/VRAM thực tế. Nếu PA=(correct−wrong)/N, trả lời bừa có thể làm giảm điểm; dùng kết quả đánh giá để quyết định.

Cơ chế học lại là vòng phần mềm (sinh→chấm→chẩn đoán→correction/replay→optimizer→đánh giá→checkpoint), không phải chỉ thêm câu prompt “hãy học từ lỗi”. Agent phải triển khai code/tests/notebook trong bản mới rồi xác nhận chạy thực tế; tài liệu này không chứng minh cơ chế đã tồn tại.

### Quyết định vận hành bổ sung ngày 08/11/2026

Áp dụng mục 11 của `D:\ZAIC\AGENTS.md` cho cấu hình, bằng chứng và các cổng thực thi; đây là nguồn quy tắc duy nhất, không tự chọn lại phương án từ các ví dụ LR/mixture trong tài liệu này.

- Lượt correction/replay tiếp theo dùng `init_mode=warm_start` từ adapter 9B đã xác minh, optimizer/scheduler mới và run mới. Giữ checkpoint-48 nguyên trạng để phục hồi/đối chứng. `fresh|warm_start|resume` phải khai báo tường minh, không tự chuyển khi lỗi.
- Agent tự đề xuất ngân sách theo tài nguyên thực tế và giới hạn đã biết; chỉ hỏi phần còn thiếu trước train dài. Chưa có quyền chạy train dài chỉ vì cập nhật tài liệu.
- Tách `user_authorized` khỏi `competition_permission_evidence`. Bằng chứng thể lệ độc lập hiện chưa xác minh: tiếp tục chuẩn bị/test giả lập, chưa chuyển mới hoặc xử lý dữ liệu thuộc phạm vi hạn chế trên Kaggle. Giữ bằng chứng lịch sử của các upload đã xảy ra.
- Báo riêng attempt coverage (mục tiêu 100% khi hoàn tất pool, pending chưa tính) và verified correction coverage (có thể dưới 100%, ghi tử/mẫu; mẫu 0 là N/A). Không bịa nhãn; unresolved còn trong hàng đợi.
- `mode=learning|evaluation|submission` cùng split/hash phải được kiểm tra. Không tự chuyển mode; evaluation/submission không cập nhật weights hay dùng gold để sửa dự đoán.
- Mọi capability có mức Đã viết tài liệu / Đã triển khai code / Đã qua kiểm thử / Đã chạy thực tế, kèm evidence đúng phạm vi. Trạng thái trước triển khai là tài liệu. Hiện relearning_v1 đã có mã và 40 tests mới cùng 99 baseline đạt trên Kaggle Linux CPU; warm-start thực tế trên 9B, GPU smoke và checkpoint reload còn chưa kiểm; full resume chưa triển khai. Bằng chứng chi tiết: D:\ZAIC\EXPERIMENTS\relearning_v1\IMPLEMENTATION_REPORT_VI.md.
- Kiểm lại input của run mới, gắn thời điểm/phạm vi/đường dẫn/hash vào kết luận; không yêu cầu kiểm toán lại toàn bộ lịch sử mỗi vòng.

## 10. Thứ tự thực hiện cho agent tiếp quản

1. Đọc D:\ZAIC\AGENTS.md/audit/registry, xác minh adapter9B và backup baseline.
2. Dùng D:\ZAIC\AGENTS.md làm nguồn quy tắc duy nhất; các dẫn chiếu đã được thống nhất về training.md. Khi có khác biệt, forced-attempt chỉ áp dụng chế độ học; giới hạn dữ liệu và bảo toàn vẫn giữ.
3. Freeze learning/calibration/holdout, ngân sách và tiêu chí promote.
4. Triển khai learning_mode, attempt/error ledger, scorer và tests chống leak; chạy giả lập.
5. Chạy baseline first-pass và một vòng correction nhỏ, báo năng lực thật; xử lý hết pool theo kế hoạch, không âm thầm skip.
6. Tạo verified correction/replay dataset, warm-start có kiểm soát, đánh giá hồi quy và calibration.
7. Lưu/đối chiếu checkpoint hai nơi, cập nhật registry/lesson log; chỉ promote khi có bằng chứng.

Nguồn: trainning.md cũ, báo cáo bàn giao và chỉ đạo mới ngày08/11/2026. Các số lịch sử phải được đối chiếu artifacts thực tế khi chạy. Không có huấn luyện mới nào được thực hiện chỉ bằng việc viết tài liệu này.


## Cập nhật có bằng chứng 09/11/2026 — 20261109T065047772787Z_d5826809

Warm-start E7, 7 steps, 7 correction +7 replay đã xác minh. Calibration 552 câu đã lộ: 228→229 đúng, 53→52 sai, 0 correct→wrong, delta +0.00362319 <0.01: không promote. Post correction 0/7, replay 7/7, correct→wrong 0. Public1000 dùng E7 theo cổng; không phải kết quả candidate. 8 bộ checkpoint/state đã khớp hash Kaggle–máy; LATEST_CHECKPOINT/LATEST_SYNCED cập nhật, BEST giữ nguyên. GPU đã tắt. Full resume chưa triển khai/xác minh; còn1720 pending/11 unresolved. Bằng chứng: D:\ZAIC\PRIVATE_DATA\zaic\result_811\QUALITY_ROUND_20261109\ROUND_COMPLETION_REPORT_VI.md. Các mục trạng thái cũ trong tài liệu là lịch sử; xác nhận quyền hiện tại theo AGENTS.md/user attestation trong run config, không nâng thành bằng chứng thể lệ độc lập.


## Bằng chứng vòng thời gian v3 — 20261109T125033322737Z_b786bc50

Warm-start E7,9 bước,9 correction+9 replay. CPU230tests đạt. TRAIN124/124 thử,9 correctionverified,12unresolved; toànpool1693pending. Hậu xử lý GPUv3 lỗi cleanup sau khi traincompleted; không train lại, đánh giá child riêng và bảo toàn CPU fullstate. Model mới d3104b03f4e5a48b5b65fc5525a23d814eb025bc71b080be214ebc2e9e8ff11b. BEST giữ nguyên. LATEST_CHECKPOINT/LATEST_SYNCED đã khớp manifest c51d0b0f422ebb93f9a6fd927f9c7a3a6cb8d3eefffdf5987e89dca423863aaf. Full resume chưa chứng minh. Báo cáo: D:\ZAIC\PRIVATE_DATA\zaic\result_811\TIME_FOCUSED_V3_20261109\RESULT_REPORT_VI.md.


## CPU v4 — 2026-11-10T01:15:51.941534+07:00

282/282 tests LinuxCPU đạt,0skip; baseline byte giữ nguyên. Sửa source-proven annotation, không impute tiền;124 dự đoán E7 đã lưu vẫn103 đúng,0regression,3exception được xử lý nhưng chưa tăngaccuracy.12unresolved gồm1coverage và11canonical/gold lệch, không ép làm supervision. Dataset9correction+9replay phủ5nhóm thời gian,12biz; bank103replay được giữ làm regression và7lỗi cũ nằm đủ trongcorrection. Manifest a0e11f70c0c8e2b29b983e265a77f4e30ed0436ef6714fef4e4c2cfe377720c3. Runner train child+protected regression có code/testsCPU, chưaGPUv4; fullresume chưa triển khai. Không optimizer/model mới hoặc cập nhậtBEST/tracker. Chi tiết:D:\ZAIC\EXPERIMENTS\relearning_cpu_v4_20261109\CPU_READINESS_REPORT_VI.md.

## Chuẩn bị bản thi ngày 10/11/2026

Người dùng cấp ngân sách mới 240 phút GPU cho smoke, huấn luyện và đánh giá nội bộ, **không bao gồm public_test**, mục tiêu chuẩn bị bản thi trước cuối ngày 11/11 theo giờ Việt Nam. Không cộng ngân sách lượt cũ hoặc tự suy ra ngân sách public_test mới. Theo dõi trong `D:\ZAIC\EXPERIMENTS\competition_ready_20261110\GPU_BUDGET.json`.

Driver frozen pilot dùng bank đã xác minh, tránh thu lại dự đoán cũ; warm-start E7, optimizer/scheduler mới, đề xuất 18 bước và tối đa 4 epoch tương đương, LR1e-5, accumulation4, time weight4. Source af590010e0277770820b1c8ab13d55d35a851a8cc45375351c2f715bc24f02d1 đã qua 288/288 tests Linux CPU, không skip. Có thêm 3 kiểm tra wrapper tại máy. Không gọi kết quả CPU là bằng chứng GPU.

Wrapper GPU v1 bị chặn trước smoke/optimizer vì cấu hình lịch sử trong bootstrap ghi đè CONFIG mới. Log được giữ; notebook đã lưu CPU v2. Wrapper sửa chỉ giữ một CONFIG; preflight input/cache thật trên Kaggle CPU đã passed tại `zaic-competition-wrapper-cpu-20261110/1`. Đã khởi chạy `lnghongphc/zaic-competition-pilot-20261110/3`, timeout 6000 giây, gồm các child smoke/train/regression. Tại lúc ghi quyết định này job đang RUNNING; chưa xác nhận train hoặc cải thiện chất lượng. Không public_test, không promote BEST hoặc cập nhật LATEST_SYNCED từ trạng thái RUNNING. Kết luận cuối phải đọc output và đối chiếu hash.
