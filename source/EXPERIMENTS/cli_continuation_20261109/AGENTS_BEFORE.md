# AGENTS.md — Quy tắc chuẩn cho agent ZAIC DataMate

Ngày lập: 08/11/2026, giờ Việt Nam. Phạm vi: tiếp quản, bảo toàn, cải thiện và đánh giá model; vận hành bằng Codex desktop và Kaggle web. Đọc tiếp `training.md` trước khi lập thí nghiệm.

## GPU Kaggle — ưu tiên nghiêm ngặt của người dùng (08/11/2026)

- Quota GPU có hạn. Mặc định Accelerator=None; dùng CPU cho chuẩn bị mã, upload, checksum và kiểm thử giả lập không cần GPU.
- Chỉ bật GPU ngay trước lượt huấn luyện hoặc phép test thực sự cần GPU, khi mã, input và cấu hình đã sẵn sàng.
- Khi train/test kết thúc, thất bại hoặc bị gián đoạn: ghi checkpoint/output cần thiết vào vùng lưu bền vững rồi dừng phiên GPU ngay (không giữ GPU để đợi tải về máy), đưa Accelerator về None và kiểm tra giao diện xác nhận đã tắt. Không để GPU chạy rỗi trong lúc phân tích, sửa mã, tải file hoặc chờ người dùng.
- Chạy Save & Run All có thể tạo phiên nền riêng: phải kiểm tra cả phiên tương tác và phiên nền; không coi đóng tab là đã dừng GPU.
- Nếu mất quyền điều khiển, báo rõ chưa xác nhận tắt được và hướng dẫn người dùng Stop Session; không hứa tự tắt khi agent đã ngắt.

## 1. Mục tiêu và cách làm việc

- Bảo toàn công sức của mọi lượt huấn luyện: trọng số, trạng thái checkpoint, mã, dữ liệu/manifest, cấu hình, đánh giá và bài học.
- Cải thiện PA bằng kết quả đo được; không hứa PA=1. Loss giảm, nhiều steps hơn hoặc model lớn hơn không chứng minh chất lượng tăng.
- Người dùng không muốn tự nhập CLI. Agent có thể chạy kiểm tra bằng công cụ nội bộ đã được cấp; hướng dẫn người dùng bằng file và nút bấm GUI. Không bắt cài Kaggle CLI hay dán API token.
- Tự hoàn thành công việc đã được cho phép; chỉ hỏi khi thiếu thông tin/quyền có ảnh hưởng thực sự. Tiếp tục các phần độc lập khi bị chặn.
- Kiểm tra công cụ Browser/Computer Use thực tế trong phiên hiện tại; không mặc định chọn This computer là đã điều khiển được Kaggle. Bàn giao đăng nhập/xác minh/chọn file khi công cụ yêu cầu.
- Tài liệu này không tự cấp quyền mới, không vượt sandbox hoặc thay thế các chỉ dẫn cấp cao hơn. Không tự nộp bài, công khai tài sản hoặc xóa bản gốc.

## 2. Thứ tự đọc và cập nhật trạng thái

1. File D:\ZAIC\AGENTS.md này và các chỉ dẫn cấp cao hơn áp dụng.
2. D:\ZAIC\training.md trước mọi thao tác huấn luyện/đánh giá; đây là đặc tả cần triển khai, không phải bằng chứng đã triển khai.
3. Báo cáo bàn giao ZAIC, TRAINING_HISTORY_AUDIT.md, RUN_REGISTRY.csv, WEIGHTS_INVENTORY.json, báo cáo kiểm checksum và các manifest thực tế, nếu đã có.
4. Mã nguồn, model_lock, cấu hình và evidence của run định dùng.

Nguồn quy tắc duy nhất là D:\ZAIC\AGENTS.md. File AGENTS.md trong project con chỉ dẫn về file này, không duy trì bộ quy tắc độc lập. training.md được giữ riêng làm đặc tả huấn luyện. Bản hướng dẫn cũ được lưu ngoài luồng vận hành trong GUIDE_ARCHIVE.

Ưu tiên bằng chứng file/hash và log thực tế hơn diễn giải hội thoại. Ghi rõ nguồn, thời điểm và phạm vi mỗi kết luận. Không biến checklist hoặc ảnh báo cáo thành bằng chứng đã chạy kiểm tra mới.

## 3. Hiện trạng được ghi nhận, chưa phải chứng nhận mới

- Base 9B: Qwen/Qwen3.5-9B; revision c202236235762e1c871ad0ccb60c8ee5ba337b9a.
- Adapter 9B cuối: 86.623.432 byte; SHA256 e7dfa635beccce128c99f9d3fd993d33dbb4c88e73e3ddeccbc2bc915eec4a4d.
- Các checkpoint 9B đã thấy: 40, 45, 48. Các bản cũ: bản 20 steps; run1 150 steps với checkpoint125/150; candidate4B 83 steps với checkpoint80/83.
- Báo cáo trong ảnh người dùng cung cấp ghi 6 mục run/thử nghiệm, 4 giai đoạn có adapter thành công, 2 lần thử 9B lỗi chưa tìm thấy trọng số riêng; 14 file adapter và 137 tệp artifact/state/support; 7 checkpoint chính đủ trạng thái, chưa thử resume. Đây là kết quả được báo lại, cần mở báo cáo/file gốc khi tiếp quản.
- Cùng báo cáo: 68 đối chiếu kích thước Drive khớp, nhưng checksum Drive chưa xác minh; chưa thấy dataset/model Kaggle đã lưu tại thời điểm báo cáo. Không coi đây là trạng thái hiện tại nếu chưa kiểm tra lại.
- 9B tạo adapter mới từ base, không kế thừa trọng số 4B. Candidate4B có bằng chứng kế thừa run1/final; run1 chỉ có dấu hiệu tiếp tục model trong RAM từ bản20steps, mức chắc chắn thấp hơn. Không suy diễn thêm.
- v2/v2.1 sửa loader/contracts/executor, không huấn luyện lại adapter. Public v2.1 đã xuất 1000 dòng, 235 answered/765 abstain; chưa có PA chính thức mới trong bằng chứng bàn giao.

## 4. Bảo toàn và kiểm toán trọng số

Trước thay đổi đầu tiên đối với tài sản model/dữ liệu/run, lập hoặc cập nhật inventory theo file: đường dẫn, bytes, SHA256, vai trò, base/adapter, run/parent, checkpoint step và trạng thái xác minh. Kiểm cả ZIP chưa giải nén và file tải dở trong thư mục người dùng cho phép.

- Không xóa, đổi tên, di chuyển, ghi đè, merge hoặc loại bản trùng một cách tự động. Không gọi file trùng tên là trùng nội dung nếu chưa so hash.
- Source và run đã đóng băng giữ nguyên; mọi sửa đổi/thí nghiệm vào bản sao hoặc nhánh mới.
- Base cần config/tokenizer/index và mọi shard index tham chiếu. Không ghép shard của các bộ khác nhau; không tự đổi hậu tố tên file để “đủ bộ”.
- Adapter cần trọng số, adapter_config và base tương thích. Không nạp adapter4B vào base9B.
- Resume cần trainer/optimizer/scheduler/scaler khi có/RNG cùng cấu hình và dữ liệu tương thích. Có file trạng thái không đồng nghĩa đã chứng minh resume thành công.
- Không deserialize pickle/bin/pt không rõ nguồn để kiểm kê. Chỉ load trạng thái nguồn đáng tin cậy sau khi xác minh nguồn và checksum, dùng cơ chế an toàn phù hợp.
- Giữ log lỗi và bằng chứng run thất bại. “Chưa tìm thấy” khác “đã mất”; một lượt lỗi có thể chưa từng tạo checkpoint.
- Bản mới nhất không mặc định tốt nhất. Trọng số cuối không tự chứa mọi lần thử trước. Không gộp adapter để cố “cộng kinh nghiệm”.

## 5. Ranh giới dữ liệu và đánh giá

Nguồn dữ liệu: challenge.zalo.ai. Đọc thể lệ chính thức hiện hành và điều kiện challenge trước xử lý trên nền tảng khác. Bản ghi trước có hạn chế chuyển dữ liệu ra ngoài đội/bên thứ ba; Private không tự chứng minh được phép.

- Không đưa câu hỏi, gold, giao dịch, SFT chi tiết hoặc journal từng câu vào chat, dịch vụ LLM ngoài, public repo hay tài liệu bàn giao công khai.
- Raw/SFT xử lý bằng chương trình trong môi trường người dùng được phép; agent chỉ đưa hash, số lượng, tổng hợp hoặc ví dụ giả lập vào ngữ cảnh AI.
- Chỉ dùng TRAIN để tạo supervision, chọn chính sách và cải thiện nghiệp vụ. Không dùng public/private gold, không gán nhãn thủ công hoặc sửa đáp án test theo từng ID.
- Tách theo doanh nghiệp, ghi exposure lịch sử, freeze evaluation trước thay đổi. Tập đã mở không tiếp tục được gọi untouched holdout.
- Không bỏ guard dùng chung hoặc giả đáp án để tăng coverage. Riêng LEARNING_FORCE_ATTEMPT: thử mọi câu của learning pool TRAIN bằng đường chạy riêng, ghi lỗi/unresolved và retry hữu hạn theo training.md. Chế độ đánh giá/nộp bài giữ chính sách đã khóa; không lấy gold để sửa dự đoán.
- Ngày tương đối phải neo theo as_of_date/quy tắc nguồn, không theo ngày hệ thống.

## 6. Vận hành máy cá nhân và Kaggle

Đường dẫn mẫu, phải thay bằng đường dẫn đã xác minh:
- CODEX_PROJECT: mã, tài liệu, notebook sạch, báo cáo tổng hợp.
- PRIVATE_DATA: base, adapters, dữ liệu, checkpoints và output chi tiết, nằm ngoài project mã.
- DOWNLOADS_ARCHIVE: ZIP gốc, chưa xóa trước khi đối chiếu.

Phân tách thư mục chỉ giúp quản lý ngữ cảnh, không phải sandbox. Không đọc đệ quy raw vào tool output.

Trên Kaggle, xác minh input/read-only và vùng output có thể ghi trong phiên thực tế. Base cố định giữ ở input; tránh xuất lại 19GB vào output. Kiểm phiên bản Python/CUDA/driver/Transformers/PEFT/bnb và GPU được cấp. Không cài đè nguyên môi trường Colab hoặc mặc định hai T4 có bộ nhớ chung. Loader baseline chỉ dùng cuda:0.

Checkpoint/journal ghi trên local disk của runtime, dùng ghi tạm rồi đổi tên nguyên tử. Đã từng xảy ra Drive/FUSE dùng chung tạo file trùng tên và trạng thái cũ; không append journal đang chạy lên đó. Backup bằng snapshot bất biến có count/hash, xác minh sau lưu.

Mỗi lượt có run_id riêng. Khôi phục journal chỉ khi source/data/model/config fingerprint khớp; bản ghi cùng ID phải giống hệt. Không sửa fingerprint để ép resume sau đổi môi trường/config.

Đồng bộ theo phiên bản đã lưu: lưu output bền vững trên Kaggle → tải về máy → kiểm đủ file/hash → cập nhật LATEST_SYNCED. BEST_MODEL chỉ cập nhật khi đánh giá đạt. Không quảng cáo đồng bộ liên tục khi agent/browser đã dừng. Lưu cả phiên bản input/notebook của run; không nhầm notebook đã lưu với checkpoint đã lưu.

## 7. Các cổng bắt buộc trước huấn luyện

- Inventory/hash baseline và phân biệt fresh/warm-start/resume rõ ràng.
- Kiểm source/test hiện có: baseline có 75 app+24 tools=99 tests giả lập; sửa mới phải thêm test nghiệp vụ liên quan.
- GPU smoke test: model/tokenizer load đúng, loss hữu hạn, gradient hữu hạn, completion-mask/shift đúng, bộ nhớ và thời gian trong ngân sách.
- Split/data/schema/manifest đã freeze; tập đánh giá không lẫn training.
- Run config, parent, tiêu chí chọn model, checkpoint/backup, điều kiện dừng, output mới có thể xem xét được.

Không tự bắt đầu train dài chỉ vì nhiệm vụ là “setup/kiểm tra”. Khi người dùng đã giao train với cấu hình/ngân sách rõ thì tiến hành trong phạm vi đó, không hỏi lại từng bước thường lệ. Dừng và báo nếu loss/gradient không hữu hạn, sai base/hash, rò rỉ split, checkpoint ghi lỗi hoặc hết không gian. Giữ evidence, không đánh dấu hoàn tất giả.

## 8. Báo cáo sau mỗi lượt và điều kiện bàn giao

Ghi RUN_REGISTRY: run_id, parent/evidence, init_mode, source hash, data hash/split, model revision, adapter/checkpoint hash, versions/GPU, seed/config, steps, thời gian/VRAM, metrics, trạng thái backup và hạn chế.

Phân biệt bốn kết luận: dùng được cho inference; đủ để resume; đủ bằng chứng tái lập; còn thiếu/chưa kiểm. Chỉ tuyên bố điều đã kiểm tra. Không khẳng định Kaggle đã nhận trọng số nếu chưa có tài nguyên tồn tại, đủ file và đối chiếu thành công.

Bàn giao gồm config, source, manifest, adapter, checkpoint cần thiết, metrics, log đã lọc, lesson log, checksum và liên kết/phiên bản lưu. Cập nhật D:\ZAIC\training.md bằng quyết định có bằng chứng; giữ lịch sử quyết định cũ. Trả lời người dùng ngắn gọn bằng tiếng Việt: đã làm, kết quả, phần thiếu, bước tiếp theo.

## 9. Quy tắc kỹ thuật bổ sung được giữ từ AGENTS.md project

- Đọc README_FIRST_VI.md, docs/HANDOFF_CODEX_VI.md, docs/DOWNLOAD_AND_GUI_VI.md, docs/TRANSFER_MANIFEST.json và reports/VALIDATION.json trong project khi tiếp quản. Gói bàn giao chỉ chứa mã; không thay thế kiểm kê dữ liệu/trọng số thật.
- frozen_v21/app và frozen_v21/tools gồm 27 file nguồn bất biến; xác minh SOURCE_SHA256.json. Không sửa tại chỗ. Các recovery script Colab ở root là lịch sử, không phải entrypoint Kaggle; không Run All notebook Colab lịch sử.
- Baseline rank 8, không phải 16. Final submission lịch sử SHA256 e0815c96524e9db0badf987e45e13227eec8e57c7f3195f5aa93a580a0508780. 235 answered không có nghĩa 235 đúng hay PA=1.
- Baseline có 75 app + 24 tools tests giả lập; gui/kaggle_runner.py chạy unittest không cần pytest. Test hồi quy phải kiểm ngữ nghĩa nghiệp vụ. Đổi runtime/dependency phải kiểm batch parity.
- /kaggle/input chỉ đọc; /kaggle/working có thể ghi. Để base ở input; không sao chép/xuất lại 19GB vào outputs. Ghi checkpoint/journal vào local disk runtime, không append trực tiếp lên Drive/FUSE dùng chung.
- Khi đổi fingerprint, tạo thư mục output mới; không sửa fingerprint để ép resume. Bản ghi trùng ID phải giống hệt.
- Không ép cài stack torch+cu130 của Colab lên Kaggle. Loader baseline chỉ dùng cuda:0; T4 x2 không tự gộp VRAM.

## 10. Đường dẫn và bằng chứng trạng thái

- Project mã đã xác định: D:\ZAIC\drive-download-20241211T124801Z-001\ZAIC_Local_GUI\CODEX_PROJECT. Đường dẫn D:\ZAIC\CODEX_PROJECT trong tài liệu cũ chỉ là gợi ý, không coi là thư mục thật.
- Dữ liệu/model riêng: D:\ZAIC\PRIVATE_DATA. ZIP gốc: D:\ZAIC\DOWNLOADS_ARCHIVE. Không đính kèm toàn bộ kho riêng vào ngữ cảnh AI.
- Tài liệu huấn luyện chuẩn: D:\ZAIC\training.md. Hướng dẫn agent chuẩn: D:\ZAIC\AGENTS.md. Không tiếp tục cập nhật agent.md đã lưu trữ.
- Mục 3 là lịch sử tham chiếu, không phải kiểm tra mới. Đối chiếu TRAINING_HISTORY_AUDIT.md, RUN_REGISTRY.csv, WEIGHTS_INVENTORY_AFTER_RENAME.json và WEIGHTS_CHECK_REPORT.md trước quyết định tác động tài sản.
- Báo cáo upload sau kiểm toán: D:\ZAIC\KAGGLE_UPLOAD_REPORT_VI.md; biên nhận chi tiết: D:\ZAIC\KAGGLE_STAGING\20261107T191003Z\KAGGLE_UPLOAD_STATUS.json. Báo cáo ghi bốn input Private và notebook version 2 đã xác minh; đọc báo cáo không chứng minh trạng thái phiên hiện tại.
- Notebook mới mặc định preflight. Tại thời điểm báo cáo upload, wrapper chỉ hỗ trợ fresh adapter, chưa triển khai warm-start/full resume hoặc vòng học correction/replay. training.md yêu cầu các tính năng này nhưng không chứng minh đã có code hay đã chạy smoke.
- LATEST_SYNCED là lượt đã tải về và khớp manifest, không mặc định model tốt nhất. BEST_MODEL cần đánh giá; LATEST_CHECKPOINT là mốc checkpoint mới nhất có bằng chứng, tách riêng hai trạng thái trên. Không tự mở rộng schema tracker khi chưa thực hiện/thẩm định thay đổi.

## 11. Quy tắc thực hiện theo cấu hình và bằng chứng (08/11/2026)

### Khởi tạo và ngân sách

Mọi lượt train phải khai báo `init_mode`: `fresh`, `warm_start` hoặc `resume`. Không tự chuyển chế độ khi gặp lỗi; dừng có kiểm soát, giữ evidence và báo nguyên nhân. Fresh dùng base đã pin; warm_start nạp adapter đã học ở chế độ trainable nhưng tạo optimizer/scheduler mới; resume khôi phục đầy đủ trạng thái cần thiết và xác minh data/order/config tương thích. Ghi đường dẫn, phiên bản và hash base/adapter/checkpoint nguồn, kết quả kiểm trạng thái và parent run. Có đủ file chưa chứng minh resume chạy được.

Phương án đã thống nhất cho lượt học lại correction/replay tiếp theo: `init_mode=warm_start`, dùng adapter 9B đã xác minh ở mục 3, tạo run mới và optimizer/scheduler mới. Checkpoint-48 giữ nguyên để phục hồi/đối chứng. Không chọn resume chỉ vì checkpoint đó mới nhất; nếu lịch cũ đã hoàn tất, resume có thể không tạo thêm bước học. Đổi phương án phải ghi quyết định mới trước chạy; không fallback sang fresh.

Agent tự đề xuất cấu hình/ngân sách dựa trên GPU, dung lượng và thời gian khả dụng, ghi run config có thể xem xét. Chỉ hỏi phần ngân sách còn thiếu trước train dài; không hỏi lại giới hạn đã xác nhận. Quota hiển thị không phải quyền sử dụng hết quota. Các cổng GPU smoke, quyền dữ liệu và bảo toàn vẫn bắt buộc. Chỉ bật GPU khi code/input/config đã sẵn sàng và tắt theo quy tắc đầu file.

### Chế độ chạy, split và coverage

Khai báo `mode=learning|evaluation|submission` và manifest/hash split. Learning chỉ dùng learning pool TRAIN đã freeze; mỗi câu phải được thử và có bản ghi kết quả hoặc lỗi, kể cả timeout/OOM/plan lỗi. Evaluation/submission không cập nhật trọng số và không truy cập gold để sửa câu trả lời; bộ chấm độc lập có thể dùng gold hợp lệ theo split sau khi khóa dự đoán. Không tự chuyển sang learning khi đánh giá gặp câu khó.

- `attempt_coverage = số ID duy nhất đã thực sự thử và có trạng thái / tổng ID learning pool`. Mục tiêu hoàn tất vòng là 100%; pending chưa thử không tính đã coverage. Reconcile expected_ids với attempt_ids, phát hiện thiếu/trùng.
- `verified_correction_coverage = số ID câu sai/lỗi cần khắc phục đã có correction được xác minh / tổng ID câu sai/lỗi cần khắc phục` trong cùng pool/vòng. Ghi tử số, mẫu số, quy ước và phiên bản scorer. Nếu mẫu số bằng 0, báo N/A. Chỉ target đã qua kiểm ngữ nghĩa/thực thi/chấm mới tính verified.
- Correction coverage có thể dưới 100% dù attempt coverage đạt 100%. Câu chưa giải được giữ trong hàng đợi unresolved với error_class/next_action; không bịa đáp án/target để báo hoàn tất, không retry hoặc đốt GPU vô hạn.

### Bằng chứng và mức triển khai

Số liệu cũ là bằng chứng tại thời điểm kiểm tra. Mỗi kết luận ghi `observed_at`, `scope`, đường dẫn/phiên bản tài sản, hash, phép kiểm, kết quả và đường dẫn evidence. Trước run mới phải đối chiếu lại các input cần dùng; không bắt kiểm toán lại toàn bộ kho sau mọi lần chạy. Số tổng file/thư mục/checkpoint không thay kiểm tra từng tài sản của run. Đổi input/source/config phải có manifest và output mới.

Theo dõi từng capability độc lập qua bốn mức: **Đã viết tài liệu → Đã triển khai code → Đã qua kiểm thử → Đã chạy thực tế**. Mỗi mức cần đường dẫn/hash source, test report và run evidence tương ứng; không tự thăng mức chỉ vì có checklist. Ghi rõ test giả lập hay GPU, smoke hay train thật và phạm vi nào chưa đạt.

Trước triển khai relearning_v1, warm-start, full resume và vòng học correction/replay mới được xác nhận ở mức Đã viết tài liệu. Cập nhật triển khai ngày 08/11/2026: xem D:\ZAIC\EXPERIMENTS\relearning_v1\IMPLEMENTATION_REPORT_VI.md và IMPLEMENTATION_STATUS.json. Mã warm-start đã viết và qua unit test dùng mock; vòng learning/correction/replay đã qua test và demo CPU giả lập; 99 baseline + 40 test mới đạt trên Kaggle Linux CPU. Chưa chạy GPU hoặc optimizer trên model thật; full resume chưa triển khai. Không nâng trạng thái GPU từ bằng chứng CPU. Phải triển khai và kiểm thử cả ba luồng, rồi chứng minh từng luồng bằng thực thi phù hợp trước khi gọi chúng là tính năng hoạt động. Các bằng chứng preflight cũ không chứng minh các capability mới này.

### Khung run config cần hoàn thiện trước thực thi

Đây là khung tài liệu, không phải cấu hình đã đủ điều kiện chạy. Giá trị null phải được điền và kiểm tra trước tác vụ phụ thuộc.

```json
{
  "run_id": null,
  "mode": "learning",
  "init_mode": "warm_start",
  "parent_run_id": null,
  "base": {"model": "Qwen/Qwen3.5-9B", "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a", "manifest_sha256": null},
  "source_adapter": {"path": null, "sha256": "e7dfa635beccce128c99f9d3fd993d33dbb4c88e73e3ddeccbc2bc915eec4a4d"},
  "source_checkpoint": null,
  "optimizer_init": "new",
  "scheduler_init": "new",
  "user_authorized": {"granted": true, "scope": "previously authorized Kaggle account operations; no new long training authorization inferred", "evidence_path": null},
  "competition_permission_evidence": {"status": "unverified", "user_attestation_path": null, "official_source": null, "verified_at": null, "scope": null},
  "source_sha256": null,
  "data_manifest_sha256": null,
  "split_manifest_sha256": null,
  "input_verification_report": null,
  "budget": {"max_gpu_hours": null, "max_rounds": null, "steps_per_round": null, "attempts_per_question": 3},
  "stop_conditions": null,
  "promotion_criteria": null,
  "implementation_evidence": null,
  "test_evidence": null,
  "execution_evidence": null
}
```

