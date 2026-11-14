# BÁO CÁO BÀN GIAO CHI TIẾT — ZAIC DataMate 9B

**Ngày chốt:** 07/11/2026. **Đích tiếp quản:** ứng dụng Codex trên Windows, điều khiển Kaggle web bằng Browser/Computer Use khi có; không yêu cầu người dùng nhập CLI. **Trạng thái:** mã nguồn và tài liệu đã chuẩn bị; dữ liệu/trọng số chưa xác nhận tải đủ về máy; chưa chạy huấn luyện GPU trên Kaggle trong lần bàn giao này.

## 1. Đọc phần này trước khi hành động

Dự án đã có adapter 9B được huấn luyện và một bản runtime v2.1 hoàn tất public_test 1.000 câu. Việc chuyển máy/Kaggle không bắt buộc huấn luyện lại. Mục tiêu trước mắt là bảo toàn thành quả, kiểm tra tài sản, tái lập môi trường và sau đó cải thiện có đo lường.

Người dùng từng đặt mục tiêu PA = 1. Đây là mong muốn, **không phải kết quả đã đạt hay mức có thể bảo đảm**. Bản public v2.1 có 235 câu trả lời và 765 abstain; chưa có điểm chấm chính thức mới. 235 answered không được diễn giải thành 235 correct.

Gói này không phải bản sao toàn bộ Drive: không có dữ liệu giao dịch, câu hỏi/nhãn từng câu, SFT chi tiết, trọng số base, adapter nhị phân hay journal public chi tiết. Các tài sản đó phải đi trực tiếp từ Drive của người dùng về máy của người dùng. Không đẩy chúng vào chat để agent “đọc hết”.

**Không chạy Run All notebook Colab lịch sử. Không ghi đè adapter cũ. Không nối tiếp journal trên Drive/FUSE dùng chung. Không sửa fingerprint để tiếp tục một config đã đổi.**

## 2. Yêu cầu người dùng và mô hình vận hành

Người dùng muốn dùng ứng dụng Codex có các mục Choose project và This computer như ảnh, kết hợp Kaggle web; không muốn tự thao tác CLI. Cách làm gồm:

- Máy Windows là nơi lưu bản gốc, mã, manifest, trọng số và kết quả tải về.
- Codex làm việc với project mã nguồn, đọc tài liệu và báo cáo tổng hợp, tạo/sửa notebook.
- Nếu Browser/Computer Use hoạt động trong đúng phiên desktop, Codex có thể thao tác web Kaggle; phải chứng minh đọc được đúng notebook trước.
- Nếu trình duyệt tích hợp không hỗ trợ tự upload, người dùng chọn file bằng hộp thoại. Nếu phiên không có điều khiển web, dùng cùng notebook và người dùng bấm theo hướng dẫn.
- Kaggle cung cấp compute từ xa, nên cần input được chép/gắn lên Kaggle. Không có cơ chế cho GPU Kaggle đọc trực tiếp D:\ của người dùng mà mọi dữ liệu vẫn chỉ nằm trên PC.
- Không cần Kaggle CLI hay API key cho quy trình web. Không yêu cầu người dùng dán token vào notebook/chat.

Việc chọn This computer cho Codex không tự cấp GPU Kaggle hoặc quyền điều khiển tab Chrome. Kết nối Browser hiện tại của ChatGPT cũng không tự chứng minh app/phiên khác đã kết nối. Xem hướng dẫn GUI kèm theo.

## 3. Bản đồ nguồn và mức độ chắc chắn

| Tài sản | Định danh / vị trí | Trạng thái bàn giao |
|---|---|---|
| Thư mục gốc | Drive ID `1fdICQT1q-1847vsO6Sh8NN1rKhs_4mrO` | Đã mở lại và thấy các mục gốc; yêu cầu tải cả thư mục thất bại |
| Notebook lịch sử | `ZAIC_QLoRA_intent_v0.ipynb`, Drive ID `1GYahVt46wyDeGT3l3aSJGEdvdPxVMTip` | Có trong zaic; bản có nhiều cell/lịch sử 4B và 9B |
| Base/adapter 9B | `zaic/datamate_v1/qwen35_9b` | Đường dẫn lịch sử đã dùng; chưa kiểm đủ file trên PC |
| Repair v2.1 | Drive ID `1XJjbIdPBKDCWNFmGPu_hRpdlj5yZRHFB` | Nguồn source/gate/calibration/fragment; cần tải riêng |
| Resume của tài khoản GPU | Drive ID `1bPHpPmC29_qar42kEIPeXSGgbg_PxNkh` | Nguồn checkpoint bất biến và thư mục kết quả cuối |
| Public v2.1 verified | Drive ID `1TupBKE0PL5DFsYGoBVq5qqmok-4Vn4WV` | Bộ cuối đã được đối chiếu ở lượt chạy trước, gồm 9 file |
| Mã nguồn v2.1 | `frozen_v21/` trong gói | 27 file app/tools, checksum giữ nguyên |
| Hạ tầng khôi phục Colab | 2 script gốc trong frozen_v21 | Chỉ làm bằng chứng/hướng dẫn; có đường dẫn/ID riêng, không portable |
| Thư mục lịch sử ngoài các nguồn trên | Repair_v2, 9B public v1 nếu còn | Chưa có kiểm kê đệ quy đầy đủ; cần tìm trên Drive của người dùng |
| Dữ liệu/weights trên PC | Đường dẫn thực do người dùng chọn | Chưa kiểm chứng; không được đánh dấu migrated=true |

Xem DOWNLOAD_AND_GUI_VI.md và TRANSFER_MANIFEST.json để lấy URL có thể mở. Link có authuser chỉ là chỉ số phiên trình duyệt; không dùng số đó làm định danh chủ tài khoản cố định. Không mở quyền public để giải quyết lỗi truy cập.

Ngày bàn giao, notebook Kaggle được quan sát ở `https://www.kaggle.com/code/lnghongphc/notebookd504e83869/edit`: Draft saved, 0 versions, session off, chưa gắn Input và chỉ có cell mẫu. Không có bằng chứng model đang train tại Kaggle.

## 4. Cây thư mục nên dùng trên máy

| Vị trí | Nội dung chính |
|---|---|
| `D:\ZAIC\CODEX_PROJECT` | README, AGENTS, docs, frozen_v21, notebook mới, GUI utilities |
| `D:\ZAIC\PRIVATE_DATA\zaic\train` | Bộ TRAIN gốc và nhãn TRAIN |
| `D:\ZAIC\PRIVATE_DATA\zaic\public_test` | Câu hỏi và dữ liệu public, không có gold được phép dùng |
| `D:\ZAIC\PRIVATE_DATA\zaic\datamate_v1\qwen35_9b\assets\base` | Config/tokenizer/index và toàn bộ weight shards |
| `...\qwen35_9b\candidate\final` | adapter_model.safetensors và adapter_config/tokenizer liên quan |
| `...\qwen35_9b\data` | SFT/manifest/split/calibration/audit đã đóng băng |
| `...\qwen35_9b\candidate\checkpoints` | Checkpoint huấn luyện cũ, nếu có |
| `D:\ZAIC\PRIVATE_DATA\repair_v21` | Evidence và fragment sửa v2.1 |
| `D:\ZAIC\PRIVATE_DATA\resume_own` | Checkpoint khôi phục và bộ verified cuối |
| `D:\ZAIC\DOWNLOADS_ARCHIVE` | Mọi ZIP tải từ Drive, chưa xóa |
| `D:\ZAIC\PRIVATE_DATA\runs` | Các run mới và outputs Kaggle tải về |

Base khoảng 19 GB; chưa có tổng dung lượng dự án. Phải tính cả nhiều bản checkpoint/model cũ, ZIP và file giải nén. Không dùng kích thước ZIP mã nhỏ để suy ra mô hình nhỏ. Kiểm tra shortcut, file trùng tên và thư mục lồng khi giải nén.

## 5. Bài toán và kiến trúc giải pháp

DataMate nhận câu hỏi kinh doanh tiếng Việt, ngày tham chiếu as_of_date và tập dữ liệu của một doanh nghiệp. Dữ liệu gồm bảng giao dịch/danh mục, README theo doanh nghiệp và rulebook dùng chung. Mô hình không được dùng như bộ nhớ chứa các đáp án public.

Pipeline hiện tại:

1. Đọc schema/dữ liệu, phân loại vai trò file/sheet, chuẩn hóa cột/ngày/số, nối header hóa đơn và dòng chi tiết khi có mã đáng tin cậy.
2. Tạo ngữ cảnh rút gọn: phạm vi thời gian, schema, thực thể ứng viên và các quy tắc liên quan. Chọn quy tắc có liên quan thay vì bỏ mất điều kiện quan trọng.
3. Qwen3.5-9B + LoRA sinh **kế hoạch JSON có cấu trúc**. Thinking tắt; decoding greedy.
4. Kiểm tra schema và hợp đồng kế hoạch với câu hỏi: metric, thời gian, bộ lọc, đơn vị, group_by, order, limit, answer_type, unresolved.
5. Bộ thực thi xác định tính kết quả từ dữ liệu nguồn. Nếu không xác minh đủ nghĩa/phép toán/phạm vi, trả abstain với lý do, không đoán số.
6. Xuất JSONL theo thứ tự ID câu hỏi, kiểm format và nhật ký.

Các kiểu output quan sát: number, string, list, table, percent, date, abstain. Unanswerable có ý nghĩa nghiệp vụ/khả năng trả lời riêng; không dùng thay abstain để làm số liệu đẹp.

Không có API LLM bên ngoài trong inference. Quy trình backup/khôi phục từng dùng Drive API bên ngoài tiến trình inference. Vì vậy không được gọi toàn bộ controller Colab là entrypoint offline/network-none.

## 6. Bản đồ mã nguồn cho agent

| File / nhóm | Vai trò / lưu ý |
|---|---|
| `app/legacy_engine.py` | Đọc/chuẩn hóa nguồn và thực thi nền, nhiều nghiệp vụ đã kiểm thử |
| `app/planner.py` | Prompt, context, schema/parse plan, kiểm tra và trả lời |
| `app/v2_contracts.py` | Ràng buộc metric/thời gian/điều kiện/phạm vi giúp tránh trả sai |
| `app/modeling.py` | Loader Qwen3_5ForCausalLM, NF4, local_files_only, cuda:0 |
| `app/train.py` | QLoRA fresh adapter, completion-only loss, rank8, checkpoints |
| `app/build_data.py` | Split theo doanh nghiệp; tạo SFT từ TRAIN và weak plan được lọc bằng thực thi |
| `app/smoke_model.py` | So loss full/sliced logits, backward giả lập và finite gradients |
| `app/infer.py`, `tools/infer_batched.py` | Inference và bản batch có journal/fingerprint/atomic outputs |
| `app/scoring.py`, `app/evaluate_saved.py` | Chấm nội bộ TRAIN, không phải dịch vụ chấm của ban tổ chức |
| `app/abstention_policy.py`, `tools/fit_abstention.py` | Chính sách tùy chọn; chính sách học từng không được chọn vì giảm PA |
| `tools/audit_output.py`, `tools/release_test.py` | Format/ID, cổng kiểm tra phát hành |
| `tools/run_repair_v2.py`, `tools/run_repair_v21.py` | Điều phối thử nghiệm sửa đã có, nhiều path/assumption cần đọc trước |
| `tools/replay_train.py`, `tools/inventory.py`, `tools/verify_ingestion.py` | Chẩn đoán tổng hợp, kiểm input/loader |
| Các `test_*.py` trong app/tools | 99 tests tổng hợp/giả lập, không cần dữ liệu cuộc thi |
| `app/workflow.py` | Workflow lịch sử có train và mở audit; không chạy lại máy móc vì audit đã dùng |
| `resume_public_v21.py`, `run_public_local_with_checkpoints.py` | 2 script khôi phục Colab riêng thí nghiệm; không chạy trên Kaggle nguyên xi |
| `gui/kaggle_runner.py` | Wrapper mới cho notebook GUI, tách khỏi mã đóng băng |
| `gui/local_check.pyw` | Checker chọn thư mục bằng cửa sổ, không có mạng |

Mã v2.1 đóng băng trong gói lấy từ archive nguồn đã lưu phiên bản 2; 27 file Python app/tools và 2 recovery script. File verify_ingestion.py đã được chuẩn hóa đúng một newline cuối trong archive nguồn để khớp bản gốc. Không sửa bytes source để tiện chuyển nền tảng. Các thay đổi portability mới chỉ nằm ngoài frozen_v21.

## 7. Model, trọng số và huấn luyện đã thực hiện

| Thuộc tính | Giá trị đã ghi nhận |
|---|---|
| Base | Qwen/Qwen3.5-9B |
| Revision | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| License trong model_lock | apache-2.0; giữ LICENSE/model card thực tế để đối chiếu khi đóng gói |
| Model loader | Qwen3_5ForCausalLM, text-only |
| Quantization | 4-bit NF4, double quantization, fp16 compute |
| Adapter cũ 9B | SHA256 `e7dfa635beccce128c99f9d3fd993d33dbb4c88e73e3ddeccbc2bc915eec4a4d` |
| Reuse adapter 4B | Không; 9B đã train adapter mới |
| LoRA | r=8, alpha=16, dropout=0.05, all-linear, bias none |
| Optimizer | paged_adamw_8bit |
| Learning rate | 2e-5, cosine, warmup khoảng 5% steps |
| Epoch / steps | 1 epoch, 48 steps |
| Rows / max tokens | 768 rows dùng train; max_length 2048 |
| Batch / accumulation | 1 / 16 |
| Trainable / total | 21.639.168 / 8.975.442.432 parameters |
| Gradient checkpointing | Có, use_reentrant=False |
| FP16 loss scale | init_scale=16, max_grad_norm=0.3; dừng sau 3 grad_norm không hữu hạn liên tiếp |
| Checkpoint | mỗi 5 steps, giữ tối đa 3 |
| Seed | 20261107 |
| Thời gian train / VRAM | 3.670,781 giây / 8,660 GiB peak |

Completion-only loss: prefix có label -100, chỉ phần plan completion tham gia loss. Dùng logits_to_keep=n+1 để tránh materialize full vocab logits của toàn prompt. Trước train, smoke test đối chiếu loss với full logits, sai lệch dưới 0.02 và backward phải có gradient hữu hạn. Embedding/head bị freeze rất lớn nên giữ fp16; norms và tham số LoRA phù hợp được giữ fp32. Không gọi blanket upcast toàn mô hình gây OOM T4.

v2/v2.1 **không huấn luyện lại**. Những cải thiện PA nội bộ của v2.1 do sửa loader/contracts/executor/runtime. Không mô tả đó là adapter mới tốt hơn.

### Phân chia TRAIN và SFT

Split theo doanh nghiệp, không trộn câu của cùng doanh nghiệp vào train và evaluation:

| Vai trò | Doanh nghiệp |
|---|---|
| Train (18) | biz102, biz103, biz104, biz105, biz107, biz108, biz110, biz113, biz115, biz116, biz118, biz119, biz120, biz122, biz123, biz126, biz129, biz130 |
| Calibration (6) | biz101, biz109, biz112, biz114, biz121, biz128 |
| Audit/confirmation (6) | biz106, biz111, biz117, biz124, biz125, biz127 |

SFT đã chọn 768/1.078 mẫu khả dụng: 393 constructed_plan_template, 375 execution_filtered_weak_plan. Weak plan được chuyển sang hợp đồng mới và chỉ giữ khi thực thi khớp nhãn TRAIN. Không dùng gold public/private, không viết đáp án theo ID.

SHA256 tập SFT đã đóng băng: `5057658ac10c19579c675c9f926cc13f1e17641b88eb312318e48be75e6bb170`. Tái lập baseline cần giữ chính xác file và manifest. Nếu tạo SFT mới sau sửa loader, đó là thí nghiệm mới, phải có split/hash/selection/evaluation mới; không sửa manifest cũ cho vừa.

Các doanh nghiệp audit đã từng xuất hiện trong lịch sử phát triển 4B. Không được gọi chúng là chưa từng tiếp xúc trong toàn dự án. Confirmation2 loại ID từng đánh giá, cố định trước sửa loader; sau khi đã mở thì cũng không còn là holdout mới cho vòng tiếp theo.

## 8. Kết quả đã đo và cách đọc

Mã chấm nội bộ đang dùng PA=(correct−wrong)/N. Đối chiếu lại định nghĩa chính thức từ thể lệ/scorer trước nộp; screenshot điểm âm không tự chứng minh sai format.

| Tập / lượt | N | Đúng | Sai | Abstain | PA |
|---|---:|---:|---:|---:|---:|
| 9B ban đầu calibration nhỏ | 24 | 10 | 4 | 10 | 0,25 |
| 9B ban đầu audit nhỏ | 48 | 14 | 3 | 31 | 0,229167 |
| TRAIN calibration 200, runtime cũ | 200 | 89 | 29 | 82 | 0,30 |
| TRAIN calibration 200, v2/v2.1 sinh mới | 200 | 91 | 21 | 88 | 0,35 |
| TRAIN confirmation2 v2.1 | 100 | 36 | 6 | 58 | 0,30 |
| Public v2.1 | 1.000 | Chưa biết | Chưa biết | 765 | Chưa chấm mới |

Answered accuracy confirmation2 = 36/(36+6)=85,71%; coverage=42%; PA=0,30. Calibration200 answered accuracy=91/112=81,25%. Phải báo cả PA, coverage, answered accuracy và số correct/wrong/abstain.

Điểm public lịch sử: 4B −0,20700; 9B trước repair −0,09700 từ thông tin/screenshot người dùng. Không có biên bản đối chiếu upload hash của ảnh cũ nên không gán hai điểm đó cho submission v2.1.

Confirmation2 SHA256: `fa7250d85bf5c162c7934c909fae1fa21cc34c0e9a8559b88d3e6b6d881cfc8c`. Được freeze trước sửa loader và loại các ID evaluation trước đó; dữ liệu vẫn thuộc TRAIN và có khai báo exposure lịch sử.

99 kiểm thử: 75 app +24 tools. Kiểm thử đều đạt tại môi trường sửa/khôi phục trước, và được chạy lại từ mã đóng băng khi chuẩn bị gói bàn giao này. Không suy từ 99 tests sang độ chính xác public hay tương thích GPU Kaggle.

## 9. Những lỗi đã sửa trong v2/v2.1

### Lỗi nhận dạng file và nối giao dịch quan trọng nhất

`BanHang_HoaDon.xlsx` được chuẩn hóa thành `banhanghoadon`, vô tình chứa chuỗi `hanghoa` qua ranh giới từ, dẫn đến nhận nhầm hóa đơn bán hàng thành danh mục hàng hóa. Sửa ưu tiên nhận dạng chi tiết giao dịch/hóa đơn trước danh mục chung. Sheet có vai trò rõ được ưu tiên; sheet không rõ dùng tên file làm fallback.

Bổ sung các biến thể tên cột mã hóa đơn/mã giao dịch để nối header và dòng hàng. Kiểm tra TRAIN phục hồi 95.845 dòng chi tiết và 51.044 đơn; 95.845 dòng có order_id và ngày bán, phạm vi 01/01/2025–25/05/2026. Đây là kiểm tra cấu trúc/ingestion, không phải chứng minh toàn bộ nghiệp vụ đã đúng.

### Các sửa hợp đồng và độ an toàn kết quả

- Kiểm tra thời gian lồng nhau; không đoán năm từ đồng hồ hệ thống; không gọi month_bounds sai số đối số.
- Giữ quy tắc README/rulebook liên quan trong context; đối chiếu thời gian câu hỏi với kế hoạch.
- Chặn mâu thuẫn rõ về metric, thời gian, bộ lọc không tiêu thụ, đơn vị và điều kiện phủ dữ liệu.
- Xử lý ranking khách thiếu tên qua mã định danh; tránh nhập nhằng khách lẻ.
- Bảo toàn lượng thập phân; không lấy tổng tiền phải trả của hóa đơn làm doanh thu từng dòng.
- Ghi vị trí/lý do lỗi, xuất JSONL có audit; tránh im lặng trả kết quả bị lệch nghĩa.
- Release gate yêu cầu không có loader/network/OOM. Chính sách abstention được fit thử đã không được chọn vì PA giảm so raw.

Không có nhánh theo một business/question ID để làm đẹp điểm. Cải thiện tiếp nên tập trung nghiệp vụ chung có bằng chứng và test TRAIN.

## 10. Khôi phục public_test và bài học checkpoint

Lượt public đầu lưu được 611 plans rồi Colab ngắt và hết hạn mức GPU. Người dùng cung cấp phiên T4 ở tài khoản khác. Môi trường được khôi phục đúng dependency và 99 tests; quyền truy cập Drive được người dùng xử lý qua UI.

Lượt tiếp trên Drive dùng chung sinh thêm 68 plans nhưng file journal/submission xuất hiện nhiều bản trùng tên do hành vi Drive/FUSE. Đã dừng process, không xóa/mất fragment. 17 file được đặt tên riêng `recovery_plan_part_001..017.jsonl`; thư mục cũ giữ dưới tên `public_test_v21_preserved_fragments`.

Sau xác thực Colab thông thường do người dùng chấp thuận, tải từng fragment qua Drive API về ổ cục bộ, kiểm MD5 từng file, merge theo ID và yêu cầu bản ghi trùng phải giống hệt. Kết quả: 679 unique plans, 0 conflicts, không sửa giá trị. Fingerprint giữ nguyên bytes; key `96945a4b5bc607922f1039219d7ce659c751ace5fccd2ce73b8d57a7771ba5a7`.

Chạy phần còn lại trên local disk: 679 journal_replay, 321 model calls mới, đủ 1.000 kết quả. Tiến trình bên ngoài inference lưu checkpoint bất biến lên Drive riêng với tên chứa count/SHA, kiểm MD5/kích thước. Không append/overwrite journal đang sống trên shared FUSE nữa.

Mọi đường dẫn `/content/...` mô tả phiên Colab đã chạy, có thể không còn tồn tại. Không dùng việc notebook có output để suy ra runtime/RAM/GPU còn sống.

## 11. Bộ kết quả cuối cần bảo toàn

Thư mục `public_test_v21_verified_20261107T090319Z`, Drive ID `1TupBKE0PL5DFsYGoBVq5qqmok-4Vn4WV`, có:

1. submission.jsonl
2. plans.jsonl
3. question_logs.jsonl
4. fingerprint.json
5. format_audit.json
6. runtime.json
7. REPAIR_V21_RESUMED_COMPLETED.json
8. DRIVE_CHECKPOINT_MANIFEST.json
9. JOURNAL_RECOVERY_VERIFIED.json

Submission verified: **85.573 bytes**, SHA256 **`e0815c96524e9db0badf987e45e13227eec8e57c7f3195f5aa93a580a0508780`**. Không lấy file cùng tên ở vùng fragment/phiên lỗi.

Đã xác minh trước khi bàn giao: 1.000 dòng, 1.000 unique IDs đúng thứ tự câu hỏi; 1.000 plans; 1.000 logs; audit passed. Kiểu output: number174, string22, list16, percent11, table8, date4, abstain765. Không có unanswerable. Loader errors={}, OOM retries=0, network attempts trong inference=0, không có executor_exception trong lý do bỏ qua cuối.

Lượt chạy cuối 1.491,575 giây, gồm startup95,665 và generation1.303,404; peak8,7906GiB. Không cộng nhầm thành tổng thời gian toàn dự án: chưa gồm các lượt gián đoạn, sửa code, xác thực, khôi phục và backup.

Chưa nộp bản cuối lên ban tổ chức bằng phiên làm việc này. Chưa biết PA public mới. Chưa xây/chạy chứng nhận Docker network-none hoặc RTX5090.

## 12. Những phần còn yếu và ưu tiên tiếp theo

Các lý do từ executor public (không phải nhãn chấm public):

| Nhóm lý do | Số lần |
|---|---:|
| Unparsed source dates | 133 |
| invalid_plan | 106 |
| Plan period contradicts explicit question period | 83 |
| Composite/threshold operation chưa hỗ trợ | 60 |
| Unparsed numeric amounts | 59 |
| Metric contradiction | 43 |
| Required semantic columns chưa xác minh | 37 |
| Snapshot/movement executor chưa xác minh | 30 |
| Percent metric mismatch | 29 |
| Numeric comparator operation chưa xác minh | 26 |

Dùng phân bố tổng hợp để đặt giả thuyết kỹ thuật. Kiểm tra và sửa trên TRAIN; không xem câu public rồi viết nhãn/đáp án thủ công. Không ép “trả lời hết” bằng cách bỏ các guard: với PA có trừ sai, coverage tăng có thể làm điểm giảm.

Lộ trình v2.2 đề xuất:

1. Hoàn tất kiểm kê/tải và bảo toàn baseline.
2. Freeze tập đánh giá TRAIN mới theo kế hoạch exposure rõ ràng trước sửa. Calibration/confirmation2 hiện tại chỉ dùng tham khảo/hồi quy, không gọi untouched holdout.
3. Sửa parser ngày/số Việt Nam với dữ liệu giả lập đại diện, kiểm join/currency/unit và coverage.
4. Mở rộng hợp đồng plan cùng executor cho comparator, threshold, distinct count, tồn kho/công nợ; test nghiệp vụ có tính tay được.
5. Nếu đổi schema plan, đồng bộ validation/prompt/SFT/executor và version; không lặng lẽ đọc journal schema cũ.
6. Tạo SFT TRAIN mới khi cần, loại mẫu sai/overlength, kiểm split/hash, rồi train adapter mới vào run riêng.
7. Đánh giá base so candidate trên cùng tập/tiêu chí, chọn bằng calibration, mở holdout đã freeze một lần, báo uncertainty.
8. Kiểm batch parity và format trên môi trường mới; chỉ chạy public sau khi cổng phát hành đạt, không dùng nhãn public để chọn.
9. Chuẩn bị artifact triển khai offline và đo GPU/timing thực tế theo yêu cầu chính thức trước tuyên bố sẵn sàng nộp.

## 13. Môi trường Colab đã dùng và chuyển sang Kaggle

Snapshot lịch sử:

| Thành phần | Phiên bản |
|---|---|
| Python | 3.13.16 |
| torch | 2.11.0+cu130 |
| transformers | 5.22.0 |
| peft | 0.21.2 |
| accelerate | 1.15.0 |
| bitsandbytes | 0.50.2 |
| pandas | 2.2.3 |
| numpy | 2.4.6 |
| safetensors | 0.8.0 |
| huggingface-hub | 1.33.0 |
| openpyxl | 3.1.5 |

Engine/planner `datamate-qwen35-9b-0.2.1`, legacy parser `datamate-engine-0.4.2`, batched runtime `batch-0.2.0`. Public: batch4, input<=3072, max_new_tokens320, do_sample=False, thinking=False.

Snapshot không phải cam kết wheel tương thích Kaggle. Driver/CUDA/Python/GPU có thể khác. Không cài đè nguyên torch+cu130 trên Kaggle mà không kiểm tra. Ghi versions/GPU capability, thử import Qwen3_5ForCausalLM/PEFT/bnb rồi smoke test. Source test không tải GPU, nên vẫn có thể pass khi stack GPU sai.

Kaggle `/kaggle/input` là input được gắn, `/kaggle/working` là nơi ghi outputs. Base nên symlink/read trực tiếp từ input, không chép vào working. Model loader hiện device_map={'':0}; không có DDP/model sharding cho T4×2. Không tự đổi multi-GPU trong bước migration cơ bản.

## 14. Notebook mới trong gói làm được gì

`notebooks/ZAIC_Kaggle_GUI.ipynb` là notebook sạch, không có output, không mount Drive và không dùng Kaggle CLI. Nó cần ZIP mã làm input, sau đó:

- ACTION='preflight' (mặc định): giải nén code an toàn, verify 27 source hashes, chạy 99 tests giả lập, ghi báo cáo versions; không dùng weights/raw/SFT, không train.
- ACTION='smoke': sau khi người dùng bật GPU và điền base input, chạy smoke_model.py trên dữ liệu giả lập, kiểm loss/gradient.
- ACTION='train': kiểm quyền xử lý dữ liệu cuộc thi, SFT frozen checksum/manifest; tạo run mới, smoke rồi train fresh adapter với cấu hình cũ. Không load adapter cũ để tiếp tục fine-tune, không resume optimizer cũ, không chấm/nộp public tự động.

Mỗi lượt GPU dùng timestamp mới. Notebook không tự cài package hoặc tự tải model từ Internet. Nếu môi trường không tương thích, dừng và đưa báo cáo versions/lỗi tổng hợp cho Codex để sửa cell môi trường. Cần đánh giá TRAIN và portability/batch-parity trước dùng model mới.

**Chưa chạy GPU Kaggle trong lần bàn giao này.** Những thứ đã kiểm là source hashes, cú pháp notebook/wrapper và kiểm thử cục bộ được ghi ở reports/VALIDATION.json. Không thay thế bằng lời hứa notebook sẽ chạy GPU ngay ở mọi tài khoản.

## 15. Điều kiện dữ liệu và an toàn vận hành liên quan trực tiếp dự án

Bản ghi thể lệ trước đây có yêu cầu không chia sẻ/chuyển dữ liệu cuộc thi ra ngoài đội hoặc đến bên thứ ba; dữ liệu dùng cho mục đích phi thương mại/học thuật, ghi nguồn challenge.zalo.ai. Agent tiếp quản cần đọc lại bản chính thức hiện hành trước khi đưa raw/SFT lên Kaggle. SFT có thể chứa schema/thực thể/thông tin từ dữ liệu, không được coi tự động là dữ liệu công khai vì mang tên training.

Handoff này chưa xác minh ngoại lệ cho Kaggle; không kết luận dùng Kaggle bị cấm tuyệt đối, cũng không kết luận Private là đủ. Nếu có xác nhận của ban tổ chức/thể lệ cho phép môi trường đó, ghi lại bằng chứng rồi mới stage input trong phạm vi cho phép. Nếu không được phép, giữ dữ liệu trên máy hoặc môi trường được cho phép, dùng Kaggle chỉ với dữ liệu giả lập/công khai phù hợp.

Không tải raw competition data vào chat/Library hoặc public repository. Việc đặt PRIVATE_DATA ngoài CODEX_PROJECT giảm lẫn dữ liệu vào context nhưng không phải sandbox. Tự thực thi script cục bộ có thể xử lý raw; chỉ đưa hash/count/aggregate hoặc tái hiện giả lập vào hội thoại AI.

Không chỉnh quyền Drive để ai có link cũng xem. Không đưa API token/password vào notebook. Không tự nộp bài/công khai artifacts. Thực hiện yêu cầu tắt mạng ở inference triển khai bằng test thực tế, không suy từ local_files_only là mọi networking đều đã bị chặn.

## 16. Checklist định nghĩa “đã chuyển toàn bộ”

Chỉ tuyên bố hoàn tất khi có bằng chứng trên chính máy người dùng:

- [ ] Mọi ZIP tải xong, không file tải dở, giải nén không lỗi.
- [ ] Kiểm kê nguồn/đích theo file, kích thước và checksum phù hợp; xử lý mọi shortcut/trùng tên.
- [ ] TRAIN có questions/answers/rulebook và đủ thư mục dữ liệu; public có questions/rulebook/data đúng phiên bản.
- [ ] Dữ liệu SFT/manifest/split/training metadata/selection và lịch sử cần thiết đã giữ.
- [ ] Base có config/tokenizer/index và mọi shard theo index; so hash với nguồn tin cậy.
- [ ] Adapter 9B cũ đúng SHA256 và adapter_config; các adapter 4B được tách rõ.
- [ ] Repair v2/v2.1, fragment và resume_own/checkpoint được bảo toàn; các thư mục ngoài zaic đã kiểm kê.
- [ ] Bộ verified đủ 9 file; submission 1.000 unique IDs và đúng SHA256 nêu trên.
- [ ] Notebook gốc lưu riêng; không lẫn output lịch sử vào notebook mới gửi AI.
- [ ] Project Codex chọn đúng CODEX_PROJECT; checker xuất báo cáo không lộ chi tiết dữ liệu.
- [ ] 99 source tests đạt; trạng thái GPU/Kaggle được báo riêng.
- [ ] Chưa xóa Drive/ZIP gốc trước khi kiểm xong và có bản sao lưu phù hợp.

Checker GUI chỉ xác minh một tập tài sản baseline và submission hash; nó cố ý không đặt full_project_transfer_complete=True. Cần kiểm kê đầy đủ các nguồn để đóng mục cuối cùng.

## 17. Công việc đầu tiên cho agent tiếp quản

Đọc tài liệu và AGENTS; đối chiếu reports/VALIDATION.json. Xác định người dùng đã giải nén gói và các nguồn Drive ở đâu; chỉ hỏi đường dẫn khi chưa thể suy ra từ project/config. Kiểm code/hash/tests trước, xử lý bản thiếu bằng GUI, tránh tải lặp toàn bộ19GB.

Sau đó xác nhận Browser trong đúng app/phiên có truy cập đúng notebook Kaggle. Kiểm tra Input/session đang off/on và GPU có được cấp. Chưa upload raw/SFT đến khi quyền xử lý được xác minh. Nhập notebook preflight; nếu built-in browser yêu cầu upload thủ công, bàn giao đúng hộp thoại cho người dùng chọn file.

Báo ngắn: phần đã đủ, phần thiếu, kết quả kiểm tra và bước GUI kế tiếp. Không nói “đã huấn luyện trên Kaggle” khi mới import notebook hoặc source test pass. Không gọi bản sửa loader là huấn luyện mới. Không hứa PA1.

## 18. Căn cứ và tài liệu ưu tiên khi có mâu thuẫn

Ưu tiên: file/manifest/hash thực tế trong bộ verified và source đã freeze → báo cáo tổng hợp v2.1 hiện hành → báo cáo lịch sử → diễn giải trong hội thoại. Thời điểm snapshot cũ không đại diện runtime đang sống.

Tài liệu gốc đi kèm: `reports/ZAIC_repair_v21_worklog.md`, `frozen_v21/AGGREGATE_RESULTS.json`, `frozen_v21/model_lock.json`, `frozen_v21/versions.json`, source/checksum. Chi tiết huấn luyện/split ở trên kế thừa biên bản lịch sử; cần so lại training_complete/manifest khi người dùng đã tải về, không truy xuất raw để in vào chat.

Nguồn hướng dẫn giao diện: tài liệu chính thức OpenAI tại learn.chatgpt.com/docs/projects và /docs/browser; Kaggle tại kaggle.com/docs/notebooks; giao diện Kaggle/Drive được trực tiếp đọc trong lần chuẩn bị. Không dùng ảnh ứng dụng để suy ra mọi công cụ Browser đã bật.

Nguồn dữ liệu cuộc thi: **challenge.zalo.ai**.


---

# PHỤ LỤC — TẢI DỰ ÁN VÀ THAO TÁC GUI

# Chuyển dự án về Windows và điều khiển Kaggle bằng giao diện

## 1. Tình trạng của lần chuẩn bị này

Đã đóng gói nguồn v2.1, tài liệu và notebook mới. Đã thấy notebook Kaggle `notebookd504e83869` là Draft, chưa chạy, chưa gắn Input. Đã mở lại Drive `zaic` và yêu cầu Download cả thư mục: Drive báo Zipping rồi **Can't download file**. Vì vậy **chưa có bằng chứng toàn bộ dự án đã tải xuống máy**. Không bấm tải lại cả thư mục liên tục. Chuyển sang tải các thư mục nhỏ bên dưới hoặc dùng ứng dụng Google Drive chính thức nếu cần sao chép lớn; chưa thực hiện cài đặt ứng dụng đó.

## 2. Tổ chức thư mục trên máy

Dùng File Explorer để tạo các thư mục sau. Có thể đổi D: thành ổ còn đủ chỗ.

| Thư mục | Nội dung | Mở làm project Codex? |
|---|---|---|
| `D:\ZAIC\CODEX_PROJECT` | Toàn bộ nội dung CODEX_PROJECT trong ZIP bàn giao | Có |
| `D:\ZAIC\PRIVATE_DATA\zaic` | Dữ liệu, SFT, base/adapter và lịch sử gốc | Không đưa cả thư mục vào ngữ cảnh AI |
| `D:\ZAIC\PRIVATE_DATA\repair_v21` | Thư mục sửa v2.1, tập hiệu chỉnh, các fragment cũ | Không |
| `D:\ZAIC\PRIVATE_DATA\resume_own` | Checkpoint bất biến và bộ kết quả cuối | Không |
| `D:\ZAIC\PRIVATE_DATA\history_extra` | Các thư mục lịch sử ngoài zaic, nếu có | Không |
| `D:\ZAIC\DOWNLOADS_ARCHIVE` | ZIP tải từ Drive, giữ nguyên cho đến đối chiếu xong | Không |
| `D:\ZAIC\KAGGLE_STAGING` | Input được phép đưa lên Kaggle, chuẩn bị riêng | Không |

Thư mục chọn cho Codex giúp quản lý ngữ cảnh, không phải cơ chế sandbox chắc chắn chặn mọi truy cập ổ đĩa. Agent phải tuân thủ AGENTS.md, xử lý dữ liệu bằng chương trình cục bộ và chỉ đọc báo cáo tổng hợp.

## 3. Danh sách cần tải

| Nguồn | Cần giữ | Đích gợi ý |
|---|---|---|
| [zaic gốc](https://drive.google.com/drive/folders/1fdICQT1q-1847vsO6Sh8NN1rKhs_4mrO) | Mọi file/thư mục bên trong, kể cả bản cũ | `PRIVATE_DATA\zaic` |
| [ZAIC_9B_Repair_v21_20261107](https://drive.google.com/drive/folders/1XJjbIdPBKDCWNFmGPu_hRpdlj5yZRHFB) | Source, gate/freeze, calibration/confirmation, fragment bảo toàn | `PRIVATE_DATA\repair_v21` |
| [ZAIC_9B_Resume_Own_20261107](https://drive.google.com/drive/folders/1bPHpPmC29_qar42kEIPeXSGgbg_PxNkh) | Tất cả checkpoint bất biến và thư mục verified | `PRIVATE_DATA\resume_own` |
| [Kết quả verified cuối](https://drive.google.com/drive/folders/1TupBKE0PL5DFsYGoBVq5qqmok-4Vn4WV) | Đủ 9 file; có thể đã nằm trong resume_own | Đối chiếu, không cần tạo bản trùng nếu đã đủ |
| Notebook gốc `ZAIC_QLoRA_intent_v0.ipynb` trong zaic | Bản gốc có lịch sử output, dùng để tham khảo | Giữ riêng, không Run All và không upload nội dung có dữ liệu vào AI |
| Các thư mục 9B v1 / Repair_v2 ngoài zaic, nếu tìm thấy | Lịch sử đánh giá và cấu hình cũ | `PRIVATE_DATA\history_extra` |

Hai nguồn cuối của lịch sử không có biên bản liệt kê đệ quy đầy đủ ở lần bàn giao này. Để tuyên bố “toàn bộ dự án”, cần kiểm kê thêm các thư mục liên quan ngoài ba thư mục chính. Không thay file trong verified bằng file cùng tên ở thư mục cũ.

Tại zaic, giao diện đã xác nhận các mục: train, sft_data, public_test, datamate_v1, adapter_intent_v0_run1_20261106_044117, adapter_intent_v0_20steps_20261106_043640, adapter_intent_v0, ZAIC_QLoRA_intent_v0.ipynb, normalize_vi.py, intents.py. Không chỉ lấy file notebook.

### Tải bằng Drive web

1. Mở đúng nguồn và tài khoản có quyền. Nếu báo không có quyền, đổi sang tài khoản sở hữu/được chia sẻ; không đổi link thành public.
2. Với thư mục lớn, tải từng thư mục con bằng menu **Download**. Base model nên tải từng shard nếu tải cả thư mục thất bại. Không dùng phím/đường dẫn để né cảnh báo bảo mật.
3. Giữ Chrome mở trong lúc Drive chuẩn bị ZIP. Kiểm tra danh sách Downloads của trình duyệt; ZIP phải tải xong, không còn `.crdownload`.
4. Drive có thể phát nhiều ZIP độc lập, không mặc định là một bộ `.zip.001`. Giải nén từng ZIP vào vùng tạm riêng, sau đó gộp đúng cây thư mục. Khi tên trùng nhưng nội dung khác, giữ hai bản và so hash; không chọn Replace all.
5. Trong File Explorer, nhấp phải ZIP → **Extract All**. Tránh cấu trúc lồng `zaic\zaic\train`; thư mục chọn cho checker phải trực tiếp chứa train/public_test/datamate_v1.
6. Đảm bảo không chỉ tải shortcut trỏ đến thư mục khác. Mở đích của shortcut và lấy các file thực. Ảnh chụp thư mục hoặc file `.url` không thay được trọng số.
7. Lưu toàn bộ ZIP nguyên bản ở DOWNLOADS_ARCHIVE, rồi mới kiểm tra.

Base 9B khoảng 19 GB; toàn bộ lịch sử còn có checkpoint và các model cũ. Không có số tổng đã kiểm kê. Cần chỗ cho cả ZIP, dữ liệu giải nén và output mới; đo dung lượng thực trước, không suy từ dung lượng RAM/GPU.

## 4. Kiểm tra sau khi tải

Nhấp đúp `gui/local_check.pyw` nếu máy có Python 3.10+ với tkinter. Công cụ không có mạng, không cần Kaggle key, không hiển thị câu hỏi/đáp án/giao dịch.

- Chọn thư mục `zaic`, lưu JSON kiểm tra tại thư mục riêng.
- Chọn thư mục verified, kiểm tra đủ 9 file và hash submission.
- Có thể chọn tính hash các shard; việc này tạo bản kiểm kê cục bộ. Phải so hash nguồn/manifest nguồn mới chứng minh được tính toàn vẹn của trọng số.
- Các checks cơ bản đạt vẫn chưa chứng nhận toàn bộ file lịch sử Drive đã về đủ. Phải đối chiếu kiểm kê file nguồn với đích, đặc biệt repair_v21/resume_own và các thư mục lịch sử ngoài zaic.
- Nếu JSON báo thiếu train/public hoặc base, sửa đường dẫn/cây thư mục trước khi tải lại.

Nếu Python chưa có hoặc double-click mở bằng trình soạn thảo, Codex có thể giúp thiết lập Python chính thức bằng giao diện hoặc chạy checker nội bộ. Người dùng không cần tự mở terminal.

## 5. Codex desktop và điều khiển Kaggle web

1. Trong ảnh ứng dụng: **Choose project**, chọn CODEX_PROJECT; giữ **This computer**. Dán START_CODEX_PROMPT_VI.txt để bàn giao ngữ cảnh.
2. Để điều khiển web, phiên desktop cần Browser/Computer Use hoạt động. Với trình duyệt tích hợp, dùng `@Browser`; nếu có Settings → Browser → Agent permissions, cho phép tác vụ tại kaggle.com. Trình duyệt tích hợp có phiên đăng nhập riêng.
3. Muốn dùng tab Chrome đang mở, dùng kết nối extension chính thức được ứng dụng hỗ trợ. Agent phải xác nhận đọc được đúng notebook trước khi thao tác; có extension ở phiên ChatGPT này không tự chứng minh ứng dụng khác đã kết nối.
4. Nếu không thấy công cụ Browser, cập nhật ứng dụng chính thức/kiểm tra tính năng tài khoản; không có nút bật nào được bảo đảm xuất hiện ở mọi phiên bản. Vẫn có thể dùng Codex viết notebook và người dùng bấm các nút Kaggle.
5. Đăng nhập, OTP/CAPTCHA và xác minh tài khoản do người dùng xử lý. Trình duyệt tích hợp hiện không tự động upload file; khi cần người dùng chọn file. Extension có thể hỗ trợ upload tùy phiên/quyền thực tế. Không hứa chạy không cần can thiệp.

Câu giao việc đề xuất:

> Dùng Browser điều khiển notebook Kaggle của tôi. Trước hết xác nhận đúng tài khoản và notebook, đọc trạng thái Input/Session options. Dùng mã nguồn tại project để chuẩn bị notebook kiểm tra môi trường, chưa huấn luyện, chưa upload dữ liệu cuộc thi. Sau khi tôi gắn các input đã được phép và chọn file nếu cần, hãy chạy preflight, báo kết quả, cấu hình GPU và smoke test. Chỉ tiếp tục huấn luyện khi preflight/smoke đạt và cấu hình lượt chạy đã được tôi chấp thuận. Lưu phiên bản và tải outputs về máy, không công khai hay nộp bài.

Agent có thể hỗ trợ mở notebook, nhập/sửa cell, chọn GPU, chạy, đọc trạng thái, kiểm tra lỗi, lưu version và tải kết quả. GPU do Kaggle cấp; ứng dụng Codex không biến GPU máy cá nhân thành GPU Kaggle. Phiên GPU có hạn mức/tạm ngắt, cần checkpoint và Save Version.

## 6. Nhập notebook mới vào Kaggle

1. Mở Kaggle **Code → New Notebook**, hoặc tạo bản mới từ notebook nháp trống. Giữ notebook ở chế độ Private.
2. Menu **File → Import Notebook** (tên có thể khác theo giao diện); chọn `notebooks/ZAIC_Kaggle_GUI.ipynb` từ máy. Không nhập notebook Colab lịch sử rồi Run All.
3. Dùng **Add Input/Upload** để gắn gói mã ZIP nguồn. Gói mã chỉ chứa code/docs. Không gắn nhầm báo cáo cũ có log chi tiết hoặc thư mục dữ liệu cuộc thi.
4. Điền đường dẫn ZIP mã ở cell cấu hình; chạy ACTION='preflight'. Chỉ giải nén code, kiểm checksum và 99 tests giả lập, không cần GPU/model/SFT.
5. Kiểm tra quyền sử dụng dữ liệu với Kaggle theo thể lệ trước khi tạo/gắn input SFT/raw. Nếu được phép, input chứa SFT và weights phải Private; không đưa credentials/answers public/private vào input.
6. Gắn base model dưới dạng input đọc được, đủ config/tokenizer/shard index. Gắn `sft_train.jsonl` và `manifest.json` từ `datamate_v1/qwen35_9b/data` cho lượt train tái lập.
7. Mở **Session options/Settings → Accelerator → GPU**; chọn loại thực sự được tài khoản cấp. Mã hiện tại dùng GPU số 0. T4×2 không tự thành bộ nhớ 32 GB chung.
8. Điền BASE_INPUT, chọn ACTION='smoke', đặt GPU_RUN_CONFIRMED=True. Chạy và kiểm báo cáo, không có dữ liệu cuộc thi trong smoke.
9. Sau smoke đạt, để huấn luyện mới: ACTION='train', đúng SFT_INPUT, COMPETITION_CLOUD_USE_ALLOWED=True, TRAIN_FRESH_ADAPTER=True. Mỗi lần bấm chạy tạo experiment mới; không tiếp tục adapter cũ.
10. Nếu thiếu/incompatible Transformers/PEFT/bitsandbytes, đưa **báo cáo phiên bản và lỗi đã lọc** cho Codex. Agent sửa cell môi trường phù hợp Kaggle. Không cài đè torch/cu130 theo Colab một cách máy móc. Notebook bàn giao không tự cài thư viện hay gọi Internet.
11. Theo dõi finite gradients, số hàng, bước và file training_complete.json. Khi đạt, **Save Version** theo tùy chọn lưu output của phiên hiện tại. Cẩn thận **Save & Run All** có thể chạy lại từ đầu và tạo experiment mới; đọc mô tả hộp thoại trước khi chọn.
12. Sau version hoàn tất, tải output bằng giao diện, lưu vào PRIVATE_DATA/runs/<run_id>, kiểm checksum adapter mới và metadata. Không chỉ tải notebook rồi tưởng checkpoint đã được lưu.

Base giữ ở input; output chỉ gồm adapter/checkpoint/log/metadata. Không copy lại base 19 GB vào /kaggle/working rồi xuất ZIP. Kiểm tra giới hạn lưu output thực tế ở tài khoản. Sau khi có adapter mới, vẫn cần notebook đánh giá TRAIN và kiểm tra batch parity; notebook này chưa tự đánh giá/triển khai/nộp bài.

## 7. Nếu chỉ muốn chạy thành quả đã huấn luyện

Không cần ACTION='train'. Tải đúng base/adapter cũ, kiểm hash, yêu cầu Codex tạo một notebook suy luận riêng từ `frozen_v21/tools/infer_batched.py` với đường dẫn mới và output mới. Đo parity trên TRAIN trước public. Không nạp trực tiếp journal/fingerprint cũ vào config mới để ép resume.

## 8. Nguồn hướng dẫn giao diện

Đã tra cứu ngày 07/11/2026:

- https://learn.chatgpt.com/docs/projects — local project/folder và phạm vi truy cập.
- https://learn.chatgpt.com/docs/browser — browser tích hợp, extension, quyền và giới hạn upload.
- https://www.kaggle.com/docs/notebooks — notebook/input/output/session/version; trang có thể cần JavaScript để hiển thị đầy đủ.
- Giao diện notebook Kaggle đã trực tiếp quan sát: File, Run All, Save Version, Add Input, Upload, Session options; bản nháp chưa chạy, chưa có input.

Tên nút và tính năng tùy phiên bản/tài khoản. Không hứa hạn mức GPU cố định hay hỗ trợ điều khiển trình duyệt nếu chưa kiểm tra trong phiên thực tế.
