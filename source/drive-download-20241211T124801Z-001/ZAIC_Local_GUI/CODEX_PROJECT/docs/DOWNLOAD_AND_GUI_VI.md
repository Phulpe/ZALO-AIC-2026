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
