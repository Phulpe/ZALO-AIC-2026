from pathlib import Path
from datetime import datetime
import hashlib, json, shutil, re

root=Path('D:/ZAIC').resolve()
project=root/'drive-download-20241211T124801Z-001/ZAIC_Local_GUI/CODEX_PROJECT'
paths=[root/'AGENTS.md',root/'agent.md',project/'AGENTS.md',project/'agent.md',root/'training.md']
archive=root/'GUIDE_ARCHIVE'/datetime.now().strftime('%Y%m%dT%H%M%S%f')
assert archive.resolve().is_relative_to(root)
archive.mkdir(parents=True,exist_ok=False)
records=[]
for i,p in enumerate(paths):
    assert p.resolve().is_relative_to(root) and p.is_file()
    target=archive/f'{i}_{p.name}.bak'
    shutil.copy2(p,target)
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    assert hashlib.sha256(target.read_bytes()).hexdigest()==digest
    records.append({'original':str(p),'backup':str(target),'sha256':digest})

body=(root/'agent.md').read_text(encoding='utf-8')
body=body.split('## 9. Quy tắc bổ sung')[0].rstrip()
body=body.replace('# agent.md — Nguyên tắc cho agent ZAIC DataMate','# AGENTS.md — Quy tắc chuẩn cho agent ZAIC DataMate',1)
body=body.replace('1. AGENTS.md hiện có của project và các chỉ dẫn hợp lệ liên quan.\n2. File này và training.md.', '1. File D:\\ZAIC\\AGENTS.md này và các chỉ dẫn cấp cao hơn áp dụng.\n2. D:\\ZAIC\\training.md trước mọi thao tác huấn luyện/đánh giá; đây là đặc tả cần triển khai, không phải bằng chứng đã triển khai.')
body=re.sub(r'`agent.md` là tên người dùng yêu cầu;.*?Nếu chưa có, tạo tệp hướng dẫn ngắn sau khi kiểm tra project\.', 'Nguồn quy tắc duy nhất là D:\\\\ZAIC\\\\AGENTS.md. File AGENTS.md trong project con chỉ dẫn về file này, không duy trì bộ quy tắc độc lập. training.md được giữ riêng làm đặc tả huấn luyện. Bản hướng dẫn cũ được lưu ngoài luồng vận hành trong GUIDE_ARCHIVE.',body,flags=re.S)
body=body.replace('- Không bỏ guard, ép trả lời hoặc thay abstain bằng unanswerable chỉ để tăng coverage.', '- Không bỏ guard dùng chung hoặc giả đáp án để tăng coverage. Riêng LEARNING_FORCE_ATTEMPT: thử mọi câu của learning pool TRAIN bằng đường chạy riêng, ghi lỗi/unresolved và retry hữu hạn theo training.md. Chế độ đánh giá/nộp bài giữ chính sách đã khóa; không lấy gold để sửa dự đoán.')
body=body.replace('Trước thay đổi đầu tiên, lập hoặc cập nhật inventory theo file:', 'Trước thay đổi đầu tiên đối với tài sản model/dữ liệu/run, lập hoặc cập nhật inventory theo file:')
body=body.replace('Cập nhật training.md bằng quyết định có bằng chứng;', 'Cập nhật D:\\ZAIC\\training.md bằng quyết định có bằng chứng;')
gpu=(root/'AGENTS.md').read_text(encoding='utf-8').split('## GPU Kaggle')[1].split('Hướng dẫn dự án chi tiết')[0]
gpu='## GPU Kaggle'+gpu
gpu=gpu.replace('bảo toàn checkpoint/output cần thiết, dừng phiên GPU ngay,', 'ghi checkpoint/output cần thiết vào vùng lưu bền vững rồi dừng phiên GPU ngay (không giữ GPU để đợi tải về máy),')
body=body.replace('## 1. Mục tiêu và cách làm việc',gpu.strip()+'\n\n## 1. Mục tiêu và cách làm việc',1)
body+='''

## 9. Quy tắc kỹ thuật bổ sung được giữ từ AGENTS.md project

- Đọc README_FIRST_VI.md, docs/HANDOFF_CODEX_VI.md, docs/DOWNLOAD_AND_GUI_VI.md, docs/TRANSFER_MANIFEST.json và reports/VALIDATION.json trong project khi tiếp quản. Gói bàn giao chỉ chứa mã; không thay thế kiểm kê dữ liệu/trọng số thật.
- frozen_v21/app và frozen_v21/tools gồm 27 file nguồn bất biến; xác minh SOURCE_SHA256.json. Không sửa tại chỗ. Các recovery script Colab ở root là lịch sử, không phải entrypoint Kaggle; không Run All notebook Colab lịch sử.
- Baseline rank 8, không phải 16. Final submission lịch sử SHA256 e0815c96524e9db0badf987e45e13227eec8e57c7f3195f5aa93a580a0508780. 235 answered không có nghĩa 235 đúng hay PA=1.
- Baseline có 75 app + 24 tools tests giả lập; gui/kaggle_runner.py chạy unittest không cần pytest. Test hồi quy phải kiểm ngữ nghĩa nghiệp vụ. Đổi runtime/dependency phải kiểm batch parity.
- /kaggle/input chỉ đọc; /kaggle/working có thể ghi. Để base ở input; không sao chép/xuất lại 19GB vào outputs. Ghi checkpoint/journal vào local disk runtime, không append trực tiếp lên Drive/FUSE dùng chung.
- Khi đổi fingerprint, tạo thư mục output mới; không sửa fingerprint để ép resume. Bản ghi trùng ID phải giống hệt.
- Không ép cài stack torch+cu130 của Colab lên Kaggle. Loader baseline chỉ dùng cuda:0; T4 x2 không tự gộp VRAM.

## 10. Đường dẫn và bằng chứng trạng thái

- Project mã đã xác định: D:\\ZAIC\\drive-download-20241211T124801Z-001\\ZAIC_Local_GUI\\CODEX_PROJECT. Đường dẫn D:\\ZAIC\\CODEX_PROJECT trong tài liệu cũ chỉ là gợi ý, không coi là thư mục thật.
- Dữ liệu/model riêng: D:\\ZAIC\\PRIVATE_DATA. ZIP gốc: D:\\ZAIC\\DOWNLOADS_ARCHIVE. Không đính kèm toàn bộ kho riêng vào ngữ cảnh AI.
- Tài liệu huấn luyện chuẩn: D:\\ZAIC\\training.md. Hướng dẫn agent chuẩn: D:\\ZAIC\\AGENTS.md. Không tiếp tục cập nhật agent.md đã lưu trữ.
- Mục 3 là lịch sử tham chiếu, không phải kiểm tra mới. Đối chiếu TRAINING_HISTORY_AUDIT.md, RUN_REGISTRY.csv, WEIGHTS_INVENTORY_AFTER_RENAME.json và WEIGHTS_CHECK_REPORT.md trước quyết định tác động tài sản.
- Báo cáo upload sau kiểm toán: D:\\ZAIC\\KAGGLE_UPLOAD_REPORT_VI.md; biên nhận chi tiết: D:\\ZAIC\\KAGGLE_STAGING\\20261107T191003Z\\KAGGLE_UPLOAD_STATUS.json. Báo cáo ghi bốn input Private và notebook version 2 đã xác minh; đọc báo cáo không chứng minh trạng thái phiên hiện tại.
- Notebook mới mặc định preflight. Tại thời điểm báo cáo upload, wrapper chỉ hỗ trợ fresh adapter, chưa triển khai warm-start/full resume hoặc vòng học correction/replay. training.md yêu cầu các tính năng này nhưng không chứng minh đã có code hay đã chạy smoke.
- LATEST_SYNCED là lượt đã tải về và khớp manifest, không mặc định model tốt nhất. BEST_MODEL cần đánh giá; LATEST_CHECKPOINT là mốc checkpoint mới nhất có bằng chứng, tách riêng hai trạng thái trên. Không tự mở rộng schema tracker khi chưa thực hiện/thẩm định thay đổi.

## 11. Điểm cần làm rõ trước tác vụ phụ thuộc

- Quyền dùng cloud: tài liệu cũ ghi hạn chế bên thứ ba; người dùng sau đó xác nhận ban tổ chức cho phép nền tảng thứ ba để huấn luyện. Giữ CLOUD_PERMISSION_RECORD.json và giới hạn dữ liệu; chưa có URL ngoại lệ chính thức để kiểm chứng độc lập. Không coi Private là bằng chứng quyền, cũng không phủ nhận xác nhận đã có của người dùng hay hỏi lại cùng quyền chỉ do đọc tài liệu cũ. Nếu tác vụ mở rộng phạm vi ngoài xác nhận hiện có, hỏi đúng thông tin còn thiếu.
- Trước lượt train tiếp theo, chốt init_mode và ngân sách: warm-start adapter với optimizer/schedule mới khác full resume checkpoint-48 với trạng thái/data/order/config tương thích. Không tự dùng luồng fresh chỉ vì wrapper hiện có hỗ trợ. Không coi yêu cầu gộp tài liệu là quyền chạy train.
- Thống kê run/checkpoint/adapter trong báo cáo lịch sử có phạm vi và đơn vị khác nhau (file, thư mục, checkpoint chính). Không dùng các số tổng đó làm tiêu chuẩn đầy đủ thay kiểm kê từng đường dẫn/hash; nêu chênh lệch nếu kiểm tra sau phát hiện.
'''
(root/'AGENTS.md').write_text(body+'\n',encoding='utf-8')
(project/'AGENTS.md').write_text('# Điểm đọc hướng dẫn ZAIC\n\nToàn bộ quy tắc agent nằm duy nhất tại `D:\\ZAIC\\AGENTS.md`. Đọc file đó trước mọi công việc ở project này; đọc `D:\\ZAIC\\training.md` trước huấn luyện/đánh giá. File này chỉ là dẫn chiếu, không có bộ quy tắc riêng.\n',encoding='utf-8')
training=(root/'training.md').read_text(encoding='utf-8')
training=training.replace('Đọc agent.md và báo cáo kiểm toán thực tế', 'Đọc D:\\ZAIC\\AGENTS.md và báo cáo kiểm toán thực tế')
training=training.replace('Đọc agent.md/audit/registry', 'Đọc D:\\ZAIC\\AGENTS.md/audit/registry')
(root/'training.md').write_text(training,encoding='utf-8')
for i,p in [(1,root/'agent.md'),(3,project/'agent.md')]:
    target=archive/f'{i}_agent.md.retired'
    assert p.resolve().is_relative_to(root) and target.resolve().is_relative_to(root) and not target.exists()
    p.rename(target)
(archive/'MANIFEST.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
assert not (root/'agent.md').exists() and not (project/'agent.md').exists()
assert '## 11.' in (root/'AGENTS.md').read_text(encoding='utf-8')
print(json.dumps({'canonical':str(root/'AGENTS.md'),'archive':str(archive),'rules_in_child':False,'training_changes':'references only'},ensure_ascii=False,indent=2))
