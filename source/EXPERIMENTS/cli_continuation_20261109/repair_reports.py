"""Repair report encoding only; frozen model/source/run artifacts stay untouched."""
from pathlib import Path
import json,hashlib,datetime
root=Path('D:/ZAIC');exp=root/'EXPERIMENTS/time_learning_v3_20261109'
dest=root/'PRIVATE_DATA/zaic/result_811/TIME_FOCUSED_V3_20261109'
archive=Path(__file__).parent
notes='''

## Bổ sung vận hành sau khi nối lại

Trong lúc chat bị ngắt, phiên đánh giá đã tự được Kaggle cấp GPU và hoàn tất (436,6 giây). Agent không thể thực hiện mốc hủy hàng đợi khi đã ngắt. Không có optimizer bổ sung. Phiên CPU xác nhận mặc định bị xếp hàng đã được dừng trước CreateSession; Accelerator=None được lưu và kiểm bằng giao diện.

Đã phát hiện thêm phiên GPU tương tác public cũ đang nhàn rỗi, dừng ngay và đưa về None. Khoảng 70,54 giây của phiên này được cộng bảo thủ vào tổng 35,74 phút. Không có cell public được chạy trong thao tác này. Ảnh GPU_OFF_VERIFIED.png ghi nhận Draft Session off, Accelerator=None và cảnh báo sự cố kết nối notebook của Kaggle.

12 unresolved gồm 8 lỗi model và 4 lỗi executor. Trước vòng train tiếp theo, ưu tiên sửa lỗi executor, mở rộng correction được xác minh và tách process train/eval. Không tăng số bước hoặc dùng gold public để ép đạt cổng.

Chỉ đạo mới: ưu tiên Kaggle CLI/SDK; chỉ dùng web khi CLI/API thiếu chức năng cần thiết. Lần kiểm tra tiếp nối được lưu trong CLI_RECHECK_20261109.json. SDK có cancel_kernel_session yêu cầu session ID đã xác minh; chưa kiểm chứng gọi dừng thực tế bằng API trong lượt này.
'''
for name,p in [('project_report',exp/'RESULT_REPORT_VI.md'),('private_report',dest/'RESULT_REPORT_VI.md')]:
 s=p.read_text(encoding='utf-8');(archive/(name+'_before_encoding_fix.md')).write_text(s,encoding='utf-8')
 anchor='Bằng chứng: LOCAL_VERIFICATION.json'
 assert anchor in s
 end=s.index('\n',s.index(anchor));p.write_text(s[:end]+notes,encoding='utf-8')
progress='''# Vòng học thời gian v3 đã kết thúc

Đã train 9 bước warm-start E7. CPU: 99 baseline +131 tests mới =230 đạt. Dataset 9 correction +9 replay; thu dự đoán thật 124/124 câu TRAIN trong pool đã chọn.

Đánh giá GPU: sửa 2/9 lỗi mới, 0/7 lỗi cũ; giữ replay đúng 9/9 và 7/7. Cổng lỗi cũ chưa đạt: không calibration, không promote BEST, không chạy public mới.

10 bộ checkpoint/state đã đồng bộ máy–Kaggle, khớp checksum. LATEST_CHECKPOINT/LATEST_SYNCED cập nhật; BEST_MODEL giữ nguyên. GPU đã tắt ở lần kiểm cuối; tổng 35,74/120 phút bao gồm phiên public nhàn rỗi đã được tắt.

Đọc RESULT_REPORT_VI.md, FINAL_STATUS.json và CAPABILITY_STATUS.json. Full resume chưa triển khai. Runner một process lỗi cleanup sau train; đánh giá đã được phục hồi thành công bằng process riêng.

Ưu tiên CLI/SDK, chỉ dùng web khi thiếu chức năng. Đây là vòng đã đóng, không có quyền mở optimizer vòng mới từ báo cáo này.
'''
(exp/'PROGRESS_VI.md').write_text(progress,encoding='utf-8')
handoff=json.loads((exp/'ACTIVE_HANDOFF.json').read_text(encoding='utf-8'))
current={'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':'closed_and_verified','run_id':'20261109T125033322737Z_b786bc50','training_steps':9,'evaluation_version':367314230,'preservation_version':367316919,'no_active_training_job':True,'no_new_optimizer_authorization':True,'current_authoritative_evidence':['FINAL_STATUS.json','RESULT_REPORT_VI.md','CLI_RECHECK_20261109.json'],'historical_handoff_warning':'ACTIVE_HANDOFF retains historical intermediate fields; use this current state and final evidence, not old active/version/timeout fields','cli_first':True,'web_condition':'Only when official CLI/SDK lacks the required capability or cannot identify the session safely'}
(exp/'CURRENT_STATE.json').write_text(json.dumps(current,indent=2),encoding='utf-8')
receipt={'observed_at':current['observed_at'],'scope':'CLI preference and UTF-8 reports only; no model/source/run mutations','files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'AGENTS.md',exp/'RESULT_REPORT_VI.md',dest/'RESULT_REPORT_VI.md',exp/'PROGRESS_VI.md',exp/'CURRENT_STATE.json']}}
(archive/'MAINTENANCE_RECEIPT.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
for p in [exp/'RESULT_REPORT_VI.md',dest/'RESULT_REPORT_VI.md',exp/'PROGRESS_VI.md']:
 assert '?' not in p.read_text(encoding='utf-8'),'Unexpected corrupted Vietnamese text'
print('CLI_POLICY_AND_UTF8_REPORTS_UPDATED; NO_MODEL_MUTATION')
