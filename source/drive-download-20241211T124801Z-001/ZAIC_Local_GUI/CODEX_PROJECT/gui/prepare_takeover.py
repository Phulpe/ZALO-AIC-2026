"""Internal audit/build. Reads private files only for byte counts and hashes."""
from pathlib import Path
from collections import Counter
import ast
import base64
import hashlib
import importlib.machinery
import importlib.util
import io
import json
import platform
import shutil
import sys
import zipfile
from run_sync import digest, write_json, write_dashboard
from kaggle_runner import environment_report, verify_frozen

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
PRIVATE = WORKSPACE / 'PRIVATE_DATA'


def audit():
    files = []
    for p in sorted(PRIVATE.rglob('*')):
        if p.is_file() and not p.is_symlink() and 'runs' not in p.relative_to(PRIVATE).parts:
            files.append({'path': p.relative_to(PRIVATE).as_posix(), 'bytes': p.stat().st_size,
                          'sha256': digest(p)})
    # Full filename inventory stays private; AI consumes only categories below.
    write_json(PRIVATE / 'LOCAL_FILE_INVENTORY_20261107.json', {'files': files})
    loader = importlib.machinery.SourceFileLoader('local_check', str(PROJECT / 'gui/local_check.pyw'))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    checker = importlib.util.module_from_spec(spec); loader.exec_module(checker)
    original = checker.inspect_original(PRIVATE / 'zaic', True)
    finals = list(PRIVATE.glob('*/public_test_v21_verified_20261107T090319Z'))
    verified = checker.inspect_final(finals[0]) if len(finals) == 1 else {'baseline_checks_passed': False}
    categories = []
    for label, prefix in [('TRAIN', 'zaic/train/'), ('public', 'zaic/public_test/'),
        ('SFT/manifest', 'zaic/datamate_v1/qwen35_9b/data/'),
        ('base/tokenizer', 'zaic/datamate_v1/qwen35_9b/assets/base/'),
        ('adapter 9B', 'zaic/datamate_v1/qwen35_9b/candidate/final/'),
        ('checkpoints', 'zaic/datamate_v1/qwen35_9b/candidate/checkpoints/'),
        ('repair_v21', 'ZAIC_9B_Repair_v21_20261107/'),
        ('resume_own', 'ZAIC_9B_Resume_Own_20261107/')]:
        subset = [x for x in files if x['path'].startswith(prefix)]
        categories.append({'category': label, 'files': len(subset), 'bytes': sum(x['bytes'] for x in subset),
            'status': 'chưa xác minh đầy đủ' if subset else 'còn thiếu'})
    summary = {'file_count': len(files), 'bytes': sum(x['bytes'] for x in files),
        'all_present_files_sha256_recorded': True, 'source_inventory_complete': False,
        'full_transfer_verified': False, 'categories': categories,
        'baseline': original, 'verified_final': verified,
        'safetensors_files': sum(x['path'].endswith('.safetensors') for x in files),
        'original_notebook_present': (PRIVATE / 'zaic/ZAIC_QLoRA_intent_v0.ipynb').is_file(),
        'download_archives': sum(p.is_file() for p in (WORKSPACE / 'DOWNLOADS_ARCHIVE').rglob('*'))}
    write_json(PROJECT / 'reports/ASSET_INVENTORY_20261107.json', summary)
    lines = ['# Kiểm kê tài sản trên máy', '',
        f"Có {summary['file_count']} tệp / {summary['bytes']:,} bytes. SHA256 đã tính cho từng tệp hiện có.", '',
        'Hash mới là kiểm kê cục bộ; chỉ hash có đối chứng nguồn mới xác nhận truyền nguyên vẹn.', '',
        '| Nhóm | Tệp hiện có | Bytes | Trạng thái |', '|---|---:|---:|---|']
    for row in categories:
        lines.append(f"| {row['category']} | {row['files']} | {row['bytes']} | {row['status']} |")
    lines += ['', 'Bộ verified: ' + ('đạt kiểm tra 9 tên tệp, 1000 ID duy nhất và hash submission' if verified['baseline_checks_passed'] else 'chưa đạt'),
        '', 'Thiếu: TRAIN/public questions + rulebook, TRAIN answers; base config/tokenizer_config/index và toàn bộ shards; adapter 9B + config; frozen SFT + manifest; notebook gốc.',
        '', 'Checkpoint chỉ hiện diện một phần: chưa có bằng chứng đủ optimizer/scheduler/RNG/adapter ở cùng step.',
        'Repair/resume và lịch sử ngoài zaic: cần kiểm kê đệ quy từ nguồn Drive để đối chiếu đầy đủ.',
        'Không tải lại bộ verified nếu đối chứng nguồn còn khớp. Không coi .metadata/.lock là file đích.']
    (PROJECT / 'reports/ASSET_INVENTORY_VI.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    env = {'os': platform.platform(), 'python_executable': sys.executable, 'environment': environment_report(),
           'utf8_mode': sys.flags.utf8_mode, 'source_files_verified': verify_frozen(PROJECT / 'frozen_v21'),
           'disk': {str(p): dict(zip(['total','used','free'],shutil.disk_usage(p))) for p in ['C:/','D:/']}}
    write_json(PROJECT / 'reports/LOCAL_ENVIRONMENT_READY_20261107.json', env)
    return summary


def build_notebook():
    out = PROJECT / 'notebooks/ZAIC_Kaggle_Versioned_20261107.ipynb'
    bundle = io.BytesIO()
    names = [PROJECT / 'frozen_v21' / name for name in json.loads((PROJECT / 'frozen_v21/SOURCE_SHA256.json').read_text())]
    names += [PROJECT / 'frozen_v21' / name for name in ['SOURCE_SHA256.json','model_lock.json']]
    names += [PROJECT / 'gui' / name for name in ['kaggle_runner.py','local_check.pyw','run_sync.py','versioned_runner.py']]
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(names)):
            z.writestr(p.relative_to(PROJECT).as_posix(), p.read_bytes())
    raw = bundle.getvalue()
    sha = hashlib.sha256(raw).hexdigest()
    payload = base64.b64encode(raw).decode()
    code = '''# Code + synthetic tests only. No competition data or weights embedded.
from pathlib import Path
from datetime import datetime, timezone
import base64, hashlib, io, json, stat, sys, zipfile
PAYLOAD = "__PAYLOAD__"
EXPECTED_SHA256 = "__SHA__"
raw = base64.b64decode(PAYLOAD)
assert hashlib.sha256(raw).hexdigest() == EXPECTED_SHA256, 'Code archive checksum mismatch'
source = Path('/kaggle/temp/zaic_code') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
source.mkdir(parents=True, exist_ok=False)
with zipfile.ZipFile(io.BytesIO(raw)) as z:
    assert sum(i.file_size for i in z.infolist()) < 25 * 1024 * 1024
    for i in z.infolist():
        assert (source / i.filename).resolve().is_relative_to(source.resolve())
        assert not stat.S_ISLNK(i.external_attr >> 16)
    z.extractall(source)
sys.path.insert(0, str(source / 'gui'))
from versioned_runner import run
RESULT = run(CONFIG, source)
'''.replace('__PAYLOAD__', payload).replace('__SHA__', sha)
    config = {'ACTION':'preflight', 'BASE_INPUT':'/kaggle/input/REVIEW_BASE/base',
        'SFT_INPUT':'/kaggle/input/REVIEW_APPROVED_SFT/data', 'GPU_RUN_CONFIRMED':False,
        'COMPETITION_CLOUD_USE_ALLOWED':False, 'CLOUD_PERMISSION_EVIDENCE':'',
        'TRAIN_FRESH_ADAPTER':False, 'BASE_FILE_HASHES':{}}
    def cell(kind, source, identifier):
        c = {'cell_type':kind,'id':identifier,'metadata':{},'source':source.splitlines(keepends=True)}
        if kind == 'code': c.update(execution_count=None,outputs=[])
        return c
    notebook = {'nbformat':4,'nbformat_minor':5,
        'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},
                    'language_info':{'name':'python'}},
        'cells':[cell('markdown','# ZAIC DataMate — preflight và đối chiếu phiên bản\n\nNotebook mới chỉ chứa code và kiểm thử giả lập. Mặc định CPU preflight; không tải model, không huấn luyện, không nộp bài.\n\nGPU smoke 9B cần đủ base và hash nguồn. Train cần bằng chứng được phép đưa dữ liệu/SFT lên Kaggle. Private không thay bằng chứng.\n','zaic-info'),
            cell('code','CONFIG = '+repr(config)+'\n','zaic-config'),cell('code',code,'zaic-run'),
            cell('markdown','## Lưu và đối chiếu\nSau completed, lưu phiên bản với output hiện tại (tránh chạy lại train). Đợi version hoàn tất. Tải ZIP của đúng run; đối chiếu ARTIFACT_MANIFEST SHA256 của bản đã lưu với máy. Chỉ sau đó cập nhật LATEST_SYNCED. BEST_MODEL chỉ chọn bằng đánh giá TRAIN.\n\nBase và SFT giữ ở input, scratch ở /kaggle/temp; output chỉ có checkpoint/adapter và báo cáo. Runner này chưa hỗ trợ resume: giữ checkpoint đầy đủ để triển khai resume riêng sau khi kiểm fingerprint.\n','zaic-save')]}
    write_json(out, notebook)
    for c in notebook['cells']:
        if c['cell_type'] == 'code': ast.parse(''.join(c['source']))
    write_json(PROJECT / 'reports/NOTEBOOK_BUILD_20261107.json',
        {'path':str(out),'sha256':digest(out),'embedded_code_sha256':sha,'embedded_archive_bytes':len(raw),
         'default_action':'preflight','raw_data_included':False,'weights_included':False})
    return out


if __name__ == '__main__':
    summary = audit()
    path = build_notebook()
    runs = PRIVATE / 'runs'
    if not (runs / 'SYNC_STATE.json').exists():
        write_json(runs / 'SYNC_STATE.json', {'LATEST_SYNCED':None,'BEST_MODEL':None,'runs':[]})
        write_dashboard(runs, {'LATEST_SYNCED':None,'BEST_MODEL':None,'runs':[]})
    print(json.dumps({'files':summary['file_count'],'bytes':summary['bytes'],
        'verified_final':summary['verified_final'],'notebook':str(path)}, indent=2))
