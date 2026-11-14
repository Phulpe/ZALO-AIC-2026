"""Authorized shard rename and local-only staging. No network or model loading."""
from pathlib import Path
from datetime import datetime, timezone
import ast, base64, hashlib, io, json, os, shutil, sys, uuid, zipfile

ROOT=Path('D:/ZAIC').resolve()
PROJECT=ROOT/'drive-download-20241211T124801Z-001/ZAIC_Local_GUI/CODEX_PROJECT'
PRIVATE=ROOT/'PRIVATE_DATA'
BASE9=PRIVATE/'zaic/datamate_v1/qwen35_9b/assets/base'
BASE4=PRIVATE/'zaic/datamate_v1/offline_export/assets/base'
STAMP=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
STAGE=ROOT/'KAGGLE_STAGING'/STAMP
STAGE.mkdir(parents=True,exist_ok=False)
REPORT=STAGE/'PREPARATION_REPORT.json'
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
inventory=json.loads((ROOT/'WEIGHTS_INVENTORY.json').read_text(encoding='utf-8'))
records={f['path']:f for f in inventory['files']}
moves=[]
for base in inventory['base_sets']:
    for shard in base['shards']:
        for match in shard['matches']:
            src=Path(match['path']);dst=Path(base['directory'])/shard['index_filename']
            if src==dst:continue
            src=src.resolve();dst=dst.resolve()
            if not src.is_relative_to(ROOT) or not dst.is_relative_to(ROOT):raise ValueError('Path escapes authorized workspace')
            if not src.is_file() or src.stat().st_size!=match['bytes'] or dst.exists():raise ValueError('Source changed or destination already exists')
            if not match['trusted_handoff_manifest_match']:raise ValueError('Source hash evidence missing')
            moves.append({'source':str(src),'destination':str(dst),'bytes':match['bytes'],'expected_sha256':match['sha256'],'state':'planned'})
report={'created_utc':datetime.now(timezone.utc).isoformat(),'state':'renaming','stage':str(STAGE),
        'upload_performed':False,'training_started':False,'moves':moves,'base4_metadata_issue_preserved':True}
write(REPORT,report)
for move in moves:
    src=Path(move['source']);dst=Path(move['destination'])
    if dst.exists():raise FileExistsError(dst)
    os.rename(src,dst) # Windows fails if target exists; no overwrite or merge.
    move['state']='renamed_pending_hash';write(REPORT,report)
    print('RENAMED',src.name,'->',str(dst),flush=True)
report['state']='verifying_after_rename';write(REPORT,report)
for move in moves:
    dst=Path(move['destination']);before=dst.stat();actual=sha(dst);after=dst.stat()
    move['actual_sha256']=actual
    move['verified']=actual==move['expected_sha256'] and after.st_size==move['bytes'] and before.st_mtime_ns==after.st_mtime_ns
    move['state']='verified' if move['verified'] else 'checksum_conflict_preserved'
    write(REPORT,report)
    print('SHA256',dst.name,'PASS' if move['verified'] else 'FAIL',flush=True)
    if not move['verified']:raise ValueError('Renamed file checksum mismatch; staging blocked')

# Base input is immutable and shared across runs. Stage only the 15 source assets,
# without cache locks, training data, adapters, checkpoints or old notebooks.
manifest_path=PRIVATE/'zaic/datamate_v1/qwen35_9b/training_code/base_assets_manifest.json'
manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
base_stage=STAGE/'01_BASE_QWEN35_9B/base';base_stage.mkdir(parents=True)
hashes={}
for name,expected in manifest.items():
    src=(BASE9/name).resolve()
    if not src.is_relative_to(BASE9.resolve()) or not src.is_file():raise ValueError('Base asset missing')
    moved=next((m for m in moves if m['destination']==str(src)),None)
    actual=moved['actual_sha256'] if moved else sha(src)
    if actual!=expected['sha256'] or src.stat().st_size!=expected['bytes']:raise ValueError('Base source manifest mismatch')
    hashes[name]=actual
    os.link(src,base_stage/name)
write(STAGE/'01_BASE_QWEN35_9B/BASE_UPLOAD_MANIFEST.json',{'model':'Qwen/Qwen3.5-9B',
    'revision':'c202236235762e1c871ad0ccb60c8ee5ba337b9a','files':manifest,
    'staging_uses_hardlinks':True,'independent_backup':False})

# Process SFT privately and print only aggregate checks. No samples in reports.
data=PRIVATE/'zaic/datamate_v1/qwen35_9b/data'
sft=data/'sft_train.jsonl';sft_sha=sha(sft)
expected_sft='5057658ac10c19579c675c9f926cc13f1e17641b88eb312318e48be75e6bb170'
if sft_sha!=expected_sft:raise ValueError('SFT baseline mismatch')
dm=json.loads((data/'manifest.json').read_text(encoding='utf-8'))
if dm['training_sha256']!=sft_sha:raise ValueError('SFT manifest mismatch')
split=dm['split'];train=set(split['train']);cal=set(split['calibration']);audit=set(split['audit'])
if train&cal or train&audit or cal&audit:raise ValueError('Business split overlaps')
rows=0;invalid=0
with sft.open(encoding='utf-8') as f:
    for line in f:
        if not line.strip():continue
        row=json.loads(line);rows+=1
        if row.get('business_id') not in train or not isinstance(row.get('prompt'),str) or not isinstance(row.get('response'),str):invalid+=1
if rows!=768 or invalid:raise ValueError('SFT row/split/schema mismatch')
data_stage=STAGE/'02_APPROVED_TRAIN_SFT/data';data_stage.mkdir(parents=True)
for name in ['sft_train.jsonl','manifest.json','split.json']:
    shutil.copy2(data/name,data_stage/name)
permission={'status':'user_attested','independently_verified_official_exception':False,
 'user_statement':'Ban tổ chức cho phép tất cả các thí sinh thực hiện sử dụng nền tảng thứ 3 để tiến hành huấn luyện mô hình AI phục vụ cho mục đích thi',
 'scope':'training for this competition; private input; no public/private test data or predictions',
 'source':'User confirmation in this chat; official source URL not provided'}
write(STAGE/'CLOUD_PERMISSION_RECORD.json',permission)
write(STAGE/'02_APPROVED_TRAIN_SFT/SFT_UPLOAD_MANIFEST.json',{'rows':rows,'invalid_rows':invalid,
    'sha256':sft_sha,'source_manifest_sha256':sha(data/'manifest.json'),'permission':permission,
    'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in data_stage.iterdir()}})

# Curated code archive. No docs, private reports, raw data or historical notebooks.
sys.path.insert(0,str(PROJECT/'gui'))
from kaggle_runner import verify_frozen
if verify_frozen(PROJECT/'frozen_v21')!=27:raise ValueError('Frozen source mismatch')
files=[PROJECT/'frozen_v21'/n for n in json.loads((PROJECT/'frozen_v21/SOURCE_SHA256.json').read_text())]
files += [PROJECT/'frozen_v21'/n for n in ['SOURCE_SHA256.json','model_lock.json']]
files += [PROJECT/'gui'/n for n in ['kaggle_runner.py','local_check.pyw','run_sync.py','versioned_runner.py']]
code_dir=STAGE/'03_CODE_AND_NOTEBOOK';code_dir.mkdir()
code_zip=code_dir/'ZAIC_CODE_ONLY.zip'
with zipfile.ZipFile(code_zip,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(set(files)):z.write(p,'CODEX_PROJECT/'+p.relative_to(PROJECT).as_posix())
code_sha=sha(code_zip)
config={'ACTION':'preflight','BASE_INPUT':'','SFT_INPUT':'','GPU_RUN_CONFIRMED':False,
 'COMPETITION_CLOUD_USE_ALLOWED':True,'CLOUD_PERMISSION_EVIDENCE':'User confirmation recorded in local CLOUD_PERMISSION_RECORD.json; independently unverified',
 'TRAIN_FRESH_ADAPTER':False,'BASE_FILE_HASHES':hashes}
config_text='CONFIG = '+repr(config)+'\n'
launch='''from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, stat, sys, zipfile
archives=list(Path('/kaggle/input').rglob('ZAIC_CODE_ONLY.zip'))
assert len(archives)==1, 'Attach exactly one ZAIC code archive as input'
archive=archives[0]
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='__CODE_SHA__', 'Code archive checksum mismatch'
source=Path('/kaggle/temp/zaic_source')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
source.mkdir(parents=True,exist_ok=False)
with zipfile.ZipFile(archive) as z:
    assert sum(i.file_size for i in z.infolist())<25*1024*1024
    for i in z.infolist():
        assert (source/i.filename).resolve().is_relative_to(source.resolve())
        assert not stat.S_ISLNK(i.external_attr>>16)
    z.extractall(source)
source=source/'CODEX_PROJECT'
sys.path.insert(0,str(source/'gui'))
if CONFIG['ACTION']!='preflight':
    bases=list(Path('/kaggle/input').rglob('BASE_UPLOAD_MANIFEST.json'))
    assert len(bases)==1, 'Attach exactly one version of the base dataset'
    CONFIG['BASE_INPUT']=str(bases[0].parent/'base')
if CONFIG['ACTION']=='train':
    datasets=list(Path('/kaggle/input').rglob('SFT_UPLOAD_MANIFEST.json'))
    assert len(datasets)==1, 'Attach exactly one approved training dataset'
    CONFIG['SFT_INPUT']=str(datasets[0].parent/'data')
from versioned_runner import run
RESULT=run(CONFIG,source)
'''.replace('__CODE_SHA__',code_sha)
def cell(kind,text,ident):
    c={'cell_type':kind,'id':ident,'metadata':{},'source':text.splitlines(keepends=True)}
    if kind=='code':c.update(execution_count=None,outputs=[])
    return c
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','language':'python','display_name':'Python 3'}},
 'cells':[cell('markdown','# ZAIC 9B — upload đã kiểm hash, mặc định preflight\n\nGắn ZAIC_CODE_ONLY.zip rồi chạy kiểm tra nguồn + 99 tests CPU. Base và SFT là hai input riêng, Private. Chỉ sau preflight mới xem xét GPU smoke. Không chạy notebook Colab lịch sử.\n','info'),
 cell('code',config_text,'config'),cell('code',launch,'run'),
 cell('markdown','## Sau khi kiểm tra đạt\nSmoke: ACTION=smoke, GPU_RUN_CONFIRMED=True. Train mới: ACTION=train và TRAIN_FRESH_ADAPTER=True sau khi xem xét cấu hình; không resume adapter cũ. Mặc định cả hai cổng này tắt.\n\nLưu version có outputs, tải ZIP run về máy, đối chiếu hash rồi mới LATEST_SYNCED. BEST_MODEL chọn bằng đánh giá TRAIN.\n','save')]}
for c in nb['cells']:
    if c['cell_type']=='code':ast.parse(''.join(c['source']))
write(code_dir/'ZAIC_9B_READY_PREFLIGHT.ipynb',nb)
write(code_dir/'REVIEW_FIRST_TRAIN_CONFIG.json',{'ready_for_review':True,'execution_authorized_by_this_file':False,
 'mode':'fresh adapter from frozen base; existing adapters/checkpoints not loaded',
 'base':'Qwen/Qwen3.5-9B','revision':'c202236235762e1c871ad0ccb60c8ee5ba337b9a',
 'sft_sha256':sft_sha,'rows':rows,'lora_rank':8,'lora_alpha':16,'lora_dropout':0.05,
 'learning_rate':2e-5,'epochs':1,'expected_steps_if_all_768_rows_retained':48,'batch':1,
 'gradient_accumulation':16,'max_length':2048,'seed':20261107,
 'gates':['99 synthetic tests','GPU dependencies and Qwen3_5ForCausalLM import','9B model synthetic smoke: loss delta <0.02 and finite gradients',
          'GPU and output space','review training configuration'],
 'resume_support':'Existing full checkpoints preserved locally; this notebook creates a new run, does not resume.'})
report.update(state='local_upload_preparation_complete',base_asset_files=len(hashes),base_asset_bytes=sum(x['bytes'] for x in manifest.values()),
    sft_rows=rows,sft_sha256=sft_sha,code_zip_sha256=code_sha,
    existing_adapters_and_checkpoints_modified=False,base4_upload_prepared=False,
    base4_note='Renamed to index paths; total_size inconsistency remains. Excluded from 9B training upload.',
    staging_uses_hardlinks_for_base=True,hardlinks_are_not_independent_backups=True)
write(REPORT,report)
guide='''# Sẵn sàng upload Kaggle — chỉ dành cho 9B

1. Notebook mới: File → Import Notebook → chọn 03_CODE_AND_NOTEBOOK/ZAIC_9B_READY_PREFLIGHT.ipynb.
2. Add Input / Upload: tạo dataset mã Private chỉ chứa ZAIC_CODE_ONLY.zip. Chạy mặc định preflight (CPU): checksum nguồn + 99 tests. Không gắn notebook Colab cũ.
3. Dataset base Private: upload toàn bộ NỘI DUNG thư mục 01_BASE_QWEN35_9B, giữ base/ và BASE_UPLOAD_MANIFEST.json cùng cấp. Giữ cố định phiên bản này cho mọi run.
4. Dataset TRAIN SFT Private: upload toàn bộ NỘI DUNG 02_APPROVED_TRAIN_SFT, giữ data/ và SFT_UPLOAD_MANIFEST.json cùng cấp. Chỉ bộ SFT baseline; không có public/private test, câu hỏi đánh giá, transaction hay journal.
5. Gắn đúng một version mỗi dataset. Notebook tự xác định đường dẫn thực tế từ manifest, không phụ thuộc tên slug Kaggle.
6. Bật GPU, xem báo cáo phiên bản thư viện, chạy smoke 9B giả lập. Chưa train cho tới khi đạt cổng smoke và cấu hình được xem xét. Không cài đè stack Colab theo thói quen.
7. REVIEW_FIRST_TRAIN_CONFIG.json ghi cấu hình train fresh adapter. Không tiếp tục checkpoint cũ bằng notebook này. Adapter/checkpoint cũ giữ ở PRIVATE_DATA.
8. Sau run, Save Version kèm output của phiên hiện tại, tránh chạy lại train. Đợi version thành công, tải ZIP về runs/<run_id>, so manifest SHA256 trước khi đánh dấu đã đồng bộ. Giữ checkpoint đầy đủ để chuẩn bị resume riêng.

Base trong staging là hardlink tới file chuẩn trên cùng ổ D: không tốn thêm 19 GB nhưng cũng KHÔNG phải bản sao lưu độc lập. Chỉ dùng để upload, không sửa nội dung các file staging.

Quyền cloud: lưu lời xác nhận của người dùng trong CLOUD_PERMISSION_RECORD.json; chưa có URL ngoại lệ chính thức để đối chứng độc lập. Không thay đổi chia sẻ thành Public.

Bộ 4B không nằm trong gói upload 9B. Đã sửa tên/vị trí theo index nhưng vẫn giữ nguyên bất nhất metadata total_size, chưa chứng nhận runtime.
'''
(STAGE/'UPLOAD_GUIDE_VI.md').write_text(guide,encoding='utf-8')
write(ROOT/'KAGGLE_UPLOAD_LATEST.json',{'stage':str(STAGE),'report':str(REPORT),'upload_performed':False,'training_started':False})
write(PROJECT/'reports/KAGGLE_UPLOAD_PREPARATION.json',report)
print('PREPARED',str(STAGE),flush=True)
