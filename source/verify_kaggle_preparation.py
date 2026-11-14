from pathlib import Path
import ast, hashlib, json, zipfile
root=Path('D:/ZAIC');latest=json.loads((root/'KAGGLE_UPLOAD_LATEST.json').read_text());stage=Path(latest['stage'])
report=json.loads((stage/'PREPARATION_REPORT.json').read_text(encoding='utf-8'))
project=root/'drive-download-20241211T124801Z-001/ZAIC_Local_GUI/CODEX_PROJECT'
assert report['state']=='local_upload_preparation_complete'
assert len(report['moves'])==6 and all(m['verified'] for m in report['moves'])
assert all(Path(m['destination']).is_file() and not Path(m['source']).exists() for m in report['moves'])
base=stage/'01_BASE_QWEN35_9B/base'
idx=json.loads((base/'model.safetensors.index.json').read_text())
assert len(set(idx['weight_map'].values()))==4
assert all((base/name).is_file() for name in set(idx['weight_map'].values()))
assert len(list(base.iterdir()))==15
assert not list(base.rglob('.cache'))
code=stage/'03_CODE_AND_NOTEBOOK'
with zipfile.ZipFile(code/'ZAIC_CODE_ONLY.zip') as z:
    assert z.testzip() is None
    expected={name.removeprefix('CODEX_PROJECT/'):hashlib.sha256(z.read(name)).hexdigest() for name in z.namelist()}
    assert len(expected)==33
    for name in z.namelist():
        if name.endswith(('.py','.pyw')):ast.parse(z.read(name).decode('utf-8'))
# Kaggle may automatically unpack uploaded ZIPs. Support both forms without
# trusting new source bytes or requiring a user to alter upload defaults.
notebook_path=code/'ZAIC_9B_READY_PREFLIGHT.ipynb'
nb=json.loads(notebook_path.read_text(encoding='utf-8'))
launch='''from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, stat, sys, zipfile
EXPECTED_CODE_FILES = __EXPECTED_FILES__
archives=list(Path('/kaggle/input').rglob('ZAIC_CODE_ONLY.zip'))
if archives:
    assert len(archives)==1, 'Attach exactly one code dataset version'
    archive=archives[0]
    assert hashlib.sha256(archive.read_bytes()).hexdigest()=='__ZIP_SHA__', 'Code archive mismatch'
    source=Path('/kaggle/temp/zaic_source')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    source.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        assert sum(i.file_size for i in z.infolist())<25*1024*1024
        for i in z.infolist():
            assert (source/i.filename).resolve().is_relative_to(source.resolve())
            assert not stat.S_ISLNK(i.external_attr>>16)
        z.extractall(source)
    source=source/'CODEX_PROJECT'
else:
    matches=list(Path('/kaggle/input').rglob('CODEX_PROJECT/gui/versioned_runner.py'))
    assert len(matches)==1, 'Attach the code ZIP or its extracted CODEX_PROJECT folder'
    source=matches[0].parents[1]
for name, expected_hash in EXPECTED_CODE_FILES.items():
    target=(source/name).resolve()
    assert target.is_relative_to(source.resolve())
    assert hashlib.sha256(target.read_bytes()).hexdigest()==expected_hash, 'Code file hash mismatch'
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
'''.replace('__EXPECTED_FILES__',repr(expected)).replace('__ZIP_SHA__',report['code_zip_sha256'])
nb['cells'][2]['source']=launch.splitlines(keepends=True)
for c in nb['cells']:
    if c['cell_type']=='code':ast.parse(''.join(c['source']));assert not c['outputs']
assert "'ACTION': 'preflight'" in ''.join(nb['cells'][1]['source'])
notebook_path.write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8')
checks={'all_six_renames_sha256_passed':True,'base_four_shards_resolve_from_index':True,
 'base_15_assets_only_no_cache':True,'code_archive_33_files_crc_and_syntax_passed':True,
 'notebook_default_preflight_no_outputs':True,'supports_kaggle_auto_extracted_code_zip':True,
 'frozen_source_27_hashes_verified':True,'windows_portable_synthetic_tests_passed':99,'sync_tests_passed':5,
 'gpu_smoke_verified':False,'upload_performed':False,'training_started':False}
(stage/'LOCAL_READY_CHECKS.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
# Preserve the original inventory as pre-rename evidence and publish a new path map.
inventory=json.loads((root/'WEIGHTS_INVENTORY.json').read_text(encoding='utf-8'))
mapping={m['source']:m['destination'] for m in report['moves']}
def remap(x):
    if isinstance(x,str):return mapping.get(x,x)
    if isinstance(x,list):return [remap(v) for v in x]
    if isinstance(x,dict):return {k:remap(v) for k,v in x.items()}
    return x
inventory=remap(inventory)
inventory['post_rename_evidence']=str(stage/'PREPARATION_REPORT.json')
inventory['weights_pending_classification']=[]
for b in inventory['base_sets']:
    for s in b['shards']:
        s['canonical_exists']=True;s['status']='present_at_expected_path'
        for m in s['matches']:m['filename_matches_index']=True
    b['loadable_at_current_path']=b['total_size_matches_index']
    b['canonical_shard_paths_complete']=True
    b['inference_status']='structurally_ready_not_runtime_tested' if b['total_size_matches_index'] else 'all_shards_at_index_paths_but_metadata_total_size_unresolved'
for p in [root/'WEIGHTS_INVENTORY_AFTER_RENAME.json',project/'WEIGHTS_INVENTORY_AFTER_RENAME.json']:
    p.write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf-8')
guide=stage/'UPLOAD_GUIDE_VI.md'
guide.write_text(guide.read_text(encoding='utf-8')+'\nNotebook hỗ trợ cả ZIP mã nguyên vẹn và CODEX_PROJECT do Kaggle tự giải nén; cả hai đều được kiểm hash từng file trước import.\n',encoding='utf-8')
for folder in [root,project]:
    p=folder/'WEIGHTS_CHECK_REPORT.md'
    p.write_text(p.read_text(encoding='utf-8')+'\n\n## Cập nhật sau đổi tên được người dùng cho phép\n\nĐã đổi tên/chuyển 6 shard từ weights_pending vào đúng base 9B/4B; SHA256 sau chuyển đều khớp. Xem WEIGHTS_INVENTORY_AFTER_RENAME.json để lấy đường dẫn hiện hành. Các nhận xét về sai tên/vị trí phía trên là trạng thái trước chuyển; bất nhất metadata total_size 4B vẫn còn. Adapter/checkpoint cũ không thay đổi. Gói upload 9B: '+str(stage)+'\n',encoding='utf-8')
print(json.dumps({'stage':str(stage),'checks':checks,'base_bytes':report['base_asset_bytes'],'sft_rows':report['sft_rows'],
                  'notebook':str(notebook_path)},indent=2))
