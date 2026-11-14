"""Local run verification. No network, no pickle loading, no raw data output."""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import shutil
import uuid
import zipfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def seal(root, metadata):
    root = Path(root)
    write_json(root / 'RUN_METADATA.json', metadata)
    files = {}
    for p in sorted(root.rglob('*')):
        if p.is_symlink():
            raise ValueError('Export contains a symlink')
        if p.is_file() and p.name != 'ARTIFACT_MANIFEST.json':
            name = p.relative_to(root).as_posix()
            if name.startswith(('assets/', 'data/')):
                raise ValueError('Base/data must never be exported with a run')
            files[name] = {'bytes': p.stat().st_size, 'sha256': digest(p)}
    manifest = {'schema': 'zaic-artifacts-1', 'run_id': metadata['run_id'], 'files': files}
    write_json(root / 'ARTIFACT_MANIFEST.json', manifest)
    return digest(root / 'ARTIFACT_MANIFEST.json')


def safe_name(name):
    p = PurePosixPath(name)
    if not name or '\\' in name or ':' in name or p.is_absolute() or '..' in p.parts or str(p) != name:
        raise ValueError('Unsafe artifact path')
    return p


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'ARTIFACT_MANIFEST.json').read_text(encoding='utf-8'))
    if manifest.get('schema') != 'zaic-artifacts-1' or not manifest.get('files'):
        raise ValueError('Invalid manifest')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', manifest['run_id']):
        raise ValueError('Invalid run ID')
    expected = set(manifest['files']) | {'ARTIFACT_MANIFEST.json'}
    actual = set()
    for p in root.rglob('*'):
        if p.is_symlink():
            raise ValueError('Symlinks are not accepted')
        if p.is_file():
            actual.add(p.relative_to(root).as_posix())
    if actual != expected:
        raise ValueError('Missing or unexpected files')
    for name, item in manifest['files'].items():
        safe_name(name)
        p = root / name
        if p.stat().st_size != item['bytes'] or digest(p) != item['sha256']:
            raise ValueError('Artifact checksum mismatch')
    meta = json.loads((root / 'RUN_METADATA.json').read_text(encoding='utf-8'))
    if meta['run_id'] != manifest['run_id'] or meta.get('state') != 'completed':
        raise ValueError('Run not completed')
    return manifest, meta


def import_saved_run(source, runs_root, receipt):
    """Receipt must record an OBSERVED completed Kaggle version and remote manifest hash."""
    source, runs_root = Path(source), Path(runs_root)
    if not receipt.get('saved_version_confirmed') or not re.fullmatch(
        r'https://www\.kaggle\.com/code/[\w-]+/[\w-]+\?scriptVersionId=\d+', receipt.get('version_url', '')
    ):
        raise ValueError('Need an observed completed Kaggle version URL')
    runs_root.mkdir(parents=True, exist_ok=True)
    staging = runs_root / ('incoming_' + uuid.uuid4().hex)
    staging.mkdir()
    if source.is_dir():
        for p in source.rglob('*'):
            if p.is_symlink():
                raise ValueError('Source contains a symlink')
        shutil.copytree(source, staging, dirs_exist_ok=True)
    else:
        with zipfile.ZipFile(source) as z:
            seen = set()
            total = sum(i.file_size for i in z.infolist())
            if total > shutil.disk_usage(runs_root).free - 1024**3:
                raise ValueError('Insufficient space for extraction')
            for item in z.infolist():
                name = item.filename.rstrip('/')
                safe_name(name)
                if name.casefold() in seen or (item.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Duplicate path or archive symlink')
                seen.add(name.casefold())
            z.extractall(staging)
    candidates = list(staging.rglob('ARTIFACT_MANIFEST.json'))
    if len(candidates) != 1:
        raise ValueError('Select one run export archive')
    root = candidates[0].parent
    manifest, meta = verify(root)
    manifest_sha = digest(root / 'ARTIFACT_MANIFEST.json')
    if manifest_sha != receipt.get('remote_manifest_sha256'):
        raise ValueError('Downloaded manifest differs from saved Kaggle manifest')
    target = runs_root / manifest['run_id']
    if target.exists():
        verify(target)
        if digest(target / 'ARTIFACT_MANIFEST.json') != manifest_sha:
            raise ValueError('Conflict: kept previous run and incoming download')
    else:
        shutil.copytree(root, target)
        verify(target)
    # Commit pointers only AFTER all bytes have been verified on both sides.
    state_path = runs_root / 'SYNC_STATE.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {
        'LATEST_SYNCED': None, 'BEST_MODEL': None, 'runs': []}
    record = {'run_id': manifest['run_id'], 'action': meta['action'], 'manifest_sha256': manifest_sha,
              'kaggle_version': receipt['version_url'], 'local_path': str(target),
              'synced_utc': datetime.now(timezone.utc).isoformat(), 'status': 'synced',
              'metrics': meta.get('metrics', {})}
    if not any(x['run_id'] == record['run_id'] for x in state['runs']):
        state['runs'].append(record)
    # Imported order is not run chronology.
    newest = max(state['runs'], key=lambda x: json.loads(
        (runs_root / x['run_id'] / 'RUN_METADATA.json').read_text(encoding='utf-8'))['created_utc'])
    state['LATEST_SYNCED'] = newest['run_id']
    write_json(state_path, state)
    write_json(runs_root / ('RECEIPT_' + manifest['run_id'] + '.json'), receipt)
    write_dashboard(runs_root, state)
    return record


def choose_best(runs_root, run_id, evaluation):
    """Explicit selection from comparable TRAIN evaluation, never from loss alone."""
    runs_root = Path(runs_root)
    state = json.loads((runs_root / 'SYNC_STATE.json').read_text(encoding='utf-8'))
    manifest, meta = verify(runs_root / run_id)
    if meta['action'] != 'train' or not any(r['run_id'] == run_id for r in state['runs']):
        raise ValueError('Best model must be a verified training run')
    if evaluation.get('split') != 'TRAIN' or not evaluation.get('evaluation_set_sha256') or not evaluation.get('metrics'):
        raise ValueError('Need frozen TRAIN evaluation and metrics')
    write_json(runs_root / ('EVALUATION_' + run_id + '.json'), evaluation)
    state['BEST_MODEL'] = run_id
    write_json(runs_root / 'SYNC_STATE.json', state)
    write_dashboard(runs_root, state)


def write_dashboard(root, state):
    lines = ['# Theo dõi Kaggle – máy cá nhân', '',
             'LATEST_SYNCED: ' + str(state['LATEST_SYNCED']),
             '', 'BEST_MODEL: ' + str(state['BEST_MODEL']), '',
             '| Run | Tác vụ | Trạng thái | Phiên bản Kaggle |', '|---|---|---|---|']
    for r in state['runs']:
        lines.append(f"| {r['run_id']} | {r['action']} | đã đồng bộ SHA256 | {r['kaggle_version']} |")
    (Path(root) / 'RUNS_VI.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    import tkinter as tk
    from tkinter import filedialog, simpledialog, messagebox
    app = tk.Tk(); app.withdraw()
    try:
        source = filedialog.askopenfilename(title='Chọn ZIP output của một run Kaggle', filetypes=[('ZIP', '*.zip')])
        if source:
            version = simpledialog.askstring('Phiên bản Kaggle', 'URL phiên bản đã lưu hoàn tất (có scriptVersionId):')
            sha = simpledialog.askstring('SHA256 nguồn', 'Hash ARTIFACT_MANIFEST của output trên phiên bản Kaggle đã lưu:')
            if version and sha:
                result = import_saved_run(source, Path(__file__).resolve().parents[2] / 'PRIVATE_DATA/runs',
                    {'saved_version_confirmed': True, 'version_url': version, 'remote_manifest_sha256': sha.strip()})
                messagebox.showinfo('Đã đối chiếu', 'Đã đồng bộ: ' + result['run_id'])
    except Exception as exc:
        messagebox.showerror('Chưa đồng bộ', str(exc))
