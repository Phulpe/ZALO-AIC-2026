"""Checksum inventory only: never deserialize optimizer or RNG pickle files."""
from pathlib import Path
import json, hashlib, datetime, sys


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(root):
    manifest = root/'ARTIFACT_MANIFEST.json'
    data = json.loads(manifest.read_text(encoding='utf-8'))
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p != manifest}
    if actual != set(data['files']):
        raise ValueError('Incomplete download or unexpected extra files')
    total = 0
    for name, item in data['files'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()) or path.is_symlink():
            raise ValueError('Unsafe downloaded path')
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('Downloaded checksum mismatch')
        total += item['bytes']
    return {'manifest_sha256': sha(manifest), 'files':len(actual), 'bytes':total}


if __name__ == '__main__':
    download = Path(sys.argv[1])
    matches = list(download.rglob('WORKFLOW_COMPLETED.json'))
    if len(matches) != 1:
        raise ValueError('Ambiguous downloaded pilot')
    pilot = matches[0].parent
    proof = verify(pilot)
    expected = 'bd8904b7e920a2aeb40de8ecb86334c6bb3012f70f24c57f8d319ee33195ad95'
    if proof['manifest_sha256'] != expected:
        raise ValueError('Wrong saved pilot version')
    result = json.loads(matches[0].read_text())
    run = pilot/'model_runs'/result['run_id']
    checkpoints = []
    for folder in sorted(run.rglob('*')):
        if folder.is_dir() and (folder/'adapter_model.safetensors').is_file():
            required = ['adapter_model.safetensors','adapter_config.json','optimizer.pt',
                        'scheduler.pt','rng_state.pth','trainer_state.json']
            missing = [n for n in required if not (folder/n).is_file()]
            checkpoints.append({'path':str(folder),'weight_sha256':sha(folder/'adapter_model.safetensors'),
                'bytes':(folder/'adapter_model.safetensors').stat().st_size,'missing_resume_files':missing,
                'scaler_present':(folder/'scaler.pt').is_file(),'full_resume_executed':False})
    if sha(run/'checkpoint/adapter_model.safetensors') != result['candidate_sha256']:
        raise ValueError('Final adapter differs')
    report = {'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'state':'synced_checksum_verified','pilot':str(pilot),'run_id':result['run_id'],
        'Kaggle_version':'lnghongphc/zaic-competition-pilot-20261110/3',
        'verification':proof,'checkpoints':checkpoints,'pickle_deserialized':False,
        'BEST_MODEL_updated':False,'full_resume_verified':False}
    Path(sys.argv[2]).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'state':report['state'],'files':proof['files'],'checkpoint_sets':len(checkpoints)}))
