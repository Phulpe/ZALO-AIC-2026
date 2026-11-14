from pathlib import Path
import json, hashlib
roots=list(Path('/kaggle/input').rglob('RUNTIME_DATA_MANIFEST.json'))
assert len(roots)==1,'Need exactly one runtime manifest'
p=roots[0]
assert hashlib.sha256(p.read_bytes()).hexdigest()=='523db34eb8e8bab52aa52d1d046307f19dbb3668495025ca05d66abd1e4c6e70'
m=json.loads(p.read_text())
total=0
for name,entry in m['files'].items():
    f=p.parent/name
    assert f.is_file() and f.stat().st_size==entry['bytes']
    assert hashlib.sha256(f.read_bytes()).hexdigest()==entry['sha256']
    total+=f.stat().st_size
assert not (p.parent/'audit_exposed').exists()
r={'verified_files':len(m['files']),'bytes':total,'runtime_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'audit_gold_excluded':True}
Path('/kaggle/working/RUNTIME_INPUT_RECEIPT_V2.json').write_text(json.dumps(r,indent=2))
print('RUNTIME_INPUT_RECEIPT',json.dumps(r))
