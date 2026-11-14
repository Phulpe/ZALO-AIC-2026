"""Private runtime readers; gold is lazy and limited to the declared split."""
from pathlib import Path
import json,copy
from relearning import sha,atomic

class LazyGoldReader:
    def __init__(self,path,allowed_ids,expected_sha256):
        self.path=Path(path);self.allowed=set(allowed_ids);self.expected=expected_sha256;self.rows=None
    def __call__(self,question_id):
        if question_id not in self.allowed:raise ValueError('Gold requested outside allowed split')
        if self.rows is None:
            if sha(self.path)!=self.expected:raise ValueError('Gold source hash mismatch')
            rows=[json.loads(line) for line in self.path.read_text(encoding='utf-8').splitlines() if line.strip()]
            if len({r['id'] for r in rows})!=len(rows) or {r['id'] for r in rows}!=self.allowed:raise ValueError('Gold split IDs mismatch')
            self.rows={r['id']:r for r in rows}
        return copy.deepcopy(self.rows[question_id])

def load_learning_pool(root,expected_manifest_sha256,expected_split_sha256):
    root=Path(root)
    if sha(root/'DATA_MANIFEST.json')!=expected_manifest_sha256 or sha(root/'SPLIT_MANIFEST.json')!=expected_split_sha256:raise ValueError('Dataset/split fingerprint mismatch')
    manifest=json.loads((root/'DATA_MANIFEST.json').read_text())
    split=json.loads((root/'SPLIT_MANIFEST.json').read_text())
    for name in ['learning/model_pool.jsonl','learning/gold.jsonl']:
        if sha(root/name)!=manifest['files'][name]['sha256']:raise ValueError('Learning input changed')
    pool=[json.loads(l) for l in (root/'learning/model_pool.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    ids={q['id'] for q in pool}
    if len(ids)!=len(pool) or any(q['split']!='learning' or q['business_id'] not in split['groups']['learning'] for q in pool):raise ValueError('Learning IDs or business split invalid')
    return pool,split,LazyGoldReader(root/'learning/gold.jsonl',ids,manifest['files']['learning/gold.jsonl']['sha256'])

def pending_coverage(path,pool,parent_hash,split_hash):
    path=Path(path)
    if path.exists():raise FileExistsError('Preserve previous coverage ledger')
    obj={'parent_sha256':parent_hash,'split_manifest_sha256':split_hash,'questions':{q['id']:{'status':'pending','attempted':False} for q in pool}}
    if len(obj['questions'])!=len(pool):raise ValueError('Duplicate expected IDs')
    atomic(path,obj);return obj

def reconcile_coverage(path,round_ledger):
    state=json.loads(Path(path).read_text())
    if not set(round_ledger['questions'])<=set(state['questions']):raise ValueError('Unexpected IDs in round')
    for question_id,row in round_ledger['questions'].items():
        attempted=any(a.get('state') in {'predicted','correct','wrong','invalid_plan','error'} for a in row['attempts'])
        state['questions'][question_id]={'status':row['status'],'attempted':attempted}
    attempted=sum(r['attempted'] for r in state['questions'].values());expected=len(state['questions'])
    state['metrics']={'expected':expected,'attempted':attempted,'attempt_coverage':attempted/expected if expected else None,'pending':sum(r['status']=='pending' for r in state['questions'].values())}
    atomic(path,state);return state['metrics']
