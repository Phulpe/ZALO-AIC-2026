"""Prepare a gated, time-bounded smoke notebook; does not execute it."""
from pathlib import Path
import json,ast,base64,hashlib
root=Path(__file__).parent
nb=json.loads(Path('D:/ZAIC/EXPERIMENTS/relearning_v1/ZAIC_RELEARNING_V1.ipynb').read_text(encoding='utf-8'))
supplement={name:(root/name).read_text(encoding='utf-8') for name in ['smoke_validation.py','test_smoke_validation.py']}
text=json.dumps(supplement,ensure_ascii=False)
payload=base64.b64encode(text.encode()).decode();digest=hashlib.sha256(text.encode()).hexdigest()
launch='''# Additional CPU-only validators. No GPU model is loaded here.
import base64,hashlib,json,subprocess,sys
extra=base64.b64decode(PAYLOAD)
assert hashlib.sha256(extra).hexdigest()==EXPECTED
for name,source_text in json.loads(extra).items():
    target=workspace/name
    assert target.resolve().is_relative_to(workspace.resolve())
    with target.open('x',encoding='utf-8') as f:f.write(source_text)
check=subprocess.run([sys.executable,'-X','utf8','-m','unittest','test_smoke_validation'],cwd=workspace,capture_output=True,text=True)
assert check.returncode==0,'Additional CPU smoke validators failed; do not enable GPU'
print('Additional smoke validators: 8 CPU tests passed; no GPU execution')
'''.replace('PAYLOAD',repr(payload)).replace('EXPECTED',repr(digest))
gpu='''# Default remains preflight. Complete evidence and CPU input checks before GPU.
if CONFIG['action']=='gpu_smoke':
    assert CONFIG['gpu_confirmed'], 'Review limited smoke budget first'
    assert CONFIG['competition_permission_evidence']['status']=='verified', 'Competition evidence is still unverified'
    assert CONFIG['max_steps']==1, 'This notebook only permits one smoke optimizer step'
    from pathlib import Path
    bases=list(Path('/kaggle/input').rglob('BASE_UPLOAD_MANIFEST.json'))
    adapters=list(Path('/kaggle/input').rglob('final/adapter_model.safetensors'))
    assert len(bases)==len(adapters)==1
    assert CONFIG['base_hashes']=={k:v['sha256'] for k,v in json.loads(bases[0].read_text())['files'].items()}
    CONFIG['base']=str(bases[0].parent/'base')
    CONFIG['adapter']=str(adapters[0].parent)
    CONFIG['output_root']='/kaggle/working/relearning_model_runs'
    request=workspace/'GPU_SMOKE_REQUEST.json'
    request.write_text(json.dumps(CONFIG,indent=2))
    child=workspace/'execute_gpu_smoke.py'
    with child.open('x',encoding='utf-8') as f:
        f.write(CHILD)
    # Bound the child process, including model load, training and reload.
    # Stopping the Kaggle GPU session itself is still an explicit GUI action.
    try:
        with (workspace/'GPU_SMOKE_PRIVATE.log').open('w',encoding='utf-8') as log:
            job=subprocess.run([sys.executable,str(child),str(request)],cwd=workspace,stdout=log,stderr=subprocess.STDOUT,timeout=900)
        assert job.returncode==0,'GPU smoke failed: retain private log and stop GPU session now'
        print((workspace/'GPU_SMOKE_RECEIPT.json').read_text())
    finally:
        print('STOP SESSION NOW, set Accelerator=None and verify off. Download saved output afterwards.')
elif CONFIG['action']!='preflight':
    raise ValueError('Long training is disabled in this smoke notebook')
'''
child='''from pathlib import Path
import sys,json
from train_warm import train
from smoke_targets import targets
from relearning import verify_export,atomic
from smoke_validation import validate_after_training,reload_forward_checkpoint
config=json.loads(Path(sys.argv[1]).read_text())
rows=targets()
run=train(config,rows)
receipt=validate_after_training(Path(config['adapter']),run/'checkpoint',verify_export,lambda ck:reload_forward_checkpoint(ck,Path(config['base']),rows))
receipt.update(run_id=run.name,init_mode='warm_start',steps=1,gpu_session_stop_required=True)
atomic(Path('GPU_SMOKE_RECEIPT.json'),receipt)
'''
gpu=gpu.replace('CHILD',repr(child))
def code(source):
    ast.parse(source)
    return {'cell_type':'code','source':source.splitlines(keepends=True),'metadata':{},'execution_count':None,'outputs':[]}
nb['cells'][3:]=[code(launch),code(gpu)]
nb['cells'][0]['source']+=['\n\nPrepared revision adds CPU post-smoke validator tests, a 900-second child timeout, saved-adapter reload and native/masked loss parity. This revision has not run on Kaggle/GPU yet. No long training. Stop GPU session explicitly after work.']
target=root/'ZAIC_FINAL_SMOKE_REVIEW.ipynb'
target.write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8')
(root/'SUPPLEMENT_MANIFEST.json').write_text(json.dumps({'supplement_sha256':digest,'notebook_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'cpu_tests':8,'gpu_executed':False,'kaggle_revision_executed':False},indent=2),encoding='utf-8')
print('Prepared gated final smoke notebook; not executed:',target)
