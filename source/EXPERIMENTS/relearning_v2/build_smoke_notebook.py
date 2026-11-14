"""Merge tested v2 CPU package with bounded GPU smoke + reload checks."""
from pathlib import Path
import json,ast,hashlib
root=Path(__file__).parent
nb=json.loads((root/'ZAIC_RELEARNING_V2.ipynb').read_text(encoding='utf-8'))
config=ast.literal_eval(''.join(nb['cells'][1]['source']).split('=',1)[1])
config['output_root']='/kaggle/working/relearning_model_runs_v2'
nb['cells'][1]['source']=['CONFIG='+repr(config)]
nb['cells'][0]['source']=['# ZAIC Relearning v2 — CPU preflight then bounded GPU smoke\n\nOriginal model/notebook/checkpoints preserved. Permission status is user_attested under revised AGENTS.md, not independent verification. Default preflight; GPU smoke must be explicitly selected. Stop GPU immediately after work. No long training.']
gpu='''sys.path.insert(0,str(workspace))
if CONFIG['action']=='gpu_smoke':
    from permission import permitted
    assert CONFIG['gpu_confirmed'] and permitted(CONFIG['competition_permission_evidence'])
    assert CONFIG['max_steps']==1
    bases=list(Path('/kaggle/input').rglob('BASE_UPLOAD_MANIFEST.json'))
    adapters=list(Path('/kaggle/input').rglob('final/adapter_model.safetensors'))
    assert len(bases)==len(adapters)==1, 'Attach exactly one base and trained parent version'
    assert CONFIG['base_hashes']=={k:v['sha256'] for k,v in json.loads(bases[0].read_text())['files'].items()}
    CONFIG['base']=str(bases[0].parent/'base');CONFIG['adapter']=str(adapters[0].parent)
    runtime_manifests=list(Path('/kaggle/input').rglob('RUNTIME_DATA_MANIFEST.json'))
    if runtime_manifests:
        assert len(runtime_manifests)==1
        CONFIG['runtime_root']=str(runtime_manifests[0].parent)
        CONFIG['runtime_manifest_sha256']='523db34eb8e8bab52aa52d1d046307f19dbb3668495025ca05d66abd1e4c6e70'
        CONFIG['collect_learning_probe']=True
        CONFIG['learning_probe_max_questions']=8
    request=workspace/'GPU_SMOKE_REQUEST.json';request.write_text(json.dumps(CONFIG,indent=2))
    child=workspace/'gpu_smoke_entry.py'
    child.write_text(CHILD,encoding='utf-8')
    try:
        with (workspace/'GPU_SMOKE_PRIVATE.log').open('w',encoding='utf-8') as log:
            job=subprocess.run([sys.executable,str(child),str(request)],cwd=workspace,stdout=log,stderr=subprocess.STDOUT,timeout=900)
        assert job.returncode==0,'GPU smoke failed; retain private log and stop GPU session'
        print((workspace/'GPU_SMOKE_RECEIPT.json').read_text())
    finally:print('Stop session now; Accelerator=None. Save outputs before shutdown; download after GPU is off.')
elif CONFIG['action']!='preflight':raise ValueError('Long training is disabled')
'''
child='''from pathlib import Path
import sys,json
from train_warm import train
from smoke_targets import targets
from relearning import verify_export,atomic
from smoke_validation import validate_after_training,reload_forward_checkpoint
config=json.loads(Path(sys.argv[1]).read_text())
rows=targets();run=train(config,rows)
receipt=validate_after_training(Path(config['adapter']),run/'checkpoint',verify_export,lambda ck:reload_forward_checkpoint(ck,Path(config['base']),rows))
receipt.update(run_id=run.name,run_path=str(run),init_mode='warm_start',steps=1,gpu_stop_required=True)
if (run/'LEARNING_PROBE_REFERENCE.json').exists():
    private_probe=json.loads((run/'LEARNING_PROBE_REFERENCE.json').read_text())
    receipt['learning_probe']={k:v for k,v in private_probe.items() if k not in {'path'}}
atomic(Path('GPU_SMOKE_RECEIPT.json'),receipt)
'''
gpu=gpu.replace('CHILD',repr(child));ast.parse(gpu)
nb['cells'][3]['source']=gpu.splitlines(keepends=True)
target=root/'ZAIC_RELEARNING_V2_SMOKE.ipynb'
target.write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8')
print('Notebook ready for CPU preflight:',hashlib.sha256(target.read_bytes()).hexdigest())
