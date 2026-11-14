from pathlib import Path
import json,ast,hashlib

root=Path(__file__).parent
source=root.parent/'relearning_v3'
nb=json.loads((source/'ZAIC_RELEARNING_V3_READY.ipynb').read_text(encoding='utf-8'))
config=ast.literal_eval(''.join(nb['cells'][1]['source']).split('=',1)[1])
split=json.loads(Path('D:/ZAIC/PRIVATE_DATA/relearning_v2_prepared/20261108T070936492119Z/SPLIT_MANIFEST.json').read_text(encoding='utf-8'))
config.update(action='preflight',budget_approved=False,gpu_confirmed=False,max_steps=4,approved_max_steps=4,max_seconds=900,approved_max_seconds=900,learning_businesses=split['groups']['learning'],output_root='/kaggle/working/relearning_pilot_runs_v3',verified_dataset_directory=None,verified_dataset_manifest_sha256=None)
config.update(data_manifest_sha256='dc5f6c3a4894a61d3048a91484f6ca0887021ddb901777ab340353b0e702ccad',split_manifest_sha256='1fc5342af2e81405fc32d0dbae49237815276541a50443e6aadbcbb3638c9f4f')
(root/'PILOT_REQUEST.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
nb['cells'][0]['source']=['# Lượt correction/replay kế tiếp — chưa mở train\n\nCPU preflight mặc định. V3 đã qua 99+67 tests và GPU warm-start/save/reload smoke. Wrapper pilot có thêm 8 tests trên máy. Chưa có correction được xác minh; đường dẫn/hash dataset đang null. Không dùng candidate giả lập làm parent. Ngân sách đề xuất 4 bước/900 giây chỉ cho pha train/test, không bao gồm thu thập toàn pool hay đánh giá 552 câu. Giữ Accelerator=None; không Run All khi action=pilot_train. BEST_MODEL giữ nguyên cho đến khi đánh giá đủ.']
nb['cells'][1]['source']=['CONFIG='+repr(config)]
launch=''.join(nb['cells'][2]['source'])
for name in ['pilot_entry.py','test_pilot_entry.py']:
    launch+=f"\n(workspace/{name!r}).write_text({(root/name).read_text(encoding='utf-8')!r},encoding='utf-8')\n"
launch+="\npilot_tests=subprocess.run([sys.executable,'-m','unittest','test_pilot_entry'],cwd=workspace,capture_output=True,text=True)\n(workspace/'PILOT_TEST_DETAILS.txt').write_text(pilot_tests.stdout+pilot_tests.stderr)\nassert pilot_tests.returncode==0,'Pilot gate tests failed'\nprint(json.dumps({'pilot_gate_tests':8,'passed':True,'training_enabled':False,'verified_corrections_available':0}))\n"
nb['cells'][2]['source']=launch.splitlines(keepends=True)
bootstrap=(source/'OFFLINE_BNB_BOOTSTRAP.py').read_text(encoding='utf-8')
child="""from pathlib import Path
import json,sys
from pilot_entry import run_pilot
from smoke_validation import validate_after_training,reload_forward_checkpoint
from smoke_targets import targets
from relearning import verify_export,atomic
config=json.loads(Path(sys.argv[1]).read_text())
run=run_pilot(config)
receipt=validate_after_training(Path(config['adapter']),run/'checkpoint',verify_export,lambda ck:reload_forward_checkpoint(ck,Path(config['base']),targets()))
receipt.update(run_id=run.name,mode='learning',init_mode='warm_start',synthetic_training=False,best_model_promoted=False,evaluation_pending=True)
atomic(Path('PILOT_RECEIPT.json'),receipt)
"""
gpu="""if CONFIG['action']=='pilot_train':
    assert CONFIG['budget_approved'] and CONFIG['gpu_confirmed'],'Reviewed budget required'
    assert CONFIG['verified_dataset_directory'] and CONFIG['verified_dataset_manifest_sha256'],'Correction/replay dataset not ready'
    exec(BOOTSTRAP)
    sys.path.insert(0,str(workspace))
    from pilot_entry import validate_config
    from runtime_gpu import single_gpu_env
    validate_config(CONFIG)
    bases=list(Path('/kaggle/input').rglob('BASE_UPLOAD_MANIFEST.json'))
    adapters=list(Path('/kaggle/input').rglob('final/adapter_model.safetensors'))
    assert len(bases)==len(adapters)==1
    CONFIG['base']=str(bases[0].parent/'base');CONFIG['adapter']=str(adapters[0].parent)
    request=workspace/'PILOT_REQUEST.json';request.write_text(json.dumps(CONFIG))
    child=workspace/'pilot_gpu_entry.py';child.write_text(CHILD)
    try:
        with (workspace/'PILOT_PRIVATE.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,str(child),str(request)],cwd=workspace,env=single_gpu_env(os.environ),stdout=log,stderr=subprocess.STDOUT,timeout=900)
        assert result.returncode==0,'Pilot failed; preserve private logs and stop GPU'
        print((workspace/'PILOT_RECEIPT.json').read_text())
    finally:print('Save outputs, stop GPU immediately, select None. Download and verify after GPU is off.')
elif CONFIG['action']!='preflight':raise ValueError('Unsupported action; no fallback')
""".replace('BOOTSTRAP',repr(bootstrap)).replace('CHILD',repr(child))
nb['cells'][3]['source']=gpu.splitlines(keepends=True)
# Keep the independent CPU runtime checksum cell; training dependencies bootstrap only inside the gated GPU block.
nb['cells']=nb['cells'][:5]
for c in nb['cells']:
    if c['cell_type']=='code':ast.parse(''.join(c['source']))
nb['metadata']['kaggle']={'accelerator':'none','isInternetEnabled':False}
target=root/'ZAIC_NEXT_CORRECTION_REPLAY_PILOT.ipynb'
target.write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8')
report={'pilot_gate_tests':8,'local_tests_passed':True,'notebook_code_ast_valid':True,'notebook_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'default_action':'preflight','default_accelerator':'none','training_enabled':False,'blocked_by':['verified_correction_dataset_missing','next_training_budget_not_yet_accepted'],'gpu_execution_of_this_wrapper':False,'gpu_model_runner_evidence':'relearning_v3 GPU smoke version 366867761'}
(root/'PREPARATION_STATUS.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
