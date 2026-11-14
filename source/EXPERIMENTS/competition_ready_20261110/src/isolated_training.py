"""One optimizer child, waited to exit before any parent evaluation.

CPU orchestration tests prove ordering/timeout/preservation, not GPU VRAM parity.
"""
from pathlib import Path
import sys,json,os,subprocess,time,math,tempfile
from relearning import atomic,sha,verify_export

class ChildStageError(RuntimeError):pass

def run_stage(command,folder,timeout,env=None):
    if isinstance(timeout,bool) or not isinstance(timeout,(int,float)) or not math.isfinite(timeout) or timeout<=0:raise ValueError('Finite child timeout required')
    folder=Path(folder);folder.mkdir(exist_ok=False,parents=True);start=time.monotonic()
    with (folder/'PRIVATE_CHILD.log').open('x',encoding='utf-8') as log:
        try:r=subprocess.run(list(command),stdout=log,stderr=subprocess.STDOUT,timeout=timeout,env=env,check=False,shell=False)
        except subprocess.TimeoutExpired:
            atomic(folder/'STAGE_RESULT.json',{'state':'timeout','child_waited_after_kill':True,'elapsed_seconds':time.monotonic()-start,'outputs_preserved':True});raise ChildStageError('Child stage timed out; artifacts preserved') from None
    result={'state':'completed' if r.returncode==0 else 'failed','returncode':r.returncode,'elapsed_seconds':time.monotonic()-start,'outputs_preserved':True};atomic(folder/'STAGE_RESULT.json',result)
    if r.returncode:raise ChildStageError('Child stage failed; inspect private log')
    return result

def train_verified_isolated(config,rows,evidence):
    if config.get('mode')!='learning' or config.get('init_mode')!='warm_start':raise ValueError('Explicit warm-start only; no fallback')
    if config.get('collect_learning_probe'):raise ValueError('Collection belongs to the prior phase')
    if config.get('user_authorized') is not True or config.get('budget_approved') is not True:raise ValueError('Reviewed optimizer budget required')
    seconds=config.get('max_seconds');timeout=config.get('child_timeout_seconds')
    if type(seconds) not in {int,float} or not math.isfinite(seconds) or seconds<=0:raise ValueError('Optimizer budget required')
    if type(timeout) not in {int,float} or not seconds<timeout<=seconds+300:raise ValueError('Bounded child load/train/save budget required')
    output=Path(config['output_root']).resolve();parent=output.parent
    for item in [config['base'],config['adapter'],str(Path(__file__).parent/'baseline')]:
        protected=Path(item).resolve()
        if parent==protected or parent.is_relative_to(protected) or protected.is_relative_to(parent):raise ValueError('Request/output overlaps protected assets')
    parent.mkdir(parents=True,exist_ok=True);request_dir=Path(tempfile.mkdtemp(prefix='optimizer_request_',dir=parent));request=request_dir/'PRIVATE_REQUEST.json';receipt=request_dir/'CHILD_RECEIPT.json'
    atomic(request,{'config':config,'rows':rows,'evidence':evidence,'receipt':str(receipt)});request_hash=sha(request)
    stage=run_stage([sys.executable,'-X','utf8',str(Path(__file__).resolve()),str(request)],request_dir/'process',timeout,{**os.environ,'CUDA_VISIBLE_DEVICES':'0'})
    if sha(request)!=request_hash:raise ValueError('Training request changed')
    result=json.loads(receipt.read_text(encoding='utf-8'));run=Path(result['run_path']).resolve()
    if not run.is_relative_to(output) or run==output or result['parent_sha256']!=config['parent_sha256']:raise ValueError('Wrong child run/parent')
    if verify_export(run)!=result['manifest_sha256']:raise ValueError('Child checkpoint checksum mismatch')
    training=json.loads((run/'training_result.json').read_text(encoding='utf-8'))
    if training['state']!='completed' or training['init_mode']!='warm_start':raise ValueError('Incomplete training; do not evaluate')
    atomic(request_dir/'PROCESS_ISOLATION.json',{'state':'completed','parent_pid':os.getpid(),'child_pid':result['child_pid'],'child_exited_before_return':True,'run_manifest_sha256':result['manifest_sha256'],'request_sha256':request_hash,'optimizer_rounds':1})
    return run

def child(path):
    request=json.loads(Path(path).read_text(encoding='utf-8'));c=request['config']
    from quality_train import train_verified
    run=Path(train_verified(c,request['rows'],request['evidence']))
    atomic(request['receipt'],{'run_path':str(run),'manifest_sha256':verify_export(run),'parent_sha256':c['parent_sha256'],'child_pid':os.getpid()})

if __name__=='__main__':child(sys.argv[1])
