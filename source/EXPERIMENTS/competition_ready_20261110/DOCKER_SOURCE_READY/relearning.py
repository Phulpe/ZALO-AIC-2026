"""Private runtime learning orchestration. No external services or raw-data printing."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, uuid, copy, math, re, sys, importlib.metadata

PARENT_SHA='e7dfa635beccce128c99f9d3fd993d33dbb4c88e73e3ddeccbc2bc915eec4a4d'
REVISION='c202236235762e1c871ad0ccb60c8ee5ba337b9a'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()

def atomic(p,obj):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def new_run(output_root,config,protected=()):
    root=Path(output_root).resolve()
    for path in protected:
        path=Path(path).resolve()
        if root==path or root.is_relative_to(path) or path.is_relative_to(root):
            raise ValueError('Output overlaps protected assets')
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:8]
    run=root/run_id;run.mkdir(parents=True,exist_ok=False)
    atomic(run/'run_config.json',{**config,'run_id':run_id,'config_hash':fingerprint(config)})
    source=Path(__file__).parent
    atomic(run/'source_manifest.json',{p.relative_to(source).as_posix():sha(p) for p in source.rglob('*.py') if '__pycache__' not in p.parts and 'synthetic_runs' not in p.parts})
    versions={}
    for name in ['torch','transformers','peft','accelerate','bitsandbytes','numpy','pandas']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    atomic(run/'environment.json',{'python':sys.version,'platform':sys.platform,'packages':versions})
    return run

def verify_parent(adapter,expected=PARENT_SHA):
    adapter=Path(adapter)
    if sha(adapter/'adapter_model.safetensors')!=expected:raise ValueError('Parent hash mismatch')
    c=json.loads((adapter/'adapter_config.json').read_text())
    # Verified baseline adapter records its original Colab path, not HF model ID.
    if c['base_model_name_or_path'] not in {'Qwen/Qwen3.5-9B','/content/drive/MyDrive/zaic/datamate_v1/qwen35_9b/assets/base'} or c['r']!=8:
        raise ValueError('Incompatible adapter')
    return expected

def warm_start(base,adapter,config,peft_loader=None):
    if config.get('init_mode')!='warm_start':raise ValueError('Explicit warm_start required; resume is separate')
    parent=verify_parent(adapter,config.get('parent_sha256',PARENT_SHA))
    if peft_loader is None:
        from peft import PeftModel
        peft_loader=PeftModel.from_pretrained
    # No fresh fallback and no checkpoint state deserialization.
    model=peft_loader(base,str(adapter),is_trainable=True,local_files_only=True)
    if not any(p.requires_grad for p in model.parameters()):raise ValueError('No trainable adapter parameters')
    return model,{'init_mode':'warm_start','parent_sha256':parent,'optimizer_init':'new','scheduler_init':'new'}

class LearningRound:
    """Callbacks receive allowlisted question fields, never the gold store.

    semantic_check must independently verify plan meaning against source/rulebook;
    scorer matching alone is insufficient. Callbacks run in the trusted runtime.
    """
    def __init__(self,run,config,pool,gold_reader,predict,execute,validate,semantic_check,scorer,correct=None,recover=False):
        if config.get('mode')!='learning':raise ValueError('Learning mode required')
        n=config.get('attempts_per_question',3)
        if type(n)!=int or not 1<=n<=3:raise ValueError('Retry budget must be 1..3')
        self.pool=copy.deepcopy(pool)
        ids=[q['id'] for q in pool]
        if not ids or len(ids)!=len(set(ids)):raise ValueError('Empty or duplicate learning IDs')
        allowed=set(config['learning_businesses']);blocked=set(config['calibration_businesses'])|set(config['holdout_businesses'])
        if allowed & blocked or any(q['split']!='learning' or q['business_id'] not in allowed for q in pool):raise ValueError('Wrong split')
        for q in pool:
            if set(q)-{'id','split','business_id','prompt','source_ref','as_of_date'}:raise ValueError('Question contains forbidden fields')
        self.run=Path(run);self.config=config;self.gold=gold_reader;self.predict=predict;self.execute=execute
        self.validate=validate;self.semantic=semantic_check;self.scorer=scorer;self.correct=correct
        self.ledger={'fingerprint':fingerprint({'config':config,'pool':pool}), 'questions':{i:{'status':'pending','attempts':[]} for i in ids}}
        if (self.run/'ledger.json').exists():
            if not recover:raise FileExistsError('Do not overwrite ledger; explicit recovery required')
            saved=json.loads((self.run/'ledger.json').read_text())
            if saved['fingerprint']!=self.ledger['fingerprint'] or set(saved['questions'])!=set(ids):raise ValueError('Recovery fingerprint mismatch')
            if any(row['status']=='pending' and row['attempts'] for row in saved['questions'].values()):raise ValueError('Interrupted partial attempt requires recovery audit; preserve ledger')
            self.ledger=saved
        self.save()

    def save(self):atomic(self.run/'ledger.json',self.ledger)

    def verified(self,q,plan,gold):
        self.validate(copy.deepcopy(plan))
        if not self.semantic(copy.deepcopy(q),copy.deepcopy(plan)):return False,None
        answer=self.execute(copy.deepcopy(q),copy.deepcopy(plan))
        return bool(self.scorer(answer,gold)),answer

    def process(self,stop_requested=None):
        from prediction_errors import BlindPredictionError
        replay=[];corrections=[]
        if (self.run/'verified_bank.json').exists():
            bank=json.loads((self.run/'verified_bank.json').read_text());replay=bank['replay'];corrections=bank['correction']
        for q in self.pool:
            if stop_requested is not None and stop_requested():break
            row=self.ledger['questions'][q['id']]; gold=None;first_correct=False;first_verified=False;last_correct=False
            if row['status']!='pending':continue
            for attempt in range(len(row['attempts'])+1,self.config.get('attempts_per_question',3)+1):
                rec={'attempt_no':attempt,'state':'started','error_class':None}
                row['attempts'].append(rec);self.save()
                try:
                    try:plan=self.predict(copy.deepcopy(q),attempt)
                    except (TimeoutError,MemoryError):rec['error_class']='infrastructure';raise
                    except Exception:rec['error_class']='model';raise
                    rec['plan']=copy.deepcopy(plan);rec['state']='predicted';self.save()
                    # Gold opens only after the blind prediction is persisted.
                    if gold is None:gold=self.gold(q['id'])
                    try:self.validate(copy.deepcopy(plan))
                    except Exception:rec['error_class']='model';raise
                    try:meaning=self.semantic(copy.deepcopy(q),copy.deepcopy(plan))
                    except Exception:rec['error_class']='ingestion';raise
                    rec['semantic_verified']=bool(meaning)
                    if not meaning:
                        rec['state']='invalid_plan';rec['error_class']='model'
                        continue
                    try:answer=self.execute(copy.deepcopy(q),copy.deepcopy(plan))
                    except (TimeoutError,MemoryError):rec['error_class']='infrastructure';raise
                    except Exception:rec['error_class']='executor';raise
                    rec['answer']=answer
                    last_correct=bool(self.scorer(answer,gold))
                    rec['state']='correct' if last_correct else 'wrong'
                    if not last_correct:rec['error_class']='model'
                    if attempt==1:first_correct=last_correct;first_verified=last_correct and meaning
                    if last_correct and meaning:
                        replay.append({'id':q['id'],'prompt':q['prompt'],'response':json.dumps(plan,ensure_ascii=False),'business_id':q['business_id'],'kind':'replay','parent_sha256':self.config['parent_sha256'],'target_verified':True,'split':'learning'})
                        break
                except Exception as exc:
                    rec['state']='error';rec['error_class']=rec['error_class'] or 'infrastructure';rec['exception_type']=type(exc).__name__
                    if isinstance(exc,BlindPredictionError):rec['prediction_text']=exc.prediction_text
                finally:self.save()
            row.update(first_correct=first_correct,first_verified=first_verified,last_correct=last_correct,status='resolved' if replay and replay[-1]['id']==q['id'] else 'unresolved',next_action='retain_and_diagnose')
            # Invalid JSON is still a real blind prediction. It must be saved
            # before gold opens; transport/OOM failures without output do not qualify.
            if gold is None and any('prediction_text' in a for a in row['attempts']):
                gold=self.gold(q['id'])
            if row['status']=='unresolved' and self.correct is not None and gold is not None:
                try:
                    plan=self.correct(copy.deepcopy(q),copy.deepcopy(row),copy.deepcopy(gold))
                    ok,answer=self.verified(q,plan,gold)
                    if ok:
                        corrections.append({'id':q['id'],'prompt':q['prompt'],'response':json.dumps(plan,ensure_ascii=False),'business_id':q['business_id'],'kind':'correction','parent_sha256':self.config['parent_sha256'],'target_verified':True,'split':'learning'})
                        row.update(status='correction_verified',corrected_plan=plan,next_action='train_and_recheck')
                except Exception as exc:row['correction_error_type']=type(exc).__name__
            atomic(self.run/'verified_bank.json',{'replay':replay,'correction':corrections})
            self.save()
        rows=list(self.ledger['questions'].values())
        attempted=[r for r in rows if any(a.get('state') in {'predicted','correct','wrong','invalid_plan','error'} for a in r['attempts'])]
        needed=sum(not r.get('first_verified',False) for r in attempted)
        metrics={'expected':len(rows),'attempted':len(attempted),'attempt_coverage':len(attempted)/len(rows),'first_pass_correct':sum(r.get('first_correct',False) for r in attempted),'correction_needed':needed,'verified_corrections':len(corrections),'verified_correction_coverage':len(corrections)/needed if needed else None,'unresolved':sum(r['status']=='unresolved' for r in rows),'pending':len(rows)-len(attempted)}
        assert set(self.ledger['questions'])=={q['id'] for q in self.pool}
        atomic(self.run/'metrics.json',metrics)
        atomic(self.run/'verified_bank.json',{'replay':replay,'correction':corrections})
        return metrics,replay,corrections

def mixture(replay,correction,learning_businesses):
    """Balanced without oversampling; retain unused verified targets in bank."""
    all_rows=replay+correction
    if any(r.get('target_verified') is not True or r.get('split')!='learning' or r['business_id'] not in learning_businesses for r in all_rows):raise ValueError('Unverified or wrong-split target')
    if len({r['id'] for r in all_rows})!=len(all_rows):raise ValueError('Duplicate target ID')
    n=min(len(replay),len(correction));out=[]
    for a,b in zip(replay[:n],correction[:n]):out.extend([a,b])
    if not n:raise ValueError('Need both verified correction and replay')
    return out

def regressions(before,after):
    if set(before)!=set(after):raise ValueError('Missing regression IDs')
    return sorted(k for k in before if before[k] is True and after[k] is not True)

def seal(run):
    run=Path(run);files={}
    for p in sorted(run.rglob('*')):
        if p.is_symlink():raise ValueError('Symlink in export')
        if p.is_file() and p!=run/'ARTIFACT_MANIFEST.json':files[p.relative_to(run).as_posix()]={'bytes':p.stat().st_size,'sha256':sha(p)}
    atomic(run/'ARTIFACT_MANIFEST.json',{'files':files})
    return sha(run/'ARTIFACT_MANIFEST.json')

def verify_export(run):
    run=Path(run).resolve();m=json.loads((run/'ARTIFACT_MANIFEST.json').read_text())
    if any(p.is_symlink() for p in run.rglob('*')):raise ValueError('Symlink in export')
    actual={p.relative_to(run).as_posix() for p in run.rglob('*') if p.is_file() and p!=run/'ARTIFACT_MANIFEST.json'}
    if actual!=set(m['files']):raise ValueError('Missing/extra artifact')
    for name,item in m['files'].items():
        p=(run/name).resolve()
        if not p.is_relative_to(run) or p.is_symlink() or p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:raise ValueError('Artifact mismatch')
    return sha(run/'ARTIFACT_MANIFEST.json')

def track(path,run,action,evidence):
    """Separate pointers; promotion requires comparable calibration, sync saved-version receipt."""
    path=Path(path);state=json.loads(path.read_text()) if path.exists() else {'LATEST_CHECKPOINT':None,'BEST_MODEL':None,'LATEST_SYNCED':None}
    manifest=verify_export(run);record={'run':str(run),'manifest_sha256':manifest}
    if action=='checkpoint':
        required=['adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json']
        if not all((Path(run)/'checkpoint'/n).is_file() for n in required):raise ValueError('Incomplete checkpoint')
        state['LATEST_CHECKPOINT']=record
    elif action=='promote':
        for key in ['candidate_metric','minimum_metric','regression_count','max_regressions']:
            if type(evidence.get(key)) not in {int,float} or not math.isfinite(evidence[key]):return False
        if not evidence.get('calibration_manifest_sha256') or not evidence.get('gates_passed') or not evidence.get('candidate_metric',-math.inf)>=evidence.get('minimum_metric',math.inf) or evidence.get('regression_count',math.inf)>evidence.get('max_regressions',0):return False
        state['BEST_MODEL']={**record,'evaluation':evidence}
    elif action=='sync':
        if not evidence.get('saved_version_completed') or not re.fullmatch(r'https://www\.kaggle\.com/code/[\w-]+/[\w-]+\?scriptVersionId=\d+',evidence.get('saved_version_url','')) or evidence.get('remote_manifest_sha256')!=manifest:raise ValueError('Unverified saved version')
        state['LATEST_SYNCED']={**record,'version':evidence['saved_version_url']}
    else:raise ValueError('Unknown state transition')
    atomic(path,state);return True
