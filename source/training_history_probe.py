from pathlib import Path
import ast, hashlib, json, re
ROOT=Path('D:/ZAIC');PRIVATE=ROOT/'PRIVATE_DATA'
OUT=ROOT/'TRAINING_AUDIT_EVIDENCE';OUT.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def numeric(d):
    allowed={'step','global_step','max_steps','epoch','num_train_epochs','train_loss','train_runtime','train_rows','rows','steps',
       'learning_rate','rank','batch_size','gradient_accumulation','max_length','loss','grad_norm','gradient_finite',
       'initial_loss_scale','old_adapter_reused','public_test_used','state','seed','epochs','seconds','peak_gpu_GB',
       'PA','correct','wrong','abstain','n','answered_accuracy','coverage','data_sha256','training_sha256'}
    if isinstance(d,dict):return {k:v for k,v in d.items() if k in allowed and isinstance(v,(str,int,float,bool,type(None))) and (not isinstance(v,str) or len(v)<150)}
    return {}
notebooks=[]
for p in ROOT.rglob('*.ipynb'):
    if 'KAGGLE_STAGING' in p.parts:continue
    try:nb=json.loads(p.read_text(encoding='utf-8'))
    except Exception:continue
    cells=[]
    for i,c in enumerate(nb.get('cells',[])):
        if c.get('cell_type')!='code':continue
        source=''.join(c.get('source',[]));events=[];assignments={};paths=[]
        if not re.search(r'\.train\(|save_pretrained|resume_from_checkpoint|TrainOutput',source):continue
        try:tree=ast.parse(source)
        except SyntaxError:
            try:tree=ast.parse('\n'.join(l for l in source.splitlines() if not l.lstrip().startswith(('!','%'))))
            except SyntaxError:tree=None
        if tree:
            for node in ast.walk(tree):
                if isinstance(node,ast.Call):
                    name=ast.unparse(node.func)
                    if any(x in name for x in ['.train','.save_pretrained','.from_pretrained','LoraConfig','TrainingArguments','Trainer','copytree']):
                        def val(n):
                            if isinstance(n,ast.Constant) and isinstance(n.value,(str,int,float,bool,type(None))):
                                s=n.value
                                return s if not isinstance(s,str) or (len(s)<200 and not '\n' in s) else '[redacted long literal]'
                            return ast.unparse(n)[:120] if isinstance(n,(ast.Name,ast.Attribute)) else type(n).__name__
                        events.append({'call':name,'args':[val(n) for n in node.args[:3]],'kwargs':{k.arg:val(k.value) for k in node.keywords if k.arg not in {'train_dataset','eval_dataset','data_collator'}}})
                if isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant):
                    for target in node.targets:
                        if isinstance(target,ast.Name) and re.search(r'BASE|OUT|LR|EPOCH|STEP|RANK|MAX_LEN',target.id,re.I):
                            value=node.value.value
                            if not isinstance(value,str) or len(value)<250:assignments[target.id]=value
        output='\n'.join(''.join(o.get('text',[]))+'\n'+''.join(o.get('data',{}).get('text/plain',[])) for o in c.get('outputs',[]))
        trainout=re.findall(r'TrainOutput\([^\n]{0,700}',output)
        progress=re.findall(r'\b\d+/\d+\s+\d+:\d+[^\n]{0,60}',output)
        errors=[o.get('ename') for o in c.get('outputs',[]) if o.get('output_type')=='error']
        cells.append({'cell_index':i,'execution_count':c.get('execution_count'),'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
             'events':events,'assignments':assignments,'train_outputs':trainout,'progress':progress,'error_types':errors})
    notebooks.append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size,'cell_count':len(nb.get('cells',[])),'training_cells':cells})
summaries=[]
for p in PRIVATE.rglob('*.json'):
    if p.name in ['training_complete.json','metadata.json','train_results.json','training_progress.json'] or p.name.startswith('progress_scale'):
        try:d=json.loads(p.read_text(encoding='utf-8'));summaries.append({'path':str(p),'sha256':sha(p),'summary':numeric(d)})
        except Exception:pass
logs=[]
for p in PRIVATE.rglob('*.log'):
    if not re.search('train|workflow|smoke|build',p.name,re.I):continue
    text=p.read_text(encoding='utf-8',errors='replace')
    records=[]
    for prefix,raw in re.findall(r'(TRAIN_9B_CONFIG|TRAIN_9B_COMPLETED|TRAIN_PROGRESS|MODEL9B_SMOKE_PASSED):\s*(\{[^\n]+\})',text):
        try:records.append({'event':prefix,'aggregate':numeric(json.loads(raw))})
        except Exception:pass
    logs.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'events_first_last':records[:2]+records[-2:],
        'events_count':len(records),'exception_types':sorted(set(re.findall(r'\b(?:FloatingPointError|OutOfMemoryError|AssertionError|RuntimeError|ValueError|ImportError|ModuleNotFoundError)\b',text))),
        'nonfinite_mentions':len(re.findall(r'nonfinite|non.finite|NaN|nan',text))})
report={'notebooks':notebooks,'training_metadata':summaries,'training_logs':logs}
(OUT/'LOCAL_TRAINING_PROBE.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'notebooks':notebooks,'training_metadata':summaries,'training_logs':logs},ensure_ascii=False,indent=2))
