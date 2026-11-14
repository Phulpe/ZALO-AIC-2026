"""Inspect embedded code as bytes/AST only; never execute notebook code."""
from pathlib import Path
import ast,base64,hashlib,json,zlib
root=Path('D:/ZAIC');nb=json.loads((root/'ZAIC_QLoRA_intent_v0.ipynb').read_text(encoding='utf-8'))
snapshots=[];seen=set();blobs=0
def inspect(text,cell,origin,name=None):
 if not isinstance(text,str) or len(text)>4*1024*1024:return
 if not any(s in text for s in ['CompletionTrainer','def train_candidate','TrainerCallback','trainer_full.train']):return
 try:t=ast.parse(text)
 except SyntaxError:return
 h=hashlib.sha256(text.encode()).hexdigest()
 if (cell,h) in seen:return
 seen.add((cell,h));calls=[]
 for n in ast.walk(t):
  if not isinstance(n,ast.Call):continue
  fn=ast.unparse(n.func)
  if any(s in fn for s in ['GradScaler','TrainingArguments','LoraConfig']) or fn.endswith('.train'):
   fields={}
   for k in n.keywords:
    if k.arg in ['train_dataset','eval_dataset','data_collator']:continue
    v=k.value
    if isinstance(v,ast.Constant) and isinstance(v.value,(int,float,bool,type(None))):fields[k.arg]=v.value
    elif isinstance(v,ast.Constant) and isinstance(v.value,str) and len(v.value)<80 and '\n' not in v.value:fields[k.arg]=v.value
    elif isinstance(v,(ast.Name,ast.Attribute)):fields[k.arg]=ast.unparse(v)
   calls.append({'call':fn,'kwargs':fields})
 snapshots.append({'cell_index':cell,'origin':origin,'filename':name,'source_sha256':h,'bytes':len(text.encode()),'calls':calls})
for i,c in enumerate(nb['cells']):
 if c.get('cell_type')!='code':continue
 s=''.join(c.get('source',[]))
 try:t=ast.parse('\n'.join(l for l in s.splitlines() if not l.lstrip().startswith(('!','%'))))
 except SyntaxError:continue
 inspect(s,i,'cell source')
 for n in ast.walk(t):
  if isinstance(n,ast.Constant) and isinstance(n.value,str) and len(n.value)>500:
   text=n.value;inspect(text,i,'literal embedded source')
   if len(text)>8*1024*1024:continue
   try:
    raw=base64.b64decode(text,validate=True)
    if len(raw)>4*1024*1024:continue
    decoder=zlib.decompressobj();raw=decoder.decompress(raw,4*1024*1024+1)
    if len(raw)>4*1024*1024 or decoder.unconsumed_tail:continue
    obj=json.loads(raw);blobs+=1
   except Exception:continue
   if isinstance(obj,dict):
    for name,content in obj.items():
     if isinstance(name,str) and name.endswith('.py'):inspect(content,i,'decoded JSON code bundle',name)
result={'safe_decode_only':True,'executed_code':False,'decoded_json_bundles':blobs,'snapshots':snapshots,
 'limit':'Source snapshots in mutable notebook are evidence of code availability, not a complete ordered execution trace.'}
(root/'TRAINING_AUDIT_EVIDENCE/EMBEDDED_CODE_SNAPSHOTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
