from pathlib import Path
import ast, json, re
root=Path('D:/ZAIC');p=root/'ZAIC_QLoRA_intent_v0.ipynb';n=json.loads(p.read_text(encoding='utf-8'))
out=[]
for i,c in enumerate(n['cells']):
 s=''.join(c.get('source',[]));text='\n'.join(''.join(o.get('text',[]))+'\n'+''.join(o.get('data',{}).get('text/plain',[])) for o in c.get('outputs',[]))
 facts={}
 for key in ['global_step','training_loss','train_loss','train_runtime','max_steps','epoch','learning_rate']:
  vals=re.findall(r"['\"]?"+key+r"['\"]?\s*[:=]\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",text)
  if vals:facts[key]=vals[-5:]
 paths=sorted(set(re.findall(r'/content/drive/MyDrive/zaic/[A-Za-z0-9_/.-]*(?:adapter[A-Za-z0-9_/.-]*|candidate[A-Za-z0-9_/.-]*)',s+'\n'+text)))
 env={k:v for k,v in re.findall(r'(transformers|peft|bnb|torch)\s*[=: ]+\s*(\d+\.\d+(?:\.\d+)?(?:\+cu\d+)?)',text)}
 if facts or paths or env:out.append({'cell':i,'execution_count':c.get('execution_count'),'numeric_training_facts':facts,'artifact_paths':paths,'library_versions':env})
metadata=[]
keys={'rows','seed','dataset_rows','selected_rows','available_training_rows','public_test_used','training_sha256',
      'base_model','base_revision','previous_adapter','learning_rate','max_length','skipped_overlength',
      'epochs','rank','batch_size','gradient_accumulation','trainable_parameters','total_parameters'}
for path in (root/'PRIVATE_DATA').rglob('*.json'):
 if path.name not in ['run_metadata.json','summary.json','evaluation.json','manifest.json','all_results.json']:continue
 if path.stat().st_size>2*1024*1024:continue
 try:x=json.loads(path.read_text(encoding='utf-8'))
 except Exception:continue
 if not isinstance(x,dict):continue
 result={k:v for k,v in x.items() if k in keys and isinstance(v,(str,int,float,bool,type(None))) and (not isinstance(v,str) or len(v)<200)}
 metrics={k:v for k,v in x.items() if re.fullmatch(r'(n|N|correct|wrong|abstain|PA|answered_accuracy|coverage|train_loss|train_runtime|epoch)',k) and isinstance(v,(int,float))}
 if result or metrics:metadata.append({'path':str(path),'metadata':result,'metrics':metrics,'keys':list(x)})
result={'notebook_facts':out,'metadata':metadata}
(root/'TRAINING_AUDIT_EVIDENCE/EXTRA_PROBE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
