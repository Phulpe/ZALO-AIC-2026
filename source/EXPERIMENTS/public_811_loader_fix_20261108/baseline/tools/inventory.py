"""Project metadata inventory for handoff; excludes raw business contents."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import argparse,hashlib,json,os

def inventory(root):
 out=[]
 for folder,dirs,files in os.walk(root):
  dirs[:]=[d for d in dirs if not d.startswith('.') and d!='__pycache__']
  for name in sorted(files):
   f=Path(folder)/name
   if name.startswith('.'):continue
   try:stat=f.stat()
   except OSError:continue
   item={'path':str(f.relative_to(root)),'bytes':stat.st_size,'mtime_utc':datetime.fromtimestamp(stat.st_mtime,timezone.utc).isoformat()}
   if f.suffix=='.py':
    content=f.read_bytes();item['sha256']=hashlib.sha256(content).hexdigest()
    import re
    v=re.search(r'^VERSION\s*=\s*[\"\']([^\"\']+)',content.decode(errors='replace'),re.M)
    if v:item['version']=v.group(1)
   out.append(item)
 return out

def main():
 pa=argparse.ArgumentParser();pa.add_argument('--project',required=True);a=pa.parse_args();p=Path(a.project);root=p.parent
 inv=inventory(root)
 aggregate={}
 for name in sorted({r['path'].split('/')[0] for r in inv}):
  rs=[r for r in inv if r['path'].split('/')[0]==name];aggregate[name]={'files':len(rs),'bytes':sum(r['bytes'] for r in rs)}
 datasets={}
 for name in ['train','public_test']:
  d=root.parent/name
  datasets[name]={'exists':d.is_dir(),'questions_count':sum(bool(l.strip()) for l in (d/'questions.jsonl').read_text(encoding='utf-8-sig').splitlines()) if (d/'questions.jsonl').is_file() else None,'reference_answers_present':(d/'answers.jsonl').is_file(),'business_directories':len([f for f in (d/'data').iterdir() if f.is_dir()]) if (d/'data').is_dir() else None,'file_extensions':dict(Counter(f.suffix for f in d.rglob('*') if f.is_file() and not any(x.startswith('.') for x in f.relative_to(d).parts)))}
 root_assets={}
 for entry in sorted(root.parent.iterdir()):
  if entry.name.startswith('.') or entry.name in {'datamate_v1','train','public_test'}:continue
  extra=inventory(entry) if entry.is_dir() else [{'path':entry.name,'bytes':entry.stat().st_size}]
  root_assets[entry.name]={'files':len(extra),'bytes':sum(i['bytes'] for i in extra)}
 keys=['model_lock.json','versions.json','smoke_model.json','data/manifest.json','candidate/metadata.json','training_progress.json','training_complete.json','calibration_base/evaluation.json','calibration_candidate/evaluation.json','audit_selected/evaluation.json','selection.json','UPGRADE_COMPLETED.json','runtime_comparison.json','runtime_selection.json','PUBLIC_9B_VERIFIED.json']
 status={}
 for name in keys:
  f=p/name
  if f.is_file():status[name]=json.loads(f.read_text())
 (p/'project_inventory.json').write_text(json.dumps({'generated_utc':datetime.now(timezone.utc).isoformat(),'root':str(root),'root_assets':root_assets,'top_level':aggregate,'datasets':datasets,'files':inv},ensure_ascii=False,indent=2))
 summary={'generated_utc':datetime.now(timezone.utc).isoformat(),'datasets':datasets,'root_assets':root_assets,'top_level':aggregate,'status':status,'file_count':len(inv)}
 (p/'handoff_state.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
 print('HANDOFF_STATE_READY:',json.dumps({'generated_utc':summary['generated_utc'],'file_count':len(inv),'upgrade_complete':'UPGRADE_COMPLETED.json' in status},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
