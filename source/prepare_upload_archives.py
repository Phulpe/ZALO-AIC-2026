"""Local GUI-upload archives; stream hashes, preserve source artifacts."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,zipfile
ROOT=Path('D:/ZAIC');STAGE=Path(json.loads((ROOT/'KAGGLE_UPLOAD_LATEST.json').read_text())['stage'])
OUT=STAGE/'UPLOAD_ARCHIVES';OUT.mkdir(exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def package(name,entries):
 target=OUT/name;partial=target.with_suffix('.zip.partial')
 if target.exists() or partial.exists():raise FileExistsError('Preserve existing archive; do not overwrite')
 records={}
 with zipfile.ZipFile(partial,'w',zipfile.ZIP_STORED,allowZip64=True) as z:
  for arc,p,expected in entries:
   h=hashlib.sha256();before=p.stat()
   with p.open('rb') as source,z.open(arc,'w',force_zip64=True) as dest:
    for b in iter(lambda:source.read(8*1024*1024),b''):h.update(b);dest.write(b)
   after=p.stat();actual=h.hexdigest()
   if expected and actual!=expected:raise ValueError('Source changed or hash mismatch; incomplete archive preserved')
   if before.st_size!=after.st_size or before.st_mtime_ns!=after.st_mtime_ns:raise ValueError('Source changed while packaging')
   records[arc]={'bytes':after.st_size,'sha256':actual}
   print('PACKED',name,arc,flush=True)
 partial.rename(target)
 with zipfile.ZipFile(target) as z:
  assert len(z.infolist())==len(entries)
  assert sum(i.file_size for i in z.infolist())==sum(r['bytes'] for r in records.values())
 result={'path':str(target),'bytes':target.stat().st_size,'sha256':sha(target),'files':records}
 print('ARCHIVE_READY',name,flush=True)
 return result
base=STAGE/'01_BASE_QWEN35_9B'
manifest=json.loads((base/'BASE_UPLOAD_MANIFEST.json').read_text(encoding='utf-8'))['files']
entries=[('base/'+n,base/'base'/n,item['sha256']) for n,item in manifest.items()]
entries.append(('BASE_UPLOAD_MANIFEST.json',base/'BASE_UPLOAD_MANIFEST.json',None))
results=[package('ZAIC_BASE_QWEN35_9B.zip',entries)]
data=STAGE/'02_APPROVED_TRAIN_SFT';sm=json.loads((data/'SFT_UPLOAD_MANIFEST.json').read_text(encoding='utf-8'))
entries=[('data/'+n,data/'data'/n,item['sha256']) for n,item in sm['files'].items()]
entries.append(('SFT_UPLOAD_MANIFEST.json',data/'SFT_UPLOAD_MANIFEST.json',None))
results.append(package('ZAIC_TRAIN_SFT_768.zip',entries))
# Preserve learned 9B weights and full final checkpoint as a separate cloud input,
# so a later reviewed resume workflow can retain optimizer/scaler/RNG state.
Q=ROOT/'PRIVATE_DATA/zaic/datamate_v1/qwen35_9b'
prior=json.loads((ROOT/'TRAINING_AUDIT_EVIDENCE/CHECKSUM_MANIFEST.json').read_text(encoding='utf-8'))['files']
entries=[]
for directory,arcroot in [(Q/'candidate/final','final'),(Q/'candidate/checkpoints/checkpoint-48','checkpoints/checkpoint-48')]:
 for p in sorted(directory.iterdir()):
  if p.is_file():entries.append((arcroot+'/'+p.name,p,prior.get(str(p),{}).get('sha256')))
for name in ['model_lock.json','training_complete.json','versions.json']:
 entries.append((name,Q/name,None))
results.append(package('ZAIC_TRAINED_9B_FINAL_CHECKPOINT48.zip',entries))
index={'created_utc':datetime.now(timezone.utc).isoformat(),'archives':results,
 'code_archive':str(STAGE/'03_CODE_AND_NOTEBOOK/ZAIC_CODE_ONLY.zip'),
 'upload_complete':False,'training_started':False,
 'trained_9b_archive_note':'Checkpoint 48 and final adapter preserved; current prepared notebook trains fresh, does not yet implement resume.'}
(OUT/'UPLOAD_ARCHIVES_MANIFEST.json').write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf-8')
print('ALL_UPLOAD_ARCHIVES_READY',flush=True)
