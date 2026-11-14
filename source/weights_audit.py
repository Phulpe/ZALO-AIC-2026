from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
import hashlib, json, math, re, struct, zipfile

ROOT=Path('D:/ZAIC')
PRIVATE=ROOT/'PRIVATE_DATA'
PROJECT=ROOT/'drive-download-20241211T124801Z-001/ZAIC_Local_GUI/CODEX_PROJECT'
EXPECTED='e7dfa635beccce128c99f9d3fd993d33dbb4c88e73e3ddeccbc2bc915eec4a4d'
def read(p):
    if p.stat().st_size>32*1024*1024: raise ValueError('JSON exceeds safety limit')
    return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def unique(pairs):
    d={}
    for k,v in pairs:
        if k in d:raise ValueError('Duplicate JSON key')
        d[k]=v
    return d
def header(p):
    with p.open('rb') as f:
        prefix=f.read(8)
        if len(prefix)!=8:raise ValueError('Truncated length')
        n=struct.unpack('<Q',prefix)[0]
        if n<2 or n>100*1024*1024 or n+8>p.stat().st_size:raise ValueError('Invalid header size')
        d=json.loads(f.read(n),object_pairs_hook=unique)
    tensors={k:v for k,v in d.items() if k!='__metadata__'}
    spans=[]; data_bytes=p.stat().st_size-8-n
    sizes={'BOOL':1,'U8':1,'I8':1,'I16':2,'U16':2,'F16':2,'BF16':2,'I32':4,'U32':4,'F32':4,'I64':8,'U64':8,'F64':8,'F8_E4M3':1,'F8_E5M2':1}
    for k,v in tensors.items():
        shape=v['shape'];a,b=v['data_offsets'];dtype=v['dtype']
        if not all(type(s)==int and s>=0 for s in shape):raise ValueError('Invalid shape')
        if not 0<=a<=b<=data_bytes:raise ValueError('Invalid data offsets')
        if dtype not in sizes or math.prod(shape)*sizes[dtype]!=b-a:raise ValueError('Shape/dtype byte mismatch')
        spans.append((a,b))
    end=0
    for a,b in sorted(spans):
        if a!=end:raise ValueError('Overlap or hole')
        end=b
    if end!=data_bytes:raise ValueError('Truncated or trailing payload')
    return tensors,{'valid':True,'tensor_count':len(tensors),'header_bytes':n,'tensor_data_bytes':data_bytes,
                    'dtypes':dict(Counter(v['dtype'] for v in tensors.values()))}

report={'checked_utc':datetime.now(timezone.utc).isoformat(),'workspace':str(ROOT),'source_project':str(PROJECT),
        'scope':[str(ROOT)],'full_source_inventory_available':False,'raw_data_read':False,
        'gpu_or_pickle_loading':False,'files':[],'adapters':[],'base_sets':[],'archives':[],
        'partial_downloads':[],'read_errors':[],'weights_pending':str(PRIVATE/'weights_pending')}
allfiles=[]
for p in ROOT.rglob('*'):
    if p.is_file() and not p.is_symlink() and '.venv' not in p.parts and '.git' not in p.parts:allfiles.append(p)
headers={};records={}
names={'adapter_config.json','config.json','tokenizer.json','tokenizer_config.json','tokenizer.model','vocab.json','merges.txt',
       'optimizer.pt','scheduler.pt','scaler.pt','rng_state.pth','trainer_state.json','training_args.bin','training_complete.json',
       'metadata.json','model_lock.json','base_model_lock.json','generation_config.json','chat_template.jinja'}
for p in allfiles:
    if p.suffix=='.safetensors' or p.name in names or p.name.endswith('.index.json'):
        st=p.stat();r={'path':str(p),'bytes':st.st_size,'sha256':sha(p),'empty':st.st_size==0}
        after=p.stat();r['stable_during_hash']=st.st_size==after.st_size and st.st_mtime_ns==after.st_mtime_ns
        if p.suffix=='.safetensors':
            try:headers[str(p)],r['safetensors_header']=header(p)
            except Exception as e:r['safetensors_header']={'valid':False,'error':str(e)}
        records[str(p)]=r;report['files'].append(r)
    if p.suffix.lower() in {'.crdownload','.part','.partial','.tmp','.lock'}:report['partial_downloads'].append({'path':str(p),'bytes':p.stat().st_size})
    if p.suffix.lower()=='.zip':
        r={'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
        try:
            with zipfile.ZipFile(p) as z:
                items=[{'name':i.filename,'bytes':i.file_size,'crc32':f'{i.CRC:08x}'} for i in z.infolist() if re.search(r'\.safetensors$|checkpoint-|adapter_config|model.*index\.json',i.filename)]
                r.update(weight_entries=items,crc_payload_tested=False)
        except Exception as e:r['error']=type(e).__name__
        report['archives'].append(r)
print('HASHING_COMPLETE',len(records),flush=True)

# Content membership, not filename similarity, identifies renamed shards.
for index in [p for p in allfiles if p.name=='model.safetensors.index.json']:
    try:
        idx=read(index);mapping=idx['weight_map'];config=read(index.parent/'config.json')
        model={'model_type':config.get('model_type'),'architectures':config.get('architectures'),
               'text_hidden_size':config.get('text_config',config).get('hidden_size'),
               'text_vocab_size':config.get('text_config',config).get('vocab_size')}
        source_path=PRIVATE/'zaic/datamate_v1/qwen35_9b/training_code/base_assets_manifest.json'
        source=read(source_path) if source_path.is_file() and config.get('model_type')=='qwen3_5' else {}
        base={'directory':str(index.parent),'index':str(index),'model':model,'shards':[],
              'expected_tensor_count':len(mapping),'index_total_size':idx.get('metadata',{}).get('total_size')}
        selected={};total=0
        for name in sorted(set(mapping.values())):
            expected={k for k,v in mapping.items() if v==name}
            matches=[Path(k) for k,v in headers.items() if set(v)==expected]
            canonical=index.parent/name
            row={'index_filename':name,'expected_tensors':len(expected),'canonical_exists':canonical.is_file(),
                 'content_matches':[str(p) for p in matches],'matches':[]}
            for p in matches:
                rec=records[str(p)];src=source.get(name,{})
                source_ok=bool(src) and src.get('bytes')==rec['bytes'] and src.get('sha256')==rec['sha256']
                row['matches'].append({'path':str(p),'bytes':rec['bytes'],'sha256':rec['sha256'],
                     'filename_matches_index':p.name==name,'trusted_handoff_manifest_match':source_ok})
            if matches:
                selected[name]=str(matches[0]);total+=records[str(matches[0])]['safetensors_header']['tensor_data_bytes']
            row['status']='present_at_expected_path' if canonical.is_file() and str(canonical) in headers and set(headers[str(canonical)])==expected else ('downloaded_but_not_at_index_path' if matches else 'missing_or_unidentified')
            base['shards'].append(row)
        base['all_shard_contents_found']=len(selected)==len(base['shards'])
        base['tensor_payload_total_bytes']=total
        base['total_size_matches_index']=total==base['index_total_size']
        base['loadable_at_current_path']=all(r['status']=='present_at_expected_path' for r in base['shards'])
        base['config_source_hash_matches']=bool(source.get('config.json')) and records[str(index.parent/'config.json')]['sha256']==source['config.json']['sha256']
        base['source_manifest']=str(source_path) if source else None
        base['unmapped_safetensors_in_directory']=[str(p) for p in index.parent.glob('*.safetensors') if p.name not in set(mapping.values())]
        report['base_sets'].append(base)
    except Exception as e:report['read_errors'].append({'path':str(index),'error':str(e)})

for cfgpath in [p for p in allfiles if p.name=='adapter_config.json']:
    folder=cfgpath.parent;weight=folder/'adapter_model.safetensors'
    try:
        cfg=read(cfgpath);base=cfg.get('base_model_name_or_path');cfg_revision=cfg.get('revision')
        # Base path in config must be supported by its own config/lock, not folder naming alone.
        localbase=None
        if isinstance(base,str) and '/zaic/' in base:localbase=PRIVATE/'zaic'/base.split('/zaic/',1)[1]
        basecfg=read(localbase/'config.json') if localbase and (localbase/'config.json').is_file() else {}
        evidence={'config_base':base,'config_revision':cfg_revision,'resolved_base_path':str(localbase) if localbase else None,
                  'resolved_model_type':basecfg.get('model_type'),'rank':cfg.get('r'),'alpha':cfg.get('lora_alpha')}
        family='Qwen3.5-9B' if basecfg.get('model_type')=='qwen3_5' and basecfg.get('text_config',{}).get('hidden_size')==4096 else (base if isinstance(base,str) and base.startswith('Qwen/') else 'chưa xác định từ config')
        statefile=folder/'trainer_state.json';state=read(statefile) if statefile.is_file() else {}
        missing=[n for n in ['optimizer.pt','scheduler.pt','scaler.pt','rng_state.pth','trainer_state.json'] if not (folder/n).is_file() or (folder/n).stat().st_size==0]
        valid=str(weight) in headers;rec=records.get(str(weight),{})
        row={'path':str(folder),'model':family,'evidence':evidence,'checkpoint_step':state.get('global_step'),
             'checkpoint_name_step':int(folder.name.split('-')[-1]) if re.fullmatch(r'checkpoint-\d+',folder.name) else None,
             'adapter_bytes':rec.get('bytes'),'adapter_sha256':rec.get('sha256'),'safetensors_valid':valid,
             'adapter_files_structurally_complete':valid and cfgpath.stat().st_size>0,
             'missing_resume_files':missing,'resume_files_present':valid and not missing,
             'resume_execution_verified':False,'inference_execution_verified':False,
             'tokenizer_files':[p.name for p in folder.iterdir() if p.name.startswith('tokenizer') or p.name in {'vocab.json','merges.txt'}],
             'source_integrity':'known_9b_hash_match' if rec.get('sha256')==EXPECTED else 'local_hash_only_no_source_hash'}
        report['adapters'].append(row)
    except Exception as e:report['read_errors'].append({'path':str(cfgpath),'error':str(e)})
known=['datamate_v1/qwen35_9b/candidate/final']+['datamate_v1/qwen35_9b/candidate/checkpoints/checkpoint-'+str(s) for s in [40,45,48]]+['adapter_intent_v0_run1_20261106_044117/final']+['adapter_intent_v0_run1_20261106_044117/checkpoints/checkpoint-'+str(s) for s in [125,150]]+['adapter_intent_v0_20steps_20261106_043640','datamate_v1/candidate_20261106_090435/final']+['datamate_v1/candidate_20261106_090435/checkpoints/checkpoint-'+str(s) for s in [80,83]]
report['known_expected_adapters']=[{'path':str(PRIVATE/'zaic'/n),'found':any(r['path']==str(PRIVATE/'zaic'/n) for r in report['adapters'])} for n in known]
report['unclassified_safetensors']=[p for p in headers if Path(p).name!='adapter_model.safetensors' and not any(p in s['content_matches'] for b in report['base_sets'] for s in b['shards'])]
report['checkpoint_directories']=[{'path':str(p),'adapter_config_present':(p/'adapter_config.json').is_file()} for p in ROOT.rglob('checkpoint-*') if p.is_dir()]
report['final_9b']=next((r for r in report['adapters'] if r['path']==str(PRIVATE/'zaic/datamate_v1/qwen35_9b/candidate/final')),None)
report['full_preservation_verified']=False
lines=['# Kiểm kê bảo toàn trọng số ZAIC — 08/11/2026','',
 'Phạm vi: D:\\ZAIC, gồm PRIVATE_DATA/weights_pending và project mã đã chuyển vào thư mục giải nén. Không sửa trọng số, không unpickle, không GPU, không upload.',
 '', 'SHA256 tính trên toàn bộ bytes; header safetensors kiểm length, JSON, shape/dtype, offsets, không chồng lấn/lỗ hổng và kích thước payload. Không chứng nhận giá trị tensor hay tương thích runtime.',
 '', '## Adapter và checkpoint','', '| Thư mục | Base từ config/bằng chứng | Step | Bytes adapter | Cấu trúc adapter | Bộ trạng thái resume |','|---|---|---:|---:|---|---|']
for r in report['adapters']:
    lines.append(f"| {r['path']} | {r['model']} | {r['checkpoint_step']} | {r['adapter_bytes']} | {'đạt' if r['adapter_files_structurally_complete'] else 'thiếu/sai'} | {'đủ tệp, chưa chạy xác minh' if r['resume_files_present'] else 'thiếu: '+', '.join(r['missing_resume_files'])} |")
lines+=['','## Bộ base và weights_pending','']
for b in report['base_sets']:
    lines += [f"### {b['directory']}",'',f"Config: {b['model']}. Đủ nội dung shard: {b['all_shard_contents_found']}; khớp tổng bytes tensor index: {b['total_size_matches_index']}; nạp được theo đường dẫn hiện tại: {b['loadable_at_current_path']}.",'', '| Tên index yêu cầu | Tệp khớp header | SHA256 đối chứng nguồn | Trạng thái |','|---|---|---|---|']
    for s in b['shards']:
        lines.append(f"| {s['index_filename']} | {'; '.join(s['content_matches']) or 'THIẾU'} | {'; '.join(str(x['trusted_handoff_manifest_match']) for x in s['matches']) or 'chưa có'} | {s['status']} |")
lines+=['','## Giới hạn và bổ sung','',
 'Adapter 9B cuối: '+json.dumps(report['final_9b'],ensure_ascii=False),
 '', 'Không kết luận đã bảo toàn đầy đủ: chưa có danh sách nguồn đầy đủ mọi lần huấn luyện. Các hash không có đối chứng nguồn chỉ là dấu vân tay cục bộ.',
 'Base 9B nếu đủ nội dung nhưng sai tên/vị trí: chưa cần tải lại các shard đã khớp hash; chỉ đề xuất sắp xếp sau phê duyệt, chưa thực hiện.',
 'Base ba shard: xem dòng missing_or_unidentified để biết file cần tìm/tải bổ sung. Không trộn với bộ 9B.',
 'Đủ tệp resume không chứng minh optimizer/scaler/RNG đọc được hoặc resume đúng; chỉ kiểm bytes/hash, không thực thi pickle.',
 '', '## File chưa xác định',*report['unclassified_safetensors'],'',
 '## ZIP và file tải dở',json.dumps({'archives':report['archives'],'partials':report['partial_downloads']},ensure_ascii=False,indent=2)]
for folder in [ROOT,PROJECT]:
    (folder/'WEIGHTS_INVENTORY.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (folder/'WEIGHTS_CHECK_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'adapters':len(report['adapters']),'known':report['known_expected_adapters'],'final_9b':report['final_9b'],
 'bases':report['base_sets'],'unclassified':report['unclassified_safetensors'],'errors':report['read_errors'],
 'archives':len(report['archives']),'partials':len(report['partial_downloads'])},ensure_ascii=False,indent=2),flush=True)
