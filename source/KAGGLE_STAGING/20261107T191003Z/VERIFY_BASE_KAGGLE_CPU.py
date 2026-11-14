from pathlib import Path
import hashlib, json, struct, math
root=Path('/kaggle/input')
base_manifests=list(root.rglob('BASE_UPLOAD_MANIFEST.json'))
assert len(base_manifests)==1, 'Attach exactly one reviewed base version'
base=base_manifests[0].parent/'base'
expected=CONFIG['BASE_FILE_HASHES']
assert len(expected)==15
assert {p.name for p in base.iterdir() if p.is_file()}==set(expected)
for name,h in expected.items():
    assert input_sha(base/name)==h, 'Base checksum mismatch'
index=json.loads((base/'model.safetensors.index.json').read_text())
shards=set(index['weight_map'].values())
assert len(shards)==4
sizes={'F64':8,'F32':4,'F16':2,'BF16':2,'I64':8,'I32':4,'I16':2,'I8':1,'U8':1,'BOOL':1,'F8_E4M3':1,'F8_E5M2':1}
tensor_bytes=0
for name in shards:
    p=base/name
    with p.open('rb') as f:
        header_size=struct.unpack('<Q',f.read(8))[0]
        assert 0<header_size<100*1024*1024
        header=json.loads(f.read(header_size))
    tensor_records={k:v for k,v in header.items() if k!='__metadata__'}
    assert set(tensor_records)=={k for k,v in index['weight_map'].items() if v==name}
    intervals=[]
    for record in tensor_records.values():
        a,b=record['data_offsets']
        assert 0<=a<=b<=p.stat().st_size-8-header_size
        assert b-a==math.prod(record['shape'])*sizes[record['dtype']]
        intervals.append((a,b)); tensor_bytes+=b-a
    intervals.sort()
    assert intervals[0][0]==0
    assert all(a[1]==b[0] for a,b in zip(intervals,intervals[1:]))
    assert intervals[-1][1]==p.stat().st_size-8-header_size
assert tensor_bytes==index['metadata']['total_size']
base_receipt={'base_model':'Qwen/Qwen3.5-9B','revision':'c202236235762e1c871ad0ccb60c8ee5ba337b9a','base_files_sha256_matched':len(expected),'base_shards':len(shards),'safetensors_headers_and_index':'passed','tensor_bytes':tensor_bytes,'gpu_used':False,'training_started':False}
Path('/kaggle/working/BASE_INTEGRITY_RECEIPT.json').write_text(json.dumps(base_receipt,indent=2))
print(json.dumps(base_receipt,indent=2))
