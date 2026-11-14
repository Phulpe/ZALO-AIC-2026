from pathlib import Path
import json
root=Path('D:/ZAIC');r=json.loads((root/'WEIGHTS_INVENTORY.json').read_text(encoding='utf-8'))
source=root/'PRIVATE_DATA/zaic/datamate_v1/offline_export/manifest.json'
manifest=json.loads(source.read_text(encoding='utf-8'))['files']
for b in r['base_sets']:
    if b['identified_model']!='Qwen/Qwen3-4B-Instruct-2507':continue
    b['source_manifest']=str(source)
    for s in b['shards']:
        exp=manifest.get('assets/base/'+s['index_filename'],{})
        for m in s['matches']:
            m['trusted_handoff_manifest_match']=m['sha256']==exp.get('sha256') and m['bytes']==exp.get('bytes')
    b['metadata_total_size_discrepancy_bytes']=b['index_total_size']-b['tensor_payload_total_bytes']
    b['integrity_note']='All three shard byte sizes and SHA256 match offline_export manifest; index total_size disagrees with summed valid tensor payload by 655360 bytes. Cause unverified; no data/index edits made.'
for f in r['files']:
    f['source_manifest_matches']=[{'manifest':str(source),'entry':name} for name,v in manifest.items() if isinstance(v,dict) and v.get('sha256')==f['sha256'] and v.get('bytes')==f['bytes']]
for a in r['adapters']:
    matches=[name for name,v in manifest.items() if isinstance(v,dict) and v.get('sha256')==a['adapter_sha256'] and v.get('bytes')==a['adapter_bytes']]
    if matches:a['source_integrity']='matches_offline_export_manifest';a['source_manifest_matches']=matches
r['lock_files']=[x for x in r['partial_downloads'] if x['path'].endswith('.lock')]
r['partial_downloads']=[x for x in r['partial_downloads'] if not x['path'].endswith('.lock')]
text=(root/'WEIGHTS_CHECK_REPORT.md').read_text(encoding='utf-8')
text=text.replace('Bộ 4B được nhận diện bởi config Qwen3ForCausalLM, cấu trúc tensor theo index và model lock Qwen/Qwen3-4B-Instruct-2507. Hash shard 4B chỉ có đối chứng cục bộ, chưa có manifest hash nguồn đầy đủ.',
 'Bộ 4B được nhận diện bởi config Qwen3ForCausalLM, cấu trúc tensor theo index và model lock Qwen/Qwen3-4B-Instruct-2507. Đã tìm thấy offline_export/manifest.json và xác nhận SHA256/kích thước cả 3 shard đều khớp; xem bất nhất metadata dưới đây.')
text=text.replace('Cần bổ sung bằng chứng: manifest nguồn cho các adapter 4B/checkpoint, optimizer/RNG và shard 4B; danh sách đầy đủ mọi run trên Drive để xác nhận toàn bộ lịch sử.',
 'Cần bổ sung bằng chứng: hash nguồn cho checkpoint/optimizer/RNG và các adapter chưa có đối chứng trong JSON; danh sách đầy đủ mọi run trên Drive để xác nhận toàn bộ lịch sử. Không cần tải lại các shard đã khớp manifest.')
text+='''
## Kết quả đối chiếu cuối cùng với manifest 4B

- Cả 3 shard Qwen3-4B-Instruct-2507 khớp kích thước và SHA256 trong offline_export/manifest.json; tên tensor khớp index và mọi header/payload boundary đều hợp lệ.
- Tuy nhiên index ghi total_size=8.045.591.552, còn tổng payload tensor thực tế=8.044.936.192 bytes: lệch **655.360 bytes**. Đây là bất nhất metadata đã ghi nhận; chưa xác minh nguyên nhân. Không có bằng chứng từ việc này rằng file tải bị cắt, vì cả 3 hash khớp manifest và header không bị cắt. Chưa chứng nhận loadable/runtime, chưa sửa index.
- Base 9B: cả 4 shard khớp manifest nguồn; 775 tên tensor khớp index, tổng payload 19.306.216.416 bytes khớp index. Config cũng khớp manifest nguồn. Khóa model ghi revision c202236235762e1c871ad0ccb60c8ee5ba337b9a; adapter_config tự nó không ghi revision, nên định danh revision dựa thêm vào model_lock và manifest.
- Không có safetensors nào chưa xác định trong phạm vi quét. Hai file model-... trong weights_pending thuộc bộ 4B, không phải 9B.
- Không tìm thấy ZIP hay .crdownload/.part/.partial/.tmp trong D:\\ZAIC tại lúc kiểm. Có 16 tệp .lock rỗng: đây là khóa phối hợp/cache, không tự chứng minh download đang dở và không phải shard rỗng.
- Đủ tệp trạng thái ở cả 8 thư mục checkpoint (gồm bản sao checkpoint-48 trong training_code); trainer_state.global_step khớp tên 40/45/48/80/83/125/150. Chưa deserialize hoặc chạy resume.

**Kết luận:** các vị trí adapter/checkpoint đã nêu đều tìm thấy; trọng số base hai bộ đã có và khớp manifest bàn giao. Chưa sẵn sàng suy luận ngay vì 6 shard còn ở weights_pending với tên khác index; bộ 4B còn bất nhất metadata total_size. Chưa kết luận bảo toàn toàn bộ lịch sử ngoài danh sách nguồn đã biết.
'''
for folder in [root,Path(r['source_project'])]:
    (folder/'WEIGHTS_INVENTORY.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
    (folder/'WEIGHTS_CHECK_REPORT.md').write_text(text,encoding='utf-8')
print(json.dumps({'base_source_hash_checks':[{ 'model':b['identified_model'],'all_shards_match':all(m['trusted_handoff_manifest_match'] for s in b['shards'] for m in s['matches'])} for b in r['base_sets']],
 'partial_downloads':len(r['partial_downloads']),'locks':len(r['lock_files']),'reports_written':True},indent=2))
