from pathlib import Path
import json
from collections import defaultdict
root=Path('D:/ZAIC');path=root/'WEIGHTS_INVENTORY.json'
r=json.loads(path.read_text(encoding='utf-8'))
private=root/'PRIVATE_DATA'
r['model_locks']={}
for p in [private/'zaic/datamate_v1/base_model_lock.json',private/'zaic/datamate_v1/qwen35_9b/model_lock.json']:
    r['model_locks'][str(p)]=json.loads(p.read_text(encoding='utf-8'))
for b in r['base_sets']:
    b['identified_model']='Qwen/Qwen3.5-9B' if b['model']['model_type']=='qwen3_5' else 'Qwen/Qwen3-4B-Instruct-2507'
    b['identification_basis']='config architecture/dimensions, exact tensor-key membership in index, local model lock; 9B additionally source SHA256 manifest'
    b['inference_status']='structurally_ready_not_runtime_tested' if b['loadable_at_current_path'] else 'all_content_found_but_wrong_path_or_name' if b['all_shard_contents_found'] else 'missing_shards'
    b['required_support_files']={n:(Path(b['directory'])/n).is_file() for n in ['config.json','tokenizer_config.json','tokenizer.json']}
byname=defaultdict(list)
for f in r['files']:
    p=Path(f['path']);byname[p.name].append(f)
    f['type']='base_or_training_metadata'
    if p.name in ['optimizer.pt','scheduler.pt','scaler.pt','rng_state.pth','training_args.bin']:
        f['type']='training_state_not_deserialized'
    if p.name=='adapter_model.safetensors':
        f['type']='learned_adapter'
        a=next((a for a in r['adapters'] if a['path']==str(p.parent)),None)
        f['model']=a['model'] if a else 'unidentified'
        f['checkpoint_step']=a['checkpoint_step'] if a else None
    elif p.suffix=='.safetensors':
        f['type']='unidentified_safetensors';f['model']='unidentified'
        for b in r['base_sets']:
            for shard in b['shards']:
                if f['path'] in shard['content_matches']:
                    f.update(type='base_weight_shard',model=b['identified_model'],index_filename=shard['index_filename'])
r['same_name_different_hash_preserved']=[{'filename':n,'files':[{'path':f['path'],'sha256':f['sha256']} for f in fs]} for n,fs in byname.items() if len({f['sha256'] for f in fs})>1]
r['notebook_count']=sum(1 for _ in root.rglob('*.ipynb'))
r['notebook_and_predictions_are_not_weights']=True
r['weights_pending_classification']=[f for f in r['files'] if 'weights_pending' in Path(f['path']).parts]
r['summary']={'adapter_directories':len(r['adapters']),'checkpoint_directories':len(r['checkpoint_directories']),
              'known_adapter_locations_found':sum(x['found'] for x in r['known_expected_adapters']),
              'known_adapter_locations_expected':len(r['known_expected_adapters']),
              'all_safetensors_header_valid':all(f.get('safetensors_header',{}).get('valid',True) for f in r['files']),
              'unstable_files':[f['path'] for f in r['files'] if not f['stable_during_hash']]}
append=['','## Phân loại bổ sung từng tệp weights_pending','',
 '| File hiện tại | Base | Tên index yêu cầu | Bytes | SHA256 |','|---|---|---|---:|---|']
for f in r['weights_pending_classification']:
    append.append(f"| {Path(f['path']).name} | {f['model']} | {f.get('index_filename','chưa xác định')} | {f['bytes']} | {f['sha256']} |")
append += ['', 'Đã tìm thấy '+str(r['summary']['known_adapter_locations_found'])+'/'+str(len(r['known_expected_adapters']))+' vị trí adapter/checkpoint đã nêu; các bản bổ sung ghi riêng trong JSON.',
 '', 'Bộ 4B được nhận diện bởi config Qwen3ForCausalLM, cấu trúc tensor theo index và model lock Qwen/Qwen3-4B-Instruct-2507. Hash shard 4B chỉ có đối chứng cục bộ, chưa có manifest hash nguồn đầy đủ.',
 '', 'Không cần tải bổ sung 4 shard 9B đã khớp manifest nguồn. Không có shard nào thiếu nội dung trong hai bộ index đã kiểm; tên/vị trí hiện tại vẫn chưa đáp ứng loader.',
 '', 'Cần bổ sung bằng chứng: manifest nguồn cho các adapter 4B/checkpoint, optimizer/RNG và shard 4B; danh sách đầy đủ mọi run trên Drive để xác nhận toàn bộ lịch sử.',
 '', 'Mọi bản cùng tên khác hash được giữ riêng. Không chọn BEST_MODEL từ thời gian mới nhất. Không thử nạp file pt/bin/pickle.',
 '', 'Phạm vi không bao gồm các ổ/thư mục ngoài D:\\ZAIC; chưa chứng nhận mọi tài sản trên máy/Drive đã được sao lưu đầy đủ.']
text=(root/'WEIGHTS_CHECK_REPORT.md').read_text(encoding='utf-8')+'\n'.join(append)+'\n'
for folder in [root,Path(r['source_project'])]:
    (folder/'WEIGHTS_INVENTORY.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
    (folder/'WEIGHTS_CHECK_REPORT.md').write_text(text,encoding='utf-8')
print(json.dumps({'summary':r['summary'],'final_9b':r['final_9b'],
 'bases':[{'model':b['identified_model'],'contents_found':b['all_shard_contents_found'],'total_matches':b['total_size_matches_index'],
 'source_hash_matches':[m['trusted_handoff_manifest_match'] for s in b['shards'] for m in s['matches']]} for b in r['base_sets']],
 'partials':r['partial_downloads'],'read_errors':r['read_errors']},ensure_ascii=False,indent=2))
