from pathlib import Path
import csv,hashlib,json
root=Path('D:/ZAIC');E=root/'TRAINING_AUDIT_EVIDENCE';project=root/'drive-download-20241211T124801Z-001/ZAIC_Local_GUI/CODEX_PROJECT'
audit=json.loads((E/'TRAINING_HISTORY_DETAILS.json').read_text(encoding='utf-8'))
snapshots=json.loads((E/'EMBEDDED_CODE_SNAPSHOTS.json').read_text(encoding='utf-8'))
rows=list(csv.DictReader((root/'RUN_REGISTRY.csv').open(encoding='utf-8-sig',newline='')))
assert len(rows)==6 and len({r['run_id'] for r in rows})==6
assert all(x['stable_during_hash'] for x in audit['current_artifact_hashes'].values())
artifacts=[a for r in audit['runs'] for a in r['artifacts'] if a.get('weight_found')]
assert len([a for a in artifacts if a.get('step') is not None])==7
assert all(a['matches_prior_inventory'] for a in artifacts)
assert all(not a['missing_resume_files'] for a in artifacts if a.get('step') is not None)
stable=next(r for r in audit['runs'] if r['run_id']=='9B_stable_48')
assert stable['artifacts'][0]['weight']['sha256']=='e7dfa635beccce128c99f9d3fd993d33dbb4c88e73e3ddeccbc2bc915eec4a4d'
data=next(d for d in stable['data'] if d['path'].endswith('sft_train.jsonl'))
assert data['sha256']=='5057658ac10c19579c675c9f926cc13f1e17641b88eb312318e48be75e6bb170' and data['rows']==768
assert all(x['adapter_sha256_equal'] for x in audit['backup_aliases'])
assert all(all(x['state_hash_equal'].values()) for x in audit['backup_aliases'])
assert not audit['remote_size_mismatches']
nb=next(x for x in json.loads((E/'LOCAL_TRAINING_PROBE.json').read_text(encoding='utf-8'))['notebooks'] if x['path'].endswith('ZAIC_QLoRA_intent_v0.ipynb'))
assert all(x['execution_count'] is None for x in nb['training_cells'])

note='''
## Bổ sung: các phiên bản mã cũ nằm trong notebook

Đã giải mã an toàn 17 gói JSON nén trong các string literal của notebook (giới hạn dung lượng, chỉ parse bytes/AST, không exec). Tìm thấy source train_candidate.py ở cell 23 và 24 với hash khác nhau; train.py 9B ở cell 53 và 54 cũng có hash khác nhau. Đây là bằng chứng mã lịch sử vẫn còn trong notebook, chưa phải bằng chứng mọi phiên bản đều thực thi thành công. Chi tiết từng hash/cấu hình nằm trong TRAINING_AUDIT_EVIDENCE/EMBEDDED_CODE_SNAPSHOTS.json.

Do vậy không kết luận mã thử nghiệm thất bại đã mất. Phần chưa đầy đủ là liên kết chắc chắn phiên bản mã nào đã thực thi ở từng lần lỗi, vì execution_count đều null và chưa có receipt bất biến ghi source hash + resume checkpoint + environment cho từng lần chạy.

Kiểm tra chéo báo cáo: 6 mục run/attempt không trùng ID; 4 giai đoạn có adapter; 7 checkpoint chính đầy đủ bộ state; 1 bản sao checkpoint-48 khớp toàn bộ hash state; 14 adapter được tính hash lại, 137 tệp state/support/adapter được tính hash lại; 68 phép so size Drive đều khớp. SFT 9B hiện có đúng 768 dòng và đúng hash đã ghi trong training_complete. Không phép so size nào được nâng thành xác nhận checksum Drive.
'''
for folder in [root,project]:
 with (folder/'TRAINING_HISTORY_AUDIT.md').open('a',encoding='utf-8') as f:f.write(note)
audit['embedded_source_snapshot_evidence']='TRAINING_AUDIT_EVIDENCE/EMBEDDED_CODE_SNAPSHOTS.json'
audit['validation']={'csv_unique_runs':6,'main_checkpoints_complete':7,'fresh_hashes_stable':137,
 'adapter_hashes_match_prior_inventory':True,'stable_sft_rows_and_hash_match':True,
 'remote_size_checks_passed':68,'remote_checksum_checks_passed':0,
 'historical_notebook_execution_counters_null':True,'embedded_json_code_bundles_safely_decoded':17}
(E/'TRAINING_HISTORY_DETAILS.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
checks={'reports_readable':True,'unique_training_entries':6,'full_canonical_checkpoints':7,'all_adapter_hashes_unchanged':True,
 'code_snapshots_recovered_without_execution':True,'raw_data_in_report':False,'training_or_upload_started':False,
 'outputs':[{'path':str(root/n),'sha256':hashlib.sha256((root/n).read_bytes()).hexdigest()} for n in ['TRAINING_HISTORY_AUDIT.md','RUN_REGISTRY.csv']]}
(E/'AUDIT_VALIDATION.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(json.dumps(checks,indent=2))
