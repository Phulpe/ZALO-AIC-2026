"""Finalize only after independent post-evaluation and CPU preservation succeed."""
from pathlib import Path
import json,sys,csv,shutil,datetime,zipfile,stat
exp=Path(__file__).parent;sys.path.insert(0,str(exp/'src'))
from relearning import sha,verify_export,atomic,track,seal
from pilot_gate import learning_gate
from quality_gate import promotion
def main():
 dest=Path('D:/ZAIC/PRIVATE_DATA/zaic/result_811/TIME_FOCUSED_V3_20261109');v=json.loads((dest/'LOCAL_VERIFICATION.json').read_text(encoding='utf-8'));run=Path(v['run_path']).resolve();assert verify_export(run)==v['training_manifest_sha256'];assert verify_export(dest/'time_pilot')==v['outer_manifest_sha256']
 recovery=dest/'evaluation_recovery';recovery_manifest=verify_export(recovery);done=json.loads((recovery/'RECOVERY_COMPLETED.json').read_text(encoding='utf-8'));assert done['state']=='completed' and done['optimizer_executed'] is False and done['public_ran'] is False and done['candidate_sha256']==v['candidate_sha256']
 post=recovery/'post';verify_export(post);original=json.loads((post/'original14/REGRESSION_SUMMARY.json').read_text(encoding='utf-8'));selected=json.loads((post/'selected_regression/REGRESSION_SUMMARY.json').read_text(encoding='utf-8'))
 for name in ['original14','selected_regression']:verify_export(post/name)
 gates={'original14':learning_gate(original),'selected':learning_gate(selected)};assert gates['original14']==done['post']['original14_gate'] and gates['selected']==done['post']['selected_gate']
 field=json.loads((post/'NEW_CANDIDATE_FIELD_PROBE.json').read_text(encoding='utf-8'));assert field['loaded']['all_tensor_values_match'] and field['loaded']['adapter_sha256']==v['candidate_sha256']
 preservation=json.loads((dest/'PRESERVATION_RECEIPT.json').read_text(encoding='utf-8'));version=json.loads((dest/'SAVED_VERSION_EVIDENCE.json').read_text(encoding='utf-8'));off=json.loads((dest/'GPU_SHUTDOWN_RECEIPT.json').read_text(encoding='utf-8'));assert preservation['state']=='completed' and preservation['training_manifest_sha256']==v['training_manifest_sha256'] and preservation['checkpoint_sets']==10 and not preservation['gpu_executed'];assert version['preservation_status']=='COMPLETE' and version['evaluation_status']=='COMPLETE';assert off['gpu_reserved_seconds']==0 and off['active_events']==0 and off['accelerator']=='None' and off['total_this_round_seconds']<=7200
 cfg=json.loads((run/'run_config.json').read_text(encoding='utf-8'));assert cfg['parent_sha256']=='e7dfa635beccce128c99f9d3fd993d33dbb4c88e73e3ddeccbc2bc915eec4a4d' and cfg['init_mode']=='warm_start' and v['training']['steps']==9
 for item in json.loads((exp/'PRE_RUN_SNAPSHOT.json').read_text(encoding='utf-8'))['protected_files']:assert sha(item['path'])==item['sha256']
 source_hash=sha(exp/'CODE_V3.zip');assert source_hash==cfg['source_sha256']=='ed9cd9a0c70618b6d992c14526280a6e6df168810480bb4c05e02e4a9dc223da'
 tests=json.loads((exp/'CPU_DEFAULT_V4/source/TEST_RESULTS.json').read_text(encoding='utf-8'));assert tests['baseline']['tests']==99 and tests['new']['tests']==131 and tests['baseline']['passed'] and tests['new']['passed']
 cal=None;decision=None;promoted=False;promotion_evidence=None
 if done['calibration_ran']:
  before=json.loads(Path('D:/ZAIC/PRIVATE_DATA/time_v3_prior_reference/bundle/parent_calibration/EVALUATION.json').read_text(encoding='utf-8'));after=json.loads((recovery/'candidate_calibration/EVALUATION.json').read_text(encoding='utf-8'));decision=promotion(before,after,cfg['promotion_policy']);assert decision['promote']==done['promoted'];cal={k:value for k,value in after.items() if k!='verdicts'}
  assert not done['promoted'] or all(g['passed'] for g in gates.values())
  promotion_evidence={'calibration_manifest_sha256':after['manifest_sha256'],'candidate_metric':after['metric'],'minimum_metric':before['metric']+.01,'regression_count':decision['correct_to_wrong'],'max_regressions':0,'gates_passed':all(g['passed'] for g in gates.values())}
 else:assert done['promoted'] is False
 tracker=dest.parent/'RUN_STATE.json';prior_snapshot=dest/'RUN_STATE_AFTER_PRESERVATION.json';before_state=json.loads((prior_snapshot if prior_snapshot.exists() else exp/'RUN_STATE_BEFORE.json').read_text(encoding='utf-8'));assert json.loads(tracker.read_text(encoding='utf-8'))==before_state,'Concurrent tracker change'
 code=dest/'DEPLOYED_CODE';code.mkdir(exist_ok=False);source=code/'source_snapshot';source.mkdir()
 with zipfile.ZipFile(exp/'CODE_V3.zip') as archive:
  assert archive.testzip() is None
  for item in archive.infolist():assert (source/item.filename).resolve().is_relative_to(source.resolve()) and not stat.S_ISLNK(item.external_attr>>16)
  archive.extractall(source)
 source_manifest=json.loads((run/'source_manifest.json').read_text(encoding='utf-8'));assert all(sha(source/name)==digest for name,digest in source_manifest.items())
 for p,name in [(exp/'REPAIR_01/GPU_DEPLOY/gpu.py','training_gpu_version3.py'),(exp/'EVALUATION_RECOVERY/gpu.py','evaluation_gpu_version1.py'),(exp/'EVALUATION_RECOVERY/evaluation_child.py','evaluation_child.py'),(exp/'CPU_PRESERVATION/preserve.py','preservation_cpu.py')]:shutil.copy2(p,code/name)
 code_manifest=seal(code)
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();col=v['COLLECTION_SUMMARY.json'];sel=v['SELECTION_REPORT.json'];stats=v['PREVIOUS_ROUND_FIELD_PROBE.json'];teacher=field['teacher']
 result={'observed_at':now,'state':'bounded_optimizer_round_completed_and_evaluated','run_id':run.name,'init_mode':'warm_start','parent_sha256':cfg['parent_sha256'],'candidate_sha256':v['candidate_sha256'],'training':v['training'],'collection':col,'selection':sel,'original14':original,'selected_regression':selected,'learning_gates':gates,'calibration':cal,'calibration_decision':decision,'promoted':done['promoted'],'checkpoint_sets':10,'unique_adapter_hashes':9,'artifact_files_verified':v['artifact_files_verified'],'training_manifest_sha256':v['training_manifest_sha256'],'recovery_manifest_sha256':recovery_manifest,'preservation':preservation,'gpu_shutdown':off,'code_manifest_sha256':code_manifest,'public_ran':False,'full_resume_executed':False,'source_training_wrapper_failed_after_checkpoint':True}
 nll='\n'.join(f"| {name} | {data['groups']['correction']['time_mean_nll']:.6f} | {data['groups']['replay']['time_mean_nll']:.6f} |" for name,data in [('E7',stats['parent']),('28 cũ',stats['prior_candidate']),('Candidate mới',teacher)])
 cal_text=json.dumps({'evaluation':cal,'decision':decision},ensure_ascii=False) if cal is not None else 'Không chạy calibration đầy đủ: cổng 7 lỗi TRAIN cũ chưa đạt (0/7 sửa đúng), dù tập mới sửa được2/9 và giữ replay9/9. Không chọn model từ loss.'
 report=f'''# Kết quả vòng học thời gian v3

Quan sát UTC: {now}. Run `{run.name}`. Một vòng optimizer, giới hạn 120 phút GPU.

## Thực thi và bảo toàn

- Warm-start từ E7 đã xác minh, base Qwen3.5-9B revision c202236235762e1c871ad0ccb60c8ee5ba337b9a; optimizer/scheduler mới. Không resume checkpoint48, không fallback fresh.
- 99 baseline +131 tests mới =230 đạt trên Kaggle Linux CPU; baseline source không đổi. Input, split, tokenizer/mask/shift/EOS được kiểm trước GPU.
- Pool TRAIN 124/124 được thử thật; 103 đúng ngay; 21 cần sửa, 9 correction được xác minh, 12 unresolved còn ledger. Toàn learning pool1817 còn1693 chưa thử ở vòng này; không tuyên bố coverage toàn bộ100%.
- Dataset:9 correction+9 replay,11 doanh nghiệp. Prediction được lưu trước đọc gold; target correction qua kiểm ngữ nghĩa/thực thi/scorer. Không dùng gold public.
- Train9 bước,1.889 epoch, LR1e-5, rank8, batch1/accum4. Loss trung bình {v['training']['metrics']['train_loss']:.6f}. Objective chỉ tăng trọng số4 cho giá trị trường time, không tăng trọng số dấu JSON cố định.
- Adapter mới `{v['candidate_sha256']}`; 86.623.432 byte. 10 bộ checkpoint gồm step1–9 và final,9 hash adapter khác nhau. Optimizer/scheduler/scaler/RNG/trainer state đủ file, checksum khớp; chưa thực thi full resume.
- 148 artifact đã khớp manifest tải từ GPUv3. Notebook GPUv3 ERROR sau train vì guard phát hiện hơn512MiB bộ nhớ chưa giải phóng; training_result đã completed và checkpoint đã lưu. Giữ nguyên log lỗi. Không train lại; đánh giá bằng child inference riêng để giải phóng model theo vòng đời process.
- Bản bảo toàn CPU hoàn tất,128 file chỉ gồm model/state/tokenizer/metadata; không journal/câu hỏi/gold/SFT. Manifest run `{v['training_manifest_sha256']}` khớp bản máy. Bản zip đầy đủ có journal bị auto-review chặn và không upload; phương án whitelist state-only đã thay thế.
- E7, B306, adapter28 và submission cũ giữ nguyên hash. Không ghi đè kết quả public cũ, không nộp bài, không chạy public lần này.

## Đánh giá thật sau train

| Tập | Correction trước | Correction sau | Replay sau | Correct→wrong |
|---|---:|---:|---:|---:|
| 14 mẫu cũ | 0/{original['correction_total']} | {original['correction_after_correct']}/{original['correction_total']} | {original['replay_after_correct']}/{original['replay_before_correct']} | {original['correct_to_wrong']} |
| 18 mẫu vòng mới | 0/{selected['correction_total']} | {selected['correction_after_correct']}/{selected['correction_total']} | {selected['replay_after_correct']}/{selected['replay_before_correct']} | {selected['correct_to_wrong']} |

Cả hai tập là TRAIN đã tiếp xúc, không phải holdout. Đánh giá không cập nhật trọng số, nạp adapter strict và so đủ496 tensor với file nguồn. Chỉ kết quả calibration đủ điều kiện mới cho phép đổi BEST_MODEL.

| Adapter | NLL giá trị time trên7 correction cũ | NLL time replay cũ |
|---|---:|---:|
{nll}

Teacher-forcing là chẩn đoán token, không phải độ chính xác trả lời mù.

{cal_text}

BEST_MODEL {'được cập nhật theo cổng đã khóa' if done['promoted'] else 'giữ nguyên'}. LATEST_CHECKPOINT và LATEST_SYNCED trỏ tới run9 bước được bảo toàn trên phiên CPU hoàn tất và khớp hash.

## Giới hạn và bước tiếp theo

- Đủ artifact cho inference; đã load/test trên GPU. Chưa chứng minh model tốt hơn trên dữ liệu độc lập nếu calibration không đạt.
- Đủ file trạng thái để khảo sát resume; chưa triển khai/xác minh full resume.
- Có source/config/data/split/environment/hash cho kiểm toán; chưa chứng minh tái lập bit-for-bit.
- Còn12 unresolved trong pool124 và1693 pending toàn pool. Không tự tạo target hoặc retry vô hạn để lấp số lượng.
- GPU dùng tổng {off['total_this_round_seconds']/60:.2f}/120 phút, gồm setup lỗi và đánh giá. GPU đã dừng, quota reserved0, Active Events0, Accelerator=None. Không bảo đảm điều khiển khi agent ngắt.
- Dữ liệu calibration552 đã tiếp xúc lịch sử; không gọi untouched holdout. Không có PA public mới của candidate này.

Bằng chứng: LOCAL_VERIFICATION.json, FINAL_STATUS.json, GPU_SHUTDOWN_RECEIPT.json, SAVED_VERSION_EVIDENCE.json, PRESERVATION_RECEIPT.json, time_pilot và evaluation_recovery trong thư mục này.
'''
 # All evidence has passed before any shared pointer/history changes.
 if not prior_snapshot.exists():
  shutil.copy2(tracker,dest/'RUN_STATE_BEFORE.json');track(tracker,run,'checkpoint',{});track(tracker,run,'sync',{'saved_version_completed':True,'saved_version_url':version['preservation_url'],'remote_manifest_sha256':v['training_manifest_sha256']})
 if done['promoted']:promoted=track(tracker,run,'promote',promotion_evidence)
 assert promoted==done['promoted']
 result.update(LATEST_CHECKPOINT_updated=True,LATEST_SYNCED_updated=True);atomic(dest/'FINAL_STATUS.json',result);atomic(exp/'FINAL_STATUS.json',result)
 capabilities={'observed_at':now,'warm_start':{'status':'da_chay_GPU','scope':'E7 trainable;new optimizer/scheduler;9 steps'},'learning':{'status':'da_chay_GPU','scope':'124/124 frozen TRAIN subset;not full1817'},'correction_replay':{'status':'da_chay_GPU','scope':'9 verified correction+9 replay;12 unresolved retained'},'temporal_objective':{'status':'da_chay_GPU','scope':'payload weight4;unit tests+9 actual optimizer steps'},'post_evaluation':{'status':'da_chay_GPU','scope':'strict saved adapter reload;14 old and18 current TRAIN rows;no optimizer'},'checkpoint_preservation':{'status':'da_chay_thuc_te','scope':'10 state sets;manifest hashes match CPU saved version and local'},'full_resume':{'status':'chua_trien_khai','scope':'state files present;resume execution unverified'},'single_process_full_pipeline':{'status':'GPU_integration_failed_after_training','scope':'memory cleanup guard;recovered with separate inference child;future integrated runner needs process isolation'}}
 atomic(dest/'CAPABILITY_STATUS.json',capabilities);atomic(exp/'CAPABILITY_STATUS.json',capabilities)
 (dest/'RESULT_REPORT_VI.md').write_text(report,encoding='utf-8');(exp/'RESULT_REPORT_VI.md').write_text(report,encoding='utf-8')
 atomic(dest/'WEIGHTS_INVENTORY_RUN.json',{'observed_at':now,'run_id':run.name,'checkpoints':v['checkpoints'],'manifest_sha256':v['training_manifest_sha256'],'pickle_deserialized':False})
 registry=Path('D:/ZAIC/RUN_REGISTRY.csv');shutil.copy2(registry,dest/'RUN_REGISTRY_BEFORE.csv')
 with registry.open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;existing=list(reader)
 assert not any(row['run_id']==run.name for row in existing)
 row={key:'' for key in fields};row.update(run_id=run.name,kind='warm_start_temporal_correction_replay',time_and_basis=now,base_model='Qwen/Qwen3.5-9B',base_revision=cfg['revision'],initialization='warm_start E7; new optimizer/scheduler;9 steps',lineage_confidence='verified parent and source hashes',status='training completed; evaluated; preserved',final_adapter=str(run/'checkpoint'),all_checkpoints='checkpoint1-9 and final;10 sets;9 unique hashes',resume_state='optimizer/scheduler/scaler/RNG/trainer present; full resume not executed',code_configuration=str(dest/'DEPLOYED_CODE'),data_manifest_hash=cfg['data_manifest_sha256'],library_environment=str(run/'environment.json'),logs='Private original training failure log + evaluation logs preserved',metrics_evaluation=json.dumps({'old14':original,'new18':selected,'calibration':cal,'promoted':done['promoted']}),lessons='Cleanup guard stopped post stage after successful train; inference isolated in child; no optimizer retry',local_copies=str(dest),drive_copies='not checked this round',drive_checksum_status='unverified',kaggle_copy=version['preservation_url'],use_results='inference tested; BEST only if frozen gates pass',continue_training='state files complete; full resume unverified',reproduce='source/data/split/runtime hashes; no bit-for-bit reproduction proof',missing_unverified='12 unresolved;1693 pending;no new public;no full resume')
 with registry.open('a',encoding='utf-8',newline='') as f:writer=csv.DictWriter(f,fields);writer.writerow(row)
 training=Path('D:/ZAIC/training.md');shutil.copy2(training,dest/'training_BEFORE.md')
 with training.open('a',encoding='utf-8') as f:f.write(f'\n\n## Bằng chứng vòng thời gian v3 — {run.name}\n\nWarm-start E7,9 bước,9 correction+9 replay. CPU230tests đạt. TRAIN124/124 thử,9 correctionverified,12unresolved; toànpool1693pending. Hậu xử lý GPUv3 lỗi cleanup sau khi traincompleted; không train lại, đánh giá child riêng và bảo toàn CPU fullstate. Model mới {v["candidate_sha256"]}. BEST {"promoted theo cổng" if done["promoted"] else "giữ nguyên"}. LATEST_CHECKPOINT/LATEST_SYNCED đã khớp manifest {v["training_manifest_sha256"]}. Full resume chưa chứng minh. Báo cáo: {dest/"RESULT_REPORT_VI.md"}.\n')
 print(json.dumps({'state':'finalized','run_id':run.name,'promoted':done['promoted'],'gpu_minutes':off['total_this_round_seconds']/60,'report':str(dest/'RESULT_REPORT_VI.md')},ensure_ascii=True))
if __name__=='__main__':main()
