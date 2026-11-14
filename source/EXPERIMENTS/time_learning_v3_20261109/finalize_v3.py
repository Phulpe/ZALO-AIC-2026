"""Finalize from saved-version checksums, locked gates and observed GPU off."""
from pathlib import Path
import json,sys,csv,shutil,datetime,hashlib,zipfile,stat
exp=Path(__file__).parent;sys.path.insert(0,str(exp/'src'))
from relearning import atomic,sha,verify_export,seal,track
from quality_gate import promotion

def main(folder):
 dest=Path(folder);out=dest/'time_pilot';v=json.loads((dest/'LOCAL_VERIFICATION.json').read_text(encoding='utf-8'));assert v['WORKFLOW_COMPLETED.json']['state']=='completed';assert verify_export(out)==v['outer_manifest_sha256']
 run=Path(v['run_path']);assert verify_export(run)==v['training_manifest_sha256'];off=json.loads((dest/'GPU_SHUTDOWN_RECEIPT.json').read_text());assert off['gpu_reserved_seconds']==0 and off['active_events']==0 and off['accelerator']=='None'
 cfg=json.loads((run/'run_config.json').read_text(encoding='utf-8'));assert cfg['source_sha256']=='ed9cd9a0c70618b6d992c14526280a6e6df168810480bb4c05e02e4a9dc223da'
 complete=v['WORKFLOW_COMPLETED.json'];promoted=False;cal=None
 if complete['promoted']:
  prior=Path('D:/ZAIC/PRIVATE_DATA/time_v3_prior_reference/bundle/parent_calibration/EVALUATION.json');before=json.loads(prior.read_text());after=json.loads((out/'candidate_calibration/EVALUATION.json').read_text());cal=promotion(before,after,cfg['promotion_policy']);assert cal['promote'] and v['original14_gate_recomputed']['passed'] and v['selected_regression_gate_recomputed']['passed']
  promotion_evidence={'calibration_manifest_sha256':after['manifest_sha256'],'candidate_metric':after['metric'],'minimum_metric':before['metric']+.01,'regression_count':cal['correct_to_wrong'],'max_regressions':0,'gates_passed':True}
 parent=Path('D:/ZAIC/PRIVATE_DATA/zaic/result_811');tracker=parent/'RUN_STATE.json';before_state=json.loads((exp/'RUN_STATE_BEFORE.json').read_text());assert json.loads(tracker.read_text())==before_state,'Concurrent tracker change; preserve files and review'
 shutil.copy2(tracker,dest/'RUN_STATE_BEFORE.json')
 track(tracker,run,'checkpoint',{});track(tracker,run,'sync',{'saved_version_completed':True,'saved_version_url':'https://www.kaggle.com/code/lnghongphc/zaic-time-focused-learning-v3-20261109?scriptVersionId=367303123','remote_manifest_sha256':v['training_manifest_sha256']})
 if complete['promoted']:promoted=track(tracker,run,'promote',promotion_evidence)
 assert promoted==complete['promoted']
 # Store exact code and wrapper separately from immutable remote run folders.
 code=dest/'DEPLOYED_CODE';code.mkdir(exist_ok=False);source=code/'source_snapshot';source.mkdir();archive=exp/'CODE_V3.zip';assert sha(archive)==cfg['source_sha256']
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None
  for f in z.infolist():assert (source/f.filename).resolve().is_relative_to(source.resolve()) and not stat.S_ISLNK(f.external_attr>>16)
  z.extractall(source)
 sm=json.loads((run/'source_manifest.json').read_text());assert all(sha(source/n)==h for n,h in sm.items())
 shutil.copy2(exp/'REPAIR_01/GPU_DEPLOY/gpu.py',code/'gpu_version3.py');shutil.copy2(exp/'REPAIR_01/kernel-metadata.gpu.json',code/'kernel-metadata.gpu-v3.json');code_manifest=seal(code)
 inventory={'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':run.name,'checkpoint_sets':v['checkpoints'],'unique_adapter_hashes':v['unique_adapter_hashes'],'all_state_file_hashes':json.loads((run/'ARTIFACT_MANIFEST.json').read_text())['files'],'pickle_deserialized':False};atomic(dest/'WEIGHTS_INVENTORY_RUN.json',inventory)
 setup=json.loads((exp/'PRE_RUN_SNAPSHOT.json').read_text());preserved=[]
 for item in setup['protected_files']:
  assert sha(item['path'])==item['sha256'];preserved.append(item)
 atomic(dest/'ORIGINALS_PRESERVED.json',{'files':preserved,'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat()})
 col=v['COLLECTION_SUMMARY.json'];sel=v['SELECTION_REPORT.json'];old=v['original14'];post=v['selected_regression'];stats=v['PREVIOUS_ROUND_FIELD_PROBE.json'];new=v['NEW_CANDIDATE_FIELD_PROBE.json'];now=datetime.datetime.now(datetime.timezone.utc).isoformat()
 result={'observed_at':now,'state':'completed_bounded_round','run_id':run.name,'init_mode':'warm_start','parent_run_id':'9B_stable_48','parent_sha256':cfg['parent_sha256'],'candidate_sha256':v['candidate_sha256'],'source_sha256':cfg['source_sha256'],'gpu_kernel_version':3,'script_version_id':367303123,'metrics':col,'selection':sel,'training':v['training'],'original14':old,'selected_regression':post,'calibration':v.get('candidate_calibration'),'promoted':promoted,'LATEST_SYNCED_updated':True,'LATEST_CHECKPOINT_updated':True,'gpu_shutdown':off,'checkpoint_sets':len(v['checkpoints']),'full_resume_executed':False,'public_ran':False,'remaining_limitations':['Full learning pool not complete','Calibration historically exposed','No public score for candidate','No full resume or bit-for-bit reproduction proof']}
 atomic(dest/'FINAL_STATUS.json',result);atomic(exp/'FINAL_STATUS.json',result)
 nll_table='\n'.join(f"| {name} | {data['groups']['correction']['time_mean_nll']:.6f} | {data['groups']['replay']['time_mean_nll']:.6f} |" for name,data in [('E7 parent',stats['parent']),('28 cũ',stats['prior_candidate']),('Candidate mới',new['teacher'])])
 cal_text=('Calibration candidate: '+json.dumps(v['candidate_calibration'],ensure_ascii=False)) if complete['calibration_ran'] else 'Chưa chạy calibration đầy đủ: cổng học chưa đạt hoặc thời gian còn lại dưới ngưỡng đã khóa. Không promote từ loss hay kết quả trên tập đã học.'
 text=f'''# Vòng học thời gian v3 — kết quả thực tế

Quan sát UTC: {now}. Run `{run.name}`. Một vòng optimizer, giới hạn mới 120 phút GPU, tách ngân sách vòng trước.

## Đã làm và bằng chứng

- Giữ nguyên E7, B306, candidate28 và submission cũ; hash kiểm lại khớp. Base9B cố định giữ ở input, không xuất lại 19GB.
- Mã mới trong {exp/'src'}. Source archive `{cfg['source_sha256']}`; 99 baseline +131 tests mới =230, không skip, đạt Kaggle Linux CPU. Baseline/app không sửa.
- Đối chiếu 7 lỗi cũ: 5 today và2 last_month so với target yesterday; cả7 khác khoảng thời gian thực khi neo as_of_date. Không phải khác cú pháp JSON tương đương.
- Prompt train/inference, completion mask/shift/EOS đều đúng. 7 correction cũ có398 target tokens; time object49, giá trị time14. Objective mới chỉ tăng trọng số4 vào giá trị time; giữ label/token IDs, không tăng trọng số dấu JSON cố định.
- LR1e-5, rank8, batch1/accum4, warm-start E7 với optimizer/scheduler mới; tối đa32steps/2epochquyđổi. Đây là cấu hình thử có đo lường, chưa chứng minh tối ưu toàn cục.
- Thu dự đoán mù gold: {col['metrics']['attempted']}/{col['metrics']['expected']} trong pool đã freeze124; first-pass đúng {col['metrics']['first_pass_correct']}. Gold mở sau prediction được lưu. Một generate/câu, lỗi không bị bỏ khỏi ledger.
- Bank correction/replay qua semantic, executor, scorer và provenance; chọn {sel['correction']} correction +{sel['replay']} replay, {sel['unique_businesses']} doanh nghiệp. {col['metrics']['unresolved']} unresolved được giữ, không ép target.
- Train thực tế {v['training']['steps']} bước; loss {v['training']['metrics']['train_loss']:.6f}; objective `{v['training']['loss_objective']}`. Gradient/loss hữu hạn theo guard; tensor delta được xác minh. Không gọi đây là resume48.

## Đo loss và hiệu quả sửa lỗi

Các giá trị dưới là NLL teacher-forcing trên14mẫu TRAIN cũ, **không phải độ chính xác suy luận mù**. Trước mỗi lần đo, so toàn bộ tensor adapter đã nạp với safetensors nguồn.

| Adapter | NLL giá trị time — correction | NLL giá trị time — replay |
|---|---:|---:|
{nll_table}

Suy luận thật sau train, không cập nhật weights:

- 14mẫu cũ: correction {old['correction_after_correct']}/{old['correction_total']} (parent trước0), replay giữ đúng {old['replay_after_correct']}/{old['replay_before_correct']}; đúng→sai {old['correct_to_wrong']}.
- Tập vừa học: correction {post['correction_after_correct']}/{post['correction_total']}; replay {post['replay_after_correct']}/{post['replay_before_correct']}; đúng→sai {post['correct_to_wrong']}.
- Cổng học yêu cầu correction tăng và không quên replay ở cả hai bộ. Post-tests chạy ngay trong **cùng phiên GPU**, trước calibration/public, tránh phiên GPU đánh giá riêng và hàng đợi như vòng trước.

{cal_text}

**BEST_MODEL {'được cập nhật theo toàn bộ cổng đã khóa' if promoted else 'giữ nguyên'}**. Full calibration nếu chạy dùng552câu đã từng lộ, không gọi untouched holdout. Parent calibration E7 được tái sử dụng từ snapshot có hash, chỉ khi GPU/CUDA/packages, model/data/split/decoder/scorer khớp; không đổi ngưỡng +0,01/0regression sau khi xem kết quả. Không chạy public_test, không dùng public gold, không nộp bài.

## Bảo toàn, đồng bộ và giới hạn

- {len(v['checkpoints'])} bộ adapter/state, {v['unique_adapter_hashes']} hash adapter khác nhau; giữ cả bản trùng. Header safetensors/kích thước/hash hợp lệ; không deserialize pickle/bin/pt để kiểm kê.
- {v['artifact_files_verified']} file output khớp manifest; training manifest `{v['training_manifest_sha256']}`; outer `{v['outer_manifest_sha256']}`. Source snapshot {len(sm)} file Python khớp hash; deployed code manifest {code_manifest}.
- LATEST_CHECKPOINT/LATEST_SYNCED cập nhật sau khớp hash với saved Kaggle version3. BEST tách riêng. Checkpoint, config, manifests, metrics, private ledger và mã đã lưu tại `{dest}`.
- Full learning pool: {col['full_pool']['attempted']}/{col['full_pool']['expected']}, còn {col['full_pool']['pending']} pending. Không cộng lặp97câu cũ thành lượt mới đã bao phủ.
- Dùng inference: candidate đã reload từ đĩa và tạo output thật. Tiếp tục train: có adapter để warm-start và các file state đã kiểm hash; **full resume chưa triển khai/chạy kiểm chứng**.
- Tái lập: giữ source/config/manifest/environment/weights/state; chưa rerun tái lập và chưa đủ lock toàn môi trường/container/driver để cam kết bit-for-bit.
- Adapter config giữ đường dẫn Colab lịch sử, không sửa file đã seal. Base thật được pin bằng15hash: Qwen/Qwen3.5-9B revision c202236235762e1c871ad0ccb60c8ee5ba337b9a.
- Config workflow toàn cục là learning; calibration dùng entrypoint inference riêng, không có optimizer. Một số khóa giới hạn từ wrapper cũ còn lưu làm lịch sử; trần đang thực thi là soft6300s/hard6750s, cộng setup lỗi70,011s vàreserve180s =7000,011s dưới7200s.

## Lỗi chuẩn bị đã bảo toàn

GPU version1 thất bại trước entrypoint học vì input notebook thiếu biên nhận cũ cần dùng; không có optimizer step. GPU dùng khoảng70giây đã tính vào120phút. Sau đó chuyển CPU ngay, giữ log/metadata gốc.

Tạo reference Private bất biến khoảng80MB từ adapter28+config+environment+parent evaluation đã kiểm hash, không sao chép base. Kaggle tự giải nén ZIP: wrapper tái lập đúng tập file theo manifest, chống path traversal/symlink. CPU xác minh cả15filebase/E7/28/pool124 trước GPUversion3. Reference manifest91976ad72e4791a57747dc215859495cb6874e039e5524591ba2f84046d47ced. Không dựa vào hậu tố version trong input notebook làm bằng chứng đã pin đúng artifact.

## GPU và vận hành

GPU đã dừng; quota API reserved=0, giao diện ActiveEvents=0, Accelerator=None; đã kiểm phiên nền và tình trạng phiên tương tác theo biên nhận. Xem GPU_SHUTDOWN_RECEIPT.json. Không giữ GPU để tải file/phân tích. Không hứa đồng bộ liên tục khi agent ngắt.

Kaggle GPU version3: https://www.kaggle.com/code/lnghongphc/zaic-time-focused-learning-v3-20261109?scriptVersionId=367303123 . Mặc định hiện trở lại CPU; không Run All notebook Colab lịch sử. Lượt optimizer mới tiếp theo cần cấu hình/ngân sách riêng, không coi120phút còn nguyên.
'''
 for folder in [dest,exp]:(folder/'RESULT_REPORT_VI.md').write_text(text,encoding='utf-8')
 # Preserve existing CSV bytes and schema; append failed setup and actual training.
 registry=Path('D:/ZAIC/RUN_REGISTRY.csv');csv.field_size_limit(10_000_000)
 with registry.open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;old=list(reader)
 assert not any(r['run_id']==run.name for r in old)
 shutil.copy2(registry,dest/'RUN_REGISTRY_BEFORE.csv');original_bytes=registry.read_bytes()
 def js(v):return json.dumps(v,ensure_ascii=False)
 row=dict.fromkeys(fields,'');row.update(run_id=run.name,kind='training_run',time_and_basis=f'{now}; KaggleGPUv3 and sealed artifacts',base_model='Qwen/Qwen3.5-9B',base_revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a',initialization='warm_start E7 /9B_stable_48; optimizer/scheduler new;28for diagnostic only',lineage_confidence='high: parent hash, strict in-memory tensor match, run config and artifacts',status=f'{v["training"]["steps"]} steps completed; synced; promoted={promoted}',final_adapter=js({'path':str(run/'checkpoint'),'sha256':v['candidate_sha256']}),all_checkpoints=js(v['checkpoints']),resume_state='Files inventoried with hashes; full resume not implemented/tested',code_configuration=js({'source_sha256':cfg['source_sha256'],'config':str(run/'run_config.json'),'snapshot':str(code)}),data_manifest_hash=js({'runtime':cfg['runtime_manifest_sha256'],'selected':sel['dataset_sha256']}),library_environment=js(json.loads((out/'environment.json').read_text())),logs=js([str(out),str(exp/'REPAIR_01')]),metrics_evaluation=js({'collection':col,'original14':old if False else v['original14'],'selected':post,'calibration':v.get('candidate_calibration'),'promoted':promoted}),lessons='Uniform total loss did not show original time errors; payload objective and same-session post gates added. No public baseline rerun.',local_copies=js([str(dest)]),drive_copies='[]',drive_checksum_status='not checked for this run',kaggle_copy='https://www.kaggle.com/code/lnghongphc/zaic-time-focused-learning-v3-20261109?scriptVersionId=367303123',use_results='Saved adapter reload and real inference verified',continue_training='Warm-start possible; full resume unverified',reproduce='Pinned source/data/model/config/state; no rerun or full environment lock',missing_unverified='Remaining learning pool;full resume;official public score;unseen holdout')
 failed=dict.fromkeys(fields,'');failed.update(run_id='time_v3_setup_gpu_v1_20261109',kind='setup_attempt_not_training',time_and_basis='2026-11-09T12:12:31Z quota observation;70.011GPUseconds',base_model='Qwen/Qwen3.5-9B',base_revision=row['base_revision'],initialization='Intent warm-start; stopped before time_pilot entry',lineage_confidence='source stack proves no optimizer/model training reached',status='failed preflight; no learned weights',code_configuration=str(exp/'KAGGLE_GPU'),logs='D:/ZAIC/PRIVATE_DATA/time_learning_v3_gpu_v1_failed',lessons='Notebook input did not expose expected old workflow receipt; immutable Private reference bundle and actual all-input CPU check added',local_copies='D:/ZAIC/PRIVATE_DATA/time_learning_v3_gpu_v1_failed',kaggle_copy='lnghongphc/zaic-time-focused-learning-v3-20261109/1',use_results='No model update',continue_training='Not a checkpoint',reproduce='Failure log/source retained')
 rows=[failed,row] if not any(r['run_id']==failed['run_id'] for r in old) else [row]
 if original_bytes and not original_bytes.endswith(b'\n'):
  with registry.open('ab') as f:f.write(b'\r\n')
 with registry.open('a',encoding='utf-8',newline='') as f:csv.DictWriter(f,fieldnames=fields).writerows(rows)
 assert registry.read_bytes().startswith(original_bytes)
 with (dest/'RUN_REGISTRY_THIS_ROUND.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 training=Path('D:/ZAIC/training.md');shutil.copy2(training,dest/'training_BEFORE.md')
 with training.open('a',encoding='utf-8') as f:f.write(f'\n\n## Bằng chứng time v3 ngày09/11/2026 — {run.name}\n\nPayload-time objective4/LR1e-5; warm-startE7, khôngresume. {v["training"]["steps"]}steps; original7correction {v["original14"]["correction_after_correct"]}/7, replay{v["original14"]["replay_after_correct"]}/7. Fullcalibration={complete["calibration_ran"]}, promote={promoted}; không chạypublic. 124poolpilot, còn{col["full_pool"]["pending"]}pending. 230testsCPUđạt; GPUtrain/reload/fieldprobe/post đãchạythật trongcùngphiên. Checkpointsync/hash vàGPUoff xem {dest/"RESULT_REPORT_VI.md"}. Giữnguyên vòngcũ vàmọicheckpoint; fullresume chưa kiểm. LỗiGPUsetup70giây đãtínhvàongânsách120phút, khôngreset.\n')
 atomic(exp/'ACTIVE_HANDOFF.json',{**result,'next_gpu_job':None,'report':str(dest/'RESULT_REPORT_VI.md'),'next_optimizer_round_requires_new_config_and_budget':True})
 print(json.dumps({'state':'completed','run_id':run.name,'promoted':promoted,'original14':v['original14'],'selected':post,'report':str(dest/'RESULT_REPORT_VI.md')},ensure_ascii=True))

if __name__=='__main__':main(sys.argv[1])
