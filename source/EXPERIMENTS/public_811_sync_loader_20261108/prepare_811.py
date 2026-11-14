"""Assemble verified TRAIN replay and a separate public inference input, no gold export."""
from pathlib import Path
import sys,json,shutil,hashlib,zipfile,datetime
root=Path('D:/ZAIC');exp=Path(__file__).parent;sys.path.insert(0,str(exp))
from relearning import verify_export,sha,atomic,seal,PARENT_SHA
train=root/'PRIVATE_DATA/zaic/train';public=root/'PRIVATE_DATA/zaic/public_test'
assert sha(train/'questions.jsonl')=='b258ebd978a6d51b3b0b431b26926f958e56522beeb7da2984fea8333e21cec1'
parent=root/'PRIVATE_DATA/zaic/datamate_v1/qwen35_9b/candidate/final'
assert sha(parent/'adapter_model.safetensors')==PARENT_SHA
seed_meta=json.loads((root/'EXPERIMENTS/replay_seed_preparation_20261108/RESULT.json').read_text(encoding='utf-8'))
seed=Path(seed_meta['private_output_path']);seed_hash=verify_export(seed)
probe=next((root/'PRIVATE_DATA/KAGGLE_RUNS/relearning_v3_smoke_366867761/output/learning_probe_runs_v2').iterdir());probe_hash=verify_export(probe)
rows=json.loads((seed/'verified_replay_seed.json').read_text(encoding='utf-8'))+json.loads((probe/'verified_bank.json').read_text(encoding='utf-8'))['replay']
assert len(rows)==10 and len({r['id'] for r in rows})==10
prepared=root/'PRIVATE_DATA/relearning_v2_prepared/20261108T070936492119Z'
split=json.loads((prepared/'SPLIT_MANIFEST.json').read_text(encoding='utf-8'))
assert all(r['kind']=='replay' and r['target_verified'] is True and r['split']=='learning' and r['business_id'] in split['groups']['learning'] and r['parent_sha256']==PARENT_SHA for r in rows)
bundle=root/'PRIVATE_DATA/zaic/result_811/INPUT_811';bundle.mkdir(exist_ok=False)
atomic(bundle/'TRAIN_VERIFIED_REPLAY.json',rows)
shutil.copytree(public,bundle/'PUBLIC')
qs=[json.loads(l) for l in (bundle/'PUBLIC/questions.jsonl').read_text(encoding='utf-8-sig').splitlines() if l.strip()]
assert len(qs)==len({q['id'] for q in qs})==1000
assert all(set(q)=={'id','business_id','question','as_of_date'} for q in qs)
assert not (bundle/'PUBLIC/answers.jsonl').exists()
public_prompt_ids={json.dumps([q['business_id'],q['question'],q['as_of_date']],ensure_ascii=False) for q in qs}
for row in rows:
    q=json.loads(row['prompt'])
    assert json.dumps([row['business_id'],q['question'],q['as_of_date']],ensure_ascii=False) not in public_prompt_ids
atomic(bundle/'TRAIN_PROVENANCE.json',{'objective':'verified_replay_refresh','correction_count':0,'rows':10,'parent_sha256':PARENT_SHA,'historical_seed_manifest_sha256':seed_hash,'gpu_parent_probe_manifest_sha256':probe_hash,'raw_train_questions_sha256':sha(train/'questions.jsonl'),'raw_train_answers_sha256':sha(train/'answers.jsonl'),'split_manifest_sha256':sha(prepared/'SPLIT_MANIFEST.json'),'data_manifest_sha256':sha(prepared/'DATA_MANIFEST.json'),'learning_businesses':split['groups']['learning'],'scope':'10 verified TRAIN replay targets; not a full pass through TRAIN or a correction/replay round'})
manifest=seal(bundle)
zip_path=bundle.parent/'ZAIC_RUN_811_INPUTS.zip'
with zipfile.ZipFile(zip_path,'x',zipfile.ZIP_DEFLATED) as archive:
    for p in sorted(bundle.rglob('*')):
        if p.is_file():archive.write(p,Path('INPUT_811')/p.relative_to(bundle))
report={'bundle':str(bundle),'bundle_manifest_sha256':manifest,'zip_path':str(zip_path),'zip_bytes':zip_path.stat().st_size,'zip_sha256':sha(zip_path),'train_rows':10,'correction_rows':0,'public_questions':1000,'public_files':len([p for p in (bundle/'PUBLIC').rglob('*') if p.is_file()]),'public_questions_sha256':sha(bundle/'PUBLIC/questions.jsonl'),'public_gold_included':False,'parent_sha256':PARENT_SHA}
atomic(exp/'INPUT_PACKAGE.json',report);print(json.dumps(report))
