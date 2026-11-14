"""Verify historical parent-SFT replay targets locally; print aggregates only."""
from pathlib import Path
import sys,json,datetime,hashlib,collections
workspace=Path('D:/ZAIC')
source=workspace/'EXPERIMENTS/relearning_v3'
sys.path[:0]=[str(source),str(source/'baseline/app')]
from baseline_bridge import BaselineBridge,matches
from legacy_engine import prepare_shop,question_norm
from relearning import sha,atomic,seal

runtime=workspace/'PRIVATE_DATA/relearning_v2_prepared/20261108T070936492119Z'
old=workspace/'PRIVATE_DATA/zaic/datamate_v1/qwen35_9b/data/sft_train.jsonl'
assert sha(old)=='5057658ac10c19579c675c9f926cc13f1e17641b88eb312318e48be75e6bb170'
assert sha(runtime/'DATA_MANIFEST.json')=='dc5f6c3a4894a61d3048a91484f6ca0887021ddb901777ab340353b0e702ccad'
split=json.loads((runtime/'SPLIT_MANIFEST.json').read_text(encoding='utf-8'))
pool={q['id']:q for q in map(json.loads,(runtime/'learning/model_pool.jsonl').read_text(encoding='utf-8').splitlines())}
gold={q['id']:q for q in map(json.loads,(runtime/'learning/gold.jsonl').read_text(encoding='utf-8').splitlines())}
manifest=json.loads((runtime/'DATA_MANIFEST.json').read_text(encoding='utf-8'))
runtime_manifest=json.loads((runtime/'RUNTIME_DATA_MANIFEST.json').read_text(encoding='utf-8'))
assert sha(runtime/'learning/gold.jsonl')==manifest['files']['learning/gold.jsonl']['sha256']
rows=[json.loads(l) for l in old.read_text(encoding='utf-8').splitlines() if l.strip()]
selected=[r for r in rows if r['id'] in set(split['pilot_ids']) and r['id'] in pool]
shops={};replay={};errors=collections.Counter()
for row in selected:
    try:
        q=pool[row['id']];assert row['business_id']==q['business_id']
        assert q['business_id'] in split['groups']['learning'] and q['split']=='learning'
        old_context=json.loads(row['prompt']);context=json.loads(q['prompt'])
        assert question_norm(old_context['question'])==question_norm(context['question'])
        assert old_context['as_of_date']==q['as_of_date']
        plan=json.loads(row['response']);biz=q['business_id']
        if biz not in shops:
            directory=runtime/'data'/biz
            for p in directory.rglob('*'):
                if p.is_file():assert sha(p)==runtime_manifest['files'][p.relative_to(runtime).as_posix()]['sha256']
            shops[biz]=prepare_shop(directory,cache_dir=None)
        bridge=BaselineBridge(None,None,shops)
        assert bridge.semantic(q,plan)
        assert matches(bridge.execute(q,plan),gold[q['id']])
        item={'id':q['id'],'prompt':q['prompt'],'response':row['response'],'business_id':biz,'kind':'replay','split':'learning','target_verified':True,'parent_sha256':'e7dfa635beccce128c99f9d3fd993d33dbb4c88e73e3ddeccbc2bc915eec4a4d','origin':'verified_historical_parent_training_sft','model_prediction_correct_not_asserted':True,'source_sft_sha256':sha(old)}
        if q['id'] in replay:assert json.loads(replay[q['id']]['response'])==plan
        replay[q['id']]=item
    except Exception as exc:errors[type(exc).__name__]+=1
out=workspace/'PRIVATE_DATA/relearning_replay_seed'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
out.mkdir(parents=True,exist_ok=False)
atomic(out/'verified_replay_seed.json',list(replay.values()))
report={'source_sft_sha256':sha(old),'selected_rows':len(selected),'verified_unique_replay':len(replay),'rejected_error_types':dict(errors),'source':'historical parent training targets, rechecked against TRAIN source/executor/scorer','current_model_accuracy_measured':False,'split':'learning','data_manifest_sha256':sha(runtime/'DATA_MANIFEST.json'),'split_manifest_sha256':sha(runtime/'SPLIT_MANIFEST.json')}
atomic(out/'SUMMARY.json',report);seal(out)
atomic(workspace/'EXPERIMENTS/replay_seed_preparation_20261108/RESULT.json',{**report,'private_output_path':str(out)})
print(json.dumps(report))
