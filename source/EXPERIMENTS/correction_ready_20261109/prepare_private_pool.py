"""Prepare TRAIN-only runtime assets locally. Emit aggregate counts only."""
from pathlib import Path
import sys,json,hashlib,contextlib,datetime,shutil
sys.path.insert(0,str(Path(__file__).parent/'baseline/app'))
from legacy_engine import prepare_shop
from planner import context,prompt_text
from baseline_bridge import canonical_total
from relearning import sha,atomic

def prepare():
    source=Path('D:/ZAIC/PRIVATE_DATA/zaic/train')
    output=Path('D:/ZAIC/PRIVATE_DATA/relearning_v2_prepared')/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True,exist_ok=False)
    groups={'learning':{f'biz_{i}' for i in [102,103,104,105,107,108,110,113,115,116,118,119,120,122,123,126,129,130]},'calibration':{f'biz_{i}' for i in [101,109,112,114,121,128]},'audit_exposed':{f'biz_{i}' for i in [106,111,117,124,125,127]}}
    assert not groups['learning']&groups['calibration'] and not groups['learning']&groups['audit_exposed']
    questions=[json.loads(l) for l in (source/'questions.jsonl').read_text(encoding='utf-8-sig').splitlines() if l.strip()]
    assert len({q['id'] for q in questions})==len(questions)
    labels={q['id']:role for role,businesses in groups.items() for q in questions if q['business_id'] in businesses}
    assert len(labels)==len(questions)
    def jsonl(path,rows):
        with path.open('x',encoding='utf-8') as f:
            for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    # Partition known TRAIN labels for storage isolation. Prediction providers
    # only receive model_pool records; gold_reader opens labels after prediction.
    gold=[json.loads(l) for l in (source/'answers.jsonl').read_text(encoding='utf-8-sig').splitlines() if l.strip()]
    assert {r['id'] for r in gold}==set(labels)
    for role in groups:
        (output/role).mkdir()
        jsonl(output/role/'questions.jsonl',[q for q in questions if labels[q['id']]==role])
        jsonl(output/role/'gold.jsonl',[r for r in gold if labels[r['id']]==role])
    rulebook=(source/'rulebook.md').read_text(encoding='utf-8-sig')
    shutil.copy2(source/'rulebook.md',output/'rulebook.md');shutil.copy2(source/'calendar.csv',output/'calendar.csv')
    pool=[];supported=[];errors=[];source_files={}
    with (output/'PRIVATE_PREPARE.log').open('x',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        for business in sorted(groups['learning']):
            try:shop=prepare_shop(source/'data'/business,cache_dir=None)
            except Exception as exc:shop=None;errors.append({'business':business,'error_type':type(exc).__name__})
            for q in [q for q in questions if q['business_id']==business]:
                try:
                    if shop is None:raise ValueError('Source ingestion unavailable')
                    prompt=prompt_text(context(shop,q['question'],q['as_of_date'],rulebook))
                except Exception as exc:
                    # Keep the question in the expected pool; runtime must record
                    # an ingestion error instead of dropping it.
                    prompt=json.dumps({'question':q['question'],'as_of_date':q['as_of_date'],'ingestion_error':type(exc).__name__},ensure_ascii=False)
                record={'id':q['id'],'business_id':business,'split':'learning','prompt':prompt,'source_ref':business,'as_of_date':q['as_of_date']}
                pool.append(record)
                if canonical_total(record) is not None:supported.append(q['id'])
            for p in sorted((source/'data'/business).rglob('*')):
                if p.is_file():source_files[p.relative_to(source).as_posix()]={'bytes':p.stat().st_size,'sha256':sha(p)}
    assert len(pool)==sum(role=='learning' for role in labels.values())
    jsonl(output/'learning/model_pool.jsonl',pool)
    atomic(output/'SPLIT_MANIFEST.json',{'groups':{k:sorted(v) for k,v in groups.items()},'counts':{k:sum(v==k for v in labels.values()) for k in groups},'historical_exposure':True,'audit_is_untouched':False,'pilot_ids':supported,'pilot_selection':'strict canonical support based only on question; gold not used for selection','remaining_learning_ids':[q['id'] for q in pool if q['id'] not in supported]})
    files={p.relative_to(output).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in output.rglob('*') if p.is_file() and p.name!='PRIVATE_PREPARE.log'}
    atomic(output/'DATA_MANIFEST.json',{'source':'TRAIN only','source_questions_sha256':sha(source/'questions.jsonl'),'source_gold_sha256':sha(source/'answers.jsonl'),'files':files,'business_source_files':source_files,'gold_access':'scorer reader only after persisted blind prediction','synthetic':False,'correction_replay_dataset_ready':False})
    summary={'path':str(output),'learning_pool':len(pool),'pilot_supported_questions':len(supported),'calibration':sum(v=='calibration' for v in labels.values()),'audit_exposed':sum(v=='audit_exposed' for v in labels.values()),'ingestion_business_errors':len(errors),'data_manifest_sha256':sha(output/'DATA_MANIFEST.json'),'split_manifest_sha256':sha(output/'SPLIT_MANIFEST.json'),'correction_replay_dataset_ready':False}
    atomic(Path(__file__).parent/'PRIVATE_POOL_PREPARATION.json',summary)
    print(json.dumps(summary,indent=2))
    return output
if __name__=='__main__':prepare()
