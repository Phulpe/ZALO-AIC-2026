"""Frozen calibration/public inference. Gold opens only after predictions persist."""
from pathlib import Path
import sys,json,tempfile,shutil
sys.path[:0]=[str(Path(__file__).parent),str(Path(__file__).parent/'baseline/app')]
from relearning import sha,atomic

def evaluate(config,adapter,output,public=False):
 from scoring import matches
 out=Path(output);out.mkdir(parents=True,exist_ok=False)
 source=Path(config['public_root'] if public else config['runtime_root'])
 questions=source/'questions.jsonl' if public else source/'calibration/questions.jsonl'
 expected=config['public_questions_sha256'] if public else config['calibration_questions_sha256']
 assert sha(questions)==expected
 with tempfile.TemporaryDirectory() as scratch:
  p=Path(scratch);(p/'assets').mkdir();(p/'assets/base').symlink_to(Path(config['base']),target_is_directory=True);(p/'app').symlink_to(Path(__file__).parent/'baseline/app',target_is_directory=True);(p/'candidate').mkdir();(p/'candidate/final').symlink_to(Path(adapter),target_is_directory=True);shutil.copy2(Path(__file__).parent/'baseline/model_lock.json',p/'model_lock.json')
  data=source # Keep fingerprint paths rooted at the real immutable dataset root.
  import public_infer_811
  sys.argv=['inference','--project',str(p),'--root',str(data),'--questions',str(questions),'--variant','candidate','--output-dir',str(out/'predictions'),'--batch-size','4','--cache-dir',str(p/'cache'),'--max-input-tokens','3072']
  public_infer_811.main()
 rows=[json.loads(l) for l in (out/'predictions/submission.jsonl').read_text().splitlines() if l.strip()];qs=[json.loads(l) for l in questions.read_text(encoding='utf-8-sig').splitlines() if l.strip()]
 runtime=json.loads((out/'predictions/runtime.json').read_text());assert runtime['state']=='completed' and runtime['question_logs']==len(qs) and runtime['network_attempts']==0 and [r['id'] for r in rows]==[q['id'] for q in qs]
 if public:
  assert len(rows)==1000;atomic(out/'PUBLIC_VALIDATION.json',{'state':'completed','rows':1000,'submission_sha256':sha(out/'predictions/submission.jsonl'),'public_gold_accessed':False,'runtime':{k:v for k,v in runtime.items() if k not in ('batch_timings','log_path')}});return
 # Predictions and journal are complete before evaluation-only labels are opened.
 assert sha(source/'calibration/gold.jsonl')==config['calibration_gold_sha256']
 gold={r['id']:r for r in map(json.loads,(source/'calibration/gold.jsonl').read_text().splitlines())};assert set(gold)=={q['id'] for q in qs}
 verdicts={row['id']:bool(matches(row,gold[row['id']])) for row in rows};correct=sum(verdicts.values());abstain=sum(row['type']=='abstain' for row in rows);wrong=len(rows)-correct-abstain
 atomic(out/'EVALUATION.json',{'manifest_sha256':config['calibration_manifest_sha256'],'decoder_sha256':sha(Path(__file__).parent/'public_infer_811.py'),'scorer_sha256':sha(Path(__file__).parent/'baseline/app/scoring.py'),'verdicts':verdicts,'metric':(correct-wrong)/len(rows),'correct':correct,'wrong':wrong,'abstain':abstain,'n':len(rows),'gates_passed':True,'weights_updated':False,'gold_after_predictions':True,'exposure':'historically exposed calibration'})
if __name__=='__main__':
 c=json.loads(Path(sys.argv[1]).read_text());evaluate(c,sys.argv[2],sys.argv[3],len(sys.argv)>4 and sys.argv[4]=='public')
