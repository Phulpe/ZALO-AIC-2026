"""Revalidate real bank against persisted blind ledger before selection."""
import json
from pathlib import Path
from quality_gate import select,digest
from relearning import atomic,sha

def export(run,pool,bridge,gold,config):
 run=Path(run);bank=json.loads((run/'verified_bank.json').read_text());ledger=json.loads((run/'ledger.json').read_text());qs={q['id']:q for q in pool};evidence={};rows=[];excluded={}
 for row in bank['correction']+bank['replay']:
  q=qs[row['id']];record=ledger['questions'][row['id']]
  if not record['attempts'] or not any('plan' in a or 'prediction_text' in a for a in record['attempts']):raise ValueError('No persisted blind prediction')
  first=record['first_verified']
  if row['kind']=='replay' and first is not True:excluded[row['id']]='retry-recovered replay lacks first-pass success';continue
  plan=json.loads(row['response'])
  from planner import validate_plan
  from baseline_bridge import matches
  validate_plan(plan)
  if not bridge.semantic(q,plan) or not matches(bridge.execute(q,plan),gold(q['id'])):raise ValueError('Revalidation failed')
  ev={k:config[k] for k in ('parent_sha256','source_manifest_sha256','split_manifest_sha256','verifier_sha256','scorer_sha256')}
  ev.update({k:True for k in ('schema','semantic','execution','scorer','blind_prediction_persisted','gold_after_prediction')});ev.update(plan_sha256=digest(plan),prompt_sha256=digest(row['prompt']),first_pass_verified_correct=first)
  evidence[row['id']]=ev;rows.append(row)
 atomic(run/'PRIVATE_TARGET_EVIDENCE.json',evidence)
 try:
  dataset,report=select(rows,evidence,config);atomic(run/'QUALITY_DATASET.json',dataset);report.update(state='passed',optimizer_executed=False,dataset_file_sha256=sha(run/'QUALITY_DATASET.json'))
 except ValueError as exc:report={'state':'blocked','reason':str(exc),'optimizer_executed':False,'available_correction':sum(r['kind']=='correction' for r in rows),'available_replay':sum(r['kind']=='replay' for r in rows)}
 report['excluded_retry_replay']=len(excluded);atomic(run/'QUALITY_GATE_RESULT.json',report)
 return report
