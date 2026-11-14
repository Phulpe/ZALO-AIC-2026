"""Evaluate full frozen replay once; derive critical TRAIN subgroup on CPU."""
from pathlib import Path
import json,copy
from relearning import verify_export,atomic
from quality_gate import digest
from field_eval import evaluate_rows
from baseline_bridge import BaselineBridge
from scoring import matches

def evaluate_protected(model,tok,bank_root,manifest_sha,parent_sha,allowed_businesses,pool,gold,shops,out,critical_reference,critical_digest,critical_count=7,stop_requested=lambda:False):
    bank_root=Path(bank_root).resolve();out=Path(out).resolve()
    if out==bank_root or out.is_relative_to(bank_root):raise ValueError('Evaluation output overlaps frozen bank')
    if verify_export(bank_root)!=manifest_sha:raise ValueError('Frozen regression bank changed')
    if digest(critical_reference)!=critical_digest:raise ValueError('Critical reference changed before evaluation')
    lineage=json.loads((bank_root/'EVIDENCE_LINEAGE.json').read_text(encoding='utf-8'))
    replay=json.loads((bank_root/'PRIVATE_REGRESSION_BANK.json').read_text(encoding='utf-8'))
    selected=json.loads((bank_root/'PRIVATE_SELECTED.json').read_text(encoding='utf-8'));correction=[r for r in selected if r['kind']=='correction'];rows=correction+replay
    critical={r['id'] for r in critical_reference if r['kind']=='correction'};ids={r['id'] for r in rows};qs={q['id']:q for q in pool}
    if len(critical)!=critical_count or not critical<={r['id'] for r in correction}:raise ValueError('Missing frozen critical corrections')
    if len(replay)!=lineage['full_replay_regression_cases'] or len(ids)!=len(rows):raise ValueError('Incomplete/duplicate regression bank')
    if any(r['kind']!='replay' for r in replay):raise ValueError('Wrong replay role')
    for r in rows:
        if r['parent_sha256']!=parent_sha or r['target_verified'] is not True or r['split']!='learning' or r['business_id'] not in allowed_businesses:raise ValueError('Wrong regression parent/split/evidence')
        q=qs[r['id']]
        if q['split']!='learning' or q['business_id']!=r['business_id'] or q['prompt']!=r['prompt']:raise ValueError('Regression question changed')
    if any(r['parent_sha256']!=parent_sha for r in critical_reference):raise ValueError('Wrong critical parent')
    # Exactly one model generation per unique case, never a second generation
    # after a verdict. All gold access remains evaluation-only.
    full=evaluate_rows(model,tok,rows,pool,gold,shops,out,stop_requested)
    records=json.loads((out/'PRIVATE_PREDICTIONS.json').read_text(encoding='utf-8'));bridge=BaselineBridge(None,None,shops);correct=0
    for key in sorted(critical):
        plan=records[key].get('plan')
        if plan is not None:
            try:correct+=int(bridge.semantic(qs[key],plan) and matches(bridge.execute(qs[key],copy.deepcopy(plan)),gold(key)))
            except Exception:pass
    subset={'mode':'evaluation','scope':'frozen old TRAIN corrections; derived without new generation','expected':len(critical),'attempted':len(critical),'real_output_ids':sum('plan' in records[k] or 'prediction_text' in records[k] for k in critical),'correction_before_correct':0,'correction_after_correct':correct,'correction_total':len(critical),'replay_before_correct':0,'replay_after_correct':0,'correct_to_wrong':0,'weights_updated':False,'source_predictions_sha256':__import__('relearning').sha(out/'PRIVATE_PREDICTIONS.json'),'extra_model_generations':0}
    atomic(out.with_name(out.name+'_CRITICAL_SUMMARY.json'),subset)
    return full,subset
