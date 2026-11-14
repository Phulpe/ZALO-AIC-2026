"""Bind orchestration to the unchanged business executor and scorer.

No gold enters generation. Accepted correction plans are supplied by a trusted
local correction provider and revalidated here. This module cannot invent one.
"""
from pathlib import Path
import json, sys, re
sys.path.insert(0,str(Path(__file__).parent/'baseline/app'))
from planner import validate_plan, execute, parse_plan, SYSTEM
from v2_contracts import check_question_semantics,expected_question_time,ContractError
from legacy_engine import question_norm
from scoring import matches
from prediction_errors import BlindPredictionError

class BaselineBridge:
    def __init__(self,model,tokenizer,shops,max_tokens=320,max_seconds=30):
        self.model=model;self.tok=tokenizer;self.shops=shops
        if not 1<=max_tokens<=320 or not 0<max_seconds<=60:raise ValueError('Generation budget')
        self.max_tokens=max_tokens;self.max_seconds=max_seconds
    def predict(self,q,attempt):
        import torch
        prompt=self.tok.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':q['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        inputs=self.tok(prompt,return_tensors='pt',add_special_tokens=False).to('cuda:0')
        if inputs['input_ids'].shape[1]>2048:raise ValueError('Context overlength; retain in ledger')
        self.model.eval()
        with torch.inference_mode():
            generated=self.model.generate(**inputs,max_new_tokens=self.max_tokens,max_time=self.max_seconds,do_sample=False,use_cache=True,pad_token_id=self.tok.pad_token_id,eos_token_id=self.tok.eos_token_id)
        text=self.tok.decode(generated[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
        plan=parse_plan(text)
        if plan is None:raise BlindPredictionError(text)
        return plan
    def execute(self,q,plan):
        context=json.loads(q['prompt'])
        if context['as_of_date']!=q['as_of_date']:raise ValueError('as_of mismatch')
        return execute(self.shops[q['source_ref']],context['question'],q['as_of_date'],plan)
    def semantic(self,q,plan):
        context=json.loads(q['prompt'])
        try:check_question_semantics(context['question'],plan,question_norm)
        except ContractError:return False
        canonical=canonical_total(q)
        if canonical is None:return False
        actual=dict(plan);expected=dict(canonical)
        # Ordering has no effect for a single aggregate without grouping.
        if actual.get('group_by') is None:actual.pop('order',None);expected.pop('order',None)
        return actual==expected

def canonical_total(q):
    """Conservative independent semantic verifier/correction for monthly totals.

    Unsupported questions remain unresolved, never broadened heuristically.
    The scorer still must verify the executed answer against learning gold.
    """
    ctx=json.loads(q['prompt']);text=question_norm(ctx['question']).strip(' ?.')
    match=re.fullmatch(r'(?:tong )?(doanh thu|so luong ban|so don hang) (?:trong )?thang (\d{1,2})(?: nam | / |/| )(\d{4})(?: la bao nhieu)?',text)
    if not match:return None
    metric={'doanh thu':'revenue','so luong ban':'quantity','so don hang':'orders'}[match[1]]
    period=expected_question_time(ctx['question'],q['as_of_date'],question_norm)
    if period is None or period['kind']!='month':return None
    return {'metric':metric,'time':period,'filters':[],'group_by':None,'order':'desc','limit':None,'unit':None,'answer_type':'number','unresolved':[]}

def canonical_correction(q,row,gold):
    # Does not derive a fake plan from gold or a mistaken model prediction.
    plan=canonical_total(q)
    if plan is None:raise ValueError('Unsupported correction; keep unresolved')
    return plan

def verified_semantics(bridge,evidence_reader):
    from relearning import fingerprint
    def check(q,plan):
        context=json.loads(q['prompt'])
        check_question_semantics(context['question'],plan,question_norm)
        evidence=evidence_reader(q['id'],fingerprint(plan))
        return bool(evidence and evidence.get('plan_sha256')==fingerprint(plan) and evidence.get('prompt_sha256')==fingerprint(q['prompt']) and evidence.get('semantic_verified') is True and evidence.get('source_manifest_sha256') and evidence.get('verifier_version'))
    return check
