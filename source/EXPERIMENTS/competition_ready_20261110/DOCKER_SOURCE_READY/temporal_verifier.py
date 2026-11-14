"""Conservative TRAIN-only total-query grammar; no labels, no time guessing."""
import json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'baseline/app'))
from legacy_engine import question_norm,time_bounds,Unsupported
from v2_contracts import expected_question_time,canonical_time

VERSION='temporal_total_grammar_v3'

def canonical_total(q):
    ctx=json.loads(q['prompt']);text=question_norm(ctx['question']).strip(' ?.')
    metric=r'(doanh thu|so luong ban|so luong hang ban|so luong hang da ban|so don hang|so don|don hang)'
    explicit=r'thang (?:0?[1-9]|1[0-2])(?: nam | / |/| )\d{4}'
    short=r'thang (?:0?[1-9]|1[0-2])'
    day=r'ngay \d{1,2} \d{1,2} \d{4}'
    relative=r'(?:ngay )?(?:hom nay|hom qua)|thang nay|thang truoc|thang roi|tuan nay|tuan truoc|tuan roi|nam nay|nam truoc'
    period=r'('+explicit+'|'+short+'|'+day+'|'+relative+')'
    lead=r'(?:(?:cho toi biet|hay cho biet|cho biet|tinh|cho minh biet|giup toi tinh|hay tinh|vui long cho biet) )?'
    tail=r'(?: la bao nhieu| bao nhieu| la bao nhieu a| bao nhieu a)?'
    match=re.fullmatch(lead+r'(?:tong )?'+metric+r' (?:trong |vao |cua )?'+period+tail,text)
    if match:word=match[1]
    else:
        match=re.fullmatch(lead+r'(?:trong |vao )?'+period+r' (?:tong )?'+metric+tail,text)
        if match:word=match[2]
        else:
            match=re.fullmatch(lead+r'(?:co )?bao nhieu don hang (?:trong |vao )?'+period,text)
            if not match:return None
            word='don hang'
    try:time=expected_question_time(ctx['question'],q['as_of_date'],question_norm)
    except (ValueError,TypeError):return None
    if time is None:return None
    metric_name='revenue' if word=='doanh thu' else 'orders' if word in {'so don','so don hang','don hang'} else 'quantity'
    return {'metric':metric_name,'time':time,'filters':[],'group_by':None,'order':'desc','limit':None,'unit':None,'answer_type':'number','unresolved':[]}

def same_time(left,right,as_of):
    """Compare resolved intervals with source anchor, without question overrides."""
    try:
        a=canonical_time(left);b=canonical_time(right)
        return time_bounds({'time':a},as_of)==time_bounds({'time':b},as_of)
    except (ValueError,TypeError,KeyError,Unsupported):return False
