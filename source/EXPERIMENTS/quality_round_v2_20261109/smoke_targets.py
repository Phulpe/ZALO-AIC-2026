"""Two synthetic plan targets verified by baseline executor, no contest data."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).parent/'baseline/app'))
from test_core import fixture
from planner import context,prompt_text,validate_plan
from baseline_bridge import BaselineBridge,canonical_total,matches

def targets():
    shop=fixture();bridge=BaselineBridge(None,None,{'fixture':shop});rows=[]
    for i,(text,gold) in enumerate([('<REDACTED_COMPETITION_QUESTION>',300),('Tổng số lượng bán tháng 1 năm 2026 là bao nhiêu?',12)]):
        prompt=prompt_text(context(shop,text,'2026-01-31',''))
        q={'id':f'synthetic_{i}','prompt':prompt,'source_ref':'fixture','as_of_date':'2026-01-31'}
        plan=canonical_total(q);validate_plan(plan)
        assert bridge.semantic(q,plan) and matches(bridge.execute(q,plan),{'type':'number','answer':gold})
        rows.append({'id':q['id'],'business_id':'synthetic','prompt':prompt,'response':json.dumps(plan,ensure_ascii=False),'target_verified':True,'split':'learning'})
    return rows
