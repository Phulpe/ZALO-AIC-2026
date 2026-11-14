"""Versioned TRAIN executor: remove only source-proven annotation rows.

Never fills money, dates or IDs; never weakens the frozen planner's guards.
The source table, before metadata forward fill, must prove the row is a note.
"""
from copy import copy
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent/'baseline/app'))
import re,json,hashlib
import pandas as pd
from planner import execute as frozen_execute,PlanError

VERSION='source_annotation_repair_v2'
NOTE=re.compile(r'^\s*(?:ghi chú|ghi chu|lưu ý|luu y|chú ý|chu y)(?:\s*[:：]|\s*$)',re.I)

def empty(value):
    return value is None or (not isinstance(value,(list,dict,tuple)) and bool(pd.isna(value))) or isinstance(value,str) and not value.strip()

def annotation_evidence(shop):
    raw=shop.tables.get('sales',pd.DataFrame());approved=set();proof=[]
    if not {'_source','product','date','qty','price','line_total'}<=set(raw):return approved,proof
    sources={a['file']+':'+str(a['sheet']) for a in shop.audit if a['kind']=='sales'}
    for index,row in raw.iterrows():
        label=row.get('product');source=row.get('_source')
        if source not in sources or not isinstance(label,str) or not NOTE.match(label):continue
        if not all(empty(row.get(k)) for k in ['date','order_id','qty','price','line_total','unit','base_qty','base_unit']):continue
        approved.add((source,label));proof.append({'raw_index':str(index),'source_ref_sha256':hashlib.sha256(source.encode()).hexdigest(),'label_sha256':hashlib.sha256(label.encode()).hexdigest(),'source_missing_identity_date_amounts':True})
    return approved,proof

def repair_shop(shop,source_proofs=None):
    candidates,cache_proof=annotation_evidence(shop);frame=shop.lines
    approved,proof=source_proofs if source_proofs is not None else (set(),[])
    approved=candidates&approved
    if not approved or not {'_source','product','qty','price','line_total'}<=set(frame):return shop,{'version':VERSION,'removed':0,'proof':proof}
    drop=[]
    counts=frame.groupby(['_source','product'],dropna=False).size()
    for position,(_,row) in enumerate(frame.iterrows()):
        pair=(row.get('_source'),row.get('product'))
        if pair in approved and counts.get(pair,0)==1 and all(empty(row.get(k)) for k in ['qty','price','line_total','unit','base_qty','base_unit']):drop.append(position)
    if not drop:return shop,{'version':VERSION,'removed':0,'proof':proof}
    result=copy(shop);removed=set(drop);result.lines=frame.iloc[[i for i in range(len(frame)) if i not in removed]].copy(deep=True)
    if result.lines.empty:raise PlanError('No verified transaction lines after annotation repair')
    return result,{'version':VERSION,'removed':len(drop),'proof':proof,'preserved_original':True,'amounts_imputed':0}

def execute_learning(shop,question,as_of,plan,split,source_proofs=None):
    if split!='learning':raise ValueError('Learning executor cannot be used on evaluation/submission split')
    try:return frozen_execute(shop,question,as_of,plan)
    except PlanError as exc:
        if str(exc)!='Unparsed numeric amounts' or source_proofs is None or plan.get('metric')!='revenue' or plan.get('group_by') is not None or plan.get('filters'):raise
        proofs=source_proofs() if callable(source_proofs) else source_proofs
        repaired,_=repair_shop(shop,proofs)
        if repaired is shop:raise
        return frozen_execute(repaired,question,as_of,plan)
