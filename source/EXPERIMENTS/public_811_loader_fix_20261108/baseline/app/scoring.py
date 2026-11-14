"""Official matching specification; train/development labels only."""
import math,re,unicodedata
from decimal import Decimal,ROUND_HALF_UP

def text_key(value):
    s=unicodedata.normalize('NFC',str(value)).lower()
    return ' '.join(''.join(c if c.isalpha() or c.isdigit() or c.isspace() else ' ' for c in s).split())

def numeric(x):return type(x) in {int,float} and math.isfinite(x)
def rounded(x):return Decimal(str(x)).quantize(Decimal('.01'),rounding=ROUND_HALF_UP)
def matches(pred,gold):
    typ=gold['type']
    if pred.get('type')!=typ:return False
    if typ=='unanswerable':return True
    if typ=='abstain':return False
    a=pred.get('answer');b=gold.get('answer');aliases=gold.get('alias') or []
    if typ=='number':return numeric(a) and numeric(b) and a==b
    if typ=='percent':return numeric(a) and numeric(b) and rounded(a)==rounded(b)
    if typ=='boolean':return type(a) is bool and a==b
    if typ=='date':return isinstance(a,str) and a==b
    if typ=='string':return isinstance(a,str) and any(text_key(a)==text_key(x) for x in [b,*aliases])
    def element(x,y,alt=()):
        if numeric(y):return numeric(x) and x==y
        return isinstance(x,str) and any(text_key(x)==text_key(z) for z in [y,*alt])
    if typ=='list':
        return isinstance(a,list) and len(a)==len(b) and all(element(x,y,aliases[i] if i<len(aliases) and isinstance(aliases[i],list) else []) for i,(x,y) in enumerate(zip(a,b)))
    if typ=='table':
        if not isinstance(a,dict) or set(a)!={'columns','rows'} or len(a['columns'])!=len(b['columns']) or len(a['rows'])!=len(b['rows']):return False
        order=[]
        for i,col in enumerate(b['columns']):
            js=[j for j,c in enumerate(a['columns']) if text_key(c)==text_key(col)]
            order.append(js[0] if js else i)
        if len(set(order))!=len(order):return False
        pending=list(a['rows'])
        for i,row in enumerate(b['rows']):
            alt=aliases[i] if i<len(aliases) and isinstance(aliases[i],list) else []
            found=next((j for j,r in enumerate(pending) if len(r)==len(row) and all(element(r[order[k]],v,alt if k==0 else []) for k,v in enumerate(row))),None)
            if found is None:return False
            pending.pop(found)
        return True
    return False

def summarize(rows):
    c=sum(r['state']=='correct' for r in rows);w=sum(r['state']=='wrong' for r in rows);a=len(rows)-c-w
    return {'n':len(rows),'correct':c,'wrong':w,'abstain':a,'PA':(c-w)/len(rows) if rows else None,'answered_accuracy':c/(c+w) if c+w else None}
