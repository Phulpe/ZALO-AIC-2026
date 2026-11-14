"""Strict time contracts and lexical rule retrieval; no answer or test-ID lookup."""
from __future__ import annotations
from datetime import date
import calendar
import re


class ContractError(ValueError):
    pass


def iso_day(value):
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        raise ContractError('Day must be an explicit ISO date')
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise ContractError('Invalid calendar date') from exc


def canonical_time(value):
    if not isinstance(value, dict) or not isinstance(value.get('kind'), str):
        raise ContractError('Invalid time object')
    t = dict(value)
    kind = t['kind']
    if kind == 'date':
        kind = t['kind'] = 'day'
    if kind in {'day', 'until'}:
        if set(t) not in ({'kind', 'date'}, {'kind', 'value'}):
            raise ContractError('Day/until requires exactly one date')
        return {'kind': kind, 'date': iso_day(t.get('date', t.get('value')))}
    if kind == 'month':
        if set(t) != {'kind', 'value'} or not isinstance(t['value'], str) or not re.fullmatch(r'\d{4}-(?:0[1-9]|1[0-2])', t['value']):
            raise ContractError('Month must be YYYY-MM; a day is not a month')
        iso_day(t['value'] + '-01')
        return t
    if kind == 'year':
        y = t.get('year', t.get('value'))
        if set(t) not in ({'kind', 'year'}, {'kind', 'value'}) or isinstance(y, bool) or not re.fullmatch(r'\d{4}', str(y)):
            raise ContractError('Invalid explicit year')
        return {'kind': 'range', 'start': iso_day(f'{int(y):04d}-01-01'), 'end': iso_day(f'{int(y):04d}-12-31')}
    if kind == 'quarter':
        if set(t) == {'kind', 'value'}:
            m = re.fullmatch(r'(\d{4})-Q([1-4])', str(t['value']), re.I)
            if not m:
                raise ContractError('Quarter must be YYYY-Q1..Q4')
            y, q = map(int, m.groups())
        elif set(t) == {'kind', 'year', 'quarter'} and type(t['year']) is int and type(t['quarter']) is int:
            y, q = t['year'], t['quarter']
        else:
            raise ContractError('Invalid quarter fields')
        if not 1 <= q <= 4:
            raise ContractError('Invalid quarter number')
        month = q * 3
        return {'kind': 'range', 'start': iso_day(f'{y:04d}-{month-2:02d}-01'), 'end': iso_day(f'{y:04d}-{month:02d}-{calendar.monthrange(y, month)[1]:02d}')}
    if kind == 'range':
        if set(t) != {'kind', 'start', 'end'}:
            raise ContractError('Range requires start and end only')
        start, end = iso_day(t['start']), iso_day(t['end'])
        if start > end:
            raise ContractError('Reversed time range')
        return {'kind': kind, 'start': start, 'end': end}
    if kind == 'rolling_days':
        if set(t) != {'kind', 'n'} or type(t['n']) is not int or not 1 <= t['n'] <= 3660:
            raise ContractError('Invalid rolling day count')
        return t
    relative = {'none', 'today', 'yesterday', 'this_week', 'last_week', 'this_month', 'last_month', 'this_year', 'last_year', 'month_to_date', 'year_to_date'}
    if kind in relative and set(t) == {'kind'}:
        return t
    raise ContractError('Unsupported or malformed time expression')


def select_rules(text, question, normalize, limit=6, width=420, budget=1320):
    """Retrieve complete short rules from all sections, not just the document start."""
    query = set(normalize(question).split())
    themes = [
        ({'doanh', 'thu', 'tien', 'ban'}, {'doanh', 'thu', 'giam', 'thue', 'vat', 'phi', 'huy', 'tra'}),
        ({'luong', 'chay', 'nhieu', 'don', 'vi'}, {'luong', 'quy', 'doi', 'don', 'vi'}),
        ({'khach'}, {'khach', 'ma', 'vang', 'lai'}),
        ({'no'}, {'no', 'cod', 'tra', 'thu'}),
        ({'ton', 'kho'}, {'ton', 'kho', 'nhap', 'dau'}),
        ({'lai', 'von'}, {'lai', 'von', 'lo', 'chi', 'phi'}),
        ({'thang', 'quy', 'ngay', 'tuan', 'tet'}, {'thang', 'quy', 'ngay', 'tuan', 'tet', 'nam'}),
    ]
    expanded = set(query)
    for trigger, related in themes:
        if query & trigger:
            expanded |= related
    rows = []
    for index, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line or line.startswith('#') or re.fullmatch(r'[| :\-]+', line):
            continue
        if not (line.startswith('- ') or line.startswith('|')):
            continue
        # Do not extract arbitrary column dictionaries or cut a rule mid-sentence.
        if line.startswith('- **'):
            continue
        cleaned = re.sub(r'[*`]', '', line).strip('| ').replace('|', ': ')
        tokens = set(normalize(cleaned).split())
        score = 3 * len(tokens & query) + len(tokens & expanded)
        if score and len(cleaned) <= width:
            rows.append((score, index, cleaned))
    rows.sort(key=lambda row: (-row[0], row[1]))
    result=[]
    for _,_,line in rows:
        if len(result)>=limit:break
        if sum(map(len,result))+len(line)<=budget:result.append(line)
    return result


def check_question_semantics(question, plan, normalize):
    """Reject clear mismatches; never infer a numeric answer or consult labels."""
    q=normalize(question)
    metric=plan['metric']
    threshold=(re.search(r'(?:it nhat|toi thieu|it hon|nhieu hon)\s+\d+(?:\s+\d+)*\s*(?:lan|don|dong|nghin|trieu)\b',q)
               or re.search(r'\btu\s+\d+(?:\s+\d+)*\s*(?:dong|nghin|trieu|k|tr)\b',q)
               or re.search(r'\btu\s+\d+\s+tro len\b',q))
    if threshold or re.search(r'[<>]=?\s*\d',question):
        raise ContractError('Numeric condition requires a verified comparator operation')
    if 'lien tiep' in q:raise ContractError('Consecutive-period condition requires its own operation')
    measures=sum(bool(re.search(pattern,q)) for pattern in [r'\bdoanh thu\b',r'\blai gop\b',r'\bso don\b'])
    if measures>1 and ' va ' in q:raise ContractError('Multiple requested measures require a composite output plan')
    if any(x in q for x in ['ty le','ti le','phan tram','ty trong','ti trong','chiem bao nhieu']):
        if plan['answer_type']!='percent':raise ContractError('Question requests a percentage, not an absolute number')
        if metric=='cancellation_rate' and not any(x in q for x in ['huy','cancel']):raise ContractError('Cancellation percentage does not answer this ratio question')
    if any(x in q for x in ['bao nhieu khach','so khach','bao nhieu loai','so loai hang','so mat hang']):
        raise ContractError('Distinct-entity count requires its own verified operation')
    if 'lai rong' in q or 'loi nhuan rong' in q:raise ContractError('Net profit requires expense execution')
    if any(x in q for x in ['gia ban','don gia','gia nhap']) and not ('binh quan' in q and metric=='avg_import_cost'):
        raise ContractError('A unit-price request cannot use an aggregate metric')
    required=None
    if any(x in q for x in ['lai gop','loi nhuan gop','hang loi nhat']):required='gross_profit'
    elif any(x in q for x in ['gia tri trung binh mot don','gia tri trung binh moi don','trung binh moi don','trung binh mot don']):required='avg_order_value'
    elif 'doanh thu' in q:required='revenue'
    elif re.search(r'\b(?:so don|bao nhieu don|may don)\b',q):
        required='cancellation_rate' if any(x in q for x in ['ty le','ti le','phan tram']) else 'cancelled_orders' if 'huy' in q else 'orders'
    elif any(x in q for x in ['ban chay','hot nhat','so luong ban','tong so luong','ban nhieu nhat']):required='quantity'
    if required and metric!=required:raise ContractError('Metric contradicts the explicit requested measure')
    if plan['group_by']=='product' and any(x in q for x in ['theo khach','khach nao','khach hang nao']):
        raise ContractError('Product grouping contradicts customer ranking')
    if plan['group_by']=='customer' and any(x in q for x in ['san pham nao','mat hang nao','mon nao']):
        raise ContractError('Customer grouping contradicts product ranking')


def expected_question_time(question, as_of, normalize):
    """Independent anchors for single-period questions. Ambiguous cases return None."""
    q=normalize(question)
    if any(x in q for x in ['so voi','chenh lech','lien tuc','ca sang','ca chieu','truoc tet','sau tet']):return None
    # Explicit date ranges are inclusive. Require the date keyword to avoid
    # interpreting product names or model numbers as dates.
    dates=re.findall(r'(?:ngày|ngay)\s+(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})',question,re.I)
    if dates:
        ds=[iso_day(f'{int(y):04d}-{int(m):02d}-{int(d):02d}') for d,m,y in dates]
        if len(ds)==2 and 'tu ' in q and ' den ' in q:return {'kind':'range','start':ds[0],'end':ds[1]}
        if len(ds)==1 and not any(x in q for x in ['tu ',' den ','truoc ','sau ']):return {'kind':'day','date':ds[0]}
        return None
    relative=[('hom qua','yesterday'),('hom nay','today'),('thang nay','this_month'),('thang truoc','last_month'),('thang roi','last_month'),('tuan nay','this_week'),('tuan truoc','last_week'),('tuan roi','last_week'),('nam nay','this_year'),('nam truoc','last_year')]
    if 'tu dau nam' in q:return {'kind':'this_year'}
    for phrase,kind in relative:
        if phrase in q:
            if kind=='today' and any(x in q for x in ['tinh den','den het','con no','con ton']):return None
            return {'kind':kind}
    raw=question.lower()
    months=re.findall(r'(?:tháng|thang)\s*(\d{1,2})(?:\s*[/.-]\s*|\s+(?:năm|nam)\s+)(\d{4})',raw)
    if len(months)==1:
        m,y=map(int,months[0]);return canonical_time({'kind':'month','value':f'{y:04d}-{m:02d}'})
    if len(months)>1:return None
    short=re.findall(r'\bthang\s+([1-9]|1[0-2])\b',q)
    if len(short)==1 and not re.search(r'\bnam\s+\d{4}\b',q):
        now=date.fromisoformat(iso_day(as_of));m=int(short[0]);y=now.year-(m>now.month)
        return {'kind':'month','value':f'{y:04d}-{m:02d}'}
    return None
