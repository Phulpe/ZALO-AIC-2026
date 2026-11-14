"""Doi chieu "Code gợi ý" (hints.py) voi dap an bo train: moi cau co dong code goi y thi chay that trong sandbox va cham theo §9.

Chay: PYTHONPATH=src .venv/bin/python scripts/code_eval.py train - data/train/answers.jsonl > training_code/eval/code_hints_train.txt
Dong TONG: so cau co code goi y dung / sai / loi; khong_co_code = cau de model tu viet code."""
import sys, json, datetime, collections, re, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
from datamate.kernel import BusinessKernel, load_business_cached
from datamate.hints import question_hints, EntityIndex
from datamate.dates import LunarCalendar
from datamate.answers import score

def main():
    split, ids_file, gold_file = sys.argv[1], sys.argv[2], sys.argv[3]
    lun = LunarCalendar.load(f'data/{split}/calendar.csv')
    qs = [json.loads(l) for l in open(f'data/{split}/questions.jsonl', encoding='utf-8')]
    if ids_file != '-':
        ids = set(l.strip() for l in open(ids_file) if l.strip()); qs = [q for q in qs if q['id'] in ids]
    gold = {json.loads(l)['id']: json.loads(l) for l in open(gold_file, encoding='utf-8')}
    by = collections.defaultdict(list)
    for q in qs: by[q['business_id']].append(q)
    res = collections.Counter(); kinds = collections.defaultdict(collections.Counter)
    for b in sorted(by):
        biz = load_business_cached(f'data/{split}/data/{b}', '/tmp/datamate_cache_dev'); idx = EntityIndex.from_business(biz)
        k = None
        for q in by[b]:
            h = question_hints(q['question'], datetime.date.fromisoformat(q['as_of_date']), lun, idx)
            m = re.search(r"Code gợi ý \([^)]*\): (.*)$", h, re.M)
            if not m: res['khong_co_code'] += 1; continue
            code = m.group(1)
            fn = re.search(r"answer\('\w+', (\w+)", code); fn = fn.group(1) if fn else '?'
            if k is None: k = BusinessKernel(f'data/{split}/data/{b}', '/tmp/datamate_cache_dev', f'data/{split}/calendar.csv')
            r = k.execute(q['id'], code, q['as_of_date'])
            g = gold.get(q['id'])
            if r.get('error') or not r.get('answer'):
                res['loi'] += 1; kinds[fn]['loi'] += 1
                print('LOI ', b, q['id'], code[:120], '|', str(r.get('error'))[-150:].replace('\n', ' ')); continue
            sc = score(r['answer'], g) if g else None
            tag = 'dung' if sc == 1 else ('sai' if sc == -1 else 'khac')
            res[tag] += 1; kinds[fn][tag] += 1
            if tag != 'dung':
                print('SAI ' if tag == 'sai' else 'KHAC', b, q['id'], q['question'][:70], '|', code[:110], '| du doan', json.dumps(r['answer'], ensure_ascii=False)[:90], '| gold', json.dumps(g, ensure_ascii=False)[:90])
        if k: k.close()
    print('TONG', dict(res))
    for fn, c in sorted(kinds.items(), key=lambda x: -sum(x[1].values())): print('  ', fn, dict(c))

if __name__ == '__main__': main()
