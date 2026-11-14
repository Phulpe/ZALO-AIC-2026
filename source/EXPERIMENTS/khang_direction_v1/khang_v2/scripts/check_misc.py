"""Doi chieu: mat hang e (unsold_items), ngay ban nhieu tien nhat / dat khach nhat (by_day), gio cao diem (by_hour)."""
import json, os, re, sys, collections, datetime as dt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datamate.kernel import load_business_cached
from datamate.metrics import Metrics
from datamate.dates import period_vars
from datamate.parsing import norm_key_ascii as plain
from datamate.answers import same_answer
DATA = "data/train"
Q = [json.loads(l) for l in open(f"{DATA}/questions.jsonl")]; G = {g["id"]: g for g in map(json.loads, open(f"{DATA}/answers.jsonl"))}
res = collections.defaultdict(list); cache = {}
for q in Q:
    s = plain(q["question"]); g = G[q["id"]]
    if g["type"] == "unanswerable": continue
    pv = period_vars(q["as_of_date"])
    m = re.search(r"tu (\d{1,2})/(\d{1,2})/(\d{4}) den (?:het )?(\d{1,2})/(\d{1,2})/(\d{4})", s)
    mm = re.search(r"thang (\d{1,2})[/ ](?:nam )?(\d{4})", s)
    if m:
        per = (dt.date(int(m[3]), int(m[2]), int(m[1])), dt.date(int(m[6]), int(m[5]), int(m[4])))
    elif mm:
        per = pv["THANG"](int(mm[1]), int(mm[2]))
    elif "thang nay" in s:
        per = pv["THANG_NAY"]
    elif "tuan truoc" in s or "tuan roi" in s:
        per = pv["TUAN_TRUOC"]
    else:
        continue
    b = q["business_id"]
    if b not in cache:
        x = load_business_cached(os.path.join(DATA, "data", b), "/tmp/datamate_cache_dev"); cache[b] = Metrics(x.sales.S, x.sales.O) if x.sales is not None else None
    M = cache[b]
    if M is None: continue
    kind = pred = None
    if re.search(r"ban cham|khong ban duoc|mat hang e\b|hang e\b|bi e\b", s) and g["type"] in ("number", "list"):
        u = M.unsold_items(per); kind = "unsold"; pred = len(u) if g["type"] == "number" else u
    elif g["type"] == "date" and re.search(r"ngay nao", s) and re.search(r"tien|doanh thu|dthu", s):
        d = M.by_day(per); kind = "best_day_rev"; r = d.sort_values(["doanh_thu", "order_date"], ascending=[False, True]).iloc[0]; pred = r.order_date.strftime("%Y-%m-%d")
    elif g["type"] == "date" and re.search(r"dat khach|nhieu don", s):
        d = M.by_day(per); kind = "best_day_orders"; r = d.sort_values(["so_don", "order_date"], ascending=[False, True]).iloc[0]; pred = r.order_date.strftime("%Y-%m-%d")
    elif re.search(r"gio cao diem|khung gio", s) and g["type"] == "number":
        h = M.by_hour(per); kind = "peak_hour"; pred = int(h.iloc[0].hour) if len(h) else None
    if kind is None: continue
    res[kind].append((b, pred is not None and same_answer({"type": g["type"], "answer": pred}, g), json.dumps(pred, ensure_ascii=False)[:80], json.dumps(g["answer"], ensure_ascii=False)[:80], q["question"][:70]))
for kind, rows in res.items():
    print(f"{kind}: {sum(r[1] for r in rows)}/{len(rows)} khop")
    for r in rows:
        if not r[1]: print("   SAI", r)
