"""Doi chieu 'hieu ung tang gia' (§5): so luong ban N ngay ke tu ngay D (tinh ca D) so voi N ngay ngay truoc D."""
import json, os, re, sys, collections, datetime as dt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datamate.kernel import load_business_cached
from datamate.metrics import Metrics
from datamate.helpers import pct_change
from datamate.parsing import norm_key_ascii as plain
from datamate.answers import same_answer
DATA = "data/train"
Q = [json.loads(l) for l in open(f"{DATA}/questions.jsonl")]; G = {g["id"]: g for g in map(json.loads, open(f"{DATA}/answers.jsonl"))}
res = []; cache = {}
for q in Q:
    s = plain(q["question"]); g = G[q["id"]]
    if "tang gia" not in s and "len gia" not in s: continue
    m = re.search(r"(\d+) ngay ke tu ngay (\d{1,2})/(\d{1,2})/(\d{4})", s) or re.search(r"(\d+) ngay ke tu (\d{1,2})/(\d{1,2})/(\d{4})", s)
    if not m: continue
    n = int(m.group(1)); D = dt.date(int(m.group(4)), int(m.group(3)), int(m.group(2)))
    b = q["business_id"]
    if b not in cache:
        x = load_business_cached(os.path.join(DATA, "data", b), "/tmp/datamate_cache_dev"); cache[b] = Metrics(x.sales.S, x.sales.O)
    M = cache[b]
    items = sorted(M.S["item"].dropna().unique(), key=lambda v: -len(v))
    it = next((v for v in items if plain(v) in s), None)
    if it is None: continue
    after = M.qty_sold(D, D + dt.timedelta(days=n - 1), item=it)
    before = M.qty_sold(D - dt.timedelta(days=n), D - dt.timedelta(days=1), item=it)
    pred = {"type": "percent", "answer": pct_change(after, before)} if g["type"] == "percent" else {"type": g["type"], "answer": None}
    if g["type"] != "percent": continue
    res.append((b, same_answer(pred, g), pred["answer"], g["answer"], q["question"][:80]))
print(f"price_effect: {sum(r[1] for r in res)}/{len(res)} khop")
for r in res:
    if not r[1]: print("   SAI", r)

# ---- gia ban ngay D
res = []
for q in Q:
    s = plain(q["question"]); g = G[q["id"]]
    if g["type"] != "number" or not re.search(r"gia ban|don gia", s): continue
    m = re.search(r"ngay (\d{1,2})/(\d{1,2})/(\d{4})", s)
    if not m: continue
    D = dt.date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    b = q["business_id"]
    if b not in cache:
        x = load_business_cached(os.path.join(DATA, "data", b), "/tmp/datamate_cache_dev"); cache[b] = Metrics(x.sales.S, x.sales.O)
    M = cache[b]
    items = sorted(M.S["item"].dropna().unique(), key=lambda v: -len(v))
    it = next((v for v in items if plain(v) in s), None)
    if it is None: continue
    v = M.price_on(D, it)
    res.append((b, v is not None and same_answer({"type": "number", "answer": v}, g), v, g["answer"], q["question"][:80]))
print(f"price_on: {sum(r[1] for r in res)}/{len(res)} khop")
for r in res:
    if not r[1]: print("   SAI", r)
