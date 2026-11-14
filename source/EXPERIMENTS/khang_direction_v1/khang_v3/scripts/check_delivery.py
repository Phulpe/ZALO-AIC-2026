"""Doi chieu so dot giao hang (rulebook 'Giao nhieu dot') voi dap an train: delivery_rounds(ky) tren bang D_giao."""
import json, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datamate.kernel import load_business_cached
from datamate.delivery import build_delivery, delivery_rounds
from datamate.hints import question_periods, _norm
from datamate.answers import same_answer

DATA = "data/train"
Q = [json.loads(l) for l in open(f"{DATA}/questions.jsonl")]
G = {g["id"]: g for g in map(json.loads, open(f"{DATA}/answers.jsonl"))}
res, cache = [], {}
for q in Q:
    s = _norm(q["question"]); g = G[q["id"]]
    if not re.search(r"\bdot giao\b|\bgiao (hang )?(tong cong )?(bao nhieu|may|duoc bao nhieu) dot\b|\bmay dot\b", s) or g["type"] != "number":
        continue
    ps = [p for p in question_periods(q["question"], q["as_of_date"]) if p.code.startswith(("THANG", "QUY", "TUAN", "N_NGAY"))]
    if len(ps) != 1 or re.search(r"\bdat\b|\btu \d+ dot\b", s):
        continue
    b = q["business_id"]
    if b not in cache:
        B = load_business_cached(os.path.join(DATA, "data", b), "/tmp/datamate_cache_dev")
        cache[b] = build_delivery(B.tables, B.sales.O if B.sales else None)
    D = cache[b]
    pred = {"type": "number", "answer": None if D is None else delivery_rounds(D, ps[0].a, ps[0].b, q["as_of_date"])}
    res.append((b, pred["answer"] is not None and same_answer(pred, g), pred["answer"], g["answer"], q["question"][:80]))
print(f"delivery_rounds: {sum(r[1] for r in res)}/{len(res)} khop")
for r in res:
    if not r[1]:
        print("   SAI", r)
