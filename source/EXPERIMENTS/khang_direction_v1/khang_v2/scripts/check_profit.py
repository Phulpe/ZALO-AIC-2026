"""Doi chieu lai gop (S.cogs) voi dap an train."""
import json, os, re, sys, collections
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datamate.kernel import load_business_cached
from datamate.parsing import norm_key_ascii as plain
from datamate.answers import same_answer, round_money
DATA = "data/train"
Q = [json.loads(l) for l in open(f"{DATA}/questions.jsonl")]; G = {g["id"]: g for g in map(json.loads, open(f"{DATA}/answers.jsonl"))}
res = collections.defaultdict(list); cache = {}
for q in Q:
    s = plain(q["question"]); g = G[q["id"]]
    if not re.search(r"lai gop|loi gop|loi nhuan gop|loi nhat|lai nhat", s): continue
    mm = re.search(r"thang (\d{1,2})[/ ](?:nam )?(\d{4})", s)
    if not mm: continue
    b = q["business_id"]
    if b not in cache: cache[b] = load_business_cached(os.path.join(DATA, "data", b), "/tmp/datamate_cache_dev").sales
    M = cache[b]
    if M is None or "cogs" not in M.S:
        res["no_cost"].append((b, g["type"] == "unanswerable", None, g["answer"], q["question"][:70])); continue
    S = M.S; y, mo = int(mm.group(2)), int(mm.group(1))
    sel = S.revenue_ok & S.revenue_line & (S.order_date.dt.year == y) & (S.order_date.dt.month == mo)
    if g["type"] == "number":
        pred = {"type": "number", "answer": round_money(S.loc[sel, "amount"].sum() - S.loc[sel, "cogs"].sum())}; kind = "gross_month"
    elif g["type"] == "string":
        t = S[sel & S.is_item].assign(gp=lambda d: d.amount - d.cogs).groupby("item")["gp"].sum().sort_values(ascending=False)
        pred = {"type": "string", "answer": t.index[0]}; kind = "top_profit_item"
    else:
        continue
    res[kind].append((b, same_answer(pred, g), pred["answer"], g["answer"], q["question"][:70]))
for kind, rows in res.items():
    print(f"{kind}: {sum(r[1] for r in rows)}/{len(rows)} khop")
    for r in rows:
        if not r[1]: print("   SAI", r)
