"""Doi chieu doanh thu theo hoat chat / thuoc ke don (S.dm_*) voi dap an train."""
import json, os, re, sys, collections
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datamate.kernel import load_business_cached
from datamate.parsing import norm_key_ascii as plain
from datamate.answers import same_answer, round_money
from datamate.helpers import pct
DATA = "data/train"
Q = [json.loads(l) for l in open(f"{DATA}/questions.jsonl")]; G = {g["id"]: g for g in map(json.loads, open(f"{DATA}/answers.jsonl"))}
res = collections.defaultdict(list); cache = {}
for q in Q:
    s = plain(q["question"]); g = G[q["id"]]
    mm = re.search(r"thang (\d{1,2})[/ ](?:nam )?(\d{4})", s)
    if not mm or not re.search(r"hoat chat|ke don", s): continue
    b = q["business_id"]
    if b not in cache: cache[b] = load_business_cached(os.path.join(DATA, "data", b), "/tmp/datamate_cache_dev").sales
    S = cache[b].S
    ing = next((c for c in S.columns if c.startswith("dm_") and any(k in plain(c).replace("_", " ") for k in ("hoat chat", "thanh phan"))), None)
    rx = next((c for c in S.columns if c.startswith("dm_") and any(k in plain(c).replace("_", " ") for k in ("ke don", "theo don"))), None)
    y, mo = int(mm.group(2)), int(mm.group(1))
    base = S.revenue_ok & S.revenue_line & (S.order_date.dt.year == y) & (S.order_date.dt.month == mo)
    if "hoat chat" in s and ing:
        names = sorted(S[ing].dropna().unique(), key=len, reverse=True)
        hit = next((n for n in names if plain(n) in s), None)  # ten dai nhat co trong cau hoi (hoat chat phoi hop truoc)
        if hit is None: continue
        pred = {"type": "number", "answer": round_money(S.loc[base & (S[ing] == hit), "amount"].sum())}; kind = "ingredient"
    elif "ke don" in s and rx:
        flag = S[rx].astype(float) == 1
        part = ~flag if re.search(r"khong ke don|k ke don|hok ke don|ko ke don|hong ke don|tru thuoc ke don|khong can ke don|khong phai thuoc ke don", s) else flag
        if g["type"] == "percent":
            pred = {"type": "percent", "answer": pct(S.loc[base & part, "amount"].sum(), S.loc[base, "amount"].sum())}; kind = "rx_share"
        else:
            pred = {"type": "number", "answer": round_money(S.loc[base & part, "amount"].sum())}; kind = "rx_revenue"
    else:
        continue
    if g["type"] == "unanswerable": res[kind + "_unans"].append((b, False, pred["answer"], None, q["question"][:70])); continue
    res[kind].append((b, same_answer(pred, g), pred["answer"], g["answer"], q["question"][:70]))
for kind, rows in res.items():
    print(f"{kind}: {sum(r[1] for r in rows)}/{len(rows)} khop")
    for r in rows:
        if not r[1]: print("   SAI", r)
