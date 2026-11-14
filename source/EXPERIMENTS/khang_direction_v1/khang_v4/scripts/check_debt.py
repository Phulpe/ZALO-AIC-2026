"""Doi chieu ham cong no (debt.py) voi dap an train."""
import json, os, re, sys, collections, calendar, datetime as dt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datamate.kernel import load_business_cached
from datamate.debt import build_debt, debt_as_of, debt_by_customer, debt_paid, debt_added
from datamate.parsing import norm_key_ascii as plain
from datamate.answers import same_answer, round_money

split = sys.argv[1] if len(sys.argv) > 1 else "train"
DATA = f"data/{split}"
Q = [json.loads(l) for l in open(f"{DATA}/questions.jsonl")]
G = {g["id"]: g for g in map(json.loads, open(f"{DATA}/answers.jsonl"))}
models = {}
res = collections.defaultdict(list)
for q in Q:
    s = plain(q["question"]); g = G[q["id"]]
    if not re.search(r"\bno\b|cong no|mua chiu|thu no|tra no|ghi no", s):
        continue
    b = q["business_id"]
    if b not in models:
        biz = load_business_cached(os.path.join(DATA, "data", b), "/tmp/datamate_cache_dev")
        models[b] = build_debt(biz, biz.sales)
    M = models[b]
    m = re.search(r"het ngay (\d{1,2})/(\d{1,2})/(\d{4})", s)
    mm = re.search(r"thang (\d{1,2})[/ ](?:nam )?(\d{4})", s)
    kind, pred = None, None
    if M is None:
        res["no_model"].append((b, g["type"] == "unanswerable", None, g["answer"], q["question"][:80])); continue
    if m:
        D = dt.date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        if g["type"] == "table":
            k = int(re.search(r"(\d+) khach", s).group(1)) if re.search(r"(\d+) khach", s) else 3
            t = debt_by_customer(M, D).head(k)
            kind, pred = "top_table", {"type": "table", "answer": {"columns": ["khach_hang", "so_tien_no"], "rows": [[r.khach, round_money(r.con_no)] for r in t.itertuples()]}}
        elif re.search(r"bao nhieu khach|may khach|so khach|bnhiu khach|bao nhiu khach", s) and g["type"] == "number":
            kind, pred = "n_debtors", {"type": "number", "answer": len(debt_by_customer(M, D))}
        elif g["type"] == "number" and (ag := re.search(r"(?:hon|qua|tren) (\d+) ngay", s)):  # tuoi no: 'qua N' khong gom N (§2)
            d = debt_as_of(M, D)
            kind, pred = "aging", {"type": "number", "answer": round_money(d.loc[d.tuoi_no > int(ag.group(1)), "con_no"].sum())}
        elif g["type"] == "number":
            kind, pred = "total", {"type": "number", "answer": round_money(debt_as_of(M, D)["con_no"].sum())}
    elif mm and g["type"] == "number" and re.search(r"tra no|thu hoi|thu no|tra bot", s) and "khach " not in s.replace("khach tra", ""):
        y, mo = int(mm.group(2)), int(mm.group(1))
        a, z = dt.date(y, mo, 1), dt.date(y, mo, calendar.monthrange(y, mo)[1])
        kind, pred = "paid_month", {"type": "number", "answer": round_money(debt_paid(M, a, z)["amount"].sum())}
    if kind is None or g["type"] == "unanswerable":
        continue
    ok = same_answer(pred, g)
    res[kind].append((b, ok, json.dumps(pred["answer"], ensure_ascii=False)[:90], json.dumps(g["answer"], ensure_ascii=False)[:90], q["question"][:70]))
for kind, rows in res.items():
    print(f"{kind}: {sum(r[1] for r in rows)}/{len(rows)} khop")
    for r in rows:
        if not r[1]:
            print("   SAI", r)
