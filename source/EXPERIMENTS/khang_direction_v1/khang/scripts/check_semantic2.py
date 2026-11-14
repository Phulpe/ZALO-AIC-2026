"""Doi chieu S/O voi dap an train cho: so luong ban cua mot mat hang trong thang, top-k ban chay trong thang,
so don huy trong thang, gia tri trung binh mot don trong thang (khong dung model)."""
import json, os, re, sys, unicodedata, collections
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
from datamate.kernel import load_business_cached
from datamate.parsing import norm_key_ascii
from datamate.answers import same_answer, round_money

DATA = "data/train"
Q = [json.loads(l) for l in open(f"{DATA}/questions.jsonl")]
G = {g["id"]: g for g in map(json.loads, open(f"{DATA}/answers.jsonl"))}
plain = norm_key_ascii
cache = {}
res = collections.defaultdict(list)


def find_item(s, items):
    best = None
    for it in items:
        k = plain(it)
        if k and k in s and (best is None or len(k) > len(plain(best))):
            best = it
    return best


for q in Q:
    s = plain(q["question"])
    m = re.search(r"thang (\d{1,2})[/ ](?:nam )?(\d{4})", s)
    if not m:
        continue
    mm, yy = int(m.group(1)), int(m.group(2))
    g = G[q["id"]]
    b = q["business_id"]
    if b not in cache:
        cache[b] = load_business_cached(os.path.join(DATA, "data", b), "/tmp/datamate_cache_dev").sales
    M = cache[b]
    if M is None or not len(M.S) or "item" not in M.S:
        continue
    S, O = M.S, M.O
    inm = (S.order_date.dt.year == yy) & (S.order_date.dt.month == mm)
    kind, pred = None, None
    if re.search(r"(ban chay|ban nhieu nhat|hot nhat|top)", s) and g["type"] in ("list", "string") and "khach" not in s and "nhom" not in s:
        k = re.search(r"(\d+) (mat hang|mon|san pham|sp|hang)", s)
        k = int(k.group(1)) if k else 1
        sub = S[inm & S.revenue_ok & S.is_item]
        agg = sub.groupby("item")["qty_base"].sum().reset_index()
        agg["_n"] = agg["item"].map(plain)
        agg = agg.sort_values(["qty_base", "item"], ascending=[False, True])
        top = agg["item"].head(k).tolist()
        kind = "top"
        pred = {"type": g["type"], "answer": top if g["type"] == "list" else top[0]}
    elif re.search(r"(ban duoc|ban ra|ban dc).{0,30}(may|bao nhieu|bnhiu|bao nhiu|bn)", s) and "tien" not in s and g["type"] == "number":
        it = find_item(s, S["item"].dropna().unique())
        if not it:
            continue
        kind = "qty_item"
        pred = {"type": "number", "answer": float(S.loc[inm & S.revenue_ok & (S.item == it), "qty_base"].sum())}
    elif re.search(r"(so don|may don|bao nhieu don|bnhiu don).{0,25}huy|huy.{0,15}(so don|may don|bao nhieu don)|dem so don da huy", s) and g["type"] == "number":
        kind = "n_cancel"
        sub = O[(O.order_date.dt.year == yy) & (O.order_date.dt.month == mm)]
        pred = {"type": "number", "answer": int((sub.status == "huỷ").sum())}
    elif re.search(r"trung binh (mot|moi|tren mot) don|gia tri trung binh", s) and g["type"] == "number":
        kind = "aov"
        sub = O[(O.order_date.dt.year == yy) & (O.order_date.dt.month == mm) & O.revenue_ok]
        grp = re.search(r"khach (?:nhom|thuoc nhom|hang) ([a-z ]+?)(?:(?: la| thi| co)\b|$)", s)
        if grp and "customer_group" in sub:  # AOV cua mot nhom khach (§1: don vang lai khong thuoc nhom nao)
            gk = grp.group(1).strip()
            sub = sub[sub.customer_group.fillna("").map(plain) == gk]
            kind = "aov_group"
        pred = {"type": "number", "answer": round_money(sub.revenue.sum() / len(sub)) if len(sub) else None}
    if kind is None or g["type"] == "unanswerable":
        continue
    ok = pred["answer"] is not None and same_answer(pred, g)
    res[kind].append((b, ok, json.dumps(pred["answer"], ensure_ascii=False)[:70], json.dumps(g["answer"], ensure_ascii=False)[:70], q["question"][:80]))
for kind, rows in res.items():
    print(f"{kind}: {sum(r[1] for r in rows)}/{len(rows)} khop")
    for r in rows:
        if not r[1]:
            print("   SAI", r)

# ---- ca sang / ca chieu (doanh thu theo ca trong tuan truoc / thang)
from datamate.dates import week_range
import datetime as _dt
res2 = collections.defaultdict(list)
for q in Q:
    s = plain(q["question"]); g = G[q["id"]]
    if not re.search(r"\bca (sang|chieu)\b", s) or g["type"] != "number" or "%" in s or "hon" in s or "kem" in s:
        continue
    b = q["business_id"]
    M = cache.get(b) or load_business_cached(os.path.join(DATA, "data", b), "/tmp/datamate_cache_dev").sales
    cache[b] = M
    if M is None or "ca" not in M.S:
        res2["shift"].append((b, False, "no shift", g["answer"], q["question"][:80])); continue
    S = M.S; shift = "sáng" if "ca sang" in s else "chiều"
    D = _dt.date.fromisoformat(q["as_of_date"])
    if "tuan truoc" in s or "tuan roi" in s:
        a, _ = week_range(D); a = a - _dt.timedelta(days=7); bnd = a + _dt.timedelta(days=6)
    else:
        m = re.search(r"thang (\d{1,2})[/ ](?:nam )?(\d{4})", s)
        if not m: continue
        import calendar
        a = _dt.date(int(m.group(2)), int(m.group(1)), 1); bnd = a.replace(day=calendar.monthrange(a.year, a.month)[1])
    sel = S.revenue_ok & S.revenue_line & (S.order_date >= pd.Timestamp(a)) & (S.order_date <= pd.Timestamp(bnd)) & (S["ca"] == shift)
    if re.search(r"\b(don|bill)\b", s) and not re.search(r"doanh thu|tien|dthu", s):  # dem don theo ca (§1: don khong huy, khong tra)
        got = int(S.loc[sel, "order_id"].nunique()); kind = "shift_orders"
    else:
        got = round_money(S.loc[sel, "amount"].sum()); kind = "shift"
    res2[kind].append((b, got == g["answer"], got, g["answer"], q["question"][:80]))
for kind, rows in res2.items():
    print(f"{kind}: {sum(r[1] for r in rows)}/{len(rows)} khop")
    for r in rows:
        if not r[1]:
            print("   SAI", r)
