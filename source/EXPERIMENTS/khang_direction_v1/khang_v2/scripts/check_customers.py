"""Doi chieu phan tich khach (metrics.customers) voi dap an train."""
import json, os, re, sys, collections, datetime as dt, calendar
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datamate.kernel import load_business_cached
from datamate.metrics import Metrics
from datamate.dates import week_range
from datamate.helpers import pct
from datamate.parsing import norm_key_ascii as plain
from datamate.answers import same_answer
DATA = "data/train"
Q = [json.loads(l) for l in open(f"{DATA}/questions.jsonl")]; G = {g["id"]: g for g in map(json.loads, open(f"{DATA}/answers.jsonl"))}
res = collections.defaultdict(list); cache = {}
for q in Q:
    s = plain(q["question"]); g = G[q["id"]]
    if "khach" not in s or re.search(r"\bno\b|nhom|tien mat|chuyen khoan|ghi no|mua chiu|thanh vien|dang ky|duong|dia chi", s): continue
    D = dt.date.fromisoformat(q["as_of_date"])
    m = re.search(r"thang (\d{1,2})[/ ](?:nam )?(\d{4})", s)
    if m:
        y, mo = int(m.group(2)), int(m.group(1)); a, z = dt.date(y, mo, 1), dt.date(y, mo, calendar.monthrange(y, mo)[1])
    elif "tuan truoc" in s or "tuan roi" in s:
        a, _ = week_range(D); a -= dt.timedelta(days=7); z = a + dt.timedelta(days=6)
    elif "thang nay" in s:
        a, z = D.replace(day=1), D
    else:
        continue
    b = q["business_id"]
    if b not in cache:
        x = load_business_cached(os.path.join(DATA, "data", b), "/tmp/datamate_cache_dev"); cache[b] = Metrics(x.sales.S, x.sales.O) if x.sales is not None else None
    M = cache[b]
    if M is None: continue
    C = M.customers(a, z)
    if C.empty:
        res["empty"].append((b, g["type"] == "unanswerable", None, g["answer"], q["question"][:70])); continue
    key = C.columns[0]
    kind = pred = None
    if g["type"] == "string" and re.search(r"nhieu tien nhat|chi nhieu|mua nhieu nhat.*tien|tien nhieu nhat", s):
        kind = "top_spender"; t = C.sort_values(["doanh_thu", key], ascending=[False, True]); pred = t.iloc[0].khach
    elif g["type"] == "string" and "trung thanh" in s:
        kind = "loyal"; t = C.sort_values(["so_ngay_mua", key], ascending=[False, True]); pred = t.iloc[0].khach
    elif g["type"] == "string" and "sop" in s:
        kind = "sop"; t = C[C.so_don >= 3].sort_values(["gia_tri_tb_don", key], ascending=[False, True]); pred = t.iloc[0].khach if len(t) else None
    elif g["type"] == "string" and "doi y" in s:
        kind = "doi_y"; t = C.sort_values(["so_don_huy", key], ascending=[False, True]); pred = t.iloc[0].khach
    elif g["type"] == "string" and re.search(r"(so don|nhieu don).*nhieu nhat|mua nhieu don nhat|nhieu don nhat", s):
        kind = "most_orders"; t = C.sort_values(["so_don", key], ascending=[False, True]); pred = t.iloc[0].khach
    elif g["type"] == "percent" and (mm := re.search(r"tu (\d+) don tro len", s)):
        kind = "pct_k_orders"; pred = pct((C.so_don >= int(mm.group(1))).sum(), len(C))
    if kind is None: continue
    gp = {"type": g["type"], "answer": pred}
    res[kind].append((b, pred is not None and same_answer(gp, g), pred, g["answer"], q["question"][:70]))
for kind, rows in res.items():
    print(f"{kind}: {sum(r[1] for r in rows)}/{len(rows)} khop")
    for r in rows:
        if not r[1]: print("   SAI", r)
