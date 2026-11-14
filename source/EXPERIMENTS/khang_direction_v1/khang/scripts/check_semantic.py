"""Doi chieu bang chuan S/O voi dap an train cho 3 loai cau don gian (khong dung model):
doanh thu thang M/Y, so don thang M/Y, ti le huy thang M/Y."""
import json, os, re, sys, unicodedata, collections
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
from datamate.kernel import load_business_cached
from datamate.semantic import build_sales
from datamate.helpers import pct, round_money

split = sys.argv[1] if len(sys.argv) > 1 else "train"
DATA = f"data/{split}"
Q = [json.loads(l) for l in open(f"{DATA}/questions.jsonl")]
G = {g["id"]: g for g in map(json.loads, open(f"{DATA}/answers.jsonl"))} if os.path.exists(f"{DATA}/answers.jsonl") else {}


def plain(s):
    from datamate.hints import _norm  # bo dau + sua viet tat / loi go telex ("binh quaan mooix don")
    return _norm(s)


EXCL = ("nhom", "mat hang", "khach", "ca ", "kenh", "nhan vien", "ngay", "tuan", "trung binh", "so voi", "chenh", "tang", "giam", "moi ", "thuoc", "hoat chat", "tien mat", "chuyen khoan", "no ", "tu ", "tro len", "duoi", "tren", "gio", "ban duoc bao nhieu", "loai", "size", "sua", "bim", "topping", "cua ", "qua ", "ke don", "binh quan", "tai quay", ">", "≥", "<", "zalo", "shopee", "online", "tai cho", "mang ve", "giao di", "phong", "cod", "tra bang", "thanh toan", "vi dien tu")
models = {}
res = collections.defaultdict(list)
for q in Q:
    if q["id"] not in G:
        continue
    s = plain(q["question"])
    m = re.search(r"thang (\d{1,2})[/ ](?:nam )?(\d{4})", s)
    if not m or any(k in s for k in EXCL):
        continue
    kind = None
    if re.search(r"(ti le|ty le).{0,20}huy|huy.{0,20}(ti le|ty le|bao nhieu %|phan tram)", s):
        kind = "cancel_rate"
    elif re.search(r"(doanh thu|dthu)", s) and G[q["id"]]["type"] == "number":
        kind = "revenue"
    elif re.search(r"(bao nhieu|may|bnhiu|bao nhiu) (don|hoa don)", s) and "huy" not in s and "tra" not in s and G[q["id"]]["type"] == "number":
        kind = "orders"
    if not kind:
        continue
    b = q["business_id"]
    if b not in models:
        biz = load_business_cached(os.path.join(DATA, "data", b), "/tmp/datamate_cache_dev")
        models[b] = build_sales(biz)
    M = models[b]
    gold = G[q["id"]]["answer"]
    if M is None:
        res[kind].append((b, False, None, gold, q["question"][:70], "khong dung duoc S"))
        continue
    mm, yy = int(m.group(1)), int(m.group(2))
    S, O = M.S, M.O
    in_m = lambda d: (d.dt.year == yy) & (d.dt.month == mm)
    if kind == "revenue":
        got = round_money(S.loc[S.revenue_ok & in_m(S.order_date), "amount"].sum()) if len(S) else round_money(O.loc[O.revenue_ok & in_m(O.order_date), "revenue"].sum())
    elif kind == "orders":
        got = int((O.revenue_ok & in_m(O.order_date)).sum()) if len(O) else None
    else:
        sub = O[in_m(O.order_date)] if len(O) else None
        got = pct((sub.status == "huỷ").sum(), len(sub)) if sub is not None and len(sub) else None
    if G[q["id"]]["type"] == "unanswerable":  # ham tra None (thang khong co don) -> model tra unanswerable
        res[kind + "_unans"].append((b, got is None, got, None, q["question"][:70], "")); continue
    ok = got is not None and gold is not None and abs(got - gold) < 1e-6
    res[kind].append((b, ok, got, gold, q["question"][:70], ""))
for kind, rows in res.items():
    print(f"{kind}: {sum(r[1] for r in rows)}/{len(rows)} khop")
    for r in rows:
        if not r[1]:
            print("   SAI", r)
