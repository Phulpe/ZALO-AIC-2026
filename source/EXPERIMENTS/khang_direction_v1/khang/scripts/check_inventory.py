"""Doi chieu ton kho (inventory.py) voi dap an train: ton cuoi ngay cua mot mat hang, gia tri ton theo gia von."""
import json, os, re, sys, collections, datetime as dt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datamate.kernel import load_business_cached
from datamate.inventory import build_inventory, stock_at, stock_value_at
from datamate.parsing import norm_key_ascii as plain, compact_key
from datamate.answers import same_answer, round_money
DATA = "data/train"
Q = [json.loads(l) for l in open(f"{DATA}/questions.jsonl")]; G = {g["id"]: g for g in map(json.loads, open(f"{DATA}/answers.jsonl"))}
res = collections.defaultdict(list); cache = {}
for q in Q:
    s = plain(q["question"]); g = G[q["id"]]
    if not re.search(r"\bton\b|trong kho", s) or re.search(r"\blo\b|het han|doanh thu|ban ra|ban duoc", s): continue
    b = q["business_id"]
    if b not in cache:
        biz = load_business_cached(os.path.join(DATA, "data", b), "/tmp/datamate_cache_dev"); cache[b] = build_inventory(biz, biz.sales)
    M = cache[b]
    m = re.search(r"het ngay (\d{1,2})/(\d{1,2})/(\d{4})|cuoi ngay (\d{1,2})/(\d{1,2})/(\d{4})|ngay (\d{1,2})/(\d{1,2})/(\d{4})", s)
    D = dt.date.fromisoformat(q["as_of_date"])
    if m:
        gr = [x for x in m.groups() if x]; D = dt.date(int(gr[2]), int(gr[1]), int(gr[0]))
    if M is None:
        res["no_model"].append((b, g["type"] == "unanswerable", None, g["answer"], q["question"][:70])); continue
    if "gia von" in s:
        v = stock_value_at(M, D)
        res["value"].append((b, v is not None and same_answer({"type": "number", "answer": round_money(v)}, g) if g["type"] == "number" else v is None, v, g["answer"], q["question"][:70])); continue
    st = stock_at(M, D)
    if st is None:
        res["no_stock_for_day"].append((b, g["type"] == "unanswerable", None, g["answer"], q["question"][:70])); continue
    best = None
    for _, r in st.iterrows():
        nm = r.get("item")
        if isinstance(nm, str) and plain(nm) in s and (best is None or len(plain(nm)) > len(plain(best["item"]))): best = r
    if best is None: continue
    if g["type"] == "number":
        res["item_stock"].append((b, same_answer({"type": "number", "answer": float(best["ton"])}, g), best["ton"], g["answer"], q["question"][:70]))
    elif g["type"] == "list":
        biz = load_business_cached(os.path.join(DATA, "data", b), "/tmp/datamate_cache_dev")
        from datamate.semantic import _conversion_table
        conv = _conversion_table(biz)
        from datamate.parsing import norm_key
        um = re.search(r"may (\w+) chan|bao nhieu (\w+) chan|bao nhieu (\w+) nguyen|may (\w+) nguyen|(\w+) chan|(?:bao nhieu|bnhiu|bao nhiu|may) (\w+) .*?(?:chan|nguyen)", s)
        u = next((x for x in (um.groups() if um else []) if x), None)
        f = None
        for (k, unit), fac in conv.items():
            if (plain(unit) == u or plain(unit).startswith(f"{u} ")) and (k == norm_key(best.get("item") or "") or k == norm_key(best.get("item_code") or "")):
                f = fac
        pred = [int(best["ton"] // f), int(best["ton"] % f)] if f else None
        res["item_stock_big_unit"].append((b, pred == g["answer"], pred, g["answer"], q["question"][:70]))
for kind, rows in res.items():
    print(f"{kind}: {sum(1 for r in rows if r[1])}/{len(rows)} khop")
    for r in rows:
        if r[1] is False: print("   SAI", r)
