"""Doi chieu nhap hang (imports.py) voi dap an train: tong tien phieu nhap trong thang / thang nay, gia nhap binh quan."""
import json, os, re, sys, collections, calendar, datetime as dt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datamate.kernel import load_business_cached
from datamate.imports import build_imports
from datamate.parsing import norm_key_ascii as plain, compact_key
from datamate.answers import same_answer, round_money

DATA = "data/train"
Q = [json.loads(l) for l in open(f"{DATA}/questions.jsonl")]
G = {g["id"]: g for g in map(json.loads, open(f"{DATA}/answers.jsonl"))}
models = {}; res = collections.defaultdict(list)
for q in Q:
    s = plain(q["question"]); g = G[q["id"]]
    if "nhap" not in s:
        continue
    b = q["business_id"]
    if b not in models:
        models[b] = build_imports(load_business_cached(os.path.join(DATA, "data", b), "/tmp/datamate_cache_dev"))
    M = models[b]
    D = dt.date.fromisoformat(q["as_of_date"])
    mm = re.search(r"thang (\d{1,2})[/ ](?:nam )?(\d{4})", s)
    if mm:
        y, mo = int(mm.group(2)), int(mm.group(1)); a, z = dt.date(y, mo, 1), dt.date(y, mo, calendar.monthrange(y, mo)[1])
    elif "thang nay" in s:
        a, z = D.replace(day=1), D
    else:
        continue
    if M is None:
        res["no_model"].append((b, g["type"] == "unanswerable", None, g["answer"], q["question"][:80])); continue
    N = M.N; sel = ~N.cancelled & (N.date >= str(a)) & (N.date <= str(z))
    kind = pred = None
    if re.search(r"nha cung cap nao|ncc nao", s) and g["type"] == "string":
        from datamate.imports import ImportMetrics
        by = "receipts" if re.search(r"nhieu phieu|nhieu lan|chinh", s) else "amount"
        t = ImportMetrics(M).top_suppliers(a, z, by=by)
        kind, pred = "top_supplier", {"type": "string", "answer": None if t is None or t.empty else t.iloc[0]["supplier"]}
    elif re.search(r"tong gia tri|tong tien|bao nhieu tien nhap|tien nhap hang", s) and "binh quan" not in s:
        kind, pred = "total", {"type": "number", "answer": round_money(N.loc[sel, "line_total"].sum())}
    elif "binh quan" in s and "item" in N:
        best = None
        for it in N["item"].dropna().unique():
            k = plain(it)
            if k in s and (best is None or len(k) > len(plain(best))): best = it
        um = re.search(r"(?:mot|1) (\w+)", s.split("binh quan", 1)[1]) or re.search(r"(?:mot|1) (\w+)", s)
        if best is None or not um: continue
        unit = um.group(1)
        from datamate.imports import ImportMetrics
        IM = ImportMetrics(M)
        units = [u for u in N.get("unit", []).dropna().unique() if plain(u) == unit] if "unit" in N else []
        v = IM.avg_import_price(a, z, best, units[0] if units else None)
        kind, pred = "avg_price", {"type": "number", "answer": v}
    if kind is None or g["type"] == "unanswerable" or pred["answer"] is None:
        if kind and g["type"] == "unanswerable": res[kind + "_unans"].append((b, pred["answer"] is None, pred["answer"], None, q["question"][:80]))  # helper tra None -> model tra unanswerable
        continue
    res[kind].append((b, same_answer(pred, g), pred["answer"], g["answer"], q["question"][:80]))
for kind, rows in res.items():
    print(f"{kind}: {sum(r[1] for r in rows)}/{len(rows)} khop")
    for r in rows:
        if not r[1]: print("   SAI", r)
