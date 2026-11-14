"""Ngan hang vi du few-shot tu TRAIN learning (khang_v3 fewshot.py).

Moi cau TRAIN learning co code goi y ma Tier1 da xac nhan dung gold TRAIN (TIER1_V2_PER_ID our_score==1)
-> {id, business_id, question, code, fn}. Khong calibration/audit/public. stdout chi in so tong hop.

Dung: python build_fewshot_bank.py <khang_src> <data_root_train> <tier1_per_id_jsonl> <out_jsonl>
"""
import collections
import hashlib
import json
import os
import re
import sys

CODE_LINE = re.compile(r"^Code gợi ý \([^)]*\): (.*)$", re.M)


def main(src, data_root, per_id, out):
    sys.path.insert(0, src)
    from datamate.dates import LunarCalendar
    from datamate.hints import EntityIndex, question_hints
    from datamate.kernel import load_business_cached

    ok = set()
    for l in open(per_id, encoding="utf-8"):
        r = json.loads(l)
        if r.get("split") == "train" and r.get("status") == "hint_answer" and r.get("our_score") == 1:
            ok.add(r["id"])
    qs = [q for q in map(json.loads, open(os.path.join(data_root, "questions.jsonl"), encoding="utf-8")) if q["id"] in ok]
    lun = LunarCalendar.load(os.path.join(data_root, "calendar.csv"))
    by = collections.defaultdict(list)
    for q in qs:
        by[q["business_id"]].append(q)
    rows, st = [], collections.Counter()
    for b in sorted(by):
        idx = EntityIndex.from_business(load_business_cached(os.path.join(data_root, "data", b), "/tmp/datamate_cache_khang_direction_v1"))
        for q in by[b]:
            m = CODE_LINE.search(question_hints(q["question"], q["as_of_date"], lun, idx))
            if not m:
                st["no_code"] += 1
                continue
            code = m.group(1).strip()
            fns = [f for f in re.findall(r"\b([a-z_]+)\(", code) if f != "answer"]
            rows.append({"id": q["id"], "business_id": b, "question": q["question"], "code": code,
                         "fn": fns[0] if fns else ("unanswerable" if "unanswerable" in code else "literal")})
    rows.sort(key=lambda r: r["id"])
    with open(out + ".tmp", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(out + ".tmp", out)
    print(json.dumps({"rows": len(rows), "stats": dict(st), "n_fn": len({r["fn"] for r in rows}),
                      "sha256": hashlib.sha256(open(out, "rb").read()).hexdigest()}))


if __name__ == "__main__":
    main(*sys.argv[1:5])
