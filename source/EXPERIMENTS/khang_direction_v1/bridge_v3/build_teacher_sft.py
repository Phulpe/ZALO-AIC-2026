"""Huong A (12/11/2026): du lieu SFT "loi giai mau" tu code goi y tat dinh cua Khang, CHI tren learning pool TRAIN.

Moi cau TRAIN learning co dong "Code gợi ý" -> chay code trong sandbox -> chi giu neu answers.score == 1 voi gold TRAIN.
Prompt = dung dinh dang runner luc suy luan nhung BO dong "Code gợi ý" (mo phong cau khong co hint);
tra loi mau = mot khoi ```python``` chua dung dong code do (agent trich khoi code cuoi, chay, answer() nop dap an).
Khong dung calibration/audit/public. stdout chi in so tong hop; noi dung chi ghi vao file SFT (PRIVATE_DATA).

Dung: python build_teacher_sft.py <khang_src> <data_root_train> <tier1_per_id_jsonl> <gold_jsonl> <out_jsonl>
          [--max-items N] [--per-fn K] [--seed S] [--measure-only]
"""
import argparse
import collections
import datetime
import hashlib
import json
import os
import random
import re
import sys

CODE_LINE = re.compile(r"^Code gợi ý \([^)]*\): (.*)$\n?", re.M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("khang_src")
    ap.add_argument("data_root")
    ap.add_argument("per_id")
    ap.add_argument("gold")
    ap.add_argument("out")
    ap.add_argument("--max-items", type=int, default=400)
    ap.add_argument("--per-fn", type=int, default=14)
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--max-steps", type=int, default=6)
    a = ap.parse_args()
    sys.path.insert(0, a.khang_src)
    from datamate.answers import score
    from datamate.dates import LunarCalendar, date_context
    from datamate.hints import EntityIndex, question_hints
    from datamate.kernel import BusinessKernel, load_business_cached
    from datamate.profile import business_profile
    from datamate.prompts import build_first_user, build_system

    learn = set()
    for l in open(a.per_id, encoding="utf-8"):
        r = json.loads(l)
        if r.get("split") == "train" and r.get("status") == "hint_answer" and r.get("our_score") == 1:
            learn.add(r["id"])
    gold = {g["id"]: g for g in map(json.loads, open(a.gold, encoding="utf-8"))}
    qs = [q for q in map(json.loads, open(os.path.join(a.data_root, "questions.jsonl"), encoding="utf-8"))
          if q["id"] in learn and q["id"] in gold]
    rnd = random.Random(a.seed)
    rnd.shuffle(qs)
    rulebook = open(os.path.join(a.data_root, "rulebook.md"), encoding="utf-8").read()
    system = build_system(rulebook, a.max_steps)
    cal = os.path.join(a.data_root, "calendar.csv")
    lun = LunarCalendar.load(cal)
    cache = "/tmp/datamate_cache_khang_direction_v1"
    st = collections.Counter()
    per_fn = collections.Counter()
    rows = []
    by_b = collections.defaultdict(list)
    for q in qs:
        by_b[q["business_id"]].append(q)
    for b in sorted(by_b):
        biz = load_business_cached(os.path.join(a.data_root, "data", b), cache)
        prof, end, idx = business_profile(biz), (biz.data_end.isoformat() if biz.data_end else "không ghi"), EntityIndex.from_business(biz)
        del biz
        k = BusinessKernel(os.path.join(a.data_root, "data", b), cache, cal)
        for q in by_b[b]:
            st["candidates"] += 1
            h = question_hints(q["question"], q["as_of_date"], lun, idx)
            m = CODE_LINE.search(h)
            if not m:
                st["no_code_line"] += 1
                continue
            code = m.group(1).strip()
            fns = [f for f in re.findall(r"\b([a-z_]+)\(", code) if f != "answer"]
            fn = fns[0] if fns else ("unanswerable" if "unanswerable" in code else "literal")
            if per_fn[fn] >= a.per_fn:
                st["capped_fn"] += 1
                continue
            r = k.execute(q["id"], code, q["as_of_date"], soft_timeout=20.0)
            k.end(q["id"])
            ans = r.get("answer")
            if r.get("error") or not ans or score(dict(ans, id=q["id"]), gold[q["id"]]) != 1:
                st["not_correct"] += 1
                continue
            hints_off = CODE_LINE.sub("", h)
            user = build_first_user(prof, q["id"], q["as_of_date"], end, date_context(q["as_of_date"], lun), q["question"], hints_off)
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": user},
                    {"role": "assistant", "content": "```python\n" + code + "\n```"}]
            rows.append({"qid": q["id"], "k": 0, "fn": fn, "messages": msgs})
            per_fn[fn] += 1
            st["kept"] += 1
            if len(rows) >= a.max_items:
                break
        k.close()
        if len(rows) >= a.max_items:
            break
    rows.sort(key=lambda r: r["qid"])
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, a.out)
    chars = [len(r["messages"][0]["content"]) + len(r["messages"][1]["content"]) for r in rows]
    print(json.dumps({"stats": dict(st), "n_fn": len(per_fn), "top_fn": per_fn.most_common(12),
                      "prompt_chars_min_med_max": [min(chars), sorted(chars)[len(chars) // 2], max(chars)] if chars else None,
                      "system_chars": len(system), "sha256": hashlib.sha256(open(a.out, "rb").read()).hexdigest()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
