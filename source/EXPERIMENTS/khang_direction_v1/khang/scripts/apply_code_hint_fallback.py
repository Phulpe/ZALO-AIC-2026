"""Ap luat du phong cua runner (src/datamate/runner.py) len mot file nop da co, khong goi model.

Runner hien tai: cau co "Code gợi ý" thi chay thu dong do trong sandbox truoc; neu model
  - khong nop duoc (abstain: het buoc / het gio) -> nop ket qua chay thu;
  - bao unanswerable trong khi chay thu ra mot gia tri that -> nop ket qua chay thu.
Script nay tinh lai goi y + chay thu bang code hien tai va ap dung dung hai luat do cho file nop cu
(dung khi file nop duoc tao boi ban runner truoc khi co luat). Cac cau khac giu nguyen.

Chay: PYTHONPATH=src .venv/bin/python scripts/apply_code_hint_fallback.py <split> <submission_vao> <submission_ra> [cache]
"""

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datamate.dates import LunarCalendar  # noqa: E402
from datamate.hints import EntityIndex, question_hints  # noqa: E402
from datamate.kernel import KernelPool, load_business_cached  # noqa: E402
from datamate.runner import _prerun_code_hint  # noqa: E402


def main() -> None:
    split, src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
    cache = sys.argv[4] if len(sys.argv) > 4 else "/tmp/datamate_cache_dev"
    data = os.path.join("data", split)
    sub = [json.loads(line) for line in open(src, encoding="utf-8") if line.strip()]
    qs = {json.loads(line)["id"]: json.loads(line) for line in open(os.path.join(data, "questions.jsonl"), encoding="utf-8") if line.strip()}
    cal = os.path.join(data, "calendar.csv")
    lunar = LunarCalendar.load(cal) if os.path.exists(cal) else None
    pool = KernelPool(cache, cal if lunar else None, workers_per_business=1)
    idx, changed = {}, []
    for r in sub:
        if r.get("type") not in ("abstain", "unanswerable"):
            continue
        q = qs[r["id"]]
        b = q["business_id"]
        path = os.path.join(data, "data", b)
        if b not in idx:
            idx[b] = EntityIndex.from_business(load_business_cached(path, cache))
        pre: dict = {}
        _prerun_code_hint(pool, path, q, question_hints(q["question"], q["as_of_date"], lunar, idx[b]), out=pre)
        a = pre.get("answer")
        if a and (r["type"] == "abstain" or a.get("type") not in ("unanswerable", "abstain")):
            changed.append((r["id"], r["type"], a))
            r.clear()
            r.update({"id": q["id"], **a})
    pool.close()
    with open(dst, "w", encoding="utf-8") as fh:
        for r in sub:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    for qid, old, a in changed:
        print(f"{qid}: {old} -> {json.dumps(a, ensure_ascii=False)}")
    print(f"{len(sub)} dong, doi {len(changed)} cau -> {dst}")


if __name__ == "__main__":
    main()
