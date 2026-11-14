"""Cham submission tren 38 cau no-hint TRAIN/calibration sau khi khoa du doan (gold TRAIN). stdout chi so tong hop.
Dung: python score_eval38.py <khang_src> <gold_jsonl> <out_jsonl> <name=submission.jsonl> [...]
Ghi tung ID -> <out_jsonl> (PRIVATE_DATA).
"""
import json, sys

src, gold_f, out_f = sys.argv[1:4]
sys.path.insert(0, src)
from datamate.answers import score  # noqa: E402

gold = {json.loads(l)["id"]: json.loads(l) for l in open(gold_f, encoding="utf-8")}
per = {}
for arg in sys.argv[4:]:
    name, f = arg.split("=", 1)
    sub = [json.loads(l) for l in open(f, encoding="utf-8")]
    c = w = a = 0
    for r in sub:
        if r.get("type", "abstain") == "abstain":
            st = "abstain"; a += 1
        elif score(r, gold[r["id"]]) == 1:
            st = "correct"; c += 1
        else:
            st = "wrong"; w += 1
        per.setdefault(r["id"], {})[name] = st
    print(json.dumps({"run": name, "n": len(sub), "correct": c, "wrong": w, "abstain": a, "net": c - w}))
with open(out_f, "w", encoding="utf-8") as fh:
    fh.write("".join(json.dumps({"id": i, **v}) + "\n" for i, v in sorted(per.items())))
names = [a.split("=", 1)[0] for a in sys.argv[4:]]
if len(names) > 1:
    b = names[0]
    for n in names[1:]:
        print(json.dumps({"vs": f"{n} vs {b}", "correct_to_wrong": sum(1 for v in per.values() if v.get(b) == "correct" and v.get(n) == "wrong"),
                          "correct_to_abstain": sum(1 for v in per.values() if v.get(b) == "correct" and v.get(n) == "abstain"),
                          "fixed": sum(1 for v in per.values() if v.get(b) != "correct" and v.get(n) == "correct")}))
