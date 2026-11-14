"""Kiem tra ket qua mot lan chay image: du cau, dung format, log du truong bat buoc; cham diem neu co dap an."""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datamate.answers import normalize, score

ap = argparse.ArgumentParser()
ap.add_argument("questions"); ap.add_argument("out"); ap.add_argument("--gold", default="")
a = ap.parse_args()
qs = [json.loads(l) for l in open(a.questions, encoding="utf-8")]
sub = {}
bad = 0
for line in open(os.path.join(a.out, "submission.jsonl"), encoding="utf-8"):
    d = json.loads(line)
    if d["id"] in sub:
        print("TRUNG ID", d["id"]); bad += 1
    sub[d["id"]] = d
    if d.get("type") not in ("abstain",):
        try:
            normalize(d["type"], d.get("answer"), d.get("reason"))
        except Exception as e:
            print("SAI FORMAT", d, e); bad += 1
missing_logs = [q["id"] for q in qs if not os.path.exists(os.path.join(a.out, "logs", q["id"] + ".json"))]
for q in qs[:5000]:
    p = os.path.join(a.out, "logs", q["id"] + ".json")
    if os.path.exists(p):
        lg = json.load(open(p, encoding="utf-8"))
        for k in ("id", "answer", "elapsed_ms", "thinking"):
            if k not in lg:
                print("LOG THIEU", k, q["id"]); bad += 1
types = {}
for d in sub.values():
    types[d.get("type")] = types.get(d.get("type"), 0) + 1
print(f"cau hoi {len(qs)}, dong submission {len(sub)}, thieu log {len(missing_logs)}, loi {bad}, theo type {types}")
times = []
for q in qs:
    p = os.path.join(a.out, "logs", q["id"] + ".json")
    if os.path.exists(p):
        times.append(json.load(open(p))["elapsed_ms"])
if times:
    print(f"elapsed_ms trung binh {sum(times)/len(times):.0f}, max {max(times)}")
if a.gold:
    gold = {g["id"]: g for g in map(json.loads, open(a.gold, encoding="utf-8"))}
    s = [score(sub.get(q["id"]), gold[q["id"]]) for q in qs if q["id"] in gold]
    print(f"dung {s.count(1)}, sai {s.count(-1)}, bo {s.count(0)}, PA {(s.count(1)-s.count(-1))/max(1,len(s)):.4f}")
