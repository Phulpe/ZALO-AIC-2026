"""Phan tich tong hop cau sai/bo qua cua run vLLM (chi TRAIN/calibration, gold dung sau khi khoa du doan).
stdout chi in dac trung tong hop (khong noi dung cau hoi/code). Chi tiet tung ID -> <out>.jsonl (PRIVATE_DATA).
Dung: python error_analysis.py <khang_src> <gold1,gold2> <out_jsonl> <logs_dir>...
"""
import sys, os, json, glob, collections, re


def bucket(p, g):
    if not isinstance(p, (int, float)) or not isinstance(g, (int, float)):
        return "non_numeric"
    if g == 0:
        return "gold_zero"
    r = p / g
    if abs(r - 1) < 0.005: return "close<0.5%"
    if abs(r - 1) < 0.05: return "close<5%"
    if r < 0: return "sign"
    for k, n in ((1000, "x1000"), (0.001, "x0.001"), (100, "x100"), (0.01, "x0.01")):
        if abs(r / k - 1) < 0.02: return n
    return "over" if r > 1 else "under"


def main():
    src, golds, out, dirs = sys.argv[1], sys.argv[2].split(","), sys.argv[3], sys.argv[4:]
    sys.path.insert(0, src)
    from datamate.answers import score
    gold = {}
    for g in golds:
        for l in open(g, encoding="utf-8"):
            d = json.loads(l); gold[d["id"]] = d
    agg = collections.defaultdict(collections.Counter)
    with open(out, "w", encoding="utf-8") as fo:
        for d in dirs:
            for f in sorted(glob.glob(os.path.join(d, "*.json"))):
                L = json.load(open(f, encoding="utf-8"))
                qid = L["id"]; a = L["answer"]; g = gold[qid]
                if a.get("type") == "abstain":
                    st = "abstain"
                else:
                    st = "correct" if score(dict(a, id=qid), g) == 1 else "wrong"
                ss = L["samples"]
                feats = {
                    "status": st, "gold_type": g["type"], "pred_type": a.get("type"),
                    "ratio": bucket(a.get("answer"), g.get("answer")) if st == "wrong" else None,
                    "n_samples": len(ss), "top_votes": L["vote"].get("top_votes"), "n_distinct": len(L["vote"].get("votes", {})),
                    "truncated": sum(s.get("completion_tokens", 0) >= 2000 for s in ss),
                    "sample_err": sum(bool(s.get("error")) for s in ss),
                    "exec_err": sum(bool(re.search(r"Traceback|Error", st_.get("output") or "")) for s in ss for st_ in s.get("steps") or []),
                    "steps_max": max((len(s.get("steps") or []) for s in ss), default=0),
                    "any_sample_correct": any(s.get("answer") and s["answer"].get("type") not in (None, "abstain")
                                              and score(dict(s["answer"], id=qid), g) == 1 for s in ss),
                }
                fo.write(json.dumps(dict(id=qid, **feats)) + "\n")
                for k, v in feats.items():
                    if k != "status":
                        agg[st][f"{k}={v}"] += 1
                agg["_all"][st] += 1
    print(json.dumps({k: dict(sorted(v.items())) for k, v in agg.items()}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
