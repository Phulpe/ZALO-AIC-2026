"""Chon nguong dong thuan tu log runner (samples[k].answer), cham cuc bo sau khi khoa du doan.

stdout chi in tong hop: voi moi nguong m (so phieu toi thieu cho dap an pho bien nhat), so cau tra loi / dung / sai / PA rong.
Dung: python vote_threshold.py <khang_src> <gold_jsonl[,gold2...]> <logs_dir> [<logs_dir> ...]
"""
import sys, os, json, glob, collections


def main():
    khang_src, golds, dirs = sys.argv[1], sys.argv[2].split(","), sys.argv[3:]
    sys.path.insert(0, khang_src)
    from datamate.answers import score
    gold = {}
    for g in golds:
        for l in open(g, encoding="utf-8"):
            d = json.loads(l); gold[d["id"]] = d
    per_q = {}
    for d in dirs:
        for f in glob.glob(os.path.join(d, "*.json")):
            qid = os.path.basename(f)[:-5]
            ss = json.load(open(f, encoding="utf-8")).get("samples") or []
            per_q[qid] = [s.get("answer") for s in ss]
    n = len(per_q)
    rows = []
    max_k = max((len(v) for v in per_q.values()), default=0)
    for m in range(1, max_k + 1):
        c = collections.Counter()
        for qid, ans in per_q.items():
            valid = [a for a in ans if a and a.get("type") not in (None, "abstain")]
            votes = collections.Counter(json.dumps(a, ensure_ascii=False, sort_keys=True) for a in valid)
            if not votes or votes.most_common(1)[0][1] < m or qid not in gold:
                c["abstain"] += 1
                continue
            top = json.loads(votes.most_common(1)[0][0])
            c["correct" if score(top, gold[qid]) == 1 else "wrong"] += 1
        rows.append({"min_votes": m, **c, "net": c["correct"] - c["wrong"], "PA_on_subset": round((c["correct"] - c["wrong"]) / n, 4) if n else None})
    print(json.dumps({"n_questions": n, "by_threshold": rows}, indent=1))


if __name__ == "__main__":
    main()
