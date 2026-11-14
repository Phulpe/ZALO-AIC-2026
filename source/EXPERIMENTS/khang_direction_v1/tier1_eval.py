"""GD1: do tang code goi y (Khang c6379f5) tren 3 tap TRAIN da khoa theo doanh nghiep.

Chay trong WSL Linux (sandbox can SIGALRM). Khong in cau hoi/gold/dap an ra stdout:
stdout chi co so tong hop; ket qua tung ID (id, split, hint_present, status, score) ghi vao PRIVATE_DATA.
Cham bang ca scorer cua Khang (answers.score) va scorer da pin cua minh (scoring.matches, df6dea2f...).

Dung: python tier1_eval.py <khang_dir> <our_scoring_dir> <split_json> <out_jsonl> [--hints-off-list <path>]
"""
import sys, os, json, re, datetime, collections, hashlib, warnings
warnings.filterwarnings("ignore")

def main():
    khang, our_dir, split_path, out_path = sys.argv[1:5]
    sys.path.insert(0, os.path.join(khang, "src"))
    sys.path.insert(0, our_dir)
    os.chdir(khang)
    from datamate.kernel import BusinessKernel, load_business_cached  # noqa: E402
    from datamate.hints import question_hints, EntityIndex  # noqa: E402
    from datamate.dates import LunarCalendar  # noqa: E402
    from datamate.answers import score as khang_score  # noqa: E402
    import scoring as our_scoring  # noqa: E402

    CACHE = "/tmp/datamate_cache_khang_direction_v1"
    split = json.load(open(split_path, encoding="utf-8"))
    biz_split = {b: s for s, bs in split.items() for b in bs}
    lun = LunarCalendar.load(f"{os.environ.get('TIER1_DATA', 'data/train')}/calendar.csv")
    qs = [json.loads(l) for l in open(f"{os.environ.get('TIER1_DATA', 'data/train')}/questions.jsonl", encoding="utf-8")]
    data_root = os.environ.get("TIER1_DATA", "data/train")
    gold = {}
    if os.path.exists(f"{data_root}/answers.jsonl"):
        for l in open(f"{data_root}/answers.jsonl", encoding="utf-8"):
            g = json.loads(l); gold[g["id"]] = g
    only = set(os.environ.get("TIER1_SPLITS", "").split(",")) - {""}

    by = collections.defaultdict(list)
    for q in qs:
        by[q["business_id"]].append(q)

    agg = collections.defaultdict(collections.Counter)
    disagree = 0
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path + ".tmp", "w", encoding="utf-8") as out:
        for b in sorted(by):
            s = biz_split.get(b, "?")
            biz = load_business_cached(f"{data_root}/data/{b}", CACHE)
            idx = EntityIndex.from_business(biz)
            k = None
            for q in by[b]:
                h = question_hints(q["question"], datetime.date.fromisoformat(q["as_of_date"]), lun, idx)
                m = re.search(r"Code gợi ý \([^)]*\): (.*)$", h, re.M)
                rec = {"id": q["id"], "business_id": b, "split": s, "hint_present": bool(m)}
                if not m:
                    rec["status"] = "no_hint"; agg[s]["no_hint"] += 1
                else:
                    if k is None:
                        k = BusinessKernel(f"{data_root}/data/{b}", CACHE, f"{data_root}/calendar.csv")
                    r = k.execute(q["id"], m.group(1), q["as_of_date"])
                    ans = r.get("answer")
                    if r.get("error") or not ans:
                        rec["status"] = "hint_error"; agg[s]["hint_error"] += 1
                    else:
                        rec.update(status="hint_answer",
                                   answer_sha256=hashlib.sha256(json.dumps(ans, ensure_ascii=False, sort_keys=True).encode()).hexdigest())
                        g = gold.get(q["id"])
                        if g is None:
                            agg[s]["answered_unscored"] += 1
                        else:
                            ks = khang_score(ans, g)
                            os_ = 1 if our_scoring.matches(ans, g) else -1
                            if ks != os_:
                                disagree += 1
                            rec.update(khang_score=ks, our_score=os_)
                            agg[s]["correct" if os_ == 1 else "wrong"] += 1
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if k:
                k.close()
    os.replace(out_path + ".tmp", out_path)

    summary = {}
    for s, c in sorted(agg.items()):
        n = sum(c.values())
        summary[s] = dict(c, n=n, PA_tier1_only=round((c["correct"] - c["wrong"]) / n, 4),
                          hint_coverage=round((n - c["no_hint"]) / n, 4))
    summary["scorer_disagreements_khang_vs_ours"] = disagree
    print(json.dumps(summary, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
