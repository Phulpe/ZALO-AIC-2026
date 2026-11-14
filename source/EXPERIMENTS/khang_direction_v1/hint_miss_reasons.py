"""Chan doan ly do khong co code hint (chi so tong hop; khong in noi dung cau hoi).

Dung: python hint_miss_reasons.py <khang_dir> <data_root> <per_id_jsonl> <status_filter_split|ALL> <out_json>
Ghi tung ID -> ly do vao <out_json>.jsonl; stdout chi in Counter.
"""
import sys, os, json, re, datetime, collections, warnings
warnings.filterwarnings("ignore")


def main():
    khang, data_root, per_id, split_f, out = sys.argv[1:6]
    sys.path.insert(0, os.path.join(khang, "src"))
    os.chdir(khang)
    from datamate import hints as H
    from datamate.kernel import load_business_cached
    from datamate.dates import LunarCalendar

    ids = set()
    for l in open(per_id, encoding="utf-8"):
        r = json.loads(l)
        if r.get("status") == "no_hint" and (split_f == "ALL" or r.get("split") == split_f):
            ids.add(r["id"])
    trace = {}
    for name in ("_special_code", "_code_hint", "_guard_code", "_unexplained"):
        f = getattr(H, name)
        def w(*a, _f=f, _n=name, **k):
            r = _f(*a, **k)
            trace.setdefault(_n, []).append(r)
            return r
        setattr(H, name, w)
    CACHE = "/tmp/datamate_cache_khang_direction_v1"
    lun = LunarCalendar.load(f"{data_root}/calendar.csv")
    qs = [json.loads(l) for l in open(f"{data_root}/questions.jsonl", encoding="utf-8")]
    qs = [q for q in qs if q["id"] in ids]
    by = collections.defaultdict(list)
    for q in qs:
        by[q["business_id"]].append(q)
    agg = collections.Counter()
    with open(out + ".jsonl", "w", encoding="utf-8") as fo:
        for b in sorted(by):
            idx = H.EntityIndex.from_business(load_business_cached(f"{data_root}/data/{b}", CACHE))
            for q in by[b]:
                trace.clear()
                txt = H.question_hints(q["question"], datetime.date.fromisoformat(q["as_of_date"]), lun, idx)
                n_per = len(H.question_periods(q["question"], datetime.date.fromisoformat(q["as_of_date"]), lun))
                has_measure = "Dạng số đo:" in txt
                cand = [c for c in trace.get("_special_code", []) + trace.get("_code_hint", []) if c]
                guarded = [c for c in trace.get("_guard_code", []) if c]
                unexpl = [u for u in trace.get("_unexplained", []) if u]
                if not has_measure and not cand:
                    rs = "no_measure_recognized"
                elif not cand:
                    rs = "measure_but_no_code" + ("_multi_period" if n_per > 1 else "")
                elif not guarded:
                    rs = "code_rejected_by_guard"
                elif unexpl:
                    rs = "code_rejected_unexplained_words"
                else:
                    rs = "other"
                rs += f"|periods={min(n_per, 2)}"
                agg[rs] += 1
                fo.write(json.dumps({"id": q["id"], "reason": rs}) + "\n")
    print(len(qs), json.dumps(dict(agg.most_common()), indent=1))


if __name__ == "__main__":
    main()
