"""Cham rui ro cau tra loi hint (khong gold): 13/11/2026.

Hai tin hieu, khong dung dap an public:
 (1) do ben: bien the giu nguyen nghia (VARIANTS cua hint_robustness_v3) lam dap an hint DOI hoac MAT hint.
     Tren TRAIN 23 bien the x 600 cau: 0 doi -> tren public, doi = dau hieu luat khop sai.
 (2) do hiem cua khuon code: chuan hoa code hint (thay so/chuoi/ngay bang placeholder) -> dem so cau TRAIN
     (3000, luat Khang viet tu dap an TRAIN) dung cung khuon. Khuon it gap tren TRAIN de khop qua sat.
Ghi tung ID -> <out>.jsonl (PRIVATE_DATA). stdout chi in so tong hop.

Dung: python public_hint_risk.py <khang_dir> <train_data_root> <public_data_root> <out_prefix> [variants_csv]
"""
import sys, os, json, re, datetime, collections, hashlib, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hint_robustness_v3 import VARIANTS  # noqa: E402

CODE = re.compile(r"Code gợi ý \([^)]*\): (.*)$", re.M)


def shape(code):
    s = re.sub(r"'[^']*'|\"[^\"]*\"", "S", code)
    s = re.sub(r"\b\d+(?:\.\d+)?\b", "N", s)
    return re.sub(r"\s+", "", s)


def main():
    khang, train_root, pub_root, out = sys.argv[1:5]
    only = set(sys.argv[5].split(",")) if len(sys.argv) > 5 and sys.argv[5] else None
    sys.path.insert(0, os.path.join(khang, "src"))
    os.chdir(khang)
    from datamate import hints as H
    from datamate.kernel import BusinessKernel, load_business_cached
    from datamate.dates import LunarCalendar
    CACHE = "/tmp/datamate_cache_hint_risk"
    hsh = lambda a: hashlib.sha256(json.dumps(a, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    def hints_for(root):
        lun = LunarCalendar.load(f"{root}/calendar.csv")
        qs = [json.loads(l) for l in open(f"{root}/questions.jsonl", encoding="utf-8")]
        by = collections.defaultdict(list)
        for q in qs:
            by[q["business_id"]].append(q)
        return lun, by

    # (2) khuon TRAIN
    sup = collections.Counter()
    lun, by = hints_for(train_root)
    n_train = 0
    for b in sorted(by):
        idx = H.EntityIndex.from_business(load_business_cached(f"{train_root}/data/{b}", CACHE))
        for q in by[b]:
            m = CODE.search(H.question_hints(q["question"], datetime.date.fromisoformat(q["as_of_date"]), lun, idx))
            if m:
                sup[shape(m.group(1))] += 1
                n_train += 1
    # (1)+(2) public
    lun, by = hints_for(pub_root)
    vs = {k: v for k, v in VARIANTS.items() if not only or k in only}
    agg = collections.Counter()
    sup_hist = collections.Counter()
    rows = []
    for b in sorted(by):
        idx = H.EntityIndex.from_business(load_business_cached(f"{pub_root}/data/{b}", CACHE))
        k = BusinessKernel(f"{pub_root}/data/{b}", CACHE, f"{pub_root}/calendar.csv")
        for q in by[b]:
            D = datetime.date.fromisoformat(q["as_of_date"])
            m0 = CODE.search(H.question_hints(q["question"], D, lun, idx))
            if not m0:
                continue
            r0 = k.execute(q["id"], m0.group(1), q["as_of_date"])
            if r0.get("error") or not r0.get("answer"):
                continue
            a0 = hsh(r0["answer"])
            ch, lost, tried = [], [], 0
            for vn, fn in vs.items():
                v = fn(q["question"])
                if v == q["question"]:
                    continue
                tried += 1
                m = CODE.search(H.question_hints(v, D, lun, idx))
                if not m:
                    lost.append(vn)
                    continue
                r = k.execute(q["id"], m.group(1), q["as_of_date"])
                if r.get("error") or not r.get("answer") or hsh(r["answer"]) != a0:
                    ch.append(vn)
            s = sup[shape(m0.group(1))]
            sb = "0" if s == 0 else "1" if s == 1 else "2-3" if s <= 3 else "4-9" if s <= 9 else "10+"
            sup_hist[sb] += 1
            agg["hint"] += 1
            agg["changed_any"] += bool(ch)
            agg["lost_any"] += bool(lost)
            rows.append({"id": q["id"], "business_id": b, "train_support": s, "variants_tried": tried,
                         "changed": ch, "lost": lost, "answer_sha256": a0})
        k.close()
    with open(out + ".jsonl.tmp", "w", encoding="utf-8") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in rows))
    os.replace(out + ".jsonl.tmp", out + ".jsonl")
    summ = {"train_hint_questions": n_train, "train_shapes": len(sup), "public": dict(agg),
            "public_support_hist": dict(sup_hist),
            "changed_by_variant": dict(collections.Counter(v for r in rows for v in r["changed"]).most_common()),
            "changed_and_support<=3": sum(1 for r in rows if r["changed"] and r["train_support"] <= 3)}
    json.dump(summ, open(out + ".summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
