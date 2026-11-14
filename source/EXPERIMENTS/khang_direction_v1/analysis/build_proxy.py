"""Tap proxy cho cai thien phan LLM (14/11/2026), chi TRAIN, khong public.
 - nohint: moi cau TRAIN khong co Code goi y (51 = 38 cu + 13), khong co trong ngan hang few-shot (chi gom cau co hint).
 - hintoff: cau calibration/audit CO hint (khong thuoc learning -> khong co trong ngan hang few-shot), chay an hint;
   uu tien khuon code it gap tren TRAIN (giong cau LLM tren public). Lay N_HINTOFF cau theo support tang dan.
Chia dev/holdout phan tang theo (nguon, nhom support), seed co dinh. Ghi ids + manifest sha. stdout chi so luong.
Dung: python build_proxy.py <khang_dir> <train_root(3000)> <tier1_per_id> <out_dir> [n_hintoff=110]
"""
import sys, os, json, re, random, hashlib, datetime, collections, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from public_hint_risk import CODE, shape  # noqa: E402


def main():
    khang, root, per_id, out = sys.argv[1:5]
    n_ho = int(sys.argv[5]) if len(sys.argv) > 5 else 110
    assert not os.path.exists(out)
    split = {}
    for l in open(per_id, encoding="utf-8"):
        r = json.loads(l)
        split[r["id"]] = (r["split"], r["status"])
    sys.path.insert(0, os.path.join(khang, "src"))
    os.chdir(khang)
    from datamate import hints as H
    from datamate.kernel import load_business_cached
    from datamate.dates import LunarCalendar
    lun = LunarCalendar.load(f"{root}/calendar.csv")
    qs = [json.loads(l) for l in open(f"{root}/questions.jsonl", encoding="utf-8")]
    by = collections.defaultdict(list)
    for q in qs:
        by[q["business_id"]].append(q)
    shp = {}
    for b in sorted(by):
        idx = H.EntityIndex.from_business(load_business_cached(f"{root}/data/{b}", os.environ.get("PROXY_CACHE", os.path.join(out + "_cache"))))
        for q in by[b]:
            m = CODE.search(H.question_hints(q["question"], datetime.date.fromisoformat(q["as_of_date"]), lun, idx))
            if m:
                shp[q["id"]] = shape(m.group(1))
    sup = collections.Counter(shp.values())
    SPL = set(os.environ.get("PROXY_SPLITS", "train,calibration,audit").split(","))
    nohint = sorted(i for i, (s, st) in split.items() if st == "no_hint" and i not in shp and s in SPL)
    cand = [i for i, (s, st) in split.items() if s in ({"calibration", "audit"} & SPL) and st == "hint_answer" and i in shp]
    rng = random.Random(20261114)
    rng.shuffle(cand)
    cand.sort(key=lambda i: sup[shp[i]])
    hintoff = cand[:n_ho]
    bucket = lambda i: "nohint" if i in nohint else ("ho_s<=3" if sup[shp[i]] <= 3 else "ho_s>3")
    dev, hold = [], []
    groups = collections.defaultdict(list)
    for i in nohint + hintoff:
        groups[bucket(i)].append(i)
    for g, ids in sorted(groups.items()):
        ids = sorted(ids)
        rng.shuffle(ids)
        h = len(ids) // 2
        dev += ids[:len(ids) - h]
        hold += ids[len(ids) - h:]
    os.makedirs(out)
    meta = {}
    for name, ids in (("dev", sorted(dev)), ("holdout", sorted(hold))):
        p = os.path.join(out, f"proxy_{name}_ids.txt")
        open(p, "w", newline="\n").write("".join(i + "\n" for i in ids))
        meta[name] = {"n": len(ids), "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest(),
                      "by_bucket": dict(collections.Counter(bucket(i) for i in ids))}
    meta["hintoff_ids_note"] = "cau co hint: phai chay an dong Code goi y (run_hintsoff.py)"
    meta["hintoff_set"] = sorted(hintoff)
    meta["nohint_n"] = len(nohint)
    meta["splits"] = sorted(SPL)
    meta["support_of_hintoff"] = dict(collections.Counter(min(sup[shp[i]], 99) for i in hintoff))
    json.dump(meta, open(os.path.join(out, "PROXY_MANIFEST.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k in ("dev", "holdout", "nohint_n")}))


if __name__ == "__main__":
    main()
