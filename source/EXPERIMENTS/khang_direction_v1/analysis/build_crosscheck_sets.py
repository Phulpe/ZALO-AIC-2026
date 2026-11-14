"""Tao 2 tap cho kiem cheo hint bang LLM (13/11/2026), khong gold:
 - public_risk: cau public co hint rui ro (PUBLIC_HINT_RISK_V4: doi dap an khi viet lai, hoac khuon code gap <=1 lan tren TRAIN)
 - train_calib: cau TRAIN co hint (dung 100% tren TRAIN) dung khuon hiem (<=3 cau TRAIN) -> do ty le LLM 'phan doi nham'
Ghi data dir (questions.jsonl + calendar/rulebook + data/<biz>) cho tung tap. stdout chi so luong.
Dung: python build_crosscheck_sets.py <khang_dir> <train_root> <public_root> <risk_jsonl> <out_dir>
"""
import sys, os, json, re, shutil, datetime, collections, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from public_hint_risk import CODE, shape  # noqa: E402


def write_set(root, qs, out):
    os.makedirs(f"{out}/data")
    for n in ("calendar.csv", "rulebook.md"):
        shutil.copy(f"{root}/{n}", out)
    for b in sorted({q["business_id"] for q in qs}):
        shutil.copytree(f"{root}/data/{b}", f"{out}/data/{b}")
    with open(f"{out}/questions.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in qs))


def main():
    khang, train_root, pub_root, risk, out = sys.argv[1:6]
    assert not os.path.exists(out)
    rows = [json.loads(l) for l in open(risk)]
    pub_ids = {r["id"] for r in rows if r["changed"] or r["train_support"] <= 1}
    sys.path.insert(0, os.path.join(khang, "src"))
    os.chdir(khang)
    from datamate import hints as H
    from datamate.kernel import load_business_cached
    from datamate.dates import LunarCalendar
    lun = LunarCalendar.load(f"{train_root}/calendar.csv")
    tq = [json.loads(l) for l in open(f"{train_root}/questions.jsonl", encoding="utf-8")]
    by = collections.defaultdict(list)
    for q in tq:
        by[q["business_id"]].append(q)
    shp = {}
    for b in sorted(by):
        idx = H.EntityIndex.from_business(load_business_cached(f"{train_root}/data/{b}", "/tmp/datamate_cache_hint_risk"))
        for q in by[b]:
            m = CODE.search(H.question_hints(q["question"], datetime.date.fromisoformat(q["as_of_date"]), lun, idx))
            if m:
                shp[q["id"]] = shape(m.group(1))
    sup = collections.Counter(shp.values())
    calib = sorted([q for q in tq if q["id"] in shp and sup[shp[q["id"]]] <= 3], key=lambda q: q["id"])
    pq = sorted([json.loads(l) for l in open(f"{pub_root}/questions.jsonl", encoding="utf-8")], key=lambda q: q["id"])
    pub = [q for q in pq if q["id"] in pub_ids]
    write_set(pub_root, pub, f"{out}/data_public_risk")
    write_set(train_root, calib, f"{out}/data_train_calib")
    print(json.dumps({"public_risk": len(pub), "train_calib": len(calib)}))


if __name__ == "__main__":
    main()
