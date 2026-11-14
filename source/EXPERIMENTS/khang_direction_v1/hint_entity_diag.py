"""Chan doan: vi sao loi dem lam mat dong 'Ten trong du lieu khop'. Chi in ten cot, so luong va token loi dem (do script sinh),
khong in noi dung cau hoi hay gia tri du lieu.
Dung: python hint_entity_diag.py <khang_dir> <data_root> <robust_jsonl> <variants_csv>
"""
import sys, os, json, collections, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hint_robustness_v3 import VARIANTS  # noqa: E402

FILLER = {"cho", "minh", "hoi", "a", "shop", "oi", "vay", "phan", "tram"}


def main():
    khang, data_root, rj, vnames = sys.argv[1:5]
    vnames = vnames.split(",")
    sys.path.insert(0, os.path.join(khang, "src"))
    os.chdir(khang)
    from datamate import hints as H
    from datamate.kernel import load_business_cached
    lost = collections.defaultdict(set)
    for l in open(rj, encoding="utf-8"):
        r = json.loads(l)
        if r["status"] != "same" and r["status"] != "unchanged" and r["variant"] in vnames:
            lost[r["id"]].add(r["variant"])
    qs = {q["id"]: q for q in map(json.loads, open(f"{data_root}/questions.jsonl", encoding="utf-8")) if q["id"] in lost}
    by = collections.defaultdict(list)
    for q in qs.values():
        by[q["business_id"]].append(q)
    agg = collections.Counter()
    for b in sorted(by):
        idx = H.EntityIndex.from_business(load_business_cached(f"{data_root}/data/{b}", "/tmp/datamate_cache_khang_direction_v1"))
        ent_tok = {(c, v): t for c, v, t in idx.entries}
        for q in by[b]:
            m0 = idx.match(q["question"])
            for vn in sorted(lost[q["id"]]):
                v = VARIANTS[vn](q["question"])
                m1 = idx.match(v)
                gone, new = set(m0) - set(m1), set(m1) - set(m0)
                newtok = sorted({x for e in new for x in ent_tok.get(e, []) if x in FILLER})
                print(json.dumps({"id_sha": q["id"][-6:], "variant": vn, "n0": len(m0), "n1": len(m1),
                                  "gone_cols": sorted(c for c, _ in gone), "new_cols": sorted(c for c, _ in new),
                                  "new_filler_tokens": newtok, "new_ntok": sorted(len(ent_tok.get(e, [])) for e in new)}))
                agg[(vn, bool(gone), bool(new))] += 1
    print({f"{a}|gone={b}|new={c}": n for (a, b, c), n in agg.items()})


if __name__ == "__main__":
    main()
