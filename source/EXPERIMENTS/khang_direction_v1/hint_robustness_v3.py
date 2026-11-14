"""v3 (12/11/2026): kiem do ben hint voi bien the giu nguyen nghia rong hon v2 + chan doan giai doan that bai.
Chi tap TRAIN learning co hint dung gold. stdout chi in so tong hop + tien to mau goi y (van ban do code sinh,
khong phai noi dung cau hoi). Chi tiet tung ID -> PRIVATE_DATA.

Dung: python hint_robustness_v3.py <khang_dir> <data_root> <tier1_per_id_jsonl> <out_jsonl> [max_q] [variants,csv]
"""
import sys, os, json, re, datetime, collections, hashlib, random, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hint_robustness_v2 import VARIANTS as V2  # noqa: E402

_PER = r"(?:tháng|quý)\s+\d{1,2}(?:/\d{4}|\s+năm\s+\d{4})|năm \d{4}"
_ANY_PER = re.compile(r"\b(?:tháng|quý|năm|tuần|ngày|hôm|từ|đến|tới|so với)\b|\d{1,2}/\d{1,2}", re.I)


def period_front(s):
    # chi doi khi cau co DUNG MOT cum ky day du (thang/quy kem nam, hoac nam) va khong co moc thoi gian nao khac
    ms = list(re.finditer(rf"\b(?:trong |của |vào )?({_PER})\b", s))
    if len(ms) != 1 or ms[0].start() == 0:
        return s
    m = ms[0]
    if _ANY_PER.search(s[:m.start()] + " " + s[m.end():]):
        return s
    rest = (s[:m.start()] + s[m.end():]).strip()
    rest = re.sub(r"\s+([?,.!])", r"\1", re.sub(r"\s+", " ", rest))
    return m.group(1)[0].upper() + m.group(1)[1:] + ", " + rest[0].lower() + rest[1:]


def month_pad(s):
    return re.sub(r"\btháng (\d)/(\d{4})\b", r"tháng 0\1/\2", s)


def month_dash(s):
    return re.sub(r"\btháng (\d{1,2})/(\d{4})\b", r"tháng \1-\2", s)


def la_bao_nhieu(s):
    return re.sub(r"(?<!là )\bbao nhiêu\b", "là bao nhiêu", s, count=1)


def greet_tail(s):
    return "Shop ơi, " + s[0].lower() + s[1:].rstrip(" ?.") + " vậy?"


def upper(s):
    return s.upper()


def old_tone(s):
    for a, b in (("hóa", "hoá"), ("hòa", "hoà"), ("thủy", "thuỷ"), ("khỏe", "khoẻ"), ("lũy", "luỹ"), ("úy", "uý"), ("ủy", "uỷ")):
        s = s.replace(a, b) if a in s else s.replace(b, a)
    return s


def trail_marks(s):
    return s.rstrip(" ?.!") + "??? 🙏"


def no_space_comma(s):
    return re.sub(r",\s+", ",", s)


def no_punct_safe(s):
    # bo ?,.! nhung giu dau trong so (1.500.000, 2,5%, 3.5)
    return re.sub(r"\s+", " ", re.sub(r"(?<!\d)[?,.!]|[?,.!](?!\d)", " ", s)).strip()


def filler_head(s):
    return "Cho mình hỏi " + s[0].lower() + s[1:] if s else s


def filler_tail(s):
    return s + " ạ"


def greet_head(s):
    return "Shop ơi, " + s[0].lower() + s[1:]


def tail_vay(s):
    return s.rstrip(" ?.") + " vậy?"


VARIANTS = dict(V2)
VARIANTS["no_punct"] = no_punct_safe
VARIANTS.update({"period_front": period_front, "month_pad": month_pad, "month_dash": month_dash, "la_bao_nhieu": la_bao_nhieu,
                 "greet_tail": greet_tail, "upper": upper, "old_tone": old_tone, "trail_marks": trail_marks,
                 "no_space_comma": no_space_comma, "filler_head": filler_head, "filler_tail": filler_tail,
                 "greet_head": greet_head, "tail_vay": tail_vay})


def main():
    khang, data_root, per_id, out_path = sys.argv[1:5]
    max_q = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    only = set(sys.argv[6].split(",")) if len(sys.argv) > 6 and sys.argv[6] else None
    sys.path.insert(0, os.path.join(khang, "src"))
    os.chdir(khang)
    from datamate import hints as H
    from datamate.kernel import BusinessKernel, load_business_cached
    from datamate.dates import LunarCalendar

    trace = {}
    for name in ("_special_code", "_code_hint", "_guard_code", "_unexplained"):
        f = getattr(H, name)

        def w(*a, _f=f, _n=name, **k):
            r = _f(*a, **k)
            trace.setdefault(_n, []).append(r)
            return r
        setattr(H, name, w)

    def reason(txt):
        cand = [c for c in trace.get("_special_code", []) + trace.get("_code_hint", []) if c]
        if "Dạng số đo:" not in txt and not cand:
            return "no_measure"
        if not cand:
            return "measure_no_code"
        if not [c for c in trace.get("_guard_code", []) if c]:
            return "guard_rejected"
        if [u for u in trace.get("_unexplained", []) if u]:
            return "unexplained_words"
        return "other"

    def templ(txt):
        return {l.strip()[:48] for l in txt.splitlines() if l.strip() and not l.startswith("Code gợi ý")}

    CACHE = "/tmp/datamate_cache_khang_direction_v1"
    ok_ids = {}
    for l in open(per_id, encoding="utf-8"):
        r = json.loads(l)
        if r.get("split") == "train" and r.get("status") == "hint_answer" and r.get("our_score") == 1:
            ok_ids[r["id"]] = r["answer_sha256"]
    lun = LunarCalendar.load(f"{data_root}/calendar.csv")
    qs = [q for q in map(json.loads, open(f"{data_root}/questions.jsonl", encoding="utf-8")) if q["id"] in ok_ids]
    if max_q:
        random.Random(20261111).shuffle(qs)
        qs = qs[:max_q]
    by = collections.defaultdict(list)
    for q in qs:
        by[q["business_id"]].append(q)
    vs = {k: v for k, v in VARIANTS.items() if not only or k in only}
    agg = collections.defaultdict(collections.Counter)
    why = collections.Counter()
    lost_t = collections.Counter()
    hsh = lambda a: hashlib.sha256(json.dumps(a, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    with open(out_path + ".tmp", "w", encoding="utf-8") as out:
        for b in sorted(by):
            idx = H.EntityIndex.from_business(load_business_cached(f"{data_root}/data/{b}", CACHE))
            k = BusinessKernel(f"{data_root}/data/{b}", CACHE, f"{data_root}/calendar.csv")
            for q in by[b]:
                D = datetime.date.fromisoformat(q["as_of_date"])
                h0 = H.question_hints(q["question"], D, lun, idx)
                for vn, fn in vs.items():
                    v = fn(q["question"])
                    if v == q["question"]:
                        agg[vn]["unchanged"] += 1
                        continue
                    trace.clear()
                    h = H.question_hints(v, D, lun, idx)
                    m = re.search(r"Code gợi ý \([^)]*\): (.*)$", h, re.M)
                    rs = ""
                    if not m:
                        st, rs = "lost_hint", reason(h)
                        why[(vn, rs)] += 1
                        for t in templ(h0) - templ(h):
                            lost_t[(vn, t)] += 1
                    else:
                        r = k.execute(q["id"], m.group(1), q["as_of_date"])
                        a = r.get("answer")
                        st = "error" if (r.get("error") or not a) else ("same" if hsh(a) == ok_ids[q["id"]] else "changed")
                    agg[vn][st] += 1
                    out.write(json.dumps({"id": q["id"], "variant": vn, "status": st, "reason": rs}) + "\n")
            k.close()
    os.replace(out_path + ".tmp", out_path)
    print(json.dumps({"n_questions": len(qs), "by_variant": {k: dict(v) for k, v in agg.items()},
                      "lost_reason": {f"{a}|{b}": c for (a, b), c in why.most_common()},
                      "lost_templates": {f"{a}|{b}": c for (a, b), c in lost_t.most_common(30)}}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
