"""v2 (11/11/2026): them bien the giu nguyen nghia (loi chao/dem, T3/Q1, bo dau cau, %, khach/don). Buoc 2: kiem do ben cua hint Khang voi bien the cach hoi (chi tap TRAIN learning).

Moi cau goc co hint dung -> sinh bien the bang chuong trinh (khong dau, teencode, doi cach ghi thang/nam,
viet thuong/hoa) -> chay lai hint + kernel -> so dap an voi dap an goc.
stdout chi in so tong hop theo loai bien the; chi tiet tung ID ghi vao PRIVATE_DATA.

Dung: python hint_robustness.py <khang_dir> <data_root> <split_json> <tier1_per_id_jsonl> <out_jsonl> [max_q]
"""
import sys, os, json, re, datetime, collections, hashlib, unicodedata, random, warnings
warnings.filterwarnings("ignore")


def strip_accents(s):
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


TEEN = [(r"\bkhông\b", "ko"), (r"\bđược\b", "dc"), (r"\bmình\b", "mik"), (r"\bcủa\b", "kua"),
        (r"\bbao nhiêu\b", "bn"), (r"\bgiúp\b", "jum"), (r"\bvới\b", "vs")]


def teencode(s):
    for a, b in TEEN:
        s = re.sub(a, b, s, flags=re.I)
    return s


def month_style(s):
    # "tháng 3/2025" <-> "tháng 3 năm 2025"; "3/2025" -> "03/2025"
    s2 = re.sub(r"tháng (\d{1,2})/(\d{4})", r"tháng \1 năm \2", s)
    if s2 == s:
        s2 = re.sub(r"tháng (\d{1,2}) năm (\d{4})", r"tháng \1/\2", s)
    if s2 == s:
        s2 = re.sub(r"(?<![\d/])(\d)/(\d{4})\b", r"0\1/\2", s)
    return s2


def lower(s):
    return s.lower()




def filler(s):
    return "Cho mình hỏi " + s[0].lower() + s[1:] + " ạ" if s else s


def month_abbr(s):
    s2 = re.sub(r"\btháng (\d{1,2})\b", r"T\1", s)
    return re.sub(r"\bquý (\d)\b", r"Q\1", s2)


def no_punct(s):
    return re.sub(r"\s+", " ", re.sub(r"[?,.!]", " ", s)).strip()


def pct_word(s):
    s2 = s.replace("%", " phần trăm") if "%" in s else re.sub(r"\bphần trăm\b", "%", s)
    return re.sub(r" +", " ", s2)


def short_words(s):
    s2 = re.sub(r"\bkhách hàng\b", "khách", s)
    return re.sub(r"\bđơn hàng\b", "đơn", s2)


def spaces(s):
    return "  " + s.replace(" ", "  ") + "  "


VARIANTS = {"no_accent": strip_accents, "teencode": teencode, "month_style": month_style, "lower": lower,
            "filler": filler, "month_abbr": month_abbr, "no_punct": no_punct, "pct_word": pct_word,
            "short_words": short_words, "spaces": spaces}


def main():
    khang, data_root, split_path, per_id, out_path = sys.argv[1:6]
    max_q = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    sys.path.insert(0, os.path.join(khang, "src"))
    os.chdir(khang)
    from datamate.kernel import BusinessKernel, load_business_cached
    from datamate.hints import question_hints, EntityIndex
    from datamate.dates import LunarCalendar

    CACHE = "/tmp/datamate_cache_khang_direction_v1"
    ok_ids = {}
    for l in open(per_id, encoding="utf-8"):
        r = json.loads(l)
        if r.get("split") == "train" and r.get("status") == "hint_answer" and r.get("our_score") == 1:
            ok_ids[r["id"]] = r["answer_sha256"]
    lun = LunarCalendar.load(f"{data_root}/calendar.csv")
    qs = [json.loads(l) for l in open(f"{data_root}/questions.jsonl", encoding="utf-8")]
    qs = [q for q in qs if q["id"] in ok_ids]
    if max_q:
        random.Random(20261111).shuffle(qs)
        qs = qs[:max_q]
    by = collections.defaultdict(list)
    for q in qs:
        by[q["business_id"]].append(q)

    agg = collections.defaultdict(collections.Counter)
    hsh = lambda a: hashlib.sha256(json.dumps(a, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    with open(out_path + ".tmp", "w", encoding="utf-8") as out:
        for b in sorted(by):
            biz = load_business_cached(f"{data_root}/data/{b}", CACHE)
            idx = EntityIndex.from_business(biz)
            k = BusinessKernel(f"{data_root}/data/{b}", CACHE, f"{data_root}/calendar.csv")
            for q in by[b]:
                for vn, fn in VARIANTS.items():
                    v = fn(q["question"])
                    if v == q["question"]:
                        agg[vn]["unchanged"] += 1
                        continue
                    h = question_hints(v, datetime.date.fromisoformat(q["as_of_date"]), lun, idx)
                    m = re.search(r"Code gợi ý \([^)]*\): (.*)$", h, re.M)
                    if not m:
                        st = "lost_hint"
                    else:
                        r = k.execute(q["id"], m.group(1), q["as_of_date"])
                        a = r.get("answer")
                        st = "error" if (r.get("error") or not a) else ("same" if hsh(a) == ok_ids[q["id"]] else "changed")
                    agg[vn][st] += 1
                    out.write(json.dumps({"id": q["id"], "variant": vn, "status": st}) + "\n")
            k.close()
    os.replace(out_path + ".tmp", out_path)
    print(json.dumps({"n_questions": len(qs), "by_variant": {k: dict(v) for k, v in agg.items()}}, indent=1))


if __name__ == "__main__":
    main()
