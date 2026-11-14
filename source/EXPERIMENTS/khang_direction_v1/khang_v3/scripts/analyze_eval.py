"""Phan tich mot lan danh gia: diem theo nhom cau hoi, theo type, ty le bo trong; in cac cau sai kem code.

Dung: python scripts/analyze_eval.py <thu_muc_ket_qua> <answers.jsonl> [--show 15]
"""
import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datamate.answers import score  # noqa: E402
from datamate.parsing import norm_key_ascii as plain  # noqa: E402

FAMILIES = [
    ("cong_no", r"\bno\b|cong no|mua chiu|thu no|tra no|ghi no|qua han"),
    ("ton_kho_lo", r"\bton\b|het hang|het han|\blo\b|trong kho"),
    ("nhap_hang", r"nhap|nha cung cap|ncc"),
    ("lai_gia_von", r"lai|loi nhuan|gia von|loi gop"),
    ("giao_hang", r"giao|dot"),
    ("gia_ban", r"gia ban|don gia|tang gia"),
    ("ca", r"\bca (sang|chieu)\b"),
    ("nhan_vien", r"nhan vien|\btho\b|cham chi"),
    ("khach", r"khach"),
    ("huy_tra", r"huy|tra hang|tra lai"),
    ("top_ban_chay", r"ban chay|nhieu nhat|hot nhat|top|uu chuong"),
    ("so_don", r"\bdon\b"),
    ("so_luong", r"ban duoc|ban ra"),
    ("doanh_thu", r"doanh thu|dthu|tien|thu ve"),
]


def family(q: str) -> str:
    s = plain(q)
    return next((f for f, p in FAMILIES if re.search(p, s)), "khac")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("gold")
    ap.add_argument("--show", type=int, default=15)
    a = ap.parse_args()
    gold = {g["id"]: g for g in map(json.loads, open(a.gold, encoding="utf-8"))}
    logs = []
    for f in os.listdir(os.path.join(a.out, "logs")):
        if f.endswith(".json"):
            logs.append(json.load(open(os.path.join(a.out, "logs", f), encoding="utf-8")))
    by_fam = collections.defaultdict(collections.Counter)
    by_type = collections.defaultdict(collections.Counter)
    wrong = []
    times = []
    for d in logs:
        g = gold.get(d["id"])
        if g is None:
            continue
        s = score(d["answer"], g)
        tag = {1: "dung", -1: "sai", 0: "bo"}[s]
        by_fam[family(d.get("question", ""))][tag] += 1
        by_type[g["type"]][tag] += 1
        times.append(d.get("elapsed_ms", 0))
        if s == -1:
            wrong.append((d, g))
    def line(name, c):
        n = sum(c.values())
        return f"{name:14s} n={n:4d} dung={c['dung']:4d} sai={c['sai']:4d} bo={c['bo']:4d} PA={(c['dung'] - c['sai']) / max(1, n):+.3f}"
    tot = collections.Counter()
    for c in by_fam.values():
        tot.update(c)
    print(line("TONG", tot))
    if times:
        print(f"elapsed_ms trung binh {sum(times) / len(times):.0f}, max {max(times)}")
    print("-- theo nhom")
    for k, c in sorted(by_fam.items(), key=lambda x: -sum(x[1].values())):
        print(line(k, c))
    print("-- theo type dap an chuan")
    for k, c in sorted(by_type.items(), key=lambda x: -sum(x[1].values())):
        print(line(k, c))
    print(f"-- {min(a.show, len(wrong))} cau sai dau tien")
    for d, g in wrong[: a.show]:
        print("=" * 100)
        print(d["business_id"], d["id"], "|", d.get("question"))
        print("du doan:", json.dumps(d["answer"], ensure_ascii=False)[:200], "| dung:", json.dumps(g.get("answer"), ensure_ascii=False)[:200], "| phieu:", d.get("vote", {}).get("votes"))
        for st in (d.get("samples") or [{}])[0].get("steps", [])[-2:]:
            print("--- code:\n" + (st.get("code") or "")[:1200])
            print("--- output:\n" + (st.get("output") or "")[:600])


if __name__ == "__main__":
    main()
