"""Ap quy tac kiem cheo hint (khoa 13/11/2026 truoc khi xem ket qua), khong dung gold public.

Phieu "manh" cua LLM (hints-off): answered va votes=a/b voi a==b>=2 (moi lan chay dong y).
1) Hieu chinh tren TRAIN (hint dung 100%): FF = so cau train_calib LLM manh ma KHAC gold (= phan doi nham).
   Quy tac chi duoc bat khi FF <= 1.
2) Public: cau hint rui ro co LLM manh va dap an KHAC hint -> doi thanh abstain (khong thay bang dap an LLM).
Ghi submission moi (khong ghi de ban goc) + bao cao. stdout chi so tong hop.

Dung: python apply_crosscheck.py <khang_src> <train_gold> <xc_train_dir> <xc_public_dir> <base_submission> <out_dir>
"""
import json, os, re, sys

src, gold_f, xt, xp, base_f, out = sys.argv[1:7]
sys.path.insert(0, src)
from datamate.answers import same_answer  # noqa: E402

PROG = re.compile(r"^PROG \| \S+ \| (q_\w+) \| src=(\w+) \| (\w+) \| type=\S+ \| votes=(\d+)/(\d+)")


def strong(d):
    votes = {}
    for l in open(os.path.join(d, "runner.log"), encoding="utf-8", errors="replace"):
        m = PROG.match(l)
        if m and m.group(3) == "answered" and m.group(4) == m.group(5) and int(m.group(4)) >= 2:
            votes[m.group(1)] = True
    sub = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(d, "submission.jsonl"), encoding="utf-8")}
    return {i: sub[i] for i in votes if i in sub and sub[i].get("type", "abstain") != "abstain"}, len(sub)


gold = {json.loads(l)["id"]: json.loads(l) for l in open(gold_f, encoding="utf-8")}
st, nt = strong(xt)
ff = [i for i, r in st.items() if not same_answer(r, gold[i])]
sp, npub = strong(xp)
base = [json.loads(l) for l in open(base_f, encoding="utf-8")]
bd = {r["id"]: r for r in base}
flag = sorted(i for i, r in sp.items() if bd[i].get("type", "abstain") != "abstain" and not same_answer(r, bd[i]))
active = len(ff) <= 1
rep = {"train_calib_n": nt, "train_strong": len(st), "train_false_flags": len(ff), "rule_active": active,
       "public_risk_n": npub, "public_strong": len(sp), "public_disagree_strong": len(flag)}
os.makedirs(out, exist_ok=True)
if active and flag:
    new = [{"id": r["id"], "type": "abstain"} if r["id"] in set(flag) else r for r in base]
    with open(os.path.join(out, "submission.jsonl"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in new))
    rep["submission_written"] = True
with open(os.path.join(out, "CROSSCHECK_REPORT.json"), "w") as fh:
    json.dump({**rep, "abstained_ids": flag if active else []}, fh, indent=1)
print(json.dumps(rep))
