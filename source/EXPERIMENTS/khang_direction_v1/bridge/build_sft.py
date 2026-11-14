"""GD3 — dung du lieu SFT tu quy dao (rejection sampling) tren learning pool TRAIN.

Vao: TRAJ_LOG (moi dong = 1 loi goi LLM: messages truoc loi goi + reply), thu muc run cua Khang (logs/<qid>.json),
gold learning (chi TRAIN). Giu quy dao cuoi cua moi sample co dap an dung theo answers.score cua Khang.
Ra: sft.jsonl {qid, k, messages(day du, ket thuc bang assistant)} + stats (chi so tong hop, khong noi dung).

Ghep quy dao: agent goi llm.chat(seed = sample_seed + step), step = so assistant da co + 1.
sample_seed = qseed + 1000*k, qseed = base_seed + (int(qid[-6:] bytes) % 100000) * 10.
"""
import json
import os
import re
import sys
from collections import Counter

QID = re.compile(r"^- id: (\S+)$", re.M)


def qseed(qid: str, base_seed: int) -> int:
    return base_seed + (int.from_bytes(qid.encode()[-6:], "big") % 100000) * 10


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def final_trajectories(traj_path, base_seed):
    """{(qid, k): messages_day_du} — ban dai nhat cua moi sample."""
    best = {}
    for r in read_jsonl(traj_path):
        msgs = r["messages"]
        m = QID.search(msgs[1]["content"]) if len(msgs) > 1 else None
        if not m or r.get("seed") is None:
            continue
        qid = m.group(1)
        step = sum(1 for x in msgs if x["role"] == "assistant") + 1
        k, rem = divmod(r["seed"] - step - qseed(qid, base_seed), 1000)
        if rem or k < 0:
            continue
        key = (qid, k)
        if r["reply"].get("finish_reason") == "length" or best.get(key, []) is None:
            best[key] = None  # buoc nao bi cat thi loai ca sample
            continue
        full = msgs + [{"role": "assistant", "content": r["reply"]["content"]}]
        if len(full) > len(best.get(key) or []):
            best[key] = full
    return best


def build(traj_path, run_dir, gold_path, out_path, base_seed=1234, score_fn=None):
    if score_fn is None:
        from datamate.answers import score as score_fn
    gold = {g["id"]: g for g in read_jsonl(gold_path)}
    trajs = final_trajectories(traj_path, base_seed)
    st = Counter()
    rows = []
    for (qid, k), msgs in sorted(trajs.items()):
        st["samples"] += 1
        if msgs is None:
            st["truncated"] += 1
            continue
        lp = os.path.join(run_dir, "logs", f"{qid}.json")
        if qid not in gold or not os.path.exists(lp):
            st["no_gold_or_log"] += 1
            continue
        log = json.load(open(lp, encoding="utf-8"))
        samples = log.get("samples") or []
        if k >= len(samples):
            st["no_sample"] += 1
            continue
        s = samples[k]
        last = [x for x in s.get("steps", []) if x.get("code")]
        if s.get("error") or s.get("answer") is None or not last:
            st["no_answer"] += 1
            continue
        sc = score_fn(s["answer"], gold[qid])
        st[{1: "correct", -1: "wrong", 0: "abstain"}[sc]] += 1
        if sc == 1:
            rows.append({"qid": qid, "k": k, "messages": msgs})
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, out_path)
    st["kept"] = len(rows)
    st["kept_questions"] = len({r["qid"] for r in rows})
    return dict(st)


if __name__ == "__main__":
    sys.path.insert(0, os.environ.get("KHANG_SRC", os.path.join(os.path.dirname(__file__), "..", "khang", "src")))
    print(json.dumps(build(*sys.argv[1:5], base_seed=int(os.environ.get("BASE_SEED", "1234")))))
