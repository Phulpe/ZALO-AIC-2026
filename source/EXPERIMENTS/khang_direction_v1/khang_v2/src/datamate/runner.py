"""Chay toan bo bo cau hoi: /data -> /result/submission.jsonl + /result/logs/.

Dung:
  python -m datamate.runner --data /data --out /result            (che do nop bai, tham so mac dinh co dinh)
  python -m datamate.runner --data data/train --out runs/x --gold data/train/answers.jsonl --limit 50 ...  (danh gia)
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import random
import sys
import threading
import time
import traceback
from collections import Counter, defaultdict

from .agent import AgentConfig, SampleResult, run_sample, vote
from .answers import score
from .dates import LunarCalendar, date_context
from .hints import EntityIndex, question_hints
from .kernel import KernelPool, load_business_cached, warm_cache
from .llm import LLMClient
from .profile import business_profile
from .prompts import build_first_user, build_system


class ResultWriter:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.log_dir = os.path.join(out_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.sub_path = os.path.join(out_dir, "submission.jsonl")
        self.lock = threading.Lock()
        self.done: dict[str, dict] = {}
        open(self.sub_path, "w").close()

    def write(self, qid: str, answer: dict, log: dict):
        line = {"id": qid, **answer}
        with self.lock:
            with open(self.sub_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            with open(os.path.join(self.log_dir, f"{qid}.json"), "w", encoding="utf-8") as fh:
                json.dump(log, fh, ensure_ascii=False, default=str)
            self.done[qid] = line

    def finalize(self, order: list[str]):
        """Viet lai submission theo thu tu cau hoi; cau chua co dap an ghi abstain."""
        with self.lock:
            tmp = self.sub_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                for qid in order:
                    fh.write(json.dumps(self.done.get(qid, {"id": qid, "type": "abstain"}), ensure_ascii=False) + "\n")
            os.replace(tmp, self.sub_path)


def _read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data")
    ap.add_argument("--out", default="/result")
    ap.add_argument("--cache", default=os.environ.get("DATAMATE_CACHE", "/tmp/datamate_cache"))
    ap.add_argument("--base-urls", default=os.environ.get("DATAMATE_BASE_URLS", "http://127.0.0.1:8000"))
    ap.add_argument("--backend", default=os.environ.get("DATAMATE_BACKEND", "vllm"), choices=("vllm", "mlx"))
    ap.add_argument("--model", default=os.environ.get("DATAMATE_MODEL", "qwen3-14b"))
    ap.add_argument("--workers", type=int, default=48, help="so cau hoi chay dong thoi")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--min-votes", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--budget-seconds", type=float, default=5.4 * 3600, help="tinh tu luc runner bat dau")
    ap.add_argument("--deadline-epoch", type=float, default=0.0, help="moc dung tuyet doi (epoch giay), uu tien hon --budget-seconds")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample-seed", type=int, default=0, help="chon ngau nhien --limit cau theo seed nay (0: lay theo thu tu)")
    ap.add_argument("--ids", default="", help="file chua danh sach id can chay")
    ap.add_argument("--businesses", default="", help="danh sach business_id, phan cach dau phay")
    ap.add_argument("--gold", default="", help="answers.jsonl de cham diem tai cho")
    ap.add_argument("--kernel-workers", type=int, default=2)
    ap.add_argument("--llm-on-code-hint", action="store_true",
                    help="van goi model ca khi code goi y chay ra dap an (mac dinh: nop thang ket qua code goi y, khong goi model)")
    return ap.parse_args(argv)


def _prerun_code_hint(pool, biz_path: str, q: dict, hints: str, out: dict | None = None) -> str:
    """Chay thu dong 'Code gợi ý' trong mot phien sandbox rieng (khong phai phien cua model, khong nop dap an)
    va ghi ket qua vao goi y. Loi thi bo dong code gợi ý de model tu viet code.
    out (neu co) nhan out['answer'] = ket qua chay thu (dap an du phong khi model khong nop duoc / het gio)."""
    import re

    m = re.search(r"^Code gợi ý \([^)]*\): (.*)$", hints, re.M)
    if not m:
        return hints
    sid = f"hint-{q['id']}"
    try:
        k = pool.get(biz_path, sid)
        r = k.execute(sid, m.group(1), q["as_of_date"], soft_timeout=20.0)
        k.end(sid)
    except Exception as exc:  # sandbox loi: giu nguyen goi y
        print(f"[runner] chay thu code goi y loi {q['id']}: {exc}", flush=True)
        return hints
    if r.get("answer") and not r.get("error"):
        if out is not None:
            out["answer"] = r["answer"]
            out["code"] = m.group(1)
            out["output"] = r.get("stdout") or json.dumps(r["answer"], ensure_ascii=False)
        res = json.dumps(r["answer"], ensure_ascii=False)
        return hints + (f"\nKết quả chạy thử code gợi ý trong sandbox (đã chuẩn hoá §9): {res}. "
                        "Hợp lý với câu hỏi thì chạy lại đúng dòng code gợi ý để nộp; thấy sai thì kiểm tra dữ liệu rồi tự tính.")
    err = ((r.get("error") or "").strip().splitlines() or ["không nộp được đáp án"])[-1][:200]
    return hints.replace(m.group(0), "") + f"\n(Code gợi ý chạy thử bị lỗi: {err} -> tự viết code, không dùng dòng đó.)"


def main(argv=None) -> int:
    args = parse_args(argv)
    t_start = time.time()
    deadline = args.deadline_epoch if args.deadline_epoch > 0 else t_start + args.budget_seconds
    os.makedirs(args.out, exist_ok=True)
    questions = _read_jsonl(os.path.join(args.data, "questions.jsonl"))
    order = [q["id"] for q in questions]
    if args.businesses:
        keep = set(args.businesses.split(","))
        questions = [q for q in questions if q["business_id"] in keep]
    if args.ids:
        keep = {line.strip() for line in open(args.ids) if line.strip()}
        questions = [q for q in questions if q["id"] in keep]
    if args.limit:
        if args.sample_seed:
            rnd = random.Random(args.sample_seed)
            questions = rnd.sample(questions, min(args.limit, len(questions)))
        else:
            questions = questions[: args.limit]
    order_sel = [q["id"] for q in questions]
    final_order = order if not (args.limit or args.ids or args.businesses) else order_sel
    writer = ResultWriter(args.out)  # file rong ngay tu dau: cau thieu dong = abstain; moi dap an ghi ngay khi co
    rulebook = open(os.path.join(args.data, "rulebook.md"), encoding="utf-8").read()
    cal_path = os.path.join(args.data, "calendar.csv")
    lunar = LunarCalendar.load(cal_path) if os.path.exists(cal_path) else None
    data_root = os.path.join(args.data, "data")
    biz_ids = sorted({q["business_id"] for q in questions})
    biz_paths = {b: os.path.join(data_root, b) for b in biz_ids}
    print(f"[runner] {len(questions)} cau, {len(biz_ids)} cua hang", flush=True)

    # 1. nap + lam sach du lieu song song (cache pickle trong /tmp)
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=min(8, max(1, len(biz_ids))), mp_context=ctx) as ex:
        futs = {ex.submit(warm_cache, p, args.cache): b for b, p in biz_paths.items()}
        for fut in cf.as_completed(futs):
            try:
                fut.result()
            except Exception:
                print(f"[runner] LOI nap cua hang {futs[fut]}", flush=True)
                traceback.print_exc()
    profiles, data_end, entities = {}, {}, {}
    for b, p in biz_paths.items():
        try:
            biz = load_business_cached(p, args.cache)
            profiles[b] = business_profile(biz)
            data_end[b] = biz.data_end.isoformat() if biz.data_end else "không ghi"
            try:
                entities[b] = EntityIndex.from_business(biz)
            except Exception:
                traceback.print_exc()
            del biz
        except Exception:
            traceback.print_exc()
    questions = [q for q in questions if q["business_id"] in profiles]  # cua hang loi -> bo trong
    print(f"[runner] nap du lieu xong sau {time.time() - t0:.1f}s", flush=True)

    cfg = AgentConfig(
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        enable_thinking=not args.no_thinking,
        n_samples=args.samples,
        min_votes=args.min_votes,
        base_seed=args.seed,
    )
    system = build_system(rulebook, cfg.max_steps)
    llm = LLMClient(args.base_urls.split(","), args.model, backend=args.backend)
    try:
        llm.wait_ready(max_wait=max(60.0, min(1800.0, deadline - time.time())))
    except RuntimeError as exc:
        print(f"[runner] {exc}; nop toan bo abstain", flush=True)
        writer.finalize(final_order)
        return 1
    print(f"[runner] LLM san sang sau {time.time() - t_start:.1f}s", flush=True)
    pool = KernelPool(args.cache, cal_path if lunar else None, workers_per_business=args.kernel_workers)

    stats = {"lock": threading.Lock(), "sample_secs": [], "done": 0}

    def planned_samples() -> int:
        """Giam so lan chay moi cau khi du kien khong kip gio (cau chua lam tinh la bo trong)."""
        with stats["lock"]:
            secs = stats["sample_secs"][-200:]
            done = stats["done"]
        if len(secs) < 10:
            return cfg.n_samples
        avg = sum(secs) / len(secs)
        left_q = max(0, len(questions) - done)
        left_t = deadline - time.time()
        for k in range(cfg.n_samples, 0, -1):
            if left_q * k * avg / max(1, args.workers) <= 0.85 * left_t:
                return k
        return 1

    def solve(q: dict) -> tuple[str, dict]:
        qstart = time.time()
        b = q["business_id"]
        try:
            hints = question_hints(q["question"], q["as_of_date"], lunar, entities.get(b))
        except Exception:
            traceback.print_exc()
            hints = ""
        pre: dict = {}
        hints = _prerun_code_hint(pool, biz_paths[b], q, hints, out=pre)
        if pre.get("answer") and not args.llm_on_code_hint:
            # Code gợi ý (luật tất định ghép từ kỳ, bộ lọc và hàm dựng sẵn; train 2949/2949 đúng) đã chạy ra đáp án:
            # nộp thẳng, không gọi model. Trên public, model luôn nộp lại đúng kết quả này (587/587 câu có code gợi ý).
            final = dict(pre["answer"])
            with stats["lock"]:
                stats["done"] += 1
            writer.write(q["id"], final, {
                "id": q["id"], "answer": final, "elapsed_ms": int((time.time() - qstart) * 1000),
                "thinking": None,  # khong goi model (muc 3.3 huong dan BTC)
                "code": [pre["code"]], "output": [pre["output"]], "source": "code_hint",
                "hints": hints, "question": q["question"], "business_id": b, "as_of_date": q["as_of_date"],
            })
            return q["id"], final
        first_user = build_first_user(
            profiles[b], q["id"], q["as_of_date"], data_end[b], date_context(q["as_of_date"], lunar), q["question"], hints
        )
        samples: list[SampleResult] = []
        qseed = cfg.base_seed + (int.from_bytes(q["id"].encode()[-6:], "big") % 100000) * 10
        n_plan = planned_samples()
        min_votes = min(cfg.min_votes, n_plan)
        for k in range(n_plan):
            if time.time() > deadline:
                break
            kernel = pool.get(biz_paths[b], f"{q['id']}#{k}")
            samples.append(run_sample(llm, kernel, system, first_user, q["as_of_date"], cfg, qseed + 1000 * k, affinity=b, deadline=deadline))
            # dung som khi cac lan chay con lai khong the doi ket qua bieu quyet
            with stats["lock"]:
                stats["sample_secs"].append(samples[-1].seconds)
            _, info = vote(samples, min_votes)
            remaining = n_plan - len(samples)
            counts = sorted(info["votes"].values(), reverse=True) + [0, 0]
            top, second = counts[0], counts[1]
            if (top >= min_votes and top - second > remaining) or (top + remaining < min_votes):
                break
        final, info = vote(samples, min_votes)
        info["n_planned"] = n_plan
        pa = pre.get("answer")
        if final.get("type") == "abstain" and pa:
            # model khong nop duoc (het buoc / het gio): dung ket qua code goi y tat dinh (train: 1777/1777 dung khi co code goi y)
            final = dict(pa)
            info["fallback"] = "code_hint"
        elif final.get("type") == "unanswerable" and pa and pa.get("type") not in ("unanswerable", "abstain"):
            # code goi y chay that tren du lieu ra gia tri ma model lai bao khong tra loi duoc (public: model tu viet lai ham, ra None)
            final = dict(pa)
            info["fallback"] = "code_hint_over_unanswerable"
        with stats["lock"]:
            stats["done"] += 1
        log = {
            "id": q["id"],
            "answer": final,
            "elapsed_ms": int((time.time() - qstart) * 1000),
            # moi loi goi model mot phan tu (dung thu tu goi); khong co suy nghi nao -> null (muc 3.3 huong dan BTC)
            "thinking": (lambda th: th if any(t for t in th) else None)([t or "" for s in samples for t in s.thinking]),
            "code": [st["code"] for s in samples for st in s.steps if st.get("code")] or None,
            "output": [st["output"] for s in samples for st in s.steps if st.get("code")] or None,
            "vote": info,
            "samples": [
                {
                    "answer": s.answer,
                    "error": s.error,
                    "seconds": s.seconds,
                    "prompt_tokens": s.prompt_tokens,
                    "completion_tokens": s.completion_tokens,
                    "steps": s.steps,
                }
                for s in samples
            ],
            "question": q["question"],
            "business_id": b,
            "as_of_date": q["as_of_date"],
        }
        writer.write(q["id"], final, log)
        return q["id"], final

    results: dict[str, dict] = {}
    # xen ke cua hang de cac worker sandbox khong bi don mot cho
    by_biz = defaultdict(list)
    for q in questions:
        by_biz[q["business_id"]].append(q)
    queue = []
    while any(by_biz.values()):
        for b in list(by_biz):
            if by_biz[b]:
                queue.append(by_biz[b].pop(0))
    n_done = 0
    try:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(solve, q): q["id"] for q in queue}
            for fut in cf.as_completed(futs):
                qid = futs[fut]
                try:
                    _, ans = fut.result()
                    results[qid] = ans
                except Exception:
                    traceback.print_exc()
                    results[qid] = {"type": "abstain"}
                n_done += 1
                if n_done % 10 == 0 or n_done == len(queue):
                    print(f"[runner] {n_done}/{len(queue)} cau, {time.time() - t_start:.0f}s", flush=True)
    finally:
        pool.close()
        writer.finalize(final_order)
    print(f"[runner] xong {len(results)} cau sau {time.time() - t_start:.1f}s", flush=True)

    if args.gold:
        evaluate(args.gold, results, questions, os.path.join(args.out, "eval.json"))
    return 0


def evaluate(gold_path: str, results: dict, questions: list[dict], out_path: str):
    gold = {g["id"]: g for g in _read_jsonl(gold_path)}
    per_type = defaultdict(Counter)
    total = Counter()
    rows = []
    for q in questions:
        g = gold.get(q["id"])
        if g is None:
            continue
        p = results.get(q["id"])
        s = score(p, g)
        tag = {1: "dung", -1: "sai", 0: "bo"}[s]
        total[tag] += 1
        per_type[g["type"]][tag] += 1
        rows.append({"id": q["id"], "business_id": q["business_id"], "gold_type": g["type"], "score": s, "pred": p, "gold": g.get("answer"), "question": q["question"]})
    n = sum(total.values())
    pa = (total["dung"] - total["sai"]) / n if n else 0.0
    summary = {
        "n": n,
        "dung": total["dung"],
        "sai": total["sai"],
        "bo_trong": total["bo"],
        "PA": round(pa, 4),
        "do_chinh_xac_khi_tra_loi": round(total["dung"] / max(1, total["dung"] + total["sai"]), 4),
        "theo_type": {t: dict(c) for t, c in per_type.items()},
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "rows": rows}, fh, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.exit(main())
