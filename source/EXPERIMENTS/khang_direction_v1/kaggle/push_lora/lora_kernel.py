"""ZAIC khang_direction_v1 — PUBLIC_TEST (plan vllm_public): Tier1 hint Khang + vLLM Qwen3.5-9B cho cau khong hint; khong gold, khong tu nop.
Cac plan khac giu nguyen tu gd3_kernel.py.

GD3 tren Kaggle (mot phien GPU, tan dung toi da khi den luot).

Pha A  thu quy dao: Qwen3.5-9B base (khong adapter) chay runner Khang --hints-off tren learning pool TRAIN.
        Runner KHONG nhan gold. Moi loi goi LLM ghi vao trajectories_learning.jsonl.
Pha B  dung SFT: build_sft.py cham bang gold learning (chi TRAIN) va giu quy dao cuoi cua sample dung.
Pha C  train LoRA fresh (init_mode=fresh, khong dung adapter cu) — train_lora.py, han gio cung.
Pha D  do lai: server nap base + adapter moi, chay 120 cau calibration da dong bang (hints-off, khong gold).
Moi pha co han gio rieng; het gio thi luu va sang pha sau. Khong public_test. Ket qua: /kaggle/working/out.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

CONFIG = {
    "mode": "gd3",
    "run_id": "khang_dir_v3_fewshot_confirm_20261112",
    "init_mode": "fresh",
    "runtime_manifest_sha256": "523db34eb8e8bab52aa52d1d046307f19dbb3668495025ca05d66abd1e4c6e70",
    "code_manifest_sha256": "2597b7456de9507392930e07c8de18c2ccfba99f12d754e2d416aa517db81470",
    "base_small_hashes": {"LICENSE": "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a", "README.md": "c5f5a8c2dddab69cfbf05279235aa5fddb137939a06539c4c7637aa900fef6d0", "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715", "config.json": "d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05", "merges.txt": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d", "model.safetensors.index.json": "26d3539b516be613f39563617cb9d33b3f83d401298125be392c80cefb8f7fe5", "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516", "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42", "tokenizer_config.json": "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8", "video_preprocessor_config.json": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13", "vocab.json": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"},
    "eval_ids_file": "eval_ids_120.txt",
    "eval_ids_sha256": "f337f7f88f71544dc942d1aeb23d98243b964ea1faacfb092c076d399109e26d",
    "smoke_ids_sha256": "64ee76bf459213d9552cb5118d7bd56bf5584f7db99991fc430304c7a4e5fd5c",
    "max_error_rate": 0.2,
    "max_servers": 2,
    "plan": "fewshot_eval",
    "bank_sha256": "f74eb15e44e2995fd1131d096f573cd50e59fb2807dbc606f0c2fe7ca1cb630e",
    "fewshot_k": 3,
    "fewshot_eval_args_key": "v2_runner_args",  # v8: cau hinh public v2 (fixed-samples, 16 workers) + few-shot
    "fewshot_run_name": "eval38_fewshot_v2cfg",  # lora_preflight (CPU) | lora (GPU) | v2_bench | v2_public
    "sft_sha256": "bd4b8bda093673a0276a4920202f0ad65f587c8d2c270b87b45e0c859688087d",
    "train_seconds": 9000,
    "reserve_after_train": 7800,
    "lora_train_args": ["--maxlen", "16384", "--per-question", "1", "--epochs", "1", "--grad-accum", "8", "--rank", "16",
                   "--lr", "1e-4", "--target", "all-linear", "--scheduler", "constant_with_warmup", "--warmup", "3",
                   "--save-steps", "5", "--min-items", "16", "--multi-gpu", "--gpu-mem-gib", "11",
                        "--lora-dropout", "0"],
    "eval_v5_args": ["--samples", "3", "--min-votes", "2", "--no-thinking", "--temperature", "0.6", "--max-tokens", "2048",
                     "--max-steps", "6", "--workers", "8", "--kernel-workers", "2"],
    "public_questions_sha256": "b33ae585ed5d4acdacbd6a2873d442e3a5673a8238b6e932827f40c48c158832",
    "public_calendar_sha256": "c2579905e8b4eb152a4833b05f1f72de3e3d2c517b5d610d7664aeeb2d6a75b3",
    "public_probe_n": 40,
    "vllm_wheel_glob": "vllm-0.30.0-*.whl",
    "khang_src": "khang_v3/src",
    "vllm_pip": "vllm==0.30.0",
    "vllm_args": ["--dtype", "float16", "--max-model-len", "32768", "--gpu-memory-utilization", "0.90",
                  "--max-num-seqs", "16", "--enable-prefix-caching", "--seed", "0"],
    "spec_config": None,  # bench v1: MTP tren T4 chi 2 request song song -> tat (chi thu tren 5090)
    "v2_runner_args": ["--samples", "3", "--min-votes", "2", "--no-thinking", "--temperature", "0.6", "--max-tokens", "2048",
                       "--max-steps", "6", "--workers", "16", "--kernel-workers", "4", "--fixed-samples", "--parallel-first", "1",
                       "--question-timeout", "0", "--progress-every", "600"],
    "vllm_ready_seconds": 2400,
    "nohint_ids_file": "nohint_ids.txt",
    "nohint_ids_sha256": "854cecebeedc89a252daa0d6efab81777e848a3195acf77ab50f0e663ed1c368",
    "probe_n": 4,
    "nohint_runner_args": ["--samples", "3", "--min-votes", "2", "--no-thinking", "--temperature", "0.6",
                           "--max-tokens", "2048", "--max-steps", "6", "--workers", "16", "--kernel-workers", "4"],
    "learning_limit": 0,           # 0 = ca learning pool
    "learning_sample_seed": 20261110,
    "collect_runner_args": ["--samples", "1", "--min-votes", "1", "--no-thinking", "--temperature", "0.6",
                            "--max-tokens", "4096", "--max-steps", "6", "--workers", "8", "--kernel-workers", "2"],
    "eval_runner_args": ["--samples", "1", "--min-votes", "1", "--no-thinking", "--max-tokens", "4096",
                         "--max-steps", "6", "--workers", "8", "--kernel-workers", "2"],
    "server_args": ["--max-batch", "8", "--wait-ms", "80", "--max-new-cap", "4096"],
    "train_args": ["--maxlen", "6144", "--per-question", "2", "--lr", "1e-4", "--epochs", "1", "--grad-accum", "16"],
    "phase_seconds": {"collect": 14400, "train": 9000, "eval": 6600, "public": 9000, "bench": 4200},
    "max_gpu_seconds": 8700,  # fewshot confirm: ~2,4h (quota con 3,18h)
    "preflight_cpu": False,
    "code_manifest_override": None,
    "public_test": False,  # chi chay sau khi nguoi dung duyet; khong gold, khong tu nop
}

T0 = time.monotonic()
OUT = Path("/kaggle/working/out")
OUT.mkdir(parents=True, exist_ok=True)
STATUS = {"config": CONFIG, "steps": []}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def save_status(**kw):
    STATUS.update(kw, elapsed_s=round(time.monotonic() - T0, 1))
    tmp = OUT / "STATUS.json.tmp"
    tmp.write_text(json.dumps(STATUS, ensure_ascii=False, indent=1))
    os.replace(tmp, OUT / "STATUS.json")


def step(name, **kw):
    STATUS["steps"].append(dict(name=name, t=round(time.monotonic() - T0, 1), **kw))
    save_status()
    print("STEP", name, json.dumps(kw, ensure_ascii=False)[:400], flush=True)


def one(pattern):
    hits = list(Path("/kaggle/input").rglob(pattern))
    assert len(hits) == 1, f"{pattern}: {len(hits)} matches"
    return hits[0]


def finish(state):
    manifest = {str(p.relative_to(OUT)): {"bytes": p.stat().st_size, "sha256": sha(p)}
                for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "ARTIFACT_MANIFEST.json"}
    (OUT / "ARTIFACT_MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    save_status(state=state)


assert CONFIG["init_mode"] == "fresh" and (not CONFIG["public_test"] or CONFIG["plan"] in ("vllm_public", "v2_public"))
HARD_END = T0 + CONFIG["max_gpu_seconds"]

# 1. Input
code_man = one("KHANG_DIRECTION_CODE_MANIFEST.json")
code_root = code_man.parent
if CONFIG["code_manifest_sha256"]:
    assert sha(code_man) == CONFIG["code_manifest_sha256"]
for rel, d in json.loads(code_man.read_text())["files"].items():
    assert sha(code_root / rel) == d, rel
runtime_man = one("RUNTIME_DATA_MANIFEST.json")
assert sha(runtime_man) == CONFIG["runtime_manifest_sha256"]
runtime = runtime_man.parent
rfiles = json.loads(runtime_man.read_text())["files"]
for rel in ("learning/questions.jsonl", "learning/gold.jsonl", "calibration/questions.jsonl"):
    exp = rfiles[rel] if isinstance(rfiles[rel], str) else rfiles[rel].get("sha256")
    assert sha(runtime / rel) == exp, rel
step("inputs_verified", code_manifest=sha(code_man))


def read_q(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8-sig").splitlines() if l.strip()]


def build_dm(name, subset):
    dm = Path("/kaggle/working/dm") / name
    if dm.exists():
        shutil.rmtree(dm)
    (dm / "data").mkdir(parents=True)
    for n in ("calendar.csv", "rulebook.md"):
        os.symlink(runtime / n, dm / n)
    for b in sorted({q["business_id"] for q in subset}):
        os.symlink(runtime / "data" / b, dm / "data" / b)
    (dm / "questions.jsonl").write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in subset), encoding="utf-8")
    return dm


learn = read_q(runtime / "learning/questions.jsonl")
cal = read_q(runtime / "calibration/questions.jsonl")
assert not ({q["business_id"] for q in learn} & {q["business_id"] for q in cal}), "split leak"
if CONFIG["learning_limit"]:
    import random
    learn = sorted(learn, key=lambda q: q["id"])
    random.Random(CONFIG["learning_sample_seed"]).shuffle(learn)
    learn = learn[: CONFIG["learning_limit"]]
ids_f = code_root / CONFIG["eval_ids_file"]
assert sha(ids_f) == CONFIG["eval_ids_sha256"]
keep = set(ids_f.read_text().split())
ev = [q for q in cal if q["id"] in keep]
assert len(ev) == len(keep)
sm_f = code_root / "eval_ids.txt"
assert sha(sm_f) == CONFIG["smoke_ids_sha256"]
sm_ids = set(sm_f.read_text().split())
assert sm_ids <= keep
dm_learn, dm_eval = build_dm("learning", learn), build_dm("eval120", ev)
dm_smoke = build_dm("smoke10", [q for q in ev if q["id"] in sm_ids])
dm_eval_rest = build_dm("eval120_base", [q for q in ev if q["id"] not in sm_ids])
(OUT / "learning_ids.txt").write_text("\n".join(sorted(q["id"] for q in learn)) + "\n")
step("data_layout", learning=len(learn), eval=len(ev), smoke=len(sm_ids))

env = dict(os.environ, PYTHONIOENCODING="utf-8", KHANG_SRC=str(code_root / CONFIG.get("khang_src", "khang/src")))

# 2. GPU
import torch  # noqa: E402
if CONFIG.get("plan") == "lora_preflight":
    # CPU: kiem merge_lora (test gia lap), so token that cua SFT voi tokenizer base, phien ban thu vien. Khong GPU.
    res = {}
    r = subprocess.run([sys.executable, str(code_root / "bridge_v3/test_merge_lora.py")], capture_output=True, text=True, timeout=600)
    res["test_merge_rc"], res["test_merge_tail"] = r.returncode, (r.stdout + r.stderr)[-400:]
    sft_man = one("TEACHER_SFT_MANIFEST.json")
    sft = sft_man.parent / "sft.jsonl"
    res["sft_sha_ok"] = sha(sft) == CONFIG["sft_sha256"]
    b = one("BASE_UPLOAD_MANIFEST.json").parent / "base"
    from transformers import AutoTokenizer
    import transformers, peft
    tok = AutoTokenizer.from_pretrained(str(b), local_files_only=True)
    lens, tl = [], []
    for l in open(sft, encoding="utf-8"):
        m = json.loads(l)["messages"]
        pre = tok.apply_chat_template(m[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        lens.append(len(tok(pre, add_special_tokens=False)["input_ids"]))
        tl.append(len(tok(m[-1]["content"], add_special_tokens=False)["input_ids"]))
    lens.sort()
    q = lambda p: lens[min(len(lens) - 1, int(p * len(lens)))]
    res.update(n=len(lens), prompt_tok_p50=q(0.5), p90=q(0.9), max=lens[-1], target_tok_mean=round(sum(tl) / len(tl), 1),
               transformers=transformers.__version__, peft=peft.__version__, torch=torch.__version__)
    step("lora_preflight", **res)
    finish("lora_preflight_ok" if res["test_merge_rc"] == 0 and res["sft_sha_ok"] else "lora_preflight_failed")
    sys.exit(0)

assert CONFIG.get("preflight_cpu") or torch.cuda.is_available(), "GPU mode without GPU"
if CONFIG.get("preflight_cpu"):
    # CPU, khong GPU: kiem input public + hash + cai vLLM tu wheel offline; khong chay model.
    qf = one("INPUT_811/PUBLIC/questions.jsonl")
    ok_q = sha(qf) == CONFIG["public_questions_sha256"]
    ok_c = sha(qf.parent / "calendar.csv") == CONFIG["public_calendar_sha256"]
    pq = read_q(qf)
    whl = sorted(Path("/kaggle/input").glob("**/" + CONFIG["vllm_wheel_glob"]))
    t = time.monotonic()
    with open(OUT / "pip_vllm.log", "w") as lg:
        rc = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", str(whl[0].parent),
                             CONFIG["vllm_pip"]], stdout=lg, stderr=subprocess.STDOUT, timeout=2400).returncode if whl else -1
    ver = subprocess.run([sys.executable, "-c", "import vllm,torch,sys;print(vllm.__version__, torch.__version__, sys.version.split()[0])"],
                         capture_output=True, text=True).stdout.strip()
    b = one("BASE_UPLOAD_MANIFEST.json").parent / "base"
    ok_b = all(sha(b / n) == d for n, d in CONFIG["base_small_hashes"].items())
    step("preflight_cpu", public_q_ok=ok_q, calendar_ok=ok_c, n_public=len(pq), wheel_dir=str(whl[0].parent) if whl else None,
         n_wheels=len(list(whl[0].parent.glob("*.whl"))) if whl else 0, pip_rc=rc, version=ver,
         pip_seconds=round(time.monotonic() - t, 1), base_small_ok=ok_b)
    finish("preflight_ok" if (ok_q and ok_c and ok_b and rc == 0 and ver.startswith("0.30.0")) else "preflight_failed")
    sys.exit(0)

base = one("BASE_UPLOAD_MANIFEST.json").parent / "base"
for name, d in CONFIG["base_small_hashes"].items():
    assert sha(base / name) == d, name
exec(compile((code_root / "kaggle/OFFLINE_BNB_BOOTSTRAP.py").read_text(), "bnb_bootstrap", "exec"))
env["PYTHONPATH"] = os.environ.get("PYTHONPATH", "")
chk = subprocess.run([sys.executable, "-c", "import bitsandbytes as b; print(b.__version__)"], env=env,
                     capture_output=True, text=True, timeout=300)
assert chk.returncode == 0 and chk.stdout.strip() == "0.50.2", chk.stderr[-500:]
step("gpu_ready", gpu=torch.cuda.get_device_name(0), count=torch.cuda.device_count(), bnb=chk.stdout.strip())


def prefetch_base(src):
    from concurrent.futures import ThreadPoolExecutor
    files = [p for p in src.iterdir() if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    dst = None
    for d in ("/kaggle/temp", "/tmp"):
        try:
            Path(d).mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(d).free > total + 4_000_000_000:
                dst = Path(d) / "base_local"
                break
        except Exception:
            continue
    t = time.monotonic()

    def cp(p):
        if dst is None:
            with open(p, "rb") as f:
                while f.read(64 << 20):
                    pass
            return p.stat().st_size
        tmp = dst / (p.name + ".part")
        with open(p, "rb") as fi, open(tmp, "wb") as fo:
            shutil.copyfileobj(fi, fo, 64 << 20)
        os.replace(tmp, dst / p.name)
        return p.stat().st_size

    if dst is not None:
        dst.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(8) as ex:
        n = sum(ex.map(cp, sorted(files, key=lambda p: -p.stat().st_size)))
    secs = time.monotonic() - t
    if dst is not None:
        for p in files:
            assert (dst / p.name).stat().st_size == p.stat().st_size, p.name
        for name, d in CONFIG["base_small_hashes"].items():
            assert sha(dst / name) == d, name
    step("base_prefetch", target=str(dst) if dst else "page_cache_only", gb=round(n / 1e9, 2),
         seconds=round(secs, 1), mb_per_s=round(n / 1e6 / max(secs, 1e-3), 1))
    return dst or src


base = prefetch_base(base)


N_GPU = min(torch.cuda.device_count(), CONFIG.get("max_servers", 2))


def start_server(tag, adapter=None):
    """Moi GPU mot server doc lap (ban sao model rieng, khong gop VRAM). Nap tuan tu de han che dinh RAM."""
    procs = []
    for g in range(N_GPU):
        log = open(OUT / f"server_{tag}_gpu{g}.log", "w")
        args = [sys.executable, str(code_root / "bridge/hf_chat_server.py"), "--base", str(base), "--port", str(8000 + g),
                *CONFIG["server_args"]] + (["--adapter", str(adapter)] if adapter else [])
        p = subprocess.Popen(args, env=dict(env, CUDA_VISIBLE_DEVICES=str(g), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"),
                             stdout=log, stderr=subprocess.STDOUT)
        t = time.monotonic()
        ok = False
        while time.monotonic() < HARD_END and p.poll() is None:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{8000 + g}/v1/models", timeout=5)
                ok = True
                break
            except Exception:
                time.sleep(5)
        step("server", tag=tag, gpu=g, ready=ok, load_seconds=round(time.monotonic() - t, 1), rc=p.poll())
        if ok:
            procs.append((g, p, log))
        else:
            log.close()
            if g == 0:
                break
    return procs


def stop_server(procs):
    for g, p, log in procs:
        try:
            stats = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{8000 + g}/stats", timeout=10).read())
        except Exception as e:
            stats = {"error": str(e)}
        p.terminate()
        try:
            p.wait(60)
        except Exception:
            p.kill()
        log.close()
        step("server_stopped", gpu=g, stats=stats)


def urls(procs):
    return ",".join(f"http://127.0.0.1:{8000 + g}" for g, _, _ in procs)


def error_rate(name):
    n = bad = 0
    for f in (OUT / "run" / name / "logs").glob("*.json"):
        for s in json.loads(f.read_text(encoding="utf-8")).get("samples") or []:
            n += 1
            bad += bool(s.get("error"))
    return n, bad


def run_runner(name, dm, seconds, extra, procs, hints_off=True, base_urls=None):
    left = int(min(seconds, HARD_END - time.monotonic() - 600))
    if left < 300:
        step("runner_skipped", stage=name, seconds_left=left)
        return None
    t = time.monotonic()
    args = [sys.executable, str(code_root / "bridge/run_khang.py"), "--data", str(dm), "--out", str(OUT / "run" / name),
            "--cache", "/kaggle/temp/dm_cache", "--model", "qwen3.5-9b", "--budget-seconds", str(left),
            "--base-urls", base_urls or urls(procs), *extra] + (["--hints-off"] if hints_off else [])
    with open(OUT / f"runner_{name}.log", "w") as lg:
        try:
            rc = subprocess.run(args, env=dict(env, TRAJ_LOG=str(OUT / f"trajectories_{name}.jsonl")), stdout=lg,
                                stderr=subprocess.STDOUT, timeout=left + 600).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    sub = OUT / "run" / name / "submission.jsonl"
    n_ans = sum(1 for l in open(sub) if l.strip() and json.loads(l).get("type") != "abstain") if sub.exists() else 0
    sec = round(time.monotonic() - t, 1)
    n_s, n_err = error_rate(name)
    step("runner", stage=name, rc=rc, seconds=sec, answered=n_ans, samples=n_s, sample_errors=n_err,
         logs=len(list((OUT / "run" / name / "logs").glob("*.json"))))
    return rc


def start_vllm():
    """Mot server vLLM tensor-parallel tren moi GPU (2xT4: fp16, khong bf16). Cai vLLM luc chay (can internet cho pip)."""
    t = time.monotonic()
    whl = sorted(Path("/kaggle/input").glob("**/" + CONFIG.get("vllm_wheel_glob", "none")))
    mode = "online"
    with open(OUT / "pip_vllm.log", "w") as lg:
        rc = 1
        if whl:
            mode = "offline"
            rc = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", str(whl[0].parent),
                                 CONFIG["vllm_pip"]], stdout=lg, stderr=subprocess.STDOUT, timeout=2400).returncode
        if rc != 0:
            mode = "online_fallback" if whl else "online"
            rc = subprocess.run([sys.executable, "-m", "pip", "install", "-q", CONFIG["vllm_pip"]], stdout=lg,
                                stderr=subprocess.STDOUT, timeout=2400).returncode
    ver = subprocess.run([sys.executable, "-c", "import vllm,torch;print(vllm.__version__, torch.__version__)"],
                         capture_output=True, text=True).stdout.strip()
    step("vllm_install", rc=rc, mode=mode, version=ver, seconds=round(time.monotonic() - t, 1))
    if rc != 0:
        return []
    hlp = subprocess.run([sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--help"], capture_output=True,
                         text=True, timeout=600).stdout
    bad = [x for x in CONFIG["vllm_args"] if x.startswith("--") and x not in hlp]
    step("vllm_args_check", unknown=bad)
    if bad:
        return []
    log = open(OUT / "server_vllm.log", "w")
    args = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", str(base),
            "--served-model-name", "qwen3.5-9b", "--host", "127.0.0.1", "--port", "8000",
            "--tensor-parallel-size", str(N_GPU), *CONFIG["vllm_args"]]
    p = subprocess.Popen(args, env=dict(env, VLLM_NO_USAGE_STATS="1", NCCL_P2P_DISABLE="1"), stdout=log, stderr=subprocess.STDOUT)
    t = time.monotonic()
    ok = False
    while time.monotonic() < HARD_END and time.monotonic() - t < CONFIG.get("vllm_ready_seconds", 2400) and p.poll() is None:
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=5)
            ok = True
            break
        except Exception:
            time.sleep(10)
    step("server", tag="vllm", ready=ok, load_seconds=round(time.monotonic() - t, 1), rc=p.poll())
    if not ok:
        p.terminate(); log.close()
        return []
    return [(0, p, log)]


def stop_vllm(procs):
    for g, p, log in procs:
        try:
            m = urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=10).read().decode()
            keep = [l for l in m.splitlines() if l.startswith(("vllm:prompt_tokens_total", "vllm:generation_tokens_total",
                                                               "vllm:prefix_cache_hits", "vllm:prefix_cache_queries",
                                                               "vllm:request_success_total", "vllm:num_preemptions"))]
        except Exception as e:
            keep = [f"error {e}"]
        p.terminate()
        try:
            p.wait(60)
        except Exception:
            p.kill()
        log.close()
        step("server_stopped", tag="vllm", metrics=keep)


# ===================== v2 (11/11/2026): hint chong voi nap vLLM, log tien do truc tiep, MTP thu + du phong =====================
import threading as _th


def install_vllm():
    t = time.monotonic()
    whl = sorted(Path("/kaggle/input").glob("**/" + CONFIG.get("vllm_wheel_glob", "none")))
    mode, rc = "none", 1
    with open(OUT / "pip_vllm.log", "w") as lg:
        if whl:
            mode = "offline"
            rc = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", str(whl[0].parent),
                                 CONFIG["vllm_pip"]], stdout=lg, stderr=subprocess.STDOUT, timeout=2400).returncode
    ver = subprocess.run([sys.executable, "-c", "import vllm,torch;print(vllm.__version__, torch.__version__)"],
                         capture_output=True, text=True).stdout.strip()
    step("vllm_install", rc=rc, mode=mode, version=ver, seconds=round(time.monotonic() - t, 1))
    hlp = ""
    if rc == 0:
        for flag in ("--help=all", "--help"):
            hlp = subprocess.run([sys.executable, "-m", "vllm.entrypoints.openai.api_server", flag], capture_output=True,
                                 text=True, timeout=600).stdout
            if "--max-num-seqs" in hlp:
                break
    bad = [x for x in CONFIG["vllm_args"] if x.startswith("--") and x not in hlp]
    step("vllm_args_check", unknown=bad, speculative_flag="--speculative-config" in hlp)
    return rc == 0 and not bad, "--speculative-config" in hlp


def launch_vllm(spec):
    tag = "vllm_mtp" if spec else "vllm"
    log = open(OUT / f"server_{tag}.log", "w")
    args = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", str(base),
            "--served-model-name", "qwen3.5-9b", "--host", "127.0.0.1", "--port", "8000",
            "--tensor-parallel-size", str(N_GPU), *CONFIG["vllm_args"]] + (["--speculative-config", spec] if spec else [])
    p = subprocess.Popen(args, env=dict(env, VLLM_NO_USAGE_STATS="1", NCCL_P2P_DISABLE="1"), stdout=log, stderr=subprocess.STDOUT)
    return {"p": p, "log": log, "tag": tag, "t": time.monotonic(), "spec": spec}


def wait_vllm(h):
    ok = False
    while time.monotonic() < HARD_END and time.monotonic() - h["t"] < CONFIG.get("vllm_ready_seconds", 2400) and h["p"].poll() is None:
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=5)
            ok = True
            break
        except Exception:
            time.sleep(10)
    step("server", tag=h["tag"], ready=ok, load_seconds=round(time.monotonic() - h["t"], 1), rc=h["p"].poll())
    if not ok:
        h["p"].terminate()
        try:
            h["p"].wait(60)
        except Exception:
            h["p"].kill()
        h["log"].close()
        tail = (OUT / f"server_{h['tag']}.log").read_text(errors="replace").splitlines()[-15:]
        step("server_fail_tail", tag=h["tag"], tail=[x[-300:] for x in tail])
    return ok


def run_live(name, dm, seconds, extra, base_urls, ids_file=None):
    """Runner ghi file log va dong thoi in dong PROG/SUMMARY ra stdout (tab Logs Kaggle) de theo doi truc tiep."""
    left = int(min(seconds, HARD_END - time.monotonic() - 600))
    if left < 300:
        step("runner_skipped", stage=name, seconds_left=left)
        return None
    t = time.monotonic()
    args = [sys.executable, "-u", str(code_root / "bridge/run_khang.py"), "--data", str(dm), "--out", str(OUT / "run" / name),
            "--cache", "/kaggle/temp/dm_cache", "--model", "qwen3.5-9b", "--budget-seconds", str(left),
            "--base-urls", base_urls, *extra] + (["--ids", str(ids_file)] if ids_file else [])
    print(f"=== STAGE {name} bat dau (han {left}s) ===", flush=True)
    stop_mon = _th.Event()

    def _mon():
        while not stop_mon.wait(600):
            if "8000" in base_urls:
                m = vllm_metrics()
                print(f"[{name}] VLLM | running={m.get('vllm:num_requests_running')} | waiting={m.get('vllm:num_requests_waiting')} | "
                      f"kv={m.get('vllm:kv_cache_usage_perc', m.get('vllm:gpu_cache_usage_perc'))} | gen_tokens={m.get('vllm:generation_tokens_total')} | "
                      f"preempt={m.get('vllm:num_preemptions_total')}", flush=True)

    _th.Thread(target=_mon, daemon=True).start()
    with open(OUT / f"runner_{name}.log", "w") as lg:
        p = subprocess.Popen(args, env=dict(env, PYTHONUNBUFFERED="1"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace")
        for line in p.stdout:
            lg.write(line)
            if line.startswith(("PROG |", "SUMMARY |", "[runner]")):
                print(f"[{name}] {line.rstrip()}", flush=True)
        try:
            rc = p.wait(timeout=max(60, left + 600 - (time.monotonic() - t)))
        except subprocess.TimeoutExpired:
            p.kill()
            rc = "timeout"
    stop_mon.set()
    sub = OUT / "run" / name / "submission.jsonl"
    n_ans = sum(1 for l in open(sub) if l.strip() and json.loads(l).get("type") != "abstain") if sub.exists() else 0
    n_s, n_err = error_rate(name)
    step("runner", stage=name, rc=rc, seconds=round(time.monotonic() - t, 1), answered=n_ans, samples=n_s, sample_errors=n_err,
         logs=len(list((OUT / "run" / name / "logs").glob("*.json"))))
    return rc


def vllm_metrics():
    try:
        m = urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=10).read().decode()
    except Exception as e:
        return {"error": str(e)}
    out = {}
    for l in m.splitlines():
        if l.startswith(("vllm:generation_tokens_total", "vllm:prompt_tokens_total", "vllm:spec_decode_num_accepted_tokens_total",
                         "vllm:spec_decode_num_draft_tokens_total", "vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total",
                         "vllm:num_preemptions_total", "vllm:num_requests_running", "vllm:num_requests_waiting",
                         "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")):
            k, v = l.rsplit(" ", 1)
            k = k.split("{")[0]
            out[k] = out.get(k, 0.0) + float(v)
    return out


def fake_llm(port=8099):
    import http.server

    class _Fake(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            b = json.dumps({"data": [{"id": "qwen3.5-9b"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b)

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error":"hint_stage_no_llm"}')

        def log_message(self, *a):
            pass

    s = http.server.ThreadingHTTPServer(("127.0.0.1", port), _Fake)
    _th.Thread(target=s.serve_forever, daemon=True).start()
    return s


def start_server_v2(ok_install, has_spec, overlap=None):
    """Khoi dong vLLM (thu MTP neu co), trong luc nap chay ham overlap (vd buoc chi-hint). Loi MTP -> chay lai khong MTP."""
    if not ok_install:
        if overlap:
            overlap()
        return None
    spec = CONFIG.get("spec_config") if has_spec else None
    h = launch_vllm(spec)
    if overlap:
        overlap()
    if wait_vllm(h):
        return h
    if spec:
        h = launch_vllm(None)
        if wait_vllm(h):
            return h
    return None


def stop_server_v2(h):
    m = vllm_metrics()
    h["p"].terminate()
    try:
        h["p"].wait(60)
    except Exception:
        h["p"].kill()
    h["log"].close()
    step("server_stopped", tag=h["tag"], metrics=m)


if CONFIG.get("plan") == "fewshot_eval":
    # Vi du giai mau TRAIN dua vao goi y cua cau khong co code goi y (khong train, khong tham so moi).
    # Danh gia 38 cau no-hint TRAIN/calibration (khong nam trong ngan hang), cung tham so voi base v5; cham gold ngoai kernel.
    bank_man = one("FEWSHOT_BANK_MANIFEST.json")
    bank_f = bank_man.parent / "bank.jsonl"
    assert sha(bank_f) == json.loads(bank_man.read_text())["bank_sha256"] == CONFIG["bank_sha256"]
    nh_f = code_root / CONFIG["nohint_ids_file"]
    assert sha(nh_f) == CONFIG["nohint_ids_sha256"]
    nh = sorted(set(nh_f.read_text().split()))
    bank_ids = {json.loads(l)["id"] for l in open(bank_f, encoding="utf-8")}
    assert not (bank_ids & set(nh)), "eval leak"
    allq = {q["id"]: q for q in read_q(runtime / "learning/questions.jsonl") + cal}
    dm_e = build_dm("eval38", [allq[i] for i in nh])
    step("fewshot_layout", bank=len(bank_ids), eval=len(nh), leak=0)
    ok_i, _ = install_vllm()
    h = start_server_v2(ok_i, False)
    if not h:
        finish("failed_vllm_server")
        sys.exit(1)
    m0 = vllm_metrics()
    t0 = time.monotonic()
    run_live(CONFIG.get("fewshot_run_name", "eval38_fewshot"), dm_e, CONFIG["phase_seconds"]["eval"],
             CONFIG[CONFIG.get("fewshot_eval_args_key", "eval_v5_args")] + ["--fewshot", str(bank_f), "--fewshot-k", str(CONFIG["fewshot_k"])], "http://127.0.0.1:8000")
    m1 = vllm_metrics()
    step("eval_throughput", wall_s=round(time.monotonic() - t0, 1),
         gen_tokens=m1.get("vllm:generation_tokens_total", 0) - m0.get("vllm:generation_tokens_total", 0))
    stop_server_v2(h)
    finish("fewshot_eval_done")
    sys.exit(0)

if CONFIG.get("plan") == "lora":
    # Huong A: LoRA fresh tren loi giai mau TRAIN (init_mode=fresh, khong dung adapter cu) -> gop vao ban sao base
    # -> vLLM -> danh gia 38 cau no-hint TRAIN/calibration (khong co trong SFT; cham gold ngoai kernel). Khong public/private.
    sft_man = one("TEACHER_SFT_MANIFEST.json")
    sft = sft_man.parent / "sft.jsonl"
    assert sha(sft) == json.loads(sft_man.read_text())["sft_sha256"] == CONFIG["sft_sha256"]
    nh_f = code_root / CONFIG["nohint_ids_file"]
    assert sha(nh_f) == CONFIG["nohint_ids_sha256"]
    nh = sorted(set(nh_f.read_text().split()))
    sft_q = {json.loads(l)["qid"] for l in open(sft, encoding="utf-8")}
    assert not (sft_q & set(nh)), "eval leak"
    allq = {q["id"]: q for q in read_q(runtime / "learning/questions.jsonl") + cal}
    dm_e = build_dm("eval38", [allq[i] for i in nh])
    step("lora_layout", sft_items=len(sft_q), eval=len(nh), leak=0)
    lora_dir = OUT / "lora"
    train_end = time.time() + min(CONFIG["train_seconds"], HARD_END - time.monotonic() - CONFIG["reserve_after_train"])
    t = time.monotonic()
    targs = [sys.executable, "-u", str(code_root / "bridge_v3/train_lora_v3.py"), "--base", str(base), "--sft", str(sft),
             "--out", str(lora_dir), "--deadline-epoch", str(train_end), *CONFIG["lora_train_args"]]
    print("=== STAGE train bat dau ===", flush=True)
    with open(OUT / "train.log", "w") as lg:
        p = subprocess.Popen(targs, env=dict(env, CUDA_VISIBLE_DEVICES="0,1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                                             PYTHONUNBUFFERED="1"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace")
        for line in p.stdout:
            lg.write(line)
            if line.startswith(("TRAIN_", "Traceback")) or "Error" in line[:80]:
                print("[train] " + line.rstrip()[:600], flush=True)
        rc = p.wait()
    done = lora_dir / "training_complete.json"
    info = json.loads(done.read_text()) if done.exists() else {}
    step("train", rc=rc, seconds=round(time.monotonic() - t, 1), **{k: info.get(k) for k in
         ("steps", "items", "planned_steps", "train_loss", "peak_gpu_GB", "stopped_at_deadline", "skipped_overlength")})
    adapter = lora_dir / "final"
    if rc != 0 or not (adapter / "adapter_model.safetensors").exists():
        ck = sorted((lora_dir / "checkpoints").glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[1]))
        adapter = ck[-1] if ck else None
        step("train_fallback_checkpoint", adapter=str(adapter) if adapter else None)
    if adapter is None:
        finish("failed_train")
        sys.exit(1)
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    merged = Path("/kaggle/temp/merged_lora")
    t = time.monotonic()
    with open(OUT / "merge.log", "w") as lg:
        rc = subprocess.run([sys.executable, str(code_root / "bridge_v3/merge_lora.py"), str(base), str(adapter), str(merged)],
                            env=env, stdout=lg, stderr=subprocess.STDOUT, timeout=3600).returncode
    step("merge", rc=rc, adapter=str(adapter), seconds=round(time.monotonic() - t, 1),
         report=json.loads((merged / "MERGE_REPORT.json").read_text()) if (merged / "MERGE_REPORT.json").exists() else None)
    if rc != 0:
        finish("failed_merge")
        sys.exit(1)
    base = merged
    ok_i, has_spec = install_vllm()
    h = start_server_v2(ok_i, False)
    if not h:
        finish("failed_vllm_server_merged")
        sys.exit(1)
    m0 = vllm_metrics()
    t0 = time.monotonic()
    run_live("eval38_lora", dm_e, CONFIG["phase_seconds"]["eval"], CONFIG["eval_v5_args"], "http://127.0.0.1:8000")
    m1 = vllm_metrics()
    step("eval_throughput", wall_s=round(time.monotonic() - t0, 1),
         gen_tokens=m1.get("vllm:generation_tokens_total", 0) - m0.get("vllm:generation_tokens_total", 0))
    stop_server_v2(h)
    finish("lora_done")
    sys.exit(0)

if CONFIG.get("plan") == "v2_bench":
    # 38 cau no-hint TRAIN/calibration (co gold, cham sau khi khoa du doan, ngoai kernel). Hai luot A/B chay DONG THOI:
    # do do on dinh 2 lan chay (yeu cau BTC) va tai cao (~76 cau dang chay) de uoc tinh thong luong cho public.
    nh_f = code_root / CONFIG["nohint_ids_file"]
    assert sha(nh_f) == CONFIG["nohint_ids_sha256"]
    nh = sorted(set(nh_f.read_text().split()))
    allq = {q["id"]: q for q in read_q(runtime / "learning/questions.jsonl") + cal}
    dm_b = build_dm("bench38", [allq[i] for i in nh])
    step("bench_layout", n=len(nh), khang_src=CONFIG["khang_src"])
    ok_i, has_spec = install_vllm()
    h = start_server_v2(ok_i, has_spec)
    if not h:
        finish("failed_vllm_server")
        sys.exit(1)
    m0 = vllm_metrics()
    t0 = time.monotonic()
    ths = [_th.Thread(target=run_live, args=(n, dm_b, CONFIG["phase_seconds"]["bench"], CONFIG["v2_runner_args"], "http://127.0.0.1:8000"))
           for n in ("bench_A", "bench_B")]
    for x in ths:
        x.start()
    for x in ths:
        x.join()
    m1 = vllm_metrics()
    wall = time.monotonic() - t0
    gen = m1.get("vllm:generation_tokens_total", 0) - m0.get("vllm:generation_tokens_total", 0)
    step("bench_throughput", wall_s=round(wall, 1), gen_tokens=gen, gen_tok_per_s=round(gen / max(wall, 1), 1), spec=h["spec"],
         accepted=m1.get("vllm:spec_decode_num_accepted_tokens_total"), drafted=m1.get("vllm:spec_decode_num_draft_tokens_total"))
    stop_server_v2(h)
    finish("v2_bench_done")
    sys.exit(0)

if CONFIG.get("plan") == "v2_public":
    # public_test, khong gold, khong tu nop. Buoc chi-hint (server gia) chay TRONG LUC vLLM nap; sau do vLLM cho cau con abstain.
    qf = one("INPUT_811/PUBLIC/questions.jsonl")
    pub = qf.parent
    assert sha(qf) == CONFIG["public_questions_sha256"]
    assert sha(pub / "calendar.csv") == CONFIG["public_calendar_sha256"]
    pq = read_q(qf)
    rt_saved = runtime
    runtime = pub
    dm_all = build_dm("public_all", pq)
    runtime = rt_saved
    step("public_layout", n=len(pq), khang_src=CONFIG["khang_src"])
    ok_i, has_spec = install_vllm()

    def hint_stage():
        fk = fake_llm(8099)
        run_live("public_hint", dm_all, 3600, CONFIG["v2_runner_args"], "http://127.0.0.1:8099")
        fk.shutdown()

    def load_sub(name):
        f = OUT / "run" / name / "submission.jsonl"
        return {r["id"]: r for r in map(json.loads, open(f, encoding="utf-8")) if r.get("id")} if f.exists() else {}

    def merge(state):
        import collections
        hh, ll = load_sub("public_hint"), load_sub("public_llm")
        rows = [ll.get(q["id"]) if ll.get(q["id"], {}).get("type", "abstain") != "abstain"
                else hh.get(q["id"], {"id": q["id"], "type": "abstain"}) for q in pq]
        tmp = OUT / "submission_public.jsonl.tmp"
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        os.replace(tmp, OUT / "submission_public.jsonl")
        step("public_merged", n=len(rows), by_type=dict(collections.Counter(r.get("type") for r in rows)),
             from_hint=sum(1 for q in pq if hh.get(q["id"], {}).get("type", "abstain") != "abstain"),
             from_llm=sum(1 for q in pq if ll.get(q["id"], {}).get("type", "abstain") != "abstain"))
        finish(state)

    h = start_server_v2(ok_i, has_spec, overlap=hint_stage)
    hint = load_sub("public_hint")
    rest = [q for q in pq if hint.get(q["id"], {}).get("type", "abstain") == "abstain"]
    ids_f = OUT / "public_llm_ids.txt"
    ids_f.write_text("".join(q["id"] + "\n" for q in rest))
    step("public_split", hint_answered=len(pq) - len(rest), llm_todo=len(rest))
    merge("public_hint_only")
    if not h:
        merge("failed_vllm_server_hint_only")
        sys.exit(1)
    m0 = vllm_metrics()
    t0 = time.monotonic()
    rc0 = run_live("public_llm", dm_all, CONFIG["phase_seconds"]["public"], CONFIG["v2_runner_args"], "http://127.0.0.1:8000", ids_file=ids_f)
    m1 = vllm_metrics()
    gen = m1.get("vllm:generation_tokens_total", 0) - m0.get("vllm:generation_tokens_total", 0)
    step("public_throughput", wall_s=round(time.monotonic() - t0, 1), gen_tokens=gen, spec=h["spec"])
    stop_server_v2(h)
    n_s, n_err = error_rate("public_llm")
    merge("v2_public_done" if rc0 == 0 and n_err <= CONFIG["max_error_rate"] * max(n_s, 1) else f"public_llm_rc{rc0}_err{n_err}of{n_s}")
    sys.exit(0)

if CONFIG.get("plan") == "vllm_public":
    # public_test, khong gold, khong tu nop. Buoc 1: chi code hint (server gia tra 400 -> cau khong hint = abstain ngay),
    # dam bao moi cau co hint co dap an truoc khi het gio. Buoc 2: vLLM chi cho cau con abstain. Gop theo thu tu goc.
    import http.server, threading

    class _Fake(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            b = json.dumps({"data": [{"id": "qwen3.5-9b"}]}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(b)

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(400); self.end_headers(); self.wfile.write(b'{"error":"hint_stage_no_llm"}')

        def log_message(self, *a):
            pass

    fake = http.server.ThreadingHTTPServer(("127.0.0.1", 8099), _Fake)
    threading.Thread(target=fake.serve_forever, daemon=True).start()
    qf = one("INPUT_811/PUBLIC/questions.jsonl")
    pub = qf.parent
    assert sha(qf) == CONFIG["public_questions_sha256"]
    assert sha(pub / "calendar.csv") == CONFIG["public_calendar_sha256"]
    pq = read_q(qf)
    rt_saved = runtime
    runtime = pub
    dm_all = build_dm("public_all", pq)
    step("public_layout", n=len(pq), khang_src=CONFIG.get("khang_src"))
    run_runner("public_hint", dm_all, 3600, CONFIG["nohint_runner_args"], [], hints_off=False, base_urls="http://127.0.0.1:8099")
    fake.shutdown()

    def load_sub(name):
        f = OUT / "run" / name / "submission.jsonl"
        return {r["id"]: r for r in map(json.loads, open(f, encoding="utf-8")) if r.get("id")} if f.exists() else {}

    def merge(state):
        h, l = load_sub("public_hint"), load_sub("public_llm")
        rows = [l.get(q["id"]) if l.get(q["id"], {}).get("type", "abstain") != "abstain" else h.get(q["id"], {"id": q["id"], "type": "abstain"})
                for q in pq]
        tmp = OUT / "submission_public.jsonl.tmp"
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        os.replace(tmp, OUT / "submission_public.jsonl")
        import collections
        step("public_merged", n=len(rows), by_type=dict(collections.Counter(r.get("type") for r in rows)),
             from_hint=sum(1 for q in pq if h.get(q["id"], {}).get("type", "abstain") != "abstain"),
             from_llm=sum(1 for q in pq if l.get(q["id"], {}).get("type", "abstain") != "abstain"))
        finish(state)

    hint = load_sub("public_hint")
    rest = [q for q in pq if hint.get(q["id"], {}).get("type", "abstain") == "abstain"]
    dm_llm = build_dm("public_llm", rest)
    runtime = rt_saved
    step("public_split", hint_answered=len(pq) - len(rest), llm_todo=len(rest))
    merge("public_hint_only")  # ban an toan neu buoc vLLM loi
    procs = start_vllm()
    if not procs:
        merge("failed_vllm_server_hint_only")
        sys.exit(1)
    rc0 = run_runner("public_llm", dm_llm, CONFIG["phase_seconds"]["public"], CONFIG["nohint_runner_args"], procs, hints_off=False)
    n_s, n_err = error_rate("public_llm")
    stop_vllm(procs)
    merge("vllm_public_done" if rc0 == 0 and n_err <= CONFIG["max_error_rate"] * max(n_s, 1) else f"public_llm_rc{rc0}_err{n_err}of{n_s}")
    sys.exit(0)

if CONFIG.get("plan") == "vllm_nohint":
    nh_f = code_root / CONFIG["nohint_ids_file"]
    assert sha(nh_f) == CONFIG["nohint_ids_sha256"]
    nh = sorted(set(nh_f.read_text().split()))
    allq = {q["id"]: q for q in read_q(runtime / "learning/questions.jsonl") + cal}
    assert all(i in allq for i in nh)
    stages = [("vllm_probe", build_dm("vllm_probe", [allq[i] for i in nh[:CONFIG["probe_n"]]])),
              ("vllm_rest", build_dm("vllm_rest", [allq[i] for i in nh[CONFIG["probe_n"]:]]))]
    step("nohint_layout", n=len(nh), probe=CONFIG["probe_n"], khang_src=CONFIG.get("khang_src"))
    procs = start_vllm()
    if not procs:
        finish("failed_vllm_server")
        sys.exit(1)
    state = "vllm_nohint_done"
    for name, dm_s in stages:
        rc0 = run_runner(name, dm_s, CONFIG["phase_seconds"]["eval"], CONFIG["nohint_runner_args"], procs, hints_off=False)
        n_s, n_err = error_rate(name)
        if rc0 != 0 or n_s == 0 or n_err > CONFIG["max_error_rate"] * n_s:
            state = f"stopped_{name}_errors"
            break
    stop_vllm(procs)
    finish(state)
    sys.exit(0)

if CONFIG.get("plan") == "nohint":
    # Do 9B base (du goi y Khang, khong --hints-off) tren cac cau Tier1 khong phu; probe nho truoc de chan loi
    nh_f = code_root / CONFIG["nohint_ids_file"]
    assert sha(nh_f) == CONFIG["nohint_ids_sha256"]
    nh = sorted(set(nh_f.read_text().split()))
    allq = {q["id"]: q for q in read_q(runtime / "learning/questions.jsonl") + cal}
    assert all(i in allq for i in nh)
    stages = [("nohint_probe", build_dm("nohint_probe", [allq[i] for i in nh[:CONFIG["probe_n"]]])),
              ("nohint_rest", build_dm("nohint_rest", [allq[i] for i in nh[CONFIG["probe_n"]:]]))]
    if CONFIG.get("probe_only"):
        stages = stages[:1]
    step("nohint_layout", n=len(nh), probe=CONFIG["probe_n"], probe_only=bool(CONFIG.get("probe_only")))
    procs = start_server("base")
    state = "nohint_done"
    for name, dm_s in stages:
        rc0 = run_runner(name, dm_s, CONFIG["phase_seconds"]["eval"], CONFIG["nohint_runner_args"], procs, hints_off=False)
        n_s, n_err = error_rate(name)
        if rc0 != 0 or n_s == 0 or n_err > CONFIG["max_error_rate"] * n_s:
            state = f"stopped_{name}_errors"
            break
    stop_server(procs)
    finish(state)
    sys.exit(0)

# Pha 0 + A: base tren 120 cau (doi chung GD2), roi thu quy dao tren learning pool
procs = start_server("base")
if not procs:
    finish("failed_server_base")
    sys.exit(1)
for name, dm_s, secs in (("smoke10", dm_smoke, 1200), ("eval120_base", dm_eval_rest, CONFIG["phase_seconds"]["eval"])):
    rc0 = run_runner(name, dm_s, secs, CONFIG["eval_runner_args"], procs)
    n_s, n_err = error_rate(name)
    if rc0 != 0 or n_s == 0 or n_err > CONFIG["max_error_rate"] * n_s:
        stop_server(procs)
        finish(f"stopped_{name}_errors")
        sys.exit(1)
rc_a = run_runner("learning", dm_learn, CONFIG["phase_seconds"]["collect"], CONFIG["collect_runner_args"], procs)
stop_server(procs)

# Pha B
sft = OUT / "sft.jsonl"
r = subprocess.run([sys.executable, str(code_root / "bridge/build_sft.py"), str(OUT / "trajectories_learning.jsonl"),
                    str(OUT / "run" / "learning"), str(runtime / "learning/gold.jsonl"), str(sft)],
                   env=env, capture_output=True, text=True, timeout=1800)
try:
    sft_stats = json.loads(r.stdout.strip().splitlines()[-1])
except Exception:
    sft_stats = {"error": r.stderr[-1500:]}
step("build_sft", rc=r.returncode, **{k: v for k, v in sft_stats.items() if not isinstance(v, (dict, list))})
if r.returncode != 0 or sft_stats.get("kept", 0) < 64:
    finish("stopped_insufficient_sft")
    sys.exit(0)

# Pha C
torch.cuda.empty_cache()
adapter_dir = Path("/kaggle/temp/lora")
train_deadline = time.time() + min(CONFIG["phase_seconds"]["train"],
                                   HARD_END - time.monotonic() - CONFIG["phase_seconds"]["eval"] - 900)
with open(OUT / "train.log", "w") as tl:
    rt = subprocess.run([sys.executable, str(code_root / "bridge/train_lora.py"), "--base", str(base), "--sft", str(sft),
                         "--out", str(adapter_dir), "--deadline-epoch", str(train_deadline), *CONFIG["train_args"]],
                        env=dict(env, CUDA_VISIBLE_DEVICES="0"), stdout=tl, stderr=subprocess.STDOUT,
                        timeout=CONFIG["phase_seconds"]["train"] + 1800).returncode
done_p = adapter_dir / "training_complete.json"
if rt != 0 or not done_p.exists():
    step("train", rc=rt)
    if adapter_dir.exists():
        for f in ("metadata.json", "progress.json"):
            if (adapter_dir / f).exists():
                shutil.copy(adapter_dir / f, OUT / f"train_{f}")
    finish("failed_train")
    sys.exit(1)
done = json.loads(done_p.read_text())
final = adapter_dir / "final"
shutil.copytree(final, OUT / "adapter_final")
shutil.copy(adapter_dir / "metadata.json", OUT / "train_metadata.json")
shutil.copy(done_p, OUT / "training_complete.json")
step("train", rc=rt, **{k: v for k, v in done.items() if not isinstance(v, (dict, list))},
     adapter_sha256={f.name: sha(f) for f in (OUT / "adapter_final").iterdir() if f.is_file()})

# Pha D
procs = start_server("adapter", OUT / "adapter_final")
rc_d = run_runner("eval120_adapter", dm_eval, CONFIG["phase_seconds"]["eval"], CONFIG["eval_runner_args"], procs) if procs else None
stop_server(procs)
finish("gd3_done" if rc_d == 0 else "gd3_incomplete")
