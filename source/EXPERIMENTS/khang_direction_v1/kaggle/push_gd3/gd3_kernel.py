"""ZAIC khang_direction_v1 — GD3 tren Kaggle (mot phien GPU, tan dung toi da khi den luot).

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
    "run_id": "khang_dir_v1_vllm_nohint_20261111b",
    "init_mode": "fresh",
    "runtime_manifest_sha256": "523db34eb8e8bab52aa52d1d046307f19dbb3668495025ca05d66abd1e4c6e70",
    "code_manifest_sha256": "ee6e9e9883a3f58c01de9312ea38ee341b097f0070980e05194d427803217b33",
    "base_small_hashes": {"LICENSE": "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a", "README.md": "c5f5a8c2dddab69cfbf05279235aa5fddb137939a06539c4c7637aa900fef6d0", "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715", "config.json": "d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05", "merges.txt": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d", "model.safetensors.index.json": "26d3539b516be613f39563617cb9d33b3f83d401298125be392c80cefb8f7fe5", "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516", "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42", "tokenizer_config.json": "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8", "video_preprocessor_config.json": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13", "vocab.json": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"},
    "eval_ids_file": "eval_ids_120.txt",
    "eval_ids_sha256": "f337f7f88f71544dc942d1aeb23d98243b964ea1faacfb092c076d399109e26d",
    "smoke_ids_sha256": "64ee76bf459213d9552cb5118d7bd56bf5584f7db99991fc430304c7a4e5fd5c",
    "max_error_rate": 0.2,
    "max_servers": 2,
    "plan": "vllm_nohint",
    "khang_src": "khang_v2/src",
    "vllm_pip": "vllm==0.30.0",
    "vllm_args": ["--dtype", "float16", "--max-model-len", "32768", "--gpu-memory-utilization", "0.90",
                  "--max-num-seqs", "16", "--enable-prefix-caching", "--seed", "0"],
    "vllm_ready_seconds": 2400,
    "nohint_ids_file": "nohint_ids.txt",
    "nohint_ids_sha256": "854cecebeedc89a252daa0d6efab81777e848a3195acf77ab50f0e663ed1c368",
    "probe_n": 4,
    "nohint_runner_args": ["--samples", "3", "--min-votes", "2", "--no-thinking", "--temperature", "0.6",
                           "--max-tokens", "2048", "--max-steps", "6", "--workers", "8", "--kernel-workers", "2"],
    "learning_limit": 0,           # 0 = ca learning pool
    "learning_sample_seed": 20261110,
    "collect_runner_args": ["--samples", "1", "--min-votes", "1", "--no-thinking", "--temperature", "0.6",
                            "--max-tokens", "4096", "--max-steps", "6", "--workers", "8", "--kernel-workers", "2"],
    "eval_runner_args": ["--samples", "1", "--min-votes", "1", "--no-thinking", "--max-tokens", "4096",
                         "--max-steps", "6", "--workers", "8", "--kernel-workers", "2"],
    "server_args": ["--max-batch", "8", "--wait-ms", "80", "--max-new-cap", "4096"],
    "train_args": ["--maxlen", "6144", "--per-question", "2", "--lr", "1e-4", "--epochs", "1", "--grad-accum", "16"],
    "phase_seconds": {"collect": 14400, "train": 9000, "eval": 5400},
    "max_gpu_seconds": 9000,
    "public_test": False,
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


assert not CONFIG["public_test"] and CONFIG["init_mode"] == "fresh"
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
assert torch.cuda.is_available(), "GPU mode without GPU"
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


def run_runner(name, dm, seconds, extra, procs, hints_off=True):
    left = int(min(seconds, HARD_END - time.monotonic() - 600))
    if left < 300:
        step("runner_skipped", stage=name, seconds_left=left)
        return None
    t = time.monotonic()
    args = [sys.executable, str(code_root / "bridge/run_khang.py"), "--data", str(dm), "--out", str(OUT / "run" / name),
            "--cache", "/kaggle/temp/dm_cache", "--model", "qwen3.5-9b", "--budget-seconds", str(left),
            "--base-urls", urls(procs), *extra] + (["--hints-off"] if hints_off else [])
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
    with open(OUT / "pip_vllm.log", "w") as lg:
        rc = subprocess.run([sys.executable, "-m", "pip", "install", "-q", CONFIG["vllm_pip"]], stdout=lg,
                            stderr=subprocess.STDOUT, timeout=2400).returncode
    ver = subprocess.run([sys.executable, "-c", "import vllm,torch;print(vllm.__version__, torch.__version__)"],
                         capture_output=True, text=True).stdout.strip()
    step("vllm_install", rc=rc, version=ver, seconds=round(time.monotonic() - t, 1))
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
