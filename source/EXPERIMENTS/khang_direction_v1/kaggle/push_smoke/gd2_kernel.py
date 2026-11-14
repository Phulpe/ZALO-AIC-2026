"""ZAIC khang_direction_v1 — GD2 tren Kaggle. Ba che do trong CONFIG['mode']:
- cpu_preflight: kiem input/hash, moi truong, test bridge + test kernel Khang, tang code goi y tren calibration (khong gold).
- gpu_smoke: nap Qwen3.5-9B (nf4, cuda:0) qua hf_chat_server, chay runner Khang --hints-off tren vai cau, do toc do/VRAM.
- gpu_eval: nhu smoke tren danh sach ID da dong bang.
Khong dua gold vao runner. Khong public_test. Ket qua ghi /kaggle/working/out; journal ghi tam roi doi ten.
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
    "mode": "gpu_eval",
    "run_id": "khang_dir_v1_gd2_smoke_eval120_20261110",
    "runtime_manifest_sha256": "523db34eb8e8bab52aa52d1d046307f19dbb3668495025ca05d66abd1e4c6e70",
    "calibration_questions_sha256": "703dae803191682a4222416028ec9733fcb12fb71bb15e04498c5d1385e7ff8c",
    "code_manifest_sha256": "a38735a23e45cd88fd462f1de938897662edaab1857b69469c2cef14c114f1da",
    "base_small_hashes": {"LICENSE": "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a", "README.md": "c5f5a8c2dddab69cfbf05279235aa5fddb137939a06539c4c7637aa900fef6d0", "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715", "config.json": "d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05", "merges.txt": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d", "model.safetensors.index.json": "26d3539b516be613f39563617cb9d33b3f83d401298125be392c80cefb8f7fe5", "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516", "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42", "tokenizer_config.json": "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8", "video_preprocessor_config.json": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13", "vocab.json": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"},
    "stages": [{"name": "smoke10", "ids_file": "eval_ids.txt", "ids_sha256": "64ee76bf459213d9552cb5118d7bd56bf5584f7db99991fc430304c7a4e5fd5c"}, {"name": "eval120_rest", "ids_file": "eval_ids_120.txt", "ids_sha256": "f337f7f88f71544dc942d1aeb23d98243b964ea1faacfb092c076d399109e26d"}],  # [{"name","ids_file","ids_sha256"}]; cau da chay o stage truoc bi loai o stage sau
    "min_seconds_next_stage": 600,
    "runner_args": ["--samples", "1", "--min-votes", "1", "--no-thinking", "--max-tokens", "4096",
                    "--max-steps", "6", "--workers", "8", "--kernel-workers", "2"],
    "hints_off": True,
    "server_args": ["--max-batch", "8", "--wait-ms", "80", "--max-new-cap", "4096"],
    "max_gpu_seconds": 10800,
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
    print("STEP", name, json.dumps(kw, ensure_ascii=False)[:300], flush=True)


def one(pattern):
    hits = list(Path("/kaggle/input").rglob(pattern))
    assert len(hits) == 1, f"{pattern}: {len(hits)} matches"
    return hits[0]


assert not CONFIG["public_test"]
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
calq = runtime / "calibration/questions.jsonl"
assert sha(calq) == CONFIG["calibration_questions_sha256"]
step("inputs_verified", code_manifest=sha(code_man))

# 2. Bo du lieu dang Khang cho calibration (khong gold)
qs = [json.loads(l) for l in calq.read_text(encoding="utf-8-sig").splitlines() if l.strip()]
split = {"calibration": sorted({q["business_id"] for q in qs})}
(OUT / "split_calibration.json").write_text(json.dumps(split))


def build_dm(name, subset):
    dm = Path("/kaggle/working/dm") / name
    if dm.exists():
        shutil.rmtree(dm)
    (dm / "data").mkdir(parents=True)
    for n in ("calendar.csv", "rulebook.md"):
        os.symlink(runtime / n, dm / n)
    for b in sorted({q["business_id"] for q in qs}):
        os.symlink(runtime / "data" / b, dm / "data" / b)
    (dm / "questions.jsonl").write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in subset), encoding="utf-8")
    return dm


STAGES = []  # (ten, thu muc du lieu, so cau)
if CONFIG["mode"] == "cpu_preflight":
    dm = build_dm("calibration", qs)
    step("data_layout", n=len(qs), businesses=len(split["calibration"]))
else:
    done = set()
    for st in CONFIG["stages"]:
        f = code_root / st["ids_file"]
        assert sha(f) == st["ids_sha256"], st["ids_file"]
        keep = set(f.read_text().split()) - done
        sub = [q for q in qs if q["id"] in keep]
        assert len(sub) == len(keep), st["name"]
        done |= keep
        STAGES.append((st["name"], build_dm(st["name"], sub), len(sub)))
    step("data_layout", stages=[(n, k) for n, _, k in STAGES], businesses=len(split["calibration"]))

env = dict(os.environ, PYTHONIOENCODING="utf-8", KHANG_SRC=str(code_root / "khang/src"))


def versions():
    v = {"python": sys.version.split()[0]}
    for m in ("pandas", "openpyxl", "numpy", "torch", "transformers", "peft"):
        try:
            v[m] = __import__(m).__version__
        except Exception as e:
            v[m] = f"missing: {type(e).__name__}"
    return v


step("versions", **versions())

if CONFIG["mode"] == "cpu_preflight":
    rc = {}
    for t in ("bridge/test_hf_chat_server.py", "bridge/test_run_khang.py", "khang/tests/test_kernel.py",
              "khang/tests/test_dates_args.py"):
        work = Path("/kaggle/working/t") / t.replace("/", "_")
        shutil.copytree(code_root, work, symlinks=True)
        tr = work / "khang/data/train"
        tr.mkdir(parents=True)
        os.symlink(runtime / "calendar.csv", tr / "calendar.csv")
        os.symlink(runtime / "data", tr / "data")
        cwd = work / "khang" if t.startswith("khang/") else work
        r = subprocess.run([sys.executable, str(work / t)], cwd=cwd, env=dict(env, PYTHONPATH=str(work / "khang/src")),
                           capture_output=True, text=True, timeout=1200)
        rc[t] = r.returncode
        (OUT / f"test_{t.replace('/', '_')}.log").write_text(r.stdout[-4000:] + "\n" + r.stderr[-4000:])
    step("tests", **rc)
    env_t = dict(env, TIER1_DATA=str(dm), TIER1_SPLITS="calibration")
    r = subprocess.run([sys.executable, str(code_root / "tier1_eval.py"), str(code_root / "khang"),
                        str(code_root / "scoring"), str(OUT / "split_calibration.json"), str(OUT / "TIER1_CALIBRATION.jsonl")],
                       env=env_t, capture_output=True, text=True, timeout=7200)
    (OUT / "tier1.log").write_text(r.stdout[-6000:] + "\n" + r.stderr[-6000:])
    step("tier1", rc=r.returncode, summary=r.stdout[-1500:])
    # Template co ho tro enable_thinking? (CPU, chi tokenizer)
    try:
        from transformers import AutoTokenizer
        base_man = one("BASE_UPLOAD_MANIFEST.json")
        tok = AutoTokenizer.from_pretrained(str(base_man.parent / "base"), local_files_only=True)
        m = [{"role": "user", "content": "x"}]
        a = tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        b = tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        step("chat_template", differs=a != b, no_think_tail=a[-40:], think_tail=b[-40:])
        small = {}
        for f in sorted((base_man.parent / "base").iterdir()):
            if f.is_file() and f.stat().st_size < 50_000_000:
                small[f.name] = sha(f)
        (OUT / "BASE_SMALL_HASHES.json").write_text(json.dumps(small, indent=1))
        step("base_small_hashes", n=len(small))
    except Exception as e:
        step("chat_template_error", error=f"{type(e).__name__}: {e}")
    save_status(state="cpu_preflight_done")
    sys.exit(0)

# 3. GPU
import torch  # noqa: E402
assert torch.cuda.is_available(), "GPU mode without GPU"
base = one("BASE_UPLOAD_MANIFEST.json").parent / "base"
for name, d in CONFIG["base_small_hashes"].items():
    assert sha(base / name) == d, name
exec(compile((code_root / "kaggle/OFFLINE_BNB_BOOTSTRAP.py").read_text(), "bnb_bootstrap", "exec"))
step("gpu_ready", gpu=torch.cuda.get_device_name(0), count=torch.cuda.device_count())
env["PYTHONPATH"] = os.environ.get("PYTHONPATH", "")
chk = subprocess.run([sys.executable, "-c", "import bitsandbytes as b; print(b.__version__)"], env=env,
                     capture_output=True, text=True, timeout=300)
assert chk.returncode == 0 and chk.stdout.strip() == "0.50.2", chk.stderr[-500:]
step("server_env_bnb", version=chk.stdout.strip())


def prefetch_base(src):
    """Chep base tu /kaggle/input (doc mang, rat cham khi nap tung tensor) sang dia cuc bo bang 8 luong doc tuan tu."""
    from concurrent.futures import ThreadPoolExecutor
    files = [p for p in src.iterdir() if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    for d in ("/kaggle/temp", "/kaggle/working", "/tmp"):
        dst = Path(d) / "base_local"
        try:
            Path(d).mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(d).free
        except Exception:
            continue
        if free > total + 4_000_000_000 and d != "/kaggle/working":
            break
    else:
        dst = None
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

srv_log = open(OUT / "server.log", "w")
srv = subprocess.Popen([sys.executable, str(code_root / "bridge/hf_chat_server.py"), "--base", str(base), "--port", "8000",
                        *CONFIG["server_args"]], env=dict(env, CUDA_VISIBLE_DEVICES="0"), stdout=srv_log, stderr=subprocess.STDOUT)
deadline = T0 + CONFIG["max_gpu_seconds"]
ready = False
while time.monotonic() < deadline and srv.poll() is None:
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=5)
        ready = True
        break
    except Exception:
        time.sleep(5)
step("server", ready=ready, load_seconds=round(time.monotonic() - T0, 1))
rr = None
if ready:
    for name, dm_s, n in STAGES:
        left = int(deadline - time.monotonic() - 120)
        if left < CONFIG["min_seconds_next_stage"]:
            step("stage_skipped", stage=name, seconds_left=left)
            break
        t_s = time.monotonic()
        args = [sys.executable, str(code_root / "bridge/run_khang.py"), "--data", str(dm_s), "--out", str(OUT / "run" / name),
                "--cache", "/kaggle/temp/dm_cache", "--model", "qwen3.5-9b", "--budget-seconds", str(left),
                *CONFIG["runner_args"]] + (["--hints-off"] if CONFIG["hints_off"] else [])
        with open(OUT / f"runner_{name}.log", "w") as lg:
            try:
                rr = subprocess.run(args, env=dict(env, TRAJ_LOG=str(OUT / f"trajectories_{name}.jsonl")), stdout=lg,
                                    stderr=subprocess.STDOUT, timeout=left + 300).returncode
            except subprocess.TimeoutExpired:
                rr = "timeout"
        try:
            stats = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/stats", timeout=10).read())
        except Exception as e:
            stats = {"error": str(e)}
        sec = round(time.monotonic() - t_s, 1)
        step("runner", stage=name, rc=rr, n=n, stage_seconds=sec, s_per_q=round(sec / max(n, 1), 1), server_stats=stats)
        if rr != 0:
            break
srv.terminate()
try:
    srv.wait(60)
except Exception:
    srv.kill()
srv_log.close()
manifest = {str(p.relative_to(OUT)): {"bytes": p.stat().st_size, "sha256": sha(p)}
            for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "ARTIFACT_MANIFEST.json"}
(OUT / "ARTIFACT_MANIFEST.json").write_text(json.dumps(manifest, indent=1))
save_status(state="gpu_done" if rr == 0 else "gpu_incomplete")
