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
    "mode": "cpu_preflight",
    "run_id": "khang_dir_v1_cpu_preflight_20261110",
    "runtime_manifest_sha256": "523db34eb8e8bab52aa52d1d046307f19dbb3668495025ca05d66abd1e4c6e70",
    "calibration_questions_sha256": "703dae803191682a4222416028ec9733fcb12fb71bb15e04498c5d1385e7ff8c",
    "code_manifest_sha256": "15bef555384bfb7a1b99869e39104a9a59cb066a0f03c17deafff00795ed4ffa",
    "base_small_hashes": {},
    "ids_sha256": None,
    "runner_args": ["--samples", "1", "--min-votes", "1", "--no-thinking", "--max-tokens", "4096",
                    "--max-steps", "6", "--workers", "8", "--kernel-workers", "2"],
    "hints_off": True,
    "server_args": ["--max-batch", "8", "--wait-ms", "80", "--max-new-cap", "4096"],
    "max_gpu_seconds": 900,
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
dm = Path("/kaggle/working/dm/calibration")
if dm.exists():
    shutil.rmtree(dm)
(dm / "data").mkdir(parents=True)
for name in ("calendar.csv", "rulebook.md"):
    os.symlink(runtime / name, dm / name)
for b in sorted({q["business_id"] for q in qs}):
    os.symlink(runtime / "data" / b, dm / "data" / b)
ids_file = code_root / "eval_ids.txt"
if CONFIG["mode"] != "cpu_preflight":
    assert sha(ids_file) == CONFIG["ids_sha256"]
    keep = set(ids_file.read_text().split())
    qs_run = [q for q in qs if q["id"] in keep]
    assert len(qs_run) == len(keep)
else:
    qs_run = qs
(dm / "questions.jsonl").write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in qs_run), encoding="utf-8")
split = {"calibration": sorted({q["business_id"] for q in qs})}
(OUT / "split_calibration.json").write_text(json.dumps(split))
step("data_layout", n=len(qs_run), businesses=len(split["calibration"]))

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
    budget = max(60, int(deadline - time.monotonic() - 60))
    args = [sys.executable, str(code_root / "bridge/run_khang.py"), "--data", str(dm), "--out", str(OUT / "run"),
            "--cache", "/kaggle/temp/dm_cache", "--model", "qwen3.5-9b", "--budget-seconds", str(budget),
            *CONFIG["runner_args"]] + (["--hints-off"] if CONFIG["hints_off"] else [])
    with open(OUT / "runner.log", "w") as lg:
        rr = subprocess.run(args, env=env, stdout=lg, stderr=subprocess.STDOUT, timeout=budget + 600).returncode
    try:
        stats = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/stats", timeout=10).read())
    except Exception as e:
        stats = {"error": str(e)}
    step("runner", rc=rr, server_stats=stats, run_seconds=round(time.monotonic() - T0, 1))
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
