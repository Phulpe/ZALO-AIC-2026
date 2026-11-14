"""CPU: tai san wheel vllm==0.30.0 (+ phu thuoc) cho Python cua Kaggle, de kernel GPU cai offline (--no-index)."""
import subprocess, sys, json, hashlib, pathlib, platform
out = pathlib.Path("/kaggle/working/wheels"); out.mkdir(parents=True, exist_ok=True)
r = subprocess.run([sys.executable, "-m", "pip", "download", "vllm==0.30.0", "-d", str(out)], capture_output=True, text=True)
pathlib.Path("/kaggle/working/pip_download.log").write_text(r.stdout[-20000:] + r.stderr[-20000:])
files = {}
for f in sorted(out.iterdir()):
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    files[f.name] = {"bytes": f.stat().st_size, "sha256": h.hexdigest()}
man = {"rc": r.returncode, "python": platform.python_version(), "n": len(files),
       "total_gb": round(sum(v["bytes"] for v in files.values()) / 1e9, 2), "files": files}
pathlib.Path("/kaggle/working/WHEELS_MANIFEST.json").write_text(json.dumps(man, indent=1))
print(json.dumps({k: man[k] for k in ("rc", "python", "n", "total_gb")}))
