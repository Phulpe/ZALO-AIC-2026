"""Tao ban xuat ma/tai lieu ZAIC de day len GitHub (chi doc ban goc, ghi vao thu muc moi).
- Chi lay file van ban (ma, tai lieu, cau hinh), bo du lieu thi, trong so, log, output, file lon.
- Bo file chua >= MAX_IDS id cau hoi khac nhau (danh sach id / journal tung cau).
- Thay van ban cau hoi that (train/public, >= MIN_Q ky tu) bang <REDACTED_COMPETITION_QUESTION>.
- Che chuoi giong bi mat (token Kaggle/GitHub/HF, khoa SSH, email, IP cong khai).
- Notebook: xoa outputs. Ghi EXPORT_MANIFEST.json (sha goc + sha ban xuat).
stdout chi in so tong hop.
Dung: python build_export.py <ZAIC_root> <out_dir> <questions.jsonl> [questions.jsonl ...]
"""
import hashlib, json, os, re, sys

ROOT, OUT = sys.argv[1], sys.argv[2]
QFILES = sys.argv[3:]
assert not os.path.exists(OUT), "out_dir da ton tai"

TOP_DIRS = ["EXPERIMENTS", "drive-download-20241211T124801Z-001", "GUIDE_ARCHIVE", "reports",
            "TRAINING_AUDIT_EVIDENCE", "KAGGLE_STAGING"]
SKIP_DIRS = {"data", "runs", "outputs", "output", "models", "model", "__pycache__", ".git", "node_modules",
             "relearning_v2_dependencies", "site-packages", "wheels", "dl", ".ipynb_checkpoints", "cache",
             "checkpoints", "adapters", "logs", "kaggle_tmp"}
EXT = {".py", ".sh", ".md", ".txt", ".json", ".toml", ".cfg", ".ini", ".yaml", ".yml", ".ipynb", ".csv", ".ps1"}
NAMES = {"Dockerfile", ".gitignore", ".dockerignore", ".env.example", ".python-version"}
MAX_BYTES = 1_500_000
MAX_IDS = 10
MIN_Q = 25
QID = re.compile(r"q_[0-9a-f]{12}")
SECRETS = [
    (re.compile(r'("key"\s*:\s*")[0-9a-f]{32}(")'), r"\1<REDACTED_SECRET>\2"),
    (re.compile(r"KGAT_[A-Za-z0-9]{16,}"), "<REDACTED_SECRET>"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"), "<REDACTED_SECRET>"),
    (re.compile(r"hf_[A-Za-z0-9]{30,}"), "<REDACTED_SECRET>"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"), "<REDACTED_SECRET>"),
    (re.compile(r"[A-Za-z0-9._%+-]+@(gmail|yahoo|outlook|hotmail)\.com"), "<REDACTED_EMAIL>"),
    (re.compile(r"(?<![\d.])(?!(?:127|10|0)\.)(?!192\.168\.)(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?=:\d{2,5}\b)"),
     "<REDACTED_IP>"),
    (re.compile(r"(?<=@)(?!(?:127|10|0)\.)(?:\d{1,3}\.){3}\d{1,3}\b"), "<REDACTED_IP>"),
]

qs = set()
for f in QFILES:
    for l in open(f, encoding="utf-8"):
        q = json.loads(l)["question"].strip()
        if len(q) >= MIN_Q:
            qs.add(q)
qforms = sorted({f for q in qs for f in (q, json.dumps(q)[1:-1])}, key=len, reverse=True)


def candidates():
    for name in sorted(os.listdir(ROOT)):
        p = os.path.join(ROOT, name)
        if os.path.isfile(p):
            yield name
    for d in TOP_DIRS:
        for dp, dns, fns in os.walk(os.path.join(ROOT, d)):
            dns[:] = sorted(x for x in dns if x.lower() not in SKIP_DIRS and not x.startswith(".venv"))
            for fn in sorted(fns):
                yield os.path.relpath(os.path.join(dp, fn), ROOT).replace("\\", "/")


sha = lambda b: hashlib.sha256(b).hexdigest()
files, skipped = [], {"ext": 0, "size": 0, "many_ids": 0, "binary": 0}
skipped_ids = []
n_red_q = n_red_s = 0
for rel in candidates():
    base = os.path.basename(rel)
    ext = os.path.splitext(base)[1].lower()
    if ext not in EXT and base not in NAMES:
        skipped["ext"] += 1; continue
    src = os.path.join(ROOT, rel)
    raw = open(src, "rb").read()
    if len(raw) > MAX_BYTES:
        skipped["size"] += 1; continue
    try:
        txt = raw.decode("utf-8")
    except UnicodeDecodeError:
        skipped["binary"] += 1; continue
    if len(set(QID.findall(txt))) >= MAX_IDS:
        skipped["many_ids"] += 1; skipped_ids.append(rel); continue
    stripped = False
    if ext == ".ipynb":
        try:
            nb = json.loads(txt)
            for c in nb.get("cells", []):
                if c.get("cell_type") == "code":
                    c["outputs"], c["execution_count"] = [], None
            txt = json.dumps(nb, ensure_ascii=False, indent=1) + "\n"; stripped = True
        except ValueError:
            skipped["binary"] += 1; continue
    rq = 0
    for q in qforms:
        if q in txt:
            rq += txt.count(q); txt = txt.replace(q, "<REDACTED_COMPETITION_QUESTION>")
    rs = 0
    for pat, rep in SECRETS:
        txt, k = pat.subn(rep, txt); rs += k
    n_red_q += rq; n_red_s += rs
    out_rel = "source/" + rel
    dst = os.path.join(OUT, out_rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    b = txt.encode("utf-8")
    open(dst, "wb").write(b)
    files.append({"path": out_rel, "source_sha256": sha(raw), "export_sha256": sha(b),
                  "notebook_outputs_stripped": stripped, "redacted_questions": rq, "redacted_secrets": rs})

json.dump({"source_root": ROOT, "rules": {"max_bytes": MAX_BYTES, "max_distinct_question_ids": MAX_IDS,
           "min_question_len_redacted": MIN_Q, "skip_dirs": sorted(SKIP_DIRS)},
           "skipped_counts": skipped, "skipped_many_ids": skipped_ids, "files": files},
          open(os.path.join(OUT, "EXPORT_MANIFEST.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps({"exported": len(files), "bytes": sum(os.path.getsize(os.path.join(OUT, f["path"])) for f in files),
                  "skipped": skipped, "files_with_redacted_questions": sum(1 for f in files if f["redacted_questions"]),
                  "redacted_questions": n_red_q, "redacted_secrets": n_red_s,
                  "files_with_secrets": [f["path"] for f in files if f["redacted_secrets"]]}, ensure_ascii=False))
