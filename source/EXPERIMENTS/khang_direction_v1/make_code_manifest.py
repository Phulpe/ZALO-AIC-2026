"""Tao KHANG_DIRECTION_CODE_MANIFEST.json cho goi code (chi code, khong du lieu cuoc thi)."""
import hashlib
import json
import os
import sys

root = sys.argv[1]
skip = {"KHANG_DIRECTION_CODE_MANIFEST.json", "dataset-metadata.json"}
files = {}
for r, dirs, fs in os.walk(root):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    for f in fs:
        rel = os.path.relpath(os.path.join(r, f), root).replace(os.sep, "/")
        if rel in skip:
            continue
        assert not rel.endswith((".jsonl", ".xlsx", ".csv")) or rel == "eval_ids.txt", rel
        files[rel] = hashlib.sha256(open(os.path.join(root, rel), "rb").read()).hexdigest()
man = {"files": dict(sorted(files.items())), "khang_commit": "c6379f5d35b98e728806acbd226603febb48dd3f",
       "contains_competition_data": False}
p = os.path.join(root, "KHANG_DIRECTION_CODE_MANIFEST.json")
open(p, "w", encoding="utf-8").write(json.dumps(man, indent=1))
print(len(files), hashlib.sha256(open(p, "rb").read()).hexdigest())
