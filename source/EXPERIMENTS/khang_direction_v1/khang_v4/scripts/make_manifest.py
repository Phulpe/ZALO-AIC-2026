"""Tao training_code/data_manifest.json: duong dan, kich thuoc, SHA-256 cua du lieu kiem dinh va trong so model."""
import hashlib, json, os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def files_under(rel):
    base = os.path.join(ROOT, rel)
    out = []
    for dp, _, fs in os.walk(base):
        for f in sorted(fs):
            p = os.path.join(dp, f)
            out.append({"path": os.path.relpath(p, ROOT), "bytes": os.path.getsize(p), "sha256": sha256(p)})
    return sorted(out, key=lambda x: x["path"])


def count_lines(rel):
    p = os.path.join(ROOT, rel)
    return sum(1 for _ in open(p, encoding="utf-8")) if os.path.exists(p) else None


manifest = {
    "note": "Khong huan luyen / fine-tune, khong sinh du lieu. Bo train BTC cung cap chi dung de kiem dinh (validation) luat va prompt.",
    "image_paths": "Trong image, duong dan tuong doi tinh tu /code (vd training_code/data/train/...).",
    "datasets": [
        {
            "name": "Zalo AI Challenge 2026 DataMate - train (BTC cung cap)",
            "source": "https://challenge.zalo.ai (portal DataMate, tab Data)",
            "role": "validation only (khong dung de huan luyen)",
            "questions": count_lines("training_code/data/train/questions.jsonl"),
            "answers": count_lines("training_code/data/train/answers.jsonl"),
            "files": files_under("training_code/data/train") if os.path.isdir(os.path.join(ROOT, "training_code/data/train")) else [],
        }
    ],
    "models": [
        {
            "name": "Qwen3-14B",
            "source": "https://huggingface.co/Qwen/Qwen3-14B",
            "revision": "40c069824f4251a91eefaf281ebe4c544efd3e18",
            "license": "Apache-2.0",
            "parameters": 14768307200,
            "release_date": "2025-04-29 (repo tao 2025-04-27)",
            "role": "suy luan (duy nhat), khong fine-tune",
            "path_in_image": "/code/models/Qwen3-14B",
            "files": files_under("models/Qwen3-14B") if os.path.isdir(os.path.join(ROOT, "models/Qwen3-14B")) else [],
        }
    ],
}
out = os.path.join(ROOT, "training_code", "data_manifest.json")
json.dump(manifest, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", out, "datasets files:", len(manifest["datasets"][0]["files"]), "model files:", len(manifest["models"][0]["files"]))
