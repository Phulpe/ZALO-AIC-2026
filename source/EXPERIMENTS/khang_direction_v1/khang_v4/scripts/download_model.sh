#!/usr/bin/env bash
# Tai trong so Qwen3-14B (ban goc, Apache-2.0) dung revision co dinh vao models/Qwen3-14B.
# Chay tren may build image (can mang). Khoang 30 GB.
set -euo pipefail
REPO="Qwen/Qwen3-14B"
REV="40c069824f4251a91eefaf281ebe4c544efd3e18"
DEST="$(cd "$(dirname "$0")/.." && pwd)/models/Qwen3-14B"
mkdir -p "${DEST}"
python3 -m pip install -q "huggingface_hub>=0.24"
python3 - "$REPO" "$REV" "$DEST" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, rev, dest = sys.argv[1:4]
snapshot_download(repo_id=repo, revision=rev, local_dir=dest,
                  allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json", "LICENSE", "README.md"])
print("ok", dest)
PY
cd "${DEST}"
rm -rf .cache
find . -type f ! -name 'SHA256SUMS' -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
echo "Da tai xong. Kiem tra: (cd ${DEST} && sha256sum -c SHA256SUMS)"
