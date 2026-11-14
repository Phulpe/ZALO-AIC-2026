#!/usr/bin/env bash
# Dong goi image nop bai. Chay tu may Linux x86_64 co Docker (BuildKit), sau khi da chay scripts/download_model.sh.
# Dung: scripts/build_image.sh <ten_doi_viet_thuong>
set -euo pipefail
TEAM="${1:?can ten doi viet thuong, vd: scripts/build_image.sh abcteam}"
cd "$(dirname "$0")/.."
test -f models/Qwen3-14B/model.safetensors.index.json || { echo "Chua co models/Qwen3-14B: chay scripts/download_model.sh"; exit 1; }
(cd models/Qwen3-14B && sha256sum -c --quiet SHA256SUMS)
mkdir -p training_code/data
rsync -a --delete data/train/ training_code/data/train/
python3 scripts/make_manifest.py
TAG="zaic2026_datamate_${TEAM}:final"
DOCKER_BUILDKIT=1 docker build --platform linux/amd64 -f docker/Dockerfile -t "${TAG}" .
docker image inspect -f '{{.Os}}/{{.Architecture}} {{.Size}}' "${TAG}"
docker run --rm --network none --entrypoint "" "${TAG}" find /code -maxdepth 2 | sort | head -50
