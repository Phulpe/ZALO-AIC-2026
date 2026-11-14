#!/usr/bin/env bash
# Chay image dung lenh cham cua BTC tren bo public test (hoac train) va kiem tra ket qua.
# Dung: scripts/test_image.sh <ten_doi> <thu_muc_split> <thu_muc_ket_qua> [answers.jsonl de cham]
set -euo pipefail
TEAM="$1"; SPLIT="$(cd "$2" && pwd)"; OUT="$3"; GOLD="${4:-}"
mkdir -p "${OUT}"; OUT="$(cd "${OUT}" && pwd)"
TAG="zaic2026_datamate_${TEAM}:final"
START=$(date +%s)
docker run --rm --gpus '"device=0,1"' --network none --shm-size=16g \
    --entrypoint bash \
    -v "${SPLIT}":/data:ro \
    -v "${OUT}":/result \
    "${TAG}" /code/predict.sh
echo "Thoi gian: $(( $(date +%s) - START ))s"
python3 "$(dirname "$0")/check_result.py" "${SPLIT}/questions.jsonl" "${OUT}" ${GOLD:+--gold "${GOLD}"}
