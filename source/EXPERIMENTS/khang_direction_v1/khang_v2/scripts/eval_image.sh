#!/usr/bin/env bash
# Danh gia image tren mot split co dap an (train) hoac khong (public_test), dung dung lenh cham cua BTC.
# Dung: scripts/eval_image.sh <ten_doi> <split: train|public_test> <thu_muc_ket_qua> ["--limit 100 --sample-seed 7"]
set -euo pipefail
TEAM="$1"; SPLIT="$2"; OUT="$3"; EXTRA="${4:-}"
mkdir -p "${OUT}"; OUT="$(cd "${OUT}" && pwd)"
DATA="$(cd "$(dirname "$0")/.." && pwd)/data/${SPLIT}"
START=$(date +%s)
docker run --rm --gpus '"device=0,1"' --network none --shm-size=16g --entrypoint bash \
  -e DATAMATE_EXTRA_ARGS="${EXTRA}" \
  -v "${DATA}":/data:ro -v "${OUT}":/result \
  "zaic2026_datamate_${TEAM}:final" /code/predict.sh
echo "Tong thoi gian: $(( $(date +%s) - START ))s"
GOLD=""; [ -f "${DATA}/answers.jsonl" ] && GOLD="--gold ${DATA}/answers.jsonl"
python3 "$(dirname "$0")/check_result.py" "${DATA}/questions.jsonl" "${OUT}" ${GOLD}
