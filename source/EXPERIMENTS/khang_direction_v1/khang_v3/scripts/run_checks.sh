#!/usr/bin/env bash
# Doi chieu tat dinh cac ham dung san voi dap an bo train (khong dung model).
# Chay: bash scripts/run_checks.sh [file_ket_qua]
# Mac dinh ghi vao training_code/eval/deterministic_checks_train_v3.txt; dong cuoi "TONG dung/tong".
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-training_code/eval/deterministic_checks_train_v3.txt}"
PY="${PYTHON:-.venv/bin/python}"
CHECKS=(semantic semantic2 debt imports profit inventory pharmacy price_effect customers misc delivery)
TMP="$(mktemp)"
{
  echo "# Doi chieu tat dinh voi dap an train ($(date '+%Y-%m-%d %H:%M'))"
  echo "# Moi dong: <dang cau hoi>: <so cau dung>/<so cau dang do>. Chay: bash scripts/run_checks.sh"
  for c in "${CHECKS[@]}"; do
    echo "== check_$c"
    PYTHONPATH=src "$PY" "scripts/check_$c.py" | grep -E ': [0-9]+/[0-9]+ khop$' || { echo "check_$c LOI" >&2; exit 1; }
  done
} > "$TMP"
awk -F'[ /]' '/ khop$/ {split($0, a, ": "); split(a[2], b, "/"); split(b[2], c, " "); ok += b[1]; n += c[1]} END {print "TONG " ok "/" n}' "$TMP" >> "$TMP"
mv "$TMP" "$OUT"
cat "$OUT"
