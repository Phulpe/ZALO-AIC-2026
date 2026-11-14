#!/usr/bin/env bash
# Danh gia mot checkpoint LoRA tren 38 cau no-hint (may thue). Khong co gold tren may nay; cham o may local.
# Dung: bash eval_lora.sh <checkpoint-N> <ten_luot> [tham so runner them, vd --seed 1]
#   - kiem sha adapter theo $PKG/adapters/ADAPTERS_MANIFEST.json
#   - gop adapter vao ban sao base -> /workspace/models/merged_<checkpoint-N> (base goc khong sua)
#   - dung moi vLLM dang chay, chay run.sh voi MODEL_DIR=ban gop, FEWSHOT=0 (dinh dang luc train)
#   - xong: dung vLLM, xoa ban gop (KEEP_MERGED=1 de giu)
set -euo pipefail
CK=$1; NAME=$2; shift 2
PKG=${PKG:-/workspace/pkg}
BASE=${BASE_DIR:-/workspace/models/Qwen3.5-9B}
AD=$PKG/adapters/$CK
MERGED=/workspace/models/merged_$CK
python3 - "$PKG/adapters" "$CK" <<'EOF'
import hashlib, json, os, sys
root, ck = sys.argv[1], sys.argv[2]
m = json.load(open(os.path.join(root, "ADAPTERS_MANIFEST.json")))
files = {k: v for k, v in m["files"].items() if k.startswith(ck + "/")}
assert len(files) == 2, ("adapter khong co trong manifest", ck, list(files))
bad = [k for k, v in files.items() if hashlib.sha256(open(os.path.join(root, k), "rb").read()).hexdigest() != v]
print("[eval_lora] adapter", ck, "files", len(files), "bad", bad); assert not bad
EOF
pkill -f vllm.entrypoints || true
for i in $(seq 1 60); do pgrep -f vllm.entrypoints >/dev/null || break; sleep 2; done
if [ ! -f "$MERGED/MERGE_REPORT.json" ]; then
  rm -rf "$MERGED.tmp"
  T=$(date +%s)
  python3 "$PKG/code/merge_lora.py" "$BASE" "$AD" "$MERGED.tmp" 2>&1 | tail -5
  test -f "$MERGED.tmp/MERGE_REPORT.json"
  mv "$MERGED.tmp" "$MERGED"
  echo "[eval_lora] merge $CK xong $(( $(date +%s) - T ))s"
fi
python3 -c "import json; r=json.load(open('$MERGED/MERGE_REPORT.json')); print('[eval_lora] MERGE_REPORT', {k: r[k] for k in list(r)[:8]})"
MODEL_DIR=$MERGED FEWSHOT=${FEWSHOT:-0} BUDGET=${BUDGET:-7200} bash "$PKG/code/run.sh" "$NAME" "$PKG/data_nohint38" "$@"
cp "$MERGED/MERGE_REPORT.json" "/workspace/runs/$NAME.MERGE_REPORT.json"
pkill -f vllm.entrypoints || true
[ "${KEEP_MERGED:-0}" = "1" ] || rm -rf "$MERGED"
echo "[eval_lora] $CK -> /workspace/runs/$NAME.tgz"
