#!/usr/bin/env bash
# Zalo AI Challenge 2026 - DataMate
# Diem vao duy nhat: bash /code/predict.sh (khong tham so).
#   Doc  : /data (chi doc)  -> data/<business_id>/..., questions.jsonl, rulebook.md, calendar.csv
#   Ghi  : /result/submission.jsonl (ghi tung cau ngay khi xong), /result/logs/<id>.json (moi cau mot ban ghi)
# Moi tham so (model, seed, temperature, so lan chay, so luong song song) co dinh trong file nay.
set -uo pipefail

START_EPOCH=$(date +%s)
# Gioi han 6 gio tinh ca nap model; dung nhan cau moi sau 5 gio 40 phut de con thoi gian ghi file.
DEADLINE_EPOCH=$((START_EPOCH + 5 * 3600 + 40 * 60))

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}

DATA_DIR=/data
OUT_DIR=/result
MODEL_DIR=/code/models/Qwen3-14B
PORT=8000
mkdir -p "${OUT_DIR}/logs"

NGPU=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)
if [ "${NGPU:-0}" -ge 2 ]; then TP=2; else TP=1; fi
echo "[predict] $(date -u +%FT%TZ) GPU=${NGPU} tensor_parallel=${TP}"

# 1) Server vLLM noi bo (chi 127.0.0.1, khong mang ngoai). Seed co dinh.
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_DIR}" \
    --served-model-name qwen3-14b \
    --host 127.0.0.1 --port "${PORT}" \
    --tensor-parallel-size "${TP}" \
    --dtype bfloat16 \
    --max-model-len 40960 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 64 \
    --enable-prefix-caching \
    --reasoning-parser qwen3 \
    --seed 0 \
    --disable-log-requests \
    > "${OUT_DIR}/vllm_server.log" 2>&1 &
VLLM_PID=$!

cleanup() {
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# 2) Agent: nap + lam sach du lieu (song song voi luc vLLM khoi dong), roi tra loi tung cau.
cd /code
PYTHONPATH=/code/src python3 -m datamate.runner \
    --data "${DATA_DIR}" \
    --out "${OUT_DIR}" \
    --cache /tmp/datamate_cache \
    --base-urls "http://127.0.0.1:${PORT}" \
    --backend vllm \
    --model qwen3-14b \
    --workers 32 \
    --samples 3 \
    --min-votes 2 \
    --max-steps 6 \
    --max-tokens 8192 \
    --temperature 0.6 \
    --seed 1234 \
    --deadline-epoch "${DEADLINE_EPOCH}" \
    ${DATAMATE_EXTRA_ARGS:-}
# DATAMATE_EXTRA_ARGS: chi dung khi doi tu danh gia (vd "--limit 100 --sample-seed 7"); BTC chay khong co bien nay.
STATUS=$?

echo "[predict] runner exit=${STATUS}, $(wc -l < "${OUT_DIR}/submission.jsonl") dong, $(( $(date +%s) - START_EPOCH ))s"
exit 0
