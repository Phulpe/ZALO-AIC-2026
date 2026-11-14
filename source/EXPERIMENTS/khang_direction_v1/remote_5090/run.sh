#!/usr/bin/env bash
# Chay mot luot tren may thue (5090 / 4090 / 2x4090): bash run.sh <ten_luot> <thu_muc_data> [tham so runner them...]
# vd: bash run.sh nohint38_s1 /workspace/pkg/data_nohint38 --seed 1
#     bash run.sh nohint38_think /workspace/pkg/data_nohint38 --seed 1 --thinking   (khong --no-thinking)
# Log tien do: /workspace/runs/<ten>/PROGRESS.tsv + runner.log (dong PROG tung cau). Khong co gold tren may nay.
set -uo pipefail
NAME=$1; DATA=$2; shift 2
PKG=${PKG:-/workspace/pkg}
MODEL_DIR=${MODEL_DIR:-/workspace/models/Qwen3.5-9B}
OUT=/workspace/runs/$NAME
[ -e "$OUT" ] && { echo "da ton tai $OUT (khong ghi de)"; exit 2; }
mkdir -p "$OUT/logs"
THINK="--no-thinking"; EXTRA=()
for a in "$@"; do if [ "$a" = "--thinking" ]; then THINK=""; else EXTRA+=("$a"); fi; done
# Tu chon cau hinh theo GPU (ghi de bang bien moi truong PROFILE/WORKERS/MAXSEQ):
#  >=30 GB (5090, 4090-48G): bf16, 1 server 1 GPU
#  <30 GB x>=4 (4x5060Ti):   bf16, 2 server TP2 (GPU 0,1 -> :8000; 2,3 -> :8001), runner chia deu
#  <30 GB x2 (2x4090/5060Ti): bf16, 1 server TP2 (giong Kaggle 2xT4)
#  <30 GB x1:               fp8 trong so -> KHAC so hoc voi bf16, chi tham khao
NGPU=$(nvidia-smi -L | wc -l); MEMMB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$MEMMB" -ge 30000 ]; then P_DEF=bf16_1; W_DEF=32; S_DEF=64
elif [ "$MEMMB" -lt 20000 ] && [ "$NGPU" -ge 4 ]; then P_DEF=bf16_tp2x2; W_DEF=24; S_DEF=16
elif [ "$NGPU" -ge 2 ]; then
  if [ "$MEMMB" -lt 20000 ]; then P_DEF=bf16_tp2; W_DEF=12; S_DEF=16; else P_DEF=bf16_tp2; W_DEF=24; S_DEF=48; fi
else P_DEF=fp8_1; W_DEF=12; S_DEF=24; fi
PROFILE=${PROFILE:-$P_DEF}; WORKERS=${WORKERS:-$W_DEF}; MAXSEQ=${MAXSEQ:-$S_DEF}
# GPU Turing (sm_75: T4, 2080 Ti) khong co bf16 -> float16 (giong Kaggle 2xT4)
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
DTYPE=${DTYPE:-$([ "${CAP:-80}" -lt 80 ] && echo float16 || echo bfloat16)}
case "$PROFILE" in
  bf16_1)     VARGS=(--dtype $DTYPE --gpu-memory-utilization 0.90); SRV=("0:8000") ;;
  bf16_tp2)   VARGS=(--dtype $DTYPE --tensor-parallel-size 2 --gpu-memory-utilization 0.90); SRV=("0,1:8000") ;;
  bf16_tp2x2) VARGS=(--dtype $DTYPE --tensor-parallel-size 2 --gpu-memory-utilization 0.90); SRV=("0,1:8000" "2,3:8001") ;;
  fp8_1)      VARGS=(--dtype $DTYPE --quantization fp8 --gpu-memory-utilization 0.93); SRV=("0:8000") ;;
  *) echo "PROFILE la: $PROFILE"; exit 2 ;;
esac
URLS=()
for sv in "${SRV[@]}"; do
  DEV=${sv%%:*}; PORT=${sv##*:}; URLS+=("http://127.0.0.1:$PORT")
  # Server dang chay phai phuc vu dung MODEL_DIR (base hay ban gop LoRA); khac thi dung, khong dung nham model.
  RUNNING_ROOT=$(curl -sf "http://127.0.0.1:$PORT/v1/models" | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0].get('root',''))" 2>/dev/null)
  if [ -n "$RUNNING_ROOT" ] && [ "$RUNNING_ROOT" != "$MODEL_DIR" ]; then
    echo "vLLM :$PORT dang phuc vu $RUNNING_ROOT, can $MODEL_DIR -> dung server cu truoc (pkill -f vllm.entrypoints)"; exit 3
  fi
  if [ -z "$RUNNING_ROOT" ]; then
    NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1} VLLM_NO_USAGE_STATS=1 CUDA_VISIBLE_DEVICES=$DEV python3 -m vllm.entrypoints.openai.api_server --model "$MODEL_DIR" --served-model-name qwen3.5-9b       --host 127.0.0.1 --port "$PORT" "${VARGS[@]}" --max-model-len 40960       --max-num-seqs "$MAXSEQ" --enable-prefix-caching --seed 0 > "/workspace/runs/vllm_server_$PORT.log" 2>&1 &
  else
    echo "[run] vLLM :$PORT dang chay san (cau hinh cua lan khoi dong truoc)" | tee -a "$OUT/profile.txt"
  fi
done
for u in "${URLS[@]}"; do
  for i in $(seq 1 180); do curl -sf "$u/v1/models" >/dev/null && break; sleep 10; done
  curl -sf "$u/v1/models" >/dev/null || { echo "vLLM $u khong len"; tail -30 /workspace/runs/vllm_server_*.log; exit 1; }
done
BASE_URLS=$(IFS=,; echo "${URLS[*]}")
echo "[run] GPU ${NGPU}x${MEMMB}MB profile=$PROFILE dtype=$DTYPE cap=$CAP servers=$BASE_URLS workers=$WORKERS max-num-seqs=$MAXSEQ" | tee -a "$OUT/profile.txt"
# FEWSHOT=0: tat few-shot (LoRA: SFT train khong co few-shot). Mac dinh bat, k=3. BUDGET: gioi han giay cua runner.
FEWSHOT=${FEWSHOT:-1}; BUDGET=${BUDGET:-21000}
if [ "$FEWSHOT" = "1" ]; then FS_ARGS=(--fewshot "$PKG/code/fewshot/bank.jsonl" --fewshot-k 3); else FS_ARGS=(); fi
echo "[run] model=$MODEL_DIR fewshot=$FEWSHOT budget=$BUDGET extra=${EXTRA[*]:-}" | tee -a "$OUT/profile.txt"
cd "$PKG/code"
START=$(date +%s)
PYTHONPATH=src python3 -m datamate.runner --data "$DATA" --out "$OUT" --cache /tmp/datamate_cache \
  --base-urls "$BASE_URLS" --backend vllm --model qwen3.5-9b \
  --workers "$WORKERS" --kernel-workers 4 --samples 3 --min-votes 2 --max-steps 6 $THINK --max-tokens 2048 \
  --temperature 0.6 --fixed-samples ${FS_ARGS[@]+"${FS_ARGS[@]}"} --progress-every 300 \
  --budget-seconds "$BUDGET" ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee "$OUT/runner.log"
echo "[run] $NAME xong $(( $(date +%s) - START ))s, $(wc -l < "$OUT/submission.jsonl") dong"
(cd /workspace/runs && tar czf "$NAME.tgz" "$NAME" && sha256sum "$NAME.tgz" > "$NAME.tgz.sha256")
