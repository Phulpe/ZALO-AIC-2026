# Zalo AI Challenge 2026 - DataMate. Build tu thu muc goc khang_v4 (co src/, training_code/; hints v4 da kiem Tier1 TRAIN 3000/3000 nhu v2), sau khi chep thu muc nay vao khang_v4/docker/
# va bank.jsonl (few-shot) vao khang_v4/fewshot/:
#   docker build --platform linux/amd64 -f docker/Dockerfile -t zaic2026_datamate_<team>:final .
# BAN NHAP v4 (12/11/2026): vLLM 0.10.1.1 khong ho tro Qwen3_5 -> dung ban da kiem tren Kaggle (0.30.0); can xac nhan image ho tro sm_120
FROM vllm/vllm-openai:v0.30.0
ENTRYPOINT []
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PIP_NO_CACHE_DIR=1 NCCL_P2P_DISABLE=1 \
    VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
WORKDIR /code
COPY docker/requirements.txt /code/requirements.txt
# giu nguyen numpy da kiem thu cung vLLM trong image goc
RUN NUMPY_VER=$(python3 -c "import numpy; print(numpy.__version__)") && \
    pip install --no-cache-dir -r /code/requirements.txt "numpy==${NUMPY_VER}"
# cach khac phuc treo khoi tao 2 GPU theo huong dan cua BTC
RUN pip install --no-cache-dir --no-deps nvidia-nccl-cu12==2.27.7
# trong so model tach lop rieng de doi code khong phai copy lai 30 GB
COPY models/Qwen3.5-9B/ /code/models/Qwen3.5-9B/
COPY training_code/ /code/training_code/
COPY docker/predict.sh docker/README.md /code/
COPY src/ /code/src/
# ngan hang vi du few-shot: 1791 cau TRAIN learning co code hint dung gold TRAIN (khong public/private)
COPY fewshot/bank.jsonl /code/fewshot/bank.jsonl
RUN chmod +x /code/predict.sh && python3 -c "import sys, hashlib; sys.path.insert(0, '/code/src'); import datamate.runner, datamate.fewshot, pandas, openpyxl; \
assert hashlib.sha256(open('/code/fewshot/bank.jsonl', 'rb').read()).hexdigest() == 'f74eb15e44e2995fd1131d096f573cd50e59fb2807dbc606f0c2fe7ca1cb630e'; \
print('import ok', pandas.__version__)"
CMD ["bash", "/code/predict.sh"]
