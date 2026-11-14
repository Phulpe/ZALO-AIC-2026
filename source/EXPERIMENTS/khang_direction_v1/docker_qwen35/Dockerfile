# Zalo AI Challenge 2026 - DataMate. Build tu thu muc goc khang_v2 (co src/, training_code/), sau khi chep thu muc nay vao khang_v2/docker/:
#   docker build --platform linux/amd64 -f docker/Dockerfile -t zaic2026_datamate_<team>:final .
# BAN NHAP: vLLM 0.10.1.1 khong ho tro Qwen3_5 -> dung ban da kiem tren Kaggle (0.30.0); can xac nhan image ho tro sm_120
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
RUN chmod +x /code/predict.sh && python3 -c "import sys; sys.path.insert(0, '/code/src'); import datamate.runner, pandas, openpyxl; print('import ok', pandas.__version__)"
CMD ["bash", "/code/predict.sh"]
