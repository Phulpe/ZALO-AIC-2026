# syntax=docker/dockerfile:1.7
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime@sha256:eee11b3b3872a8c838e35ef48f08b2d5def2080902c7f666831310ca1a0ef2be
ENTRYPOINT []
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false \
    HF_DEACTIVATE_ASYNC_LOAD=1 CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /code
COPY requirements.txt /code/requirements.txt
RUN python -m pip install --index-url https://pypi.org/simple --no-cache-dir -r /code/requirements.txt \
    && python -m pip freeze > /code/installed_environment.txt
# Named contexts use existing local assets without making another 19GB staging copy.
COPY --from=base_model / /code/models/base/
COPY --from=train_runtime / /code/training_code/data/runtime/
COPY --from=training_evidence / /code/training_code/evidence/
COPY --from=parent_adapter adapter_model.safetensors adapter_config.json /code/training_code/parent_adapter/
COPY --from=frozen_bank / /code/training_code/data/frozen_bank/
COPY --from=selected_adapter adapter_model.safetensors adapter_config.json /code/models/adapter/
COPY --from=source_code / /code/src/
COPY training_README.md /code/training_code/README.md
COPY predict.sh entrypoint.py release_manifest.json README.md /code/
RUN chmod +x /code/predict.sh
CMD ["bash", "/code/predict.sh"]
