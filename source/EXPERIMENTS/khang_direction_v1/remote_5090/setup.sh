#!/usr/bin/env bash
# May thue RTX 5090 / 4090 / 2x4090 (image vllm/vllm-openai:v0.30.0 hoac Ubuntu + CUDA). Chay 1 lan: bash /workspace/pkg/code/setup.sh
# - kiem goi (PKG_MANIFEST), cai pandas/openpyxl, tai Qwen3.5-9B dung revision da pin, kiem hash file nho.
set -euo pipefail
PKG=${PKG:-/workspace/pkg}
MODEL_DIR=${MODEL_DIR:-/workspace/models/Qwen3.5-9B}
REV=c202236235762e1c871ad0ccb60c8ee5ba337b9a
cd "$PKG"
python3 - <<'EOF'
import hashlib, json
m = json.load(open("PKG_MANIFEST.json"))
bad = [f for f, h in m["files"].items() if hashlib.sha256(open(f, "rb").read()).hexdigest() != h]
print("PKG files", len(m["files"]), "bad", len(bad)); assert not bad, bad[:5]
EOF
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
df -h /workspace | tail -1   # can >= 60 GB trong
python3 -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda, 'cap', torch.cuda.get_device_capability(0))"
pip install -q pandas==2.2.3 openpyxl==3.1.5 "numpy==$(python3 -c 'import numpy; print(numpy.__version__)')"
if [ ! -f "$MODEL_DIR/config.json" ]; then
  pip install -q -U "huggingface_hub[hf_transfer]"
  HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.5-9B', revision='$REV', local_dir='$MODEL_DIR')"
fi
python3 - "$MODEL_DIR" <<'EOF'
import hashlib, sys
want = {"config.json": "d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05",
        "model.safetensors.index.json": "26d3539b516be613f39563617cb9d33b3f83d401298125be392c80cefb8f7fe5",
        "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
        "tokenizer_config.json": "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8",
        "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"}
bad = [n for n, h in want.items() if hashlib.sha256(open(f"{sys.argv[1]}/{n}", "rb").read()).hexdigest() != h]
print("model small-file hashes bad:", bad); assert not bad
EOF
cd "$PKG/code" && PYTHONPATH=src python3 tests/test_fewshot.py && PYTHONPATH=src python3 tests/test_hints_v4_robust.py
echo "[setup] OK $(date -u +%FT%TZ)"
