#!/usr/bin/env bash
# Chuan bi may GPU (VM Linux x86_64, 2 GPU NVIDIA) de danh gia va dong goi image. Chay tu thu muc goc repo.
set -euo pipefail
echo "== GPU"; nvidia-smi -L
if ! command -v docker >/dev/null; then
  echo "== Cai Docker"; curl -fsSL https://get.docker.com | sh
fi
if ! docker info 2>/dev/null | grep -qi nvidia; then
  echo "== Cai NVIDIA Container Toolkit"
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -qq && apt-get install -y -qq nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker && systemctl restart docker
fi
command -v pigz >/dev/null || apt-get install -y -qq pigz rsync python3-pip >/dev/null || true
echo "== Kiem tra GPU trong container"; docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi -L
echo "== Keo image goc"; docker pull vllm/vllm-openai:v0.10.1.1
echo "== Tai model"; bash scripts/download_model.sh
df -h .
