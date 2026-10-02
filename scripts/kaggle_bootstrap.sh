#!/usr/bin/env bash
# Run in the first cell of every Kaggle session (Accelerator: GPU T4 x2, Internet: On):
#   !git clone https://github.com/<you>/inference-lab.git && cd inference-lab && bash scripts/kaggle_bootstrap.sh
# Pass "vllm" or "trt" to also install that stack (do them in SEPARATE sessions/notebooks:
# vLLM replaces Kaggle's torch build).
set -euo pipefail

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
nvidia-smi topo -m || true          # expect PHB/SYS between GPUs: PCIe, no NVLink

export CUDA_DEVICE_ORDER=PCI_BUS_ID
EXTRA="dev"
case "${1:-}" in
  vllm) EXTRA="dev,vllm" ;;
  trt)  EXTRA="dev,trt" ;;
esac
pip install -q -e ".[${EXTRA}]"

infer-lab env
pytest -q
