#!/usr/bin/env bash
# Install vLLM + infer-lab into an isolated venv so vLLM's pinned torch/transformers
# never fight Kaggle's preinstalled ones (that conflict is what broke `pip install vllm==0.7.3`).
#
#   bash scripts/vllm_setup.sh            # latest vLLM
#   bash scripts/vllm_setup.sh 0.10.1.1   # a specific version
#
# Afterwards use the venv's binaries, e.g.  /tmp/vllm-env/bin/infer-lab bench ...
# /tmp is not persisted: re-run this once per Kaggle session (~3-5 min).
set -euo pipefail

VENV="${VLLM_VENV:-/tmp/vllm-env}"
SPEC="vllm${1:+==$1}"
export UV_CACHE_DIR=/tmp/uv-cache

pip install -q uv
uv venv -q --python "$(command -v python3)" "$VENV"
# One resolve for everything, so vLLM's torch/transformers pins win consistently.
uv pip install --python "$VENV/bin/python" "$SPEC" -e ".[dev]" requests aiohttp

"$VENV/bin/python" - <<'EOF'
import importlib
for m in ("vllm", "torch", "transformers"):
    print(f"{m:<13}", importlib.import_module(m).__version__)
EOF
"$VENV/bin/infer-lab" env
