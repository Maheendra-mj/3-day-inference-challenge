"""Backend registry. Imports are lazy so missing optional deps (vllm, tensorrt) only fail
when that backend is actually requested."""
from __future__ import annotations

from infer_lab.backends.base import Backend

_AVAILABLE = ("hf_torch", "vllm_offline")


def create_backend(name: str, **kwargs) -> Backend:
    if name == "hf_torch":
        from infer_lab.backends.hf_torch import HFTorchBackend
        return HFTorchBackend(**kwargs)
    if name == "vllm_offline":
        from infer_lab.backends.vllm_offline import VLLMOfflineBackend
        return VLLMOfflineBackend(**kwargs)
    raise ValueError(f"unknown backend {name!r} (available: {', '.join(_AVAILABLE)})")
