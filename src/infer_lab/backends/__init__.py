"""Backend registry. Imports are lazy so missing optional deps (vllm, tensorrt) only fail
when that backend is actually requested."""
from __future__ import annotations

from infer_lab.backends.base import Backend


def create_backend(name: str, **kwargs) -> Backend:
    if name == "hf_torch":
        from infer_lab.backends.hf_torch import HFTorchBackend
        return HFTorchBackend(**kwargs)
    raise ValueError(f"unknown backend {name!r} (available: hf_torch)")
