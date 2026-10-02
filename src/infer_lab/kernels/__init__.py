"""JIT build of the CUDA kernel lab extensions (cached under ~/.cache/torch_extensions)."""
from __future__ import annotations

import functools
import os
from pathlib import Path

_CSRC = Path(__file__).parent / "csrc"


@functools.cache
def load_ext(name: str):
    import torch
    from torch.utils.cpp_extension import load

    if "TORCH_CUDA_ARCH_LIST" not in os.environ and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"  # sm_75 on T4: faster build
    return load(
        name=f"infer_lab_{name}",
        sources=[str(_CSRC / f"{name}.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        verbose=False,
    )
