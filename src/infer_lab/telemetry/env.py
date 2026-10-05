"""Environment fingerprint stored with every result row.

Kaggle hands out different VMs per session (host CPU, P2P support differed between
sessions), so a number is only interpretable together with where it was measured.
Uses NVML only: calling torch.cuda here would initialise CUDA before vLLM forks workers.
"""
from __future__ import annotations

import importlib
import os
import platform

try:
    import pynvml
except ImportError:
    pynvml = None


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _gpus() -> dict:
    if pynvml is None:
        return {}
    try:
        pynvml.nvmlInit()
    except Exception:
        return {}
    try:
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        gpus = []
        for h in handles:
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(h)
            name = pynvml.nvmlDeviceGetName(h)
            gpus.append({
                "name": name.decode() if isinstance(name, bytes) else name,
                "sm": f"{major}{minor}",
                "mem_gib": round(pynvml.nvmlDeviceGetMemoryInfo(h).total / 2**30, 1),
            })
        out = {"driver": pynvml.nvmlSystemGetDriverVersion(), "gpus": gpus}
        if len(handles) > 1:
            try:
                # int(): newer nvidia-ml-py exposes the constant as an enum ctypes can't pass
                status = pynvml.nvmlDeviceGetP2PStatus(
                    handles[0], handles[1], int(pynvml.NVML_P2P_CAPS_INDEX_READ))
                out["p2p_0_1"] = int(status) == int(pynvml.NVML_P2P_STATUS_OK)
            except Exception:  # fingerprinting must never break a benchmark run
                out["p2p_0_1"] = None
        return out
    finally:
        pynvml.nvmlShutdown()


def env_info() -> dict:
    libs = {}
    for mod in ("torch", "transformers", "vllm", "onnxruntime", "tensorrt"):
        try:
            libs[mod] = importlib.import_module(mod).__version__
        except ImportError:
            libs[mod] = None
    try:
        gpus = _gpus()
    except Exception as e:  # never let fingerprinting break a run
        gpus = {"gpu_info_error": repr(e)}
    return {
        "python": platform.python_version(),
        "cpu": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "libs": libs,
        **gpus,
    }
