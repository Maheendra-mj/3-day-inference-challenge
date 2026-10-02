"""Background NVML sampler: GPU utilisation, VRAM and power during a benchmark.

NVML indices follow PCI bus order; CUDA indices follow CUDA_DEVICE_ORDER. On Kaggle's
T4x2 they coincide, but set CUDA_DEVICE_ORDER=PCI_BUS_ID elsewhere to be safe.
"""
from __future__ import annotations

import threading
import time

import numpy as np

try:
    import pynvml
except ImportError:  # local dev machine without NVIDIA driver
    pynvml = None


class GPUMonitor:
    def __init__(self, device_indices: list[int] | None = None, interval_s: float = 0.1):
        self.interval_s = interval_s
        self.device_indices = device_indices
        self._samples: dict[int, list[tuple[float, float, float]]] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.enabled = pynvml is not None

    def __enter__(self) -> "GPUMonitor":
        if not self.enabled:
            return self
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError:
            self.enabled = False
            return self
        if self.device_indices is None:
            self.device_indices = list(range(pynvml.nvmlDeviceGetCount()))
        self._handles = {i: pynvml.nvmlDeviceGetHandleByIndex(i) for i in self.device_indices}
        self._samples = {i: [] for i in self.device_indices}
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if not self.enabled:
            return
        self._stop.set()
        if self._thread:
            self._thread.join()
        pynvml.nvmlShutdown()

    def _loop(self) -> None:
        while not self._stop.is_set():
            for i, h in self._handles.items():
                util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                mem_mb = pynvml.nvmlDeviceGetMemoryInfo(h).used / 2**20
                power_w = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                self._samples[i].append((util, mem_mb, power_w))
            time.sleep(self.interval_s)

    def summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for i, samples in self._samples.items():
            if not samples:
                continue
            arr = np.asarray(samples)
            out[f"gpu{i}"] = {
                "util_mean_pct": float(arr[:, 0].mean()),
                "util_max_pct": float(arr[:, 0].max()),
                "mem_used_max_mb": float(arr[:, 1].max()),
                "power_mean_w": float(arr[:, 2].mean()),
                "num_samples": len(samples),
            }
        return out
