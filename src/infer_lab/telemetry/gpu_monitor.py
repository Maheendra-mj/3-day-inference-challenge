"""Background NVML sampler: GPU utilisation, VRAM, power and SM clock during a benchmark.

Caveat: NVML "utilisation" is the % of time *any* kernel was running, not how much of the
GPU it used. A launch-bound workload (many tiny kernels with idle gaps) can still read
~100%. Power draw and SM clock are better "how hard is it working" signals; use
`infer-lab profile` for the real GPU-busy time.

NVML indices follow PCI bus order; CUDA indices follow CUDA_DEVICE_ORDER. On Kaggle's
T4x2 they coincide, but set CUDA_DEVICE_ORDER=PCI_BUS_ID elsewhere to be safe.
"""
from __future__ import annotations

import csv
import threading
import time
from pathlib import Path

import numpy as np

try:
    import pynvml
except ImportError:  # local dev machine without NVIDIA driver
    pynvml = None

_FIELDS = ("t_s", "gpu", "util_pct", "mem_used_mb", "power_w", "sm_clock_mhz")


class GPUMonitor:
    def __init__(self, device_indices: list[int] | None = None, interval_s: float = 0.1):
        self.interval_s = interval_s
        self.device_indices = device_indices
        self._samples: dict[int, list[tuple[float, float, float, float, float]]] = {}
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
        self._t0 = time.perf_counter()
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
            t = time.perf_counter() - self._t0
            for i, h in self._handles.items():
                util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                mem_mb = pynvml.nvmlDeviceGetMemoryInfo(h).used / 2**20
                power_w = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                clock = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
                self._samples[i].append((t, util, mem_mb, power_w, clock))
            time.sleep(self.interval_s)

    def summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for i, samples in self._samples.items():
            if not samples:
                continue
            arr = np.asarray(samples)
            out[f"gpu{i}"] = {
                "util_mean_pct": float(arr[:, 1].mean()),
                "util_max_pct": float(arr[:, 1].max()),
                "mem_used_max_mb": float(arr[:, 2].max()),
                "power_mean_w": float(arr[:, 3].mean()),
                "power_max_w": float(arr[:, 3].max()),
                "sm_clock_mean_mhz": float(arr[:, 4].mean()),
                "sm_clock_min_mhz": float(arr[:, 4].min()),
                "num_samples": len(samples),
            }
        return out

    def write_timeseries(self, path: str | Path) -> None:
        """One CSV row per (sample, gpu): the timeline behind summary()."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(_FIELDS)
            for i, samples in self._samples.items():
                for t, util, mem, power, clock in samples:
                    w.writerow((f"{t:.3f}", i, util, f"{mem:.0f}", f"{power:.1f}", clock))
