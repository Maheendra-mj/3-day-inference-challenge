"""Correctness + roofline-style benchmarks for the kernel lab, against PyTorch references."""
from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path

import torch

from infer_lab.kernels import load_ext

log = logging.getLogger(__name__)

T4_PEAK_GBPS = 320.0
T4_PEAK_FP32_TFLOPS = 8.1


def time_cuda_ms(fn, warmup: int = 10, iters: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def bench_vector_add(n: int = 1 << 26) -> list[dict]:
    ext = load_ext("vector_add")
    a, b = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
    ref = a + b
    bytes_moved = 3 * n * 4
    rows = []
    for label, fn in [
        ("torch", lambda: a + b),
        ("scalar", lambda: ext.vector_add(a, b, False)),
        ("vec4", lambda: ext.vector_add(a, b, True)),
    ]:
        torch.testing.assert_close(fn(), ref)
        ms = time_cuda_ms(fn)
        gbps = bytes_moved / (ms * 1e-3) / 1e9
        rows.append(dict(kernel="vector_add", variant=label, n=n, ms=ms, gbps=gbps,
                         pct_peak=100 * gbps / T4_PEAK_GBPS))
    return rows


def bench_matmul(sizes: tuple[int, ...] = (512, 1024, 2048)) -> list[dict]:
    ext = load_ext("matmul")
    rows = []
    for s in sizes:
        A, B = torch.randn(s, s, device="cuda"), torch.randn(s, s, device="cuda")
        ref = A @ B
        flops = 2 * s**3
        for label, fn in [
            ("torch_cublas", lambda: A @ B),
            ("naive", lambda: ext.matmul(A, B, "naive")),
            ("tiled16", lambda: ext.matmul(A, B, "tiled")),
        ]:
            torch.testing.assert_close(fn(), ref, rtol=1e-3, atol=1e-2)
            ms = time_cuda_ms(fn, warmup=3, iters=20)
            tflops = flops / (ms * 1e-3) / 1e12
            rows.append(dict(kernel="matmul", variant=label, n=s, ms=ms, tflops=tflops,
                             pct_peak=100 * tflops / T4_PEAK_FP32_TFLOPS))
    return rows


def run_all(results_dir: Path) -> list[dict]:
    if not torch.cuda.is_available():
        raise SystemExit("kernel lab needs a CUDA GPU")
    rows = bench_vector_add() + bench_matmul()
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "kernels.json").write_text(json.dumps(rows, indent=2))
    for r in rows:
        perf = f"{r['gbps']:7.1f} GB/s" if "gbps" in r else f"{r['tflops']:6.3f} TFLOPS"
        print(f"{r['kernel']:<11} {r['variant']:<13} n={r['n']:<9} {r['ms']:8.3f} ms  "
              f"{perf}  ({r['pct_peak']:5.1f}% of T4 peak)")
    return rows
