"""Where does a decode step's time go?  GPU-busy time vs wall-clock time per step.

Method (per benchmark point):
  1. Wall time: un-profiled runs (the profiler itself adds CPU overhead), median of 3.
  2. GPU-busy time: sum of all kernel/memcpy durations recorded by torch.profiler (CUPTI).
     One profiled run with max_new_tokens=1 (prefill only) and one with the full length;
     the difference divided by (ol - 1) is the GPU work of one decode step.
  3. GPU idle % = 1 - busy / wall. High idle => the GPU waits on the CPU (launch-bound).
"""
from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

from infer_lab.backends import create_backend
from infer_lab.bench.workload import make_prompt_ids
from infer_lab.config import BenchConfig

log = logging.getLogger(__name__)


def _self_device_us(evt) -> float:
    v = getattr(evt, "self_device_time_total", None)  # torch >= 2.4
    return v if v is not None else evt.self_cuda_time_total


def _gpu_work(prof) -> tuple[float, int]:
    """(total GPU-busy ms, number of GPU kernels + memcpys) in a profiled region."""
    busy_us, n = 0.0, 0
    for evt in prof.key_averages():
        if evt.device_type == DeviceType.CUDA:
            busy_us += _self_device_us(evt)
            n += evt.count
    return busy_us / 1000.0, n


def _top_ops_table(prof, rows: int = 15) -> str:
    ka = prof.key_averages()
    for key in ("self_device_time_total", "self_cuda_time_total"):
        try:
            return ka.table(sort_by=key, row_limit=rows, max_name_column_width=70)
        except (AttributeError, KeyError):
            continue
    return ka.table(row_limit=rows)


def _profiled(backend, prompts, max_new_tokens: int):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        backend.generate(prompts, max_new_tokens, ignore_eos=True)
    return prof


def profile_point(backend, tok, bs: int, pl: int, ol: int, seed: int,
                  out_dir: Path, trace: bool) -> dict:
    if ol < 2:
        raise ValueError("output_len must be >= 2 to measure decode steps")
    prompts = make_prompt_ids(tok, bs, pl, seed=seed)
    backend.generate(prompts, ol)  # warmup

    prefill_ms, decode_ms = [], []
    for _ in range(3):
        backend.generate(prompts, ol)
        x = backend.extra_metrics()
        prefill_ms.append(x["prefill_ms"])
        decode_ms.append(x["decode_ms"])
    wall_prefill = statistics.median(prefill_ms)
    wall_step = statistics.median(decode_ms) / (ol - 1)

    p_prefill = _profiled(backend, prompts, 1)
    p_full = _profiled(backend, prompts, ol)
    busy_prefill, n_prefill = _gpu_work(p_prefill)
    busy_full, n_full = _gpu_work(p_full)
    busy_step = (busy_full - busy_prefill) / (ol - 1)
    kernels_step = (n_full - n_prefill) / (ol - 1)
    idle_pct = 100.0 * max(0.0, 1.0 - busy_step / wall_step)

    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"bs{bs}_pl{pl}_ol{ol}"
    (out_dir / f"top_ops_{name}.txt").write_text(_top_ops_table(p_full))
    if trace:
        p_full.export_chrome_trace(str(out_dir / f"trace_{name}.json"))

    return {
        "batch_size": bs, "prompt_len": pl, "output_len": ol,
        "prefill_wall_ms": wall_prefill,
        "prefill_gpu_busy_ms": busy_prefill,
        "decode_wall_ms_per_step": wall_step,
        "decode_gpu_busy_ms_per_step": busy_step,
        "decode_gpu_idle_pct": idle_pct,
        "kernels_per_decode_step": kernels_step,
        "avg_kernel_us": 1000.0 * busy_step / kernels_step if kernels_step else float("nan"),
        "verdict": "launch-bound (GPU waits on CPU)" if idle_pct > 50 else "GPU-bound",
    }


def run_profile(cfg: BenchConfig, batch_sizes: list[int], prompt_lens: list[int],
                output_len: int, trace: bool) -> list[dict]:
    backend = create_backend("hf_torch", model_name=cfg.model.name,
                             dtype=cfg.model.dtype, device=cfg.device)
    backend.load()
    tok = backend.tokenizer()
    out_dir = Path(cfg.results_dir) / "profile"
    results = []
    for bs in batch_sizes:
        for pl in prompt_lens:
            log.info("profiling bs=%d pl=%d ol=%d", bs, pl, output_len)
            r = profile_point(backend, tok, bs, pl, output_len, cfg.seed, out_dir, trace)
            results.append(r)
            with (out_dir / "profile.jsonl").open("a") as f:
                f.write(json.dumps(r) + "\n")
    backend.close()

    print(f"\n{'bs':>3} {'pl':>5} | {'prefill wall':>12} {'GPU busy':>9} | "
          f"{'decode wall/step':>16} {'GPU busy/step':>13} {'GPU idle':>8} "
          f"{'kernels/step':>12} {'avg kernel':>10} | verdict")
    for r in results:
        print(f"{r['batch_size']:>3} {r['prompt_len']:>5} | "
              f"{r['prefill_wall_ms']:>10.1f}ms {r['prefill_gpu_busy_ms']:>7.1f}ms | "
              f"{r['decode_wall_ms_per_step']:>14.2f}ms {r['decode_gpu_busy_ms_per_step']:>11.2f}ms "
              f"{r['decode_gpu_idle_pct']:>7.0f}% {r['kernels_per_decode_step']:>12.0f} "
              f"{r['avg_kernel_us']:>8.1f}us | {r['verdict']}")
    print(f"\nTop GPU ops per point: {out_dir}/top_ops_*.txt"
          + (f"   Traces (open in https://ui.perfetto.dev): {out_dir}/trace_*.json" if trace else ""))
    return results
