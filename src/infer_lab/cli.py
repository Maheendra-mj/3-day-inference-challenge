"""infer-lab command line.

  infer-lab env                                   # GPU / library report
  infer-lab bench --config configs/smoke.yaml     # offline sweep, PyTorch baseline
  infer-lab profile --bs 32 --pl 128 512          # GPU-busy vs wall time per decode step
  infer-lab kernels                               # CUDA kernel lab benchmarks
  infer-lab report                                # print results table
"""
from __future__ import annotations

import argparse
import logging
import platform
from pathlib import Path


def cmd_env(_: argparse.Namespace) -> None:
    import importlib

    print(f"python      {platform.python_version()}")
    for mod in ("torch", "transformers", "vllm", "onnxruntime", "tensorrt", "triton"):
        try:
            print(f"{mod:<11} {importlib.import_module(mod).__version__}")
        except ImportError:
            print(f"{mod:<11} -")
    from infer_lab.telemetry.env import env_info
    info = env_info()
    print(f"cpu         {info['cpu']} ({info['cpu_count']} vCPU)")
    print(f"driver      {info.get('driver', '-')}")

    import torch
    if not torch.cuda.is_available():
        print("CUDA        not available")
        return
    print(f"cuda        {torch.version.cuda}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"gpu{i}        {p.name}  sm_{p.major}{p.minor}  {p.total_memory / 2**30:.1f} GiB"
              f"  {p.multi_processor_count} SMs")
    if torch.cuda.device_count() > 1:
        print(f"p2p 0<->1   {torch.cuda.can_device_access_peer(0, 1)}")


def cmd_bench(args: argparse.Namespace) -> None:
    from infer_lab.bench.runner import run_sweep
    from infer_lab.config import BenchConfig

    cfg = BenchConfig.from_yaml(args.config)
    if args.results_dir:
        cfg.results_dir = args.results_dir
    run_sweep(cfg, args.backend, tag=args.tag)


def cmd_profile(args: argparse.Namespace) -> None:
    from infer_lab.bench.profile import run_profile
    from infer_lab.config import BenchConfig

    cfg = BenchConfig.from_yaml(args.config)
    if args.results_dir:
        cfg.results_dir = args.results_dir
    run_profile(cfg, args.bs, args.pl, args.ol, args.trace)


def cmd_kernels(args: argparse.Namespace) -> None:
    from infer_lab.kernels.bench_kernels import run_all

    run_all(Path(args.results_dir))


def cmd_report(args: argparse.Namespace) -> None:
    import pandas as pd

    from infer_lab.storage.results_db import ResultsDB

    df = ResultsDB(Path(args.results_dir) / "bench.sqlite").to_dataframe()
    if df.empty:
        print("no results yet")
        return
    cols = ["id", "tag", "backend", "batch_size", "prompt_len", "output_len",
            "output_tokens_per_s", "ttft_p50_ms", "ttft_p99_ms", "tpot_p50_ms", "e2e_p99_ms",
            "torch_peak_mem_mb", "gpu.gpu0.util_mean_pct", "gpu.gpu0.power_mean_w"]
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(df[[c for c in cols if c in df.columns]].round(2).to_string(index=False))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "huggingface_hub", "numexpr"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    p = argparse.ArgumentParser(prog="infer-lab")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("env").set_defaults(fn=cmd_env)

    b = sub.add_parser("bench")
    b.add_argument("--config", default="configs/bench.yaml")
    b.add_argument("--backend", default="hf_torch")
    b.add_argument("--tag")
    b.add_argument("--results-dir")
    b.set_defaults(fn=cmd_bench)

    pr = sub.add_parser("profile", help="GPU-busy vs wall time per decode step (torch.profiler)")
    pr.add_argument("--config", default="configs/bench.yaml")
    pr.add_argument("--bs", type=int, nargs="+", default=[32])
    pr.add_argument("--pl", type=int, nargs="+", default=[128, 512])
    pr.add_argument("--ol", type=int, default=32)
    pr.add_argument("--trace", action="store_true", help="also export Chrome/Perfetto traces")
    pr.add_argument("--results-dir")
    pr.set_defaults(fn=cmd_profile)

    k = sub.add_parser("kernels")
    k.add_argument("--results-dir", default="results")
    k.set_defaults(fn=cmd_kernels)

    r = sub.add_parser("report")
    r.add_argument("--results-dir", default="results")
    r.set_defaults(fn=cmd_report)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
