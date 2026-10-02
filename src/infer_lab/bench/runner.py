"""Offline sweep: (batch_size x prompt_len x output_len) for one backend."""
from __future__ import annotations

import itertools
import json
import logging
import time
from pathlib import Path

from infer_lab.backends import create_backend
from infer_lab.bench.metrics import summarize
from infer_lab.bench.workload import make_prompt_ids
from infer_lab.config import BenchConfig
from infer_lab.storage.results_db import ResultsDB
from infer_lab.telemetry.gpu_monitor import GPUMonitor

log = logging.getLogger(__name__)


def _gpu_index(device: str) -> int:
    return int(device.split(":")[1]) if ":" in device else 0


def run_sweep(cfg: BenchConfig, backend_name: str, tag: str | None = None) -> list[dict]:
    backend = create_backend(
        backend_name, model_name=cfg.model.name, dtype=cfg.model.dtype, device=cfg.device
    )
    log.info("loading %s on %s", cfg.model.name, backend_name)
    backend.load()
    tok = backend.tokenizer()

    db = ResultsDB(Path(cfg.results_dir) / "bench.sqlite")
    jsonl = Path(cfg.results_dir) / "bench.jsonl"
    s = cfg.sweep
    rows: list[dict] = []

    for bs, pl, ol in itertools.product(s.batch_sizes, s.prompt_lens, s.output_lens):
        prompts = make_prompt_ids(tok, bs, pl, seed=cfg.seed)
        try:
            for _ in range(s.warmup_iters):
                backend.generate(prompts, ol, cfg.ignore_eos)

            records, wall, extras = [], 0.0, []
            with GPUMonitor([_gpu_index(cfg.device)]) as mon:
                for _ in range(s.repeat):
                    t0 = time.perf_counter()
                    records += backend.generate(prompts, ol, cfg.ignore_eos)
                    wall += time.perf_counter() - t0
                    extras.append(backend.extra_metrics())
        except RuntimeError as e:  # torch.OutOfMemoryError subclasses RuntimeError
            if "out of memory" not in str(e).lower():
                raise
            log.warning("OOM at bs=%d pl=%d ol=%d, skipping", bs, pl, ol)
            _free_cuda()
            continue

        metrics = summarize(records, wall)
        if extras and extras[0]:
            metrics.update({k: max(e[k] for e in extras) for k in extras[0]})
        gpu = mon.summary()
        row = dict(backend=backend_name, model=cfg.model.name, batch_size=bs,
                   prompt_len=pl, output_len=ol, params=cfg.to_dict(),
                   metrics=metrics, gpu=gpu, tag=tag)
        db.insert(**row)
        with jsonl.open("a") as f:
            f.write(json.dumps(row) + "\n")
        rows.append(row)
        log.info(
            "bs=%-3d pl=%-5d ol=%-4d  tok/s=%8.1f  ttft_p50=%7.1fms  tpot_p50=%6.2fms",
            bs, pl, ol, metrics["output_tokens_per_s"],
            metrics["ttft_p50_ms"], metrics.get("tpot_p50_ms", float("nan")),
        )

    backend.close()
    return rows


def _free_cuda() -> None:
    import gc

    import torch
    gc.collect()
    torch.cuda.empty_cache()
