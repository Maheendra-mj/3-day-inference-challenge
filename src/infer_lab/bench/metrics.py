"""Per-request timing records and aggregate latency/throughput metrics.

Definitions (same as vLLM's benchmark_serving so numbers are comparable):
  TTFT  = first token time - submit time
  TPOT  = (end - first token) / (output_tokens - 1)    # inter-token latency
  E2E   = end - submit
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RequestRecord:
    request_id: int
    prompt_tokens: int
    output_tokens: int
    t_submit: float
    t_first_token: float
    t_end: float

    @property
    def ttft(self) -> float:
        return self.t_first_token - self.t_submit

    @property
    def e2e(self) -> float:
        return self.t_end - self.t_submit

    @property
    def tpot(self) -> float | None:
        if self.output_tokens <= 1:
            return None
        return (self.t_end - self.t_first_token) / (self.output_tokens - 1)


def _dist(name: str, values_s: list[float]) -> dict[str, float]:
    if not values_s:
        return {}
    ms = np.asarray(values_s) * 1000.0
    return {
        f"{name}_mean_ms": float(ms.mean()),
        f"{name}_p50_ms": float(np.percentile(ms, 50)),
        f"{name}_p90_ms": float(np.percentile(ms, 90)),
        f"{name}_p99_ms": float(np.percentile(ms, 99)),
    }


def summarize(records: list[RequestRecord], wall_time_s: float) -> dict[str, float]:
    if not records:
        raise ValueError("no records to summarize")
    if wall_time_s <= 0:
        raise ValueError("wall_time_s must be positive")
    out_tok = sum(r.output_tokens for r in records)
    in_tok = sum(r.prompt_tokens for r in records)
    return {
        "num_requests": len(records),
        "wall_time_s": wall_time_s,
        "request_throughput_rps": len(records) / wall_time_s,
        "output_tokens_per_s": out_tok / wall_time_s,
        "total_tokens_per_s": (out_tok + in_tok) / wall_time_s,
        **_dist("ttft", [r.ttft for r in records]),
        **_dist("tpot", [t for r in records if (t := r.tpot) is not None]),
        **_dist("e2e", [r.e2e for r in records]),
    }
