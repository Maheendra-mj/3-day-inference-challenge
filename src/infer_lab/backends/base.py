"""Backend contract every runtime (PyTorch, vLLM, ORT/TensorRT) implements.

The benchmark engine only talks to this interface, so all backends are measured with
the same workload, the same metric definitions and the same telemetry.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from infer_lab.bench.metrics import RequestRecord


class Backend(ABC):
    name: str = "base"

    @abstractmethod
    def load(self) -> None:
        """Load weights / build engine. Not included in any timing."""

    @abstractmethod
    def generate(
        self, prompts: list[list[int]], max_new_tokens: int, ignore_eos: bool = True
    ) -> list[RequestRecord]:
        """Run all prompts to completion; return one record per prompt (same order)."""

    def tokenizer(self):
        """HF tokenizer used to build workloads."""
        raise NotImplementedError

    def extra_metrics(self) -> dict:
        """Backend-specific numbers for the last generate() call (e.g. peak torch memory)."""
        return {}

    def close(self) -> None:
        pass
