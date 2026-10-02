"""Typed configuration loaded from YAML."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dtype: str = "float16"


@dataclass
class SweepConfig:
    batch_sizes: list[int] = field(default_factory=lambda: [1, 4, 16])
    prompt_lens: list[int] = field(default_factory=lambda: [128])
    output_lens: list[int] = field(default_factory=lambda: [128])
    warmup_iters: int = 1
    repeat: int = 3


@dataclass
class BenchConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    device: str = "cuda:0"
    results_dir: str = "results"
    seed: int = 0
    ignore_eos: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BenchConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls(
            model=ModelConfig(**raw.pop("model", {})),
            sweep=SweepConfig(**raw.pop("sweep", {})),
            **raw,
        )

    def to_dict(self) -> dict:
        return asdict(self)
