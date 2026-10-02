"""Synthetic workloads with exact token lengths.

Prompts are returned as token-id lists (not strings) so every backend sees exactly
`prompt_len` tokens. Each prompt is a different shuffle of the seed text, which keeps
prefix caching from silently inflating results.
"""
from __future__ import annotations

import random

import numpy as np

_SEED_TEXT = (
    "Large language model inference is dominated by two phases. The prefill phase "
    "processes the whole prompt in parallel and is compute bound, while the decode "
    "phase generates one token at a time and is memory bandwidth bound because every "
    "step must stream the model weights and the key value cache from device memory. "
    "Batching amortises weight reads across many sequences, paged attention removes "
    "fragmentation in the cache, and continuous batching admits new requests as soon "
    "as old ones finish instead of waiting for the slowest member of a static batch. "
    "Tensor parallelism splits each layer across devices at the cost of an all reduce "
    "per layer, whereas replication runs independent copies and needs no communication."
)


def make_prompt_ids(tokenizer, n: int, prompt_len: int, seed: int = 0) -> list[list[int]]:
    rng = random.Random(seed)
    words = _SEED_TEXT.split()
    prompts: list[list[int]] = []
    for _ in range(n):
        ids: list[int] = []
        while len(ids) < prompt_len:
            shuffled = words[:]
            rng.shuffle(shuffled)
            ids.extend(tokenizer(" ".join(shuffled), add_special_tokens=False).input_ids)
        prompts.append(ids[:prompt_len])
    return prompts


def poisson_arrivals(n: int, rate_rps: float, seed: int = 0) -> np.ndarray:
    """Arrival offsets (seconds from t=0) for an open-loop load test. rate=inf -> all at 0."""
    if np.isinf(rate_rps):
        return np.zeros(n)
    gaps = np.random.default_rng(seed).exponential(1.0 / rate_rps, size=n)
    return np.cumsum(gaps) - gaps[0]
