"""vLLM offline engine behind the Backend API: PagedAttention, continuous batching and
CUDA graphs, with no HTTP server in the way.

We drive the engine loop ourselves (add_request + step) instead of LLM.generate(), because
generate() only returns finished requests and so can't timestamp each request's first
token. All prompts are submitted at t=0: this measures offline/batch behaviour, directly
comparable to the hf_torch sweep. Serving under load is vllm_server's job.
"""
from __future__ import annotations

import gc
import time

from infer_lab.backends.base import Backend
from infer_lab.bench.metrics import RequestRecord


def _kv_capacity(engine) -> dict:
    """KV-cache size the engine allocated. May be absent when the engine core runs in a
    separate process (vLLM V1); vLLM also logs it at startup ("KV cache size")."""
    cc = getattr(getattr(engine, "vllm_config", None), "cache_config", None) \
        or getattr(engine, "cache_config", None)
    blocks, block_size = getattr(cc, "num_gpu_blocks", None), getattr(cc, "block_size", None)
    if blocks and block_size:
        return {"kv_gpu_blocks": blocks, "kv_block_size": block_size,
                "kv_capacity_tokens": blocks * block_size}
    return {}


class VLLMOfflineBackend(Backend):
    name = "vllm_offline"

    def __init__(
        self,
        model_name: str,
        dtype: str = "float16",
        device: str = "cuda:0",  # unused: vLLM places itself; use CUDA_VISIBLE_DEVICES
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 2048,
        max_num_seqs: int = 256,
        enforce_eager: bool = False,  # True disables CUDA graphs: isolates their effect
        enable_prefix_caching: bool = False,  # off: our prompts must not share KV
        seed: int = 0,
        **engine_kwargs,
    ):
        self.engine_args = dict(
            model=model_name, dtype=dtype, tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len,
            max_num_seqs=max_num_seqs, enforce_eager=enforce_eager,
            enable_prefix_caching=enable_prefix_caching, seed=seed, **engine_kwargs,
        )
        self._batch = 0
        self._extra: dict = {}

    def load(self) -> None:
        from vllm import LLM

        self.llm = LLM(**self.engine_args)
        self._engine = self.llm.llm_engine
        self._tok = self.llm.get_tokenizer()
        self._kv = _kv_capacity(self._engine)

    def tokenizer(self):
        return self._tok

    def extra_metrics(self) -> dict:
        return dict(self._extra)

    def generate(
        self, prompts: list[list[int]], max_new_tokens: int, ignore_eos: bool = True
    ) -> list[RequestRecord]:
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        params = SamplingParams(
            temperature=0.0, max_tokens=max_new_tokens, ignore_eos=ignore_eos,
            output_kind=RequestOutputKind.DELTA,  # only new tokens per step: cheap to count
        )
        self._batch += 1  # request ids must be unique across calls
        ids = [f"{self._batch}-{i}" for i in range(len(prompts))]
        first: dict[str, float] = {}
        end: dict[str, float] = {}
        n_tok = dict.fromkeys(ids, 0)

        t_submit = time.perf_counter()
        for rid, p in zip(ids, prompts):
            self._engine.add_request(rid, {"prompt_token_ids": p}, params)
        steps = 0
        while self._engine.has_unfinished_requests():
            outputs = self._engine.step()
            now = time.perf_counter()
            steps += 1
            for out in outputs:
                n = len(out.outputs[0].token_ids)
                if n and out.request_id not in first:
                    first[out.request_id] = now
                n_tok[out.request_id] += n
                if out.finished:
                    end[out.request_id] = now

        self._extra = {"engine_steps": steps, **self._kv}
        return [
            RequestRecord(i, len(p), n_tok[rid], t_submit, first[rid], end[rid])
            for i, (rid, p) in enumerate(zip(ids, prompts))
        ]

    def close(self) -> None:
        del self._engine, self.llm
        gc.collect()
