"""PyTorch eager baseline: static batching with an explicit prefill + decode loop.

Static batching semantics are deliberate: every request in the batch finishes when the
batch finishes, which is exactly the behaviour continuous batching (vLLM) fixes.
"""
from __future__ import annotations

import inspect
import time

import torch
import transformers
from packaging.version import Version
from torch.profiler import record_function
from transformers import AutoModelForCausalLM, AutoTokenizer

from infer_lab.backends.base import Backend
from infer_lab.bench.metrics import RequestRecord

_DTYPES = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
# `torch_dtype` was renamed to `dtype` in transformers 4.56 (old name warns in 5.x).
_DTYPE_KW = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"


class HFTorchBackend(Backend):
    name = "hf_torch"

    def __init__(
        self,
        model_name: str,
        dtype: str = "float16",
        device: str = "cuda:0",
        attn_implementation: str = "sdpa",
    ):
        self.model_name = model_name
        self.dtype = _DTYPES[dtype]
        self.device = torch.device(device)
        self.attn_implementation = attn_implementation
        self._extra: dict = {}

    def load(self) -> None:
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self.pad_id = self._tok.pad_token_id if self._tok.pad_token_id is not None else self._tok.eos_token_id
        self.eos_id = self._tok.eos_token_id
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **{_DTYPE_KW: self.dtype},
                attn_implementation=self.attn_implementation,
            )
            .to(self.device)
            .eval()
        )
        # Only materialise last-position logits during prefill: full logits for
        # B=32, L=512 and a 152k vocab would be ~5 GB of fp16.
        params = inspect.signature(self.model.forward).parameters
        if "logits_to_keep" in params:
            self._last_logits_kw = {"logits_to_keep": 1}
        elif "num_logits_to_keep" in params:
            self._last_logits_kw = {"num_logits_to_keep": 1}
        else:
            self._last_logits_kw = {}

    def tokenizer(self):
        return self._tok

    def extra_metrics(self) -> dict:
        return dict(self._extra)

    @torch.inference_mode()
    def generate(
        self, prompts: list[list[int]], max_new_tokens: int, ignore_eos: bool = True
    ) -> list[RequestRecord]:
        B = len(prompts)
        max_len = max(len(p) for p in prompts)
        input_ids = torch.full((B, max_len), self.pad_id, dtype=torch.long)
        attn = torch.zeros((B, max_len), dtype=torch.long)
        for i, p in enumerate(prompts):  # left padding so all sequences end at the same column
            input_ids[i, max_len - len(p):] = torch.tensor(p)
            attn[i, max_len - len(p):] = 1
        input_ids, attn = input_ids.to(self.device), attn.to(self.device)
        position_ids = (attn.cumsum(-1) - 1).clamp(min=0)

        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)
        t_submit = time.perf_counter()

        # ---- prefill ----
        with record_function("prefill"):
            out = self.model(
                input_ids=input_ids, attention_mask=attn, position_ids=position_ids,
                use_cache=True, **self._last_logits_kw,
            )
        past = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(-1)
        torch.cuda.synchronize(self.device)
        t_first = time.perf_counter()

        n_out = torch.ones(B, dtype=torch.long, device=self.device)
        finished = next_tok == self.eos_id if not ignore_eos else None
        cur_pos = position_ids[:, -1:] + 1

        # ---- decode ----
        for _ in range(max_new_tokens - 1):
            if finished is not None and bool(finished.all()):
                break
            attn = torch.cat([attn, attn.new_ones((B, 1))], dim=1)
            with record_function("decode_step"):
                out = self.model(
                    input_ids=next_tok[:, None], attention_mask=attn, position_ids=cur_pos,
                    past_key_values=past, use_cache=True,
                )
            past = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(-1)
            cur_pos = cur_pos + 1
            if finished is None:
                n_out += 1
            else:
                n_out += (~finished).long()
                finished |= next_tok == self.eos_id

        torch.cuda.synchronize(self.device)
        t_end = time.perf_counter()

        self._extra = {
            "torch_peak_mem_mb": torch.cuda.max_memory_allocated(self.device) / 2**20,
            "prefill_ms": (t_first - t_submit) * 1000,
            "decode_ms": (t_end - t_first) * 1000,
        }
        counts = n_out.tolist()
        return [
            RequestRecord(i, len(p), counts[i], t_submit, t_first, t_end)
            for i, p in enumerate(prompts)
        ]

    def close(self) -> None:
        del self.model
        torch.cuda.empty_cache()
