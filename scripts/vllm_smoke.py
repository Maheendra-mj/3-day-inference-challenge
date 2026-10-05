"""Is vLLM usable on this GPU?  Run with the venv's python:

    /tmp/vllm-env/bin/python scripts/vllm_smoke.py

Look in vLLM's log for the attention backend it picked (T4 = sm_75 has no
FlashAttention-2) and the "KV cache" capacity line; the script prints versions, a sample
and decode throughput at batch 32.
"""
import sys
import time


def main() -> None:
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
    llm = LLM(model=model, dtype="float16", max_model_len=1024, gpu_memory_utilization=0.6)
    sp = SamplingParams(temperature=0.0, max_tokens=128, ignore_eos=True)
    prompts = ["Explain what the KV cache does during LLM decoding."] * 32

    llm.generate(prompts[:1], sp, use_tqdm=False)  # warmup
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    dt = time.perf_counter() - t0
    n = sum(len(o.outputs[0].token_ids) for o in outs)

    print("\n================ vLLM smoke test ================")
    print(f"vllm {vllm.__version__} | torch {torch.__version__} | {torch.cuda.get_device_name(0)}")
    print(f"sample: {outs[0].outputs[0].text[:160]!r}")
    print(f"bs=32 x 128 new tokens: {n / dt:.0f} output tok/s  "
          f"(HF eager baseline at bs=32, pl=128: ~900-990 tok/s)")


if __name__ == "__main__":  # required: vLLM may spawn worker processes
    main()
