# LLM Inference Lab — Kaggle T4 ×2

One benchmark harness, several runtimes (PyTorch, vLLM, ONNX Runtime/TensorRT), one
results database. Every number is produced the same way: same workload, same metric
definitions, same GPU telemetry.

## Final architecture

```
                         configs/*.yaml  (model, sweep, serving, load profile)
                                   │
 ┌─────────────────────────────────▼─────────────────────────────────────────┐
 │  COMMON HARNESS  (src/infer_lab)                                          │
 │  workload.py   exact-length token prompts · Poisson arrivals               │
 │  metrics.py    TTFT · TPOT · E2E · p50/p90/p99 · tok/s · req/s             │
 │  gpu_monitor   NVML sampler: util · VRAM · power (per GPU)                 │
 │  results_db    SQLite + JSONL   ──►  report / plots                        │
 └──────┬─────────────────┬──────────────────────┬─────────────────┬─────────┘
        │ Backend ABC     │                      │                 │
 ┌──────▼──────┐  ┌───────▼────────┐   ┌─────────▼────────┐  ┌─────▼──────────────┐
 │ hf_torch    │  │ ONNX export    │   │ vLLM             │  │ Continuous-batch   │
 │ eager, sdpa │  │  └► ORT-CUDA   │   │  offline (LLM)   │  │ SIMULATOR          │
 │ static batch│  │  └► TensorRT   │   │  server (OpenAI) │  │ calibrated from    │
 │ (Day 1)     │  │     fp16 engine│   │  (Day 2)         │  │ hf/vLLM step costs │
 └──────┬──────┘  │  (Day 3)       │   └───┬──────────┬───┘  └────────────────────┘
        │         └───────┬────────┘       │          │
        │                 │        TP=2 (1 server)   2×TP=1 replicas + router
        │                 │        all-reduce/PCIe   no comms, 2× capacity
        └─────────────────┴────────────┬───┴──────────┘
                                       │                ┌──────────────────────┐
                              CUDA 12 · sm_75 · fp16    │ KV-CACHE OBSERVATORY │
                         ┌─────────────┴────────────┐   │ scrape vLLM /metrics │
                         │  T4 #0 16GB  │  T4 #1 16GB│   │ kv usage %, running, │
                         └──────────── PCIe ─────────┘   │ waiting, preemptions │
                                       ▲                │ + analytic KV model  │
 ┌─────────────────────────────────────┴────────────┐   └──────────────────────┘
 │ CUDA KERNEL LAB  vector_add → matmul → softmax →  │
 │ layernorm → attention   (validated vs torch,      │
 │ reported as % of T4 roofline)                     │
 └───────────────────────────────────────────────────┘
```

### What changed from the first draft, and why

| Draft | Final | Reason |
|---|---|---|
| Benchmark engine sits after multi-GPU | A common harness wraps **every** backend | Comparisons only mean something if all backends share the same workload, metrics and telemetry. |
| ONNX → TensorRT as a full serving runtime | ONNX → **ORT-CUDA and TensorRT fp16 engine**, benchmarked on prefill/forward latency plus a greedy decode loop | TensorRT-LLM is too heavy to build inside a Kaggle session. Plain TRT shows the graph-compiler gains honestly. |
| "Tensor Parallel / Replication" as one box | **Two experiments**: TP=2 vs 2× TP=1 replicas behind a router | T4s on Kaggle talk over PCIe (no NVLink). That trade-off is the main multi-GPU result. |
| Simulator and real test kept separate | The simulator is **calibrated** from measured prefill/decode costs, then checked against vLLM | This turns the simulator into a validated model instead of a toy. |
| KV observatory shows numbers only | vLLM Prometheus metrics **plus an analytic model** | KV bytes/token = `2 · layers · kv_heads · head_dim · 2B`. For Qwen2.5-0.5B: 2·24·2·64·2 = 12 KiB/token. Predict capacity, then measure it. |

### T4 constraints (these shape every design choice)
- **sm_75 (Turing)**: fp16 only (no bf16), no FlashAttention-2, no FP8. Always use `dtype=float16`.
- **16 GB per GPU, PCIe interconnect**: TP all-reduce is relatively expensive, so replication usually wins on throughput.
- **Roofline**: about 320 GB/s DRAM, 8.1 TFLOPS fp32, 65 TFLOPS fp16 tensor cores. Decode is bandwidth-bound.
- **Kaggle**: 12 h sessions, about 30 GPU-h/week, the VM is wiped on exit. Code lives in git and results go to `/kaggle/working/results`.

**Models**: `Qwen/Qwen2.5-0.5B-Instruct` for all single-GPU work. `Qwen/Qwen2.5-1.5B-Instruct` for the TP experiments. Both have 2 KV heads, so they split cleanly with TP=2.

## Repo layout
```
configs/                 bench.yaml, smoke.yaml  (+ serving / load profiles on Day 2)
src/infer_lab/
  config.py              typed YAML config
  cli.py                 infer-lab {env, bench, kernels, report}
  bench/                 metrics, workload, runner
  backends/              base (ABC), hf_torch   (+ vllm_offline, vllm_server, ort, trt)
  telemetry/             gpu_monitor (NVML)      (+ vllm_metrics scraper)
  storage/               results_db (SQLite)
  kernels/               csrc/*.cu + JIT loader + benchmarks
  sim/                   (Day 2) continuous batching simulator
tests/                   CPU tests run anywhere; GPU tests auto-skip without CUDA
scripts/kaggle_bootstrap.sh
```

## Kaggle workflow
1. Develop locally, push to GitHub.
2. In a Kaggle notebook (Settings → Accelerator **GPU T4 x2**, Internet **On**):
   ```
   !git clone https://github.com/<you>/inference-lab.git
   %cd inference-lab
   !bash scripts/kaggle_bootstrap.sh          # or: bash scripts/kaggle_bootstrap.sh vllm
   !infer-lab bench --config configs/smoke.yaml --tag smoke
   !infer-lab report
   ```
3. Before the session ends, save `results/` (it is in `/kaggle/working`, so "Save Version" keeps it).

Use separate notebooks for the **vLLM** stack and the **TensorRT** stack. vLLM installs its own torch build.

---

## 3-day plan

### Day 1: foundation, PyTorch baseline, kernel lab stage 1  ✅ code in repo
| Block | Work | Done when |
|---|---|---|
| 1 h | Repo, harness, `infer-lab env`, tests | `pytest` is green; `env` shows 2× T4 sm_75 |
| 3 h | `hf_torch` backend with explicit prefill/decode loop, sweep runner, SQLite | `bench --config configs/bench.yaml` fills the DB; `report` prints a table |
| 1 h | Analyse the baseline | Chart of tok/s vs batch size; identify the launch-bound, compute-bound and attention-heavy regimes (see results below) |
| 3 h | Kernel lab: vector_add (scalar vs vec4), matmul (naive vs tiled) | vector_add near the practical ceiling of about 245 GB/s; tiled faster than naive; all correctness checks pass |

#### Day 1 results (Kaggle T4 ×2, torch 2.10 / cu128, transformers 5.0, Qwen2.5-0.5B fp16)
- **Eager decode is launch-bound, not bandwidth-bound.** TPOT is about 29 ms from bs=1 to bs=32 (at pl=128). Reading the 0.99 GB of weights takes about 3–4 ms at about 250 GB/s, so the GPU is idle about 85% of each step while Python/HF launches kernels. Throughput therefore scales almost linearly with batch size (34 → 990 tok/s). CUDA graphs (vLLM, Day 2) remove this floor.
- **Prefill is compute-bound.** TTFT grows linearly with total prompt tokens: 2k → 175 ms, 8k → 725 ms, 16k → 1625 ms, about 0.1 ms/token or about 10 effective TFLOPS fp16. At an equal 2k tokens, 4×512 costs more than 16×128 because attention is quadratic in sequence length.
- **bs=32 × pl=512 leaves the launch floor.** TPOT doubles to 60 ms and throughput falls to 443 tok/s: per-step attention/KV work now exceeds launch overhead. Confirm with `torch.profiler`.
- **Static batching shows up in the data.** TTFT p50 ≈ p99 and E2E is identical for every request in a batch, because everyone waits for the whole batch.
- **Kernels.** vector_add: torch 245 / vec4 238 / scalar 226 GB/s. About 77% of the 320 GB/s spec is the practical ceiling with ECC on. matmul: tiled16 is 1.5–1.7× faster than naive but only 0.6–0.9 TFLOPS vs cuBLAS 3.5–5.9. It is limited by shared-memory traffic (1 output per thread). Next step: register blocking.

### Day 2: vLLM, KV-Cache Observatory, multi-GPU, continuous batching
| Block | Work | Done when |
|---|---|---|
| 2 h | `vllm_offline` backend (`LLM.generate`, `ignore_eos`, `TokensPrompt`) | The same sweep as Day 1, now on vLLM, gives a direct comparison table |
| 2 h | `vllm_server` backend: launch the OpenAI server, async streaming client, Poisson load at N req/s | TTFT/TPOT/p99 vs request rate; the saturation knee is visible |
| 1.5 h | KV observatory: scrape `/metrics` (kv-cache usage, running/waiting, preemptions) every 0.5 s during load, sweep context length × concurrency, compare to the analytic model | Plot of KV usage over time; first preemption point matches the predicted capacity within about 10% |
| 1.5 h | Multi-GPU: TP=2 on 1.5B vs 2 replicas (one per GPU) behind a round-robin router | Throughput and p99 for both at equal load, with a written conclusion on PCIe all-reduce cost |
| 1 h | Continuous batching simulator: discrete-event, static vs continuous, calibrated from measured step costs | Simulated vs real vLLM throughput within about 15% |

### Day 3: ONNX → TensorRT, kernel lab stage 2, productionising
| Block | Work | Done when |
|---|---|---|
| 3 h | Export to ONNX (with KV-cache inputs, dynamic axes), ORT-CUDA backend, TensorRT fp16 engine with optimisation profiles, decode loop | Prefill latency for torch vs ORT vs TRT at several sequence lengths; logits match within fp16 tolerance |
| 2 h | Kernels: softmax (warp shuffle), layernorm (Welford), naive attention, then a fused/tiled attention | Each passes its torch reference; bandwidth reported as % of peak |
| 2 h | Visualisation: `report --plots` builds all charts plus a single HTML/Markdown report from SQLite | One command regenerates the full results page |
| 1 h | Hardening: config-driven runs, CI-safe tests, README results section, reproducibility notes (versions, seeds) | Clean clone → bootstrap → smoke run works on a fresh Kaggle session |
