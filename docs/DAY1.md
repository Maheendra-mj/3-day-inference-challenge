# Day 1: Foundation, PyTorch Baseline, CUDA Kernel Lab (Stage 1)

This document explains Day 1 from zero: the goal, what was built, every file and
function, the experiments that were run, what the numbers mean, and every technical
term used. Read the sections in order the first time; later, use the glossary (§10) as
a reference.

---

## 1. Goal of Day 1

> **Build a trustworthy measuring instrument, then take the first measurement.**

Days 2 and 3 compare runtimes (PyTorch vs vLLM vs ONNX/TensorRT) and setups
(1 GPU vs 2 GPUs). A comparison is only meaningful if every runtime is measured
**the same way**. So Day 1 had three goals:

| # | Goal | Why it matters |
|---|---|---|
| G1 | A **common benchmark harness**: workload generator, metric definitions, GPU telemetry, results storage, CLI | Days 2–3 plug new backends into it and get directly comparable numbers |
| G2 | A **PyTorch baseline**: the simplest honest way to run the model, measured precisely | Every later optimisation is reported as "×N faster than baseline" |
| G3 | **CUDA Kernel Lab, stage 1**: write real GPU kernels and measure them against the hardware's limits | Builds the intuition (memory-bound vs compute-bound) needed to explain every LLM number |

## 2. What was achieved

| Goal | Status | Evidence |
|---|---|---|
| G1 harness | ✅ | `pytest`: 23/23 passed on Kaggle; `bench` filled SQLite + JSONL; `report` printed the table |
| G2 baseline | ✅ | 10 benchmark points (2 smoke + 8 baseline) on Qwen2.5-0.5B fp16, T4 |
| G3 kernels | ✅ | vector_add at 74–77% of spec bandwidth (practical ceiling); tiled matmul 1.5–1.7× faster than naive; all verified against PyTorch |
| Environment facts | ✅ | 2× T4 sm_75, 14.6 GiB usable each, PCIe `PHB`, **no P2P**: this shapes Day 2's multi-GPU design |

**Five findings we can now defend with data:**

1. Eager PyTorch decode on this model is **launch-bound** (≈29 ms/step regardless of batch), not memory-bound.
2. Because of (1), batching is almost free: throughput grows ~linearly from 34 → 990 tok/s.
3. Prefill is **compute-bound**: TTFT ∝ total prompt tokens (~0.1 ms/token ≈ 10 effective TFLOPS).
4. At bs=32 × 512 tokens, real GPU work exceeds the launch floor, so TPOT doubles to 60 ms.
5. Static batching makes every request in a batch finish together (p50 = p99), which is the problem continuous batching solves.

---

## 3. Background: how an LLM generates text

```
prompt "The cat sat on the"  ──tokenizer──►  [791, 8415, 7731, 389, 279]   (token IDs)

PREFILL  (1 forward pass over ALL prompt tokens, in parallel)
   └─► logits for the last position ──argmax──► first new token "mat"   ◄── TTFT measured here
         └─► also produces the KV cache for all 5 prompt tokens

DECODE   (1 forward pass PER new token, each sees only the newest token + KV cache)
   step 1: "mat" ─► "." ;  step 2: "." ─► "<eos>" ...                    ◄── TPOT = time per step
```

- **Prefill** does a lot of math per byte of weights read: *compute-bound*.
- **Decode** does very little math per step, but must read **all weights** (and the
  KV cache) every step: *memory-bound*, unless something else (e.g. CPU kernel-launch
  overhead) is even slower. On Day 1 that is exactly what we found.

Everything in this project measures or optimises one of these two phases.

---

## 4. Folder structure

```
inference-3-days/
├── pyproject.toml              Package definition, dependencies, `infer-lab` CLI entry point
├── README.md                   Final architecture + 3-day plan + results summary
├── .gitignore                  Keeps results/, engines, ONNX files, caches out of git
├── configs/
│   ├── smoke.yaml              ~1 min sanity run (2 tiny points)
│   └── bench.yaml              Full Day-1 sweep (4 batch sizes × 2 prompt lengths)
├── docs/
│   └── DAY1.md                 ← this file
├── scripts/
│   └── kaggle_bootstrap.sh     First cell of every Kaggle session: GPU check, install, tests
├── notebooks/
│   └── test1.ipynb             The Day-1 Kaggle run (outputs are the raw evidence)
├── src/infer_lab/              The Python package ("src layout")
│   ├── __init__.py             Package version
│   ├── config.py               YAML → typed dataclasses
│   ├── cli.py                  `infer-lab env | bench | kernels | report`
│   ├── bench/                  THE BENCHMARK ENGINE
│   │   ├── metrics.py            RequestRecord + summarize() (TTFT/TPOT/E2E/percentiles)
│   │   ├── workload.py           exact-length prompts + Poisson arrivals
│   │   ├── runner.py             sweep loop: warmup → measure → store
│   │   └── profile.py            GPU-busy vs wall time per decode step (torch.profiler)
│   ├── backends/               RUNTIMES (one class per inference engine)
│   │   ├── __init__.py           create_backend() registry (lazy imports)
│   │   ├── base.py               Backend abstract interface
│   │   └── hf_torch.py           PyTorch eager baseline (manual prefill/decode loop)
│   ├── telemetry/
│   │   └── gpu_monitor.py        NVML background sampler (util, VRAM, power)
│   ├── storage/
│   │   └── results_db.py         SQLite store + flattened DataFrame view
│   └── kernels/                CUDA KERNEL LAB
│       ├── __init__.py           load_ext(): JIT-compile .cu → Python module
│       ├── bench_kernels.py      correctness + timing + % of T4 peak
│       └── csrc/
│           ├── vector_add.cu     memory-bound kernel (scalar vs float4)
│           └── matmul.cu         compute-bound kernel (naive vs shared-memory tiled)
├── tests/
│   ├── test_metrics.py         CPU-only: metric math, Poisson rate
│   └── test_kernels.py         GPU-only (auto-skipped without CUDA): kernels vs torch
└── results/  (created at runtime, git-ignored)
    ├── bench.sqlite            one row per benchmark point
    ├── bench.jsonl             same rows, one JSON per line (easy to diff/grep)
    ├── kernels.json            kernel lab results
    ├── telemetry/run_<id>.csv  GPU timeline per benchmark point (util, VRAM, power, SM clock)
    └── profile/                profile.jsonl, top_ops_*.txt, trace_*.json (Perfetto)
```

**Design rules behind this layout**

- **src layout** (`src/infer_lab`): tests run against the *installed* package, not
  stray files in the working directory, so you catch packaging bugs early.
- **One responsibility per module**: the runner doesn't know how a backend generates
  tokens; a backend doesn't know about SQLite. That is why Day 2's vLLM backend is
  a single new file.
- **Config over code**: experiments are YAML files, so a run is reproducible from
  `(git commit, config file)`.
- **Lazy imports for heavy/optional deps** (vllm, tensorrt): a missing library only
  fails when you actually ask for that backend.

---

## 5. File-by-file, function-by-function

### 5.1 `pyproject.toml`
Defines the installable package `infer-lab`.
- `dependencies`: numpy, packaging, pandas, pyyaml, nvidia-ml-py (NVML bindings),
  transformers, accelerate, matplotlib, ninja (fast C++/CUDA builds).
- **torch is deliberately not listed**: Kaggle ships a CUDA-matched torch build, and
  pip could otherwise replace it with a CPU-only or mismatched one.
- `optional-dependencies`: `[vllm]`, `[trt]`, `[dev]`, so you install only what a
  given notebook needs.
- `[project.scripts] infer-lab = "infer_lab.cli:main"` creates the `infer-lab` command.
- `package-data` ships the `.cu` sources inside the package so the JIT loader finds them.

### 5.2 `configs/*.yaml`
| Key | Meaning |
|---|---|
| `model.name` | Hugging Face model ID (`Qwen/Qwen2.5-0.5B-Instruct`) |
| `model.dtype` | `float16`, the T4 has no bf16 hardware |
| `device` | Which GPU (`cuda:0`) |
| `ignore_eos` | `true` = always generate exactly `output_len` tokens (fair comparisons) |
| `sweep.batch_sizes / prompt_lens / output_lens` | The grid; every combination is one benchmark point |
| `sweep.warmup_iters` | Untimed runs before measuring (pays one-time costs) |
| `sweep.repeat` | Timed runs per point (more samples → stabler percentiles) |

### 5.3 `src/infer_lab/config.py`
- `ModelConfig`, `SweepConfig`, `BenchConfig`: **dataclasses**, i.e. typed containers with
  defaults. A typo in a YAML key raises an error immediately instead of being ignored.
- `BenchConfig.from_yaml(path)`: reads YAML, builds the nested dataclasses.
- `BenchConfig.to_dict()`: converts back to a plain dict; stored with every result row
  so you always know which settings produced a number.

### 5.4 `src/infer_lab/bench/metrics.py`
- `RequestRecord`: one request's life: `request_id`, `prompt_tokens`,
  `output_tokens`, and three timestamps `t_submit`, `t_first_token`, `t_end`.
  - `.ttft` = `t_first_token − t_submit`
  - `.e2e` = `t_end − t_submit`
  - `.tpot` = `(t_end − t_first_token) / (output_tokens − 1)`, the average gap between
    tokens *after* the first; `None` if only one token was produced.
- `_dist(name, values)`: converts seconds → ms and returns mean, p50, p90, p99.
- `summarize(records, wall_time_s)`: aggregates a run:
  `request_throughput_rps`, `output_tokens_per_s`, `total_tokens_per_s`
  (prompt + output), plus TTFT/TPOT/E2E distributions. Rejects empty input or zero time.

These definitions match vLLM's `benchmark_serving`, so Day 2 numbers are
apples-to-apples.

### 5.5 `src/infer_lab/bench/workload.py`
- `_SEED_TEXT`: a paragraph of inference-related English, the raw material for prompts.
- `make_prompt_ids(tokenizer, n, prompt_len, seed)`: returns `n` prompts, each
  **exactly** `prompt_len` token IDs. It shuffles the words differently for each prompt,
  tokenizes, repeats until long enough, truncates.
  - *Why token IDs, not strings?* Decoding then re-encoding text can change the
    length by a few tokens; IDs guarantee every backend sees identical input.
  - *Why shuffle?* Identical prompts would let vLLM's **prefix caching** skip prefill
    and inflate results.
- `poisson_arrivals(n, rate_rps, seed)`: arrival times for an open-loop load test
  (Day 2). Gaps are exponential with mean `1/rate`, which is how independent users arrive.
  `rate=inf` means "everyone at t=0".

### 5.6 `src/infer_lab/backends/base.py`
- `class Backend(ABC)`: the contract every runtime must satisfy:
  - `load()`: load weights / build engine (never timed).
  - `generate(prompts, max_new_tokens, ignore_eos) → list[RequestRecord]`: run all
    prompts to completion, one record per prompt, same order.
  - `tokenizer()`: so the runner can build workloads with the backend's own tokenizer.
  - `extra_metrics()`: backend-specific numbers (peak memory, prefill/decode split).
  - `close()`: free GPU memory.

### 5.7 `src/infer_lab/backends/__init__.py`
- `create_backend(name, **kwargs)`: the **registry**. Maps `"hf_torch"` →
  `HFTorchBackend`, importing it only when requested. Day 2 adds `"vllm_offline"`,
  `"vllm_server"` here.

### 5.8 `src/infer_lab/backends/hf_torch.py` (the baseline, in detail)
Module level:
- `_DTYPES`: string → torch dtype.
- `_DTYPE_KW`: `"dtype"` on transformers ≥ 4.56, else `"torch_dtype"` (the argument
  was renamed; the old name warns on 5.x, which is what the notebook showed).

`HFTorchBackend.__init__(model_name, dtype, device, attn_implementation="sdpa")`:
stores settings only (no GPU work). `sdpa` = PyTorch's fused attention kernel.

`load()`:
1. Loads the tokenizer; picks `pad_id` (falls back to EOS if the model has no pad token).
2. Loads the model in fp16, moves it to the GPU, `.eval()` (disables dropout).
3. Detects whether `forward()` accepts `logits_to_keep` (new name) or
   `num_logits_to_keep` (old name). Passing `1` during prefill makes the model compute
   logits **only for the last position**. Without it, bs=32 × 512 tokens × 151,936 vocab
   × 2 bytes ≈ **5 GB** of logits we would throw away.

`generate(prompts, max_new_tokens, ignore_eos)` runs under `@torch.inference_mode()`
(no autograd bookkeeping), step by step:

| Lines (approx.) | What happens | Why |
|---|---|---|
| Build `input_ids`, `attn` | Pad every prompt **on the left** to `max_len`; mask = 1 for real tokens, 0 for padding | Left padding makes all sequences end at the same column, so "the last position" is the same index for everyone during decode |
| `position_ids = (attn.cumsum(-1) − 1).clamp(min=0)` | Real tokens get positions 0,1,2…; padding gets 0 | Without this, a left-padded prompt would start at position 5 instead of 0 and the rotary position embeddings would be wrong |
| `reset_peak_memory_stats`, `synchronize`, `t_submit` | Start the clock with the GPU idle | GPU calls are **asynchronous**; without sync you time the *launch*, not the *work* |
| **Prefill** `self.model(...)` | One forward pass over all prompt tokens, `use_cache=True` | Produces logits + the KV cache (`past_key_values`) |
| `next_tok = logits[:, -1, :].argmax(-1)` | **Greedy decoding**: pick the most likely token | Deterministic, so runs are reproducible |
| `synchronize`, `t_first` | First token is now really computed | This timestamp defines TTFT |
| `n_out`, `finished`, `cur_pos` | Token counter per sequence; EOS tracker (`None` when `ignore_eos`); next position per sequence | |
| **Decode loop** | Append a `1` to the mask, feed **only the newest token** + `past`, get next token, advance position | The KV cache is what lets each step process 1 token instead of the whole history |
| `finished` / `n_out` update (line ~116) | If not ignoring EOS: count the new token only for sequences that were **not already finished**, then mark sequences whose new token is EOS as finished. `bool(finished.all())` stops the loop when everyone is done | A finished sequence keeps occupying a slot and burning compute until the slowest one ends: **static batching waste** in one line of code |
| `synchronize`, `t_end` | | Defines E2E |
| `self._extra` | Peak torch memory, prefill ms, decode ms | Extra diagnostics stored with the row |
| Return | One `RequestRecord` per prompt, **all sharing `t_submit`, `t_first`, `t_end`** | That is the honest semantics of static batching |

Why not `model.generate()`? It hides the prefill/decode boundary, so TTFT can't be
measured precisely, and it adds sampling/stopping logic we don't control.

`close()`: deletes the model and releases cached GPU memory.

### 5.9 `src/infer_lab/telemetry/gpu_monitor.py`
- `GPUMonitor(device_indices, interval_s=0.1)`: a **context manager** (`with GPUMonitor() as m:`).
- `__enter__`: initialises NVML, opens a handle per GPU, starts a **daemon thread**. If
  NVML is unavailable (your Windows laptop), it disables itself instead of crashing.
- `_loop()`: every 100 ms records `(time, utilisation %, memory used MB, power W, SM clock MHz)` per GPU.
- `__exit__`: stops the thread and shuts NVML down.
- `summary()`: per GPU, mean/max utilisation, max memory, mean/max power, mean/min SM clock, sample count.
- `write_timeseries(path)`: writes every sample as CSV; the runner saves one file per
  benchmark point at `results/telemetry/run_<id>.csv`.

### 5.9b `src/infer_lab/bench/profile.py` (`infer-lab profile`)
- `profile_point(...)`: wall time per decode step (3 un-profiled runs, median), then two
  `torch.profiler` runs (prefill only, and full length). Their GPU-busy difference ÷
  (ol − 1) = GPU work per decode step. Reports idle %, kernels per step, average kernel duration,
  and a verdict (launch-bound if idle > 50%). Saves the top-ops table and an optional Perfetto trace.
- `_gpu_work(prof)`: sums durations of all events that ran on the GPU (kernels, memcpys).
- `run_profile(...)`: loops over the requested points and prints the comparison table.
- `hf_torch.generate` labels its work with `record_function("prefill")` /
  `record_function("decode_step")` so the trace is readable.

Note: NVML "utilisation" = % of time *at least one kernel* was running, not % of the
GPU's compute in use. A launch-bound workload can show high utilisation while
doing little work per kernel.

### 5.10 `src/infer_lab/storage/results_db.py`
- `_SCHEMA`: table `runs` with indexed columns (`backend`, `model`, `batch_size`,
  `prompt_len`, `output_len`, `tag`) and JSON columns (`params`, `metrics`, `gpu`).
  New metrics never need a schema migration because they go into JSON.
- `ResultsDB(path)`: creates the folder and table if missing.
- `insert(...)`: writes one row, returns its id.
- `to_dataframe()`: reads all rows and **flattens** the JSON into columns
  (`ttft_p50_ms`, `gpu.gpu0.util_mean_pct`, ...) for analysis and plotting.

### 5.11 `src/infer_lab/bench/runner.py`
- `_gpu_index("cuda:1") → 1`: which GPU the monitor should watch.
- `run_sweep(cfg, backend_name, tag)`, the engine:
  1. Create + `load()` the backend; get its tokenizer.
  2. For every `(batch_size, prompt_len, output_len)` combination:
     - build `batch_size` prompts of exactly `prompt_len` tokens;
     - **warmup** `warmup_iters` times (untimed);
     - inside `GPUMonitor`, run `repeat` timed generations, collecting records + wall time;
     - on **OOM**, log a warning, free memory, skip the point (the sweep continues);
     - `summarize()`, merge `extra_metrics()` (max over repeats), insert into SQLite,
       append to JSONL, log a one-line summary.
  3. `close()` the backend.
- `_free_cuda()`: garbage-collect + empty the CUDA cache after an OOM.

### 5.12 `src/infer_lab/cli.py`
- `cmd_env`: prints Python and library versions, CUDA version, each GPU's name /
  compute capability / memory / SM count, and whether the GPUs support **P2P**.
- `cmd_bench`: loads YAML → `run_sweep`.
- `cmd_kernels`: runs the kernel lab.
- `cmd_report`: prints the results table (throughput, TTFT/TPOT/E2E, peak memory,
  GPU utilisation, power).
- `main()`: logging setup (silences noisy `httpx`/`huggingface_hub` logs) and
  `argparse` subcommands.

### 5.13 `src/infer_lab/kernels/__init__.py`
- `load_ext(name)`: compiles `csrc/<name>.cu` with `torch.utils.cpp_extension.load`
  (**JIT** compilation via nvcc + ninja) and returns it as an importable Python
  module. Sets `TORCH_CUDA_ARCH_LIST` to the current GPU (7.5) so it compiles for one
  architecture only, which is faster. `@functools.cache` means each kernel builds once per
  process; torch also caches the binary on disk.
  Flags: `-O3` (optimise), `--use_fast_math` (faster, slightly less precise math),
  `-lineinfo` (lets profilers map back to source lines).

### 5.14 `src/infer_lab/kernels/csrc/vector_add.cu` (memory-bound kernel)
- `vector_add_scalar`: **grid-stride loop**. Each thread handles elements
  `i, i+stride, i+2·stride…`, so any grid size covers any array length.
- `vector_add_vec4`: same, but loads/stores `float4` (16 bytes) at a time, giving 4× fewer
  memory instructions.
- `aligned16(ptr)`: float4 access requires 16-byte-aligned addresses; falls back to
  scalar otherwise.
- `vector_add(a, b, vectorized)`: host function. Validates inputs (CUDA, float32,
  same shape), makes them contiguous, chooses grid size (capped at 40 SMs × 32
  blocks), launches on PyTorch's current **stream**, checks for launch errors.
- `PYBIND11_MODULE`: exposes `vector_add` to Python.

### 5.15 `src/infer_lab/kernels/csrc/matmul.cu` (compute-bound kernel)
- `matmul_naive`: one thread per output `C[row, col]`; loops over `K`, reading `A` and
  `B` straight from global memory. `threadIdx.x → col` so neighbouring threads read
  neighbouring `B` elements (**coalesced**).
- `matmul_tiled<TILE=16>`: the block cooperatively loads a 16×16 tile of `A` and of `B`
  into **shared memory**, `__syncthreads()`, multiplies the tiles, repeats along `K`.
  Each global load is reused 16 times. Out-of-range elements are padded with 0 so any
  `M, N, K` works.
- `matmul(A, B, variant)`: host function: validation, grid of 16×16 blocks, dispatch.

### 5.16 `src/infer_lab/kernels/bench_kernels.py`
- `T4_PEAK_GBPS = 320`, `T4_PEAK_FP32_TFLOPS = 8.1`: spec-sheet ceilings for "% of peak".
- `time_cuda_ms(fn, warmup, iters)`: times with **CUDA events** (timestamps recorded
  *on the GPU*, so they are precise), returns the **median** (robust to outliers).
- `bench_vector_add(n=2^26)`: 67M floats; checks correctness vs `a + b`; bandwidth =
  3 arrays × n × 4 bytes / time.
- `bench_matmul(sizes)`: square matrices 512/1024/2048; checks vs `A @ B` (cuBLAS);
  TFLOPS = 2·n³ / time.
- `run_all(results_dir)`: runs both, writes `kernels.json`, prints the table.

### 5.17 `tests/`
- `test_metrics.py` (runs anywhere): derived times, single-token TPOT = None,
  throughput & percentile math, empty-input rejection, Poisson rate ≈ requested rate.
- `test_kernels.py` (skipped without CUDA): vector_add on awkward sizes (1, 3,
  2²⁰+3: not divisible by 4, exercising the fallback path) and matmul on non-square,
  non-multiple-of-16 shapes (17×33×9, 500×300×700) exercising the boundary checks.

### 5.18 `scripts/kaggle_bootstrap.sh`
Prints GPUs and their interconnect topology (`nvidia-smi topo -m`), sets
`CUDA_DEVICE_ORDER=PCI_BUS_ID` (CUDA and NVML number GPUs the same way), installs the
package in editable mode with the chosen extras, runs `infer-lab env` and `pytest`.

---

## 6. The Day-1 architecture (as built)

```
                 configs/bench.yaml
                        │  BenchConfig.from_yaml
                        ▼
  infer-lab bench ─► runner.run_sweep ──────────────────────────────────────────┐
                        │                                                       │
          ┌─────────────┼───────────────────┬──────────────────┐                │
          ▼             ▼                   ▼                  ▼                │
   create_backend   make_prompt_ids     GPUMonitor          summarize           │
   ("hf_torch")     (exact-length IDs)  (NVML thread,       (TTFT/TPOT/E2E,     │
          │                              100 ms samples)     p50/p90/p99, tok/s)│
          ▼                                   │                  │              │
   HFTorchBackend.generate                    │                  ▼              │
   ┌───────────────────────────────┐          │            ResultsDB.insert ────┤
   │ left-pad + mask + position_ids│          │            bench.sqlite         │
   │ sync ─ t_submit               │          │            bench.jsonl          │
   │ PREFILL (1 pass, last logits) │          │                  │              │
   │ sync ─ t_first     ◄── TTFT   │          │                  ▼              │
   │ DECODE loop w/ KV cache       │          │           infer-lab report      │
   │ sync ─ t_end       ◄── E2E    │          │                                 │
   └───────────────┬───────────────┘          │                                 │
                   ▼                          ▼                                 │
        ┌────────── CUDA 12.8 · sm_75 · fp16 ────────────┐                      │
        │  T4 #0 (all Day-1 work)   │   T4 #1 (idle)     │◄── PCIe PHB, no P2P  │
        └───────────────────────────────────────────────-┘                      │
                   ▲                                                            │
  infer-lab kernels ─► load_ext (nvcc JIT) ─► vector_add.cu / matmul.cu         │
                       bench_kernels: assert_close vs torch, CUDA-event timing ─┘
                                      → kernels.json, % of T4 peak
```

**Extension points ready for Day 2/3** (no changes to the engine needed):
`backends/vllm_offline.py`, `backends/vllm_server.py`, `backends/ort.py`,
`backends/trt.py`, `telemetry/vllm_metrics.py`, `sim/`, `kernels/csrc/softmax.cu` ...

---

## 7. The experiments and what the numbers mean

### 7.1 Environment (`kaggle_bootstrap.sh`, `infer-lab env`)
| Fact | Value | Consequence |
|---|---|---|
| GPUs | 2× Tesla T4, sm_75, 40 SMs, 14.6 GiB usable | fp16 only, no FlashAttention-2 |
| Software | torch 2.10+cu128, transformers 5.0, triton 3.6 | Our code works on transformers 5 |
| Topology | `PHB` | GPU↔GPU traffic crosses the CPU's PCIe host bridge |
| P2P | **False** | GPUs can't read each other's memory directly; tensor-parallel all-reduce is slow, so expect replication to win on Day 2 |
| ECC | On | ~5–10% less usable bandwidth than spec |

### 7.2 PyTorch baseline (`bench.yaml`)

| bs | prompt | tok/s | TTFT p50 | TPOT p50 | E2E p99 |
|---|---|---|---|---|---|
| 1 | 128 | 33.8 | 33 ms | 29.4 ms | 3.8 s |
| 1 | 512 | 33.8 | 47 ms | 29.6 ms | 3.8 s |
| 4 | 128 | 137.8 | 32 ms | 29.1 ms | 3.7 s |
| 4 | 512 | 132.4 | 175 ms | 29.0 ms | 3.9 s |
| 16 | 128 | 536.4 | 120 ms | 29.1 ms | 3.9 s |
| 16 | 512 | 407.0 | 725 ms | 33.9 ms | 5.1 s |
| 32 | 128 | 990.4 | 256 ms | 30.6 ms | 4.2 s |
| 32 | 512 | 443.2 | 1625 ms | 59.9 ms | 9.3 s |

**Finding 1: decode is launch-bound.** TPOT ≈ 29 ms whether bs = 1 or 32.
If decode were memory-bound, one step would cost ≈ time to read the weights once:
0.99 GB ÷ ~250 GB/s ≈ **4 ms**. The other ~25 ms is the GPU **waiting for the CPU** to
launch the hundreds of small kernels in 24 transformer layers (Python + HF overhead,
on Kaggle's 4 slow vCPUs). The fix is **CUDA graphs**: record the whole step once,
replay it with one launch. vLLM does this.

**Finding 2: batching is nearly free here.** Step time is constant, so tokens per step
grow with the batch: 34 → 138 → 536 → 990 tok/s (≈29× for 32× batch).

**Finding 3: prefill is compute-bound.** TTFT grows with *total* prompt tokens:
2k tokens → 175 ms, 8k → 725 ms, 16k → 1625 ms: ~0.1 ms/token ≈ 10 effective TFLOPS
fp16 (the T4's 65 TFLOPS is a peak at boost clock; at its 70 W cap it throttles).
Same 2k tokens: 16×128 = 120 ms but 4×512 = 175 ms, because attention cost grows with
**sequence length squared**.

**Finding 4: bs=32 × 512 leaves the launch floor.** TPOT doubles to 60 ms and
throughput *drops* to 443 tok/s. Per-step GPU work (attention over 32 × ~600 tokens,
HF's cache growing by concatenation each step) now exceeds the 29 ms launch overhead.
*This is a hypothesis; verify it with:*

```
infer-lab profile --bs 32 --pl 128 512 --ol 32 --trace
```
For each point this prints **decode wall time per step** next to **GPU-busy time per step**
(sum of all kernel durations from `torch.profiler`), the **GPU idle %**, and the number of
kernels launched per step. Prediction: pl=128 has ~85% idle (launch-bound); pl=512 has much
lower idle, and the top-ops file (`results/profile/top_ops_*.txt`) shows attention and
cache-concatenation kernels growing. Open `trace_*.json` in https://ui.perfetto.dev to see
the gaps between kernels with your own eyes.

**Result (Kaggle, 2026-10-03; profile of bs=32, pl=512, ol=32):**

| Evidence | Value | Meaning |
|---|---|---|
| Kernels per decode step | **1422**, same for pl=128 and pl=512 | ≈59 tiny kernels per layer × 24 layers. At ~23 µs of CPU dispatch each, that is ≈33 ms: the launch floor |
| bs=1 NVML | util 35–39%, power 33–37 W, SM clock 700–1100 MHz | GPU mostly idle, even down-clocking, so launch-bound |
| TPOT on two different Kaggle VMs | 29 ms (Oct 2) vs 33 ms (Oct 3), same GPU model | The floor moves with the **host CPU**, which only makes sense if decode is CPU-bound |
| bs=32 pl=512 NVML | util 98%, 68 W of a 70 W cap, SM clock ~1100 MHz (boost 1590) | GPU saturated and power-throttled |
| `Command Buffer Full` | 362 events, 418 ms CPU | CPU blocked because the GPU's launch queue was full, so the GPU is now the bottleneck |
| `gemv… float` kernels | 744 calls = 31 steps × 24 layers | Attention runs in **fp32** (math SDPA fallback; torch upcasts fp16 there) |
| Large copy kernels (elementwise / vectorized unary) | ~870 ms ≈ 28 ms per step | `repeat_kv` expands 2 KV heads → 14 (×7) and converts them to fp32, **every step, every layer** |
| fp16 GEMM for linear layers | ~4.3 ms per step | Reading 0.99 GB of weights at ~230 GB/s, the only "ideal" part of the step |

**Explanation.** A decode step costs `max(CPU launch floor ≈ 33 ms, GPU work)`.
GPU work ≈ (weights ≈ 4 ms) + (attention traffic ∝ batch × context). Attention here moves
the KV cache roughly **40× more bytes than necessary** (7× GQA expansion, 2× fp32 upcast,
copied again every layer, every step). At bs=16 × 512 the GPU work reaches the floor (TPOT
34 ms); at bs=32 × 512 (~18k cached tokens) it is ≈56 ms, so TPOT doubles and throughput
drops. vLLM's PagedAttention kernel reads the fp16 KV cache once with native GQA, and
CUDA graphs remove the launch floor; Day 2 measures both effects.

The NVML monitor alone can't settle this: its "utilisation" only says *some* kernel
was running in each sample window, so launch-bound decode can still read ~90–100%. The
per-run timeline in `results/telemetry/run_<id>.csv` (power, SM clock) shows how hard
the GPU works and whether it throttles.

**Finding 5: static batching is visible.** TTFT p50 ≈ p99 and E2E is identical inside
a batch: everyone starts together and finishes together. A short request would wait for
the longest one. **Continuous batching** (Day 2) admits and releases requests every step.

### 7.3 Kernel lab (`infer-lab kernels`)

| Kernel | Variant | Result | % of T4 spec |
|---|---|---|---|
| vector_add (67M) | torch | 245.5 GB/s | 76.7% |
| | scalar (ours) | 225.6 GB/s | 70.5% |
| | vec4 (ours) | 237.9 GB/s | 74.4% |
| matmul 1024² | cuBLAS | 5.91 TFLOPS | 73.0% |
| | naive | 0.43 TFLOPS | 5.3% |
| | tiled16 | 0.73 TFLOPS | 9.0% |

- **vector_add is at the practical ceiling.** With ECC on, ~245 GB/s is about the most a T4
  delivers, so all three variants are equal-ish. Lesson: **once memory-bound, code tweaks
  barely matter; only moving fewer bytes helps.** (That is why LLMs get quantised.)
- **Tiled matmul beats naive 1.5–1.7×** by reusing data from shared memory, but each
  thread still computes only one output and does 2 shared-memory reads per FMA, so it is
  now **shared-memory-bound**. cuBLAS is 6–8× faster because each thread computes a block
  of outputs held in **registers** (register blocking), uses wide loads, and double-buffers.
  That's Day 3's kernel upgrade.
- cuBLAS dropping to 44% at 2048² is most likely **thermal/power throttling** on a
  sustained run.

---

## 8. Latest results: Kaggle session 2026-10-04

Source: `notebooks/test1.ipynb`, session of 2026-10-04 (fresh VM, latest code including
the fixed profiler). Model `Qwen/Qwen2.5-0.5B-Instruct`, fp16, `cuda:0`, `ignore_eos=true`,
greedy, `repeat=3`.

### 8.1 Environment

| Item | Value |
|---|---|
| GPUs | 2× Tesla T4, sm_75, 40 SMs, 14.6 GiB usable each |
| Driver / CUDA | 580.178.04 / CUDA 12.8 (driver supports 13.0) |
| Software | Python 3.12.13, torch 2.10.0+cu128, transformers 5.0.0, triton 3.6.0 |
| Topology | `PHB` (GPU↔GPU over PCIe through the host bridge) |
| P2P 0↔1 | **True** (was False on the 2026-10-02 VM; varies by Kaggle host) |
| Tests | 23 passed (144 s, includes kernel JIT builds) |

### 8.2 Benchmark sweep: `bench.yaml`, tag `baseline_v2`

TTFT/TPOT/E2E in ms, peak memory in MB (torch allocator), util/power/SM clock = NVML mean
over the measured repeats.

| Run | bs | pl | ol | Output tok/s | TTFT p50 | TTFT p99 | TPOT p50 | E2E p99 | Peak mem | GPU util | Power | SM clock |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 1 | 128 | 128 | 32.67 | 33.03 | 38.20 | 30.48 | 3958 | 966 | 38% | 38.6 W | 885 MHz |
| 4 | 1 | 512 | 128 | 32.95 | 47.73 | 48.98 | 30.35 | 3904 | 1009 | 35% | 46.1 W | 1291 MHz |
| 5 | 4 | 128 | 128 | 134.78 | 33.52 | 34.45 | 29.69 | 3828 | 984 | 36% | 47.7 W | 1362 MHz |
| 6 | 4 | 512 | 128 | 126.50 | 199.51 | 200.42 | 30.01 | 4161 | 1155 | 49% | 64.7 W | 1548 MHz |
| 7 | 16 | 128 | 128 | 511.59 | 146.12 | 146.30 | 30.37 | 4004 | 1059 | 56% | 67.5 W | 1478 MHz |
| 8 | 16 | 512 | 128 | 381.65 | 906.67 | 918.96 | 35.10 | 5393 | 1739 | 98% | 65.4 W | 948 MHz |
| 9 | 32 | 128 | 128 | 967.73 | 287.76 | 288.73 | 31.09 | 4255 | 1160 | 87% | 66.3 W | 1183 MHz |
| 10 | 32 | 512 | 128 | 434.87 | 1720.76 | 1733.73 | **60.53** | 9437 | 2517 | 98% | 66.1 W | 946 MHz |

Smoke run (`smoke.yaml`) in the same session:

| Run | bs | pl | ol | Output tok/s | TTFT p50 | TPOT p50 | E2E p99 | Peak mem | GPU util | Power | SM clock |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 64 | 32 | 32.70 | 33.52 | 30.46 | 978 | 962 | 40% | 35.7 W | 832 MHz |
| 2 | 4 | 64 | 32 | 135.70 | 32.98 | 29.34 | 942 | 974 | 41% | 35.7 W | 952 MHz |

### 8.3 Decode profile: `infer-lab profile --bs 1 32 --pl 128 512 --ol 32` (fixed profiler)

Wall = un-profiled median; GPU busy = union of kernel intervals; idle = 1 − busy/wall.
"CPU stall" = CPU time blocked on a full GPU launch queue (`Command Buffer Full`).

| bs | pl | Prefill wall | Prefill GPU busy | Decode wall/step | Decode GPU busy/step | GPU idle | Kernels/step | Avg kernel | CPU stall | Regime |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 128 | 31.2 ms | 15.8 ms | 28.59 ms | 10.28 ms | **64%** | 1421 | 7.2 µs | 0 ms | launch-bound |
| 1 | 512 | 52.5 ms | 43.9 ms | 36.79 ms | 9.21 ms | **75%** | 1421 | 6.5 µs | 0 ms | launch-bound |
| 32 | 128 | 222.3 ms | 235.1 ms | 38.31 ms | 20.78 ms | **46%** | 1421 | 14.6 µs | 12 ms | mixed (CPU ≈ GPU) |
| 32 | 512 | 1439.9 ms | 1442.9 ms | 55.46 ms | 53.50 ms | **4%** | 1421 | 37.7 µs | 401 ms | GPU-bound |

Repeat of the bs=32 points earlier in the same session (`--bs 32 --pl 128 512 --ol 32 --trace`):

| bs | pl | Prefill wall | Prefill GPU busy | Decode wall/step | Decode GPU busy/step | GPU idle | Kernels/step | Avg kernel | CPU stall |
|---|---|---|---|---|---|---|---|---|---|
| 32 | 128 | 241.3 ms | 231.1 ms | 29.64 ms | 20.97 ms | 29% | 1421 | 14.8 µs | 18 ms |
| 32 | 512 | 1456.7 ms | 1466.5 ms | 55.43 ms | 53.34 ms | 4% | 1421 | 37.5 µs | 412 ms |

GPU busy/step is stable between repeats (20.97 vs 20.78 ms); wall/step at bs=32 × 128 is
not (29.6 vs 38.3 ms). When the CPU is (part of) the bottleneck, timings pick up host jitter.

### 8.4 Top GPU ops: bs=32, pl=512, ol=32 (full profiled run incl. prefill)

| Op / kernel | Self CUDA | % of GPU time | Calls | Avg | What it is |
|---|---|---|---|---|---|
| `aten::copy_` | 710.5 ms | 22.8% | 8803 | 83.6 µs | KV `repeat_kv` expansion + fp16→fp32 casts |
| `aten::bmm` | 621.2 ms | 19.9% | 1568 | 419.8 µs | Attention matmuls (math SDPA) + RoPE |
| `aten::mm` | 588.2 ms | 18.9% | 3104 | 227.9 µs | Linear layers (q/k/v/o, MLP, lm_head) |
| `aten::mul` | 543.1 ms | 17.4% | 8576 | 67.9 µs | Scaling, RMSNorm, RoPE, SwiGLU |
| `elementwise_kernel<128,2>` | 462.3 ms | 14.8% | 1560 | 296.3 µs | Large strided copy (expanded KV) |
| `vectorized_elementwise_kernel<4> (unary)` | 408.9 ms | 13.1% | 1600 | 255.6 µs | Large dtype cast (KV → fp32) |
| `Command Buffer Full` (CPU) | 411.7 ms CPU | n/a | 365 | 1.13 ms | CPU waiting on a full GPU queue |
| `turing_fp16_s1688gemm_256x128` | 251.3 ms | 8.1% | 48 | 5.24 ms | Prefill linear layers (fp16 tensor cores) |
| `gemv2T_kernel_val<…float…>` | 199.9 ms | 6.4% | 744 | 268.7 µs | Decode attention in **fp32** (31 steps × 24 layers) |
| `gemvx::kernel<…>` | 188.8 ms | 6.1% | 744 | 253.7 µs | Decode attention in fp32 (second matmul) |
| `unrolled_elementwise_kernel (direct_copy)` | 139.5 ms | 4.5% | 3904 | 35.7 µs | Smaller copies |
| `turing_fp16_s1688gemm_128x64_sliced1x2` | 133.4 ms | 4.3% | 1520 | 87.7 µs | Decode linear layers (≈4.3 ms/step: the weight read) |
| `volta_sgemm_128x64_tn` | 127.9 ms | 4.1% | 24 | 5.33 ms | Prefill attention in **fp32** (one per layer) |

Totals: Self CPU 3.204 s, Self CUDA 3.120 s. (The `decode_step` / `prefill` rows are our
labels and contain the kernels above, so they are not listed.)

### 8.5 Kernel lab: `infer-lab kernels`

| Kernel | Variant | Size | Time | Throughput | % of T4 spec |
|---|---|---|---|---|---|
| vector_add | torch | 67,108,864 | 3.307 ms | 243.5 GB/s | 76.1% |
| vector_add | scalar | 67,108,864 | 3.594 ms | 224.0 GB/s | 70.0% |
| vector_add | vec4 | 67,108,864 | 3.412 ms | 236.0 GB/s | 73.8% |
| matmul | cuBLAS | 512² | 0.069 ms | 3.865 TFLOPS | 47.7% |
| matmul | naive | 512² | 0.453 ms | 0.593 TFLOPS | 7.3% |
| matmul | tiled16 | 512² | 0.290 ms | 0.924 TFLOPS | 11.4% |
| matmul | cuBLAS | 1024² | 0.365 ms | 5.891 TFLOPS | 72.7% |
| matmul | naive | 1024² | 4.730 ms | 0.454 TFLOPS | 5.6% |
| matmul | tiled16 | 1024² | 2.787 ms | 0.770 TFLOPS | 9.5% |
| matmul | cuBLAS | 2048² | 4.650 ms | 3.695 TFLOPS | 45.6% |
| matmul | naive | 2048² | 40.780 ms | 0.421 TFLOPS | 5.2% |
| matmul | tiled16 | 2048² | 26.832 ms | 0.640 TFLOPS | 7.9% |

Tiled vs naive speed-up: 1.56× (512²), 1.70× (1024²), 1.52× (2048²).

### 8.6 Same sweep across three Kaggle VMs

| bs | pl | TPOT p50 Oct 2 | TPOT p50 Oct 3 | TPOT p50 Oct 4 | tok/s Oct 2 | tok/s Oct 3 | tok/s Oct 4 |
|---|---|---|---|---|---|---|---|
| 1 | 128 | 29.44 | 34.46 | 30.48 | 33.76 | 28.9 | 32.67 |
| 1 | 512 | 29.59 | 31.84 | 30.35 | 33.79 | 30.7 | 32.95 |
| 4 | 128 | 29.09 | 33.34 | 29.69 | 137.84 | 119.6 | 134.78 |
| 4 | 512 | 28.96 | 32.66 | 30.01 | 132.42 | 118.0 | 126.50 |
| 16 | 128 | 29.12 | 33.40 | 30.37 | 536.39 | 467.7 | 511.59 |
| 16 | 512 | 33.92 | 34.25 | 35.10 | 407.05 | 407.0 | 381.65 |
| 32 | 128 | 30.57 | 33.83 | 31.09 | 990.41 | 899.4 | 967.73 |
| 32 | 512 | **59.92** | **59.66** | **60.53** | 443.19 | 452.3 | 434.87 |

| bs | pl | TTFT p50 Oct 2 | TTFT p50 Oct 3 | TTFT p50 Oct 4 | SM clock Oct 3 | SM clock Oct 4 |
|---|---|---|---|---|---|---|
| 16 | 512 | 725.28 | 690.1 | 906.67 | 1270 MHz | 948 MHz |
| 32 | 512 | 1624.91 | 1487.6 | 1720.76 | 1092 MHz | 946 MHz |

### 8.7 Predictions vs measured

| Prediction (made before the run) | Measured | Verdict |
|---|---|---|
| bs=1 decode ~85–90% GPU idle | 64–75% idle | Direction right, magnitude too high: GPU busy is ~10 ms/step, not ~4 ms, because 1421 tiny kernels each cost ~7 µs even with almost no work |
| bs=32 × 128 ~40% idle | 29–46% idle, 12–18 ms CPU stall | ✅ mixed regime |
| bs=32 × 512 ~0% idle, large CPU stall | 4% idle, 401–412 ms stall | ✅ GPU-bound |
| Kernel count independent of shape | 1421 at every point | ✅ |
| Profiler busy ≤ wall after fix | Holds at all points (prefill busy within ±1–2% of wall at bs=32) | ✅ |

### 8.8 What these numbers say

1. **Two regimes, one formula.** Decode step ≈ max(CPU launch time, GPU work). CPU launch
   time ≈ 28–31 ms (1421 kernels); GPU work grows with batch × context: 10 ms (bs=1) →
   21 ms (bs=32 × 128) → 53 ms (bs=32 × 512).
2. **The cross-VM table is the cleanest proof.** Launch-bound points move with the host
   CPU (29 → 33 → 30 ms TPOT), while the GPU-bound point (bs=32 × 512) stays at ~60 ms on
   all three VMs.
3. **Prefill is compute-bound and throttles.** At bs=32 prefill GPU busy ≈ wall. The same
   prefill was 16% slower on Oct 4 than Oct 3 with a 13% lower SM clock (946 vs 1092 MHz):
   the T4 sits at its 70 W cap and down-clocks.
4. **Where bs=32 × 512 time goes:** KV copies/casts (~23% + kernels) and fp32 attention
   matmuls dominate; the "useful" fp16 weight GEMMs are ~4.3 ms/step. That gap is the
   target for vLLM (PagedAttention, CUDA graphs) on Day 2.
5. **Kernel lab is reproducible.** Within ±3% of the previous sessions.

### 8.9 Notes on the notebook

- Cells 5 and 7 still show **older outputs** (Oct 3 / Oct 2); in the Oct 4 session only
  `baseline_v2` was benchmarked, which is why its runs are numbered 3–10 in the report.
- Cell 14 reads `results/telemetry/1.csv`; the runner writes `results/telemetry/run_<id>.csv`
  (e.g. `run_10.csv` for bs=32 × 512).
- The `torch_dtype is deprecated` warning is still printed: `hf_torch.py` line 48 still passes
  `torch_dtype=` (warning only; results unaffected).

---

## 9. Limitations to remember (honest caveats)

- One model (0.5B), one GPU, greedy decoding, `ignore_eos=True`, synthetic prompts.
- `repeat=3` → p99 with few samples is close to the max; fine for static batches
  (identical timings), but Day 2's load tests need hundreds of requests.
- NVML utilisation ≠ compute efficiency (see §5.9).
- Wall-clock per-step numbers in CPU-bound regimes vary between runs and VMs (§8.3, §8.6);
  compare GPU-busy times, or results from the same session.

---

## 10. Glossary

**Model & generation**
- **LLM inference**: running a trained model to produce output (no training/gradients).
- **Token**: a chunk of text (word piece) the model works with; ~0.75 English words.
- **Tokenizer**: converts text ↔ token IDs.
- **Logits**: the model's raw score for every vocabulary token at a position (151,936 for Qwen2.5).
- **Greedy decoding / argmax**: always pick the highest-scoring token.
- **EOS**: end-of-sequence token; the model's way of saying "I'm done".
- **ignore_eos**: keep generating past EOS, so output length is fixed for benchmarking.
- **Prefill**: the forward pass over the whole prompt; produces the first token + KV cache.
- **Decode**: the token-by-token generation phase after prefill.
- **KV cache**: stored attention Keys and Values of all previous tokens, so each decode
  step only processes the newest token. Size per token = 2 · layers · kv_heads · head_dim ·
  bytes; for Qwen2.5-0.5B fp16 = 2·24·2·64·2 = **12 KiB/token**.
- **Attention / SDPA**: the operation where each token looks at all previous tokens.
  SDPA = PyTorch's fused `scaled_dot_product_attention` kernel.
- **Attention mask**: 1 = real token, 0 = padding (ignore).
- **Padding (left)**: filling shorter prompts with dummy tokens on the left so a batch is a
  rectangle and all sequences end at the same column.
- **Position IDs**: each token's index in its sequence; feeds rotary position embeddings (RoPE).
- **fp16 / bf16**: 16-bit float formats. T4 supports fp16 in hardware; bf16 needs sm_80+.
- **Safetensors**: the safe, fast file format the weights are stored in (`model.safetensors`, 988 MB).

**Metrics**
- **TTFT**: Time To First Token = prefill latency (+ queueing when serving).
- **TPOT**: Time Per Output Token = average gap between tokens after the first (also "ITL").
- **E2E latency**: submit → last token.
- **Throughput**: work per second: **tok/s** (output tokens/s) or **req/s**.
- **p50 / p90 / p99**: percentiles. p99 = 99% of requests were at least this fast; the
  "tail latency" users actually notice.
- **Wall time**: real elapsed clock time.
- **Warmup**: untimed runs that pay one-time costs (CUDA context, kernel selection, caches).
- **Sweep**: running every combination of parameters in a grid.

**Batching & serving**
- **Batch size (bs)**: number of sequences processed together.
- **Static batching**: a fixed batch starts and ends together; finished sequences waste slots.
- **Continuous batching**: the scheduler adds/removes sequences every decode step (vLLM).
- **Prefix caching**: reusing the KV cache of an identical prompt prefix.
- **Poisson arrivals / open-loop load**: requests arrive at random independent times at a
  set average rate, regardless of whether the server keeps up, which is how real traffic behaves.
- **OOM**: Out Of Memory: the GPU ran out of VRAM.

**GPU hardware**
- **T4**: NVIDIA Turing datacenter GPU: 16 GB GDDR6, 320 GB/s, 70 W, 40 SMs.
- **sm_75 / compute capability 7.5**: the T4's architecture version; decides which
  features/kernels are available.
- **SM (Streaming Multiprocessor)**: one of the GPU's 40 compute units.
- **VRAM**: the GPU's own memory.
- **Memory bandwidth (GB/s)**: how fast data moves between VRAM and the SMs.
- **FLOPS / TFLOPS**: floating-point operations per second (10¹² for TFLOPS).
  A matmul of n×n costs 2·n³ FLOPs.
- **Roofline**: the model that says a kernel's speed is capped by either compute peak or
  bandwidth peak, depending on its **arithmetic intensity** (FLOPs per byte moved).
- **Compute-bound**: limited by math throughput (prefill, big matmul).
- **Memory-bound**: limited by bytes moved (vector_add, decode in an efficient engine).
- **Launch-bound / overhead-bound**: limited by the CPU's speed at issuing kernels; the
  GPU sits idle between tiny kernels (our eager decode).
- **ECC**: error-correcting memory; safer, costs some bandwidth.
- **Throttling**: the GPU lowers its clock when it hits power/thermal limits.
- **PCIe**: the bus connecting GPUs to the CPU. **NVLink**: NVIDIA's much faster
  GPU↔GPU link (not on Kaggle).
- **PHB** (`nvidia-smi topo`): the path between GPUs goes through the PCIe Host Bridge (CPU).
- **P2P (peer-to-peer)**: one GPU directly accessing another's memory; **unavailable** here.
- **NVML**: NVIDIA Management Library: reads utilisation, memory, power (`nvidia-smi` uses it).

**CUDA programming**
- **Kernel**: a function that runs on the GPU in thousands of threads.
- **Thread / block / grid**: a thread is one execution lane; threads are grouped into
  blocks (share fast shared memory, can `__syncthreads()`); blocks form the grid.
- **Grid-stride loop**: each thread processes several elements spaced a grid apart.
- **Kernel launch**: the CPU telling the GPU to run a kernel (~5–10 µs overhead each).
- **CUDA stream**: an ordered queue of GPU work; we launch on PyTorch's current stream.
- **Asynchronous execution / `torch.cuda.synchronize()`**: the CPU queues GPU work and
  moves on; `synchronize` waits until the GPU is finished, which is required for correct timing.
- **CUDA events**: GPU-side timestamps for precise kernel timing.
- **CUDA graphs**: record a sequence of kernels once and replay it with a single launch,
  which removes launch overhead.
- **Coalesced access**: neighbouring threads read neighbouring addresses, so the hardware
  merges them into few wide memory transactions.
- **Vectorised load (float4)**: one instruction loads 4 floats (16 bytes).
- **Alignment**: an address that is a multiple of the access size (16 B for float4).
- **Global memory**: VRAM (large, slow). **Shared memory**: small, fast per-block scratchpad.
  **Registers**: the fastest storage, private per thread.
- **Tiling**: loading a sub-block of data into shared memory so it is reused many times.
- **Register blocking**: each thread computes several outputs kept in registers (next step).
- **FMA**: fused multiply-add `a*b + c`, the basic matmul operation (counts as 2 FLOPs).
- **cuBLAS**: NVIDIA's hand-tuned linear-algebra library; `torch.matmul` calls it.
- **JIT compilation**: compiling the `.cu` file at runtime on first use (nvcc + ninja).
- **pybind11**: the glue that exposes C++ functions to Python.

**Multi-GPU (Day 2 preview)**
- **Tensor parallelism (TP)**: split each layer's weights across GPUs; they exchange partial
  results every layer via **all-reduce** (sum across GPUs).
- **Replication / data parallelism**: a full model copy on each GPU, requests split
  between them by a **router**; no communication.

**Tooling**
- **Editable install (`pip install -e`)**: the installed package points at your source
  folder, so edits apply without reinstalling.
- **Dataclass**: a Python class auto-generated from typed fields.
- **Context manager (`with`)**: guarantees setup/teardown (start/stop the GPU monitor).
- **Daemon thread**: a background thread that won't keep the process alive.
- **SQLite**: a file-based SQL database. **JSONL**: one JSON object per line.
- **ABC (abstract base class)**: a class that defines methods subclasses must implement.
