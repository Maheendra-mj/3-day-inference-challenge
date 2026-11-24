# Day 2: vLLM on T4 (Part 1: Offline Engine vs PyTorch Baseline)

This document covers the first half of Day 2: getting vLLM running on a Kaggle T4,
plugging it into the Day-1 harness, and the three-way comparison
**HF eager vs vLLM without CUDA graphs vs vLLM with CUDA graphs**, all on the same VM.
It follows the same structure as [DAY1.md](DAY1.md); terms already defined there are not
repeated (see DAY1 §10), and new terms are in §10 here.

---

## 1. Goal of Day 2

From the 3-day plan ([README](../README.md)):

| # | Block | Status |
|---|---|---|
| D2.1 | `vllm_offline` backend: same sweep as Day 1, direct comparison | ✅ **this document** |
| D2.2 | `vllm_server` backend: OpenAI server + Poisson load test (real p99) | ⬜ next |
| D2.3 | KV-Cache Observatory: scrape `/metrics`, context × concurrency sweep, analytic vs measured | 🟡 analytic model validated (§6.2); observatory pending |
| D2.4 | Multi-GPU: TP=2 vs 2 × TP=1 replicas | ⬜ |
| D2.5 | Continuous batching simulator, calibrated and validated against vLLM | ⬜ |

**Goal of D2.1:**
> Show *how much* faster a production engine is than our Day-1 baseline, and **attribute
> the speed-up to its causes** using the Day-1 diagnosis (launch floor vs attention
> inefficiency).

## 2. What was achieved

| Result | Evidence |
|---|---|
| vLLM 0.31.0 runs on T4 (sm_75) | smoke test exit 0; `TRITON_ATTN` backend; CUDA graphs captured |
| KV-cache formula from Day 1 validated | 6.99 GiB ↔ 610,896 tokens = **12,288 B/token** (= 2·24·2·64·2, within rounding of the logged GiB) |
| vLLM is **2.9–5.1× faster** in throughput, **3.2–5.4× faster** in TPOT | 8-point sweep, same VM, same prompts (§6.3) |
| Speed-up **decomposed** into "CUDA graphs" vs "better kernels/engine" | `enforce_eager` on/off experiment (§6.4) |
| Day-1 diagnosis confirmed by an independent engine | eager vLLM is stuck at the same ~27 ms launch floor as HF (§7) |
| Environment now stored with every result | `env.py`: CPU, driver, GPUs, P2P, library versions |

---

## 3. The setup story (what went wrong and why the fix works)

### 3.1 Why `pip install vllm==0.7.3` failed
Kaggle's image ships **torch 2.10 / transformers 5.0 on Python 3.12**. vLLM pins an exact
torch version per release; 0.7.3 (early 2025) wants a much older torch, so pip tried
to build a dependency from source and failed (`metadata-generation-failed`). Installing
`transformers==4.46.3` first only added another conflict.

### 3.2 The fix: an isolated virtual environment
`scripts/vllm_setup.sh` creates `/tmp/vllm-env` with **uv** (a fast pip replacement) and
installs vLLM + our package there in **one dependency resolution**, so vLLM's pins win
consistently and Kaggle's system packages are untouched. Result:

| | System (notebook kernel) | `/tmp/vllm-env` |
|---|---|---|
| Python | 3.12.13 | 3.13.15 (Kaggle's `python3` binary) |
| torch | 2.10.0+cu128 | 2.13.0+cu130 |
| transformers | 5.0.0 | 5.17.0 |
| vLLM | – | 0.31.0 |
| Used for | `hf_torch` baseline | `vllm_offline` |

Both environments use the **same source folder** (editable installs), so a `git pull`
updates both. The venv lives in `/tmp` (not persisted), so setup runs once per session (~3–5 min).

### 3.3 Bugs hit along the way (all fixed)
| Symptom | Cause | Fix |
|---|---|---|
| `ctypes.ArgumentError … parameter 3` in `infer-lab env` | Newer `nvidia-ml-py` in the venv exposes `NVML_P2P_CAPS_INDEX_READ` as an enum ctypes can't pass; only `NVMLError` was caught | `int(...)` the constant; catch every exception, since fingerprinting must never break a run |
| Fix pushed but Kaggle still ran old code | Cell 0 ran `git clone` from *inside* the repo, so it created a nested clone; later `git pull`s updated the nested copy | Cell 0 now uses absolute paths: `%cd /kaggle/working`, then pull-or-clone |
| Re-running setup prints "venv exists, replace? [y/n]" | `uv venv` prompts when the folder exists | `uv venv --clear` |
| `Error in sitecustomize … No module named 'wrapt'` | Kaggle's site-wide startup hook runs inside our venv | Harmless; ignore |

---

## 4. Codebase changes for Day 2 (file by file)

```
configs/
  vllm_offline.yaml            NEW  same sweep as bench.yaml, CUDA graphs ON
  vllm_offline_eager.yaml      NEW  identical, but enforce_eager: true (graphs OFF)
scripts/
  vllm_setup.sh                NEW  isolated venv install (uv), prints versions + env
  vllm_smoke.py                NEW  go/no-go test of vLLM on this GPU
src/infer_lab/
  config.py                    CHANGED  BenchConfig.backend_args
  cli.py                       CHANGED  `env` prints CPU model + driver
  bench/runner.py              CHANGED  passes backend_args, stores env, monitors ALL GPUs
  backends/__init__.py         CHANGED  registers "vllm_offline"
  backends/vllm_offline.py     NEW  vLLM engine behind the Backend interface
  backends/hf_torch.py         FIXED  dtype= instead of deprecated torch_dtype=
  telemetry/env.py             NEW  environment fingerprint (NVML only)
tests/
  test_config.py               NEW  configs parse; vLLM configs differ only in CUDA graphs
notebooks/
  vllm.ipynb                   NEW  the Day-2 Kaggle run
```

### 4.1 `scripts/vllm_setup.sh`
1. `pip install uv`; `uv venv --clear --python python3 /tmp/vllm-env`.
2. `uv pip install "vllm[==version]" -e ".[dev]" requests aiohttp` in one resolve.
   Optional argument pins a version: `bash scripts/vllm_setup.sh 0.10.1.1`.
3. Prints vllm/torch/transformers versions and runs `infer-lab env` from the venv.

### 4.2 `scripts/vllm_smoke.py`
`main()` builds `LLM(model, dtype="float16", max_model_len=1024, gpu_memory_utilization=0.6)`,
warms up with one prompt, generates 32 × 128 tokens greedily with `ignore_eos`, and prints
versions, a text sample and output tok/s. The `if __name__ == "__main__"` guard is required
because vLLM may start worker processes with *spawn*, which re-imports the script.

### 4.3 `src/infer_lab/backends/vllm_offline.py`
- `_kv_capacity(engine)`: reads `num_gpu_blocks × block_size` from the engine's cache
  config if visible. In vLLM V1 the engine core is a separate process, so this is often
  empty; vLLM's startup log ("GPU KV cache size") is the reliable source.
- `VLLMOfflineBackend.__init__(...)`: collects vLLM engine arguments:
  `tensor_parallel_size`, `gpu_memory_utilization` (fraction of VRAM vLLM may claim),
  `max_model_len`, `max_num_seqs`, `enforce_eager` (True = no CUDA graphs / compile),
  `enable_prefix_caching=False` (our prompts must not share cached KV), `seed`.
  `device` is accepted but ignored (vLLM places itself; use `CUDA_VISIBLE_DEVICES`).
- `load()`: creates `vllm.LLM`, keeps `llm.llm_engine` and the tokenizer.
- `generate(prompts, max_new_tokens, ignore_eos)`: **why not `LLM.generate()`?** It only
  returns finished requests, so per-request TTFT is lost. Instead:
  1. `SamplingParams(temperature=0, max_tokens, ignore_eos, output_kind=DELTA)`: greedy,
     and each step returns only the *new* tokens (cheap to count).
  2. `add_request(id, {"prompt_token_ids": ids}, params)` for every prompt; ids are unique
     across calls (`"<batch>-<i>"`).
  3. Loop `while has_unfinished_requests(): step()`. For each output: first time a
     request has tokens → `t_first_token`; accumulate the count; when `finished` → `t_end`.
  4. Return one `RequestRecord` per prompt, same as `hf_torch`, so the rest of the harness
     needs no changes.
  This is exactly the loop `LLM.generate()` runs internally, plus timestamps.
- `close()`: drops the engine; garbage collection shuts the engine-core process down.

### 4.4 `src/infer_lab/telemetry/env.py`
- `_cpu_model()`: `model name` from `/proc/cpuinfo`.
- `_gpus()`: NVML only (**no `torch.cuda` calls**, which would initialise CUDA in the parent
  process before vLLM starts workers): driver, each GPU's name / compute capability /
  memory, and P2P read capability between GPU 0 and 1.
- `env_info()`: Python, CPU, vCPU count, library versions, GPU info. Any failure is recorded
  as `gpu_info_error` instead of raised.

### 4.5 `config.py`, `runner.py`, `backends/__init__.py`, `cli.py`
- `BenchConfig.backend_args: dict`: free-form constructor arguments per backend, read from
  YAML (`backend_args:` block).
- `run_sweep`: `create_backend(..., **cfg.backend_args)`; `params = {**config, "env": env_info()}`
  is stored with every row; `GPUMonitor()` now samples **all** GPUs (needed for TP=2).
- Registry: `create_backend("vllm_offline")` → `VLLMOfflineBackend` (lazy import).
- `infer-lab env` now also prints CPU model / vCPUs and driver.

### 4.6 `configs/vllm_offline*.yaml`
Identical to `bench.yaml` (4 batch sizes × 2 prompt lengths × 128 output tokens, warmup 1,
repeat 3) plus `backend_args: {gpu_memory_utilization: 0.85, max_model_len: 2048,
enforce_eager: false|true}`. `tests/test_config.py` asserts they differ **only** in
`enforce_eager` and share the Day-1 sweep, which keeps the experiment controlled.

---

## 5. The notebook (`notebooks/vllm.ipynb`) cell by cell

| Cell | Command | Result |
|---|---|---|
| 0 | `%cd /kaggle/working`, pull-or-clone, `%cd` repo, `git log -1` | Repo at latest commit |
| 1 | `bash scripts/vllm_setup.sh` | vLLM 0.31.0 / torch 2.13 / transformers 5.17 in `/tmp/vllm-env` |
| 2 | `vllm_smoke.py` → log, then grep key lines + tail | Exit 0, TRITON_ATTN, KV 610,896 tokens, 3,808 tok/s |
| 3 | One-off repair: pull the outer repo, delete nested clone | `78736d1 → 3f245e1` |
| 4 | `/tmp/vllm-env/bin/infer-lab env` | Works after the fix (table §6.1) |
| 5 | `bench --backend vllm_offline --config vllm_offline.yaml --tag vllm` | Runs 1–8 |
| 6 | Same with `vllm_offline_eager.yaml --tag vllm_eager` | Runs 9–16 |
| 7 | System Python: `pip install -e .` + `bench --config bench.yaml --tag hf_baseline` | Runs 17–24 |
| 8 | `infer-lab report` | Combined 24-row table (§6.3) |

All three sweeps ran **back-to-back on the same VM**: Day 1 showed launch-bound numbers
move ±15% between Kaggle hosts (DAY1 §8.6), so cross-session comparisons would be unfair.

---

## 6. Results: Kaggle session 2026-10-05

### 6.1 Environment
| Item | Value |
|---|---|
| CPU | Intel Xeon @ 2.00 GHz, 4 vCPU |
| GPUs | 2× Tesla T4, sm_75, 40 SMs, 14.6 GiB |
| Driver / CUDA (venv) | 580.178.04 / 13.0 |
| P2P 0↔1 | True |
| vLLM env | Python 3.13.15, torch 2.13.0+cu130, transformers 5.17.0, triton 3.7.1, vLLM 0.31.0 |
| HF env | Python 3.12.13, torch 2.10.0+cu128, transformers 5.0.0 |

### 6.2 Smoke test (`vllm_smoke.py`, `gpu_memory_utilization=0.6`)
| Item | Value | Meaning |
|---|---|---|
| Engine | V1 engine, vLLM 0.31.0 | |
| Attention backend | **TRITON_ATTN** (candidates: TRITON_ATTN, FLEX_ATTENTION) | Triton-written PagedAttention; FlashAttention needs sm_80+ |
| Compilation | torch.compile 19.75 s; graph for token range 1–8192 | Same 8,192 number shows up again as the prefill chunk size (§7.4) |
| CUDA graphs | 51 piecewise + 35 full, 5 s, 0.12 GiB | Decode steps replay as one launch |
| Memory | weights + non-torch 1.19 GiB, peak activation 0.55 GiB | |
| KV cache | **6.99 GiB → 610,896 tokens** | 6.99 GiB ÷ 610,896 ≈ **12,288 B/token**: Day-1 formula, within rounding |
| Max concurrency @ 1,024 tokens/request | 596.6× | How many full-length requests fit in the cache at once |
| Engine init | 63 s | One-off cost, excluded from benchmarks |
| Throughput | 3,808 tok/s at 32 × 128 new tokens (≈12-token prompt) | Not comparable to the sweep (short prompt) |
| Warning | Triton JIT of `kernel_unified_attention` during inference | First use of a new shape compiles a kernel; our warmup run absorbs it |

### 6.3 Full sweep (Qwen2.5-0.5B fp16, ol=128, greedy, `ignore_eos`, repeat 3)

TTFT/TPOT/E2E in ms; util/power = NVML mean on GPU 0.

| Run | Tag | bs | pl | tok/s | TTFT p50 | TTFT p99 | TPOT p50 | E2E p99 | Util | Power | SM clock |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | vllm | 1 | 128 | 173.76 | 18.57 | 23.42 | 5.64 | 739 | 99% | 68.1 W | 1164 |
| 2 | vllm | 1 | 512 | 161.19 | 44.88 | 46.48 | 5.89 | 796 | 100% | 66.8 W | 1195 |
| 3 | vllm | 4 | 128 | 668.41 | 36.39 | 40.09 | 5.73 | 768 | 99% | 68.2 W | 1328 |
| 4 | vllm | 4 | 512 | 532.93 | 155.50 | 157.80 | 6.34 | 961 | 99% | 68.3 W | 1332 |
| 5 | vllm | 16 | 128 | 1889.97 | 110.51 | 111.91 | 7.65 | 1087 | 99% | 67.5 W | 1208 |
| 6 | vllm | 16 | 512 | 1157.61 | 542.22 | 573.00 | 9.58 | 1787 | 99% | 69.1 W | 1304 |
| 7 | vllm | 32 | 128 | 2811.72 | 212.30 | 212.40 | 9.80 | 1462 | 99% | 67.3 W | 1145 |
| 8 | vllm | 32 | 512 | 1397.52 | 675.60 | **1202.68** | 17.61 | 2944 | 100% | 67.0 W | 1217 |
| 9 | vllm_eager | 1 | 128 | 37.51 | 55.45 | 56.52 | 26.69 | 3465 | 31% | 40.7 W | 585 |
| 10 | vllm_eager | 1 | 512 | 38.26 | 83.63 | 86.26 | 25.49 | 3392 | 31% | 41.8 W | 759 |
| 11 | vllm_eager | 4 | 128 | 146.27 | 80.18 | 86.18 | 26.88 | 3553 | 30% | 40.1 W | 648 |
| 12 | vllm_eager | 4 | 512 | 143.05 | 213.14 | 217.63 | 26.64 | 3614 | 29% | 50.0 W | 1284 |
| 13 | vllm_eager | 16 | 128 | 573.64 | 136.98 | 148.86 | 27.28 | 3645 | 31% | 46.3 W | 1146 |
| 14 | vllm_eager | 16 | 512 | 513.51 | 598.46 | 603.65 | 26.67 | 4034 | 43% | 58.0 W | 1430 |
| 15 | vllm_eager | 32 | 128 | 1107.80 | 225.29 | 226.79 | 27.53 | 3739 | 36% | 57.1 W | 1416 |
| 16 | vllm_eager | 32 | 512 | 877.18 | 935.67 | 1232.24 | 29.41 | 4682 | 58% | 66.1 W | 1474 |
| 17 | hf_baseline | 1 | 128 | 33.79 | 32.83 | 33.94 | 29.55 | 3816 | 38% | 38.9 W | 913 |
| 18 | hf_baseline | 1 | 512 | 33.84 | 46.35 | 48.38 | 29.46 | 3809 | 35% | 48.3 W | 1328 |
| 19 | hf_baseline | 4 | 128 | 130.20 | 34.76 | 35.98 | 30.69 | 3943 | 36% | 46.3 W | 1263 |
| 20 | hf_baseline | 4 | 512 | 126.15 | 209.47 | 210.96 | 30.29 | 4105 | 49% | 67.6 W | 1537 |
| 21 | hf_baseline | 16 | 128 | 510.00 | 136.85 | 141.91 | 30.79 | 4061 | 55% | 67.3 W | 1500 |
| 22 | hf_baseline | 16 | 512 | 400.28 | 766.06 | 777.40 | 34.26 | 5151 | 97% | 67.2 W | 1126 |
| 23 | hf_baseline | 32 | 128 | 967.63 | 255.11 | 259.50 | 31.39 | 4244 | 84% | 67.4 W | 1300 |
| 24 | hf_baseline | 32 | 512 | 444.06 | 1622.01 | 1643.55 | 59.84 | 9253 | 99% | 66.8 W | 997 |

HF peak torch memory (runs 17–24): 966 → 2,517 MB. Not reported for vLLM, which reserves
`gpu_memory_utilization` × VRAM up front.

### 6.4 Speed-up decomposition (TPOT, the decode step)

| bs | pl | HF TPOT | vLLM eager | vLLM + graphs | HF → eager ("engine/kernels") | eager → graphs ("CUDA graphs") | **Total** |
|---|---|---|---|---|---|---|---|
| 1 | 128 | 29.55 | 26.69 | 5.64 | 1.11× | **4.73×** | **5.24×** |
| 1 | 512 | 29.46 | 25.49 | 5.89 | 1.16× | **4.33×** | **5.00×** |
| 4 | 128 | 30.69 | 26.88 | 5.73 | 1.14× | **4.69×** | **5.36×** |
| 4 | 512 | 30.29 | 26.64 | 6.34 | 1.14× | **4.20×** | **4.78×** |
| 16 | 128 | 30.79 | 27.28 | 7.65 | 1.13× | 3.57× | 4.02× |
| 16 | 512 | 34.26 | 26.67 | 9.58 | 1.28× | 2.78× | 3.58× |
| 32 | 128 | 31.39 | 27.53 | 9.80 | 1.14× | 2.81× | 3.20× |
| 32 | 512 | 59.84 | 29.41 | 17.61 | **2.03×** | 1.67× | 3.40× |

### 6.5 Throughput speed-up (output tok/s vs HF)

| bs | pl | vLLM eager / HF | vLLM / HF |
|---|---|---|---|
| 1 | 128 | 1.11× | **5.14×** |
| 1 | 512 | 1.13× | 4.76× |
| 4 | 128 | 1.12× | 5.13× |
| 4 | 512 | 1.13× | 4.22× |
| 16 | 128 | 1.12× | 3.71× |
| 16 | 512 | 1.28× | 2.89× |
| 32 | 128 | 1.14× | 2.91× |
| 32 | 512 | 1.98× | 3.15× |

Throughput gains are a bit smaller than TPOT gains at large shapes because tok/s also
includes prefill time, which improves less (§7.4).

### 6.6 TTFT comparison (p50, ms)

| bs | pl | Prompt tokens in batch | HF | vLLM eager | vLLM + graphs |
|---|---|---|---|---|---|
| 1 | 128 | 128 | 32.83 | 55.45 | 18.57 |
| 1 | 512 | 512 | 46.35 | 83.63 | 44.88 |
| 4 | 128 | 512 | 34.76 | 80.18 | 36.39 |
| 4 | 512 | 2,048 | 209.47 | 213.14 | 155.50 |
| 16 | 128 | 2,048 | 136.85 | 136.98 | 110.51 |
| 16 | 512 | 8,192 | 766.06 | 598.46 | 542.22 |
| 32 | 128 | 4,096 | 255.11 | 225.29 | 212.30 |
| 32 | 512 | 16,384 | 1622.01 | 935.67 | 675.60 (p99 **1202.68**) |

---

## 7. What the numbers mean

### 7.1 The launch floor is the main story, and CUDA graphs remove it
- **vLLM eager is stuck at ~25–29 ms TPOT at every shape**, just like HF (29–31 ms). A
  completely different codebase with PagedAttention and custom kernels hits the *same* floor
  when it launches kernels one by one from Python. That independently confirms the
  Day-1 diagnosis: decode on this setup is limited by CPU kernel launches.
- NVML agrees: eager vLLM shows 30% util, ~40 W, and the SM clock drops to 585 MHz (the GPU
  is idle enough to down-clock), the same signature as HF at small batch.
- **With CUDA graphs**, the whole decode step is replayed with a single launch: TPOT
  falls to **5.6 ms at bs=1** and NVML shows **99% util, 68 W, at the 70 W cap even at
  bs=1**. The GPU is finally the bottleneck.
- At small batch, CUDA graphs alone account for **4.2–4.7×**; the rest of the engine adds
  only ~1.1×.

### 7.2 At bs=1 vLLM is close to the memory roofline
One decode step must read all weights: 0.99 GB ÷ 5.64 ms ≈ **176 GB/s**, about **72% of the
~245 GB/s** our `vector_add` measured as the T4's practical ceiling. Decode is now in the
regime textbooks describe (memory-bandwidth-bound), which eager PyTorch never reached.

### 7.3 At bs=32 × 512, better attention matters as much as graphs
- HF → vLLM eager: **2.03×** at bs=32 × 512, but only ~1.1× elsewhere. That is the
  attention fix: Day-1 profiling showed HF spending ~40 of its 56 ms per step on
  `repeat_kv` copies, fp32 casts and fp32 GEMV attention. vLLM's TRITON_ATTN reads the
  fp16 paged KV cache directly with native GQA.
- vLLM + graphs TPOT grows 5.6 → 17.6 ms as batch × context grows: that is real
  attention work over ~18k cached tokens, no longer an artefact. (17.6 ms is still far
  from bandwidth-limited; Triton attention on Turing plus power throttling at 1.2 GHz
  are the likely reasons; profile it if it matters.)

### 7.4 Prefill (TTFT) and the first sign of the scheduler
- Small prefills (≤ 512 tokens) are similar for HF and vLLM. **Eager vLLM is slower
  there** (55 vs 33 ms at bs=1): with no graphs or compile, vLLM's per-step scheduling and
  the round-trip to its engine-core process are extra overhead on top of the launches.
- Large prefills: 16,384 tokens take 676 ms in vLLM vs 1,622 ms in HF (**2.4×**), i.e.
  ~24 effective TFLOPS vs ~10, because HF ran prefill attention in fp32 (`volta_sgemm`, Day 1).
- **TTFT p99 ≠ p50 for the first time** (676 vs 1,203 ms at bs=32 × 512). vLLM schedules at
  most ~8,192 tokens per step (the same 8,192 as the compile range in §6.2). 32 × 512 =
  16,384 prompt tokens don't fit in one step, so the batch is prefilled in **two chunks**:
  the first ~16 requests get their first token after one chunk, the rest after the second.
  At bs=16 × 512 = exactly 8,192 tokens, everything fits in one step and p99 ≈ p50.
  This is **chunked prefill**, the first visible behaviour of vLLM's scheduler. (The
  exact budget, `max_num_batched_tokens`, is inferred from the log and the data.)

### 7.5 What did *not* change
- HF baseline on this VM: TPOT 29.5 ms at bs=1 and **59.84 ms at bs=32 × 512**, matching
  Day 1 (§8.6 of DAY1: 59.9 / 59.7 / 60.5). The GPU-bound point is stable across 4 VMs.

---

## 8. Predictions vs measured

| Prediction | Measured | Verdict |
|---|---|---|
| vLLM runs on T4 only with a pinned older version (risk) | Latest 0.31.0 works, TRITON_ATTN backend | ✅ better than feared |
| KV bytes/token = 2·24·2·64·2 = 12,288 B | ≈12,288 B (6.99 GiB / 610,896 tokens) | ✅ |
| CUDA graphs remove the ~30 ms launch floor | TPOT 26.7 (eager) → 5.6 ms (graphs) at bs=1 | ✅ |
| A big gain at bs=32 × 512 from PagedAttention | 2.03× from engine/kernels alone | ✅ |
| Eager vLLM ≈ HF at small batch | 1.11–1.16× | ✅ (not predicted explicitly, but follows from Day 1) |

---

## 9. Caveats and open items

- **Offline mode only**: all requests arrive at t=0. Continuous batching barely matters
  when everybody starts together with equal lengths; its real benefit (and meaningful
  p99) appears with arrivals over time, which is D2.2 (server + Poisson load).
- **`max_num_batched_tokens = 8192`** is inferred, not read from config; the server
  experiments should log the effective engine config.
- `kv_capacity_tokens` is not populated by `_kv_capacity()` in V1 (engine core runs in another
  process). Parse vLLM's startup log or scrape `/metrics` in D2.3 instead.
- Run 8/16 TTFT p99 depends on chunking, so p99 there reflects scheduling, not noise.
- Two Python environments (3.12 vs 3.13, torch 2.10 vs 2.13). Defensible because each
  backend runs in its intended stack, but note it when quoting results.
- Remaining Day-2 work: D2.2 server + load test, D2.3 KV observatory, D2.4 TP=2 vs
  replicas, D2.5 simulator.

---

## 10. Glossary (new terms)

- **vLLM**: open-source LLM serving engine (PagedAttention, continuous batching, CUDA graphs).
- **V1 engine / EngineCore**: vLLM's current architecture; the scheduler + model runner
  ("engine core") runs in a **separate process** from the Python front-end that adds
  requests and receives outputs.
- **PagedAttention**: KV cache stored in fixed-size **blocks** (like OS memory pages)
  instead of one contiguous tensor per sequence: no fragmentation, no copying when a
  sequence grows, and memory freed instantly when it finishes.
- **KV block / block_size**: one page of the KV cache, holding `block_size` tokens (typically 16).
- **TRITON_ATTN / FLEX_ATTENTION**: vLLM attention backends written in Triton / built on
  PyTorch FlexAttention; used on GPUs where FlashAttention (sm_80+) isn't available.
- **Triton**: a Python-like language for writing GPU kernels, compiled just-in-time.
- **JIT compilation warning**: a kernel compiled the first time a new shape appears;
  causes a one-off latency spike.
- **torch.compile**: PyTorch's compiler; fuses operations into fewer, faster kernels. vLLM
  compiles the model at startup.
- **CUDA graphs (piecewise / full)**: a recorded sequence of kernels replayed with one launch.
  *Full*: the whole forward pass; *piecewise*: the model split into pieces (e.g. around
  attention) and each piece captured, which is more flexible for varying shapes.
- **enforce_eager**: vLLM option that disables CUDA graphs (and compilation), running
  the model op by op. Used here as the control experiment.
- **gpu_memory_utilization**: fraction of total VRAM vLLM may use (weights + activations +
  KV cache); whatever is left after weights and activations becomes KV cache.
- **Maximum concurrency**: KV cache tokens ÷ tokens per request: how many full-length
  requests can be resident at once.
- **Chunked prefill**: splitting a large prefill across several engine steps, so one long
  prompt (or a big batch of prompts) doesn't block decoding of everyone else.
- **max_num_batched_tokens**: the per-step token budget the scheduler may process
  (prefill chunks + decode tokens).
- **SamplingParams / RequestOutputKind.DELTA**: per-request generation settings; DELTA =
  each output contains only tokens produced since the last step.
- **Prefix caching**: reusing KV blocks of identical prompt prefixes across requests;
  disabled here so every prompt is computed from scratch.
- **uv / virtual environment (venv)**: uv is a fast Python package installer; a venv is an
  isolated set of packages with its own interpreter, so vLLM's pins can't break Kaggle's.
- **Editable install**: the venv and the system both point to the same source folder.
- **sitecustomize**: a module Python imports at startup; Kaggle uses one, hence the harmless
  `wrapt` warning inside our venv.
