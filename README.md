



# Kmini-vLLM

A lightweight vLLM-style inference engine built from scratch, with ongoing kernel- and scheduler-level optimizations.

## Key Features

- **Fast offline inference** — PagedAttention, continuous batching, prefix caching, Tensor Parallelism, Torch compile, CUDA Graph
- **Readable codebase** — Core engine in ~1,200 lines of Python; hot paths extended via `nanovllm/ops/`
- **Multi-model** — Qwen3, Llama 2/3.2 (`model_type`: `qwen3` / `llama`)
- **Prefill MLP fusion** — Fused `gate_up + SiLU×Mul` on prefill (cuBLAS GEMM + CUDA / Triton epilogue)
- **Decode-only SnapKV** — KV cache compression with one-shot or periodic triggers
- **Optional PD separation** — CPU prefill + GPU decode (experimental; see trade-offs below)

Full change history and benchmark tables: **[OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md)**.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
# For CUDA fused MLP extension (recommended on sm_120 / RTX 50-series):
pip install ninja
```

## Model & Dataset Download

```bash
# Qwen3-0.6B (smoke / KV-compress demos)
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False

# MATH-500 (evaluation)
export HF_ENDPOINT=https://hf-mirror.com   # optional mirror
huggingface-cli download HuggingFaceH4/MATH-500 --repo-type dataset \
  --local-dir ~/huggingface/MATH-500
```

## Quick Start

```python
from nanovllm import LLM, SamplingParams

llm = LLM("~/huggingface/Qwen3-0.6B/", enforce_eager=True, tensor_parallel_size=1)
outputs = llm.generate(
    ["Hello, Nano-vLLM."],
    SamplingParams(temperature=0.6, max_tokens=256),
)
print(outputs[0]["text"])
llm.exit()
```

Run bundled examples:

```bash
python example.py qwen
python example.py llama
```

### Optional: KV compression (SnapKV)

```python
llm = LLM(
    model_path,
    kv_compress=True,
    kv_compress_period=1024,   # 0 = one-shot at context_len == block_size*(N+1)-1
    kv_compress_ratio=0.5,
    enforce_eager=True,        # required when kv_compress=True
)
```

### Optional: Prefill MLP fusion (GEMM backend)

```bash
export NANOVLLM_FUSED_MLP=1
export NANOVLLM_GEMM_BACKEND=auto   # auto | ref | cuda | triton
# On RTX 5090 (sm_120), auto prefers cuda (Triton tl.dot may not compile)
export TORCH_CUDA_ARCH_LIST=12.0  # optional: faster first-time extension build
```

Fusion applies only during **prefill** (`ctx.is_prefill`); decode still uses the standard MLP path.

### Optional: CPU prefill / GPU decode

```python
llm = LLM(model_path, pd_separation=True, enforce_eager=True, tensor_parallel_size=1)
```

Long prompts are much slower on CPU; see PD benchmarks in [OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md).

## Optimization Highlights

Summary of recent work (details, diagrams, and raw numbers in [OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md)).

### 1. Prefill MLP fusion (`nanovllm/ops/`)

SwiGLU MLP on prefill: merge `gate_up_proj` + `SiLU(gate)×up`.


| Path       | Flow                                                   |
| ---------- | ------------------------------------------------------ |
| **ref**    | `F.linear` → `[N, 2×inter]` → `split` → `silu` → mul   |
| **cuda**   | `F.linear` (cuBLAS) → `silu_mul_kernel` → `[N, inter]` |
| **triton** | Fused `tl.dot` kernel (fallback if unsupported)        |


**MATH-500, 50 problems** — RTX 5090 + Qwen2.5-3B, `max_tokens=512`, `batch_size=16`:


| Backend | Accuracy      | Total  | Prefill (5806 prompt tok) | Decode            |
| ------- | ------------- | ------ | ------------------------- | ----------------- |
| ref     | 54.0% (27/50) | 58.4 s | 1.0 s, 5595 tok/s         | 57.3 s, 367 tok/s |
| cuda    | 54.0% (27/50) | 49.1 s | 0.2 s, 25693 tok/s        | 48.8 s, 448 tok/s |


Prefill ~~4.6× with identical prompt tokens; end-to-end −16% (decode unchanged by design; decode delta includes length / warmup effects). Micro gate_up+SiLU alone: **~~1.03–1.07×** vs ref.

```bash
python scripts/benchmark_gemm.py /path/to/Qwen2.5-3B-Instruct
bash scripts/run_math500_gemm.sh
```

### 2. SnapKV KV cache compression (decode-only)

After `store_kvcache`, before `flash_attn_with_kvcache`: gather window → SnapKV scoring → compact KV → truncate blocks.


| Mode     | Trigger                                                                            |
| -------- | ---------------------------------------------------------------------------------- |
| One-shot | `kv_compress_period=0`, at `context_len == block_size×(N+1)−1` (default N=1 → 511) |
| Periodic | every `period` new tokens, keep `ratio × period` tokens                            |


**MATH-500, 50 problems** — Qwen2.5-3B, `period=0`, `max_tokens=512`: compression rarely fires; use `period=1024` on longer outputs for stable triggers (see Qwen3-0.6B repro in the log).

```bash
python scripts/validate_kv_compress.py ~/huggingface/Qwen3-0.6B/
bash scripts/run_math500_qwen25_3b.sh --limit 50 --modes baseline,compress
```

### 3. `store_kvcache` Triton kernel (TPB=4)

Nsight Compute showed ~2.7% SM utilization (one token per block). **4 tokens per block** cut launches ~4×; decode ~61→70–75 tok/s, prefill ~30→99 tok/s on the profiled setup.

### 4. CPU prefill / GPU decode (`pd_separation`)

Three-queue scheduler: `waiting` → CPU prefill → `prefill_ready` → GPU handoff → decode. Correctness: token-aligned with unified GPU path; **TTFT** suffers on long prompts (CPU SDPA + H2D).

```bash
python scripts/validate_pd_separation.py ~/huggingface/Qwen3-0.6B/
python scripts/benchmark_pd.py ~/huggingface/Qwen3-0.6B/ --cases short,prompt512
```

## Evaluation Scripts


| Script                             | Purpose                                                                                                   |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `scripts/eval_math500.py`          | MATH-500 accuracy + prefill/decode timing; `--gemm-backends`, `--modes baseline,compress`, `--repro-blog` |
| `scripts/benchmark_gemm.py`        | Micro + e2e prefill GEMM comparison                                                                       |
| `scripts/run_math500_gemm.sh`      | MATH-500 ref vs cuda GEMM (50 problems)                                                                   |
| `scripts/run_math500_qwen25_3b.sh` | MATH-500 baseline vs KV compress (Qwen2.5-3B)                                                             |
| `scripts/validate_kv_compress.py`  | Short-output token parity smoke test                                                                      |
| `scripts/benchmark_pd.py`          | PD separation TTFT / TPOT / e2e                                                                           |
| `bench.py`                         | Throughput vs vLLM (see below)                                                                            |


Example:

```bash
PYTHONPATH=. python scripts/eval_math500.py ~/huggingface/Qwen2.5-3B-Instruct \
  --limit 50 --max-tokens 512 --batch-size 16 \
  --gemm-backends ref,cuda --modes baseline \
  --out results/math500_gemm_ref_cuda_50.json
```

## Benchmark (vs vLLM)

See `bench.py` for the original comparison.

**Configuration:** RTX 4070 Laptop (8GB), Qwen3-0.6B, 256 requests, random input/output 100–1024 tokens.


| Engine    | Output tokens | Time (s) | Throughput (tok/s) |
| --------- | ------------- | -------- | ------------------ |
| vLLM      | 133,966       | 98.37    | 1361.84            |
| Nano-vLLM | 133,966       | 93.41    | 1434.13            |


Newer optimizations (GEMM fusion, SnapKV, TPB kernel) are documented separately in [OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md) on RTX 5090 / Qwen2.5-3B and other setups.

## Environment Variables


| Variable                | Values                          | Description                           |
| ----------------------- | ------------------------------- | ------------------------------------- |
| `NANOVLLM_GEMM_BACKEND` | `auto`, `ref`, `cuda`, `triton` | Prefill fused MLP backend             |
| `NANOVLLM_FUSED_MLP`    | `1` / `0`                       | Enable fused prefill MLP              |
| `TORCH_CUDA_ARCH_LIST`  | e.g. `12.0`                     | CUDA extension arch for RTX 50-series |


## Project Layout

```
nanovllm/
  engine/          # scheduler, model_runner, block_manager
  layers/          # attention, linear, embed, layernorm
  models/          # qwen3.py, llama.py
  ops/             # gemm.py, kv_cache.py, cuda/fused_mlp.*
scripts/           # eval, benchmarks, validation
OPTIMIZATION_LOG.md
```

## Reference Documents
[kv_Cache优化](https://github.com/TheToughCrane/nano-kvllm)
## Star History

[Star History Chart](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)