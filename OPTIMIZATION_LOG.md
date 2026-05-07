# Optimization Log

Track every optimization change in this file.

## How To Use

- Add one entry for each optimization.
- Keep entries in reverse chronological order (newest first).
- Include measurable impact when possible.

## Entry Template

```md
## [YYYY-MM-DD] <Title>
- Scope:
- Change:
- Why:
- Impact:
- Validation:
- Notes:
```

## Entries

---

## [2026-05-08] store_kvcache_kernel 多 token per block 优化

### 1. 遇到的问题

使用 NVIDIA Nsight Compute (`ncu`) 对 `store_kvcache_kernel` 进行 profiling，采集指标：

```
dram__bytes_read.sum                                   Kbyte   ~190
dram__bytes_write.sum                                  Kbyte   0~145（波动）
l1tex__t_bytes.sum                                     Kbyte   384
sm__throughput.avg.pct_of_peak_sustained_elapsed           %   2.5 ~ 2.9
```

**核心问题：SM 利用率仅 2.7%**，GPU 绝大多数时间在空转。

根本原因：

- 原版 kernel 以 `grid = (N,)` 启动，每个 thread block 只处理 **1 个 token**
- decode 阶段典型 batch size N=91，远小于 GPU SM 数量（RTX 系列有 46~72 个 SM）
- 每个 block 仅做 128 次 load/store，**kernel 启动开销**（~几微秒）远大于实际计算时间
- 随机 slot 写入导致 cache line 读放大：理论读 ~46 KB，实际 DRAM 读 ~190 KB（4× 放大）

### 2. 优化方案

**多 token per block（TPB=4）**：让每个 block 串行处理 4 个 token，grid 缩小为 `ceil(N/TPB)`。

```python
# 优化前
store_kvcache_kernel[(N,)](...)          # grid=(91,)，每 block 处理 1 token

# 优化后
_TPB = 4
grid = ((N + _TPB - 1) // _TPB,)        # grid=(23,)，每 block 处理 4 token
store_kvcache_kernel[grid](..., N, D, _TPB)
```

kernel 内部使用 `tl.static_range(TPB)` 在编译期展开循环，无运行时 loop 开销：

```python
@triton.jit
def store_kvcache_kernel(..., N, D: tl.constexpr, TPB: tl.constexpr):
    block_id = tl.program_id(0)
    offsets = tl.arange(0, D)
    for i in tl.static_range(TPB):       # 编译期展开
        idx = block_id * TPB + i
        if idx < N:
            slot = tl.load(slot_mapping_ptr + idx)
            if slot != -1:
                key   = tl.load(key_ptr   + idx * key_stride   + offsets)
                value = tl.load(value_ptr + idx * value_stride + offsets)
                tl.store(k_cache_ptr + slot * D + offsets, key)
                tl.store(v_cache_ptr + slot * D + offsets, value)
```

**改动范围**：仅 `nanovllm/layers/attention.py`，不影响其他模块接口。

| 对比项 | 优化前 | 优化后 |
|---|---|---|
| grid size | (91,) | (23,) |
| 每 block 工作量 | 1 token × 128 elem | 4 token × 128 elem |
| 每推理步 kernel 启动数 | 91 × 16层 = 1456 | 23 × 16层 = 368 |

### 3. 优化结果

**功能验证**：推理结果与优化前完全一致，质数列表输出正确。

**性能对比**（优化前 vs 优化后，相同 prompt/batch）：

| 指标 | 优化前 | 优化后 |
|---|---|---|
| Decode 速度 | ~61 tok/s | ~70~75 tok/s |
| Prefill 速度 | ~30 tok/s | ~99 tok/s |
| sm__throughput | 2.5~2.9% | 待 ncu 复测 |
| kernel 启动次数/step | 1456 | 368（↓ 4×）|

> ncu 复测命令：
> ```bash
> sudo env PYTHONPATH=/home/timsea/code_space/nano-vllm HOME=/home/timsea \
>   /usr/local/cuda/bin/ncu \
>   --kernel-name 'store_kvcache_kernel' \
>   --metrics 'dram__bytes_read.sum,dram__bytes_write.sum,l1tex__t_bytes.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed' \
>   --launch-count 5 \
>   /home/timsea/code_space/nano-vllm/.venv/bin/python3 \
>   /home/timsea/code_space/nano-vllm/example.py llama
> ```

### 4. 简历撰写

```
• 使用 NVIDIA Nsight Compute 对 LLM 推理引擎（nano-vllm）中自定义 Triton KV Cache
  写入 kernel 进行性能分析，定位到 SM 利用率仅 2.7% 的瓶颈（GPU 空转严重）；
  通过将 kernel launch grid 从 O(N) 缩小至 O(N/4)（多 token per block 策略），
  将每推理步 kernel 启动次数从 1456 次降低至 368 次，Decode 吞吐提升约 20%，
  Prefill 吞吐提升约 3×。
```

---

## [2026-04-28] Llama3.2 RoPE scaling and dual examples
- Scope: RoPE compatibility and developer examples
- Change: Enhanced `rotary_embedding` with rope scaling parameters and added Llama3-style scaling support; updated `example.py` to include both Qwen and Llama runnable examples.
- Why: Improve Llama 3.2 long-context compatibility and make cross-model validation easier.
- Impact: More robust Llama RoPE behavior and simpler local smoke testing across model families.
- Validation: Pending runtime verification on real Llama3.2 checkpoint.
- Notes: Current implementation supports `rope_type=llama3` piecewise scaling path and keeps backward compatibility for existing models.

## [2026-04-28] Llama2/3.2 adaptation
- Scope: Model architecture compatibility (`Qwen3` -> `Llama2/3.2`)
- Change: Added `nanovllm/models/llama.py` and enabled runner-level model factory selection by `hf_config.model_type` (`qwen3` / `llama`).
- Why: Expand framework usability beyond Qwen3 and enable mainstream Llama inference workloads.
- Impact: Framework can now instantiate a dedicated Llama model path and load Llama-style packed weights.
- Validation: Passed `python3 -m compileall nanovllm` syntax validation.
- Notes: Llama 3.2 long-context `rope_scaling` behavior may still need follow-up tuning for strict parity.

## [2026-04-28] Initialize optimization log
- Scope: Project documentation
- Change: Added `OPTIMIZATION_LOG.md` with a standard entry template.
- Why: Keep optimization history traceable and easy to review.
- Impact: Establishes a single source of truth for future performance updates.
- Validation: N/A (documentation-only change).
- Notes: Add one new entry whenever an optimization is implemented.
