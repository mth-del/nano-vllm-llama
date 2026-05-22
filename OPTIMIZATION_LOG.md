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

## [2026-05-17] CPU Prefill / GPU Decode 分离（pd_separation）

### Phase 1 — 控制面与状态机
- Scope: `config`, `scheduler`, `sequence` 调度路径
- Change:
  - `Config.pd_separation: bool = False`（要求 `tensor_parallel_size=1`）
  - `Scheduler.schedule_cpu_prefill()`：为 waiting 序列在 GPU 上 `allocate` block，移入 `running`
  - `Scheduler.schedule()` 在 PD 模式下 **跳过 GPU prefill**，仅 decode
- Why: 将 Prefill 与 Decode 在调度层解耦，为跨设备执行做准备
- Validation: `python -m compileall nanovllm` 通过
- Notes: 默认关闭，不影响现有单卡路径

### Phase 2 — CPU Prefill + KV handoff
- Scope: `cpu_prefill_runner`, `kv_transfer`, `attention`, `model_runner.import_kv`
- Change:
  - 新增 `CPUPrefillRunner`：CPU 上加载同结构模型，用 `cpu_prefill_capture` + `scaled_dot_product_attention` 跑 prompt
  - 每层捕获 `(K,V)`，经 `import_kv_to_gpu()` 按 `block_table`/`slot_mapping` 写入 GPU `kv_cache`
  - `ModelRunner.import_kv()` 封装 H2D `index_copy_`
  - `Attention` 增加 `_cpu_prefill_attention`；`context` 增加 `cpu_prefill_capture` / `kv_captures`
  - 修复 `get_rope()` 全局 `lru_cache` 导致 CPU/GPU 共享 RoPE buffer 的设备冲突；RoPE `cos_sin` 按 `query.device` 对齐
- Why: Prefill 算力放 CPU，GPU 显存主要用于 Decode KV 与高 batch decode
- Validation: PD 路径单请求可跑通（Qwen3-0.6B）
- Notes: PD 模式会在 Host 再加载一份 CPU 权重，系统 RAM 占用上升

### Phase 4 — 延迟 GPU KV 分配（waiting 不占 GPU block）

- Scope: `scheduler`, `sequence`, `llm_engine._step_pd`
- Change:
  - 三队列：`waiting` →（CPU prefill）→ `prefill_ready`（host KV，`cpu_kv_layers` + 可选 `pin_memory`）→（`allocate` + `import_kv`）→ `running`（仅 decode 占 GPU block）
  - `schedule_cpu_prefill()` 不再 `block_manager.allocate`
  - 新增 `schedule_gpu_handoff()`：仅当 `can_allocate` 成功才占 GPU KV
  - `Sequence.cpu_kv_layers` 保存 handoff 前 KV；handoff 后置 `None` 释放 host
  - `is_finished` 包含 `prefill_ready` 非空
- Why: 在 GPU KV 紧张时，让大量未完成 handoff 的请求只占用 **CPU 内存**，把 **有限 GPU block** 留给正在 decode 的序列
- Validation: `validate_pd_separation.py` 仍 **token 完全一致**
- Notes: `prefill_ready` 积压会升高 **Host RAM**；可用 `scripts/stress_kv_capacity.py` 压测

```bash
python scripts/stress_kv_capacity.py ~/huggingface/Qwen3-0.6B/ \
  --num-reqs 16 --prompt-tokens 1024 --max-tokens 128 --gpu-util 0.55
```

### Phase 3 — LLMEngine 集成与首 token 对齐
- Scope: `llm_engine._step_pd`
- Change:
  - 每步优先 `schedule_cpu_prefill` → CPU prefill → `import_kv` → **同一 step 内 GPU decode** 采首 token
  - 首 token 在 GPU 上采样（与 baseline 一致），避免 CPU/GPU 数值差导致分叉
  - 后续 step 走常规 `schedule()` decode
- Why: 保证与 unified GPU 路径输出一致，仅 Prefill 计算换设备
- Validation: 见下方端到端对比

### 性能对比 baseline vs PD（Qwen3-0.6B，单卡，enforce_eager）

命令（建议分场景或一次跑全）：

```bash
python scripts/benchmark_pd.py ~/huggingface/Qwen3-0.6B/ --cases short,prompt512,prompt2k
# 或单独：--cases prompt512
```

环境：本地 RTX GPU；`max_tokens=256`（prompt512 / prompt2k）；短场景 `max_tokens=64`。

| 场景 | prompt | out | TTFT baseline | TTFT PD | TTFT 倍率 | TPOT baseline | TPOT PD | e2e baseline | e2e PD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| short | ~54 | 64 | 24 ms | 275 ms | **11.6×** | 19.5 ms | 20.1 ms | 51.0 tok/s | 41.5 tok/s |
| prompt512 | 514 | 256 | 32 ms | 2361 ms | **74.4×** | 19.8 ms | 20.0 ms | 50.3 tok/s | 34.3 tok/s |
| prompt2k | 2078 | 256 | 144 ms | 9954 ms | **69.1×** | 19.0 ms | 19.6 ms | 51.4 tok/s | 17.1 tok/s |

**结论**：
- **TTFT**：PD 随 prompt 变长急剧恶化（2k prompt 时 CPU prefill + KV H2D ≈ **10s** vs GPU **144ms**）。瓶颈在 **CPU 算子（SDPA）+ 跨设备 KV 拷贝**，不是 decode。
- **TPOT**：三种场景下几乎相同（~19–20ms/token），与「decode 仍走同一 GPU 路径」一致。
- **端到端**：输出 256 token 时，prompt 越长 PD 相对越亏（2k 场景 e2e 仅 baseline 的 **33%**）；短输出时 PD 约慢 **18–32%**。

**说明**：
- 长 prompt 未做 CPU chunked prefill；`import_kv` 一次性 H2D 全部层 KV。
- 多场景同进程跑可能触发 prefix-cache 边界问题；长 prompt 场景已默认 **跳过 in-LLM warmup**，建议分 `--cases` 跑或一次只测一个场景。
- `prepare_prefill` 已加 `end_block` 与 `block_table` 长度对齐，避免极端 chunked 下标越界。

### 端到端正确性验证（Qwen3-0.6B，prompt 一句话，max_tokens=16，seed=42）

```bash
python scripts/validate_pd_separation.py ~/huggingface/Qwen3-0.6B/
```

| 指标 | 结果 |
|---|---|
| baseline vs `pd_separation=True` token_ids | **完全一致**（16/16） |
| 命令退出码 | 0 |
| 注意 | 两次运行间需 `gc` + `cuda.empty_cache()`；PD 建议 `gpu_memory_utilization=0.85`（CPU+GPU 双份权重） |

### 使用方式

```python
llm = LLM(model_path, pd_separation=True, enforce_eager=True, tensor_parallel_size=1)
```

### 已知限制 / 后续
- 未做 chunked CPU prefill、异步 H2D、多请求 PD 流水线
- `tensor_parallel_size > 1` 未支持
- 长 prompt（如 48k）需单独评估 CPU 时延与 KV 传输带宽
- 可选：Phase 4 用 `ncu` 对比 PD 下 GPU SM 占用与 decode 吞吐

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
