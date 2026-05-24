"""Optimized KV cache store kernel (Triton)."""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


# TPB (Tokens Per Block): 每个 block 处理的 token 数。
# 原版 TPB=1，grid=(N,)，N 很小时大量 SM 空转（实测 SM 利用率仅 2.7%）。
# 增大 TPB 后 grid 缩小为 ceil(N/TPB)，每个 block 串行处理 TPB 个 token，
# 每个 block 的工作量增大，启动开销摊薄，SM 利用率显著提升。
# 压测对比: NANOVLLM_KV_TPB=1|4 python scripts/benchmark_tpb.py ...
def get_kv_tpb() -> int:
    tpb = int(os.environ.get("NANOVLLM_KV_TPB", "4"))
    if tpb < 1:
        raise ValueError(f"NANOVLLM_KV_TPB must be >= 1, got {tpb}")
    return tpb


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    N,
    D: tl.constexpr,
    TPB: tl.constexpr,
):
    block_id = tl.program_id(0)
    offsets = tl.arange(0, D)
    for i in tl.static_range(TPB):
        idx = block_id * TPB + i
        if idx < N:
            slot = tl.load(slot_mapping_ptr + idx)
            if slot != -1:
                key = tl.load(key_ptr + idx * key_stride + offsets)
                value = tl.load(value_ptr + idx * value_stride + offsets)
                tl.store(k_cache_ptr + slot * D + offsets, key)
                tl.store(v_cache_ptr + slot * D + offsets, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    tpb = get_kv_tpb()
    grid = ((N + tpb - 1) // tpb,)
    store_kvcache_kernel[grid](
        key, key.stride(0), value, value.stride(0),
        k_cache, v_cache, slot_mapping,
        N, D, tpb,
    )
