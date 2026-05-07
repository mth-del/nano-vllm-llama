import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


# TPB (Tokens Per Block): 每个 block 处理的 token 数。
# 原版 TPB=1，grid=(N,)，N 很小时大量 SM 空转（实测 SM 利用率仅 2.7%）。
# 增大 TPB 后 grid 缩小为 ceil(N/TPB)，每个 block 串行处理 TPB 个 token，
# 每个 block 的工作量增大，启动开销摊薄，SM 利用率显著提升。
_TPB = 4


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


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    grid = ((N + _TPB - 1) // _TPB,)
    store_kvcache_kernel[grid](
        key, key.stride(0), value, value.stride(0),
        k_cache, v_cache, slot_mapping,
        N, D, _TPB,
    )


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
