import torch
from torch import nn
import torch.nn.functional as F

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.layers.compress_utils import kv_cache_compress
from nanovllm.ops.kv_cache import store_kvcache
from nanovllm.utils.context import get_context


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

    def _cpu_prefill_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        context = get_context()
        context.kv_captures.append((k.detach(), v.detach()))
        if self.num_heads != self.num_kv_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        qh = q.transpose(0, 1).unsqueeze(0)
        kh = k.transpose(0, 1).unsqueeze(0)
        vh = v.transpose(0, 1).unsqueeze(0)
        o = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, scale=self.scale)
        return o.squeeze(0).transpose(0, 1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        if context.cpu_prefill_capture:
            return self._cpu_prefill_attention(q, k, v)
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if (
            k_cache.numel()
            and v_cache.numel()
            and context.kv_compress_enabled
            and not context.is_prefill
            and context.num_hidden_layers > 0
        ):
            kv_cache_compress(
                q,
                k_cache,
                v_cache,
                context.current_layer_id,
                context.num_hidden_layers,
                context.kvcache_block_size,
                context.kv_compress_n,
                context.kv_compress_snap_window,
            )  # period/ratio read from context inside kv_cache_compress
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
