"""KV cache gather, SnapKV-driven compact, and per-layer compression driver."""

from __future__ import annotations

import torch

from nanovllm.layers.CompressMethod import DEFAULT_COMPRESS_FN
from nanovllm.utils.context import get_context


def get_tail_window_and_tail_slots(
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    seq_idxs: torch.Tensor,
    block_size: int,
    window_blocks: int,
):
    device = block_tables.device
    old_context_lens = context_lens.index_select(0, seq_idxs).to(torch.long)
    selected_block_tables = block_tables.index_select(0, seq_idxs).to(torch.long)

    B = block_size
    m = seq_idxs.numel()
    full_blocks = old_context_lens // B
    tail_lens = old_context_lens % B

    if torch.any(full_blocks < window_blocks):
        return None

    block_offsets = torch.arange(window_blocks, device=device, dtype=torch.long).view(1, -1)
    window_block_idx = (full_blocks - window_blocks).unsqueeze(1) + block_offsets
    window_block_ids = torch.gather(selected_block_tables, 1, window_block_idx)

    token_offsets = torch.arange(B, device=device, dtype=torch.long).view(1, 1, B)
    window_src_slots = window_block_ids.unsqueeze(-1) * B + token_offsets
    window_src_slots = window_src_slots.reshape(m, window_blocks * B)

    max_blocks = selected_block_tables.size(1)
    safe_tail_block_idx = torch.clamp(full_blocks, max=max_blocks - 1)
    tail_block_ids = torch.gather(
        selected_block_tables, 1, safe_tail_block_idx.unsqueeze(1),
    ).squeeze(1)
    tail_block_ids = torch.where(
        tail_lens > 0,
        tail_block_ids,
        torch.full_like(tail_block_ids, -1),
    )
    return window_src_slots, old_context_lens, tail_lens, tail_block_ids


def gather_kv_by_slots(k_cache: torch.Tensor, v_cache: torch.Tensor, src_slots: torch.Tensor):
    num_blocks, block_size, num_kv_heads, head_dim = k_cache.shape
    total_slots = num_blocks * block_size
    k_flat = k_cache.view(total_slots, num_kv_heads, head_dim)
    v_flat = v_cache.view(total_slots, num_kv_heads, head_dim)
    k_batch = k_flat[src_slots].permute(0, 2, 1, 3).contiguous()
    v_batch = v_flat[src_slots].permute(0, 2, 1, 3).contiguous()
    return k_batch, v_batch


def compact_kv_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    src_flat: torch.Tensor,
    dst_flat: torch.Tensor,
):
    num_blocks, block_size, num_kv_heads, head_dim = k_cache.shape
    total_slots = num_blocks * block_size
    d = num_kv_heads * head_dim
    k_flat = k_cache.reshape(total_slots, d)
    v_flat = v_cache.reshape(total_slots, d)
    vals_k = k_flat.index_select(0, src_flat).clone()
    vals_v = v_flat.index_select(0, src_flat).clone()
    k_flat.index_copy_(0, dst_flat, vals_k)
    v_flat.index_copy_(0, dst_flat, vals_v)


def compress_compact(
    q_current: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    layer_id: int,
    block_size: int,
    window_blocks: int,
    keep_blocks: int,
    keep_extra_tokens: int,
    num_layers: int,
    snap_window: int,
    compress_fn=DEFAULT_COMPRESS_FN,
    context=None,
) -> bool:
    if context is None:
        context = get_context()

    if context.is_prefill or context.context_lens is None or context.block_tables is None:
        return False

    selected = context.compress_selected_batch_indices
    if not selected:
        return False

    device = k_cache.device
    seq_idxs = torch.tensor(selected, dtype=torch.long, device=device)
    m = seq_idxs.numel()
    if m == 0:
        return False

    B = block_size
    window_tokens = window_blocks * B
    keep_tokens = keep_blocks * B + keep_extra_tokens

    q_sub = q_current.index_select(0, seq_idxs).unsqueeze(2)
    base_context_lens = context.compress_base_context_lens
    assert base_context_lens is not None

    tail_info = get_tail_window_and_tail_slots(
        context.block_tables,
        base_context_lens,
        seq_idxs,
        B,
        window_blocks,
    )
    if tail_info is None:
        return False

    window_src_slots, old_context_lens, tail_lens, tail_block_ids = tail_info
    k_sub, v_sub = gather_kv_by_slots(k_cache, v_cache, window_src_slots)

    num_keep = keep_tokens - 2
    if num_keep < 1:
        return False

    keep_idx = compress_fn(q_sub, k_sub, v_sub, num_keep=num_keep, window=snap_window)
    if keep_idx is False:
        return False

    if keep_idx.dim() == 3:
        keep_idx = keep_idx.squeeze(1)
    if keep_idx.size(1) != keep_tokens:
        return False

    src_keep = torch.gather(window_src_slots, 1, keep_idx)
    new_context_lens_tensor = old_context_lens - window_tokens + keep_tokens
    dst_keep = window_src_slots[:, :keep_tokens]

    src_keep_flat = src_keep.reshape(-1)
    dst_keep_flat = dst_keep.reshape(-1)
    tail_total = int(tail_lens.sum().item())

    if tail_total > 0:
        tail_seq_ids = torch.repeat_interleave(
            torch.arange(m, device=device, dtype=torch.long), tail_lens,
        )
        tail_offsets = torch.cat(
            [torch.arange(int(t.item()), device=device, dtype=torch.long) for t in tail_lens]
        )
        src_tail_flat = tail_block_ids[tail_seq_ids] * B + tail_offsets
        dst_tail_start = dst_keep[:, -1] + 1
        dst_tail_flat = dst_tail_start[tail_seq_ids] + tail_offsets
        src_flat = torch.cat([src_keep_flat, src_tail_flat], dim=0)
        dst_flat = torch.cat([dst_keep_flat, dst_tail_flat], dim=0)
    else:
        src_flat = src_keep_flat
        dst_flat = dst_keep_flat

    compact_kv_cache(k_cache, v_cache, src_flat, dst_flat)
    context.context_lens[seq_idxs] = new_context_lens_tensor.to(context.context_lens.dtype)

    if layer_id + 1 >= num_layers:
        if context.compression_events is None:
            context.compression_events = []
        selected_block_tables = context.block_tables.index_select(0, seq_idxs).to(torch.long)
        keep_blocks_after_tensor = (new_context_lens_tensor + B - 1) // B
        for i, bidx in enumerate(seq_idxs.tolist()):
            keep_blocks_after = int(keep_blocks_after_tensor[i].item())
            freed = selected_block_tables[i, keep_blocks_after:]
            freed_block_ids = [int(x) for x in freed.tolist() if int(x) >= 0]
            context.compression_events.append({
                "batch_index": int(bidx),
                "layer": int(layer_id),
                "new_context_len": int(new_context_lens_tensor[i].item()),
                "keep_blocks": int(keep_blocks_after),
                "freed_block_ids": freed_block_ids,
            })

    return True


def kv_cache_compress(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    layer_id: int,
    num_layers: int,
    block_size: int,
    compress_n: int,
    snap_window: int = 1,
    compress_fn=DEFAULT_COMPRESS_FN,
) -> bool:
    context = get_context()
    if not context.kv_compress_enabled:
        return False
    period = context.kv_compress_period
    if period > 0:
        window_tokens = period
        keep_tokens = max(3, int(window_tokens * context.kv_compress_ratio))
        window_blocks = window_tokens // block_size
        keep_blocks = keep_tokens // block_size
        keep_extra_tokens = keep_tokens % block_size
    else:
        window_blocks = compress_n + 1
        keep_blocks = compress_n
        keep_extra_tokens = 1
        keep_tokens = keep_blocks * block_size + keep_extra_tokens
    if keep_tokens < 3:
        return False
    return compress_compact(
        q,
        k_cache,
        v_cache,
        layer_id,
        block_size,
        window_blocks,
        keep_blocks,
        keep_extra_tokens,
        num_layers,
        snap_window,
        compress_fn=compress_fn,
    )
