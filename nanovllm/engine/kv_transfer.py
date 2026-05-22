"""KV layout helpers for CPU prefill -> GPU paged cache handoff."""

import torch

from nanovllm.engine.sequence import Sequence


def build_slot_mapping(seq: Sequence, block_size: int, start: int = 0, end: int | None = None) -> list[int]:
    end = end if end is not None else len(seq)
    slot_mapping = []
    start_block = start // block_size
    end_block = (end + block_size - 1) // block_size
    for i in range(start_block, end_block):
        slot_start = seq.block_table[i] * block_size
        if i == start_block:
            slot_start += start % block_size
        if i != end_block - 1:
            slot_end = seq.block_table[i] * block_size + block_size
        else:
            slot_end = seq.block_table[i] * block_size + end - i * block_size
        slot_mapping.extend(range(slot_start, slot_end))
    return slot_mapping


def import_kv_to_gpu(
    kv_cache: torch.Tensor,
    cpu_kv_layers: list[tuple[torch.Tensor, torch.Tensor]],
    seq: Sequence,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
):
    """Copy per-layer K/V from CPU tensors into GPU paged kv_cache."""
    assert len(cpu_kv_layers) == kv_cache.size(1)
    prompt_len = seq.num_prompt_tokens
    slots = build_slot_mapping(seq, block_size, 0, prompt_len)
    assert len(slots) == prompt_len
    d = num_kv_heads * head_dim
    slots_t = torch.tensor(slots, dtype=torch.long, device=kv_cache.device)
    for layer_id, (k_cpu, v_cpu) in enumerate(cpu_kv_layers):
        assert k_cpu.shape[0] == prompt_len
        k_view = kv_cache[0, layer_id].view(-1, d)
        v_view = kv_cache[1, layer_id].view(-1, d)
        k_view.index_copy_(0, slots_t, k_cpu.reshape(prompt_len, d).to(device=kv_cache.device, non_blocking=True))
        v_view.index_copy_(0, slots_t, v_cpu.reshape(prompt_len, d).to(device=kv_cache.device, non_blocking=True))
