from dataclasses import dataclass, field
import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    cpu_prefill_capture: bool = False
    kv_captures: list = field(default_factory=list)
    # KV compression (decode-only)
    kv_compress_enabled: bool = False
    kv_compress_trigger_len: int = 0
    kv_compress_n: int = 1
    kv_compress_snap_window: int = 1
    kv_compress_period: int = 0
    kv_compress_ratio: float = 0.5
    kvcache_block_size: int = 256
    compress_selected_batch_indices: list = field(default_factory=list)
    compress_base_context_lens: torch.Tensor | None = None
    compression_events: list | None = None
    current_layer_id: int = 0
    num_hidden_layers: int = 0


_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(
    is_prefill,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
    cpu_prefill_capture=False,
    *,
    kv_compress_enabled=False,
    kv_compress_trigger_len=0,
    kv_compress_n=1,
    kv_compress_snap_window=1,
    kv_compress_period=0,
    kv_compress_ratio=0.5,
    kvcache_block_size=256,
    compress_selected_batch_indices=None,
    compress_base_context_lens=None,
    num_hidden_layers=0,
):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        slot_mapping,
        context_lens,
        block_tables,
        cpu_prefill_capture=cpu_prefill_capture,
        kv_captures=[] if cpu_prefill_capture else [],
        kv_compress_enabled=kv_compress_enabled,
        kv_compress_trigger_len=kv_compress_trigger_len,
        kv_compress_n=kv_compress_n,
        kv_compress_snap_window=kv_compress_snap_window,
        kv_compress_period=kv_compress_period,
        kv_compress_ratio=kv_compress_ratio,
        kvcache_block_size=kvcache_block_size,
        compress_selected_batch_indices=list(compress_selected_batch_indices or []),
        compress_base_context_lens=compress_base_context_lens,
        compression_events=None,
        num_hidden_layers=num_hidden_layers,
    )

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
