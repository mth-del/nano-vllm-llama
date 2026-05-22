import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    pd_separation: bool = False
    # KV compression (decode-only).
    kv_compress: bool = False
    kv_compress_n: int = 1
    kv_compress_snap_window: int = 1
    # Periodic mode (blog repro): compress every `period` new tokens; keep `ratio` of window.
    kv_compress_period: int = 0
    kv_compress_ratio: float = 0.5
    kv_compress_no_prefix_cache: bool = True

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        if self.pd_separation:
            assert self.tensor_parallel_size == 1, "pd_separation requires tensor_parallel_size=1"
        if self.kv_compress:
            assert self.tensor_parallel_size == 1, "kv_compress requires tensor_parallel_size=1"
            assert self.kv_compress_n >= 1
            assert 0 < self.kv_compress_ratio <= 1
            if self.kv_compress_period > 0:
                assert self.kv_compress_period % self.kvcache_block_size == 0
            if self.kv_compress and not self.enforce_eager:
                self.enforce_eager = True
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
