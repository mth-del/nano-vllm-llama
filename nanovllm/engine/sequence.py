from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        # RoPE position for next decode step (logical); may exceed physical KV length after compress.
        self.rope_pos = len(self.token_ids) - 1
        self.num_prompt_tokens = len(token_ids)
        self.num_completion_tokens_stored = 0
        # After KV compression, prefix hash / block_table token mapping is no longer valid.
        self.kv_compressed = False
        self.num_cached_tokens = 0    # tokens that don't need prefill
        self.num_scheduled_tokens = 0
        # 每个sequence维护自己的KV block编号列表（PD：handoff 前为空，不占 GPU block）
        self.block_table = []
        # PD：CPU prefill 后的 KV，handoff 前仅存于 host
        self.cpu_kv_layers: list | None = None
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        if self.token_ids:
            return len(self.token_ids) - self.num_prompt_tokens
        return self.num_completion_tokens_stored

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
        self.rope_pos += 1
        if len(self.token_ids) > self.num_prompt_tokens:
            self.num_completion_tokens_stored = len(self.token_ids) - self.num_prompt_tokens

    def __getstate__(self):
        use_full_tokens = (
            self.num_completion_tokens == 0
            or self.num_cached_tokens < self.num_tokens
        )
        last_state = self.token_ids if use_full_tokens else self.last_token
        return (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.num_scheduled_tokens,
            self.block_table,
            self.rope_pos,
            self.kv_compressed,
            self.kv_compress_anchor,
            self.num_completion_tokens_stored,
            last_state,
        )

    def __setstate__(self, state):
        if len(state) == 6:
            (
                self.num_tokens,
                self.num_prompt_tokens,
                self.num_cached_tokens,
                self.num_scheduled_tokens,
                self.block_table,
                last_state,
            ) = state
            self.rope_pos = self.num_tokens - 1
            self.kv_compressed = False
            self.kv_compress_anchor = 0
            self.num_completion_tokens_stored = 0
        elif len(state) == 7:
            (
                self.num_tokens,
                self.num_prompt_tokens,
                self.num_cached_tokens,
                self.num_scheduled_tokens,
                self.block_table,
                self.rope_pos,
                last_state,
            ) = state
            self.kv_compressed = False
            self.kv_compress_anchor = 0
            self.num_completion_tokens_stored = max(0, self.num_tokens - self.num_prompt_tokens)
        elif len(state) == 9:
            (
                self.num_tokens,
                self.num_prompt_tokens,
                self.num_cached_tokens,
                self.num_scheduled_tokens,
                self.block_table,
                self.rope_pos,
                self.kv_compressed,
                self.num_completion_tokens_stored,
                last_state,
            ) = state
            self.kv_compress_anchor = 0
        else:
            (
                self.num_tokens,
                self.num_prompt_tokens,
                self.num_cached_tokens,
                self.num_scheduled_tokens,
                self.block_table,
                self.rope_pos,
                self.kv_compressed,
                self.kv_compress_anchor,
                self.num_completion_tokens_stored,
                last_state,
            ) = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
            self.num_completion_tokens_stored = max(
                0, len(self.token_ids) - self.num_prompt_tokens,
            )
        else:
            self.token_ids = []
            self.last_token = last_state
