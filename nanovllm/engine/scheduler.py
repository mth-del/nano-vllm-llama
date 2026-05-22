from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.pd_separation = config.pd_separation
        disable_prefix = config.kv_compress and config.kv_compress_no_prefix_cache
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size,
            disable_prefix_cache=disable_prefix,
        )
        self.waiting: deque[Sequence] = deque()
        # PD：CPU prefill 完成、尚未占用 GPU KV block
        self.prefill_ready: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.prefill_ready and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule_cpu_prefill(self) -> list[Sequence]:
        """Take requests from waiting; no GPU KV allocation."""
        scheduled_seqs = []
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            scheduled_seqs.append(self.waiting.popleft())
        return scheduled_seqs

    def schedule_gpu_handoff(self) -> list[Sequence]:
        """Allocate GPU blocks and hand off sequences with CPU KV ready."""
        scheduled_seqs = []
        pending: deque[Sequence] = deque()
        while self.prefill_ready:
            seq = self.prefill_ready.popleft()
            if len(scheduled_seqs) >= self.max_num_seqs or not self.block_manager.can_allocate(seq):
                pending.append(seq)
                continue
            self.block_manager.allocate(seq)
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            scheduled_seqs.append(seq)
        self.prefill_ready.extend(pending)
        return scheduled_seqs

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        ## [1]prefill (GPU path only; PD uses schedule_cpu_prefill)
        if not self.pd_separation:
            while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
                seq = self.waiting[0]
                num_tokens = max(seq.num_tokens - seq.num_cached_tokens, 1)
                remaining = self.max_num_batched_tokens - num_batched_tokens
                if remaining == 0 or (not seq.block_table and not self.block_manager.can_allocate(seq)):
                    break
                if remaining < num_tokens and scheduled_seqs:
                    break
                if not seq.block_table:
                    self.block_manager.allocate(seq)
                seq.num_scheduled_tokens = min(num_tokens, remaining)
                if seq.num_scheduled_tokens == num_tokens:
                    seq.status = SequenceStatus.RUNNING
                    self.waiting.popleft()
                    self.running.append(seq)
                scheduled_seqs.append(seq)
                num_batched_tokens += seq.num_scheduled_tokens
            if scheduled_seqs:
                return scheduled_seqs, True

        ##  [2]decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                # 抢占式
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        if seq.block_table:
            self.block_manager.deallocate(seq)
        seq.num_cached_tokens = 0
        seq.cpu_kv_layers = None
        seq.kv_compressed = False
        seq.kv_compress_anchor = 0
        self.waiting.appendleft(seq)

    def postprocess(
        self,
        seqs: list[Sequence],
        token_ids: list[int],
        is_prefill: bool,
        compression_events: list | None = None,
    ):
        if compression_events:
            for ev in compression_events:
                bidx = ev["batch_index"]
                if bidx >= len(seqs):
                    continue
                seq = seqs[bidx]
                seq.num_tokens = ev["new_context_len"]
                seq.num_cached_tokens = ev["new_context_len"]
                seq.kv_compress_anchor = ev["new_context_len"]
                seq.kv_compressed = True
                self.block_manager.truncate_blocks(seq, ev["keep_blocks"])

        for seq, token_id in zip(seqs, token_ids):
            if is_prefill:
                seq.num_cached_tokens = min(seq.num_cached_tokens + seq.num_scheduled_tokens, seq.num_tokens)
                if seq.num_cached_tokens < seq.num_tokens or seq.num_completion_tokens > 0:    # chunked prefill or re prefill after preemption
                    seq.num_scheduled_tokens = 0
                    continue
            seq.append_token(token_id)
            seq.num_cached_tokens += 1
            seq.num_scheduled_tokens = 0
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                if seq.block_table:
                    self.block_manager.deallocate(seq)
                seq.cpu_kv_layers = None
                self.running.remove(seq)
