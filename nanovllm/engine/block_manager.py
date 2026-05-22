from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int, disable_prefix_cache: bool = False):
        self.block_size = block_size
        self.disable_prefix_cache = disable_prefix_cache
        # 维护block的大小
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # 记录字典记录每一个block的值
        self.hash_to_block_id: dict[int, int] = dict()
        # 用一个队列记录空闲的block
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # 用一个集合记录占用的KV block
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def _purge_block_hash(self, block_id: int):
        block = self.blocks[block_id]
        if block.hash != -1:
            self.hash_to_block_id.pop(block.hash, None)
            block.hash = -1
            block.token_ids = []

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = -1
            if not self.disable_prefix_cache and h != -1:
                block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1 and not self.disable_prefix_cache:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        num_blocks_needed = (len(seq) + self.block_size - 1) // self.block_size
        # 压缩后的block_table变短但num_tokens继续增长
        while len(block_table) < num_blocks_needed:
            if (
                block_table
                and not seq.kv_compressed
                and not self.disable_prefix_cache
            ):
                prev = self.blocks[block_table[-1]]
                assert prev.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)

        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 0:
            if last_block.hash != -1:
                last_block.hash = -1
            if not self.disable_prefix_cache and not seq.kv_compressed:
                token_ids = seq.block(seq.num_blocks - 1)
                prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
                h = self.compute_hash(token_ids, prefix)
                last_block.update(h, token_ids)
                self.hash_to_block_id[h] = last_block.block_id
        else:
            if last_block.hash != -1 and (seq.kv_compressed or self.disable_prefix_cache):
                last_block.hash = -1

    def truncate_blocks(self, seq: Sequence, keep_blocks: int):
        """Release blocks beyond keep_blocks after KV compression."""
        if keep_blocks >= len(seq.block_table):
            return
        tail = seq.block_table[keep_blocks:]
        for block_id in reversed(tail):
            block = self.blocks[block_id]
            self._purge_block_hash(block_id)
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.block_table = seq.block_table[:keep_blocks]
        seq.num_cached_tokens = min(seq.num_cached_tokens, keep_blocks * self.block_size)
        for block_id in seq.block_table:
            self._purge_block_hash(block_id)
        if seq.block_table:
            self.blocks[seq.block_table[-1]].hash = -1
        seq.kv_compressed = True
