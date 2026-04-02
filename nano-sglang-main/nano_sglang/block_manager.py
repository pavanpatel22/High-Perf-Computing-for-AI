"""Part 5 (stretch): Paged KV Cache

Fixed-size blocks instead of contiguous allocation.
Same idea as OS virtual memory pages.
"""

import torch
import math


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, num_layers: int,
                 num_heads: int, head_dim: int, device: str = "cuda",
                 dtype: torch.dtype = torch.float16):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers

        # Physical KV storage — pre-allocated on GPU
        # Shape: [num_blocks, num_heads, block_size, head_dim]
        # Each "row" (block_id) is one page of block_size tokens
        self.k_pool = [
            torch.zeros(num_blocks, num_heads, block_size, head_dim,
                        device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.v_pool = [
            torch.zeros(num_blocks, num_heads, block_size, head_dim,
                        device=device, dtype=dtype)
            for _ in range(num_layers)
        ]

        # Free block pool — block IDs available for allocation
        self.free_blocks: list[int] = list(range(num_blocks))

        # Maps seq_id → list of physical block IDs (the block table)
        # e.g. {0: [3, 7, 1], 1: [0, 5]}
        # Logical block 0 → physical block 3, logical block 1 → physical 7, etc.
        self.seq_to_blocks: dict[int, list[int]] = {}

    def allocate(self, seq_id: int, num_tokens: int) -> list[int]:
        """
        Allocate enough physical blocks to hold num_tokens for a sequence.
        Returns the list of allocated block IDs (the sequence's block table).

        Uses ceiling division so a partial last block is still allocated:
            e.g. 34 tokens, block_size=16 → ceil(34/16) = 3 blocks (holds 48 slots)

        Raises RuntimeError if there are not enough free blocks (OOM).
        Called during prefill when a new sequence is admitted.
        """
        # How many blocks do we need? Round up to cover all tokens
        num_blocks_needed = math.ceil(num_tokens / self.block_size)

        if num_blocks_needed > self.num_free_blocks:
            raise RuntimeError(
                f"Out of KV cache memory: need {num_blocks_needed} blocks "
                f"for {num_tokens} tokens, but only {self.num_free_blocks} free. "
                f"(block_size={self.block_size})"
            )

        # Pop block IDs from the front of the free list
        allocated = []
        for _ in range(num_blocks_needed):
            block_id = self.free_blocks.pop(0)
            allocated.append(block_id)

        # Register the block table for this sequence
        self.seq_to_blocks[seq_id] = allocated

        return allocated

    def free(self, seq_id: int):
        """
        Return all blocks owned by a finished sequence back to the free pool.
        Called when a sequence hits EOS or max_tokens so its GPU memory
        is immediately available for new sequences.

        Safe to call on a seq_id that was never allocated (no-op).
        """
        if seq_id not in self.seq_to_blocks:
            return

        # Reclaim every block this sequence was using
        blocks_to_free = self.seq_to_blocks.pop(seq_id)
        self.free_blocks.extend(blocks_to_free)

    def get_block_ids(self, seq_id: int) -> list[int]:
        return self.seq_to_blocks.get(seq_id, [])

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    # -----------------------------------------------------------------------
    # Helper methods — used by the engine to read/write KV data into pages
    # -----------------------------------------------------------------------
    def write_kv(self, layer_idx: int, seq_id: int,
                 token_pos: int, key: torch.Tensor, value: torch.Tensor):
        """
        Write one token's K/V into the correct physical block slot.

        Args:
            layer_idx:  transformer layer index
            seq_id:     which sequence owns this data
            token_pos:  absolute token position (0-indexed from start of seq)
            key:        shape [num_heads, head_dim]
            value:      shape [num_heads, head_dim]

        How paging works:
            logical_block = token_pos // block_size   (which page)
            slot_in_block = token_pos  % block_size   (offset within page)
            physical_block = block_table[logical_block]
        """
        block_table = self.seq_to_blocks[seq_id]
        logical_block = token_pos // self.block_size
        slot_in_block = token_pos % self.block_size
        physical_block = block_table[logical_block]

        self.k_pool[layer_idx][physical_block, :, slot_in_block, :] = key
        self.v_pool[layer_idx][physical_block, :, slot_in_block, :] = value

    def read_kv(self, layer_idx: int, seq_id: int,
                seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Read all cached K/V for a sequence up to seq_len tokens.
        Gathers data from (potentially non-contiguous) physical blocks
        and returns a contiguous tensor the attention layer can use.

        Returns:
            keys:   shape [1, num_heads, seq_len, head_dim]
            values: shape [1, num_heads, seq_len, head_dim]
        """
        block_table = self.seq_to_blocks[seq_id]

        keys_list   = []
        values_list = []

        tokens_remaining = seq_len
        for logical_block, physical_block in enumerate(block_table):
            # How many valid tokens are in this block?
            tokens_in_block = min(self.block_size, tokens_remaining)
            if tokens_in_block <= 0:
                break

            # Slice out only the filled slots: [num_heads, tokens_in_block, head_dim]
            k = self.k_pool[layer_idx][physical_block, :, :tokens_in_block, :]
            v = self.v_pool[layer_idx][physical_block, :, :tokens_in_block, :]

            keys_list.append(k)
            values_list.append(v)
            tokens_remaining -= tokens_in_block

        # Concatenate across blocks on the seq_len (dim=1) dimension
        # [num_heads, seq_len, head_dim] → unsqueeze → [1, num_heads, seq_len, head_dim]
        keys   = torch.cat(keys_list,   dim=1).unsqueeze(0)
        values = torch.cat(values_list, dim=1).unsqueeze(0)

        return keys, values

    def append_slot(self, seq_id: int, current_seq_len: int) -> bool:
        """
        Ensure capacity exists for one more token during decode.
        Allocates a new physical block if the current last block is full.

        Returns True if a new block was allocated, False if existing block
        still has room.

        Called every decode step BEFORE engine.decode_step() so we never
        write into an unallocated block.
        """
        # If current tokens exactly fill complete blocks, we need a new one
        if current_seq_len % self.block_size == 0:
            if self.num_free_blocks == 0:
                raise RuntimeError(
                    f"Out of KV cache memory during decode for seq {seq_id}: "
                    f"no free blocks available."
                )
            new_block_id = self.free_blocks.pop(0)
            self.seq_to_blocks[seq_id].append(new_block_id)
            return True  # new block was appended

        return False  # existing last block still has space