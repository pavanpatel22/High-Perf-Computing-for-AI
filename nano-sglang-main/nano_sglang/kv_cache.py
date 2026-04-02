"""Part 1: KV Cache

Stores key/value tensors from previous forward passes so we don't
recompute them. Turns O(n^2) decode into O(n).
"""

import torch


class KVCache:
    def __init__(self, num_layers: int, num_heads: int, head_dim: int,
                 max_seq_len: int, max_batch_size: int, device: str = "cuda",
                 dtype: torch.dtype = torch.float16):
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size

        # Shape per layer: [max_batch_size, num_heads, max_seq_len, head_dim]
        self.keys = [
            torch.zeros(max_batch_size, num_heads, max_seq_len, head_dim,
                        device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.values = [
            torch.zeros(max_batch_size, num_heads, max_seq_len, head_dim,
                        device=device, dtype=dtype)
            for _ in range(num_layers)
        ]

    def update(self, layer_idx: int, batch_idx: int,
               key: torch.Tensor, value: torch.Tensor, start_pos: int):
        """
        Write key/value into cache.
        
        Args:
            layer_idx:  which transformer layer (0 to num_layers-1)
            batch_idx:  which sequence slot in the batch
            key:        shape [1, num_heads, new_seq_len, head_dim]
            value:      shape [1, num_heads, new_seq_len, head_dim]
            start_pos:  token position offset to write at (0 for prefill,
                        current seq_len for each decode step)
        """
        new_seq_len = key.shape[2]  # number of new tokens being written
        end_pos = start_pos + new_seq_len

        assert end_pos <= self.max_seq_len, (
            f"KV cache overflow: tried to write up to position {end_pos}, "
            f"but max_seq_len={self.max_seq_len}"
        )

        # key/value are [1, num_heads, new_seq_len, head_dim]
        # cache slot is [max_batch_size, num_heads, max_seq_len, head_dim]
        # We index batch_idx and slice the seq dimension
        self.keys[layer_idx][batch_idx, :, start_pos:end_pos, :] = key[0]
        self.values[layer_idx][batch_idx, :, start_pos:end_pos, :] = value[0]

    def get(self, layer_idx: int, batch_idx: int, seq_len: int):
        """
        Read all cached key/value for a sequence up to seq_len.

        Args:
            layer_idx:  which transformer layer
            batch_idx:  which sequence slot in the batch
            seq_len:    how many token positions to return

        Returns:
            key:   shape [1, num_heads, seq_len, head_dim]
            value: shape [1, num_heads, seq_len, head_dim]
        """
        # Slice out [num_heads, seq_len, head_dim] then unsqueeze batch dim
        key   = self.keys[layer_idx][batch_idx, :, :seq_len, :].unsqueeze(0)
        value = self.values[layer_idx][batch_idx, :, :seq_len, :].unsqueeze(0)
        return key, value

    def clear(self, batch_idx: int):
        """Zero out cache for a finished sequence."""
        for layer_idx in range(self.num_layers):
            self.keys[layer_idx][batch_idx].zero_()
            self.values[layer_idx][batch_idx].zero_()