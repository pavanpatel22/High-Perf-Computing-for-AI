"""Part 2: Inference Engine

Two phases:
  Prefill:  process entire prompt in one pass -> compute-bound
  Decode:   generate one token at a time from cache -> memory-bound
"""

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from .model import Model, Tokenizer
from .sampling import SamplingParams, sample_token
from .sequence import Sequence, SequenceStatus


class Engine:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model = Model(model_path, device=device)
        self.tokenizer = Tokenizer(model_path)
        self.device = device

    def prefill(self, seq: Sequence, sampling_params: SamplingParams) -> int:
        """
        Process all prompt tokens in one forward pass, return first generated token.
        Stores past_key_values in seq and sets status to DECODING.

        - Tokenizes the prompt
        - Runs a single forward pass over ALL prompt tokens at once (compute-bound)
        - Stores the resulting KV cache in seq.past_key_values
        - Samples the first output token from the final logit position
        - Sets seq.status = DECODING so the scheduler knows to move it to running
        """
        # Tokenize the prompt → [1, prompt_len]
        input_ids = self.tokenizer.encode(seq.prompt)
        input_ids = torch.tensor([input_ids], device=self.device)

        # Single forward pass over the full prompt, no prior cache
        logits, past_key_values = self.model.forward(
            input_ids,
            past_key_values=None,
        )

        # Sample the first output token from the LAST logit position
        # logits shape: [1, prompt_len, vocab_size] → take [:, -1, :]
        next_token = sample_token(logits[:, -1, :], sampling_params).item()

        # Store KV cache on the sequence object so decode_step can use it
        seq.past_key_values = past_key_values

        # Transition sequence state: WAITING → DECODING
        seq.status = SequenceStatus.DECODING

        return next_token

    def decode_step(self, seq: Sequence, sampling_params: SamplingParams) -> int:
        """Generate one token for a single sequence using cached KV."""
        last_token = seq.output_token_ids[-1]
        input_ids = torch.tensor([[last_token]], device=self.device)
        logits, past_key_values = self.model.forward(
            input_ids, past_key_values=seq.past_key_values
        )
        next_token = sample_token(logits[:, -1, :], sampling_params).item()
        seq.past_key_values = past_key_values
        return next_token

    def decode_batch(self, sequences: list[Sequence],
                     sampling_params: SamplingParams) -> list[int]:
        """Generate one token for multiple sequences in a single GPU forward pass."""
        if not sequences:
            return []
        if len(sequences) == 1:
            return [self.decode_step(sequences[0], sampling_params)]

        n = len(sequences)
        input_ids = torch.tensor(
            [[seq.output_token_ids[-1]] for seq in sequences],
            device=self.device,
        )

        cache_lens = [seq.past_key_values.get_seq_length() for seq in sequences]
        max_len = max(cache_lens)

        batched_cache = DynamicCache()
        for layer_idx in range(self.model.num_layers):
            padded_keys, padded_values = [], []
            for seq in sequences:
                k = seq.past_key_values.key_cache[layer_idx]
                v = seq.past_key_values.value_cache[layer_idx]
                pad = max_len - k.shape[2]
                if pad > 0:
                    k = F.pad(k, (0, 0, pad, 0))
                    v = F.pad(v, (0, 0, pad, 0))
                padded_keys.append(k)
                padded_values.append(v)
            batched_cache.key_cache.append(torch.cat(padded_keys, dim=0))
            batched_cache.value_cache.append(torch.cat(padded_values, dim=0))

        attn_mask = torch.zeros(n, max_len + 1, device=self.device, dtype=torch.long)
        for i, cl in enumerate(cache_lens):
            attn_mask[i, max_len - cl:] = 1

        position_ids = torch.tensor([[cl] for cl in cache_lens], device=self.device)

        logits, new_cache = self.model.forward(
            input_ids,
            past_key_values=batched_cache,
            position_ids=position_ids,
            attention_mask=attn_mask,
        )

        tokens = sample_token(logits[:, -1, :], sampling_params)

        for i, seq in enumerate(sequences):
            real_len = cache_lens[i] + 1
            pad = max_len - cache_lens[i]
            per_seq_cache = DynamicCache()
            for layer_idx in range(self.model.num_layers):
                k = new_cache.key_cache[layer_idx][i:i+1, :, pad:pad + real_len, :]
                v = new_cache.value_cache[layer_idx][i:i+1, :, pad:pad + real_len, :]
                per_seq_cache.key_cache.append(k.clone())
                per_seq_cache.value_cache.append(v.clone())
            seq.past_key_values = per_seq_cache

        return [t.item() for t in tokens]

    def generate(self, prompt: str,
                 sampling_params: SamplingParams = None) -> str:
        """
        Generate text for a single prompt end-to-end.
        Wires prefill + decode loop together for standalone use.

        Flow:
          1. Create a Sequence from the prompt
          2. prefill()  → fills KV cache, returns first token
          3. Loop decode_step() until EOS or max_new_tokens
          4. Decode output token ids → string
        """
        if sampling_params is None:
            sampling_params = SamplingParams()

        # Build a Sequence object (mirrors how scheduler uses the engine)
        seq = Sequence(prompt=prompt)

        # --- Phase 1: Prefill ---
        first_token = self.prefill(seq, sampling_params)
        seq.output_token_ids.append(first_token)

        # Early exit if the model immediately returns EOS
        if first_token == self.tokenizer.eos_token_id:
            return self.tokenizer.decode(seq.output_token_ids)

        # --- Phase 2: Decode loop ---
        for _ in range(sampling_params.max_new_tokens - 1):
            next_token = self.decode_step(seq, sampling_params)
            seq.output_token_ids.append(next_token)

            # Stop on EOS token
            if next_token == self.tokenizer.eos_token_id:
                break

        # Mark finished and decode token ids → text
        seq.status = SequenceStatus.FINISHED
        return self.tokenizer.decode(seq.output_token_ids)