"""Part 3: Continuous Batching Scheduler

Prefill each request one at a time, then batch all decodes together
using engine.decode_batch() for GPU efficiency.
"""

from .sampling import SamplingParams
from .sequence import Sequence, SequenceStatus
from .engine import Engine


class Scheduler:
    def __init__(self, model_path: str, max_batch_size: int = 64,
                 device: str = "cuda"):
        self.engine = Engine(model_path, device=device)
        self.tokenizer = self.engine.tokenizer
        self.max_batch_size = max_batch_size

        self.next_seq_id = 0
        self.waiting_queue: list[Sequence] = []
        self.running: list[Sequence] = []
        self.finished: list[Sequence] = []

    def add_request(self, prompt: str, sampling_params: SamplingParams = None):
        """Tokenize prompt, create Sequence, add to waiting queue."""
        if sampling_params is None:
            sampling_params = SamplingParams()
        token_ids = self.tokenizer.encode(prompt)
        seq = Sequence(
            seq_id=self.next_seq_id,
            prompt_token_ids=token_ids,
            max_tokens=sampling_params.max_tokens,
        )
        self.waiting_queue.append(seq)
        self.next_seq_id += 1

    def _prefill_waiting(self, sampling_params: SamplingParams):
        """Prefill one request from the waiting queue and move it to running."""
        if not self.waiting_queue:
            return
        if len(self.running) >= self.max_batch_size:
            return
        seq = self.waiting_queue.pop(0)
        first_token = self.engine.prefill(seq, sampling_params)
        seq.output_token_ids.append(first_token)
        if first_token == self.tokenizer.eos_token_id:
            seq.status = SequenceStatus.FINISHED
            self.finished.append(seq)
        else:
            self.running.append(seq)

    def _decode_running(self, sampling_params: SamplingParams):
        """
        Decode all running sequences in one batched forward pass.

        Uses engine.decode_batch() to process every running sequence
        simultaneously on the GPU — far more efficient than calling
        decode_step() one by one in a Python loop.

        After getting the new tokens:
          - Append each token to its sequence's output_token_ids
          - Check termination: EOS token OR reached max_tokens
          - Move finished sequences out of self.running into self.finished
        """
        if not self.running:
            return

        # Single batched GPU forward pass for ALL running sequences
        next_tokens = self.engine.decode_batch(self.running, sampling_params)

        still_running = []
        for seq, token in zip(self.running, next_tokens):
            seq.output_token_ids.append(token)

            # Termination condition: EOS token or hit max_tokens budget
            is_eos = (token == self.tokenizer.eos_token_id)
            is_max = (len(seq.output_token_ids) >= seq.max_tokens)

            if is_eos or is_max:
                seq.status = SequenceStatus.FINISHED
                self.finished.append(seq)
            else:
                still_running.append(seq)

        self.running = still_running

    def step(self, sampling_params: SamplingParams = None):
        """
        One scheduling iteration — the core of continuous batching.

        Order matters here:
          1. _decode_running() first  → frees slots from finished sequences
          2. _prefill_waiting() after → fills freed slots with new requests

        This ordering ensures that as soon as a sequence finishes,
        its batch slot is immediately available for a waiting request
        in the SAME step, maximising GPU utilisation.

        We prefill one waiting request per step (not all of them) because
        prefill is expensive — doing one at a time keeps decode latency
        predictable and avoids stalling the running batch.
        """
        if sampling_params is None:
            sampling_params = SamplingParams()

        # Step 1: advance all running sequences by one token (batched)
        _decode_running(self, sampling_params)

        # Step 2: admit one new request from the waiting queue
        self._prefill_waiting(sampling_params)

    def run_to_completion(self,
                          sampling_params: SamplingParams = None) -> list[str]:
        """
        Drive the scheduler loop until every request is finished.
        Returns generated texts in the original submission order (by seq_id).

        Loop continues as long as there are requests in either:
          - waiting_queue  (not yet prefilled)
          - running        (prefilled, still generating)
        """
        if sampling_params is None:
            sampling_params = SamplingParams()

        while self.waiting_queue or self.running:
            # Step 1: decode all running sequences
            self._decode_running(sampling_params)

            # Step 2: promote as many waiting requests as batch allows
            # Keep prefilling until the batch is full or queue is empty
            while self.waiting_queue and len(self.running) < self.max_batch_size:
                self._prefill_waiting(sampling_params)

        # Sort finished sequences by seq_id to preserve submission order
        self.finished.sort(key=lambda s: s.seq_id)

        # Decode token ids → strings for each finished sequence
        return [
            self.tokenizer.decode(seq.output_token_ids)
            for seq in self.finished
        ]