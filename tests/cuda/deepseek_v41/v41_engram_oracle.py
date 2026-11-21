# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Vendored test oracle: 1Cat-vLLM 4d57473db67eb77a13018d179e4d95169e82044f benchmarks/deepseek_v41/oracle_primitives.py
# (blob a45312e0e2a8fcba6ef105b94a2fe2b7079c9b89). TensorFold changes: only EngramHashState is kept (GGUF/route helpers dropped).
"""Independent CPU reference for the V4.1 Engram n-gram hash (test oracle)."""

import torch


class EngramHashState:
    """Hash from checkpoint auxiliaries with only max_ngram_size-1 cached ids.

    A masked token blocks lookback across its position. Calls must be contiguous;
    start_pos=0 explicitly starts a new sequence, as in the official reference.
    """

    DEAD = -1

    def __init__(
        self,
        token_map: torch.Tensor,
        primes: torch.Tensor,
        offsets: torch.Tensor,
        multipliers: torch.Tensor,
        pad_token_id: int,
    ):
        self.token_map = token_map.to(device="cpu", dtype=torch.int64)
        self.primes = primes.to(device="cpu", dtype=torch.int64)
        self.offsets = offsets.to(device="cpu", dtype=torch.int64)
        self.multipliers = multipliers.to(device="cpu", dtype=torch.int64)
        self.pad_id = int(self.token_map[pad_token_id])
        self.max_ngram_size = multipliers.shape[-1]
        self.history = None
        self.next_pos = 0

    @torch.inference_mode()
    def __call__(
        self,
        input_ids: torch.Tensor,
        start_pos: int,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_ids = input_ids.to(device="cpu", dtype=torch.int64)
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("Expected a non-empty [batch, sequence] token tensor")
        if torch.any(input_ids < 0) or torch.any(input_ids >= len(self.token_map)):
            raise ValueError("Token id outside compressed token map")
        batch, length = input_ids.shape
        history_length = self.max_ngram_size - 1
        if start_pos == 0:
            self.history = torch.full((batch, history_length), self.DEAD)
            self.next_pos = 0
        if start_pos != self.next_pos or self.history is None:
            raise ValueError("Hash calls must have contiguous start_pos")
        if self.history.shape[0] != batch:
            raise ValueError("Batch size changed during a sequence")
        compressed = self.token_map[input_ids]
        if token_mask is not None:
            if token_mask.shape != input_ids.shape:
                raise ValueError("Token mask shape mismatch")
            compressed = compressed.masked_fill(~token_mask.cpu().bool(), self.DEAD)
        context = torch.cat((self.history, compressed), dim=1)
        tokens = []
        blocked = torch.zeros_like(compressed, dtype=torch.bool)
        for shift in range(self.max_ngram_size):
            begin = history_length - shift
            source = context[:, begin : begin + length]
            blocked |= source == self.DEAD
            tokens.append(torch.where(blocked, self.pad_id, source))
        products = torch.stack(tokens, dim=-1).unsqueeze(2) * self.multipliers
        heads = self.primes.shape[-1] // history_length
        primes = self.primes.unflatten(-1, (history_length, heads))
        rolling, hashes = products[..., 0], []
        for shift in range(1, self.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., shift])
            hashes.append(rolling.unsqueeze(-1) % primes[:, shift - 1])
        self.history = context[:, -history_length:].clone()
        self.next_pos += length
        return torch.cat(hashes, dim=-1) + self.offsets
