"""Hash n-gram rows on the host, resetting on EOS and checking derived constants against the checkpoint."""

from __future__ import annotations

import math

import numpy as np

from ..ngram_hash import _nth_prime_after, layer_multipliers


class NGram:
    def __init__(self, *, vocab: int, ngram_size: int, heads_per_ngram: int, vocab_base: int, divisor: int,
                 seed: int, eos: int, embed_dim: int, ple_index: int = 0) -> None:
        self.n = ngram_size
        self.context = ngram_size - 1
        self.per_ngram = heads_per_ngram
        self.heads = self.context * heads_per_ngram
        self.eos = eos
        sizes, offsets, total = [], [], 0
        for head in range(self.heads):
            size = _nth_prime_after(vocab_base - 1, ple_index * self.heads + head + 1)
            sizes.append(size)
            offsets.append(total)
            total += size
        self.head_sizes = np.array(sizes, dtype=np.int64)
        self.head_offsets = np.array(offsets, dtype=np.int64)
        self.multipliers = layer_multipliers(vocab, ngram_size, ple_index, seed)
        self.rows = math.ceil(total / divisor) * divisor
        self.dims = embed_dim // self.heads

    def check(self, multipliers, offsets, sizes) -> None:
        for name, shipped, derived in (("layer_multipliers", multipliers, self.multipliers),
                                       ("ngram_heads_offsets", offsets, self.head_offsets),
                                       ("ngram_heads_vocab_sizes", sizes, self.head_sizes)):
            shipped = np.asarray(shipped).astype(np.int64).reshape(-1)
            if not np.array_equal(shipped, derived):
                raise ValueError(f"{name}: checkpoint {shipped} != derived {derived}")

    def ids(self, history: np.ndarray, tokens: np.ndarray) -> np.ndarray:
        """Row ids [L, heads] for ``tokens`` [L] after ``history`` [n-1] (EOS resets the n-grams)."""

        seq = np.concatenate([np.asarray(history, dtype=np.int64), np.asarray(tokens, dtype=np.int64)])[None]
        batch, width = seq.shape
        pos = np.arange(width)
        eos_at = np.where(seq == self.eos, pos[None], -1)
        before = np.concatenate([np.full((batch, 1), -1), np.maximum.accumulate(eos_at, axis=1)[:, :-1]], axis=1)
        in_segment = pos[None] - (before + 1)
        shifted = []
        for shift in range(self.n):
            source = pos - shift
            taken = np.take_along_axis(seq, np.broadcast_to(np.maximum(source, 0)[None], seq.shape), axis=1)
            shifted.append(np.where((in_segment >= shift) & (source[None] >= 0), taken, self.eos))
        blocks = []
        with np.errstate(over="ignore"):
            for ngram in range(2, self.n + 1):
                first = (ngram - 2) * self.per_ngram
                mixed = shifted[0] * self.multipliers[0]
                for p in range(1, ngram):
                    mixed = np.bitwise_xor(mixed, shifted[p] * self.multipliers[p])
                sizes = self.head_sizes[first:first + self.per_ngram]
                blocks.append(mixed[..., None] % sizes + self.head_offsets[first:first + self.per_ngram])
        out = np.concatenate(blocks, axis=-1)[0, -len(tokens):]
        return out

    def initial_history(self) -> np.ndarray:
        return np.full((self.context,), self.eos, dtype=np.int64)
