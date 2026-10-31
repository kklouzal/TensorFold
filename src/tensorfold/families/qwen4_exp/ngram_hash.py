"""Flash Next's n-gram hashing constants, shared by the Mac (``model``) and CUDA (``cuda.ngram``) embeddings: the
checkpoint's table sizes and layer multipliers derive from them, so both backends hash every row alike."""

from __future__ import annotations

import math

import numpy as np

_MASK64 = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15
_MIX1 = 0xBF58476D1CE4E5B9
_MIX2 = 0x94D049BB133111EB
_PRIME = 10007


def _splitmix64(value: int) -> int:
    value = (value + _GOLDEN) & _MASK64
    value = ((value ^ (value >> 30)) * _MIX1) & _MASK64
    value = ((value ^ (value >> 27)) * _MIX2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


def layer_multipliers(vocab: int, ngram: int, ple_index: int, seed: int) -> np.ndarray:
    half = max(1, (((1 << 63) - 1) // max(vocab, 1)) // 2)
    base = seed + _PRIME * ple_index
    return np.array([2 * (_splitmix64((base + _GOLDEN * (i + 1)) & _MASK64) % half) + 1 for i in range(ngram)],
                    dtype=np.int64)
