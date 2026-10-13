"""The QSA selection oracle (tests/cuda/qsa_select_oracle.py) against brute force on small rows: its key order is
fp32's total order with -0 below +0 and NaNs by sign and payload, and its lists equal pairwise rank counting."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "qsa_select_oracle", Path(__file__).resolve().parent / "cuda" / "qsa_select_oracle.py")
oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oracle)


def _bits(*values: int) -> np.ndarray:
    return np.array(values, dtype=np.uint32).view(np.float32)


def test_keys_follow_the_total_order_of_fp32_bits():
    ordered = _bits(0xFFFFFFFF, 0xFFC00001, 0xFF800001,         # negative NaNs by payload, the largest first
                    0xFF800000, 0xC0000000, 0x80800000, 0x80000001, 0x80000000,   # -inf .. -denormal, -0.0
                    0x00000000, 0x00000001, 0x00800000, 0x3F800000, 0x7F7FFFFF, 0x7F800000,   # +0.0 .. +inf
                    0x7F800001, 0x7FC00000, 0x7FFFFFFF)          # positive NaNs by payload
    keys = oracle.qsa_keys(ordered).astype(np.int64)
    assert keys[0] == 0 and np.all(np.diff(keys) > 0)
    rng = np.random.default_rng(3)
    x = rng.standard_normal(4096).astype(np.float32)
    x[::7] = 0.0
    keys = oracle.qsa_keys(np.sort(x)).astype(np.int64)
    assert np.all(np.diff(keys) >= 0)


def _brute(row: np.ndarray, end: int, top: int, ratio: int):
    """A block is chosen iff fewer than ``top`` blocks precede it in (key descending, id ascending) order."""

    complete = end // ratio
    if complete <= top:
        return None, end, 0
    k = [int(v) for v in oracle.qsa_keys(row[:complete])]
    chosen = [b for b in range(complete)
              if sum(1 for c in range(complete) if k[c] > k[b] or (k[c] == k[b] and c < b)) < top]
    assert len(chosen) == top
    ids = [ratio * b + q for b in chosen for q in range(ratio)] + list(range(ratio * complete, end))
    return ids, len(ids), 1


def _patterns(rng, n: int):
    f32 = np.float32
    yield rng.standard_normal(n).astype(f32)
    yield (np.floor(rng.standard_normal(n) * 2) / 2).astype(f32)
    yield np.zeros(n, f32)
    yield np.ones(n, f32)
    yield np.where(rng.random(n) < 0.5, f32(0.0), f32(-0.0)).astype(f32)
    x = rng.standard_normal(n).astype(f32)
    pick = rng.random(n)
    x[pick < 0.1] = np.inf
    x[(pick >= 0.1) & (pick < 0.2)] = -np.inf
    x[(pick >= 0.2) & (pick < 0.3)] = np.nan
    x[(pick >= 0.3) & (pick < 0.4)] = -np.nan
    x[(pick >= 0.4) & (pick < 0.5)] = f32(1e-45)
    x[(pick >= 0.5) & (pick < 0.55)] = f32(-1e-45)
    yield x
    yield rng.integers(0, 2 ** 32, n, dtype=np.uint64).astype(np.uint32).view(f32)
    z = np.full(n, 0xFFFFFFFF, np.uint32)                       # key 0 everywhere but a few positive scores
    z[rng.permutation(n)[:max(1, n // 20)]] = np.float32(3.0).view(np.uint32)
    yield z.view(f32)


@pytest.mark.parametrize("budget,ratio", [(16, 4), (24, 4), (2048, 4)])
def test_lists_equal_pairwise_rank_counting(budget, ratio):
    rng = np.random.default_rng(budget)
    top = budget // ratio
    ends = sorted({top * ratio - 1, top * ratio + ratio - 1, top * ratio + ratio, top * ratio + ratio + 1,
                   top * ratio + ratio + 3, 2 * top * ratio + 2, 3 * top * ratio + 1,
                   (top + 88) * ratio + 3 if top < 100 else (top + 30) * ratio + 3})
    for end in ends:
        for row in _patterns(rng, max(1, end // ratio)):
            scores = row[None, :]
            ids = np.full((1, budget + ratio), -7, np.int32)
            nk, sparse = np.full(1, -1, np.int32), np.full(1, -1, np.int32)
            oracle.qsa_select_reference(scores, [end], ids, nk, sparse, budget=budget, ratio=ratio)
            want, n, flag = _brute(row, end, top, ratio)
            assert (int(nk[0]), int(sparse[0])) == (n, flag), end
            if want is None:
                assert np.all(ids == -7)
            else:
                assert ids[0, :n].tolist() == want and np.all(ids[0, n:] == -7), end
