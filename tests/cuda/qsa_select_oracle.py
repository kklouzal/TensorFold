"""Flash Next's QSA block selection, independently of any kernel: a full sort over a total order of score bits.

A row ending at ``end`` keys considers its ``complete = end // ratio`` finished blocks. A dense row
(``complete <= top``, ``top = budget // ratio``) keeps its length and writes no ids. A sparse row lists the ``top``
blocks first in (key descending, block id ascending) order, in ascending block id, each as its ``ratio`` keys, then
the unfinished block's 0 .. ratio - 1 keys. Keys order all 2**32 fp32 bit patterns: -NaN (by payload) < -inf <
negative finite < -0.0 < +0.0 < positive finite < +inf < +NaN (by payload); the negative NaN 0xFFFFFFFF has key 0.
"""

from __future__ import annotations

import numpy as np


def qsa_keys(x_f32: np.ndarray) -> np.ndarray:
    """Order-preserving uint32 keys of fp32 bit patterns."""

    bits = np.ascontiguousarray(x_f32, dtype=np.float32).view(np.uint32)
    return np.where((bits & 0x80000000) != 0, ~bits, bits | np.uint32(0x80000000)).astype(np.uint32)


def qsa_select_reference(scores: np.ndarray, ends, ids: np.ndarray, nk: np.ndarray, sparse: np.ndarray, *,
                         budget: int = 2048, ratio: int = 4) -> None:
    """In place, as one select launch: ``scores`` [rows, NB] fp32, ``ends[r]`` row r's position + 1, ``ids``
    [rows, budget + ratio] int32, ``nk``/``sparse`` [rows] int32. Entries the selection does not write keep their
    values (pre-fill a sentinel to catch extra writes)."""

    top = budget // ratio
    for r, end in enumerate(int(e) for e in ends):
        complete = end // ratio
        if complete <= top:                                   # dense: ids[r] untouched
            nk[r], sparse[r] = end, 0
            continue
        key = qsa_keys(scores[r, :complete]).astype(np.int64)
        order = np.lexsort((np.arange(complete), -key))       # key descending, then block id ascending
        chosen = np.sort(order[:top])                         # listed in ascending block id
        ids[r, :top * ratio] = (chosen[:, None] * ratio + np.arange(ratio)).ravel()
        tail = np.arange(ratio * complete, end)               # the unfinished block's keys
        ids[r, top * ratio:top * ratio + tail.size] = tail
        nk[r], sparse[r] = top * ratio + tail.size, 1         # ids[r, nk[r]:] untouched
