"""Committed-state checks shared by Flash Next storage regressions."""

import numpy as np


def compare_committed_state(a, b, *, exact):
    assert a.pos == b.pos and a.cur == b.cur
    for index, parity in enumerate(a.cur):
        exact(a.rec[parity, index], b.rec[b.cur[index], index])
    for name in ("conv", "ple_tail"):
        exact(getattr(a, name), getattr(b, name))
    for ka, kb in zip(a.kc, b.kc):
        for name in ("k", "v", "ks", "vs"):
            exact(getattr(ka, name)[:a.pos], getattr(kb, name)[:b.pos])
    for name in ("ikc", "pooled"):
        end = a.pos if name == "ikc" else a.pos // a.ratio
        for va, vb in zip(getattr(a, name), getattr(b, name)):
            exact(va[:end], vb[:end])
    np.testing.assert_array_equal(a.ple_history, b.ple_history)

