"""Coupled drafting: a drafter's keyed proposals follow the target's rule and noise (tensorfold.engine.speculative)."""

import numpy as np
import pytest

from tensorfold.engine.exact_sampling import MARGIN, Sampling, choose_rows
from tensorfold.engine.speculative import coupled, filtered

POLICIES = [Sampling(seed=11, temperature=1.0, top_k=64, top_p=0.95),
            Sampling(seed=12, temperature=0.6, top_k=20, top_p=0.95),
            Sampling(seed=13, temperature=0.7, top_k=20, top_p=0.8, min_p=0.05),
            Sampling(seed=14, temperature=1.3, top_k=5, top_p=1.0),
            Sampling(seed=15, temperature=1.0, top_k=0, top_p=0.9),
            Sampling(seed=16, temperature=0.8, top_k=0, top_p=1.0, min_p=0.1)]


def _rows(rng, rows, vocab, spread=3.0):
    return (rng.normal(size=(rows, vocab)) * spread).astype(np.float32)


def _lse(values, temperature):
    scaled = values.astype(np.float64) / temperature
    top = scaled.max(axis=-1, keepdims=True)
    return (top + np.log(np.exp(scaled - top).sum(axis=-1, keepdims=True)))[:, 0]


@pytest.mark.parametrize("s", POLICIES)
def test_a_drafter_equal_to_the_target_proposes_the_target_choice(s):
    """q == p: every row's first pick is the serial keyed choice at that row's position."""
    rng = np.random.default_rng(s.seed)
    vocab, rows = 300, 40
    logits = _rows(rng, rows, vocab)
    ids = np.broadcast_to(np.arange(vocab, dtype=np.int64), logits.shape).copy()
    positions = np.arange(1000, 1000 + rows, dtype=np.uint64)
    want = choose_rows(logits, ids, positions, s)
    norms = _lse(logits, s.temperature)
    picks, logconf = coupled(logits, ids, positions, s, norms=norms)
    assert picks[:, 0].tolist() == want
    assert np.all(np.isfinite(logconf[:, 0])) and np.all(logconf[:, 0] <= 0)
    if s.top_k:     # the candidates the GPU reads (top_k + MARGIN, shuffled) give the same choices
        count = s.top_k + MARGIN
        top = np.argsort(-logits, axis=-1, kind="stable")[:, :count]
        perm = rng.permutation(count)
        sub_ids, sub_vals = top[:, perm], np.take_along_axis(logits, top[:, perm], axis=-1)
        assert coupled(sub_vals, sub_ids, positions, s)[0][:, 0].tolist() == want


def test_branches_are_distinct_kept_tokens_in_perturbed_order():
    s = Sampling(seed=5, temperature=1.0, top_k=8, top_p=0.9)
    rng = np.random.default_rng(5)
    logits = _rows(rng, 30, 50)
    ids = np.broadcast_to(np.arange(50, dtype=np.int64), logits.shape).copy()
    positions = np.arange(30, dtype=np.uint64)
    picks, logconf = coupled(logits, ids, positions, s, branch=12)
    kept_ids, scaled = filtered(logits, ids, s)
    for row in range(30):
        kept = set(kept_ids[row][np.isfinite(scaled[row])].tolist())
        real = [t for t in picks[row].tolist() if t >= 0]
        assert len(real) == len(kept) and set(real) == kept           # every kept token once, then -1 padding
        assert picks[row].tolist() == real + [-1] * (12 - len(real))
        finite = logconf[row][: len(real)]
        assert np.all(np.diff(finite) <= 0) and np.all(np.isinf(logconf[row][len(real):]))
        assert np.exp(finite).sum() == pytest.approx(1.0)
    assert coupled(logits, ids, positions, s)[0][:, 0].tolist() == picks[:, 0].tolist()


def test_filtering_matches_the_serial_rule_cut_for_cut():
    """filtered keeps exactly the tokens choose_rows can return (the support of the serial draw)."""
    s = Sampling(seed=3, temperature=0.7, top_k=10, top_p=0.85, min_p=0.02)
    rng = np.random.default_rng(3)
    logits = _rows(rng, 4, 40)
    ids = np.broadcast_to(np.arange(40, dtype=np.int64), logits.shape).copy()
    kept_ids, scaled = filtered(logits, ids, s)
    seen = [set() for _ in range(4)]
    for position in range(4000):
        for row, token in enumerate(choose_rows(logits, ids, [position] * 4, s)):
            seen[row].add(token)
    for row in range(4):
        assert seen[row] == set(kept_ids[row][np.isfinite(scaled[row])].tolist())


def test_coupling_beats_argmax_drafts_and_meets_the_communication_free_bound():
    """A noisy drafter: coupled proposals match the target's draw far more often than its argmax does, and at
    least (1 - TV) / (1 + TV) on average (Daliri et al. 2024)."""
    s = Sampling(seed=21, temperature=1.0, top_k=0, top_p=1.0)
    rng = np.random.default_rng(21)
    vocab, rows = 64, 3000
    target = _rows(rng, rows, vocab, spread=1.5)
    drafter = (target + rng.normal(size=target.shape) * 0.7).astype(np.float32)
    ids = np.broadcast_to(np.arange(vocab, dtype=np.int64), target.shape).copy()
    positions = np.arange(rows, dtype=np.uint64)
    serial = np.array(choose_rows(target, ids, positions, s))
    picks = coupled(drafter, ids, positions, s)[0][:, 0]
    p = np.exp(target - target.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
    q = np.exp(drafter - drafter.max(-1, keepdims=True)); q /= q.sum(-1, keepdims=True)
    tv = 0.5 * np.abs(p - q).sum(-1)
    matched = (picks == serial).mean()
    assert matched >= ((1 - tv) / (1 + tv)).mean() - 0.02
    assert matched > (drafter.argmax(-1) == serial).mean() + 0.1
    assert matched <= (1 - tv).mean() + 0.02            # never above the maximal coupling


def test_greedy_and_bad_geometry_are_refused():
    s = Sampling(seed=1, temperature=1.0, top_k=0, top_p=0.5)
    values, ids = np.zeros((2, 3), np.float32), np.zeros((2, 3), np.int64)
    with pytest.raises(ValueError):
        coupled(values, ids, [0, 1], s)                 # nucleus without complete-row norms
    with pytest.raises(ValueError):
        coupled(values, ids, [0], Sampling(seed=1))     # one position per row
    with pytest.raises(ValueError):
        coupled(values, ids, [0, 1], Sampling(seed=1), branch=0)
