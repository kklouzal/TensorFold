"""Keyed draft margins: the token is the chooser's, the margins a brute-force rescoring of the same candidates."""

import math

import numpy as np
import pytest

from tensorfold.engine.exact_sampling import MARGIN, Sampling, choose, choose_rows, keyed_margins, uniform, value_gap


def _oracle(values, ids, position, s):
    """One row, candidate by candidate: the kept set (``choose``'s filters), each score, top1 - top2 and the edge."""

    order = sorted(range(len(ids)), key=lambda c: (-float(values[c]), int(ids[c])))
    k = max(1, min(int(s.top_k) if s.top_k else len(ids), len(ids)))
    temperature = max(float(s.temperature), 1e-6)
    scaled = np.array([float(values[c]) for c in order[:k]], dtype=np.float64) / temperature
    kept = k
    if 0.0 < s.top_p < 1.0:
        probs = np.exp(scaled - scaled.max())
        probs /= probs.sum()
        kept = int(np.searchsorted(np.cumsum(probs), s.top_p) + 1)
    if s.min_p > 0.0:
        kept = min(kept, int((scaled[:kept] >= scaled[0] + math.log(s.min_p)).sum()))

    def score(c):
        noise = -np.log(-np.log(uniform(s.seed, position, np.array([ids[c]], dtype=np.int64))))[0]
        return float(values[c]) / temperature + noise

    scores = [score(c) for c in order[:kept]]
    best = int(np.argmax(scores))
    others = [x for i, x in enumerate(scores) if i != best]
    margin = scores[best] - max(others) if others and max(others) > -math.inf else math.inf
    budget = min(len(ids), s.top_k + MARGIN) if s.top_k else len(ids)
    cut = [score(c) for c in order[kept:budget]]
    edge = scores[best] - max(cut) if cut and max(cut) > -math.inf else math.inf
    return int(ids[order[best]]), margin, edge


def _rows(rng, trial, rows, width):
    if trial % 4 == 0:                                       # tie-heavy: few distinct values, ties by id
        values = rng.integers(0, 4, (rows, width)).astype(np.float32)
    else:
        values = (rng.normal(0, rng.choice([0.5, 3.0, 12.0]), (rows, width)) + 20).astype(np.float32)
    if trial % 7 == 3:
        values[:, -3:] = -np.inf                             # masked tokens (a grammar's): never kept or cut
    ids = np.stack([rng.choice(248320, width, replace=False) for _ in range(rows)]).astype(np.int64)
    return values, ids


POLICIES = [Sampling(seed=1, top_k=20, top_p=0.95), Sampling(seed=2, top_k=20, top_p=1.0),
            Sampling(seed=3, top_k=5, top_p=0.5), Sampling(seed=4, top_k=20, top_p=0.95, min_p=0.05),
            Sampling(seed=5, top_k=0, top_p=0.9), Sampling(seed=6, top_k=0, top_p=1.0, min_p=0.2),
            Sampling(seed=7, temperature=0.6, top_k=40, top_p=0.95), Sampling(seed=8, temperature=1.7, top_k=1)]


def test_margins_match_the_brute_force_oracle_and_the_chooser():
    rng = np.random.default_rng(11)
    for trial in range(400):
        base = POLICIES[trial % len(POLICIES)]
        s = Sampling(seed=int(rng.integers(0, 2**63 - 1)), temperature=base.temperature, top_k=base.top_k,
                     top_p=base.top_p, min_p=base.min_p)
        rows = int(rng.choice([1, 3]))
        width = int(rng.choice([s.top_k + MARGIN, 64])) if s.top_k else int(rng.choice([12, 300]))
        values, ids = _rows(rng, trial, rows, width)
        positions = rng.integers(0, 1 << 40, rows).astype(np.uint64)
        got = keyed_margins(values, ids, positions, s)
        assert [t for t, _, _ in got] == choose_rows(values, ids, positions, s)
        for r in range(rows):
            want = _oracle(values[r], ids[r], int(positions[r]), s)
            assert got[r] == want, (trial, r, got[r], want)
            assert want[0] == choose(values[r], ids[r], int(positions[r]), s)
            assert got[r][1] >= 0                            # the chosen token wins its own race


def test_a_lone_survivor_has_an_infinite_margin_and_an_uncut_row_an_infinite_edge():
    values = np.array([[9.0, 1.0, 0.5, 0.25]], dtype=np.float32)
    ids = np.array([[40, 10, 30, 20]], dtype=np.int64)
    tok, margin, edge = keyed_margins(values, ids, [3], Sampling(seed=1, top_k=1, top_p=1.0))[0]
    assert tok == 40 and margin == math.inf and edge < math.inf      # top-1 keeps it alone; the rest were cut
    tok, margin, edge = keyed_margins(values, ids, [3], Sampling(seed=1, top_k=4, top_p=1.0))[0]
    assert margin < math.inf and edge == math.inf                    # nothing cut
    assert keyed_margins(values, ids, [3], Sampling(seed=1, top_k=4, top_p=1.0, min_p=0.5))[0][1] == math.inf


def test_value_gap_is_the_top_two_difference():
    rng = np.random.default_rng(3)
    for _ in range(200):
        v = rng.normal(size=int(rng.integers(2, 40))).astype(np.float32)
        if rng.random() < 0.3:
            v[1] = v.max()                                   # a tie for the maximum: no gap
        top = sorted(v.astype(np.float64), reverse=True)
        assert value_gap(v) == top[0] - top[1]
    assert value_gap(np.array([1.0], dtype=np.float32)) == math.inf
    assert value_gap(np.array([2.0, -np.inf], dtype=np.float32)) == math.inf


def test_repaired_choose_scores_once_and_reports_the_margins_of_the_candidates_it_drew_from():
    pytest.importorskip("torch")
    from tensorfold.cuda.sampling import ambiguous_rows, repaired_choose

    rng = np.random.default_rng(5)
    repaired = 0
    for trial in range(150):
        s = Sampling(seed=int(rng.integers(0, 2**63 - 1)), top_k=int(rng.choice([1, 5, 20])), top_p=0.95,
                     min_p=float(rng.choice([0.0, 0.1])))
        rows, width = 3, 200
        full = (rng.integers(0, 6, (rows, width)) if trial % 2 else rng.normal(0, 3, (rows, width))).astype(np.float32)
        full_ids = rng.permutation(248320)[:width].astype(np.int64)
        count = s.top_k + MARGIN
        cols = np.argsort(-full, axis=1, kind="stable")[:, :count]           # a GPU top-k's candidates
        values, ids = np.take_along_axis(full, cols, axis=1), full_ids[cols]
        positions = rng.integers(0, 1 << 40, rows).astype(np.uint64)
        drawn = []
        tokens = repaired_choose(values, ids, positions, s, width, lambda r: (full[r], full_ids), drawn)
        assert tokens == repaired_choose(values, ids, positions, s, width, lambda r: (full[r], full_ids))
        ambiguous = set(int(r) for r in ambiguous_rows(values, s.top_k, width))
        repaired += len(ambiguous)
        for r in range(rows):
            source = (full[r], full_ids) if r in ambiguous else (values[r], ids[r])
            token, margin, edge = keyed_margins(source[0][None], source[1][None], positions[r:r + 1], s)[0]
            assert (token, (margin, edge)) == (tokens[r], drawn[r])
    assert repaired > 20                                     # the tie repair path was exercised
