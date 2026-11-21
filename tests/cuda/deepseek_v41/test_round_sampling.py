"""Decode-round sampling (families/deepseek_v4/cuda/round_sampling.py) against the per-stream ``sample_rows`` it
replaces: the compact greedy packet over 8 contiguous vocabulary shards (first-maximum ties within and across
shards, NaN / +inf / all -inf rows) and the batched full-row sampler (greedy, top-k with tied boundaries, top-k
off nucleus incl. the complete-row fallback, per-row policies), token for token."""

from types import SimpleNamespace

import pytest
import torch

from tensorfold.cuda.sampling import sample_rows
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.deepseek_v4.cuda.round_sampling import RowSampling, _Nucleus, compact_packet, greedy_tokens

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
WORLD, SHARD = 8, 16160            # DS41: 129280 = 8 x 16160


def _rows(n, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    rows = (torch.randn((n, WORLD * SHARD), generator=g) * 3).cuda()
    # adversarial rows: ties inside a shard, across shards (equal max on ranks 2 and 5), masked shards
    rows[1, 100] = rows[1, 200] = 50.0
    rows[2, 2 * SHARD + 7] = rows[2, 5 * SHARD + 3] = 60.0
    rows[3, :4 * SHARD] = float("-inf")
    rows[4] = 0.0                                       # all tied: token 0
    rows[5, 3 * SHARD:] = float("-inf")
    rows[5, 3 * SHARD - 1] = rows[5, 0] = 1e30
    return rows


def _compact(rows):
    parts = [compact_packet(rows[:, r * SHARD:(r + 1) * SHARD].contiguous(), r) for r in range(WORLD)]
    host = torch.empty((WORLD,) + tuple(parts[0].shape), dtype=torch.float32, pin_memory=True)
    host.copy_(torch.stack(parts))
    return host


def _reference(rows, positions, sampling):
    out = []
    for g in range(rows.shape[0]):
        try:
            out.append(sample_rows(rows[g:g + 1], [positions[g]], sampling)[0])
        except ValueError as error:
            out.append(str(error))
    return out


def test_compact_greedy_matches_full_row_argmax():
    rows = _rows(12)
    rows[6, 9] = float("nan")
    rows[7, 5 * SHARD + 1] = float("inf")
    rows[8] = float("-inf")
    positions = list(range(100, 112))
    want = _reference(rows, positions, None)
    got = greedy_tokens(_compact(rows), rows.shape[0])
    for g, ((token, error), ref) in enumerate(zip(got, want)):
        assert (token if error is None else str(error)) == ref, g
    assert want[1] == 100 and want[2] == 2 * SHARD + 7 and want[4] == 0
    assert isinstance(want[6], str) and isinstance(want[7], str) and isinstance(want[8], str)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_row_sampling_matches_sample_rows(seed):
    rows = _rows(16, seed)
    rows[9, 1000:3000] = 7.5                            # a wide tie at a top-k boundary (repair path)
    rows[9, :1000] = 9.0
    rows[10] = torch.randn(WORLD * SHARD, device="cuda") * 1e-3   # flat: the nucleus outgrows its candidates
    rows[11, 5] = float("nan")
    policies = [None, Sampling(seed + 1, 0.8, 0, 0.95), Sampling(seed + 2, 1.0, 40, 0.9),
                Sampling(seed + 3, 0.6, 1000, 1.0), Sampling(seed + 4, 1.3, 0, 1.0, 0.05),
                Sampling(seed + 5, 0.7, 20, 0.95, 0.1), Sampling(seed + 6, 2.0, 0, 0.5), None,
                Sampling(seed + 7, 0.8, 0, 0.95), Sampling(seed + 8, 1.0, 1500, 0.99), Sampling(seed + 9, 1.0, 0, 1.0),
                Sampling(seed + 10, 1.0, 0, 0.9), Sampling(seed + 11, 0.9, 64, 0.95), Sampling(seed + 12, 0.5, 0, 0.8),
                Sampling(seed + 13, 1.0, 1, 1.0), Sampling(seed + 14, 1e-9, 0, 0.95)]
    positions = [1000 + 7 * g for g in range(16)]
    live = [SimpleNamespace(forced=None, constraint=None, history=None, probabilities=None, sampling=p)
            for p in policies]
    got = RowSampling(live, positions, rows, list(range(16))).choose()
    for g, p in enumerate(policies):
        try:
            want = sample_rows(rows[g:g + 1], [positions[g]], p)[0]
        except ValueError as error:
            assert isinstance(got[g], ValueError) and str(got[g]) == str(error), g
            continue
        assert got[g] == want, (g, p)


def test_nucleus_scaling_bits_match_nucleus_rows():
    """The batched per-row temperature scaling gives nucleus_rows' exact FP64 words (its Python-scalar divide)."""
    rows = _rows(6, 3)
    temps = [0.8, 0.7, 1.3, 0.6, 2.0, 1e-9]
    batch = _Nucleus(rows, temps)
    for g, t in enumerate(temps):
        scores = rows[g:g + 1].float().double()
        top = scores.max(dim=-1).values
        want = (scores - top[:, None]) / max(float(t), 1e-6)
        assert torch.equal(batch.scaled[g:g + 1].view(torch.int64), want.view(torch.int64)), t
