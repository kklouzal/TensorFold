"""Exact sampling: a fixed function of (logits, position, seed); an exact top-k/top-p draw."""

import numpy as np
import pytest

from tensorfold.engine.exact_sampling import Sampling, choose, seed_for, uniform


def test_same_inputs_same_token_and_position_matters():
    rng = np.random.default_rng(0)
    values = rng.normal(size=28).astype(np.float32)
    ids = rng.permutation(1000)[:28].astype(np.int64)
    s = Sampling(seed=123, temperature=1.0, top_k=20, top_p=0.95)
    assert choose(values, ids, 7, s) == choose(values.copy(), ids.copy(), 7, s)
    picks = {choose(values, ids, p, s) for p in range(200)}
    assert len(picks) > 3          # the draw moves with the position


def test_candidate_order_does_not_matter():
    rng = np.random.default_rng(1)
    values = rng.normal(size=28).astype(np.float32)
    ids = np.arange(28, dtype=np.int64) + 500
    s = Sampling(seed=9)
    perm = rng.permutation(28)
    assert choose(values, ids, 3, s) == choose(values[perm], ids[perm], 3, s)


def test_ties_break_by_token_id():
    values = np.array([5.0, 5.0, 5.0, 1.0], dtype=np.float32)
    ids = np.array([40, 10, 30, 20], dtype=np.int64)
    s = Sampling(seed=4, temperature=1.0, top_k=2, top_p=1.0)
    for p in range(50):
        assert choose(values, ids, p, s) in (10, 30)   # ids 10 and 30 hold the top-2 slots


def test_draw_matches_the_distribution():
    values = np.log(np.array([0.5, 0.3, 0.2], dtype=np.float64)).astype(np.float32)
    ids = np.array([1, 2, 3], dtype=np.int64)
    s = Sampling(seed=77, temperature=1.0, top_k=3, top_p=1.0)
    counts = np.bincount([choose(values, ids, p, s) for p in range(20000)], minlength=4)[1:]
    assert np.allclose(counts / counts.sum(), [0.5, 0.3, 0.2], atol=0.015)


def test_top_p_cuts_the_tail():
    values = np.log(np.array([0.6, 0.3, 0.1], dtype=np.float64)).astype(np.float32)
    ids = np.array([1, 2, 3], dtype=np.int64)
    s = Sampling(seed=5, temperature=1.0, top_k=3, top_p=0.85)
    assert {choose(values, ids, p, s) for p in range(3000)} == {1, 2}


def test_uniform_in_open_interval_and_seed_for_is_stable():
    u = uniform(1, 2, np.arange(10000, dtype=np.int64))
    assert (u > 0).all() and (u < 1).all()
    assert seed_for([1, 2, 3]) == seed_for([1, 2, 3]) != seed_for([1, 2, 4])


def test_sample_rows_matches_choose_on_the_gpu():
    mx = pytest.importorskip("mlx.core")
    from tensorfold.engine.exact_sampling import sample_rows

    logits = (mx.random.normal((3, 5000)) * 3).astype(mx.bfloat16)
    s = Sampling(seed=11)
    rows = sample_rows(logits, [10, 11, 12], s)
    alone = [sample_rows(logits[i:i + 1], [10 + i], s)[0] for i in range(3)]
    assert rows == alone


def test_choose_rows_matches_choose_row_by_row_alone_or_batched():
    from tensorfold.engine.exact_sampling import choose_rows

    rng = np.random.default_rng(7)

    def bf16(a):
        b = a.astype(np.float32).view(np.uint32)
        return ((b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000).view(np.float32)

    for trial in range(300):
        s = Sampling(seed=int(rng.integers(0, 2**63 - 1)), top_p=[0.95, 1.0, 0.5][trial % 3])
        rows = int(rng.choice([1, 3, 16]))
        values = bf16(rng.normal(0, rng.choice([0.5, 3.0, 12.0]), (rows, 28)) + 20)
        if trial % 5 == 0:
            values[:, 2] = values[:, 9]                     # value ties resolve by token id
        ids = np.stack([rng.choice(248320, 28, replace=False) for _ in range(rows)]).astype(np.int64)
        positions = rng.integers(0, 100000, rows)
        want = [choose(values[r], ids[r], int(positions[r]), s) for r in range(rows)]
        assert choose_rows(values, ids, positions, s) == want
        assert [choose_rows(values[r:r + 1], ids[r:r + 1], positions[r:r + 1], s)[0] for r in range(rows)] == want


def test_nucleus_without_top_k_matches_the_whole_vocabulary_draw():
    """top_k 0: the fast nucleus (GPU candidates + normalizer) draws what sorting every logit draws."""

    mx = pytest.importorskip("mlx.core")

    from tensorfold.engine import exact_sampling as es

    rng = np.random.default_rng(5)
    drawn = 0
    for trial in range(60):
        row = (mx.array(rng.normal(size=(1, 50_000)).astype(np.float32)) * 8.0).astype(mx.bfloat16)
        s = es.Sampling(seed=int(rng.integers(1 << 62)), temperature=float(rng.choice([0.7, 1.0])), top_k=0,
                        top_p=0.95)
        fast = es._nucleus_rows(row, [trial], s)
        full = es.choose_rows(np.array(row.astype(mx.float32)), np.arange(50_000, dtype=np.int64)[None], [trial], s)
        if fast is not None:
            drawn += 1
            assert fast == full
    assert drawn > 40
    flat = mx.zeros((1, 50_000), dtype=mx.bfloat16)          # every token tied: the nucleus needs them all
    assert es._nucleus_rows(flat, [0], es.Sampling(seed=1, top_k=0, top_p=0.95)) is None


@pytest.mark.parametrize("offset", [1e30, -1e30, np.finfo(np.float64).max, -np.finfo(np.float64).max])
@pytest.mark.parametrize("top_k,top_p,min_p", [(0, 1., 0.), (2, .9, .2), (0, .5, 1.)])
def test_extreme_common_offset_cannot_erase_keyed_noise(offset, top_k, top_p, min_p):
    from tensorfold.engine.exact_sampling import choose_rows

    ids = np.array([7, 2, 19, 5], dtype=np.int64)
    baseline, shifted = np.zeros(4), np.full(4, offset)
    seen = set()
    for seed in range(32):
        policy = Sampling(seed, .7, top_k, top_p, min_p)
        expected = choose(baseline, ids, 17, policy)
        assert choose(shifted, ids, 17, policy) == expected
        assert choose_rows(np.stack([baseline, shifted]), np.stack([ids, ids]), [17, 17], policy) == [expected, expected]
        seen.add(expected)
    assert len(seen) > 1


def test_exactly_represented_score_shifts_preserve_support_and_draw():
    from tensorfold.engine.exact_sampling import choose_rows

    values = np.array([[4., 2., 1., -3.], [-5., -1., 2., 0.]])
    ids = np.array([[9, 8, 2, 3], [9, 8, 2, 3]], dtype=np.int64)
    for seed in range(32):
        for k, p, m in [(0, 1., 0.), (3, .9, .1), (0, .7, 0.)]:
            policy = Sampling(seed, .7, k, p, m)
            expected = choose_rows(values, ids, [17, 20], policy)
            assert choose_rows(values + 2.**40, ids, [17, 20], policy) == expected
            assert expected == [choose(v, i, pos, policy) for v, i, pos in zip(values, ids, [17, 20])]


def test_extreme_finite_difference_has_zero_tail_mass_without_invalid_operation():
    from tensorfold.engine.exact_sampling import choose_rows

    bound = np.finfo(np.float64).max
    values, ids = np.array([bound, -bound, -np.inf]), np.array([3, 4, 5], dtype=np.int64)
    policy = Sampling(1, 1e-6, 0, 1.)
    with np.errstate(all="raise"):
        assert choose(values, ids, 17, policy) == 3
        assert choose_rows(values[None], ids[None], [17], policy) == [3]


@pytest.mark.parametrize("offset", [1e30, -1e30])
def test_mlx_extreme_offset_nucleus_and_native_kernel_keep_noise(offset):
    mx = pytest.importorskip("mlx.core")
    from tensorfold.engine import exact_sampling, gpu_sampling

    logits = mx.full((2, 2048), offset, dtype=mx.float32)
    baseline = mx.zeros_like(logits)
    # A narrow nucleus also exercises the host MLX normalizer, not just its complete-row fallback.
    narrow = mx.full((1, 2048), -100., dtype=mx.float32)
    narrow[0, :4] = 0.
    for top_k, top_p in [(0, .9), (20, 1.)]:
        for seed in (0, 1, 6, 13):
            policy = Sampling(seed, .7, top_k, top_p)
            assert exact_sampling.sample_rows(logits, [17, 19], policy) == exact_sampling.sample_rows(baseline, [17, 19], policy)
            native = gpu_sampling.sample(logits, policy, [17, 19])
            np.testing.assert_array_equal(np.array(native), np.array(gpu_sampling.sample(baseline, policy, [17, 19])))
            assert np.array(native).tolist() == [gpu_sampling.reference(np.full(2048, offset), policy, p) for p in (17, 19)]
    policy = Sampling(6, .7, 0, .7)
    assert exact_sampling._nucleus_rows(narrow, [17], policy) == exact_sampling.sample_rows(narrow, [17], policy, keep={})


def test_uniform_grid_exact_endpoints_by_independent_inverse_hash():
    from fractions import Fraction

    from tensorfold.engine.exact_sampling import uniform_rows

    mask = (1 << 64) - 1

    def undo_xor(value, shift):
        result = value
        for _ in range(64 // shift + 1):
            result = value ^ (result >> shift)
        return result

    def unmix(value):
        value = undo_xor(value, 31)
        value = value * pow(0x94D049BB133111EB, -1, 1 << 64) & mask
        value = undo_xor(value, 27)
        value = value * pow(0xBF58476D1CE4E5B9, -1, 1 << 64) & mask
        return undo_xor(value, 30)

    # Construct exact keyed preimages of both bins and their adjacent/discarded-bit boundaries.
    for word in (0, 1, 4095, 4096, 1 << 63, mask - 4096, mask - 4095, mask):
        seed = (unmix(unmix(unmix(word)) ^ 0xD1B54A32D192ED03) - 0x9E3779B97F4A7C15) & mask
        expected = Fraction(2 * (word >> 12) + 1, 1 << 53)
        actual = float(uniform(seed, 1, np.array([0], dtype=np.int64))[0])
        assert Fraction(*actual.as_integer_ratio()) == expected
        assert Fraction(1, 1 << 53) <= expected <= 1 - Fraction(1, 1 << 53)
        assert uniform_rows(seed, [1], np.array([[0]], dtype=np.int64))[0, 0] == actual
    # Historical 53-bit midpoint failed at the top bin; no production clamp repairs it.
    assert (mask >> 11) * 2.**-53 + 2.**-54 == 1.


def test_endpoint_witness_preserves_masks_and_scalar_vector_key_bits():
    from tensorfold.engine.exact_sampling import choose_rows, uniform_rows

    seed, position = 4199689772251414973, 1
    ids = np.array([0, 1], dtype=np.int64)
    assert uniform(seed, position, ids)[0] == 1. - 2.**-53
    values = np.array([-np.inf, 0.])
    policy = Sampling(seed, 1., 0, 1.)
    with np.errstate(all="raise"):
        assert choose(values, ids, position, policy) == 1
        assert choose_rows(np.stack([values, values]), np.stack([ids, ids]), [position, position], policy) == [1, 1]
        positions = np.array([0, 1, (1 << 64) - 1], dtype=np.uint64)
        tokens = np.array([[0, 1, 7], [1 << 54, 3, 2], [(1 << 63) - 1, 0, 1]], dtype=np.int64)
        np.testing.assert_array_equal(uniform_rows(seed, positions, tokens),
                                      np.stack([uniform(seed, int(p), row) for p, row in zip(positions, tokens)]))


def test_metal_23bit_midpoints_are_already_exact_and_strictly_interior():
    from fractions import Fraction

    # Binary32 has24significandbits: (2*b+1)*2^-24 is exact for every23bit bin.
    for word in (0, 1, (1 << 22), (1 << 23) - 2, (1 << 23) - 1):
        actual = float((np.float32(word) + np.float32(.5)) * np.float32(2.**-23))
        assert Fraction(*actual.as_integer_ratio()) == Fraction(2 * word + 1, 1 << 24)
        assert 0 < actual < 1
