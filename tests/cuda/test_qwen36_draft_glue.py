"""Fused DFlash drafter stages: float64 bounds, exact copies and row invariance on one CUDA GPU."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from types import SimpleNamespace as NS

from tensorfold.families.qwen3_5_moe.cuda import draft_glue

H, HKV, D, EPS = 32, 4, 128, 1e-6
# Relative bound of a value rounded to BF16 twice (the normalized activation, then its product with the BF16 weight;
# unit roundoff 2**-8 each: 2**-7 + 2**-16), plus FP32 terms: the mean of squares of up to 2048 values summed in any
# order (2048 * 2**-24 = 2**-13, halved by the square root) with rsqrt, or the exp inside SiLU (under 2**-19 for the
# |gate| < 16 drawn here) -- 2**-13 covers both.
TWICE_ROUNDED = 2.0 ** -7 + 2.0 ** -13


def _rand(*shape, scale=1.0, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(*shape, generator=g, device="cuda") * scale).bfloat16()


def _weight(n, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (1 + 0.1 * torch.randn(n, generator=g, device="cuda")).bfloat16()


def _normed(x, weight):
    """Exact RMSNorm * weight in float64 over the last dimension."""
    v = x.double()
    return weight.double() * v * torch.rsqrt(v.square().mean(-1, keepdim=True) + EPS)


def _partner(x):
    half = x.shape[-1] // 2
    return torch.cat((x[..., half:], x[..., :half]), -1)


def _rotated(x, cos, sin):
    half = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., half:], x[..., :half]), -1) * sin


def _within(got, exact, relative):
    return bool(((got.double() - exact).abs() <= relative * exact.abs()).all())


def _rotary(rows, start=500):
    phase = torch.arange(start, start + rows, device="cuda").double()[:, None]
    phase = phase * (1.0 / 10000 ** (torch.arange(0, D, 2, device="cuda").double() / D))[None]
    phase = torch.cat((phase, phase), -1)
    return (phase.cos() * 1.07).bfloat16(), (phase.sin() * 1.07).bfloat16()


@pytest.mark.parametrize("rows", [1, 3, 16, 100, 4096])
def test_rmsnorm_is_row_invariant_within_two_roundings_and_adds_the_residual_exactly(rows):
    x, r, w = _rand(rows, 2048, seed=rows), _rand(rows, 2048, seed=rows + 1), _weight(2048, rows)
    h, y = draft_glue.rmsnorm(x, w, EPS, residual=r)
    assert torch.equal(h.view(torch.int16), (x + r).view(torch.int16))       # one FP32 add rounded, as torch
    assert _within(y, _normed(h, w), TWICE_ROUNDED)
    same, alone = draft_glue.rmsnorm(h, w, EPS)
    assert same is h and torch.equal(alone.view(torch.int16), y.view(torch.int16))
    for i in sorted({0, rows // 2, rows - 1}):
        assert torch.equal(draft_glue.rmsnorm(h[i:i + 1].contiguous(), w, EPS)[1].view(torch.int16),
                           y[i:i + 1].view(torch.int16))


@pytest.mark.parametrize("rows", [1, 16, 48])
def test_qkv_heads_norms_and_rotates_q_and_k_and_copies_v(rows):
    qkv = _rand(rows, (H + 2 * HKV) * D, seed=7 + rows)
    qn, kn = _weight(D, 1), _weight(D, 2)
    cos, sin = _rotary(rows)
    q, k, v = draft_glue.qkv_heads(qkv, qn, kn, cos, sin, EPS, heads=H, kv_heads=HKV, head_dim=D)
    heads = qkv.view(rows, H + 2 * HKV, D).transpose(0, 1)
    assert torch.equal(v, heads[H + HKV:].contiguous())
    c, s = cos.double()[None], sin.double()[None]
    for got, part, weight in ((q, heads[:H], qn), (k, heads[H:H + HKV], kn)):
        # Each normalized value (and its rotation partner) is within TWICE_ROUNDED of exact; each rotation product
        # and their sum round to BF16 once more (2**-8 each): |error| <= (2**-6 + 2**-12) (|n c| + |partner(n) s|).
        n = _normed(part, weight)
        spread = n.abs() * c.abs() + _partner(n).abs() * s.abs()
        assert bool(((got.double() - _rotated(n, c, s)).abs() <= (2.0 ** -6 + 2.0 ** -12) * spread).all())
    order = torch.arange(rows - 1, -1, -1, device="cuda")
    q2, k2, v2 = draft_glue.qkv_heads(qkv[order].contiguous(), qn, kn, cos[order].contiguous(),
                                      sin[order].contiguous(), EPS, heads=H, kv_heads=HKV, head_dim=D)
    assert all(torch.equal(a[:, order].view(torch.int16), b.view(torch.int16)) for a, b in ((q, q2), (k, k2), (v, v2)))


def test_kv_append_writes_exactly_the_blocks_keys_and_values_into_each_rows_slot_only():
    layers, rows = 3, 7
    stores = [NS(k=torch.zeros(layers, HKV, 64, D, dtype=torch.bfloat16, device="cuda"),
                 v=torch.zeros(layers, HKV, 64, D, dtype=torch.bfloat16, device="cuda")),
              NS(k=torch.zeros(layers, HKV, 32, D, dtype=torch.bfloat16, device="cuda"),
                 v=torch.zeros(layers, HKV, 32, D, dtype=torch.bfloat16, device="cuda"))]
    kv = _rand(rows, 2 * HKV * D, seed=11)
    kn = _weight(D, 3)
    cos, sin = _rotary(rows)
    dest = draft_glue.destinations(stores + [None], [10, 5, 0], [4, 3, 0], "cuda")
    draft_glue.kv_append(kv, kn, cos, sin, dest, 2, EPS, kv_heads=HKV, head_dim=D)
    # The block path's own keys for the same rows: K/V of a [q | k | v] row with any q.
    qkv = torch.cat((torch.zeros(rows, H * D, dtype=torch.bfloat16, device="cuda"), kv), 1)
    _, k, v = draft_glue.qkv_heads(qkv, kn, kn, cos, sin, EPS, heads=H, kv_heads=HKV, head_dim=D)
    expected = [NS(k=torch.zeros_like(s.k), v=torch.zeros_like(s.v)) for s in stores]
    expected[0].k[2, :, 10:14], expected[0].v[2, :, 10:14] = k[:, :4], v[:, :4]
    expected[1].k[2, :, 5:8], expected[1].v[2, :, 5:8] = k[:, 4:], v[:, 4:]
    for got, want in zip(stores, expected):
        assert torch.equal(got.k.view(torch.int16), want.k.view(torch.int16))
        assert torch.equal(got.v.view(torch.int16), want.v.view(torch.int16))


@pytest.mark.parametrize("rows", [1, 16, 4096])
def test_silu_mul_is_within_two_roundings_and_row_invariant(rows):
    gu = _rand(rows, 2 * 6144, scale=2.0, seed=rows + 3)
    out = draft_glue.silu_mul(gu)
    g, u = gu[:, :6144].double(), gu[:, 6144:].double()
    assert _within(out, g * torch.sigmoid(g) * u, TWICE_ROUNDED)
    assert torch.equal(draft_glue.silu_mul(gu[-1:].contiguous()).view(torch.int16), out[-1:].view(torch.int16))
