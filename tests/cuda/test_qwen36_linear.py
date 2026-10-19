"""Qwen3.6 BF16 projections: one row-invariant tensor-core arithmetic, within the BF16/FP32-accumulate bound."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

# CUDA availability must be checked before importing the native backend.
from tensorfold.families.qwen3_5_moe.cuda.linear import MmaLinear

# Every verify/draft/serial/prompt row count the engine issues, and beyond one prompt tile.
ROWS = (1, 2, 3, 4, 5, 7, 8, 12, 15, 16, 17, 33, 64, 128, 300)


def _data(n, k, rows, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.empty((n, k), dtype=torch.bfloat16, device="cuda").uniform_(-.05, .05, generator=g)
    x = torch.empty((rows, k), dtype=torch.bfloat16, device="cuda").uniform_(-1, 1, generator=g)
    return w, x


def _bound_ok(got, x, w):
    """|y - x.w| <= 2^-8 |x.w| + (1 + 2^-8) K 2^-22 sum|x w|.

    2^-8 is BF16's unit roundoff (the output rounding of the FP32 sum); the FP32 accumulation is allowed two FP32
    ulps of sum|x w| (2^-22 sum|x w|) per product, as the mma aligns and normalizes its sums by truncation.
    """
    exact = x.double() @ w.double().t()
    scale = x.double().abs() @ w.double().abs().t()
    k = x.shape[1]
    limit = 2.0 ** -8 * exact.abs() + (1 + 2.0 ** -8) * k * 2.0 ** -22 * scale
    return bool(((got.double() - exact).abs() <= limit).all())


@pytest.mark.parametrize("n,k", [(130, 2048), (4160, 2048), (2048, 4096), (1024, 10240), (32, 2048)])
def test_rows_keep_singleton_bits_and_the_accumulation_bound(n, k):
    w, x = _data(n, k, max(ROWS), seed=n + k)
    projection = MmaLinear(w)
    alone = torch.cat([projection(x[r:r + 1]) for r in range(x.shape[0])])
    assert _bound_ok(alone, x, w)
    for rows in ROWS:
        assert torch.equal(projection(x[:rows]).view(torch.int16), alone[:rows].view(torch.int16)), rows
    order = torch.arange(15, -1, -1, device="cuda")
    assert torch.equal(projection(x[order]).view(torch.int16), alone[order].view(torch.int16))
    assert torch.equal(projection.prefill(x).view(torch.int16), alone.view(torch.int16))


def test_stacked_gates_keep_each_projections_bits():
    """[z | b | a] in one launch: each column block keeps its own projection's bits, decode and prompt rows."""
    w, x = _data(4096 + 64, 2048, 17, seed=5)
    stacked = MmaLinear(w)
    for rows in (1, 16, 17):
        got = stacked(x[:rows])
        for lo, hi in ((0, 4096), (4096, 4128), (4128, 4160)):
            want = MmaLinear(w[lo:hi])(x[:rows])
            assert torch.equal(got[:, lo:hi].view(torch.int16), want.view(torch.int16)), (rows, lo)


def test_actual_vocabulary_shape_keeps_singleton_bits():
    w, x = _data(248320, 2048, 17, seed=17032)
    head = MmaLinear(w)
    alone = torch.cat([head(x[r:r + 1]) for r in range(17)])
    for rows in (2, 3, 8, 15, 16, 17):
        assert torch.equal(head(x[:rows]).view(torch.int16), alone[:rows].view(torch.int16)), rows
    assert _bound_ok(alone[:2, :4096], x[:2], w[:4096])
