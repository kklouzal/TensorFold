"""Load-time requantization of BF16 weights (tensorfold.cuda.requant): FP8 (E4M3 codes, one scale an output row) and
NVFP4 (E2M1 codes, an E4M3 scale per 16 inputs, one global scale)."""

import itertools

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda.requant import E4M3_MAX, requant_rows  # noqa: E402

MAGS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)                # E2M1 magnitudes by code (OCP MX), the oracles' own copy


def test_rows_scale_to_the_e4m3_range_and_reconstruct_within_its_rounding():
    gen = torch.Generator().manual_seed(3)
    w = (torch.randn(300, 128, generator=gen) * torch.logspace(-4, 1, 300)[:, None]).bfloat16()
    w[7] = 0                                                   # an all-zero row keeps scale 1 and zero codes
    codes, scales = requant_rows(w, chunk=64)                  # several chunks, one partial
    assert codes.dtype == torch.float8_e4m3fn and codes.shape == w.shape and scales.dtype == torch.float32
    top = w.float().abs().amax(1)
    expect = torch.where(top > 0, top / E4M3_MAX, torch.ones_like(top))
    assert torch.equal(scales, expect) and scales[7] == 1 and not codes[7].float().any()
    assert float(codes.float().abs().amax(1).max()) <= E4M3_MAX
    back = codes.float() * scales[:, None]
    # E4M3 keeps 3 mantissa bits: a normal value rounds within 2^-4 relative; subnormals within half a step
    err = (back - w.float()).abs()
    bound = torch.maximum(w.float().abs() * 2.0 ** -4, scales[:, None] * 2.0 ** -10)
    assert bool((err <= bound).all())


def test_codes_are_round_to_nearest_of_the_scaled_row():
    w = torch.tensor([[1.0, -0.3, 0.0, 448.0 / 3], [2.5, 2.5, -5.0, 1e-3]], dtype=torch.bfloat16)
    codes, scales = requant_rows(w)
    scaled = w.float() / scales[:, None]
    np.testing.assert_array_equal(codes.float().numpy(), scaled.to(torch.float8_e4m3fn).float().numpy())


@pytest.mark.parametrize("bad", [torch.zeros(4), torch.zeros(2, 3, dtype=torch.int32),
                                 torch.tensor([[1.0, float("nan")]])])
def test_malformed_weights_are_refused(bad):
    with pytest.raises(ValueError):
        requant_rows(bad)


def test_accounting_matches_the_copy_and_refuses_unknown_kinds():
    from tensorfold.cuda.requant import resident_bytes, staging_bytes

    assert resident_bytes("fp8", 1000, 256) == 1024 * 256 + 1024 * 4 == resident_bytes("fp8a8", 1000, 256)
    assert staging_bytes("fp8", 262144, 2816) > resident_bytes("fp8", 262144, 2816)
    for call in (resident_bytes, staging_bytes):
        with pytest.raises(ValueError):
            call("int4", 10, 64)


def _recipe_oracle(x):
    """The NVFP4 recipe as tensorfold.cuda.nvfp4.experts computed it before nvfp4_rows (an independent oracle: whole
    tensor at once, nearest magnitude by argmin over the eight E2M1 distances, ties to the first, i.e. smaller)."""
    n, k = x.shape
    x = x.float()
    g = (x.abs().amax() / (6.0 * 448.0)).clamp_min(1e-30)
    blocks = x.view(n, k // 16, 16)
    s = (blocks.abs().amax(-1) / 6.0 / g).clamp(max=448.0).to(torch.float8_e4m3fn)
    step = (s.float() * g)[..., None]
    v = torch.where(step > 0, blocks / step.clamp_min(1e-30), torch.zeros_like(blocks))
    code = ((v.abs()[..., None] - torch.tensor(MAGS)).abs().argmin(-1) + 8 * (v < 0).to(torch.int64)).view(n, k)
    code = code.to(torch.uint8)
    return code[:, 0::2] | (code[:, 1::2] << 4), s.view(torch.uint8), g


def _nvfp4_values(codes, scales, g):
    """What an NVFP4 copy holds, exactly, in FP32: the checkpoint format's numpy dequantizer (CPU tensors)."""
    from tensorfold.cuda.nvfp4.format import dequant

    return torch.from_numpy(dequant("nvfp4", codes.numpy(), scales.numpy(), float(g)))


def test_nvfp4_rows_match_the_recipe_bit_for_bit_in_any_chunking():
    from tensorfold.cuda.requant import nvfp4_rows

    gen = torch.Generator().manual_seed(4)
    w = (torch.randn(200, 128, generator=gen) * torch.logspace(-3, 0.5, 200)[:, None]).bfloat16()
    w[5] = 0
    w[9, :16] = 0                                              # an all-zero block: scale 0, zero codes
    want = _recipe_oracle(w)
    for chunk in (7, 64, 200, 4096):
        codes, scales, g = nvfp4_rows(w, chunk=chunk)
        assert codes.dtype == scales.dtype == torch.uint8 and codes.shape == (200, 64) and scales.shape == (200, 8)
        assert torch.equal(codes, want[0]) and torch.equal(scales, want[1]) and torch.equal(g, want[2]), chunk
    assert not codes[5].any() and not scales[9, 0]


def test_nvfp4_ties_round_to_the_smaller_magnitude_like_the_recipe():
    """Global 2**-10 and a unit block scale make the quotients exact: every E2M1 midpoint, signs, and the grid."""
    from tensorfold.cuda.requant import nvfp4_rows

    mids = [(a + b) / 2 for a, b in itertools.pairwise(MAGS)]
    row = [6.0] + mids + [-m for m in mids] + [0.0]            # 16 values; the block max 6 sets its scale to 1
    grid = [6.0] + list(MAGS) + [-v for v in MAGS[:7]]
    w = torch.tensor([row, grid, [6.0 * 448.0] + [0.0] * 15], dtype=torch.float32) * 2.0 ** -10
    codes, scales, g = nvfp4_rows(w.bfloat16())
    want = _recipe_oracle(w.bfloat16())
    assert torch.equal(codes, want[0]) and torch.equal(scales, want[1])
    values = _nvfp4_values(codes, scales, g) / 2.0 ** -10
    assert values[0].tolist() == [6.0] + list(MAGS[:-1]) + [-v for v in MAGS[:-1]] + [0.0]
    assert values[1].tolist() == grid                          # the grid itself is exact


def test_nvfp4_values_stay_within_one_block_step():
    from tensorfold.cuda.requant import nvfp4_rows

    gen = torch.Generator().manual_seed(8)
    w = (torch.randn(64, 256, generator=gen) * torch.logspace(-2, 1, 64)[:, None]).bfloat16()
    codes, scales, g = nvfp4_rows(w)
    step = scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, 1) * g
    # E2M1 magnitudes are at most 2 apart and the block scale rounds to E4M3 (quotients <= 6 / (1 - 2**-4)):
    # an element moves by at most one step of its block.
    assert bool(((_nvfp4_values(codes, scales, g) - w.float()).abs() <= step).all())


@pytest.mark.parametrize("bad", [torch.zeros(4, 24), torch.zeros(2, 32, dtype=torch.int32),
                                 torch.tensor([[1.0, float("inf")] * 8])])
def test_malformed_nvfp4_weights_are_refused(bad):
    from tensorfold.cuda.requant import nvfp4_rows

    with pytest.raises(ValueError):
        nvfp4_rows(bad)


def test_nvfp4_accounting():
    from tensorfold.cuda.requant import resident_bytes, staging_bytes

    assert resident_bytes("nvfp4", 1000, 256) == 1024 * 128 + 1024 * 16
    assert staging_bytes("nvfp4", 248320, 2048) > resident_bytes("nvfp4", 248320, 2048)
