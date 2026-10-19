"""Load-time requantization of BF16 weights (tensorfold.cuda.requant): one scale an output row, E4M3 codes."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda.requant import E4M3_MAX, requant_rows  # noqa: E402


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
