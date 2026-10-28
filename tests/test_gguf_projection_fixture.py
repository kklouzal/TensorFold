"""Independent finite FP16 projection corpus; preserve the original overflow reproducer."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tensorfold.families.qwen4_exp.gguf import PackedTensor, decode_rows

DIRECTORY = Path(__file__).parent / "fixtures"
# Independent GGML wire schemas and quant multiplier bounds. The common8192
# bound includes every code/sign/local scale, without reading production tables.
SCHEMAS = [(14, 256, 210, 208, 4096), (18, 256, 98, 0, 1976.25),
           (21, 256, 110, 0, 7905), (22, 256, 82, 0, 988.125),
           (23, 256, 136, 0, 4064), (42, 64, 18, 0, 2)]


def test_original_full_range_wire_still_reproduces_fp16_projection_overflow():
    original = DIRECTORY / "gguf_mixed_ggml.npz"
    assert hashlib.sha256(original.read_bytes()).hexdigest() == "25bb9fbd8f5ccbdac5482d4ce1542327d3f00a8bd5d671c512ea679d0dcc5eaa"
    with np.load(original, allow_pickle=False) as fixture:
        values = fixture["bits_18"].view(np.float32)
        assert np.isfinite(values).all()  # Valid FP32 wire data, beyond FP16 range.
        with np.errstate(over="raise", invalid="raise"), pytest.raises(FloatingPointError):
            values.astype(np.float16)


@pytest.mark.parametrize("kind,block,size,scale_at,multiplier", SCHEMAS)
def test_finite_projection_scales_preserve_quant_codes_and_have_exact_independent_bits(kind, block, size, scale_at, multiplier):
    with np.load(DIRECTORY / "gguf_mixed_ggml.npz", allow_pickle=False) as fixture:
        original = fixture[f"raw_{kind}"].reshape(64, size)
    with np.load(DIRECTORY / "gguf_mixed_projection_ggml.npz", allow_pickle=False) as fixture:
        raw, expected = fixture[f"raw_{kind}"], fixture[f"bits_{kind}"].view(np.float32)
        provenance = json.loads(str(fixture["provenance"]))
    bank = raw.reshape(64, size)
    metadata = bank[:, scale_at:scale_at + 2].copy().view("<f2").reshape(-1)
    assert np.isfinite(metadata).all() and np.max(np.abs(metadata)) <= 2.**-20
    assert np.any(np.signbit(metadata)) and np.any(metadata == 0) and np.any(metadata > 0)
    keep = [offset for offset in range(size) if offset not in (scale_at, scale_at + 1)]
    np.testing.assert_array_equal(bank[:, keep], original[:, keep])
    assert np.isfinite(expected).all() and np.max(np.abs(expected)) <= multiplier * 2.**-20
    assert np.max(np.abs(expected)) <= 2.**-7
    with np.errstate(over="raise", invalid="raise"):
        half = expected.astype(np.float16)
    assert np.isfinite(half).all()
    tensor = PackedTensor("finite", expected.shape, kind, Path("unused"), 0, raw.nbytes)
    np.testing.assert_array_equal(decode_rows(tensor, raw).view(np.uint32), expected.view(np.uint32))
    assert provenance["revision"] == "b3fa69955ea81c3b60e77e3c1e2224c4a826ffb9"
    assert provenance["scale_domain"] == "finite-fp16-projections-v1"
    assert provenance["projection_bounds"]["maximum_abs_coefficient"] == 2.**-7
