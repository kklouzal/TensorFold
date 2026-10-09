"""Native Apple qualification for scalar-aligned generic SIMD views.

Source-only checks compile this Python file without importing SDKs. These tests
must execute on a supported Apple target and skipped cases do not qualify it.
"""
import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.kernels.qwen.dense.v1 import simd_qmm  # noqa: E402


def _metal():
    if not mx.metal.is_available():
        pytest.skip("Apple Metal execution hardware is unavailable")


def _offset(value):
    shifted = mx.concatenate([mx.zeros((1,), dtype=value.dtype), value.reshape(-1)])[1:].reshape(value.shape)
    mx.eval(shifted)
    return shifted


def test_scalar_bfloat_packing_preserves_all_payloads_at_two_byte_offset():
    _metal()
    raw = mx.arange(65536, dtype=mx.uint32).astype(mx.uint16)
    source = _offset(raw.view(mx.bfloat16))
    body = """
      const uint e = thread_position_in_grid.x;
      if (e >= 8192) return;
      const uint4 v = load_bf16_8(X + 8 * e);
      for (int j = 0; j < 4; j++) OUT[4 * e + j] = v[j];
    """
    run = mx.fast.metal_kernel(name="simd_scalar_alignment_all_bfloat_payloads", input_names=["X"],
                               output_names=["OUT"], source=body, header=simd_qmm._HEADER)
    out = run(inputs=[source], grid=(8192, 1, 1), threadgroup=(256, 1, 1),
              output_shapes=[(32768,)], output_dtypes=[mx.uint32])[0]
    assert bool(mx.all(out == raw.view(mx.uint32)).item())


@pytest.mark.parametrize("kind,rows", [("scalar", 1), ("scalar", 2), ("mma", 3), ("mma", 8), ("fragments", 8)])
@pytest.mark.parametrize("group", [32, 64])
def test_current_scalar_aligned_weight_input_and_fragment_views_match(kind, rows, group):
    _metal()
    mx.random.seed(101)
    n, k = 64, 256
    weight, scales, biases = mx.quantize((mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16),
                                        bits=4, group_size=group)
    x = mx.random.normal((rows, k)).astype(mx.bfloat16)
    if kind == "fragments":
        if group != 64:
            pytest.skip("the declared fragment ABI uses groups64")
        frags = simd_qmm.fragments(x)
        expected = simd_qmm.qmm_fragments(frags, weight, scales, biases)
        actual = simd_qmm.qmm_fragments(tuple(_offset(v) for v in frags), _offset(weight), _offset(scales), _offset(biases))
    else:
        expected = simd_qmm.qmm(x, weight, scales, biases, group, kind=kind)
        actual = simd_qmm.qmm(_offset(x), _offset(weight), _offset(scales), _offset(biases), group, kind=kind)
    assert bool(mx.all(mx.isfinite(actual)).item())
    assert bool(mx.all(actual.view(mx.uint16) == expected.view(mx.uint16)).item())
