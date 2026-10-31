"""SM70 Q4 codec proof against adopted staged and independent scalar oracles."""
import math

import numpy as np
import pytest
import torch

from tensorfold.cuda.sm70 import rotorquant_q4_native as native
from tensorfold.families.qwen4_exp.cuda import rotorquant_ref as ref

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason="SM70 native Q4 codec")
VARIANTS = ((1, "planar"), (2, "isofast"), (3, "iso64-norm"),
            (4, "signed-iso128"), (5, "signed-iso64-norm"))


def status():
    return torch.tensor([0, -2], device="cuda", dtype=torch.int32)


def storage(rows, heads=1, dim=256):
    return (torch.full((rows, heads, dim // 2), 0x55, device="cuda", dtype=torch.uint8),
            torch.full((rows, heads, dim // 128), 2.0, device="cuda", dtype=torch.float32))


@pytest.fixture(scope="module", autouse=True)
def prepare():
    native.extension()                    # All captures below borrow the already prepared immutable binary.


@pytest.mark.parametrize("codec,variant", VARIANTS)
@pytest.mark.parametrize("inverse", [False, True])
def test_staged_rotation_is_bit_exact(codec, variant, inverse):
    torch.manual_seed(271)
    x = torch.randn((3, 3, 256), device="cuda")
    x[:, :, :2] = torch.tensor([-0.0, 0.0], device="cuda")
    out, errors = torch.empty_like(x), status()
    native.rotate(x, out, codec=codec, inverse=inverse, status=errors)
    expected = ref.rotate_ref(x, variant, inverse=inverse)
    assert torch.equal(out.view(torch.int32), expected.view(torch.int32))
    assert errors.tolist() == [0, -2]


@pytest.mark.parametrize("source_dtype,output_dtype", [(torch.float16, torch.float16),
                                                       (torch.float16, torch.float32),
                                                       (torch.float32, torch.float16)])
def test_rotation_has_only_the_declared_half_boundary(source_dtype, output_dtype):
    torch.manual_seed(31)
    x = torch.randn((3, 3, 256), device="cuda").to(source_dtype)
    out = torch.empty(x.shape, device="cuda", dtype=output_dtype)
    native.rotate(x, out, status=status())
    torch.testing.assert_close(out, ref.rotate_ref(x, "signed-iso128").to(output_dtype), rtol=0, atol=0)


@pytest.mark.parametrize("codec,variant", VARIANTS)
def test_encoding_satisfies_independent_index_and_metadata_envelopes(codec, variant):
    torch.manual_seed(213)
    x = torch.randn((2, 2, 256), device="cuda", dtype=torch.float16) * .3
    codes, scales = storage(2, 2)
    errors = status()
    native.encode(x, codes, scales, codec=codec, status=errors)
    assert errors.tolist() == [0, -2]
    for values, packed, rms in zip(x.float().cpu().reshape(-1, 256).tolist(),
                                  codes.cpu().reshape(-1, 128).tolist(),
                                  scales.cpu().reshape(-1, 2).tolist()):
        lo, hi = ref.index_envelope_oracle(values, 4, variant, rms=rms)
        actual = ref.unpack_indices_ref(bytes(packed), 4)
        assert all(low <= code <= high for low, code, high in zip(lo, actual, hi))
        _, _, expected, bounds = ref.rounding_envelope_oracle(values, variant)
        assert all(abs(a-b) <= bound for a, b, bound in zip(rms, expected, bounds))


def test_zero_lower_center_and_original_rms_identity():
    x = torch.zeros((2, 1, 256), device="cuda", dtype=torch.float16)
    codes, scales = storage(2)
    native.encode(x, codes, scales, status=status())
    assert bool((codes == 0x77).all())
    assert bool((scales.view(torch.int32) == 0).all())


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("codec,variant", [(4, "signed-iso128"), (5, "signed-iso64-norm")])
def test_decode_preserves_exact_existing_packing_and_basis(inverse, dtype, codec, variant):
    packed = torch.arange(256, device="cuda", dtype=torch.int32).to(torch.uint8).reshape(2, 1, 128)
    scales = torch.tensor([[[0., .25]], [[1., 3.5]]], device="cuda")
    out = torch.empty((2, 1, 256), device="cuda", dtype=dtype)
    errors = status()
    native.decode(packed, scales, out, codec=codec, inverse=inverse, status=errors)
    expected = ref.dequant_ref(packed, scales, 4, variant, inverse=inverse).to(dtype)
    assert torch.equal(out.view(torch.int16 if dtype == torch.float16 else torch.int32),
                       expected.view(torch.int16 if dtype == torch.float16 else torch.int32))
    assert errors.tolist() == [0, -2]


def test_codec5_exact_reconstruction_norm_metadata_on_uniform_coding_coordinates():
    # Independent dense FP64 matrix oracle constructs an exactly representable
    # unit-RMS source whose coding coordinates are all+1. Every chosen centroid
    # has the same magnitude, so its energy has an exact power-of-two sum and
    # the correction has a closed-form scalar FP32 contract, without tolerance.
    values = ref.rotate_oracle([1.] * 128, "signed-iso64-norm", inverse=True)
    assert math.hypot(*values) / math.sqrt(128) == 1.
    x = torch.tensor(values, device="cuda").reshape(1, 1, 128)
    codes, scales = storage(1, dim=128)
    errors = status()
    native.encode(x, codes, scales, codec=5, status=errors)
    center = ref.CENTROIDS_4[11]
    energy = ref._f32(ref._f32(center * center) * 128)
    correction = ref._f32(math.sqrt(ref._f32(128 / energy)))
    expected = torch.tensor(correction, device="cuda")
    assert bool((codes == 0xBB).all())
    assert scales.view(torch.int32).item() == expected.view(torch.int32).item()
    assert correction != 1.
    rotated, restored = torch.empty_like(x), torch.empty_like(x)
    native.decode(codes, scales, rotated, codec=5, status=errors)
    native.decode(codes, scales, restored, codec=5, inverse=True, status=errors)
    assert torch.equal(rotated, torch.ones_like(x))
    torch.testing.assert_close(restored, x, rtol=0, atol=0)
    assert errors.tolist() == [0, -2]


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_bad_source_is_contained_and_first_layer_is_sticky(bad):
    x = torch.ones((1, 1, 256), device="cuda")
    x[0, 0, 3] = bad
    codes, scales = storage(1)
    errors = torch.tensor([4, 9], device="cuda", dtype=torch.int32)
    native.encode(x, codes, scales, status=errors, error=2, layer=5)
    assert errors.tolist() == [6, 9]
    assert bool(torch.isfinite(scales).all())
    assert bool((codes[0, 0, :64] == 0x77).all())
    assert scales[0, 0, 0].item() == 0


def test_finite_high_source_that_overflows_half_reconstruction_is_refused():
    x = torch.full((1, 1, 256), 60000., device="cuda")
    codes, scales = storage(1)
    errors = status()
    native.encode(x, codes, scales, status=errors, layer=7)
    assert errors.tolist() == [8, 7]
    out = torch.empty_like(x, dtype=torch.float16)
    native.decode(codes, scales, out, status=errors)
    assert bool(torch.isfinite(out).all())


def test_query_rotation_half_overflow_is_contained_at_its_boundary():
    x = torch.full((1, 1, 256), 60000., device="cuda")
    out, errors = torch.empty_like(x, dtype=torch.float16), status()
    native.rotate(x, out, status=errors, error=1, layer=6)
    assert errors.tolist() == [1, 6]
    assert bool(torch.isfinite(out).all())


def test_nonzero_fp32_rms_underflow_is_not_silently_erased():
    x = torch.zeros((1, 1, 128), device="cuda")
    x[0, 0, 0] = np.nextafter(np.float32(0), np.float32(1)).item()
    codes, scales = storage(1, dim=128)
    errors = status()
    native.encode(x, codes, scales, status=errors, layer=10)
    assert errors.tolist() == [8, 10]
    assert bool((codes == 0x77).all()) and scales.item() == 0


def test_decode_half_overflow_boundary_is_distinct_from_f32_wire_validity():
    center = ref.CENTROIDS_4[-1]
    boundary = np.float32(65520. / center)
    lo = np.nextafter(boundary, np.float32(0))
    hi = np.nextafter(boundary, np.float32(math.inf))
    codes = torch.full((2, 1, 64), 0xFF, device="cuda", dtype=torch.uint8)
    scales = torch.tensor([lo, hi], device="cuda", dtype=torch.float32).reshape(2, 1, 1)
    wide = torch.empty((2, 1, 128), device="cuda")
    wire_status = status()
    native.decode(codes, scales, wide, status=wire_status)
    assert wire_status.tolist() == [0, -2]
    half = torch.empty_like(wide, dtype=torch.float16)
    errors = status()
    native.decode(codes, scales, half, status=errors, layer=11)
    assert errors.tolist() == [8, 11]
    assert bool(torch.isfinite(half).all())
    assert bool((half[1] == 0).all())


def test_invalid_metadata_and_inverse_overflow_cannot_escape():
    codes = torch.full((3, 1, 64), 0xFF, device="cuda", dtype=torch.uint8)
    scales = torch.tensor([math.nan, -1., math.inf], device="cuda").reshape(3, 1, 1)
    out, errors = torch.empty((3, 1, 128), device="cuda"), status()
    native.decode(codes, scales, out, inverse=True, status=errors, layer=4)
    assert errors.tolist() == [8, 4]
    assert bool((out == 0).all())


def test_dead_rows_preserve_cache_even_with_nan_and_out_of_bounds_position():
    x = torch.full((3, 1, 256), math.nan, device="cuda")
    codes, scales = storage(2)
    before = codes.clone(), scales.clone()
    errors = status()
    native.encode(x, codes, scales, position=torch.tensor([2**63-1], device="cuda"),
                  valid_rows=torch.tensor([0], device="cuda", dtype=torch.int32), status=errors)
    assert errors.tolist() == [0, -2]
    assert torch.equal(codes, before[0]) and torch.equal(scales, before[1])


def test_last_active_cache_row_and_padded_suffix_do_not_cross_ownership():
    x = torch.ones((3, 1, 256), device="cuda")
    x[1:] = math.nan
    codes, scales = storage(4)
    before = codes.clone(), scales.clone()
    errors = status()
    native.encode(x, codes, scales, position=torch.tensor([3], device="cuda"),
                  valid_rows=torch.tensor([1], device="cuda", dtype=torch.int32), status=errors)
    assert errors.tolist() == [0, -2]
    assert torch.equal(codes[:3], before[0][:3]) and torch.equal(scales[:3], before[1][:3])
    assert not torch.equal(codes[3], before[0][3])


@pytest.mark.parametrize("position", [-1, 4, 2**63-1])
def test_invalid_active_device_position_marks_index_without_mutation(position):
    x = torch.ones((1, 1, 256), device="cuda")
    codes, scales = storage(4)
    old = codes.clone(), scales.clone()
    errors = status()
    native.encode(x, codes, scales, position=torch.tensor([position], device="cuda"), status=errors, layer=3)
    assert errors.tolist() == [16, 3]
    assert torch.equal(codes, old[0]) and torch.equal(scales, old[1])


@pytest.mark.parametrize("count", [-1, 4])
def test_invalid_device_live_count_never_writes_cache(count):
    x = torch.ones((3, 1, 256), device="cuda")
    codes, scales = storage(4)
    old = codes.clone(), scales.clone()
    errors = status()
    native.encode(x, codes, scales, valid_rows=torch.tensor([count], device="cuda", dtype=torch.int32),
                  status=errors, layer=2)
    assert errors.tolist() == [16, 2]
    assert torch.equal(codes, old[0]) and torch.equal(scales, old[1])


def test_zero_rows_and_strided_projection_source():
    parent = torch.randn((3, 1024), device="cuda", dtype=torch.float16)
    x = parent[:, 128:384].reshape(3, 1, 256)
    codes, scales = storage(3)
    errors = status()
    native.encode(x, codes, scales, status=errors)
    expected_codes, expected_scales = storage(3)
    native.encode(x.contiguous(), expected_codes, expected_scales, status=status())
    assert torch.equal(codes, expected_codes) and torch.equal(scales, expected_scales)
    old = codes.clone(), scales.clone()
    native.encode(x, codes, scales, rows=0, status=errors)
    assert torch.equal(codes, old[0]) and torch.equal(scales, old[1])


def test_encoding_every_row_is_independent_of_physical_batch_geometry():
    torch.manual_seed(451)
    x = torch.randn((18, 2, 256), device="cuda", dtype=torch.float16)
    codes, scales = storage(18, 2)
    native.encode(x, codes, scales, status=status())
    for row in range(18):
        independent_codes, independent_scales = storage(1, 2)
        native.encode(x[row:row+1], independent_codes, independent_scales, status=status())
        assert torch.equal(codes[row:row+1], independent_codes)
        assert torch.equal(scales[row:row+1], independent_scales)


def test_alias_dtype_and_geometry_refusals_precede_writes():
    x = torch.ones((1, 1, 128), device="cuda")
    with pytest.raises(RuntimeError, match="disjoint"):
        native.rotate(x, x)
    codes, scales = storage(1, dim=128)
    with pytest.raises(RuntimeError, match="never BF16"):
        native.encode(x.bfloat16(), codes, scales)
    with pytest.raises(RuntimeError, match="geometry"):
        native.encode(x[:, :, :127], codes, scales)
    backing = torch.empty(64, device="cuda", dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="disjoint"):
        native.encode(x, backing.reshape(1, 1, 64), backing[:4].view(torch.float32).reshape(1, 1, 1))


@pytest.mark.parametrize("codec", [4, 5])
def test_capture_replay_dynamic_positions_live_rows_and_half_boundaries(codec):
    torch.manual_seed(37)
    x = torch.randn((3, 1, 256), device="cuda", dtype=torch.float16)
    codes, scales = storage(16)
    expected_codes, expected_scales = codes.clone(), scales.clone()
    position = torch.tensor([0], device="cuda", dtype=torch.int64)
    live = torch.tensor([3], device="cuda", dtype=torch.int32)
    errors = status()
    query = torch.empty_like(x)
    output = torch.empty((16, 1, 256), device="cuda", dtype=torch.float16)
    native.encode(x, codes, scales, codec=codec, position=position, valid_rows=live, status=errors)
    codes.copy_(expected_codes)
    scales.copy_(expected_scales)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        native.encode(x, codes, scales, codec=codec, position=position, valid_rows=live, status=errors)
        native.rotate(x, query, codec=codec, valid_rows=live, status=errors)
        native.decode(codes, scales, output, codec=codec, status=errors)
    for at, count in ((0, 1), (2, 2), (7, 0), (13, 3)):
        position.fill_(at)
        live.fill_(count)
        graph.replay()
        native.encode(x, expected_codes, expected_scales, codec=codec,
                      position=position, valid_rows=live, status=status())
        assert torch.equal(codes, expected_codes) and torch.equal(scales, expected_scales)
        expected = torch.empty_like(output)
        native.decode(expected_codes, expected_scales, expected, codec=codec, status=status())
        assert torch.equal(output, expected)
        assert bool((query[count:] == 0).all())
    assert errors.tolist() == [0, -2]


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.int8, torch.uint8])
@pytest.mark.parametrize("rank", [2, 3])
def test_bitwise_row_copy_preserves_live_extent_and_projection_stride(dtype, rank):
    parent = torch.arange(120, device="cuda").reshape(3, 40).to(dtype)
    x = parent[:, 4:12]
    if rank == 3:
        x = x.reshape(3, 2, 4)
    target = torch.full((6, *x.shape[1:]), 17, device="cuda", dtype=dtype)
    expected = target.clone()
    expected[3:5].copy_(x[:2])
    errors = status()
    native.copy_rows(x, target, torch.tensor([3], device="cuda"),
                     valid_rows=torch.tensor([2], device="cuda", dtype=torch.int32), status=errors)
    assert torch.equal(target.view(torch.uint8), expected.view(torch.uint8))
    assert errors.tolist() == [0, -2]


def test_row_copy_carries_nan_payload_bits_without_arithmetic():
    bits = torch.arange(24, device="cuda", dtype=torch.int32).reshape(3, 8)
    bits[0, 0] = 0x7FC01234
    x = bits.view(torch.float32)
    target = torch.zeros((4, 8), device="cuda")
    native.copy_rows(x, target, torch.tensor([1], device="cuda"), status=status())
    assert torch.equal(target[1:].view(torch.int32), bits)


def test_row_copy_rejects_alias_dtype_and_incorrect_trailing_layout():
    x = torch.zeros((3, 8), device="cuda")
    with pytest.raises(RuntimeError, match="disjoint"):
        native.copy_rows(x, x, None)
    with pytest.raises(RuntimeError, match="declared CUDA storage"):
        native.copy_rows(x, x.half(), None)
    with pytest.raises(RuntimeError, match="contiguous"):
        native.copy_rows(x[:, ::2], torch.empty((3, 4), device="cuda"), None)


def test_row_copy_capture_uses_dynamic_positions_and_never_clamps_dead_rows():
    x = torch.arange(24, device="cuda", dtype=torch.float32).reshape(3, 8)
    target = torch.full((6, 8), -17., device="cuda")
    expected = target.clone()
    pos = torch.tensor([0], device="cuda", dtype=torch.int64)
    live = torch.tensor([0], device="cuda", dtype=torch.int32)
    errors = status()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        native.copy_rows(x, target, pos, valid_rows=live, status=errors)
    for start, count in ((2**63-1, 0), (2, 2), (5, 1)):
        pos.fill_(start)
        live.fill_(count)
        graph.replay()
        if count:
            expected[start:start+count].copy_(x[:count])
        assert torch.equal(target, expected)
    assert errors.tolist() == [0, -2]
    pos.fill_(-1)
    live.fill_(1)
    graph.replay()
    assert errors.tolist() == [16, 0]
    assert torch.equal(target, expected)
    native.copy_rows(x[:0], target, pos, status=errors)
    assert torch.equal(target, expected)
