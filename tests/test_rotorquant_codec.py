"""The versioned packed format, independent matrix/bit oracle and FP32 encoder.

No prototype upstream quantizer is imported. CPU-only contracts run without Torch;
Torch cases check the staged differential oracle consumed by CUDA kernel tests.
"""

import bisect
import dataclasses
import math
import random
import struct

import pytest

from tensorfold.families.qwen4_exp.cuda import rotorquant_ref as rq


def _phi(x):
    return 0.0 if math.isinf(x) else math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


def _cdf(x):
    return math.erfc(-x / math.sqrt(2)) / 2


@pytest.mark.parametrize("bits", [3, 4])
def test_codebooks_are_pinned_float32_normal_lloyd_max_fixed_points(bits):
    book, thresholds = rq.centroids(bits), rq.thresholds(bits)
    assert len(book) == 1 << bits and len(thresholds) == (1 << bits) - 1
    assert tuple(sorted(book)) == book
    assert tuple(-value for value in reversed(book)) == book
    assert tuple(struct.unpack("<f", struct.pack("<f", value))[0] for value in book) == book
    assert tuple(struct.unpack("<f", struct.pack("<f", value))[0] for value in thresholds) == thresholds
    midpoint = [(a + b) / 2 for a, b in zip(book, book[1:])]
    assert tuple(struct.unpack("<f", struct.pack("<f", value))[0] for value in midpoint) == thresholds
    edges = [-math.inf, *midpoint, math.inf]
    mse = 0.0
    for value, a, b in zip(book, edges, edges[1:]):
        probability = _cdf(b) - _cdf(a)
        first = _phi(a) - _phi(b)
        assert abs(value - first / probability) < 2e-7
        second = probability + (0 if math.isinf(a) else a * _phi(a)) - (0 if math.isinf(b) else b * _phi(b))
        mse += second - 2 * value * first + value * value * probability
    assert abs(mse - {3: 0.03454776078850356, 4: 0.00950100800819157}[bits]) < 2e-12


def test_rotation_tables_are_actual_float32_values_with_explicit_provenance():
    for i, (c, s) in enumerate(zip(rq.PLANAR_COS, rq.PLANAR_SIN)):
        angle = (2 * i + 1) * math.pi / 8
        assert c == struct.unpack("<f", struct.pack("<f", math.cos(angle)))[0]
        assert s == struct.unpack("<f", struct.pack("<f", math.sin(angle)))[0]
        assert abs(c * c + s * s - 1) < 6e-8
    assert rq.ISO_QUATERNION == (0.5, 0.5, 0.5, 0.5)
    assert sum(x * x for x in rq.ISO_QUATERNION) == 1


@pytest.mark.parametrize("bits", [3, 4])
def test_exact_midpoints_choose_lower_indices(bits):
    threshold = rq.thresholds(bits)
    for index, value in enumerate(threshold):
        assert bisect.bisect_left(threshold, value) == index
        assert bisect.bisect_left(threshold, math.nextafter(value, math.inf)) == index + 1
        assert bisect.bisect_left(threshold, math.nextafter(value, -math.inf)) == index


def test_format_identity_is_stable_immutable_and_distinguishes_semantic_options():
    formats = [rq.descriptor(bits, variant) for bits in (3, 4) for variant in ("planar", "isofast")]
    assert len({item.codec_id for item in formats}) == 4
    for item in formats:
        assert item == rq.descriptor(item.bits, item.variant)
        assert item.group == 128 and item.version == 1
        assert item.scale_dtype == "float32" and item.norm_policy == "original-rms"
        assert item.group_bytes == {3: 48, 4: 64}[item.bits]
        assert item.vector_bytes_per_group == item.group_bytes + 4
        with pytest.raises(dataclasses.FrozenInstanceError):
            item.group = 32


@pytest.mark.parametrize("bits", [3, 4])
def test_packing_all_codes_at_every_component_and_every_plane_boundary(bits):
    for index in range(128):
        for value in range(1 << bits):
            codes = [0] * 128
            codes[index] = value
            payload = rq.pack_indices_ref(codes, bits)
            assert len(payload) == 128 * bits // 8
            assert rq.unpack_indices_ref(payload, bits) == codes


@pytest.mark.parametrize("bits", [3, 4])
def test_planes_are_group_local_and_pack_known_byte_order(bits):
    codes = [i % (1 << bits) for i in range(128)] + [(1 << bits) - 1] * 128
    payload = rq.pack_indices_ref(codes, bits)
    first = bytes([0xE4] * 32 + [0xF0] * 16) if bits == 3 else bytes.fromhex("1032547698badcfe") * 8
    assert payload == first + bytes([255] * (128 * bits // 8))
    assert rq.unpack_indices_ref(payload, bits) == codes
    assert rq.unpack_indices_ref(memoryview(payload), bits) == codes


@pytest.mark.parametrize("bits", [3, 4])
def test_empty_packing_is_well_defined(bits):
    assert rq.pack_indices_ref([], bits) == b""
    assert rq.unpack_indices_ref(b"", bits) == []


@pytest.mark.parametrize("bits", [False, 0, 2, 5, 9, 3.0, "4"])
def test_invalid_bit_width_is_rejected(bits):
    with pytest.raises(ValueError, match="bits"):
        rq.descriptor(bits)
    with pytest.raises(ValueError, match="bits"):
        rq.pack_indices_ref([], bits)


@pytest.mark.parametrize("value", [-1, 16, 1.0, True, None])
def test_invalid_centroid_indices_are_rejected(value):
    with pytest.raises(ValueError, match="index"):
        rq.pack_indices_ref([value] * 128, 4)


def test_partial_groups_invalid_variants_and_mismatched_metadata_are_rejected():
    with pytest.raises(ValueError, match="complete"):
        rq.pack_indices_ref([0] * 127, 3)
    with pytest.raises(ValueError, match="complete"):
        rq.unpack_indices_ref(b"\x00" * 49, 3)
    with pytest.raises(ValueError, match="byte sequence"):
        rq.unpack_indices_ref([0] * 48, 3)
    with pytest.raises(ValueError, match="variant"):
        rq.descriptor(4, "full")
    with pytest.raises(ValueError, match="multiple"):
        rq.quantize_oracle([0.0] * 127)
    with pytest.raises(ValueError, match="RMS count"):
        rq.dequantize_oracle(b"\x00" * 64, [], 4)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        rq.dequantize_oracle(b"\x00" * 64, [-1], 4)


@pytest.mark.parametrize("variant", ["planar", "isofast"])
@pytest.mark.parametrize("width", [128, 256, 384])
def test_fp64_matrix_oracle_preserves_norms_dots_and_inverse(variant, width):
    generator = random.Random(42)
    for _ in range(8):
        a = [generator.gauss(0, 1) for _ in range(width)]
        b = [generator.gauss(0, 1) for _ in range(width)]
        rotated_a, rotated_b = rq.rotate_oracle(a, variant), rq.rotate_oracle(b, variant)
        back = rq.rotate_oracle(rotated_a, variant, inverse=True)
        # Transposing rounded Planar coefficients gives (c²+s²)*I; the
        # absolute inverse error scales with the input, not a fixed magnitude.
        assert max(abs(x - y) for x, y in zip(back, a)) < max(abs(x) for x in a) * 7e-8 + 2e-15
        norm = math.fsum(x * x for x in a)
        assert abs(math.fsum(x * x for x in rotated_a) - norm) < norm * 7e-8
        dot = math.fsum(x * y for x, y in zip(a, b))
        transformed_dot = math.fsum(x * y for x, y in zip(rotated_a, rotated_b))
        assert abs(transformed_dot - dot) < math.sqrt(norm * math.fsum(x * x for x in b)) * 7e-8


@pytest.mark.parametrize("bits", [3, 4])
@pytest.mark.parametrize("variant", ["planar", "isofast"])
def test_fp64_zero_and_group_scaling_are_explicit(bits, variant):
    zero_payload, zero_scale = rq.quantize_oracle([0.0] * 256, bits, variant)
    assert zero_scale == (0.0, 0.0)
    assert rq.unpack_indices_ref(zero_payload, bits) == [(1 << (bits - 1)) - 1] * 256
    assert rq.dequantize_oracle(zero_payload, zero_scale, bits, variant, inverse=True) == [0.0] * 256
    values = [math.sin(i) * 10000 for i in range(128)] + [math.cos(i) * 0.1 for i in range(128)]
    packed, scales = rq.quantize_oracle(values, bits, variant)
    assert scales[0] > 6000 and scales[1] < 0.1
    reconstructed = rq.dequantize_oracle(packed, scales, bits, variant, inverse=True)
    assert all(math.isfinite(x) for x in reconstructed)
    assert len(reconstructed) == len(values)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, 1e39, 1e-50])
def test_fp64_oracle_rejects_nonfinite_or_unrepresentable_scale(value):
    with pytest.raises(ValueError):
        rq.quantize_oracle([value] * 128)


def _torch():
    return pytest.importorskip("torch")


@pytest.mark.parametrize("bits", [3, 4])
def test_tensor_packing_matches_independent_byte_loop(bits):
    torch = _torch()
    source = torch.arange(2 * 3 * 256).reshape(2, 3, 256) % (1 << bits)
    packed = rq.pack_ref(source, bits)
    assert packed.dtype == torch.uint8
    assert packed.shape == (2, 3, 256 * bits // 8)
    assert torch.equal(rq.unpack_ref(packed, bits), source)
    for index in range(6):
        expected = rq.pack_indices_ref(source.reshape(6, 256)[index].tolist(), bits)
        assert bytes(packed.reshape(6, -1)[index].tolist()) == expected


@pytest.mark.parametrize("bits", [3, 4])
@pytest.mark.parametrize("variant", ["planar", "isofast"])
def test_staged_float32_reference_matches_independent_fp64_oracle(bits, variant):
    torch = _torch()
    generator = torch.Generator().manual_seed(17)
    values = (torch.randn((4, 3, 256), generator=generator) * 0.5).to(torch.bfloat16)
    packed, rms = rq.quantize_ref(values, bits, variant)
    reconstructed = rq.dequant_ref(packed, rms, bits, variant)
    assert packed.dtype == torch.uint8 and rms.dtype == torch.float32
    assert rms.shape == (4, 3, 2)
    for row, scale, payload, decoded in zip(values.reshape(-1, 256), rms.reshape(-1, 2),
                                           packed.reshape(-1, 256 * bits // 8), reconstructed.reshape(-1, 256)):
        oracle_payload, oracle_rms = rq.quantize_oracle(row.float().tolist(), bits, variant)
        # Seeded BF16 vectors are well separated from thresholds: exact packed parity.
        assert bytes(payload.tolist()) == oracle_payload
        assert torch.allclose(scale, torch.tensor(oracle_rms), atol=1e-7, rtol=2e-7)
        independently_decoded = rq.dequantize_oracle(bytes(payload.tolist()), scale.tolist(), bits, variant)
        assert torch.allclose(decoded, torch.tensor(independently_decoded), atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("bits", [3, 4])
@pytest.mark.parametrize("variant", ["planar", "isofast"])
def test_stable_float32_rms_handles_zero_finite_large_and_small_groups(bits, variant):
    torch = _torch()
    values = torch.zeros((4, 256), dtype=torch.bfloat16)
    values[1] = 10000
    values[2] = 1e30
    values[3] = 1e-30
    payload, rms = rq.quantize_ref(values, bits, variant)
    assert torch.isfinite(rms).all()
    assert torch.equal(rms[0], torch.zeros(2))
    assert torch.equal(rms[1], torch.tensor([9984.0, 9984.0]))
    assert torch.equal(rq.unpack_ref(payload, bits)[0], torch.full((256,), (1 << (bits - 1)) - 1))
    reconstruction = rq.dequant_ref(payload, rms, bits, variant, inverse=True)
    assert torch.equal(reconstruction[0], torch.zeros(256))
    assert torch.isfinite(reconstruction).all()
    assert bool((rms[2:] > 0).all())


@pytest.mark.parametrize("variant", ["planar", "isofast"])
def test_float32_rotation_matches_dense_oracle_on_noncontiguous_input(variant):
    torch = _torch()
    x = torch.randn((3, 256), generator=torch.Generator().manual_seed(8)).T.contiguous().T
    assert not x.is_contiguous()
    actual = rq.rotate_ref(x, variant)
    expected = torch.tensor([rq.rotate_oracle(row.tolist(), variant) for row in x])
    assert torch.allclose(actual, expected, atol=5e-7, rtol=5e-7)
    assert torch.allclose(rq.rotate_ref(actual, variant, inverse=True), x, atol=5e-7, rtol=5e-7)


def test_tensor_reference_rejects_malformed_shapes_types_scales_and_nonfinite_values():
    torch = _torch()
    with pytest.raises(ValueError, match="floating dtype"):
        rq.quantize_ref(torch.ones((1, 128), dtype=torch.int32))
    with pytest.raises(ValueError, match="floating dtype"):
        rq.quantize_ref(torch.full((1, 128), 1e-50, dtype=torch.float64))
    for value in (math.nan, math.inf):
        with pytest.raises(ValueError, match="finite"):
            rq.quantize_ref(torch.full((1, 128), value))
    with pytest.raises(ValueError, match="multiple"):
        rq.quantize_ref(torch.ones((1, 127)))
    with pytest.raises(ValueError, match="integer dtype"):
        rq.pack_ref(torch.zeros((1, 128), dtype=torch.float32))
    with pytest.raises(ValueError, match="outside"):
        rq.pack_ref(torch.full((1, 128), 16, dtype=torch.int32))
    with pytest.raises(ValueError, match="uint8"):
        rq.unpack_ref(torch.zeros((1, 64), dtype=torch.float32))
    with pytest.raises(ValueError, match="matching group shape"):
        rq.dequant_ref(torch.zeros((1, 64), dtype=torch.uint8), torch.ones((1, 2)))
    with pytest.raises(ValueError, match="matching group shape"):
        rq.dequant_ref(torch.zeros((1, 64), dtype=torch.uint8), torch.ones((1, 1), dtype=torch.float64))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        rq.dequant_ref(torch.zeros((1, 64), dtype=torch.uint8), torch.full((1, 1), -1.0))
    with pytest.raises(ValueError, match="finite FP32"):
        rq.dequant_ref(torch.full((1, 64), 255, dtype=torch.uint8), torch.full((1, 1), 3e38))


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _float32_tree_rotated(values, variant, tree):
    """Explicit integer-indexed RN operations, independent of Torch/Triton."""

    maximum = max(abs(value) for value in values)
    z = [_f32(value / maximum) for value in values]
    terms = [_f32(value * value) for value in z]
    if tree == "pairwise":
        while len(terms) > 1:
            terms = [_f32(a + b) for a, b in zip(terms[::2], terms[1::2])]
        total = terms[0]
    else:
        total = 0.0
        for value in terms if tree == "forward" else reversed(terms):
            total = _f32(total + value)
    t = _f32(math.sqrt(total / 128))
    unit = [_f32(value / t) for value in z]
    output = []
    if variant == "planar":
        for i in range(0, 128, 2):
            a, b = unit[i:i + 2]
            c, s = rq.PLANAR_COS[i // 2 % 4], rq.PLANAR_SIN[i // 2 % 4]
            output.extend((_f32(_f32(a*c) - _f32(b*s)), _f32(_f32(a*s) + _f32(b*c))))
    else:
        for i in range(0, 128, 4):
            a, b, c, d = unit[i:i + 4]
            for signs in ((a, -b, -c, -d), (a, b, -c, d), (a, b, c, -d), (a, -b, c, d)):
                total = signs[0]
                for value in signs[1:]:
                    total = _f32(total + value)
                output.append(_f32(total * 0.5))
    return output, _f32(maximum * t)


@pytest.mark.parametrize("variant", ["planar", "isofast"])
@pytest.mark.parametrize("bits", [3, 4])
def test_analytic_rounding_envelope_covers_independent_float32_sum_trees(variant, bits):
    generator = random.Random(912)
    for _ in range(8):
        values = [_f32(generator.gauss(0, 2)) for _ in range(128)]
        for tree in ("forward", "reverse", "pairwise"):
            actual, rms = _float32_tree_rotated(values, variant, tree)
            ideal, bound, exact_rms, rms_bound = rq.rounding_envelope_oracle(values, variant, rms=[rms])
            assert abs(rms - exact_rms[0]) <= rms_bound[0]
            assert all(abs(value - exact) <= error for value, exact, error in zip(actual, ideal, bound))
            low, high = rq.index_envelope_oracle(values, bits, variant, rms=[rms])
            codes = [bisect.bisect_left(rq.thresholds(bits), value) for value in actual]
            assert all(a <= code <= b for a, code, b in zip(low, codes, high))
            assert all(b - a <= 1 for a, b in zip(low, high))


@pytest.mark.parametrize("bits", [3, 4])
def test_exact_bf16_cancellation_has_a_derived_two_bin_interval_not_exact_byte_promise(bits):
    torch = _torch()
    values = torch.randn((3, 2, 256), generator=torch.Generator().manual_seed(37)).to(torch.bfloat16)
    group = values.float().reshape(-1, 128)[3]
    assert group[20:24].tolist() == [-1.2265625, -0.259765625, 0.6875, 0.279296875]
    # Exact source dyadics cancel before normalization; FP32 dividing each
    # component can introduce a signed residual around the exact zero threshold.
    assert rq.rotate_oracle(group.tolist(), "isofast")[23] == 0
    maximum = group.abs().max()
    z = group / maximum
    base = (z.square().sum() / 128).sqrt()
    observed = set()
    for shift in range(-4, 5):
        t = base.clone()
        for _ in range(abs(shift)):
            t = torch.nextafter(t, torch.tensor(math.inf if shift > 0 else -math.inf))
        rotated = rq.rotate_ref((z/t).reshape(1, 128), "isofast")[0]
        rms = (maximum*t).item()
        ideal, error, _, _ = rq.rounding_envelope_oracle(group.tolist(), "isofast", rms=[rms])
        assert ideal[23] == 0
        assert abs(rotated[23].item()) <= error[23]
        low, high = rq.index_envelope_oracle(group.tolist(), bits, "isofast", rms=[rms])
        lower = (1 << (bits-1)) - 1
        assert (low[23], high[23]) == (lower, lower + 1)
        code = bisect.bisect_left(rq.thresholds(bits), rotated[23].item())
        assert lower <= code <= lower + 1
        observed.add(code)
    assert observed == {(1 << (bits-1)) - 1, 1 << (bits-1)}


def test_rounding_envelope_rejects_unproved_inputs_and_bad_metadata():
    zero, error, rms, rms_error = rq.rounding_envelope_oracle([0.0] * 128, rms=[0.0])
    assert zero == error == [0.0] * 128 and rms == rms_error == [0.0]
    with pytest.raises(ValueError, match="exactly represented"):
        rq.rounding_envelope_oracle([0.1] * 128)
    with pytest.raises(ValueError, match="normal RMS"):
        rq.rounding_envelope_oracle([2.0**-149] * 128)
    with pytest.raises(ValueError, match="normal RMS"):
        rq.rounding_envelope_oracle([1.0] + [2.0**-100] * 127)
    with pytest.raises(ValueError, match="RMS count"):
        rq.rounding_envelope_oracle([1.0] * 128, rms=[])
    with pytest.raises(ValueError, match="zero RMS"):
        rq.rounding_envelope_oracle([0.0] * 128, rms=[1.0])
    with pytest.raises(ValueError, match="outside the derived"):
        rq.rounding_envelope_oracle([1.0] * 128, rms=[2.0])
