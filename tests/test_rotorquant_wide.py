"""Independent dense matrix and scale bounds for fixed wide rotor contracts.

The matrices are assembled from disjoint linear maps, without importing the
optimized transform or staged Torch reference. This file needs no accelerator.
"""
from __future__ import annotations

import bisect
import math
import struct

import numpy as np
import pytest

from tensorfold.families.qwen4_exp.cuda import rotorquant_ref as ref

MASK = 0x42F85949E46A0359
COS = struct.unpack("<f", struct.pack("<f", math.sqrt(.5)))[0]
CASES = ((3, "iso64-norm", True, False, False),
         (4, "signed-iso128", False, True, True),
         (5, "signed-iso64-norm", True, True, False))


def wide_matrix(codec, *, mathematical=False):
    """Column-vector transform; inverse applies the transposed stages backwards."""
    signed = codec in (4, 5, 6, 7)
    matrix = np.diag([1 - 2 * ((MASK >> (index // 2)) & 1) for index in range(128)]) if signed else np.eye(128)
    left = np.array([[1, -1, -1, -1], [1, 1, -1, 1], [1, 1, 1, -1], [1, -1, 1, 1]], dtype=np.float64) * .5
    for low, high in ((0, 1), (2, 3), (4, 5)):
        bitmask = (1 << low) | (1 << high)
        stage = np.zeros((128, 128))
        for base in range(128):
            if base & bitmask:
                continue
            indices = [base, base | (1 << low), base | (1 << high), base | bitmask]
            stage[np.ix_(indices, indices)] = left
        matrix = stage @ matrix
    if codec in (4, 6, 7):
        c = math.sqrt(.5) if mathematical else COS
        stage = np.zeros((128, 128))
        for base in range(64):
            stage[np.ix_([base, base + 64], [base, base + 64])] = [[c, -c], [c, c]]
        matrix = stage @ matrix
    return matrix


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def wide_index_envelope(values, codec, thresholds):
    """Analytic normal-domain FP32 bin envelope, without stored-scale inference.

    Stable normalization uses gamma127 for any legal 128-term RN reduction.
    Each quaternion stage propagates absolute prior errors and gamma3 times
    its absolute signed-sum contributions. Givens uses gamma2 (two products
    and the final sum). This covers cancellation instead of imposing a fitted
    absolute tolerance. Binary64 oracle roundoff is included independently.
    """
    values = np.asarray(values, dtype=np.float64)
    require_normal = np.max(np.abs(values))
    rms = math.hypot(*values) / math.sqrt(128)
    if rms == 0:
        zero = bisect.bisect_left(thresholds, 0.)
        return [zero] * 128, [zero] * 128
    if rms < 2**-126 or any(x != 0 and (abs(x) / require_normal) ** 2 < 2 * 2**-126 for x in values):
        raise ValueError("the bin bound requires normal RMS and nonzero squared intermediates")
    u, q, oracle = 2**-24, 2**-150, 64 * 2**-53
    gamma127 = 127 * u / (1 - 127 * u)
    sum_error = (1 + u)**3 * (1 + gamma127) - 1
    t_error = max((1 + u) * math.sqrt(1 + sum_error) - 1,
                  1 - (1 - u) * math.sqrt(1 - sum_error))
    unit_error = (1 + u)**2 / (1 - t_error) - 1
    ideal = values / rms
    errors = unit_error * np.abs(ideal) + q * ((1 + u) * math.sqrt(128) / (1 - t_error) + 1)
    if codec in (4, 5, 6, 7):
        ideal *= np.array([1 - 2 * ((MASK >> (index // 2)) & 1) for index in range(128)])
    left = np.array([[1, -1, -1, -1], [1, 1, -1, 1], [1, 1, 1, -1], [1, -1, 1, 1]], dtype=np.float64) * .5
    gamma3 = 3 * u / (1 - 3 * u)
    for low, high in ((0, 1), (2, 3), (4, 5)):
        mask = (1 << low) | (1 << high)
        out, bound = np.empty(128), np.empty(128)
        for base in range(128):
            if base & mask:
                continue
            indices = [base, base | (1 << low), base | (1 << high), base | mask]
            absolute = .5 * math.fsum(abs(ideal[i]) for i in indices)
            propagated = .5 * math.fsum(errors[i] for i in indices)
            error = propagated + gamma3 * (absolute + propagated) + 4 * q / (1 - 3*u) + oracle * absolute
            out[indices] = left @ ideal[indices]
            bound[indices] = error * (1 + oracle)
        ideal, errors = out, bound
    if codec in (4, 6, 7):
        gamma2 = 2 * u / (1 - 2 * u)
        out, bound = np.empty(128), np.empty(128)
        for base in range(64):
            absolute = COS * (abs(ideal[base]) + abs(ideal[base + 64]))
            propagated = COS * (errors[base] + errors[base + 64])
            error = propagated + gamma2 * (absolute + propagated) + 4*q/(1 - 2*u) + oracle*absolute
            out[base] = COS * (ideal[base] - ideal[base + 64])
            out[base + 64] = COS * (ideal[base] + ideal[base + 64])
            bound[base] = bound[base + 64] = error * (1 + oracle)
        ideal, errors = out, bound
    if codec == 7:
        # Independently bound the actual alpha through max-interval monotonicity
        # then bound every possible signed quotient, including RN absolute
        # error for subnormal outputs. Corrected stored metadata is unrelated
        # to this original normalization denominator.
        endpoint = ref.CENTROIDS_6[-1]
        alpha = max(1., float(np.max(np.abs(ideal))) / endpoint)
        low_max = max(0., float(np.max(np.abs(ideal) - errors)))
        high_max = float(np.max(np.abs(ideal) + errors))
        low_alpha = max(1., low_max / endpoint * (1-u) - q)
        high_alpha = high_max / endpoint * (1+u) + q
        low_alpha = max(1., math.nextafter(low_alpha * (1-oracle), -math.inf))
        high_alpha = math.nextafter(max(1., high_alpha) * (1+oracle), math.inf)
        quotients = np.stack(((ideal-errors)/low_alpha, (ideal-errors)/high_alpha,
                              (ideal+errors)/low_alpha, (ideal+errors)/high_alpha))
        scaled = ideal / alpha
        magnitude = np.max(np.abs(quotients), axis=0)
        errors = np.max(np.abs(quotients-scaled), axis=0) + u*magnitude + q
        errors += oracle*(np.abs(scaled)+magnitude+1)
        errors = np.nextafter(errors*(1+oracle), math.inf)
        ideal = scaled
    lower = [bisect.bisect_left(thresholds, math.nextafter(x-e, -math.inf)) for x,e in zip(ideal, errors)]
    upper = [bisect.bisect_left(thresholds, math.nextafter(x+e, math.inf)) for x,e in zip(ideal, errors)]
    return lower, upper


def scale_interval(values, indices, centroids, corrected, *, max_scale_factor=8):
    """Conservative independent normal-domain FP32 metadata bound.

    Every legal reduction tree has <=127 additions. Gamma127 bounds its
    relative error; stable max normalization includes division, squaring,
    sqrt and reconstruction. Corrected centroid energy adds its own RN square/
    reduction/division/sqrt. Ten elementary roundings conservatively enclose
    the original stages; twenty enclose both stages. No measured tolerance is
    fitted. This bound does not cover underflow/overflow intermediates.
    """
    if len(values) != 128 or len(indices) != 128 or not all(math.isfinite(x) for x in values):
        raise ValueError("the scale bound requires one finite 128-coordinate group")
    rms = math.hypot(*values) / math.sqrt(128)
    if rms == 0:
        return (0., 0.)
    u = 2**-24
    gamma = 127 * u / (1 - 127 * u)
    error = 2 * gamma + 20 * u if corrected else gamma + 10 * u
    if corrected:
        energy = math.fsum(centroids[index] ** 2 for index in indices)
        rms *= math.sqrt(128 / energy)
    if not 2**-126 <= rms <= 2**30 * max_scale_factor:
        raise ValueError("the relative scale bound requires normal FP32 metadata in the production source domain")
    return rms * (1 - error), rms * (1 + error)


def certified_source(codec, centroids, thresholds):
    """Build BF16 inputs by inverse-transforming a fixed away-bin unit pattern."""
    # A fixed irregular dyadic sequence avoids exact zero cancellations in the
    # normalized forward image. Certification uses independent mathematical
    # distances, so no seed or production encoder is used to pick passing data.
    rotated = np.array([(.25, -.5, 1., -2.)[index % 4] for index in range(128)])
    rotated /= np.linalg.norm(rotated) / math.sqrt(128)
    matrix = wide_matrix(codec)
    # Correctly sum the independently defined inverse map. BLAS dot left
    # ~1e-19 cancellation residues here, whose squares are outside the normal
    # arithmetic proof domain; mathematically exact cancellation is zero.
    source = [math.fsum(rotated[j] * matrix[j, i] for j in range(128)) for i in range(128)]
    # Convert to BF16 via bit rounding without Torch/CUDA runtime state.
    def bf16(value):
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
        return struct.unpack("<f", struct.pack("<I", bits))[0]
    source = np.array([bf16(value) for value in source])
    unit = source / (np.linalg.norm(source) / math.sqrt(128))
    actual = unit @ wide_matrix(codec).T
    margins = np.min(np.abs(actual[:, None] - np.asarray(thresholds)[None, :]), axis=1)
    assert margins.min() > 1e-3
    indices = [bisect.bisect_left(thresholds, value) for value in actual]
    return source, indices


@pytest.mark.parametrize("codec,variant,corrected,signed,wide", CASES)
def test_independent_wide_stages_preserve_energy_and_have_true_reverse_inverse(codec, variant, corrected, signed, wide):
    del variant, corrected
    mathematical = wide_matrix(codec, mathematical=True)
    stored = wide_matrix(codec)
    assert np.max(np.abs(mathematical @ mathematical.T - np.eye(128))) < 3e-15
    assert np.linalg.slogdet(mathematical)[0] == 1
    bound = 4e-8 if wide else 0.
    assert np.max(np.abs(stored @ stored.T - np.eye(128))) <= bound
    x = np.arange(128) / 31 - 2
    y = np.sin(np.arange(128) / 7)
    assert np.allclose((x @ stored.T) @ stored, x, atol=1e-7 if wide else 4e-14, rtol=0)
    assert math.isclose(float((x @ stored.T) @ (y @ stored.T)), float(x @ y), rel_tol=4e-8, abs_tol=1e-12)
    assert not np.array_equal(stored, stored.T)
    if not wide:
        # Actual support width is64; each output mixes every source in its block.
        assert np.all(np.count_nonzero(stored, axis=1) == 64)
        assert not np.any(stored[:64, 64:]) and not np.any(stored[64:, :64])
    else:
        assert np.all(np.count_nonzero(stored, axis=1) == 128)
    if signed:
        signs = [1 - 2 * ((MASK >> (index // 2)) & 1) for index in range(128)]
        assert math.prod(signs) == 1 and all(signs[i] == signs[i + 1] for i in range(0, 128, 2))


@pytest.mark.parametrize("codec,variant,corrected,signed,wide", CASES)
def test_pinned_wide_reference_identity_and_dense_transform(codec, variant, corrected, signed, wide):
    del signed, wide
    assert ref.variant_id(variant) == codec
    assert ref.norm_corrected(variant) is corrected
    assert ref.SIGN_PAIR_NEGATIVE_MASK == MASK
    assert ref.WIDE_GIVENS_COS == ref.WIDE_GIVENS_SIN == COS
    descriptor = ref.descriptor(4, variant)
    assert descriptor.norm_policy == ("reconstruction-norm" if corrected else "original-rms")
    assert descriptor.vector_bytes_per_group == 68
    with pytest.raises(ValueError, match="four-bit"):
        ref.descriptor(3, variant)
    source = np.sin(np.arange(256) / 13)
    expected = source.reshape(2, 128) @ wide_matrix(codec).T
    actual = np.asarray(ref.rotate_oracle(source.tolist(), variant)).reshape(2, 128)
    assert np.allclose(actual, expected, atol=2e-15, rtol=0)
    inverse = np.asarray(ref.rotate_oracle(actual.flatten().tolist(), variant, inverse=True)).reshape(2, 128)
    assert np.allclose(inverse, expected @ wide_matrix(codec), atol=2e-15, rtol=0)


@pytest.mark.parametrize("codec,variant,corrected,signed,wide", CASES)
def test_wide_certified_inverse_fixture_has_exact_independent_payload_and_valid_scale(codec, variant, corrected, signed, wide):
    del signed, wide
    torch = pytest.importorskip("torch")
    source, expected = certified_source(codec, ref.CENTROIDS_4, ref.THRESHOLDS_4)
    lower, upper = wide_index_envelope(source, codec, ref.THRESHOLDS_4)
    assert lower == upper == expected
    canonical_lower, canonical_upper = ref.index_envelope_oracle(source.tolist(), 4, variant)
    assert canonical_lower == canonical_upper == expected
    x = torch.tensor(source, dtype=torch.bfloat16).repeat(2).reshape(1, 256)
    packed, scales = ref.quantize_ref(x, 4, variant)
    actual = ref.unpack_indices_ref(bytes(packed.flatten().tolist()), 4)
    assert actual == expected * 2
    for group, scale in enumerate(scales.flatten().tolist()):
        start = group * 128
        low, high = scale_interval(x.flatten().float().tolist()[start:start+128], actual[start:start+128],
                                   ref.CENTROIDS_4, corrected)
        assert low <= scale <= high
    _, _, ideal, errors = ref.rounding_envelope_oracle(x.flatten().float().tolist(), variant,
                                                     rms=scales.flatten().tolist())
    assert all(abs(got-want) <= error for got,want,error in zip(scales.flatten().tolist(),ideal,errors))
    with pytest.raises(ValueError, match="derived wide"):
        ref.rounding_envelope_oracle(x.flatten().float().tolist(), variant,
                                    rms=[scale * 2 for scale in scales.flatten().tolist()])


@pytest.mark.parametrize("codec,variant,corrected,signed,wide", CASES)
@pytest.mark.parametrize("seed", [37, 192837])
def test_wide_random_bf16_indices_and_metadata_satisfy_independent_analytic_bounds(codec, variant, corrected, signed, wide, seed):
    del signed, wide
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn((3, 256), generator=generator).to(torch.bfloat16)
    packed, scales = ref.quantize_ref(x, 4, variant)
    for row, payload, metadata in zip(x.float().tolist(), packed.tolist(), scales.tolist()):
        actual = ref.unpack_indices_ref(bytes(payload), 4)
        canonical_low, canonical_high = ref.index_envelope_oracle(row, 4, variant, rms=metadata)
        for group in range(2):
            begin = group * 128
            lower, upper = wide_index_envelope(row[begin:begin+128], codec, ref.THRESHOLDS_4)
            assert lower == canonical_low[begin:begin+128] and upper == canonical_high[begin:begin+128]
            assert all(low <= index <= high and high-low <= 1 for low,index,high in
                       zip(lower,actual[begin:begin+128],upper))
            low, high = scale_interval(row[begin:begin+128], actual[begin:begin+128], ref.CENTROIDS_4, corrected)
            assert low <= metadata[group] <= high
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("codec,variant,corrected,signed,wide", CASES)
def test_wide_normal_domain_bound_rejects_unproved_subnormal_and_three_bit_inputs(codec, variant, corrected, signed, wide):
    del codec, corrected, signed, wide
    with pytest.raises(ValueError, match="normal RMS"):
        ref.index_envelope_oracle([2**-140] * 128, 4, variant)
    with pytest.raises(ValueError, match="four-bit"):
        ref.index_envelope_oracle([1.] * 128, 3, variant)


@pytest.mark.parametrize("codec,variant,corrected,signed,wide", CASES)
def test_wide_zero_group_is_exact_and_quantizer_rejects_three_bit_mode(codec, variant, corrected, signed, wide):
    del codec, corrected, signed, wide
    torch = pytest.importorskip("torch")
    source = torch.zeros((3,256),dtype=torch.bfloat16)
    packed,metadata = ref.quantize_ref(source,4,variant)
    assert packed.flatten().tolist() == [0x77] * packed.numel()
    assert metadata.flatten().tolist() == [0.] * 6
    lower,upper = ref.index_envelope_oracle([0.] * 256,4,variant,rms=[0.,0.])
    assert lower == upper == [7] * 256
    assert ref.dequant_ref(packed,metadata,4,variant,inverse=True).eq(0).all()
    with pytest.raises(ValueError,match="four-bit"):
        ref.quantize_ref(source,3,variant)
    with pytest.raises(ValueError,match="zero scale"):
        ref.rounding_envelope_oracle([0.] * 128,variant,rms=[1.])
