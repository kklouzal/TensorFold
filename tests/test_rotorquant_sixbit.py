"""Independent analytic normal64 solver and bit-stream oracle for six-bit KV."""

import bisect
import dataclasses
import math
import random
import struct

import pytest

from tensorfold.families.qwen4_exp.cuda import rotorquant_ref as rq


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def phi(value):
    return 0.0 if math.isinf(value) else math.exp(-value * value / 2) / math.sqrt(2 * math.pi)


def mass(lower, upper):
    # Both bounds are nonnegative: use survival-function subtraction to avoid
    # losing tail precision through1-CDF cancellation.
    return (math.erfc(lower / math.sqrt(2)) - math.erfc(upper / math.sqrt(2))) / 2


def stationary_system(centers):
    edges = [0.0, *((a + b) / 2 for a, b in zip(centers, centers[1:])), math.inf]
    residual, lower, diagonal, upper = [], [], [], []
    for i, (c, a, b) in enumerate(zip(centers, edges, edges[1:])):
        probability = mass(a, b)
        average = (phi(a) - phi(b)) / probability
        derivative_a = phi(a) * (average - a) / probability if i else 0.0
        derivative_b = phi(b) * (b - average) / probability if math.isfinite(b) else 0.0
        residual.append(c - average)
        lower.append(-derivative_a / 2)
        upper.append(-derivative_b / 2)
        diagonal.append(1 - (derivative_a + derivative_b) / 2)
    return residual, lower, diagonal, upper


def solve_normal64():
    """Damped Newton/tridiagonal solve, independent of the Lloyd iteration generator."""

    centers = [(i + .5) / 8 for i in range(32)]  # Uniform positive half; no source-table initialization.
    for iteration in range(100):
        residual, lower, diagonal, upper = stationary_system(centers)
        error = max(map(abs, residual))
        if error < 2e-14:
            return centers, iteration, error
        rhs = [-value for value in residual]
        for i in range(1, 32):
            factor = lower[i] / diagonal[i - 1]
            diagonal[i] -= factor * upper[i - 1]
            rhs[i] -= factor * rhs[i - 1]
        step = [0.0] * 32
        step[-1] = rhs[-1] / diagonal[-1]
        for i in range(30, -1, -1):
            step[i] = (rhs[i] - upper[i] * step[i + 1]) / diagonal[i]
        damping = 1.0
        for _ in range(40):
            proposed = [c + damping * d for c, d in zip(centers, step)]
            if proposed[0] > 0 and all(a < b for a, b in zip(proposed, proposed[1:])) and \
                    max(map(abs, stationary_system(proposed)[0])) < error:
                centers = proposed
                break
            damping /= 2
        else:
            raise AssertionError("independent Newton solve failed to descend")
    raise AssertionError("independent Newton solve failed to converge")


def independent_payload(codes):
    """Flatten individual little-endian bits into each group-local plane."""

    output = bytearray()
    for start in range(0, len(codes), 128):
        group = codes[start:start + 128]
        assert len(group) == 128
        bitstream = [code >> bit & 1 for code in group for bit in range(4)]
        bitstream += [code >> bit & 1 for code in group for bit in (4, 5)]
        output.extend(sum(bitstream[i + b] << b for b in range(8)) for i in range(0, 768, 8))
    return bytes(output)


def test_analytic_newton_solution_reproduces_every_pinned_centroid_bit():
    positive, iterations, error = solve_normal64()
    book = tuple(-c for c in reversed(positive)) + tuple(positive)
    assert iterations < 30 and error < 2e-14
    assert tuple(map(f32, book)) == rq.CENTROIDS_6
    assert rq.centroids(6) is rq.CENTROIDS_6
    assert len(book) == 64 and rq.CENTROIDS_6 == tuple(-c for c in reversed(rq.CENTROIDS_6))


def test_pinned_midpoint_bits_stationarity_and_independent_mse():
    book = rq.CENTROIDS_6
    mids = tuple((a + b) / 2 for a, b in zip(book, book[1:]))
    assert tuple(map(f32, mids)) == rq.THRESHOLDS_6
    assert tuple(map(f32, rq.THRESHOLDS_6)) == rq.THRESHOLDS_6
    assert rq.thresholds(6) is rq.THRESHOLDS_6
    positive = list(book[32:])
    residual, *_ = stationary_system(positive)
    # Two adjacent rounded centroids move a cell boundary by at most one
    # centroid ULP; the pinned book retains the analytic stationarity solution.
    assert max(map(abs, residual)) < 2.5e-7
    edges = [0.0, *((a + b) / 2 for a, b in zip(positive, positive[1:])), math.inf]
    terms = []
    for c, a, b in zip(positive, edges, edges[1:]):
        probability = mass(a, b)
        first = phi(a) - phi(b)
        second = probability + a * phi(a) - (b * phi(b) if math.isfinite(b) else 0.0)
        terms.append(second - 2 * c * first + c * c * probability)
    mse = 2 * math.fsum(terms)
    assert .00064 < mse < .00066
    assert mse < .00950100800819157 / 14


def test_each_pinned_threshold_has_lower_tie_and_exact_fp32_neighbors():
    thresholds = rq.THRESHOLDS_6
    for index, value in enumerate(thresholds):
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        if value == 0:
            below, above = -2**-149, 2**-149
        else:
            up_bits = bits + (1 if value > 0 else -1)
            down_bits = bits - (1 if value > 0 else -1)
            below = struct.unpack("<f", struct.pack("<I", down_bits))[0]
            above = struct.unpack("<f", struct.pack("<I", up_bits))[0]
        assert bisect.bisect_left(thresholds, value) == index
        assert below < value < above and f32(below) == below and f32(above) == above
        assert bisect.bisect_left(thresholds, below) == index
        assert bisect.bisect_left(thresholds, above) == index + 1
    assert thresholds[31] == 0.0 and bisect.bisect_left(thresholds, -0.0) == 31


def test_every_code_at_every_plane_component_and_group_boundary_has_exact_bytes():
    for coordinate in range(128):
        for code in range(64):
            codes = [0] * 128
            codes[coordinate] = code
            actual = rq.pack_indices_ref(codes, 6)
            assert actual == independent_payload(codes)
            assert rq.unpack_indices_ref(actual, 6) == codes


def test_known_planes_and_cross_group_order():
    codes = list(range(64)) * 2 + [63] * 128 + list(reversed(range(64))) * 2
    packed = rq.pack_indices_ref(codes, 6)
    first = bytes.fromhex("1032547698badcfe") * 8
    first += (bytes([0] * 4 + [0x55] * 4 + [0xAA] * 4 + [0xFF] * 4)) * 2
    assert packed[:96] == first and packed[96:192] == bytes([255]) * 96
    assert packed == independent_payload(codes)
    assert rq.unpack_indices_ref(memoryview(packed), 6) == codes


def test_sixbit_descriptor_identity_and_exact_storage_are_immutable():
    fmt = rq.descriptor(6, "signed-iso128")
    assert fmt.bits == 6 and fmt.group == 128 and fmt.group_bytes == 96
    assert fmt.vector_bytes_per_group == 100 and fmt.norm_policy == "original-rms"
    assert fmt.scale_dtype == "float32" and fmt.version == 1
    assert fmt.codec_id != rq.descriptor(4, "signed-iso128").codec_id
    with pytest.raises(dataclasses.FrozenInstanceError):
        fmt.bits = 4
    for variant in ("iso64-norm", "signed-iso64-norm"):
        with pytest.raises(ValueError, match="six-bit signed-iso128"):
            rq.descriptor(6, variant)


@pytest.mark.parametrize("codes", [[-1] * 128, [64] * 128, [True] * 128, [0] * 127])
def test_sixbit_invalid_codes_and_partial_groups_fail(codes):
    with pytest.raises(ValueError):
        rq.pack_indices_ref(codes, 6)
    with pytest.raises(ValueError, match="complete"):
        rq.unpack_indices_ref(bytes(95), 6)
    assert rq.pack_indices_ref([], 6) == b"" and rq.unpack_indices_ref(b"", 6) == []


@pytest.mark.torch
def test_tensor_packing_matches_individual_bit_oracle_and_rejects_payload_tails():
    import torch

    generator = random.Random(601)
    codes = [generator.randrange(64) for _ in range(3 * 2 * 256)]
    tensor = torch.tensor(codes, dtype=torch.int32).reshape(3, 2, 256)
    packed = rq.pack_ref(tensor, 6)
    assert packed.shape == (3, 2, 192) and packed.dtype == torch.uint8
    assert packed.reshape(-1).numpy().tobytes() == independent_payload(codes)
    assert torch.equal(rq.unpack_ref(packed, 6), tensor)
    with pytest.raises(ValueError, match="complete"):
        rq.unpack_ref(torch.zeros(95, dtype=torch.uint8), 6)


@pytest.mark.torch
def test_sixbit_zero_group_uses_code31_scale0_and_decodes_finite_zero():
    import torch

    source = torch.zeros((2, 256), dtype=torch.bfloat16)
    packed, scales = rq.quantize_ref(source, 6, "signed-iso128")
    expected = (bytes([255]) * 64 + bytes([0x55]) * 32) * 4
    assert packed.reshape(-1).numpy().tobytes() == expected
    assert rq.unpack_ref(packed, 6).unique().tolist() == [31]
    assert scales.dtype == torch.float32 and scales.shape == (2, 2) and not bool(scales.any())
    decoded = rq.dequant_ref(packed, scales, 6, "signed-iso128", inverse=True)
    assert decoded.dtype == torch.float32 and decoded.shape == source.shape
    assert bool(torch.isfinite(decoded).all()) and decoded.count_nonzero() == 0


@pytest.mark.torch
def test_original_rms_and_signed128_codes_obey_independent_normalization_envelope():
    import torch

    source = torch.randn((2, 256), generator=torch.Generator().manual_seed(613)).to(torch.bfloat16)
    packed, scales = rq.quantize_ref(source, 6, "signed-iso128")
    values = source.float().reshape(-1).tolist()
    lower, upper = rq.index_envelope_oracle(values, 6, "signed-iso128", rms=scales.reshape(-1).tolist())
    actual = rq.unpack_ref(packed, 6).reshape(-1).tolist()
    assert all(lo <= got <= hi for lo, got, hi in zip(lower, actual, upper))
    assert all(a == b for a, b in zip(lower, upper)), "fixture must have certified singleton bins"
    ideal_rms = [math.sqrt(math.fsum(x * x for x in values[i:i + 128]) / 128)
                 for i in range(0, len(values), 128)]
    for actual_scale, exact in zip(scales.reshape(-1).tolist(), ideal_rms):
        assert abs(actual_scale - exact) <= 3 * 2**-24 * exact
