"""Independent normal128/256 stationarity and exact seven/eight-bit wire oracles."""

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


def independent_system(centers):
    # Deliberately use direct density subtraction and a different initialization
    # from the artifact generator's expm1/near-zero erf implementation.
    edges = [0.0, *((a + b) / 2 for a, b in zip(centers, centers[1:])), math.inf]
    residual, lower, diagonal, upper = [], [], [], []
    for i, (center, a, b) in enumerate(zip(centers, edges, edges[1:])):
        probability = (math.erfc(a / math.sqrt(2)) - math.erfc(b / math.sqrt(2))) / 2
        average = (phi(a) - phi(b)) / probability
        derivative_a = phi(a) * (average - a) / probability if i else 0.0
        derivative_b = phi(b) * (b - average) / probability if math.isfinite(b) else 0.0
        residual.append(center - average)
        lower.append(-derivative_a / 2)
        upper.append(-derivative_b / 2)
        diagonal.append(1 - (derivative_a + derivative_b) / 2)
    return residual, lower, diagonal, upper


def independent_newton(bits):
    count = 2**bits // 2
    centers = [(i + .5) * 5 / count for i in range(count)]
    for iteration in range(100):
        residual, lower, diagonal, upper = independent_system(centers)
        error = max(map(abs, residual))
        if error < 1e-12:
            return centers, iteration, error
        rhs = [-value for value in residual]
        for i in range(1, count):
            factor = lower[i] / diagonal[i - 1]
            diagonal[i] -= factor * upper[i - 1]
            rhs[i] -= factor * rhs[i - 1]
        step = [0.0] * count
        step[-1] = rhs[-1] / diagonal[-1]
        for i in range(count - 2, -1, -1):
            step[i] = (rhs[i] - upper[i] * step[i + 1]) / diagonal[i]
        damping = 1.0
        for _ in range(40):
            proposed = [center + damping * delta for center, delta in zip(centers, step)]
            if proposed[0] > 0 and all(a < b for a, b in zip(proposed, proposed[1:])) and \
                    max(map(abs, independent_system(proposed)[0])) < error:
                centers = proposed
                break
            damping /= 2
        else:
            raise AssertionError("independent high-bit Newton line search failed")
    raise AssertionError("independent high-bit Newton solve did not converge")


def independent_payload(codes, bits):
    output = bytearray()
    planes = ((0, 1, 2, 3), (4, 5), (6,)) if bits == 7 else (tuple(range(8)),)
    for start in range(0, len(codes), 128):
        group = codes[start:start + 128]
        assert len(group) == 128
        stream = [code >> bit & 1 for plane in planes for code in group for bit in plane]
        output.extend(sum(stream[i + b] << b for b in range(8)) for i in range(0, len(stream), 8))
    return bytes(output)


@pytest.mark.parametrize("bits", [7, 8])
def test_independent_newton_reproduces_every_pinned_fp32_centroid(bits):
    positive, iterations, residual = independent_newton(bits)
    exact = tuple(-center for center in reversed(positive)) + tuple(positive)
    pinned = getattr(rq, f"CENTROIDS_{bits}")
    assert iterations < 30 and residual < 1e-12
    assert tuple(map(f32, exact)) == pinned
    assert rq.centroids(bits) is pinned and len(pinned) == 2**bits
    assert pinned == tuple(-value for value in reversed(pinned))


@pytest.mark.parametrize("bits,bounds", [(7, (.00016, .00017)), (8, (.000040, .000042))])
def test_pinned_stationarity_midpoints_and_analytic_mse(bits, bounds):
    book, thresholds = rq.centroids(bits), rq.thresholds(bits)
    assert thresholds == tuple(map(f32, ((a + b) / 2 for a, b in zip(book, book[1:]))))
    positive = book[len(book) // 2:]
    residual, *_ = independent_system(positive)
    assert max(map(abs, residual)) < 5e-7
    edges = [0.0, *((a + b) / 2 for a, b in zip(positive, positive[1:])), math.inf]
    terms = []
    for center, a, b in zip(positive, edges, edges[1:]):
        probability = (math.erfc(a / math.sqrt(2)) - math.erfc(b / math.sqrt(2))) / 2
        first = phi(a) - phi(b)
        second = probability + a * phi(a) - (b * phi(b) if math.isfinite(b) else 0)
        terms.append(second - 2 * center * first + center * center * probability)
    mse = 2 * math.fsum(terms)
    assert bounds[0] < mse < bounds[1]
    assert mse < .00065 / (3.8 if bits == 7 else 15)


@pytest.mark.parametrize("bits", [7, 8])
def test_all_threshold_ties_and_actual_adjacent_fp32_values(bits):
    thresholds = rq.thresholds(bits)
    for index, value in enumerate(thresholds):
        raw = struct.unpack("<I", struct.pack("<f", value))[0]
        if value == 0:
            below, above = -2**-149, 2**-149
        else:
            below = struct.unpack("<f", struct.pack("<I", raw - (1 if value > 0 else -1)))[0]
            above = struct.unpack("<f", struct.pack("<I", raw + (1 if value > 0 else -1)))[0]
        assert below < value < above
        assert bisect.bisect_left(thresholds, below) == index
        assert bisect.bisect_left(thresholds, value) == index
        assert bisect.bisect_left(thresholds, above) == index + 1
    assert thresholds[2**(bits - 1) - 1] == 0
    assert bisect.bisect_left(thresholds, -0.0) == 2**(bits - 1) - 1


@pytest.mark.parametrize("bits", [7, 8])
def test_every_code_at_every_coordinate_has_exact_group_local_bytes(bits):
    for coordinate in range(128):
        for code in range(2**bits):
            values = [0] * 128
            values[coordinate] = code
            actual = rq.pack_indices_ref(values, bits)
            assert actual == independent_payload(values, bits)
            assert rq.unpack_indices_ref(actual, bits) == values


def test_known_seven_bit_planes_and_cross_group_order():
    values = list(range(128)) + [127] * 128 + list(reversed(range(128)))
    expected = bytes.fromhex("1032547698badcfe") * 8
    expected += bytes([0] * 4 + [0x55] * 4 + [0xAA] * 4 + [0xFF] * 4) * 2
    expected += bytes([0] * 8 + [0xFF] * 8)
    actual = rq.pack_indices_ref(values, 7)
    assert actual[:112] == expected and actual[112:224] == bytes([255]) * 112
    assert actual == independent_payload(values, 7)
    assert rq.unpack_indices_ref(memoryview(actual), 7) == values


def test_eight_bit_raw_index_bytes_preserve_every_value_across_groups():
    values = list(range(256)) + list(reversed(range(256)))
    actual = rq.pack_indices_ref(values, 8)
    assert actual == bytes(values) == independent_payload(values, 8)
    assert rq.unpack_indices_ref(memoryview(actual), 8) == values


@pytest.mark.parametrize("bits", [7, 8])
def test_original_rms_descriptor_identity_and_immutable_exact_storage(bits):
    fmt = rq.descriptor(bits, "signed-iso128")
    assert fmt.bits == bits and fmt.group == 128 and fmt.group_bytes == bits * 16
    assert fmt.vector_bytes_per_group == bits * 16 + 4 and fmt.norm_policy == "original-rms"
    assert fmt.scale_dtype == "float32" and fmt.version == 1
    assert fmt.codec_id != rq.descriptor(6, "signed-iso128").codec_id
    with pytest.raises(dataclasses.FrozenInstanceError):
        fmt.bits = 6


@pytest.mark.parametrize("bits", [7, 8])
@pytest.mark.parametrize("kind", ["negative", "overflow", "bool", "partial"])
def test_invalid_codes_and_partial_payloads_fail(bits, kind):
    values = {"negative": [-1] * 128, "overflow": [2**bits] * 128,
              "bool": [True] * 128, "partial": [0] * 127}[kind]
    with pytest.raises(ValueError):
        rq.pack_indices_ref(values, bits)
    with pytest.raises(ValueError, match="complete"):
        rq.unpack_indices_ref(bytes(bits * 16 - 1), bits)
    assert rq.pack_indices_ref([], bits) == b"" and rq.unpack_indices_ref(b"", bits) == []


@pytest.mark.torch
@pytest.mark.parametrize("bits", [7, 8])
def test_tensor_pack_bytes_match_individual_bit_oracle(bits):
    import torch

    generator = random.Random(809 + bits)
    values = [generator.randrange(2**bits) for _ in range(3 * 2 * 256)]
    tensor = torch.tensor(values, dtype=torch.int32).reshape(3, 2, 256)
    packed = rq.pack_ref(tensor, bits)
    assert packed.shape == (3, 2, bits * 32) and packed.dtype == torch.uint8
    assert packed.reshape(-1).numpy().tobytes() == independent_payload(values, bits)
    assert torch.equal(rq.unpack_ref(packed, bits), tensor)
    with pytest.raises(ValueError, match="complete"):
        rq.unpack_ref(torch.zeros(bits * 16 - 1, dtype=torch.uint8), bits)


@pytest.mark.torch
@pytest.mark.parametrize("bits", [7, 8])
def test_zero_groups_have_lower_central_codes_zero_original_rms_and_finite_decode(bits):
    import torch

    source = torch.zeros((2, 256), dtype=torch.bfloat16)
    packed, scales = rq.quantize_ref(source, bits, "signed-iso128")
    center = 2**(bits - 1) - 1
    assert packed.reshape(-1).numpy().tobytes() == independent_payload([center] * 512, bits)
    assert rq.unpack_ref(packed, bits).unique().tolist() == [center]
    assert scales.dtype == torch.float32 and scales.shape == (2, 2) and not bool(scales.any())
    decoded = rq.dequant_ref(packed, scales, bits, "signed-iso128", inverse=True)
    assert decoded.shape == source.shape and decoded.dtype == torch.float32
    assert bool(torch.isfinite(decoded).all()) and decoded.count_nonzero() == 0


@pytest.mark.torch
def test_highbit_original_rms_and_fp32_index_envelopes_retain_the_contract():
    import torch

    source = torch.randn((2, 256), generator=torch.Generator().manual_seed(811)).to(torch.bfloat16)
    values = source.float().reshape(-1).tolist()
    prior_scales = None
    for bits in (6, 7, 8):
        packed, scales = rq.quantize_ref(source, bits, "signed-iso128")
        lower, upper = rq.index_envelope_oracle(values, bits, "signed-iso128", rms=scales.reshape(-1).tolist())
        actual = rq.unpack_ref(packed, bits).reshape(-1).tolist()
        assert all(lo <= code <= hi for lo, code, hi in zip(lower, actual, upper))
        if prior_scales is not None:
            assert torch.equal(scales, prior_scales)
        prior_scales = scales
        for i, actual_scale in enumerate(scales.reshape(-1).tolist()):
            exact = math.sqrt(math.fsum(x * x for x in values[i * 128:(i + 1) * 128]) / 128)
            assert abs(actual_scale - exact) <= 3 * 2**-24 * exact
