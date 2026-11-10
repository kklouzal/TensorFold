"""Independent GGML coefficient oracle for the CPU-only owned plane producer."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from tensorfold.cuda.sm70 import onecat_moe_packing as packing
from tensorfold.families.qwen4_exp import gguf_iq
from tensorfold.families.qwen4_exp.gguf_tables import IQ2S_GRID, IQ3S_GRID, IQ3XXS_GRID, KVALUES_IQ4NL


def raw_rows(kind, rows=160, blocks=10):
    with np.load(Path(__file__).parent / "fixtures/gguf_mixed_ggml.npz", allow_pickle=False) as archive:
        fixture = archive[f"raw_{kind}"]
    raw = np.tile(fixture, (rows // 8, 2))[:, :blocks * packing.BLOCK_BYTES[kind]].copy()
    # Power-of-two d isolates representation correctness from intentional
    # pinned Half-intermediate scale differences. Independent GGML decoder
    # remains the oracle, not the producer's own inverse or a fitted margin.
    raw.reshape(rows, blocks, -1)[:, :, :2] = np.array([2. ** -7], dtype="<f2").view(np.uint8)
    return raw


def decode_plane_fields(codes, scales, kind):
    """Separate lane/word inverse using documented logical weight positions."""
    # Fixed independently read lane order from the pinned MMA operand map;
    # deriving this from the producer would conceal a shared permutation bug.
    wire_rows = [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27,
                 4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31]
    np.testing.assert_array_equal(packing.ROWMAP, wire_rows)
    inverse = np.argsort(wire_rows)
    components = 4 if kind == 23 else 2 if kind == 22 else 3
    wire = codes.view("<u4").reshape(5, 20, components, 32, 4)
    words = wire.transpose(0, 3, 1, 2, 4)[:, inverse].reshape(160, 20, components, 4)
    k = np.arange(32)
    if kind == 23:
        quant = ((words[..., None] >> (4 * np.arange(8))) & 15).reshape(160, 20, 4, 32)
        meta = scales.view("<f2").reshape(5, 20, 32, 4)
        d = meta.transpose(0, 2, 1, 3)[:, inverse].reshape(160, 20, 4).astype(np.float32)
        return (d[..., None] * np.asarray(KVALUES_IQ4NL, np.float32)[quant]).reshape(160, 2560)
    if kind == 22:
        meta = scales.view("<u4").reshape(5, 20, 32, 4).transpose(0, 2, 1, 3)[:, inverse].reshape(160, 20, 4)
        low = words[:, :, 0].copy().view(np.uint8).reshape(160, 20, 4, 4)
        high = (meta[:, :, 0, None] >> (8 * np.arange(4))) & 255
        index = low.astype(np.uint16) | (((high[..., None] >> (2 * np.arange(4))) & 3) << 8)
        grid = np.asarray(IQ2S_GRID, np.uint64)[index[..., k // 8]]
        magnitude = ((grid >> (8 * (k % 8)).astype(np.uint64)) & 255).astype(np.float32)
        sign = (words[:, :, 1, :, None] >> k) & 1
        nibble = (meta[:, :, 2, None, None] >> (8 * np.arange(4)[None, :, None] + 4 * (k // 16))) & 15
        d = (meta[:, :, 1] & 65535).astype("<u2").view("<f2").astype(np.float32)
        factor = (d[:, :, None, None] * (.5 + nibble.astype(np.float32))) * .25
    else:
        size = 2 if kind == 21 else 1
        meta = scales.view("<u4").reshape(5, 20, 32, size).transpose(0, 2, 1, 3)[:, inverse].reshape(160, 20, size)
        dword = meta[:, :, -1]
        index = words[:, :, :2].copy().reshape(160, 20, 8).view(np.uint8).reshape(160, 20, 4, 8).astype(np.uint16)
        if kind == 21:
            high = (meta[:, :, 0, None] >> (8 * np.arange(4))) & 255
            index |= (((high[..., None] >> np.arange(8)) & 1) << 8).astype(np.uint16)
        grid = np.asarray(IQ3S_GRID if kind == 21 else IQ3XXS_GRID, np.uint32)[index[..., k // 4]]
        magnitude = ((grid >> (8 * (k % 4))) & 255).astype(np.float32)
        sign = (words[:, :, 2, :, None] >> (k // 2 + 16 * (k % 2))) & 1
        nibble = (dword[:, :, None] >> (16 + 4 * np.arange(4))) & 15
        d = (dword & 65535).astype("<u2").view("<f2").astype(np.float32)
        factor = d[:, :, None] * (1 + 2 * nibble.astype(np.float32))
        if kind == 18:
            factor *= .25
        factor = factor[..., None]
    return (factor * magnitude * np.where(sign, np.float32(-1), np.float32(1))).reshape(160, 2560)


@pytest.mark.parametrize("kind", [18, 21, 22, 23])
def test_all_four_planes_match_independent_original_ggml_fields(kind):
    raw = raw_rows(kind)
    before = raw.copy()
    fmt, codes, scales = packing._pack_expert(raw, kind)
    spec = packing.requirements(kind)
    assert codes.shape == (spec["codes_shape"][1],)
    assert scales.shape == (spec["scales_shape"][1],)
    assert fmt == {18: 6, 21: 5, 22: 7, 23: 3}[kind]
    observed = decode_plane_fields(codes, scales, kind)
    original = gguf_iq.decode(raw, kind, 2560, packing.BLOCK_BYTES[kind])
    np.testing.assert_array_equal(observed.view(np.uint32), original.view(np.uint32))
    np.testing.assert_array_equal(raw, before)
    assert packing.table(kind).nbytes == {18: 3072, 21: 3072, 22: 8192, 23: 0}[kind]
    # Independent standard-library wire encoding checks the little-endian
    # lookup ABI against the existing canonical codebook integers.
    if kind == 22:
        expected_table = b"".join(int(value).to_bytes(8, "little") for value in IQ2S_GRID)
    elif kind == 23:
        expected_table = b""
    else:
        expected_table = b"".join(int(value).to_bytes(4, "little") for value in (*IQ3S_GRID, *IQ3XXS_GRID))
    assert packing.table(kind).tobytes() == expected_table


def test_full_e256_completion_with_strided_rows_nonzero_source_offset_and_lane():
    kind = 18
    source = raw_rows(kind, 640, 11)
    # Independent row-stride padding and a readonly broadcast of identical
    # experts avoid allocating a fake model. All256 pack calls still execute.
    padded = np.zeros((1280, source.shape[1] + 19), np.uint8)
    padded[::2, :source.shape[1]] = source
    raw = np.broadcast_to(padded[::2, :source.shape[1]], (512, *source.shape))
    spec = packing.requirements(kind)
    codes = np.empty(spec["codes_shape"], np.uint8)
    scales = np.empty(spec["scales_shape"], np.uint8)
    receipt = packing.pack_owned_into(raw, codes, scales, first=256, lane=3,
                                      ggml_type=kind, column_offset=256)
    chosen = source[480:640, 98:11 * 98].copy()
    _, expected_codes, expected_scales = packing._pack_expert(chosen, kind)
    np.testing.assert_array_equal(codes, np.broadcast_to(expected_codes, codes.shape))
    np.testing.assert_array_equal(scales, np.broadcast_to(expected_scales, scales.shape))
    digest = hashlib.sha256()
    for _ in range(256):
        digest.update(memoryview(chosen))
    assert receipt["source_payload_sha256"] == digest.hexdigest()
    assert receipt["completed_experts"] == 256
    assert receipt["first"] == 256 and receipt["lane"] == 3
    np.testing.assert_array_equal(raw[256], source)


@pytest.mark.parametrize("change", ["first", "lane", "offset", "kind", "codes", "scales", "alias", "readonly"])
def test_refusals_precede_any_output_write(change):
    kind = 18
    source = raw_rows(kind, 640)
    raw = np.broadcast_to(source, (512, *source.shape))
    spec = packing.requirements(kind)
    codes = np.full(spec["codes_shape"], 19, np.uint8)
    scales = np.full(spec["scales_shape"], 23, np.uint8)
    kwargs = dict(first=0, lane=0, ggml_type=kind)
    if change == "first":
        kwargs["first"] = True
    elif change == "lane":
        kwargs["lane"] = 4
    elif change == "offset":
        kwargs["column_offset"] = 128
    elif change == "kind":
        kwargs["ggml_type"] = 42
    elif change == "codes":
        codes = codes[:-1]
    elif change == "scales":
        scales = scales[:, :-1]
    elif change == "alias":
        scales = codes.reshape(-1)[:scales.size].reshape(scales.shape)
    else:
        codes.flags.writeable = False
    before_codes, before_scales = codes.copy(), scales.copy()
    with pytest.raises((ValueError, TypeError)):
        packing.pack_owned_into(raw, codes, scales, **kwargs)
    np.testing.assert_array_equal(codes, before_codes)
    np.testing.assert_array_equal(scales, before_scales)


def test_mid_bank_exhaustion_preserves_primary_and_source(monkeypatch):
    source = raw_rows(18, 640)
    before = source.copy()
    raw = np.broadcast_to(source, (512, *source.shape))
    spec = packing.requirements(18)
    codes = np.full(spec["codes_shape"], 19, np.uint8)
    scales = np.full(spec["scales_shape"], 23, np.uint8)
    original = packing._pack_expert
    failure = MemoryError("transform allocation exhausted")
    calls = 0

    def fail_after_one(staged, kind):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise failure
        return original(staged, kind)

    monkeypatch.setattr(packing, "_pack_expert", fail_after_one)
    with pytest.raises(MemoryError) as caught:
        packing.pack_owned_into(raw, codes, scales, first=0, lane=0, ggml_type=18)
    assert caught.value is failure and calls == 2
    # The partial arrays have no success receipt and must not be consumed.
    assert np.all(codes[1:] == 19) and np.all(scales[1:] == 23)
    np.testing.assert_array_equal(source, before)


@pytest.mark.parametrize("kind", [18, 21, 22])
def test_scale_headers_are_bit_transferred_without_quieting_signaling_nan(kind):
    raw = raw_rows(kind)
    markers = np.array([0, 0x8000, 1, 0x7C00, 0xFC00, 0x7E00, 0x7D01, 0xFDFF], dtype="<u2")
    blocks = raw.reshape(160, 10, -1)
    blocks[:, :, :2] = np.tile(markers.view(np.uint8).reshape(8, 2), (20, 1))[:, None, :]
    _, _, scales = packing._pack_expert(raw, kind)
    size = {18: 1, 21: 2, 22: 4}[kind]
    expected_rows = [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27,
                     4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31]
    metadata = scales.view("<u4").reshape(5, 20, 32, size).transpose(0, 2, 1, 3)
    metadata = metadata[:, np.argsort(expected_rows)].reshape(160, 20, size)
    field = 1 if kind == 22 else size - 1
    expected = np.repeat(blocks[:, :, :2].copy().view("<u2")[:, :, 0], 2, axis=1)
    np.testing.assert_array_equal(metadata[:, :, field] & 65535, expected)


@pytest.mark.parametrize("first,lane", [(0, 1), (256, 3)])
def test_distinct_global_expert_and_lane_mapping(monkeypatch, first, lane):
    # Fixed321,126,400-byte source fixture, bounded independently of real model
    # data. Each expert/row carries its own16-bit identifiers. The spy observes
    # the actual staging seam; it does not duplicate or certify packing math.
    kind = 18
    raw = np.zeros((512, 640, 10 * packing.BLOCK_BYTES[kind]), dtype=np.uint8)
    experts = np.arange(512, dtype=np.uint16)
    rows = np.arange(640, dtype=np.uint16)
    raw[:, :, 0] = (experts & 255)[:, None]
    raw[:, :, 1] = (experts >> 8)[:, None]
    raw[:, :, 2] = (rows & 255)[None, :]
    raw[:, :, 3] = (rows >> 8)[None, :]
    raw.flags.writeable = False
    spec = packing.requirements(kind)
    codes = np.empty(spec["codes_shape"], np.uint8)
    scales = np.empty(spec["scales_shape"], np.uint8)
    observed = []

    def observe_staged(staged, actual_kind):
        assert actual_kind == kind and staged.shape == (160, 980)
        global_expert = int(staged[0, 0]) | int(staged[0, 1]) << 8
        np.testing.assert_array_equal(staged[:, 0], np.full(160, global_expert & 255))
        np.testing.assert_array_equal(staged[:, 1], np.full(160, global_expert >> 8))
        staged_rows = staged[:, 2].astype(np.uint16) | staged[:, 3].astype(np.uint16) << 8
        np.testing.assert_array_equal(staged_rows, np.arange(lane * 160, (lane + 1) * 160))
        observed.append(global_expert)
        result_codes = np.zeros(spec["codes_shape"][1], np.uint8)
        result_scales = np.zeros(spec["scales_shape"][1], np.uint8)
        marker = np.array([global_expert, lane * 160], dtype="<u2").view(np.uint8)
        result_codes[:4] = marker
        result_scales[:4] = marker
        return 6, result_codes, result_scales

    monkeypatch.setattr(packing, "_pack_expert", observe_staged)
    receipt = packing.pack_owned_into(raw, codes, scales, first=first, lane=lane, ggml_type=kind)
    assert observed == list(range(first, first + 256))
    assert receipt["completed_experts"] == 256
    assert (receipt["first"], receipt["lane"]) == (first, lane)
    for local, expected_global in ((0, first), (255, first + 255)):
        marker = np.array([expected_global, lane * 160], dtype="<u2").view(np.uint8)
        np.testing.assert_array_equal(codes[local, :4], marker)
        np.testing.assert_array_equal(scales[local, :4], marker)
    assert int(raw[511, 639, 0]) | int(raw[511, 639, 1]) << 8 == 511
