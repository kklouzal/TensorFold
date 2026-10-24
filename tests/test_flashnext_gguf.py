"""Independent tiny GGUF fixtures: extents, splits, byte codecs and packed TP boundaries."""

from pathlib import Path
import os
import struct

import numpy as np
import pytest

from tensorfold.families.qwen4_exp.gguf import GGUFCheckpoint, PackedTensor, decode_rows


def string(text):
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def fixture(path, tensors=(), *, metadata=None):
    fields = {"general.architecture": "qwen4exp", **(metadata or {})}
    body = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(fields))
    for key, value in fields.items():
        body += string(key)
        body += (struct.pack("<I", 8) + string(value) if isinstance(value, str)
                 else struct.pack("<II", 4, value))
    payload = b""
    for name, shape, kind, raw in tensors:
        body += string(name) + struct.pack("<I", len(shape))
        body += struct.pack("<" + "Q" * len(shape), *reversed(shape))
        body += struct.pack("<IQ", kind, len(payload))
        payload += raw
        payload += b"\0" * (-len(payload) % 32)
    body += b"\0" * (-len(body) % 32)
    path.write_bytes(body + payload)
    return path


def test_zero_tensor_entrypoint_discovers_ordered_siblings(tmp_path):
    first = fixture(tmp_path / "model-00001-of-00002.gguf", metadata={
        "split.no": 0, "split.count": 2, "split.tensors.count": 1})
    fixture(tmp_path / "model-00002-of-00002.gguf", [("output.weight", (2, 32), 8, b"\0" * 68)], metadata={
        "split.no": 1, "split.count": 2, "split.tensors.count": 1})
    checkpoint = GGUFCheckpoint.open(first)
    descriptor = checkpoint.tensors["output.weight"]
    assert descriptor.shape == (2, 32) and descriptor.encoding == "Q8_0"
    assert descriptor.path == checkpoint.paths[1]
    offset, size = descriptor.row_span(1, 1)
    assert offset == descriptor.offset + 34 and size == 34
    assert descriptor.path.read_bytes()[offset:offset + size] == b"\0" * 34
    with pytest.raises(TypeError):
        checkpoint.tensors["other"] = descriptor


@pytest.mark.parametrize("field,value", [("split.no", 0), ("split.count", 3), ("split.tensors.count", 2)])
def test_inconsistent_split_set_is_refused(tmp_path, field, value):
    first = fixture(tmp_path / "model-00001-of-00002.gguf", metadata={
        "split.no": 0, "split.count": 2, "split.tensors.count": 1})
    fixture(tmp_path / "model-00002-of-00002.gguf", [("x", (1,), 0, b"\0" * 4)], metadata={
        **{"split.no": 1, "split.count": 2, "split.tensors.count": 1}, field: value})
    with pytest.raises(ValueError, match="split|count"):
        GGUFCheckpoint.open(first)


def test_missing_shard_is_not_silently_ignored(tmp_path):
    path = fixture(tmp_path / "model-00001-of-00002.gguf", metadata={"split.no": 0, "split.count": 2})
    with pytest.raises(FileNotFoundError):
        GGUFCheckpoint.open(path)


@pytest.mark.parametrize("kind,width,length", [(0, 8, 32), (1, 16, 32), (7, 32, 24), (8, 32, 34),
                                               (12, 256, 144), (13, 256, 176), (20, 160, 90), (30, 16, 32)])
def test_all_target_encodings_have_exact_extents(tmp_path, kind, width, length):
    path = fixture(tmp_path / "one.gguf", [("w", (2, width), kind, bytes(2 * length))])
    tensor = GGUFCheckpoint.open(path).tensors["w"]
    assert tensor.row_bytes == length and tensor.nbytes == 2 * length
    assert tensor.row_span(0, 2) == (tensor.offset, 2 * length)


@pytest.mark.parametrize("mutation", ["truncated", "duplicate", "unknown", "wrong_arch", "misblocked"])
def test_malformed_inventory_fails_before_payload_reads(tmp_path, mutation):
    entries = [("w", (32,), 8, b"\0" * 34)]
    if mutation == "duplicate":
        entries *= 2
    if mutation == "unknown":
        entries = [("w", (32,), 999, b"\0" * 34)]
    if mutation == "misblocked":
        entries = [("w", (31,), 8, b"\0" * 34)]
    path = fixture(tmp_path / "one.gguf", entries, metadata={
        "general.architecture": "other" if mutation == "wrong_arch" else "qwen4exp"})
    if mutation == "truncated":
        path.write_bytes(path.read_bytes()[:-64])
    with pytest.raises(ValueError):
        GGUFCheckpoint.open(path)


def test_huge_string_length_does_not_allocate_from_untrusted_metadata(tmp_path):
    path = tmp_path / "bad.gguf"
    path.write_bytes(b"GGUF" + struct.pack("<IQQQ", 3, 0, 1, 2**63))
    with pytest.raises(ValueError, match="header"):
        GGUFCheckpoint.open(path)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO check")
def test_fifo_is_refused_without_waiting_for_a_writer(tmp_path):
    path = tmp_path / "pipe.gguf"
    os.mkfifo(path)
    with pytest.raises(ValueError, match="regular files"):
        GGUFCheckpoint.open(path)


@pytest.mark.parametrize("start,count", [(-1, 1), (0, -1), (1, 2), (True, 1)])
def test_row_ranges_are_exact(tmp_path, start, count):
    tensor = PackedTensor("w", (2, 32), 8, tmp_path / "unused", 0, 68)
    with pytest.raises(ValueError, match="range"):
        tensor.row_span(start, count)


def test_tp8_width80_cannot_cut_packed_quantization_blocks():
    tensor = PackedTensor("down", (2560, 640), 7, Path("unused"), 0, 2560 * 480)
    with pytest.raises(ValueError, match="complete 32-value blocks"):
        tensor.column_span(0, 80)
    assert tensor.column_span(0, 160) == (0, 120)


def test_row_read_is_bounded_and_uses_existing_reader(tmp_path):
    tensor = PackedTensor("w", (10, 32), 8, tmp_path / "unused", 100, 340)

    class Reader:
        def read(self, path, offset, length, device):
            assert (path, offset, length, device) == (tensor.path, 168, 68, "cpu")
            return np.arange(length, dtype=np.uint8)

    assert tensor.read_rows(Reader(), 2, 2).shape == (2, 34)
    with pytest.raises(ValueError, match="staging budget"):
        tensor.read_rows(Reader(), 0, 10, max_bytes=64)


def test_q8_scalar_reference():
    payload = struct.pack("<e", 0.5) + struct.pack("<32b", *range(-16, 16))
    tensor = PackedTensor("w", (1, 32), 8, Path("unused"), 0, 34)
    result = decode_rows(tensor, np.frombuffer(payload, np.uint8).reshape(1, 34))
    np.testing.assert_array_equal(result[0], [n * 0.5 for n in range(-16, 16)])


def test_iq4_nl_nibble_order_and_signed_scale():
    # Low nibbles are the first sixteen values; high nibbles are the next sixteen.
    payload = struct.pack("<e", -0.25) + bytes(i | ((15 - i) << 4) for i in range(16))
    tensor = PackedTensor("ple", (1, 32), 20, Path("unused"), 0, 18)
    result = decode_rows(tensor, np.frombuffer(payload, np.uint8).reshape(1, 18))
    codebook = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]
    np.testing.assert_array_equal(result[0], [-0.25 * n for n in codebook + codebook[::-1]])


def test_q4_k_retains_its_group_scales_and_nibble_order():
    tensor = PackedTensor("expert", (1, 256), 12, Path("unused"), 0, 144)
    block = np.zeros((1, 144), np.uint8)
    block[0, :2] = np.frombuffer(struct.pack("<e", 1), np.uint8)
    block[0, 4:8] = 1
    block[0, 12:16] = 1
    block[0, 16:] = 0x21
    np.testing.assert_array_equal(decode_rows(tensor, block)[0], ([1] * 32 + [2] * 32) * 4)
