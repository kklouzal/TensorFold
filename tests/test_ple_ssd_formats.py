"""Positional BF16/FP8/FP4 table reads retain decoded bits and file ownership."""

import json
import os
import struct
import threading

import numpy as np
import pytest

from tensorfold.families.qwen4_exp.host_table import open_table
from tensorfold.families.qwen4_exp.ssd_table import EncodedSSDTable

torch = pytest.importorskip("torch")


def _write(tmp_path, kind, *, width=64, rows=19):
    rng = np.random.default_rng(819)
    codes = rng.integers(0, 256, (rows, width // 2 if kind == "nvfp4" else width), dtype=np.uint8)
    if kind == "bf16":
        values = torch.randn(rows, width, generator=torch.Generator().manual_seed(45)).to(torch.bfloat16)
        parts = [("weight", "BF16", values.view(torch.uint8).numpy().reshape(rows, width * 2))]
        wanted = values.view(torch.int16).numpy().view(np.uint16)
    elif kind == "fp8":
        codes = codes % 127
        values = torch.from_numpy(codes.copy()).view(torch.float8_e4m3fn).float() * 0.073
        parts = [("weight", "F8_E4M3", codes)]
        wanted = values.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    else:
        scales = rng.integers(0, 126, (rows, width // 16), dtype=np.uint8)
        nibbles = np.stack((codes & 15, codes >> 4), -1).reshape(rows, width)
        # Independent Torch conversion and indexing, without the reader's
        # NumPy decoder or its BF16 rounding implementation.
        mag = torch.tensor((0., .5, 1., 1.5, 2., 3., 4., 6.))
        nib = torch.from_numpy(nibbles.astype(np.int64))
        value = mag[nib & 7] * torch.where(nib >= 8, -1., 1.)
        value *= torch.from_numpy(scales.copy()).view(torch.float8_e4m3fn).float().repeat_interleave(16, 1)
        wanted = (value * 0.073).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
        parts = [("weight", "U8", codes), ("weight_scale", "F8_E4M3", scales)]
    entries, payload, offset = {}, [], 0
    for field, dtype, array in parts:
        shape = [rows, array.shape[1] // 2 if dtype == "BF16" else array.shape[1]]
        entries["ngram.shard_0." + field] = {"dtype": dtype, "shape": shape,
                                            "data_offsets": [offset, offset + array.nbytes]}
        payload.append(array.tobytes())
        offset += array.nbytes
    encoded = json.dumps(entries).encode()
    path = tmp_path / "model.safetensors"
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(payload))
    return path, wanted


@pytest.mark.parametrize("kind", ("bf16", "fp8", "nvfp4"))
def test_encoded_positional_rows_match_independent_decoder(tmp_path, kind, monkeypatch):
    path, expected = _write(tmp_path, kind)
    ids = np.array([18, 0, 7, 7, 3, 1, 18], dtype=np.int64)
    mapped = open_table(tmp_path, [(path.name, "ngram.shard_0")], lambda _: .073)
    table = open_table(tmp_path, [(path.name, "ngram.shard_0")], lambda _: .073, ssd=True)
    try:
        assert isinstance(table, EncodedSSDTable)
        assert table.width == expected.shape[1] and table.rows == expected.shape[0]
        assert table.weight_scale == (.073 if kind == "bf16" else 1.)
        assert table.prefetch() == 0 and not table.lock()
        real = os.pread
        calls = []

        def short(fd, size, offset):
            calls.append(size)
            return real(fd, min(size, 7), offset)

        monkeypatch.setattr(os, "pread", short)
        assert np.array_equal(table.gather(ids), expected[ids])
        assert np.array_equal(table.gather(ids), mapped.gather(ids))
        assert max(calls) <= 1 << 20
        assert table.gather(np.empty(0, dtype=np.int64)).shape == (0, table.width)
    finally:
        mapped.close()
        table.close()
    with pytest.raises((ValueError, RuntimeError)):
        table.gather(ids)


@pytest.mark.parametrize("kind", ("bf16", "fp8", "nvfp4"))
def test_encoded_eof_and_invalid_ids_are_explicit(tmp_path, kind):
    path, _ = _write(tmp_path, kind)
    table = open_table(tmp_path, [(path.name, "ngram.shard_0")], lambda _: .073, ssd=True)
    try:
        with pytest.raises(ValueError):
            table.gather(np.array([table.rows]))
        with path.open("r+b") as stream:
            stream.truncate(path.stat().st_size - 1)
        with pytest.raises(OSError, match="short read"):
            table.gather(np.array([table.rows - 1]))
    finally:
        table.close()


def test_encoded_close_waits_for_accepted_read(tmp_path, monkeypatch):
    path, expected = _write(tmp_path, "nvfp4")
    table = open_table(tmp_path, [(path.name, "ngram.shard_0")], lambda _: .073, ssd=True)
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    real = os.pread
    result, failures = [], []

    def delayed(fd, size, offset):
        entered.set()
        assert release.wait(5)
        return real(fd, size, offset)

    monkeypatch.setattr(os, "pread", delayed)

    def gather():
        try:
            result.append(table.gather(np.array([1, 3])))
        except BaseException as error:
            failures.append(error)

    def close():
        try:
            table.close()
        except BaseException as error:
            failures.append(error)
        finally:
            closed.set()

    reader, closer = threading.Thread(target=gather), threading.Thread(target=close)
    reader.start()
    assert entered.wait(5)
    closer.start()
    assert not closed.wait(.05)
    release.set()
    reader.join(5)
    closer.join(5)
    assert not reader.is_alive() and not closer.is_alive() and not failures
    assert np.array_equal(result[0], expected[[1, 3]])


@pytest.mark.parametrize("kind", ("bf16", "fp8", "nvfp4"))
def test_one_wide_encoded_row_still_bounds_each_positional_read(tmp_path, kind, monkeypatch):
    path, expected = _write(tmp_path, kind, width=2 * ((1 << 20) + 8), rows=1)
    table = open_table(tmp_path, [(path.name, "ngram.shard_0")], lambda _: .073, ssd=True)
    real, sizes = os.pread, []

    def record(fd, size, offset):
        sizes.append(size)
        return real(fd, size, offset)

    monkeypatch.setattr(os, "pread", record)
    try:
        assert np.array_equal(table.gather(np.array([0])), expected)
        assert max(sizes) <= 1 << 20 and len(sizes) > 1
    finally:
        table.close()
