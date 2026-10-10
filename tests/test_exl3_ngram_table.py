"""EXL3's n-gram table reports its bytes, as every n-gram table the Flash Next CUDA engine prefetches does."""

from types import SimpleNamespace
import json
import struct
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

pytestmark = pytest.mark.torch


@pytest.mark.parametrize("consolidated", [False, True])
@pytest.mark.parametrize("bits", [2, 4, 6, 8])
def test_the_exl3_ngram_table_reports_bytes_and_gathers_layouts(tmp_path, consolidated, bits):
    import torch

    from tensorfold.families.qwen4_exp.cuda import exl3_pack

    words, rows = 1 + 160 * bits // 16, [3, 5]                    # 4-bit rows: a scale word and 160 values
    data = np.arange(sum(rows) * words, dtype=np.int16).reshape(sum(rows), words)
    (tmp_path / "ngram.safetensors").write_bytes(data.tobytes())
    entries = {f"t.shard_{i}.trellis": ("ngram.safetensors", 2 * words * sum(rows[:i]),
                                        2 * words * sum(rows[:i + 1]), "I16", [n, words]) for i, n in enumerate(rows)}
    if consolidated:
        entries = {"t.trellis": ("ngram.safetensors", 0, data.nbytes, "I16", list(data.shape))}
    tensors = {"t.head_bias": torch.zeros(4), "t.head_offsets": torch.zeros(2, dtype=torch.int64),
               "t.head_vocab_sizes": torch.ones(2, dtype=torch.int64), "t.layer_multipliers": torch.ones(2)}
    pk = SimpleNamespace(dir=tmp_path, entry=entries.__getitem__, get=tensors.__getitem__)
    table = exl3_pack.NgramTable(pk, "t.", 2, "cpu")
    assert table.nbytes == data.nbytes
    ids = np.array([7, 0, 3, 2, 3, 5])
    assert table.gather(ids).tobytes() == data[ids].tobytes()
    assert table.gather([]).shape == (0, words)
    for bad in ([-1], [8], [0, 8]):
        with pytest.raises(IndexError, match="outside"):
            table.gather(bad)
    assert all(isinstance(a, np.memmap) and a.mode == "r" for a in table.words)
    table.close()


@pytest.mark.parametrize("shape,dtype,end", [([0, 61], "I16", 0), ([3, 61], "F16", 366),
                                            ([3, 61], "I16", 364), ([3, 62], "I16", 372)])
def test_ngram_table_rejects_invalid_packed_segments(tmp_path, shape, dtype, end):
    from tensorfold.families.qwen4_exp.cuda import exl3_pack

    (tmp_path / "rows").write_bytes(bytes(1024))
    pk = SimpleNamespace(dir=tmp_path, entry={"t.trellis": ("rows", 0, end, dtype, shape)}.__getitem__)
    with pytest.raises(ValueError):
        exl3_pack.NgramTable(pk, "t.", 128, "cpu")


def real_pack(tmp_path, *, consolidated=True):
    """Real indexed safetensors and production Pack/Reader, with independent row bytes."""
    from tensorfold.families.qwen4_exp.cuda.exl3_pack import Pack

    data = (np.arange(8 * 41, dtype=np.int16).reshape(8, 41) * 17 - 1200).astype(np.int16)
    tensors = ({"t.trellis": ("I16", data)} if consolidated else {
        "t.shard_0.trellis": ("I16", data[:3]), "t.shard_1.trellis": ("I16", data[3:])})
    tensors.update({"t.head_bias": ("F32", np.zeros(4, dtype=np.float32)),
                    "t.head_offsets": ("I64", np.array([0, 4], dtype=np.int64)),
                    "t.head_vocab_sizes": ("I64", np.array([4, 4], dtype=np.int64)),
                    "t.layer_multipliers": ("I64", np.array([17, 19], dtype=np.int64))})
    header, payload = {}, bytearray()
    for name, (dtype, array) in tensors.items():
        start = len(payload)
        payload.extend(array.tobytes())
        header[name] = {"dtype": dtype, "shape": list(array.shape), "data_offsets": [start, len(payload)]}
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: "model.safetensors" for name in tensors}}))
    return Pack(tmp_path), data


@pytest.mark.parametrize("consolidated", [False, True])
def test_actual_pack_prefetch_pin_gather_and_explicit_close(tmp_path, consolidated):
    from tensorfold.families.qwen4_exp.cuda.exl3_pack import NgramTable

    pk, expected = real_pack(tmp_path, consolidated=consolidated)
    table = NgramTable(pk, "t.", 2, "cpu")
    borrowed = table.words[0][1:3]
    before = borrowed.tobytes()
    try:
        assert table.prefetch(2) >= 0
        # This is the real OS pin call, including its legitimate refusal policy.
        locked = table.lock()
        assert type(locked) is bool
        ids = np.array([7, 0, 3, 2, 3, 5])
        assert table.gather(ids).tobytes() == expected[ids].tobytes()
        assert bool(table._pins) is locked
    finally:
        table.close()
        pk.io.close()
    assert table._life.closed and not table._pins
    assert table.words == table.maps == table.scales == table.biases == []
    assert borrowed.tobytes() == before
    table.close()
    for operation in (lambda: table.gather([0]), table.prefetch, table.lock):
        with pytest.raises(ValueError, match="closed or unusable"):
            operation()


def test_actual_prefetch_borrower_finishes_before_concurrent_close(tmp_path, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda.exl3_pack import NgramTable
    from tensorfold.families.qwen4_exp import host_table

    pk, expected = real_pack(tmp_path)
    table = NgramTable(pk, "t.", 2, "cpu")
    entered, release = threading.Event(), threading.Event()
    original = host_table._prefetch

    def wait_then_read(arrays, workers):
        entered.set()
        if not release.wait(20):
            raise TimeoutError("prefetch fixture release")
        return original(arrays, workers)

    monkeypatch.setattr(host_table, "_prefetch", wait_then_read)
    with ThreadPoolExecutor(2) as pool:
        read = pool.submit(table.prefetch, 2)
        assert entered.wait(5)
        close = pool.submit(table.close)
        try:
            with table._life.condition:
                for _ in range(100):
                    if table._life.closing:
                        break
                    table._life.condition.wait(.01)
                assert table._life.closing
            assert not close.done()
            assert table.words[0].tobytes() == expected.tobytes()
        finally:
            release.set()
        assert read.result(timeout=5) >= 0
        close.result(timeout=5)
    assert table._life.closed and not table.words and not table.maps
    pk.io.close()


def test_constructor_failure_retires_all_previously_mapped_arrays(tmp_path, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda.exl3_pack import NgramTable
    from tensorfold.families.qwen4_exp.host_table import _MappedTable

    pk, _ = real_pack(tmp_path, consolidated=False)
    original, captured = _MappedTable._arrays, []

    def record(self):
        result = original(self)
        captured.append((self, result))
        return result

    monkeypatch.setattr(_MappedTable, "_arrays", record)
    primary = OSError("head tensor read failure after all mappings")
    actual_get = pk.get

    def fail_after_mapping(name):
        if name == "t.head_bias":
            raise primary
        return actual_get(name)

    monkeypatch.setattr(pk, "get", fail_after_mapping)
    with pytest.raises(OSError) as error:
        NgramTable(pk, "t.", 2, "cpu")
    assert error.value is primary
    assert captured and all(not arrays for _, arrays in captured)
    assert all(table._life.closed for table, _ in captured)
    pk.io.close()


def test_actual_gather_borrower_finishes_before_concurrent_close(tmp_path, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda.exl3_pack import NgramTable

    pk, expected = real_pack(tmp_path)
    table = NgramTable(pk, "t.", 2, "cpu")
    entered, release = threading.Event(), threading.Event()
    original = table._gather_flat

    def wait_then_gather(ids):
        entered.set()
        if not release.wait(20):
            raise TimeoutError("gather fixture release")
        return original(ids)

    monkeypatch.setattr(table, "_gather_flat", wait_then_gather)
    with ThreadPoolExecutor(2) as pool:
        pending = pool.submit(table.gather, [7, 0, 2, 5])
        assert entered.wait(5)
        close = pool.submit(table.close)
        try:
            with table._life.condition:
                for _ in range(100):
                    if table._life.closing:
                        break
                    table._life.condition.wait(.01)
                assert table._life.closing
            assert not close.done() and table.maps
        finally:
            release.set()
        output = pending.result(timeout=5)
        close.result(timeout=5)
    assert output.tobytes() == expected[[7, 0, 2, 5]].tobytes()
    assert table._life.closed and not table.words and not table.maps
    pk.io.close()


def test_failed_unlock_retains_real_pin_authority_until_explicit_retry(tmp_path):
    from tensorfold.families.qwen4_exp.cuda.exl3_pack import NgramTable

    pk, expected = real_pack(tmp_path)
    table = NgramTable(pk, "t.", 2, "cpu")
    # The syscall status adapter exercises failed release without requiring the
    # host to sabotage a real munlock. The retained array itself is real NumPy.
    fail = [True]

    def unlock(address, size):
        return -1 if fail[0] else 0

    array = table.words[0]
    table._pins.append((array, unlock, array.ctypes.data, array.nbytes, False))
    try:
        with pytest.raises(OSError, match="could not unlock"):
            table.close()
        assert not table._life.closed and table._pins and table.maps
        assert array.tobytes() == expected.tobytes()
        with pytest.raises(ValueError, match="closed or unusable"):
            table.gather([0])
    finally:
        fail[0] = False
        table.close()
        pk.io.close()
    assert table._life.closed and not table._pins and not table.maps


def test_exl3_table_satisfies_shared_read_ahead_owned_gather_contract(tmp_path):
    from tensorfold.families.qwen4_exp.cuda.exl3_pack import NgramTable
    from tensorfold.families.qwen4_exp.read_ahead import ReadAhead

    pk, expected = real_pack(tmp_path)
    table = NgramTable(pk, "t.", 2, "cpu")
    ahead = ReadAhead(table)
    try:
        ids = np.array([7, 0, 3, 2, 5])
        ahead.read_ahead(ids)
        assert ahead.gather(ids).tobytes() == expected[ids].tobytes()
    finally:
        ahead.close()
        pk.io.close()
    assert table._life.closed and not table.maps
