"""The synthetic NVFP4 checkpoint's affine table obeys the public unsigned-word schema."""

import importlib.util
import json
from pathlib import Path
import struct

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
safe_open = pytest.importorskip("safetensors").safe_open

ROOT = Path(__file__).resolve().parents[1]
BASE = "model.layers.1.ple.ple_embedding.ngram_embedding.shard_0."


def writer():
    spec = importlib.util.spec_from_file_location("nvfp4_tiny_interop", ROOT / "tests/cuda/nvfp4_tiny.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shared_e2m1_table_preserves_signed_codes_and_independent_storage():
    from tensorfold.cuda.nvfp4.format import E2M1
    from tensorfold.families.qwen4_exp.cuda.nvfp4 import e2m1_table

    expected = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                             -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32)
    table = e2m1_table()
    assert torch.equal(table.view(torch.int32), expected.view(torch.int32))
    table[0] = 99
    assert E2M1[0] == 0 and torch.equal(e2m1_table().view(torch.int32), expected.view(torch.int32))


def test_affine_table_header_and_payload_roundtrip_through_independent_readers(tmp_path, monkeypatch):
    from tensorfold.families.qwen4_exp.host_table import HostTable
    from tensorfold.families.qwen4_exp.ssd_table import SSDTable

    # Include the sign bit and all-one word that the ordinary small RNG fixture misses.
    signed = [0, 0x7FFFFFFF, -(2**31), -1]
    def packed_words(low, high, shape, *, generator, dtype):
        assert (low, high, dtype) == (0, 256, torch.int32)
        assert shape[1] == len(signed)
        return torch.tensor(signed, dtype=torch.int32).repeat(shape[0], 1)
    monkeypatch.setattr(torch, "randint", packed_words)
    model = writer().write(tmp_path / "model")
    shard = model / "model-00001-of-00001.safetensors"
    data = shard.read_bytes()
    header_size = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8:8 + header_size])
    entry = header[BASE + "weight"]
    assert entry["dtype"] == "U32" and entry["shape"] == [520, 4]
    lo, hi = entry["data_offsets"]
    expected = struct.pack("<4I", 0, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF) * entry["shape"][0]
    assert data[8 + header_size + lo:8 + header_size + hi] == expected
    with safe_open(str(shard), framework="np") as file:
        words = file.get_tensor(BASE + "weight")
        assert words.dtype == np.uint32 and words.tobytes() == expected
    with safe_open(str(shard), framework="pt") as file:
        stored = file.get_tensor(BASE + "weight")
        assert stored.dtype == torch.uint32 and stored.view(torch.uint8).numpy().tobytes() == expected
        scales = file.get_tensor(BASE + "scales").view(torch.uint16).numpy()
        biases = file.get_tensor(BASE + "biases").view(torch.uint16).numpy()
    parts = [(shard, header[BASE + "weight"], header[BASE + "scales"], header[BASE + "biases"])]
    rows = np.array([0, 17, entry["shape"][0] - 1], dtype=np.int64)
    for constructor in (HostTable, SSDTable):
        table = constructor(parts)
        try:
            got = table.gather(rows)
            assert got[0].dtype == np.uint32 and np.array_equal(got[0], words[rows])
            assert np.array_equal(got[1], scales[rows]) and np.array_equal(got[2], biases[rows])
        finally:
            table.close()
