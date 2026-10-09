"""Real NumPy table boundaries and borrowed maps; run only on ROOT's authorized remote runtime."""
from __future__ import annotations

import gc
import json
import struct

import numpy as np
import pytest

from tensorfold.families.qwen4_exp import host_table
from tensorfold.families.qwen4_exp.host_table import BF16Table, FP8Table, HostTable, NVFP4Table


def write(path, tensors):
    header, payload, offset = {}, [], 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        payload.append(raw)
        offset += len(raw)
    text = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(text)) + text + b"".join(payload))
    return header


def make(path, kind):
    if kind == "mlx":
        tensors = {"t.weight": ("U32", [3, 8], np.arange(24, dtype=np.uint32).tobytes()),
                   "t.scales": ("BF16", [3, 2], np.arange(6, dtype=np.uint16).tobytes()),
                   "t.biases": ("BF16", [3, 2], np.arange(6, dtype=np.uint16).tobytes())}
        head = write(path, tensors)
        return HostTable([(path, *(head[f"t.{name}"] for name in ("weight", "scales", "biases")))])
    dtype, width, item = ("BF16", 32, 2) if kind == "bf16" else ("F8_E4M3", 32, 1) if kind == "fp8" else ("U8", 16, 1)
    tensors = {"t.weight": (dtype, [3, width], np.arange(3 * width * item, dtype=np.uint8).tobytes())}
    if kind == "nvfp4":
        tensors["t.weight_scale"] = ("F8_E4M3", [3, 2], bytes([40] * 6))
    head = write(path, tensors)
    files = [(path, head["t.weight"], head["t.weight_scale"])] if kind == "nvfp4" else [(path, head["t.weight"])]
    return NVFP4Table(files, 0.5) if kind == "nvfp4" else FP8Table(files, 0.5) if kind == "fp8" else BF16Table(files)


@pytest.mark.parametrize("kind", ["mlx", "bf16", "fp8", "nvfp4"])
def test_close_retains_borrowed_numpy_map_and_completed_gather(tmp_path, kind):
    table = make(tmp_path / "table.safetensors", kind)
    arrays = table.words if kind == "mlx" else table.values
    borrowed = arrays[0][1:]
    expected = borrowed.tobytes()
    result = table.gather(np.array([2, 1, 2], dtype=np.int64))
    result_bytes = tuple(array.tobytes() for array in result) if kind == "mlx" else result.tobytes()
    table.close()
    gc.collect()
    assert arrays == []
    assert borrowed.tobytes() == expected
    assert (tuple(array.tobytes() for array in result) if kind == "mlx" else result.tobytes()) == result_bytes
    with pytest.raises(ValueError, match="closed"):
        table.gather([0])
    table.close()


@pytest.mark.parametrize("kind", ["mlx", "bf16", "fp8", "nvfp4"])
def test_public_ids_refused_before_narrowing_and_empty_ids_supported(tmp_path, kind):
    table = make(tmp_path / "table.safetensors", kind)
    try:
        for ids in ([True], [1.5], [np.nan], np.array([2**64 - 1], dtype=np.uint64), [-1], [3]):
            with pytest.raises((ValueError, TypeError)):
                table.gather(ids)
        got = table.gather([])
        assert (all(array.shape[0] == 0 for array in got) if kind == "mlx" else got.shape[0] == 0)
    finally:
        table.close()


@pytest.mark.parametrize("cls", [HostTable, BF16Table, FP8Table, NVFP4Table])
def test_empty_shards_refused_without_resource_leak(cls):
    with pytest.raises(ValueError, match="no shards"):
        cls([], 1.0) if cls in (FP8Table, NVFP4Table) else cls([])


def test_nvfp4_block_row_count_matches_codes(tmp_path):
    path = tmp_path / "table.safetensors"
    head = write(path, {"t.weight": ("U8", [3, 16], bytes(48)),
                        "t.weight_scale": ("F8_E4M3", [2, 2], bytes(4))})
    with pytest.raises(ValueError, match="row width"):
        NVFP4Table([(path, head["t.weight"], head["t.weight_scale"])], 1.0)


def test_mlx_component_rows_and_group_geometry_match(tmp_path):
    path = tmp_path / "table.safetensors"
    for groups, rows in ((3, 3), (2, 2)):
        head = write(path, {"t.weight": ("U32", [3, 8], bytes(96)),
                            "t.scales": ("BF16", [rows, groups], bytes(rows * groups * 2)),
                            "t.biases": ("BF16", [rows, groups], bytes(rows * groups * 2))})
        with pytest.raises(ValueError, match="4-bit rows"):
            HostTable([(path, *(head[f"t.{name}"] for name in ("weight", "scales", "biases")))])


def test_checkpoint_path_admission_cannot_escape_model(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    make(tmp_path / "outside.safetensors", "bf16").close()
    (model / "escape.safetensors").symlink_to(tmp_path / "outside.safetensors")
    for shard in ("../outside.safetensors", str(tmp_path / "outside.safetensors"), "escape.safetensors"):
        with pytest.raises(ValueError):
            host_table.open_table(model, [(shard, "t")], lambda _: 1.0)


def test_later_scale_failure_closes_acquired_table(tmp_path, monkeypatch):
    path = tmp_path / "table.safetensors"
    make(path, "bf16").close()
    acquired = []
    original = host_table.BF16Table
    def record(files):
        table = original(files)
        acquired.append(table)
        return table
    def fail(_):
        raise LookupError("table scale failed")
    monkeypatch.setattr(host_table, "BF16Table", record)
    with pytest.raises(LookupError, match="table scale failed"):
        host_table.open_table(tmp_path, [(path.name, "t")], fail)
    assert len(acquired) == 1 and acquired[0]._life.closed and acquired[0].values == []


def test_later_mapping_failure_closes_earlier_owned_mappings(tmp_path, monkeypatch):
    path = tmp_path / "table.safetensors"
    table = make(path, "mlx")
    table.close()
    header = host_table.read_header(path)
    files = [(path, *(header[f"t.{name}"] for name in ("weight", "scales", "biases")))]
    original = host_table.HostTable._map
    acquired = []
    def fail_second(self, *args):
        acquired.append(self)
        if len(acquired) == 2:
            raise OSError("later map acquisition failed")
        return original(self, *args)
    monkeypatch.setattr(host_table.HostTable, "_map", fail_second)
    with pytest.raises(OSError, match="later map acquisition failed"):
        HostTable(files)
    assert acquired[0] is acquired[1] and acquired[0]._life.closed
    assert acquired[0].words == [] and not acquired[0]._files


def test_in_bounds_supplied_geometry_must_match_opened_tensor_header(tmp_path):
    path = tmp_path / "table.safetensors"
    make(path, "bf16").close()
    entry = host_table.read_header(path)["t.weight"]
    altered = {**entry, "shape": [1, 32], "data_offsets": [0, 64]}
    with pytest.raises(ValueError, match="disagrees"):
        BF16Table([(path, altered)])
