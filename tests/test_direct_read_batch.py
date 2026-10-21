"""Owned ordered F32 scalar batching; CPU source controls plus real Torch bits."""

import ast
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorfold.cuda.tensor_file import byte_range

SOURCE = Path(__file__).parents[1] / "src/tensorfold/cuda/direct_read.py"


class BytesTensor:
    """Shared byte views make owned output and typed copy integration observable."""

    def __init__(self, raw, *, dtype="u8", shape=None):
        self.raw = memoryview(raw)
        self.dtype, self.shape = dtype, shape

    def view(self, dtype):
        return BytesTensor(self.raw, dtype=dtype)

    def reshape(self, *shape):
        return BytesTensor(self.raw, dtype=self.dtype, shape=shape)

    def index_copy_(self, dimension, positions, source):
        assert dimension == 0 and self.dtype == source.dtype == "u8"
        for row, destination in enumerate(positions.indices):
            self.raw[4 * destination:4 * destination + 4] = source.raw[4 * row:4 * row + 4]
        return self


def reader(where, *, failure=None):
    tree = ast.parse(SOURCE.read_bytes())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SafeTensors")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "get_f32_scalars")
    calls, buffers, allocations = [], [], []

    def empty(shape, *, dtype, device):
        assert dtype == "f32" and device == "cpu"
        allocations.append(shape)
        return BytesTensor(bytearray(4 * shape[0]), dtype=dtype, shape=shape)

    def tensor(values, *, dtype, device):
        assert dtype == "i64" and device == "cpu"
        return SimpleNamespace(indices=tuple(values))

    scope = {"torch": SimpleNamespace(empty=empty, tensor=tensor, float32="f32", int64="i64", uint8="u8")}
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method]
    exec(compile(ast.fix_missing_locations(ast.Module(body, [])), str(SOURCE), "exec"), scope)  # noqa: S102 - actual owned method, no external code.

    def read(path, offset, count):
        if failure is not None:
            raise failure
        calls.append((path, offset, count))
        payload = path.read_bytes()
        byte_range(offset, count, len(payload), path)
        raw = bytearray(payload[offset:offset + count])
        buffers.append(raw)
        return BytesTensor(raw)

    owner = SimpleNamespace(where=where, reader=SimpleNamespace(read=read))

    def get(names, **kwargs):
        return scope["get_f32_scalars"](owner, names, **kwargs)

    get.allocations = allocations
    return get, calls, buffers


def test_contiguous_runs_reorder_exact_bytes_without_gaps_cross_shards_or_shared_storage(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(bytes(range(24)))
    b.write_bytes(b"abcd")
    entries = {"one": (a, 1, 4, "F32", [1]), "two": (a, 5, 4, "F32", [1]),
               "gap": (a, 13, 4, "F32", [1]), "shard": (b, 0, 4, "F32", [1])}
    names = ["two", "shard", "one", "gap"]
    get, calls, buffers = reader(entries)
    got = get(names, max_bytes=16)
    assert calls == [(a, 1, 8), (a, 13, 4), (b, 0, 4)]
    expected = b"".join(entries[name][0].read_bytes()[entries[name][1]:entries[name][1] + 4] for name in names)
    assert bytes(got.raw) == expected and got.dtype == "f32" and got.shape == (4,)
    for raw in buffers:
        raw[:] = b"\xff" * len(raw)
    assert bytes(got.raw) == expected
    assert get.allocations == [(4,)]


@pytest.mark.parametrize("fault", ["duplicate", "unknown", "dtype", "shape", "empty_field", "negative", "boolean", "bound"])
def test_complete_scalar_admission_precedes_io_and_output_allocation(tmp_path, fault):
    path = tmp_path / "bytes"
    path.write_bytes(b"abcdefgh")
    entries = {"good": (path, 0, 4, "F32", [1]), "late": (path, 4, 4, "F32", [1])}
    names, bound = ["good", "late"], 8
    if fault == "duplicate":
        names.append("good")
    elif fault == "unknown":
        names.append("absent")
    elif fault == "dtype":
        entries["late"] = (path, 4, 4, "F16", [2])
    elif fault == "shape":
        entries["late"] = (path, 4, 4, "F32", [])
    elif fault == "empty_field":
        entries["late"] = (path, 4, 0, "F32", [0])
    elif fault == "negative":
        entries["late"] = (path, -1, 4, "F32", [1])
    elif fault == "boolean":
        entries["late"] = (path, True, 4, "F32", [1])
    else:
        bound = 7
    get, calls, _ = reader(entries)
    with pytest.raises((ValueError, KeyError)):
        get(names, max_bytes=bound)
    assert calls == [] and get.allocations == []


@pytest.mark.parametrize("bound", [True, -1, 1.0])
def test_invalid_budget_refuses_before_io(bound):
    get, calls, _ = reader({})
    with pytest.raises(ValueError, match="byte bound"):
        get([], max_bytes=bound)
    assert calls == [] and get.allocations == []


def test_empty_batch_and_invalid_name_collection_do_not_read():
    get, calls, _ = reader({})
    empty = get([], max_bytes=0)
    assert bytes(empty.raw) == b"" and empty.shape == (0,)
    with pytest.raises(TypeError, match="sequence"):
        get("name", max_bytes=4)
    assert calls == []


def test_current_descriptor_range_validation_is_preserved(tmp_path):
    path = tmp_path / "bytes"
    path.write_bytes(b"abcdefgh")
    get, calls, _ = reader({"a": (path, 0, 4, "F32", [1]), "b": (path, 4, 4, "F32", [1])})
    path.write_bytes(b"abcd")
    with pytest.raises(OSError, match="short read"):
        get(["a", "b"], max_bytes=8)
    assert calls == [(path, 0, 8)]


def test_read_failure_preserves_exact_primary_object(tmp_path):
    primary = OSError("read failed")
    get, _, _ = reader({"a": (tmp_path / "absent", 0, 4, "F32", [1])}, failure=primary)
    with pytest.raises(OSError) as caught:
        get(["a"], max_bytes=4)
    assert caught.value is primary


@pytest.mark.parametrize("direct", [True, False])
def test_real_host_scalar_batch_exact_bits_order_and_alignment(tmp_path, direct):
    torch = pytest.importorskip("torch")
    from tensorfold.cuda.direct_read import Reader, SafeTensors

    bits = (0, 0x80000000, 1, 0x007FFFFF, 0x00800000, 0x3F800000,
            0x7F800000, 0xFF800000, 0x7FC01234, 0x7F801234, 0xFFFFFFFF)
    path = tmp_path / "bytes"
    # The unaligned run crosses a page boundary and the final scalar ends at EOF.
    path.write_bytes(b"x" * 4093 + struct.pack("<" + "I" * len(bits), *bits))
    files = SafeTensors.__new__(SafeTensors)
    files.reader = Reader()
    files.reader.direct = files.reader.direct and direct
    files.where = {f"field.{index}": (path, 4093 + 4 * index, 4, "F32", [1]) for index in range(len(bits))}
    names = list(reversed(files.where))
    try:
        got = files.get_f32_scalars(names, max_bytes=4 * len(names))
        expected = torch.cat([files.get(name).view(torch.uint8) for name in names])
        assert got.dtype == torch.float32 and got.shape == (len(names),) and got.device.type == "cpu"
        assert got.is_contiguous() and got.storage_offset() == 0
        assert got.untyped_storage().nbytes() == 4 * len(names)
        assert torch.equal(got.view(torch.uint8), expected)
    finally:
        files.close()
