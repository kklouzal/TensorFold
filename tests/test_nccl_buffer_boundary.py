"""NCCL raw-pointer admission with standard-library provider doubles only."""
import ast
import ctypes
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from tensorfold.cleanup import rollback

ROOT = Path(__file__).resolve().parents[1]


class Tensor:
    def __init__(self, count, pointer, *, device="cuda:0", dtype="F32", contiguous=True):
        self.count, self.pointer = count, pointer
        self.device, self.dtype, self.contiguous = device, dtype, contiguous

    def numel(self):
        return self.count

    def data_ptr(self):
        return self.pointer

    def element_size(self):
        return 4

    def is_contiguous(self):
        return self.contiguous


class Provider:
    def __init__(self, name):
        self.calls = []
        self.name = name

    def __call__(self, *args):
        self.calls.append(args)
        if self.name == "ncclCommInitRank":
            args[0]._obj.value = 1234
        return 0


def source(raw=b"x" * 128):
    path = ROOT / "src/tensorfold/cuda/comm.py"
    tree = ast.parse(path.read_bytes())
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    lib = SimpleNamespace(**{name: Provider(name) for name in
                             ("ncclGetErrorString", "ncclGetUniqueId", "ncclCommInitRank", "ncclAllGather",
                              "ncclCommDestroy", "ncclCommAbort")})
    streams = []
    runtime = SimpleNamespace(Tensor=Tensor, device=lambda kind, index: f"{kind}:{index}",
                              cuda=SimpleNamespace(current_device=lambda: 0,
                              synchronize=lambda device: None,
                              current_stream=lambda device: (streams.append(device) or SimpleNamespace(cuda_stream=123))))
    namespace = {"torch": runtime, "ctypes": ctypes, "_library": lambda: lib, "_DTYPES": {"F32": 7},
                 "threading": threading, "rollback": rollback}
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *classes]
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), "exec"), namespace)
    distributed = ModuleType("torch.distributed")
    distributed.TCPStore = lambda *a, **k: SimpleNamespace(get=lambda key: raw, set=lambda *a: None)
    return namespace["NCCL"], lib, streams, distributed


class BufferBoundary(unittest.TestCase):
    def test_uid_length_and_type_refuse_before_memmove_or_native_init(self):
        for raw in (b"", b"x", b"x" * 127, b"x" * 129, "x" * 128, bytearray(128)):
            cls, lib, _, distributed = source(raw)
            with patch.dict(sys.modules, {distributed.__name__: distributed}), \
                    patch.object(ctypes, "memmove", side_effect=AssertionError("unsafe copy reached")):
                with self.assertRaises(ValueError):
                    owner = cls()
                    owner.open(1, 2, "fixture", 1234)
            self.assertFalse(lib.ncclCommInitRank.calls)

    def test_valid_uid_communicator_and_device_stream_preserve_native_values(self):
        cls, lib, streams, distributed = source()
        with patch.dict(sys.modules, {distributed.__name__: distributed}):
            owner = cls()
            owner.open(1, 2, "fixture", 1234)
        self.assertEqual(len(lib.ncclCommInitRank.calls), 1)
        owner.all_gather(Tensor(3, 400), Tensor(6, 1000))
        self.assertEqual(streams, ["cuda:0"])
        args = lib.ncclAllGather.calls[0]
        self.assertEqual((args[0], args[1], args[2], args[3], args[5]), (400, 1000, 3, 7, 123))

    def test_bad_world_rank_port_fail_before_loading_library_or_network(self):
        cls, lib, _, distributed = source()
        for rank, world, port in ((True, 2, 1234), (0, True, 1234), (-1, 2, 1234), (2, 2, 1234),
                                  (0, 0, 1234), (0, 2**31, 1234), (0, 2, -1), (0, 2, 65536)):
            with self.assertRaises(ValueError):
                owner = cls()
                owner.open(rank, world, "fixture", port)
        self.assertFalse(lib.ncclCommInitRank.calls)

    def test_raw_geometry_rejects_dtype_device_strides_sizes_and_illegal_overlap(self):
        cls, lib, _, distributed = source()
        with patch.dict(sys.modules, {distributed.__name__: distributed}):
            owner = cls()
            owner.open(1, 2, "fixture", 1234)
        cases = [(Tensor(3, 400, dtype="F64"), Tensor(6, 1000, dtype="F64")),
                 (Tensor(3, 400, device="cpu"), Tensor(6, 1000)),
                 (Tensor(3, 400), Tensor(6, 1000, device="cuda:1")),
                 (Tensor(3, 400, contiguous=False), Tensor(6, 1000)),
                 (Tensor(3, 400), Tensor(6, 1000, contiguous=False)),
                 (Tensor(3, 400), Tensor(5, 1000)),
                 (Tensor(3, 1000), Tensor(6, 1000))]
        for send, recv in cases:
            with self.assertRaises(ValueError):
                owner.all_gather(send, recv)
        self.assertFalse(lib.ncclAllGather.calls)
        owner.all_gather(Tensor(3, 1012), Tensor(6, 1000))  # valid rank1 in-place slice
        self.assertEqual(len(lib.ncclAllGather.calls), 1)


if __name__ == "__main__":
    unittest.main()
