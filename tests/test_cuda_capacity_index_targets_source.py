"""Operation-local shard authorization with real isolated filesystem schedules.

The exact owned metadata/headers source runs with a stdlib regular-stream
fixture. Native FD transport, CUDA, full models and performance are separate
ROOT gates; no SDK or installed extension is imported by these controls.
"""
from __future__ import annotations
import ast
from contextlib import contextmanager
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import threading
from types import ModuleType
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def regular_fixture(path):
    with open(path, "rb") as stream:
        yield stream


def metadata_module():
    path = ROOT / "src/tensorfold/cuda/tensor_file.py"
    module = ModuleType("tensorfold.cuda.tensor_file")
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), "exec"), vars(module))
    module._regular_stream = regular_fixture
    return module


def header_function():
    path = ROOT / "src/tensorfold/cuda/capacity.py"
    tree = ast.parse(path.read_text())
    sizes = next(node for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "SIZES" for target in node.targets))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "headers")
    module = ast.fix_missing_locations(ast.Module(body=[sizes, function], type_ignores=[]))
    namespace = dict(os=os, Path=Path, __name__="tensorfold.cuda.capacity", __package__="tensorfold.cuda")
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["headers"]


def write_shard(path, count, *, prefix="x"):
    entries = {prefix+str(i): dict(dtype="U8", shape=[1], data_offsets=[i, i+1]) for i in range(count)}
    raw = json.dumps(entries).encode()
    path.write_bytes(struct.pack("<Q", len(raw))+raw+b"x"*count)
    return entries


class IndexTargets(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.metadata = metadata_module()
        self.headers = header_function()
        self.modules = patch.dict(sys.modules, {"tensorfold.cuda.tensor_file": self.metadata})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def index(self, mapping):
        (self.root / "model.safetensors.index.json").write_text(json.dumps(dict(weight_map=mapping)))

    def test_large_single_shard_retains_all_entries_and_constant_path_checks(self):
        expected = write_shard(self.root / "a.safetensors", 512)
        self.index({name: "a.safetensors" for name in expected})
        original = self.metadata.checkpoint_path
        with patch.object(self.metadata, "checkpoint_path", wraps=original) as checked:
            actual = self.headers(self.root)
        self.assertEqual(actual, {name: dict(info, split=False) for name, info in expected.items()})
        self.assertEqual(sum(call.args[1] == "a.safetensors" for call in checked.call_args_list), 4)

    def test_small_indexes_keep_original_direct_resolution_count(self):
        for count in (1, 2):
            expected = write_shard(self.root / "a.safetensors", count)
            self.index({name: "a.safetensors" for name in expected})
            with patch.object(self.metadata, "checkpoint_path", wraps=self.metadata.checkpoint_path) as checked:
                self.assertEqual(len(self.headers(self.root)), count)
            self.assertEqual(sum(call.args[1] == "a.safetensors" for call in checked.call_args_list), 2+count)

    def test_multiple_shards_preserve_mapping_and_split_flags(self):
        a = write_shard(self.root / "a.safetensors", 16, prefix="a")
        b = write_shard(self.root / "b.safetensors", 16, prefix="b")
        self.index(dict.fromkeys(a, "a.safetensors") | dict.fromkeys(b, "b.safetensors"))
        with patch.object(self.metadata, "checkpoint_path", wraps=self.metadata.checkpoint_path) as checked:
            actual = self.headers(self.root)
        self.assertEqual(actual, {name: dict(info, split=False) for name, info in (a | b).items()})
        for shard in ("a.safetensors", "b.safetensors"):
            self.assertEqual(sum(call.args[1] == shard for call in checked.call_args_list), 4)

    def test_all_target_authorization_precedes_any_header_open(self):
        (self.root / "a.safetensors").write_bytes(b"invalid")
        self.index({"x": "a.safetensors", "y": "../outside"})
        with patch.object(self.metadata, "read_header", side_effect=AssertionError("early header open")):
            with self.assertRaisesRegex(ValueError, "traverse"):
                self.headers(self.root)

    def test_missing_and_wrong_declared_shards_preserve_failure(self):
        a = write_shard(self.root / "a.safetensors", 8)
        write_shard(self.root / "b.safetensors", 8, prefix="b")
        for mapping in (dict.fromkeys(a, "b.safetensors"), dict.fromkeys(a, "a.safetensors") | {"missing": "a.safetensors"}):
            self.index(mapping)
            with self.assertRaisesRegex(ValueError, "declared tensor shard"):
                self.headers(self.root)

    def retarget(self, target):
        temporary = self.root / "replacement-link"
        temporary.symlink_to(target)
        os.replace(temporary, self.root / "linked.safetensors")

    def test_retarget_during_header_read_refuses_inside_and_outside_redirects(self):
        expected = write_shard(self.root / "a.safetensors", 8)
        write_shard(self.root / "b.safetensors", 8)
        self.index(dict.fromkeys(expected, "linked.safetensors"))
        original = self.metadata.read_header
        for target in ("b.safetensors", "../outside.safetensors"):
            linked = self.root / "linked.safetensors"
            linked.unlink(missing_ok=True)
            linked.symlink_to("a.safetensors")
            def read(path, sizes):
                result = original(path, sizes)
                self.retarget(target)
                return result
            with patch.object(self.metadata, "read_header", side_effect=read):
                with self.assertRaises(ValueError):
                    self.headers(self.root)

    def test_real_thread_retarget_after_postread_resolution_is_rechecked_before_publication(self):
        expected = write_shard(self.root / "a.safetensors", 8)
        write_shard(self.root / "b.safetensors", 8)
        (self.root / "linked.safetensors").symlink_to("a.safetensors")
        self.index(dict.fromkeys(expected, "linked.safetensors"))
        ready, completed = threading.Event(), threading.Event()
        errors = []
        def change():
            try:
                if not ready.wait(5):
                    raise TimeoutError("owned retarget schedule")
                self.retarget("b.safetensors")
            except BaseException as error:
                errors.append(error)
            finally:
                completed.set()
        actor = threading.Thread(target=change)
        original = self.metadata.checkpoint_path
        calls = 0
        def check(root, shard):
            nonlocal calls
            target = original(root, shard)
            if shard == "linked.safetensors":
                calls += 1
                if calls == 3:
                    ready.set()
                    if not completed.wait(5):
                        raise TimeoutError("owned retarget completion")
            return target
        actor.start()
        try:
            with patch.object(self.metadata, "checkpoint_path", side_effect=check):
                with self.assertRaisesRegex(ValueError, "target changed"):
                    self.headers(self.root)
        finally:
            ready.set()
            actor.join(5)
        self.assertFalse(actor.is_alive())
        self.assertEqual(errors, [])

    def test_next_invocation_rebuilds_namespace_and_metadata(self):
        old = write_shard(self.root / "a.safetensors", 8)
        new = write_shard(self.root / "b.safetensors", 9)
        (self.root / "linked.safetensors").symlink_to("a.safetensors")
        self.index(dict.fromkeys(old, "linked.safetensors"))
        self.assertEqual(set(self.headers(self.root)), set(old))
        self.retarget("b.safetensors")
        self.index(dict.fromkeys(new, "linked.safetensors"))
        self.assertEqual(set(self.headers(self.root)), set(new))

    def test_explicit_rank_and_unindexed_files_keep_original_contract(self):
        expected = write_shard(self.root / "model.rank0.safetensors", 8)
        got = self.headers(self.root, rank=0)
        self.assertEqual(got, {name: dict(info, split=True) for name, info in expected.items()})
        with self.assertRaisesRegex(ValueError, "another rank"):
            self.headers(self.root, rank=1)
        got = self.headers(self.root, files=[self.root / "model.rank0.safetensors"])
        self.assertEqual(got, {name: dict(info, split=True) for name, info in expected.items()})


if __name__ == "__main__":
    unittest.main()
