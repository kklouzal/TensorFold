"""Full canonical prefix API with real files and opaque tensor-runtime seams."""

import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from snapshot_fd_transport import substitute_owners

from tensorfold.cuda.tensor_file import read_header
from tensorfold.engine.snapshot_builtin import _token_timeline
from tensorfold.engine.snapshot_registry import LayerSchema, Registry, TensorSchema


class Cache:
    def __init__(self):
        self.keys = None
        self.offset = 0
        self.states = [None, None]


class Tensor:
    def __init__(self, shape):
        self.shape, self.dtype = list(shape), "F32"


class Codec:
    def __init__(self):
        self.reads, self.writes, self.private = [], [], []

    def kind(self, value):
        return "array" if type(value) is Tensor else None

    def describe(self, value):
        return {"dtype": value.dtype, "shape": list(value.shape)} if type(value) is Tensor else None

    def host_array(self, value):
        raise AssertionError("undeclared host array")

    def write(self, path, payload):
        self.writes.append(path)
        header, count = {"__metadata__": payload.metadata}, 0
        for name, value in payload.arrays.items():
            size = 4
            for dim in value.shape:
                size *= dim
            header[name] = {**self.describe(value), "data_offsets": [count, count + size]}
            count += size
        raw = json.dumps(header).encode()
        path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(count))

    def load(self, path):
        self.reads.append(path)
        self.private.append(path)
        self.assert_private(path)
        _, header = read_header(path, {"F32": 4})
        return {name: Tensor(value["shape"]) for name, value in header.items() if name != "__metadata__"}, header[
            "__metadata__"
        ]

    @staticmethod
    def assert_private(path):
        if path.stat().st_mode & 0o777 != 0o600 or not path.is_file():
            raise AssertionError("not an owned private regular input")


def source_module():
    path = Path(__file__).resolve().parents[1] / "src/tensorfold/engine/prefix_snapshots.py"
    spec = importlib.util.spec_from_file_location("owned_prefix_protocol", path)
    module = importlib.util.module_from_spec(spec)
    mlx, core, numpy = ModuleType("mlx"), ModuleType("mlx.core"), ModuleType("numpy")
    mlx.core = core
    with patch.dict(sys.modules, {"mlx": mlx, "mlx.core": core, "numpy": numpy}):
        spec.loader.exec_module(module)
    return module


class PrefixControls(unittest.TestCase):
    def test_malformed_json_suppressed_parser_context_remains_cache_miss_after_successful_cleanup(self):
        from snapshot_fd_transport import TransportOwner

        descriptors = []

        class Slot(TransportOwner):
            def open(self, *args, **kwargs):
                super().open(*args, **kwargs)
                descriptors.append(self.fileno())

        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "malformed.safetensors"
            raw = b'{"bad":'
            path.write_bytes(struct.pack("<Q", len(raw)) + raw)
            with patch("tensorfold.file_io._owned_slot", Slot):
                self.assertIsNone(self.load(path, "current"))
        self.assertEqual(len(descriptors), 1)
        self.assertEqual(self.codec.reads, [])
        for descriptor in descriptors:
            with self.assertRaises(OSError):
                os.fstat(descriptor)

    def test_malformed_json_and_real_consumed_close_failure_preserve_both_errors(self):
        from snapshot_fd_transport import TransportOwner

        cleanup, descriptors = OSError("consumed parser FD close"), []

        class Slot(TransportOwner):
            def open(self, *args, **kwargs):
                super().open(*args, **kwargs)
                descriptors.append(self.fileno())

            def close(self):
                was_closed = self.closed
                super().close()
                if not was_closed:
                    raise cleanup

        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "malformed.safetensors"
            raw = b'{"bad":'
            path.write_bytes(struct.pack("<Q", len(raw)) + raw)
            with patch("tensorfold.file_io._owned_slot", Slot):
                with self.assertRaises(json.JSONDecodeError) as caught:
                    self.load(path, "current")
        cause = BaseException.__cause__.__get__(caught.exception)
        causes = cause.exceptions if isinstance(cause, BaseExceptionGroup) else (cause,)
        self.assertTrue(any(error is cleanup for error in causes))
        self.assertTrue(any(isinstance(error, StopIteration) for error in causes))
        self.assertTrue(caught.exception.__notes__)
        self.assertEqual(self.codec.reads, [])
        self.assertEqual(len(descriptors), 1)
        for descriptor in descriptors:
            with self.assertRaises(OSError):
                os.fstat(descriptor)

    def test_early_source_and_target_close_failures_survive_final_drain_classification(self):
        from snapshot_fd_transport import TransportOwner

        for target in ("source", "target"):
            for kind in (ValueError, OSError):
                with self.subTest(target=target, kind=kind):
                    cleanup = kind("consumed early " + target + " close")
                    opened = []

                    class Owner(TransportOwner):
                        def open(self, path, flags, mode=0o600):
                            super().open(path, flags, mode)
                            self.role = "target" if flags & os.O_WRONLY else "source"
                            self.borrowed_fd = self.fileno()
                            opened.append(self)

                        def close(self):
                            was_closed = self.closed
                            super().close()
                            if not was_closed and self.role == target:
                                raise cleanup

                    with tempfile.TemporaryDirectory() as directory:
                        path = self.save(Path(directory), "current", [1, 2])
                        before = len(self.codec.reads)
                        with patch("tensorfold.file_io._owned_slot", Owner):
                            with self.assertRaises(kind) as caught:
                                self.load(path, "current")
                        self.assertIs(caught.exception, cleanup)
                        self.assertTrue(cleanup.__notes__)
                        self.assertEqual(len(self.codec.reads), before)
                        self.assertEqual([owner.role for owner in opened], ["source", "target"])
                        for owner in opened:
                            self.assertTrue(owner.closed)
                            with self.assertRaises(OSError):
                                os.fstat(owner.borrowed_fd)

    def test_value_error_from_consumed_close_is_operation_failure_not_cache_miss(self):
        from snapshot_fd_transport import TransportOwner

        cleanup = ValueError("owned close failure")

        class Owner(TransportOwner):
            def close(self):
                super().close()
                raise cleanup

        with tempfile.TemporaryDirectory() as directory:
            path = self.save(Path(directory), "current", [1, 2])
            with patch("tensorfold.file_io._owned_slot", Owner):
                with self.assertRaises(ValueError) as caught:
                    self.load(path, "current")
            self.assertIs(caught.exception, cleanup)
            self.assertTrue(cleanup.__notes__)

    def test_miss_bypasses_foreign_cause_note_and_dictionary_lookup_hooks(self):
        class Foreign(ValueError):
            def __getattribute__(self, name):
                if name in ("__cause__", "__notes__", "__dict__"):
                    raise LookupError("foreign hook invoked")
                return super().__getattribute__(name)

        error = Foreign("bad data")
        self.api._miss(error)
        BaseException.__dict__["__dict__"].__get__(error)["__notes__"] = ["cleanup failed"]
        with self.assertRaises(Foreign) as caught:
            self.api._miss(error)
        self.assertIs(caught.exception, error)

        class BadTruth:
            def __bool__(self):
                raise LookupError("foreign truth hook invoked")

        BaseException.__dict__["__dict__"].__get__(error)["__notes__"] = BadTruth()
        with self.assertRaises(Foreign) as caught:
            self.api._miss(error)
        self.assertIs(caught.exception, error)

    def test_unchanged_reuse_memo_avoids_full_reads_and_corruption_rebuilds(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            path = self.save(root, "current", [1, 2])
            with patch.object(self.api, "verify_snapshot", side_effect=AssertionError("unchanged snapshot rehashed")):
                for _ in range(4):
                    self.assertIsNone(self.save(root, "current", [1, 2]))
            raw = bytearray(path.read_bytes())
            raw[-1] ^= 1  # a schema-valid changed tensor word invalidates receipt identity
            path.write_bytes(raw)
            reads = len(self.codec.reads)
            self.assertIsNone(self.load(path, "current"))
            self.assertEqual(len(self.codec.reads), reads)
            self.assertIsNotNone(self.save(root, "current", [1, 2]))
            self.assertEqual(self.load(path, "current")[0], [1, 2])
            self.assertEqual(len(self.codec.writes), 2)

    def setUp(self):
        substitute_owners(self)
        self.api = source_module()
        self.codec = Codec()
        self.registry = Registry(
            (
                LayerSchema(
                    Cache(),
                    {"keys": TensorSchema(("F32",), (1, None, 2), 32)},
                    {"offset": (0, 16)},
                    token_invariants=(_token_timeline(True),),
                ),
            ),
            token_limit=16,
            token_id_limit=256,
            tensor_byte_limit=128,
            sizes={"F32": 4},
        )

    def cache(self, tokens):
        cache = Cache()
        cache.offset, cache.keys = len(tokens), Tensor([1, len(tokens), 2])
        return [cache]

    def save(self, root, model, tokens, **kwargs):
        return self.api.save_snapshot(
            root, model, tokens, self.cache(tokens), registry=self.registry, codec=self.codec, **kwargs
        )

    def load(self, path, model, **kwargs):
        return self.api.load_snapshot(path, model, registry=self.registry, codec=self.codec, **kwargs)

    def rewrite(self, path, change):
        raw = path.read_bytes()
        count = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8 : 8 + count])
        change(header)
        new = json.dumps(header).encode()
        path.write_bytes(struct.pack("<Q", len(new)) + new + raw[8 + count :])

    def test_format2_roundtrip_all_none_slots_and_valid_existing_skip(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.save(Path(root), "model-content-v1:fixture|kernels=a", [1, 2, 3])
            tokens, [cache] = self.load(path, "model-content-v1:fixture|kernels=a")
            self.assertEqual(tokens, [1, 2, 3])
            self.assertEqual(cache.offset, 3)
            self.assertEqual(cache.states, [None, None])
            self.assertIsNot(cache.states, self.registry.layers[0].prototype.states)
            self.assertIsNone(self.save(Path(root), "model-content-v1:fixture|kernels=a", tokens))
            self.assertEqual(len(self.codec.writes), 1)
            self.assertTrue(all(not private.exists() for private in self.codec.private))

    def test_legacy_and_foreign_class_are_misses_before_load_then_atomically_rebuild(self):
        with tempfile.TemporaryDirectory() as root:
            for bad in ("format", "class", "offset"):
                self.save(Path(root), "current", [1, 2])
                path = Path(root) / (self.api.snapshot_key("current", [1, 2]) + ".safetensors")

                def change(header):
                    metadata = header["__metadata__"]
                    if bad == "format":
                        metadata["format"] = "1"
                    else:
                        entries = json.loads(metadata["layers"])
                        if bad == "class":
                            entries[0]["class"] = "untrusted_snapshot_exec:Attack"
                        else:
                            entries[0]["plain"]["offset"] = 1
                        metadata["layers"] = json.dumps(entries)

                self.rewrite(path, change)
                reads = len(self.codec.reads)
                self.assertIsNone(self.load(path, "current"))
                self.assertEqual(len(self.codec.reads), reads)
                self.assertIsNotNone(self.save(Path(root), "current", [1, 2]))
                self.assertEqual(self.load(path, "current")[0], [1, 2])
        self.assertNotIn("untrusted_snapshot_exec", sys.modules)

    def test_disk_selection_replacement_with_preserved_mtime_is_revalidated_and_bound(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            first = self.save(root, "current", [1, 2])
            blocks = self.api.DiskBlocks(root, "current", registry=self.registry)
            hit = blocks.best([1, 2, 3], 0)
            self.assertEqual(hit, (first, [1, 2]))
            timestamp = first.stat().st_mtime_ns
            second = self.save(root, "current", [4, 5])
            os.replace(second, first)
            os.utime(first, ns=(timestamp, timestamp))
            self.assertIsNone(self.load(first, "current", expected_tokens=hit[1]))
            self.assertIsNone(blocks.best([1, 2, 3], 0))
            self.assertEqual(blocks.best([4, 5, 6], 0), (first, [4, 5]))

    def test_warming_and_pruning_only_same_content_owner_validated_tokens(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            for tokens in ([1], [1, 2], [3, 4]):
                self.save(root, "model-content-v1:a|kernels=old", tokens, keep=2)
            self.save(root, "model-content-v1:b|kernels=old", [9], keep=1)
            warmed = self.api.blocks_to_warm(root, "model-content-v1:a|kernels=new", registry=self.registry)
            self.assertEqual({tuple(t) for t in warmed}, {(1, 2), (3, 4)})
            self.assertEqual(
                self.api.blocks_to_warm(root, "model-content-v1:c|kernels=new", registry=self.registry), []
            )
            loaded = list(
                self.api.load_snapshots(
                    root, "model-content-v1:a|kernels=old", registry=self.registry, codec=self.codec, limit=1
                )
            )
            self.assertEqual(len(loaded), 1)
            models = [self.api.read_metadata(path)["model"] for path in root.glob("*.safetensors")]
            self.assertEqual(models.count("model-content-v1:a|kernels=old"), 2)
            self.assertEqual(models.count("model-content-v1:b|kernels=old"), 1)

    def test_explicit_authority_and_constructor_shape_fail_before_publisher(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            with self.assertRaises(TypeError):
                self.api.load_snapshot(root / "unread", "current", registry=None, codec=self.codec)
            cache = self.cache([1, 2])
            cache[0].states.extend([None] * 20)
            with self.assertRaises(ValueError):
                self.api.save_snapshot(root, "current", [1, 2], cache, registry=self.registry, codec=self.codec)
            self.assertFalse(list(root.iterdir()))
            self.assertFalse(self.codec.writes)

    def test_cleanup_failure_context_is_not_a_success_shaped_cache_miss(self):
        primary, cleanup = ValueError("invalid data"), OSError("owned cleanup failed")
        primary.__cause__ = cleanup
        with patch.object(self.api, "restore_snapshot", side_effect=primary), self.assertRaises(ValueError) as caught:
            self.load(Path("unused"), "current")
        self.assertIs(caught.exception, primary)
        self.assertIs(caught.exception.__cause__, cleanup)


if __name__ == "__main__":
    unittest.main()
