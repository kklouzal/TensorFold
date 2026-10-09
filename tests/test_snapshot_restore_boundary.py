"""Real persisted bytes cross the complete stdlib restore boundary first."""

import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest

from snapshot_fd_transport import substitute_owners

from tensorfold.cuda.tensor_file import read_header
from tensorfold.engine.snapshot_registry import LayerSchema, Registry, TensorSchema
from tensorfold.engine.snapshot_restore import restore_snapshot
from tensorfold.engine.snapshot_integrity import FIELD, ZERO, seal_snapshot


class Cache:
    def __init__(self):
        self.keys = None
        self.offset = 0
        self.states = [None, None]
        self.window = 8


class Tensor:
    def __init__(self, dtype, shape):
        self.dtype, self.shape = dtype, shape


class RestoreControls(unittest.TestCase):
    def setUp(self):
        substitute_owners(self)

    def fixture(self):
        registry = Registry(
            (LayerSchema(Cache(), {"keys": TensorSchema(("F32",), (1, None, 2), 16)}, {"offset": (0, 8)}),),
            token_limit=8,
            token_id_limit=256,
            tensor_byte_limit=64,
            sizes={"F32": 4},
        )
        metadata = {
            "format": "2",
            "model": "content-v2",
            "tokens": "[1,2]",
            "layers": json.dumps(
                [
                    {
                        "class": registry.layers[0].class_id,
                        "plain": {"offset": 2, "window": 8},
                        "arrays": ["keys"],
                        "numpy": [],
                        "lists": {"states": {"length": 2, "slots": []}},
                    }
                ]
            ),
        }
        return registry, {
            "__metadata__": metadata,
            "0.keys": {"dtype": "F32", "shape": [1, 2, 2], "data_offsets": [0, 16]},
        }

    def write(self, path, header):
        header["__metadata__"][FIELD] = ZERO
        raw = json.dumps(header).encode("utf-8")
        end = max((v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"), default=0)
        path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(end))
        seal_snapshot(path, sizes={"F32": 4}, max_bytes=1 << 20)

    def load(self, path, registry, *, mutate=None, expected_tokens=None):
        seen = []

        def loader(private):
            seen.append(private)
            self.assertTrue(private.is_file())
            self.assertNotEqual(private, path)
            self.assertEqual(private.read_bytes(), path.read_bytes())
            self.assertEqual(private.stat().st_mode & 0o777, 0o600)
            _, header = read_header(private, registry.sizes)
            arrays = {k: Tensor(v["dtype"], v["shape"]) for k, v in header.items() if k != "__metadata__"}
            metadata = copy.deepcopy(header["__metadata__"])
            if mutate is not None:
                mutate(arrays, metadata)
            return arrays, metadata

        result = restore_snapshot(
            path,
            "content-v2",
            registry,
            load_tensors=loader,
            describe_tensor=lambda a: {"dtype": a.dtype, "shape": a.shape},
            convert_numpy=lambda a: a,
            expected_tokens=expected_tokens,
        )
        self.assertTrue(seen)
        self.assertFalse(seen[0].exists())
        return result

    def test_private_whole_file_load_then_initialized_distinct_restore(self):
        registry, header = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.safetensors"
            self.write(path, header)
            tokens, [cache] = self.load(path, registry, expected_tokens=[1, 2])
            self.assertTrue(path.exists())
        self.assertEqual(tokens, [1, 2])
        self.assertIs(type(cache), Cache)
        self.assertEqual(cache.keys.shape, [1, 2, 2])
        self.assertEqual(cache.offset, 2)
        self.assertEqual(cache.states, [None, None])
        self.assertIsNot(cache, registry.layers[0].prototype)

    def test_untrusted_schema_model_tokens_lists_and_shapes_never_call_native_loader(self):
        def layer(change):
            def apply(header):
                value = json.loads(header["__metadata__"]["layers"])
                change(value[0])
                header["__metadata__"]["layers"] = json.dumps(value)

            return apply

        cases = [
            layer(lambda e: e.update({"class": "untrusted_snapshot_code:Attack"})),
            layer(lambda e: e["plain"].update({"foreign": 1})),
            layer(lambda e: e["lists"]["states"].update({"length": 10**50})),
            layer(lambda e: e["lists"]["states"].update({"slots": [2]})),
            lambda h: h["__metadata__"].update({"format": "1"}),
            lambda h: h["__metadata__"].update({"model": "other-model"}),
            lambda h: h["__metadata__"].update({"tokens": "[true]"}),
            lambda h: h["__metadata__"].update({"tokens": "[256]"}),
            lambda h: h["0.keys"].update({"shape": [2, 1, 2]}),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.safetensors"
            for change in cases:
                registry, header = self.fixture()
                change(header)
                self.write(path, header)
                with self.subTest(change=change), self.assertRaises(ValueError):
                    restore_snapshot(
                        path,
                        "content-v2",
                        registry,
                        load_tensors=lambda _: self.fail("native loader called"),
                        describe_tensor=lambda _: self.fail("descriptor called"),
                        convert_numpy=lambda _: self.fail("converter called"),
                    )

    def test_expected_disk_prefix_replacement_is_rejected_before_native_load(self):
        registry, header = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.safetensors"
            self.write(path, header)
            with self.assertRaises(ValueError):
                restore_snapshot(
                    path,
                    "content-v2",
                    registry,
                    expected_tokens=[1, 3],
                    load_tensors=lambda _: self.fail("native loader called"),
                    describe_tensor=lambda _: self.fail("descriptor called"),
                    convert_numpy=lambda _: self.fail("converter called"),
                )

    def test_loaded_metadata_and_shape_must_still_equal_prevalidated_private_header(self):
        cases = (
            lambda a, m: m.update({"tokens": "[3,4]"}),
            lambda a, m: a["0.keys"].shape.append(1),
            lambda a, m: a.pop("0.keys"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.safetensors"
            for mutate in cases:
                registry, header = self.fixture()
                self.write(path, header)
                with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                    self.load(path, registry, mutate=mutate)

    def test_loader_interruption_releases_private_copy_and_preserves_source_and_primary(self):
        registry, header = self.fixture()
        seen = []
        primary = KeyboardInterrupt("owned native read interrupted")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.safetensors"
            self.write(path, header)
            before = path.read_bytes()

            def loader(private):
                seen.append(private)
                self.assertTrue(private.exists())
                raise primary

            with self.assertRaises(KeyboardInterrupt) as caught:
                restore_snapshot(
                    path,
                    "content-v2",
                    registry,
                    load_tensors=loader,
                    describe_tensor=lambda _: None,
                    convert_numpy=lambda _: None,
                )
            self.assertIs(caught.exception, primary)
            self.assertFalse(seen[0].exists())
            self.assertEqual(path.read_bytes(), before)

    def test_truncated_header_payload_and_symlink_refuse_before_native_load(self):
        registry, header = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.safetensors"
            self.write(path, header)
            valid = path.read_bytes()
            alias = Path(directory) / "alias.safetensors"
            alias.symlink_to(path)
            for target, content in ((path, valid[:7]), (path, valid[:-1]), (alias, valid)):
                path.write_bytes(content)
                with self.subTest(target=target, size=len(content)), self.assertRaises((ValueError, OSError)):
                    restore_snapshot(
                        target,
                        "content-v2",
                        registry,
                        load_tensors=lambda _: self.fail("native loader called"),
                        describe_tensor=lambda _: None,
                        convert_numpy=lambda _: None,
                    )


if __name__ == "__main__":
    unittest.main()
