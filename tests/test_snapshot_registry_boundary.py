"""Actual registry/full header boundary under stdlib scalar tensor seams."""

import copy
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

from dataclasses import replace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tensorfold.engine.snapshot_registry import LayerSchema, Registry, TensorSchema  # noqa: E402
from tensorfold.cuda.tensor_file import read_header  # noqa: E402 - direct source control


class Cache:
    transient = ("pending",)

    def __init__(self):
        self.keys = None
        self.offset = 0
        self.states = [None, None]
        self.window = 8
        self.pending = None


class Array:
    def __init__(self, dtype, shape):
        self.dtype, self.shape = dtype, shape


class Controls(unittest.TestCase):
    def setUp(self):
        from snapshot_fd_transport import substitute_owners

        substitute_owners(self)

    def fixture(self):
        prototype = Cache()
        registry = Registry(
            (
                LayerSchema(
                    prototype,
                    {
                        "keys": TensorSchema(("F32",), (1, 2, None, 4), 64),
                        "states.0": TensorSchema(("F32",), (1, 2), 2),
                        "states.1": TensorSchema(("F32",), (1, 2), 2),
                    },
                    {"offset": (0, 8)},
                ),
            ),
            token_limit=8,
            token_id_limit=256,
            tensor_byte_limit=272,
            sizes={"F32": 4},
        )
        metadata = {
            "format": "2",
            "model": "owned-v2",
            "tokens": "[1,2]",
            "layers": json.dumps(
                [
                    {
                        "class": f"{Cache.__module__}:{Cache.__qualname__}",
                        "plain": {"offset": 2, "window": 8},
                        "arrays": ["keys"],
                        "numpy": [],
                        "lists": {"states": {"length": 2, "slots": [0]}},
                    }
                ]
            ),
        }
        header = {
            "__metadata__": metadata,
            "0.keys": {"dtype": "F32", "shape": [1, 2, 2, 4], "data_offsets": [0, 64]},
            "0.states.0": {"dtype": "F32", "shape": [1, 2], "data_offsets": [64, 72]},
        }
        return prototype, registry, metadata, header

    def invoke(self, registry, metadata, header, *, mutate=None):
        calls = []

        def loader():
            calls.append("loader")
            arrays = {
                name: Array(value["dtype"], value["shape"]) for name, value in header.items() if name != "__metadata__"
            }
            returned = copy.deepcopy(metadata)
            if mutate is not None:
                mutate(metadata, header, arrays, returned)
            return arrays, returned

        result = registry.load(
            metadata,
            header,
            model_id="owned-v2",
            tensor_loader=loader,
            describe_tensor=lambda a: {"dtype": a.dtype, "shape": a.shape},
            convert_numpy=lambda a: ("host", a),
        )
        return result, calls

    def test_whole_valid_file_header_precedes_loader_initialized_restore_and_no_shared_lists(self):
        prototype, registry, metadata, header = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "owned.safetensors"
            raw = json.dumps(header).encode()
            path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(72))
            _, actual = read_header(path, {"F32": 4})
            (tokens, [cache]), calls = self.invoke(registry, metadata, actual)
        self.assertEqual(tokens, [1, 2])
        self.assertEqual(calls, ["loader"])
        self.assertIs(type(cache), Cache)
        self.assertEqual(cache.offset, 2)
        self.assertIsNone(cache.pending)
        self.assertEqual(cache.keys.shape, [1, 2, 2, 4])
        self.assertIsNone(cache.states[1])
        self.assertIsNot(cache.states, prototype.states)
        self.assertEqual(prototype.states, [None, None])

    def test_foreign_class_unknown_fields_huge_list_badslots_and_tokens_refuse_before_loader(self):
        cases = []

        def layer(change):
            def apply(metadata, header):
                value = json.loads(metadata["layers"])
                change(value[0])
                metadata["layers"] = json.dumps(value)

            return apply

        cases.extend(
            [
                layer(lambda v: v.update({"class": "untrusted_exec_module:Attack"})),
                layer(lambda v: v["plain"].update({"__class__": "Attack"})),
                layer(lambda v: v["lists"]["states"].update({"length": 10**100})),
                layer(lambda v: v["lists"]["states"].update({"slots": [2]})),
                layer(lambda v: v["lists"]["states"].update({"slots": [0, 0]})),
                layer(lambda v: v["lists"]["states"].update({"slots": [False]})),
                layer(lambda v: v["plain"].update({"window": 10**100})),
                layer(lambda v: v["plain"].update({"offset": True})),
                layer(lambda v: v["arrays"].append("keys")),
                layer(lambda v: v["plain"].pop("window")),
                lambda m, h: m.update({"tokens": "[true]"}),
                lambda m, h: m.update({"tokens": "[256]"}),
                lambda m, h: m.update({"tokens": "[NaN]"}),
                lambda m, h: m.update({"format": "1"}),
                lambda m, h: m.update({"model": "foreign"}),
                lambda m, h: h["0.keys"].update({"shape": [1, 3, 2, 4]}),
                lambda m, h: h["0.keys"].update({"shape": [1, 2, 100, 4]}),
                lambda m, h: h["0.keys"].update({"dtype": "F16"}),
                lambda m, h: h.update({"0.foreign": h["0.keys"]}),
            ]
        )
        for change in cases:
            with self.subTest(change=change):
                _, registry, metadata, header = self.fixture()
                change(metadata, header)
                with self.assertRaises(ValueError):
                    registry.load(
                        metadata,
                        header,
                        model_id="owned-v2",
                        tensor_loader=lambda: self.fail("foreign loader called"),
                        describe_tensor=lambda _: self.fail("descriptor called"),
                        convert_numpy=lambda _: self.fail("converter called"),
                    )
        self.assertNotIn("untrusted_exec_module", sys.modules)

    def test_foreign_callback_mutation_cannot_rewrite_validated_metadata_or_list_bound(self):
        prototype, registry, metadata, header = self.fixture()

        def mutate(meta, header, arrays, returned):
            prototype.states.extend([None] * 100)
            meta["tokens"] = "[255]"
            meta["layers"] = "[]"

        (tokens, [cache]), _ = self.invoke(registry, metadata, header, mutate=mutate)
        self.assertEqual(tokens, [1, 2])
        self.assertEqual(len(cache.states), 2)

    def test_loaded_metadata_or_tensor_geometry_mismatch_refuses_before_restore(self):
        for change in (
            lambda m, h, a, r: r.update({"model": "foreign"}),
            lambda m, h, a, r: a["0.keys"].shape.append(1),
            lambda m, h, a, r: a.pop("0.keys"),
        ):
            _, registry, metadata, header = self.fixture()
            with self.assertRaises(ValueError):
                self.invoke(registry, metadata, header, mutate=change)

    def test_explicit_numpy_conversion_callback_preserves_declared_field_and_array_identity(self):
        _, registry, metadata, header = self.fixture()
        layers = json.loads(metadata["layers"])
        layers[0]["arrays"] = []
        layers[0]["numpy"] = ["keys"]
        registry = Registry(
            (replace(registry.layers[0], numpy_fields=frozenset({"keys"})),),
            token_limit=8,
            token_id_limit=256,
            tensor_byte_limit=272,
            sizes={"F32": 4},
        )
        metadata["layers"] = json.dumps(layers)
        (_, [cache]), calls = self.invoke(registry, metadata, header)
        self.assertEqual(calls, ["loader"])
        self.assertEqual(cache.keys[0], "host")
        self.assertEqual(cache.keys[1].shape, [1, 2, 2, 4])


class DeclaredRoles(unittest.TestCase):
    def fixture(self):
        class Head:
            def __init__(self):
                self.side = None
                self.chaining = False
                self.history = None
                self.drafted = 0

        schema = LayerSchema(
            Head(),
            {
                "side.0": TensorSchema(("BF16",), (1, 2, None, 4), 64),
                "side.1": TensorSchema(("BF16",), (1, 2, None, 4), 64),
                "side.2": TensorSchema(("BF16",), (1, None, 4), 32),
                "history": TensorSchema(("I64",), (1, 3), 3),
            },
            {"drafted": (0, 8)},
            list_lengths={"side": 3},
            required_list_slots={"side": frozenset({0, 1, 2})},
            numpy_fields=frozenset({"history"}),
            mutable_flags=frozenset({"chaining"}),
        )
        registry = Registry(
            (schema,), token_limit=8, token_id_limit=256, tensor_byte_limit=1024, sizes={"BF16": 2, "I64": 8}
        )
        entry = {
            "class": registry.layers[0].class_id,
            "plain": {"side": None, "chaining": True, "drafted": 2, "history": None},
            "arrays": [],
            "numpy": [],
            "lists": {},
        }
        metadata = {"format": "2", "model": "head", "tokens": "[1,2]", "layers": json.dumps([entry])}
        return registry, entry, metadata, {"__metadata__": metadata}

    def test_nullable_side_declared_three_slots_and_boolean_preserve(self):
        registry, entry, metadata, header = self.fixture()
        tokens, _ = registry.validate(metadata, header, model_id="head")
        self.assertEqual(tokens, [1, 2])
        entry["plain"].pop("side")
        entry["lists"]["side"] = {"length": 3, "slots": [0, 1, 2]}
        for slot, shape in enumerate(([1, 2, 2, 4], [1, 2, 2, 4], [1, 2, 4])):
            header[f"0.side.{slot}"] = {"dtype": "BF16", "shape": shape}
        metadata["layers"] = json.dumps([entry])
        _, layers = registry.validate(metadata, header, model_id="head")
        arrays = {
            name: Array(value["dtype"], value["shape"]) for name, value in header.items() if name != "__metadata__"
        }
        [restored] = registry.restore(layers, arrays, convert_numpy=lambda a: ("numpy", a))
        self.assertTrue(restored.chaining)
        self.assertEqual(restored.drafted, 2)
        self.assertEqual(len(restored.side), 3)
        self.assertIs(restored.side[2], arrays["0.side.2"])
        self.assertIsNone(registry.layers[0].prototype.side)

    def test_list_capacity_missing_slot_and_flag_int_refuse(self):
        for bad in ("capacity", "missing", "flag"):
            registry, entry, metadata, header = self.fixture()
            entry["plain"].pop("side")
            entry["lists"]["side"] = {"length": 3, "slots": [0, 1, 2]}
            for slot, shape in enumerate(([1, 2, 2, 4], [1, 2, 2, 4], [1, 2, 4])):
                header[f"0.side.{slot}"] = {"dtype": "BF16", "shape": shape}
            if bad == "capacity":
                entry["lists"]["side"]["length"] = 4
            elif bad == "missing":
                entry["lists"]["side"]["slots"].pop()
                header.pop("0.side.2")
            else:
                entry["plain"]["chaining"] = 1
            metadata["layers"] = json.dumps([entry])
            with self.assertRaises(ValueError):
                registry.load(
                    metadata,
                    header,
                    model_id="head",
                    tensor_loader=lambda: self.fail("loader reached"),
                    describe_tensor=lambda a: {},
                    convert_numpy=lambda a: a,
                )

    def test_numpy_role_cannot_be_laundered_into_gpu_array(self):
        for mode in ("arrays", "numpy"):
            registry, entry, metadata, header = self.fixture()
            entry["plain"].pop("history")
            entry[mode].append("history")
            header["0.history"] = {"dtype": "I64", "shape": [1, 3]}
            metadata["layers"] = json.dumps([entry])
            if mode == "arrays":
                with self.assertRaisesRegex(ValueError, "representation"):
                    registry.validate(metadata, header, model_id="head")
            else:
                registry.validate(metadata, header, model_id="head")

    def test_required_tensor_cannot_be_replaced_by_none(self):
        registry, entry, metadata, header = self.fixture()
        registry = Registry(
            (replace(registry.layers[0], required_tensors=frozenset({"history"})),),
            token_limit=8,
            token_id_limit=256,
            tensor_byte_limit=1024,
            sizes={"BF16": 2, "I64": 8},
        )
        with self.assertRaisesRegex(ValueError, "required cache tensor"):
            registry.validate(metadata, header, model_id="head")

    def test_mutable_schema_containers_and_invalid_byte_width_refuse(self):
        registry, _, _, _ = self.fixture()
        base = registry.layers[0]
        for tensor in (TensorSchema(["BF16"], (1,), 1), TensorSchema(("BF16",), [1], 1)):
            with self.assertRaises(ValueError):
                Registry(
                    (replace(base, tensors={"history": tensor}),),
                    token_limit=8,
                    token_id_limit=256,
                    tensor_byte_limit=1024,
                    sizes={"BF16": 2},
                )
        with self.assertRaises(ValueError):
            Registry((base,), token_limit=8, token_id_limit=256, tensor_byte_limit=1024, sizes={"BF16": True, "I64": 8})


class SignedZeroConfiguration(unittest.TestCase):
    def test_clone_and_persisted_scalar_preserve_negative_zero(self):
        class Floating:
            def __init__(self):
                self.scale = -0.0

            def __copy__(self):
                clone = type(self)()
                if getattr(self, "wrong_sign", False):
                    clone.scale = 0.0
                return clone

        initial = Floating()
        registry = Registry(
            (LayerSchema(initial, {}, {}),), token_limit=0, token_id_limit=1, tensor_byte_limit=0, sizes={"F32": 4}
        )
        entry = {"class": registry.layers[0].class_id, "plain": {"scale": -0.0}, "arrays": [], "numpy": [], "lists": {}}
        metadata = {"format": "2", "model": "float", "tokens": "[]", "layers": json.dumps([entry])}
        registry.validate(metadata, {"__metadata__": metadata}, model_id="float")
        entry["plain"]["scale"] = 0.0
        metadata["layers"] = json.dumps([entry])
        with self.assertRaises(ValueError):
            registry.validate(metadata, {"__metadata__": metadata}, model_id="float")
        Floating.wrong_sign = True
        with self.assertRaises(ValueError):
            Registry(
                (LayerSchema(initial, {}, {}),), token_limit=0, token_id_limit=1, tensor_byte_limit=0, sizes={"F32": 4}
            )


class CloneOwnership(unittest.TestCase):
    def fixture(self, kind):
        class Owned(Cache):
            mode = "normal"

            def __copy__(self):
                if self.mode == "self":
                    return self
                if self.mode == "type":
                    return Cache()
                clone = type(self)()
                vars(clone).update(vars(self))
                if self.mode == "dict":
                    clone.__dict__ = self.__dict__
                elif self.mode == "missing":
                    del clone.window
                elif self.mode == "config":
                    clone.window = 99
                return clone

        original = Owned()
        original.mode = kind
        return original, LayerSchema(original, {"keys": TensorSchema(("F32",), (1,), 1)}, {"offset": (0, 8)})

    def test_bad_capture_copy_never_mutates_or_narrows_caller_authority(self):
        for kind in ("self", "type", "dict", "missing", "config"):
            with self.subTest(kind=kind):
                original, layer = self.fixture(kind)
                before = dict(vars(original))
                with self.assertRaises(ValueError):
                    Registry((layer,), token_limit=8, token_id_limit=16, tensor_byte_limit=4, sizes={"F32": 4})
                self.assertEqual(vars(original), before)

    def test_bad_restore_copy_never_updates_registry_or_caller_state(self):
        for kind in ("self", "type", "dict", "missing", "config"):
            with self.subTest(kind=kind):
                original, layer = self.fixture("normal")
                registry = Registry((layer,), token_limit=8, token_id_limit=16, tensor_byte_limit=4, sizes={"F32": 4})
                registry.layers[0].prototype.mode = kind
                entry = {"plain": {"offset": 3}, "arrays": [], "numpy": [], "lists": {}}
                with self.assertRaises(ValueError):
                    registry.restore([entry], {}, convert_numpy=lambda a: a)
                self.assertEqual(original.offset, 0)
                self.assertEqual(registry.layers[0].prototype.offset, 0)


class CallbackMutation(unittest.TestCase):
    def case(self, mutate):
        class Mutating(Cache):
            def __copy__(self):
                if mutate[0]:
                    self.window = 99
                clone = type(self)()
                vars(clone).update(vars(self))
                return clone

        original = Mutating()
        layer = LayerSchema(original, {}, {"offset": (0, 8)})
        return original, layer

    def test_authority_captured_before_copy_and_failed_caller_prototype_requires_retirement(self):
        original, layer = self.case([True])
        with self.assertRaisesRegex(ValueError, "mutated authority"):
            Registry((layer,), token_limit=8, token_id_limit=16, tensor_byte_limit=0, sizes={"F32": 4})
        self.assertEqual(original.window, 99)  # Arbitrary trusted callback mutation is detected, not undone.

    def test_failed_restore_retires_registry_and_never_revalidates_mutated_authority(self):
        enabled = [False]
        original, layer = self.case(enabled)
        registry = Registry((layer,), token_limit=8, token_id_limit=16, tensor_byte_limit=0, sizes={"F32": 4})
        enabled[0] = True
        entry = {"plain": {"offset": 3}, "arrays": [], "numpy": [], "lists": {}}
        with self.assertRaisesRegex(ValueError, "mutated authority"):
            registry.restore([entry], {}, convert_numpy=lambda a: a)
        self.assertEqual(original.window, 8)
        self.assertEqual(registry.layers[0].prototype.window, 99)
        self.assertTrue(registry._unusable)
        with self.assertRaisesRegex(RuntimeError, "retired"):
            registry.restore([entry], {}, convert_numpy=lambda a: a)
        with self.assertRaisesRegex(RuntimeError, "retired"):
            registry.validate({}, {}, model_id="any")


if __name__ == "__main__":
    unittest.main()
