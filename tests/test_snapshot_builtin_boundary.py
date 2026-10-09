"""Explicit builtin profile/type/shape controls; no numerical runtime."""

import ast
import gc
from pathlib import Path
import json
from types import ModuleType
import sys
import unittest
from unittest.mock import patch
import weakref

from tensorfold.engine.snapshot_builtin import Profile, UnregisteredCache, builtin_authorities


class Array:
    def __init__(self, dtype, shape):
        self.dtype, self.shape = dtype, shape


def describe(value):
    return {"dtype": value.dtype, "shape": value.shape} if type(value) is Array else None


class Attention:
    def __init__(self):
        self.keys = self.values = self.index_keys = self.pooled = None
        self.offset = 0

    step = 256


class Head(Attention):
    drafted, chaining, side, side_base = 0, False, None, 0


class Controls(unittest.TestCase):
    def test_empty_auxiliary_context_is_valid_but_partial_main_allocation_is_not(self):
        profile, observed = self.fixture(Head)
        profile.observe([observed])
        registry = profile.registry()
        entry, metadata, header = self.document(registry, head=True)
        entry["plain"].update(offset=0, drafted=0, side_base=0, side=None, keys=None, values=None, index_keys=None)
        entry["arrays"] = []
        entry["lists"] = {}
        header = {"__metadata__": metadata}
        metadata["layers"] = json.dumps([entry])
        registry.validate(metadata, header, model_id="current")
        for offset in (0, 1):
            entry["plain"]["offset"] = offset
            entry["plain"].pop("keys", None)
            entry["arrays"] = ["keys"]
            header["0.keys"] = {"dtype": "BF16", "shape": [1, 2, 256, 4]}
            metadata["layers"] = json.dumps([entry])
            with self.assertRaises(ValueError):
                registry.validate(metadata, header, model_id="current")
        entry["arrays"] = []
        entry["plain"]["keys"] = None
        header.pop("0.keys")
        metadata["layers"] = json.dumps([entry])
        with self.assertRaises(ValueError):
            registry.validate(metadata, header, model_id="current")

    def test_side_only_auxiliary_context_retains_exact_source_timeline(self):
        profile, observed = self.fixture(Head)
        observed.side = [Array("BF16", [1, 2, 2, 4]), Array("F32", [1, 2, 2, 5]), Array("BF16", [1, 2, 3])]
        profile.observe([observed])
        registry = profile.registry()
        entry, metadata, header = self.document(registry, head=True)
        entry["plain"].update(offset=2, side_base=0, keys=None, values=None, index_keys=None)
        entry["arrays"] = []
        for name in ("keys", "values", "index_keys"):
            header.pop("0." + name)
        metadata["layers"] = json.dumps([entry])
        registry.validate(metadata, header, model_id="current")
        entry["plain"]["side_base"] = 1
        metadata["layers"] = json.dumps([entry])
        with self.assertRaises(ValueError):
            registry.validate(metadata, header, model_id="current")

    def fixture(self, cls=Attention):
        layout = {"keys": 2, "values": 2, "index_keys": 1, "pooled": 1}
        defaults = {}
        if cls is Head:
            layout.update({"side.0": 2, "side.1": 2, "side.2": 1})
            defaults = {"drafted": 0, "chaining": False, "side": None, "side_base": 0}
        profile = Profile(
            [cls()],
            describe_tensor=describe,
            token_limit=128,
            token_id_limit=256,
            tensor_byte_limit=100000,
            max_draft=8,
            sizes={"BF16": 2, "F32": 4},
            authorities={cls: (layout, defaults)},
        )
        observed = cls()
        observed.keys = Array("BF16", [1, 2, 256, 4])
        observed.values = Array("F32", [1, 2, 256, 5])
        observed.index_keys = Array("BF16", [1, 256, 3])
        observed.offset = 64
        return profile, observed

    def document(self, registry, *, head=False):
        plain = {"offset": 64, "pooled": None}
        if head:
            plain.update({"drafted": 2, "chaining": False, "side_base": 62})
        entry = {
            "class": registry.layers[0].class_id,
            "plain": plain,
            "arrays": ["keys", "values", "index_keys"],
            "numpy": [],
            "lists": {},
        }
        header = {
            "0.keys": {"dtype": "BF16", "shape": [1, 2, 256, 4]},
            "0.values": {"dtype": "F32", "shape": [1, 2, 256, 5]},
            "0.index_keys": {"dtype": "BF16", "shape": [1, 256, 3]},
        }
        if head:
            entry["lists"]["side"] = {"length": 3, "slots": [0, 1, 2]}
            for i, shape in enumerate(([1, 2, 2, 4], [1, 2, 2, 5], [1, 2, 3])):
                header[f"0.side.{i}"] = {"dtype": "F32" if i == 1 else "BF16", "shape": shape}
        metadata = {"format": "2", "model": "current", "tokens": "[1,2]", "layers": json.dumps([entry])}
        header["__metadata__"] = metadata
        return entry, metadata, header

    def test_descriptors_not_tensor_storage_retained_and_growth_exact(self):
        profile, observed = self.fixture()
        refs = [weakref.ref(a) for a in (observed.keys, observed.values, observed.index_keys)]
        profile.observe([observed])
        observed.keys.shape[-1] = 99
        del observed
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))
        registry = profile.registry()
        self.assertEqual(registry.layers[0].tensors["keys"].shape, (1, 2, None, 4))
        self.assertEqual(registry.layers[0].tensors["values"].dtypes, ("F32",))
        self.assertEqual(registry.layers[0].tensors["pooled"].shape, (1, None, 3))
        self.assertIsNone(registry.layers[0].prototype.keys)
        _, metadata, header = self.document(registry)
        registry.validate(metadata, header, model_id="current")

    def test_explicit_head_defaults_side_roles_and_source_relations(self):
        profile, observed = self.fixture(Head)
        profile.observe([observed])
        registry = profile.registry()
        fields = registry.layers[0].fields()
        self.assertEqual(
            {name: fields[name] for name in ("drafted", "chaining", "side", "side_base")},
            {"drafted": 0, "chaining": False, "side": None, "side_base": 0},
        )
        entry, metadata, header = self.document(registry, head=True)
        registry.validate(metadata, header, model_id="current")
        header["0.side.1"]["shape"][2] = 1
        with self.assertRaisesRegex(ValueError, "timeline"):
            registry.validate(metadata, header, model_id="current")
        header["0.side.1"]["shape"][2] = 2
        entry["plain"]["side_base"] = 61
        metadata["layers"] = json.dumps([entry])
        with self.assertRaisesRegex(ValueError, "timeline"):
            registry.validate(metadata, header, model_id="current")

    def test_capacity_geometry_and_missing_populated_state_refuse(self):
        profile, observed = self.fixture()
        profile.observe([observed])
        registry = profile.registry()
        for bad in ("width", "dtype", "capacity", "offset", "missing"):
            entry, metadata, header = self.document(registry)
            if bad == "width":
                header["0.keys"]["shape"][-1] = 5
            elif bad == "dtype":
                header["0.values"]["dtype"] = "BF16"
            elif bad == "capacity":
                header["0.index_keys"]["shape"][1] = 255
            elif bad == "offset":
                for name, axis in (("keys", 2), ("values", 2), ("index_keys", 1)):
                    header["0." + name]["shape"][axis] = 32
            else:
                entry["arrays"].remove("keys")
                entry["plain"]["keys"] = None
                header.pop("0.keys")
                metadata["layers"] = json.dumps([entry])
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                registry.load(
                    metadata,
                    header,
                    model_id="current",
                    tensor_loader=lambda: self.fail("loader reached"),
                    describe_tensor=describe,
                    convert_numpy=lambda a: a,
                )

    def test_only_exact_already_loaded_builtin_classes_and_no_module_lookup_callbacks(self):
        module = ModuleType("tensorfold.families.qwen4_exp.model_layers")
        exact = type("AttentionCache", (), {"__module__": module.__name__})
        module.AttentionCache = exact
        module.__getattr__ = lambda name: self.fail("foreign module getattr executed")
        impostor = type("AttentionCache", (), {"__module__": module.__name__})
        with patch.dict(sys.modules, {module.__name__: module}):
            bindings = builtin_authorities()
            self.assertIn(exact, bindings)
            self.assertNotIn(impostor, bindings)
        with self.assertRaises(UnregisteredCache):
            Profile(
                [impostor()],
                describe_tensor=describe,
                token_limit=8,
                token_id_limit=64,
                tensor_byte_limit=256,
                max_draft=0,
                sizes={"F32": 4},
                authorities=bindings,
            )

    def test_bad_profile_count_class_rank_descriptor_refuses(self):
        for bad in ("missing", "class", "negative", "rank"):
            profile, observed = self.fixture()
            if bad == "missing":
                cache = []
            elif bad == "class":
                cache = [Head()]
            else:
                observed.keys.shape = [-1] if bad == "negative" else [1]
                cache = [observed]
            with self.assertRaises(ValueError):
                profile.observe(cache)
                profile.registry()

    def test_active_prototype_cannot_retain_tensor_or_cursor_authority(self):
        for tensor in (False, True):
            prototype = Attention()
            if tensor:
                prototype.keys = Array("BF16", [1, 2, 256, 4])
            else:
                prototype.offset = 64
            with self.assertRaisesRegex(ValueError, "fresh initialized"):
                Profile(
                    [prototype],
                    describe_tensor=describe,
                    token_limit=128,
                    token_id_limit=256,
                    tensor_byte_limit=100000,
                    max_draft=8,
                    sizes={"BF16": 2},
                    authorities={Attention: ({"keys": 2, "values": 2, "index_keys": 1, "pooled": 1}, {})},
                )

    def test_glm_pool_required_and_capacity_exact_even_before_first_pool(self):
        class MLA:
            step = 256

            def __init__(self):
                self.keys = self.ik = self.ig = self.pool = None
                self.offset = 0

        profile = Profile(
            [MLA()],
            describe_tensor=describe,
            token_limit=128,
            token_id_limit=256,
            tensor_byte_limit=1 << 20,
            max_draft=8,
            sizes={"BF16": 2},
            authorities={MLA: ({"keys": 0, "ik": 0, "ig": 0, "pool": 0}, {})},
        )
        current = MLA()
        current.keys = Array("BF16", [256, 512])
        current.ik = Array("BF16", [256, 128])
        current.ig = Array("BF16", [256, 128])
        current.pool = Array("BF16", [64, 128])
        current.offset = 1
        profile.observe([current])
        registry = profile.registry()
        self.assertIn("pool", registry.layers[0].required_tensors)
        for pool in (None, 0, 1, 63, 64, 65):
            entry = {
                "class": registry.layers[0].class_id,
                "plain": {"offset": 1},
                "arrays": ["keys", "ik", "ig", "pool"],
                "numpy": [],
                "lists": {},
            }
            header = {
                f"0.{name}": {"dtype": value.dtype, "shape": list(value.shape)}
                for name, value in vars(current).items()
                if type(value) is Array
            }
            if pool is None:
                entry["arrays"].remove("pool")
                entry["plain"]["pool"] = None
                header.pop("0.pool")
            else:
                header["0.pool"]["shape"][0] = pool
            metadata = {"format": "2", "model": "current", "tokens": "[1]", "layers": json.dumps([entry])}
            header["__metadata__"] = metadata
            if pool == 64:
                registry.validate(metadata, header, model_id="current")
            else:
                with self.assertRaises(ValueError):
                    registry.validate(metadata, header, model_id="current")
        # Execute the actual source early-return seam: a retained key capacity
        # does not regrow a lost/shortened pool before future writes reach it.
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / "src/tensorfold/families/glm5_next/caches.py").read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MLACache")
        grow = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_grow")
        namespace = {}
        exec(compile(ast.Module(body=[grow], type_ignores=[]), "<actual GLM._grow>", "exec"), namespace)
        current.pool = None
        namespace["_grow"](current, 2)
        self.assertIsNone(current.pool)
        current.pool = Array("BF16", [1, 128])
        namespace["_grow"](current, 2)
        self.assertEqual(current.pool.shape, [1, 128])
        current.pool = None
        with self.assertRaisesRegex(ValueError, "must be described"):
            profile.observe([current])

    def test_rotating_pre_capacity_cursor_exact_and_post_rotation_domain_preserved(self):
        class RotatingKVCache:
            step = 256

            def __init__(self):
                self.keys = self.values = None
                self.offset = self._idx = self.keep = 0
                self.max_size = 512

        profile = Profile(
            [RotatingKVCache()],
            describe_tensor=describe,
            token_limit=2048,
            token_id_limit=256,
            tensor_byte_limit=1 << 20,
            max_draft=8,
            sizes={"BF16": 2},
            authorities={RotatingKVCache: ({"keys": 2, "values": 2}, {})},
        )
        observed = RotatingKVCache()
        observed.keys = observed.values = Array("BF16", [1, 2, 256, 4])
        observed.offset = observed._idx = 1
        profile.observe([observed])
        registry = profile.registry()
        for offset, cursor, capacity, valid in (
            (1, 1, 256, True),
            (1, 250, 256, False),
            (20, 20, 256, True),
            (20, 19, 256, False),
            (1024, 250, 512, True),
            (512, 512, 512, True),
            (1024, 250, 256, False),
            (512, 250, 256, False),
            (511, 511, 512, True),
            (1024, 250, 1024, True),
        ):
            entry = {
                "class": registry.layers[0].class_id,
                "plain": {"offset": offset, "_idx": cursor, "keep": 0, "max_size": 512},
                "arrays": ["keys", "values"],
                "numpy": [],
                "lists": {},
            }
            metadata = {"format": "2", "model": "current", "tokens": "[1]", "layers": json.dumps([entry])}
            header = {
                "__metadata__": metadata,
                **{f"0.{name}": {"dtype": "BF16", "shape": [1, 2, capacity, 4]} for name in ("keys", "values")},
            }
            if valid:
                registry.validate(metadata, header, model_id="current")
            else:
                with self.assertRaisesRegex(ValueError, "cursor|complete window"):
                    registry.validate(metadata, header, model_id="current")

    def test_metadata_invariant_callback_executed_before_loader(self):
        profile, observed = self.fixture()
        profile.observe([observed])
        registry = profile.registry()
        self.assertEqual(len(registry.layers[0].invariants), 1)
        _, metadata, header = self.document(registry)
        header["0.values"]["shape"][2] = 1
        with self.assertRaisesRegex(ValueError, "capacity"):
            registry.load(
                metadata,
                header,
                model_id="current",
                tensor_loader=lambda: self.fail("loader reached"),
                describe_tensor=describe,
                convert_numpy=lambda a: a,
            )


class TimelineControls(unittest.TestCase):
    def test_main_exact_and_auxiliary_drafts_or_less_context_are_explicit(self):
        from tensorfold.engine.snapshot_builtin import _token_timeline

        tokens = [1, 2, 3]
        main = _token_timeline(True)
        auxiliary = _token_timeline(False)
        main({"plain": {"offset": 3}}, {}, 0, tokens)
        main({"plain": {}}, {}, 0, tokens)  # recurrent state has no absolute cursor
        for offset in (0, 2, 4):
            with self.subTest(main_offset=offset), self.assertRaises(ValueError):
                main({"plain": {"offset": offset}}, {}, 0, tokens)
        for offset, drafted in ((0, 0), (2, 0), (3, 0), (5, 2)):
            auxiliary({"plain": {"offset": offset, "drafted": drafted}}, {}, 1, tokens)
        for offset, drafted in ((4, 0), (1, 2)):
            with self.subTest(auxiliary_offset=offset, drafted=drafted), self.assertRaises(ValueError):
                auxiliary({"plain": {"offset": offset, "drafted": drafted}}, {}, 1, tokens)

    def test_profile_main_count_and_registry_foreign_loader_gate(self):
        helper = Controls()
        profile, observed = helper.fixture()
        profile.main_layers = 1
        profile.observe([observed])
        registry = profile.registry()
        _, metadata, header = helper.document(registry)
        with self.assertRaisesRegex(ValueError, "stored prefix tokens"):
            registry.load(
                metadata,
                header,
                model_id="current",
                tensor_loader=lambda: self.fail("native loader called"),
                describe_tensor=describe,
                convert_numpy=lambda a: a,
            )
        metadata["tokens"] = json.dumps([1] * 64)
        registry.validate(metadata, header, model_id="current")
        self.assertEqual(len(registry.layers[0].token_invariants), 1)


if __name__ == "__main__":
    unittest.main()
