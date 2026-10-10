"""Dense EXL3 metadata/lifetime contracts with stdlib-only source oracles.

No Torch, NumPy, MLX, checkpoint payloads, or native execution are imported.
The allocation model proves logical storage bounds, not allocator/driver peaks.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import partial
import gc
from pathlib import Path
import random
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
CAPACITY = ROOT / "src/tensorfold/cuda/capacity.py"
LOADER = ROOT / "src/tensorfold/families/qwen3_5/cuda/exl3_load.py"
FORMAT = ROOT / "src/tensorfold/cuda/exl3/format.py"


def functions(path, names, namespace=None):
    namespace = {} if namespace is None else namespace
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(nodes, []), str(path), "exec", flags=__import__("__future__").annotations.compiler_flag),
         namespace)
    return namespace


def capacity_module():
    module = ModuleType("dense_capacity_source_oracle")
    with patch.dict(sys.modules, {module.__name__: module}):
        exec(compile(CAPACITY.read_text(), str(CAPACITY), "exec"), module.__dict__)
    return module


def metadata_helpers():
    namespace = {"dataclass": dataclass, "HAD": 128,
                 "BITS": (1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8)}
    # dataclasses need a real module while resolving postponed annotations.
    module = ModuleType("dense_exl3_metadata_source_oracle")
    module.__dict__.update(namespace)
    with patch.dict(sys.modules, {module.__name__: module}):
        functions(FORMAT, {"Exl3Tensor", "check_bits", "bits_of", "parse_group"}, module.__dict__)
    return module


def group(prefix, k=128, n=128, bits=2, packed=False, bias=None):
    entries = {prefix + ".trellis": {"shape": [k // 16, n // 16, int(16 * bits)], "dtype": "I16"}}
    for size, half, signs in ((k, "suh", "su"), (n, "svh", "sv")):
        entries[prefix + "." + (signs if packed else half)] = {
            "shape": [size // 16 if packed else size], "dtype": "I16" if packed else "F16"}
    if not float(bits).is_integer():
        entries[prefix + ".mul1"] = {"shape": [], "dtype": "I32"}
    if bias is not None:
        entries[prefix + ".bias"] = {"shape": [n], "dtype": bias}
    return entries


class LogicalAllocator:
    def __init__(self):
        self.live = self.peak = 0


class LogicalStorage:
    def __init__(self, allocator, size):
        self.allocator, self.size = allocator, size
        allocator.live += size
        allocator.peak = max(allocator.peak, allocator.live)

    def __del__(self):
        self.allocator.live -= self.size


class LogicalTensor:
    def __init__(self, allocator, count, width):
        self.allocator, self.count, self.dtype = allocator, count, width
        self.storage = LogicalStorage(allocator, count * width)

    def float(self):
        return self if self.dtype == 4 else LogicalTensor(self.allocator, self.count, 4)

    def to(self, dtype):
        return self if self.dtype == dtype else LogicalTensor(self.allocator, self.count, dtype)

    def __add__(self, value):
        assert value == 1.0
        return LogicalTensor(self.allocator, self.count, self.dtype)


class DenseAdmissionContract(unittest.TestCase):
    def setUp(self):
        self.capacity = capacity_module()

    def admit(self, callback=None, *, unified=False, host_free=1 << 30):
        c = self.capacity
        build = SimpleNamespace(refuse_old_gpu=lambda floor: None)
        cuda = ModuleType("tensorfold.cuda")
        cuda.build = build
        c.floor = lambda model_dir: 120
        c.config = lambda model_dir: {"max_position_embeddings": 256}
        c.unified = lambda torch: unified
        c.host_stream_bytes = lambda: host_free
        c.available_bytes = lambda torch, **kwargs: 1024
        c.page_room = lambda torch: None
        c.estimate_weights = lambda *args, **kwargs: c.Weights(100, 50, 0)
        with patch.dict(sys.modules, {"tensorfold.cuda": cuda}):
            return c.admit(Path("metadata-only"), 128, True, object(), c.Geometry(lambda slots: slots, 0),
                           lambda name, info: (1, 0), weight_estimator=callback)

    def test_generic_header_loop_matches_independent_grouping(self):
        rng = random.Random(917)
        for _ in range(400):
            entries = {f"{rng.choice(('layers', 'blocks'))}.{rng.randrange(8)}.weight.{i}":
                       (rng.randrange(10000), rng.randrange(1000)) for i in range(rng.randrange(60))}
            entries["embedding.weight"] = (rng.randrange(10000), 0)
            amounts, hosts = zip(*entries.values())
            by_layer = {}
            for name, (amount, _) in entries.items():
                parts = name.split(".")
                key = parts[1] if parts[0] in ("layers", "blocks") else name
                by_layer[key] = by_layer.get(key, 0) + amount
            expected = (sum(amounts), 3 * max(max(amounts), max(by_layer.values())), sum(hosts))
            got = self.capacity._estimate_weights_from_headers(entries, lambda name, info: info)
            self.assertEqual((got.resident, got.staging, got.mapped), expected)

    def test_generic_header_wrapper_preserves_rank_and_files(self):
        calls = []
        self.capacity.headers = lambda *args, **kwargs: (calls.append((args, kwargs)) or {"x": [3, 2]})
        got = self.capacity.estimate_weights(Path("model"), lambda name, info: info,
                                            rank=1, files=[Path("shard")])
        self.assertEqual((got.resident, got.staging, got.mapped), (3, 9, 2))
        self.assertEqual(calls, [((Path("model"),), {"rank": 1, "files": [Path("shard")]})])

    def test_generic_negative_rejected(self):
        for info in ((-1, 0), (0, -1)):
            with self.assertRaises(ValueError):
                self.capacity._estimate_weights_from_headers({"x": info}, lambda name, value: value)

    def test_default_admission_contract_unchanged(self):
        got = self.admit()
        self.assertEqual((got["weight_bytes_estimate"], got["loading_bytes_estimate"],
                          got["context_window"], got["total_bytes_estimate"]), (100, 50, 128, 228))

    def test_callback_gpu_and_host_domains_remain_distinct(self):
        calls = []
        got = self.admit(lambda path: (calls.append(path) or self.capacity.Weights(100, 20), 500), host_free=600)
        self.assertEqual(calls, [Path("metadata-only")])
        self.assertEqual(got["loading_bytes_estimate"], 20)
        with self.assertRaisesRegex(ValueError, "host staging"):
            self.admit(lambda path: (self.capacity.Weights(100, 20), 601), host_free=600)

    def test_callback_malformed_bytes_fail_closed(self):
        bad = [None, [self.capacity.Weights(100, 20), 500], (), (object(), 0)]
        for field in range(4):
            for value in (True, False, -1, 1.0, "1", None):
                values = [100, 20, 0, 500]
                values[field] = value
                bad.append((self.capacity.Weights(*values[:3]), values[3]))
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.admit(lambda path, value=value: value)

    def test_norm_dtype_storage_overlap(self):
        loader = next(n for n in ast.parse(LOADER.read_text()).body
                      if isinstance(n, ast.FunctionDef) and n.name == "load_exl3")
        norm = next(n for n in loader.body if isinstance(n, ast.FunctionDef) and n.name == "norm")
        for width in (2, 4, 8):
            for count in (1, 64, 5120, 16384):
                allocator = LogicalAllocator()
                original = LogicalTensor(allocator, count, width)
                scope = {"stored": lambda name, original=original: original}
                exec(compile(ast.Module([norm], []), str(LOADER), "exec"), scope)
                result = scope["norm"]("norm.weight")
                self.assertEqual(result.dtype, width)
                self.assertLessEqual(allocator.peak - count * width, max(8, width + 4) * count)
                del original, result, scope
                gc.collect()
                self.assertEqual(allocator.live, 0)

    def test_cast_overlap_for_accumulated_originals(self):
        for widths in ((2, 8, 4), (8, 8, 8), (2, 2, 2), (4, 2, 8, 4)):
            counts = [17 * (i + 1) for i in range(len(widths))]
            final = 4 * sum(counts)
            overhang = sum(max(0, (width - 4) * count) for width, count in zip(widths, counts))
            bound = overhang + 4 * max(counts)
            raw = sum(width * count for width, count in zip(widths, counts))
            for width, count in zip(widths, counts):
                # Stored originals plus one cast output; then old input retires.
                self.assertLessEqual(raw + 4 * count, final + bound)
                raw += (4 - width) * count

    def test_storage_groups_all_widths_signs_biases_and_norm_dtypes(self):
        fmt = metadata_helpers()
        estimator = functions(LOADER, {"_storage_estimate"})["_storage_estimate"]
        text = {"hidden_size": 128, "num_attention_heads": 1, "head_dim": 128, "vocab_size": 256}
        for bits in fmt.BITS:
            for packed in (False, True):
                for bias in (None, "F16", "F32", "F64"):
                    for norm_dtype in ("F16", "BF16", "F32", "F64"):
                        header = group("lm_head", n=256, bits=bits, packed=packed, bias=bias)
                        header.update(group("model.language_model.layers.0.mlp.up_proj", bits=bits,
                                            packed=packed, bias=bias))
                        header["model.language_model.norm.weight"] = {"shape": [16384], "dtype": norm_dtype}
                        for draft in (False, True):
                            got = estimator(header, text, fmt.parse_group, self.capacity.itemsize, 123,
                                            with_drafter=draft)
                            # Independent integer geometry, not the implementation's grouping loop.
                            payload = (128 * (256 + 128) * int(bits * 2)) // 16
                            scales = 2 * (128 + 256 + 128 + 128 + (384 if bias else 0))
                            counters = 32 * 3
                            plain = self.capacity.SIZES[norm_dtype] * 16384
                            subset = 128 * 256 * int(bits * 2) // 16 + 2 * 256 * (1 + int(bias is not None))
                            expected = payload + scales + counters + plain + 16 * 4
                            if draft:
                                expected += subset + 64 + 2 * 256 * 8
                            self.assertEqual(got["resident"], expected)
                            self.assertGreaterEqual(got["GPU_staging"],
                                                    max(8, self.capacity.SIZES[norm_dtype] + 4) * 16384)
                            self.assertGreaterEqual(got["host_staging"], 123)

    def test_storage_invalid_group_geometry_and_bias_rejected(self):
        fmt = metadata_helpers()
        estimator = functions(LOADER, {"_storage_estimate"})["_storage_estimate"]
        text = {"hidden_size": 128, "num_attention_heads": 1, "vocab_size": 256}
        for k, n in ((0, 256), (128, 0), (64, 256), (128, 64)):
            with self.assertRaises(ValueError):
                estimator(group("lm_head", k=k, n=n), text, fmt.parse_group, self.capacity.itemsize,
                          0, with_drafter=True)
        header = group("lm_head", n=256, bias="F16")
        header["lm_head.bias"]["shape"] = [128]
        with self.assertRaisesRegex(ValueError, "bias"):
            estimator(header, text, fmt.parse_group, self.capacity.itemsize, 0, with_drafter=True)

    def test_storage_cast_dtypes_include_output_and_original_overhang(self):
        fmt = metadata_helpers()
        estimator = functions(LOADER, {"_storage_estimate"})["_storage_estimate"]
        text = {"hidden_size": 128, "num_attention_heads": 1, "vocab_size": 256}
        for width, dtype in ((2, "F16"), (2, "BF16"), (4, "F32"), (8, "F64")):
            header = group("lm_head", n=256)
            for i, count in enumerate((64, 10000)):
                header[f"model.language_model.layers.{i}.linear_attn.A_log"] = {"shape": [count], "dtype": dtype}
            got = estimator(header, text, fmt.parse_group, self.capacity.itemsize, 0, with_drafter=False)
            self.assertEqual(got["plain_final_bytes"], 4 * 10064)
            self.assertEqual(got["original_plain_cast_overhang"], max(0, width - 4) * 10064)
            self.assertEqual(got["cast_conversion_temporary_bound"], 40000)
            self.assertGreaterEqual(got["GPU_staging"], max(0, width - 4) * 10064 + 40000)

    def test_engine_estimator_selection_is_discrete_exl3_text_only(self):
        engine = ROOT / "src/tensorfold/families/qwen3_5/cuda/engine.py"
        cls = next(n for n in ast.parse(engine.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "Qwen27Engine")
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        branch = next(n for n in init.body if isinstance(n, ast.If) and isinstance(n.test, ast.Name)
                      and n.test.id == "exl3" and any(isinstance(s, ast.Assign) for s in n.body))
        for exl3 in (False, True):
            for vision in (False, True):
                for unified in (False, True):
                    for draft in (None, Path("draft")):
                        def estimator(path, *, with_drafter):
                            return with_drafter
                        scope = {"exl3": exl3, "nvfp4": False, "vision": vision, "torch": object(),
                                 "unified": lambda torch, unified=unified: unified, "draft_dir": draft,
                                 "weight_estimator": None, "admission": lambda g: (g, object()),
                                 "geometry": object(), "partial": partial, "weight_estimate": estimator}
                        exec(compile(ast.Module([branch], []), str(engine), "exec"), scope)
                        if exl3 and not vision and not unified:
                            self.assertEqual(scope["weight_estimator"](Path("model")), draft is not None)
                        else:
                            self.assertIsNone(scope["weight_estimator"])

    def test_loader_read_cleanup_preserves_primary_and_owned_order(self):
        loader = next(n for n in ast.parse(LOADER.read_text()).body
                      if isinstance(n, ast.FunctionDef) and n.name == "load_exl3")
        start = next(i for i, n in enumerate(loader.body) if isinstance(n, ast.Try))
        body = loader.body[start:start + 2]  # exact read/materialize try and successful close
        block = ast.FunctionDef(name="read_owned", args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[],
                                                                     kw_defaults=[], defaults=[]),
                                body=body, decorator_list=[])
        for failing in (None, "read", "group"):
            for close_fails in (False, True):
                events = []
                primary = RuntimeError("primary read failure")
                cleanup = RuntimeError("cleanup failure")

                def read(*args):
                    events.append("read")
                    if failing == "read":
                        raise primary
                    return {}

                def groups(*args):
                    events.append("group")
                    if failing == "group":
                        raise primary
                    return {}

                def close():
                    events.append("close")
                    if close_fails:
                        raise cleanup

                scope = {"_read": read, "_read_groups": groups, "files": SimpleNamespace(close=close),
                         "where": {}, "ckpt": SimpleNamespace(plain={}), "foreign": lambda name: False,
                         "device": "metadata-only", "groups": {}, "Workspace": object}
                functions(LOADER.with_name("weights.py"),
                          {"_close_failed_checkpoint_impl", "_close_failed_checkpoint"}, scope)
                exec(compile(ast.fix_missing_locations(ast.Module([block], [])), str(LOADER), "exec"), scope)
                if failing is not None or close_fails:
                    with self.assertRaises(RuntimeError) as result:
                        scope["read_owned"]()
                    self.assertIs(result.exception, primary if failing else cleanup)
                    if failing and close_fails:
                        self.assertIs(primary.__cause__, cleanup)
                        self.assertIn("owned staging remains retained", primary.__notes__[0])
                else:
                    scope["read_owned"]()
                self.assertEqual(events, ["read", "close"] if failing == "read" else ["read", "group", "close"])

    def test_checkpoint_cleanup_helper_retains_native_roots_and_malformed_note_status(self):
        for malformed_notes in (False, True):
            helpers = functions(LOADER.with_name("weights.py"),
                                {"_close_failed_checkpoint_impl", "_close_failed_checkpoint"})
            primary = RuntimeError("primary checkpoint failure")
            cause, context = LookupError("prior cause"), KeyError("prior context")
            cleanup = OSError("reader cleanup failure")
            primary.__cause__, primary.__context__ = cause, context
            if malformed_notes:
                primary.__notes__ = 42
            events = []

            def close():
                events.append("close")
                primary.__cause__, primary.__context__ = ValueError("foreign cause"), ValueError("foreign context")
                raise cleanup

            with self.assertRaises(RuntimeError) as caught:
                helpers["_close_failed_checkpoint"](SimpleNamespace(close=close), primary)
            self.assertIs(caught.exception, primary)
            self.assertEqual(events, ["close"])
            self.assertIsInstance(primary.__cause__, BaseExceptionGroup)
            retained = primary.__cause__.exceptions
            self.assertIs(retained[0], cause)
            self.assertIs(retained[1], context)
            self.assertIs(retained[2], cleanup)
            if malformed_notes:
                self.assertEqual(len(retained), 4)
                self.assertIsInstance(retained[3], TypeError)
            else:
                self.assertEqual(len(retained), 3)


if __name__ == "__main__":
    unittest.main()
