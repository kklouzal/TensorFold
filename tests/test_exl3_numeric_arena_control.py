"""Numeric EXL3 arena control and header byte bounds without accelerator imports."""
import ast
from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
import io
import math
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def source_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LAYOUT = source_module("_numeric_arena_layout", "src/tensorfold/cuda/exl3/cache_layout.py")
RAM = source_module("_numeric_arena_ram", "src/tensorfold/families/qwen4_exp/ram_experts.py")
ExpertSpec, arena_plan, Layout = LAYOUT.ExpertSpec, LAYOUT.plan, RAM.Layout
from tensorfold.cuda.capacity import Weights  # noqa: E402 - stdlib-only source imports


def function(path, name, namespace):
    tree = ast.parse((ROOT / path).read_bytes())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node]
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(module, str(ROOT / path), "exec"), namespace)
    return namespace[name]


class OpaqueFailure(BaseException):
    def add_note(self, text):
        raise AssertionError("foreign note hook")

    def __str__(self):
        raise AssertionError("foreign string conversion")

    __repr__ = __str__


class NumericArenaControl(unittest.TestCase):
    def receipt(self, cache):
        tree = ast.parse((ROOT / "src/tensorfold/families/qwen4_exp/cuda/engine.py").read_bytes())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FlashNextEngine")
        init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_initialize")
        branch = next(node for node in ast.walk(init) if isinstance(node, ast.If)
                      and ast.unparse(node.test) == "ram_layout is not None and (not automatic_experts)")
        module = ast.fix_missing_locations(ast.Module(body=branch.body, type_ignores=[]))
        initial = {"vram_experts": {"gpu_bytes": 1024, "control_device_bytes": 128, "control_host_bytes": 1024},
                   "weight_bytes_estimate": 3000, "startup_peak_bytes_estimate": 8000,
                   "serving_peak_bytes_estimate": 9000, "total_bytes_estimate": 9000, "mapped_table_bytes": 500}
        owner = SimpleNamespace(capacity_plan=deepcopy(initial))
        namespace = {"self": owner, "w": SimpleNamespace(meta={"expert_cache": cache}), "deepcopy": deepcopy,
                     "ram_layout": SimpleNamespace(format="exl3", entry_bytes=256)}
        with redirect_stdout(io.StringIO()):
            exec(compile(module, "numeric-engine-receipt", "exec"), namespace)
        return owner.capacity_plan, initial

    def test_final_receipt_preserves_initial_admission_and_reflects_actual_capacities(self):
        cache = SimpleNamespace(gpu_bytes=960, capacity=6, resident_capacity=8,
                                control_device_bytes=160, control_host_bytes=512, host_bytes=10000)
        actual, initial = self.receipt(cache)
        self.assertEqual(actual["initial_header_pool_admission"], initial)
        self.assertIsNot(actual["initial_header_pool_admission"]["vram_experts"], actual["vram_experts"])
        self.assertEqual(actual["weight_bytes_estimate"], 2968)
        self.assertEqual(actual["serving_peak_bytes_estimate"], 8968)
        self.assertEqual(actual["total_bytes_estimate"], 8968)
        self.assertEqual(actual["vram_experts"]["resident_capacity"], 8)
        self.assertEqual(actual["vram_experts"]["safe_lease_capacity"], 6)

    def test_final_receipt_rejects_gpu_or_host_admission_overrun(self):
        for gpu, device, host in ((1024, 129, 1024), (960, 160, 1025)):
            cache = SimpleNamespace(gpu_bytes=gpu, capacity=6, resident_capacity=8,
                                    control_device_bytes=device, control_host_bytes=host, host_bytes=10000)
            with self.assertRaises(MemoryError):
                self.receipt(cache)

    def test_estimator_covers_bootstrap_overlap_without_inventing_host_staging(self):
        tree = ast.parse((ROOT / "src/tensorfold/families/qwen4_exp/cuda/engine.py").read_bytes())
        function = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef) and node.name == "numeric_weights")
        for staging in (32, 1024):
            calls = []
            rule = object()
            original = Weights(4096, staging, 8192)
            namespace = {"Weights": Weights, "rank": 0, "weight_rule": rule,
                         "ram_layout": SimpleNamespace(entry_bytes=256),
                         "estimate_weights": lambda *a, **k: (calls.append((a, k)) or original)}
            exec(compile(ast.Module(body=[function], type_ignores=[]), "numeric-startup-estimate", "exec"), namespace)
            gpu, host = namespace["numeric_weights"]("model")
            self.assertEqual(gpu, Weights(4096, max(staging, 288), 8192))
            self.assertEqual(host, staging)
            self.assertEqual(calls, [(("model", rule), {"rank": 0})])

    def loader(self, *, automatic=False, failed=None, cleanup=None):
        order = []
        plan = SimpleNamespace(gpu_bytes=1024, entry_bytes=256, automatic=automatic)
        cache = SimpleNamespace()

        def configure(size):
            order.append(("configure", size))
            if failed == "configure":
                raise self.primary

        def close():
            order.append("close")
            if cleanup:
                raise cleanup

        cache.configure_before_use, cache.close = configure, close

        def create(payload, entry, device):
            order.append(("create", payload, entry, device))
            return cache

        weights = SimpleNamespace(meta={"expert_cache": cache})
        configuration, draft_ids = SimpleNamespace(vocab=64), object()

        def loaded(*args, **kwargs):
            self.assertIs(kwargs["expert_cache"], cache)
            self.assertIs(kwargs["_config"], configuration)
            self.assertIs(kwargs["_draft_ids"], draft_ids)
            order.append("load-owner-drained")
            if failed == "load":
                raise self.primary
            return weights

        ram = ModuleType("tensorfold.families.qwen4_exp.ram_experts")
        ram.check = lambda *a, **k: None
        ram.layout = lambda *a, **k: plan
        host = ModuleType("tensorfold.cuda.exl3.host_experts")
        host.Exl3HostExpertCache = create
        types = ModuleType("tensorfold.families.qwen4_exp.cuda.weight_types")
        types.Config = SimpleNamespace(read=lambda *args, **kwargs: configuration)

        def selected_ids(request, vocab):
            self.assertEqual(vocab, configuration.vocab)
            return draft_ids

        types.draft_token_ids = selected_ids
        load = function("src/tensorfold/families/qwen4_exp/cuda/exl3.py", "load",
                        {"__package__": "tensorfold.families.qwen4_exp.cuda", "_load": loaded})
        return load, weights, order, {ram.__name__: ram, host.__name__: host, types.__name__: types}

    def test_numeric_configuration_follows_drained_load_once(self):
        load, weights, order, modules = self.loader()
        with patch.dict(sys.modules, modules):
            result = load("fixture", vram_experts=.5)
        self.assertIs(result, weights)
        self.assertEqual(order, [("create", 256, 256, "cuda"), "load-owner-drained", ("configure", 1024)])

    def test_auto_retains_bootstrap_for_decoder_owned_sizing(self):
        load, weights, order, modules = self.loader(automatic=True)
        with patch.dict(sys.modules, modules):
            self.assertIs(load("fixture", vram_experts="auto"), weights)
        self.assertEqual(order, [("create", 256, 256, "cuda"), "load-owner-drained"])

    def test_configuration_or_load_failure_closes_before_propagating(self):
        for phase in ("load", "configure"):
            self.primary = OpaqueFailure()
            load, _, order, modules = self.loader(failed=phase)
            with patch.dict(sys.modules, modules), self.assertRaises(OpaqueFailure) as raised:
                load("fixture", vram_experts=.5)
            self.assertIs(raised.exception, self.primary)
            self.assertEqual(order[-1], "close")

    def test_cleanup_preserves_primary_identity_and_owned_failure(self):
        self.primary, cleanup = OpaqueFailure(), OpaqueFailure()
        load, _, order, modules = self.loader(failed="configure", cleanup=cleanup)
        with patch.dict(sys.modules, modules), self.assertRaises(OpaqueFailure) as raised:
            load("fixture", vram_experts=.5)
        self.assertIs(raised.exception, self.primary)
        self.assertIs(self.primary.__cause__, cleanup)
        self.assertEqual(order[-1], "close")

    def test_header_host_publication_bound_matches_complete_immutable_roles(self):
        # Independent sized-header double: only the new role/control accounting
        # is under test; maintained EXL3 shape/bit validation has its own oracle.
        text = {"num_experts_per_tok": 2, "num_experts": 3, "num_hidden_layers": 2,
                "moe_intermediate_size": 128, "hidden_size": 256}
        entries = {}
        for base in ("model.language_model.layers.0.mlp", "model.language_model.layers.1.mlp",
                     "mtp.layers.0.mlp"):
            for expert in [f"experts.{i}" for i in range(3)] + ["shared_expert"]:
                for name, inputs, outputs in (("gate_proj", 256, 128), ("up_proj", 256, 128),
                                               ("down_proj", 128, 256)):
                    entries[f"{base}.{expert}.{name}.trellis"] = {
                        "shape": [1], "k": inputs, "n": outputs,
                        "bytes": 32 if expert == "shared_expert" else 16}
        fmt = ModuleType("tensorfold.cuda.exl3.format")
        fmt.PARTS = ("trellis",)
        fmt.parse_group = lambda prefix, parts: SimpleNamespace(
            k=parts["trellis"]["k"], n=parts["trellis"]["n"],
            bias=False, codebook="same", trellis_bytes=parts["trellis"]["bytes"])
        namespace = {"Layout": Layout, "headers": lambda _: entries, "math": math, "re": re, "GIB": 2**30,
                     "exl3_expert_tensor": lambda name: ".mlp.experts." in name or ".mlp.shared_expert." in name}
        make = function("src/tensorfold/families/qwen4_exp/ram_experts.py", "_exl3_layout", namespace)
        specs = tuple(ExpertSpec(layer, expert, 96 if expert == 3 else 48, expert == 3)
                      for layer in range(3) for expert in range(4))
        for cells in (1, 2, 6, 8, 12):
            with patch.dict(sys.modules, {fmt.__name__: fmt, "tensorfold.cuda.exl3.cache_layout": LAYOUT}):
                actual = make("fixture", cells * 96 / 2**30, text, True)
            expected = arena_plan(specs, cells * 96, cells * 128)
            self.assertEqual(actual.control_host_bytes, expected.publication_host_bytes)
            self.assertEqual(actual.control_device_bytes, cells * 32)
            self.assertLessEqual(expected.payload_bytes + expected.publication_device_bytes,
                                 actual.gpu_bytes + actual.control_device_bytes)
        self.assertGreater(arena_plan(specs, 8 * 96, 8 * 128).publication_host_bytes, 8 * 64)


if __name__ == "__main__":
    unittest.main()
