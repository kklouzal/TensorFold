"""SDK-free host control checks for complete Metal plan authority."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple
import unittest


ROOT = Path(__file__).resolve().parents[1]


def extracted(path, names, scope):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    exec(compile(ast.Module(nodes, []), path, "exec"), scope)
    return scope


def compilation_scope():
    made = []

    def compile_shader(**kwargs):
        made.append(kwargs)
        return kwargs

    scope = {"Any": Any, "NamedTuple": NamedTuple, "hashlib": hashlib, "_kernels": {}, "_plans": {},
             "_SCALAR": "scalar-body", "_MMA": "mma-body", "_PREP": "prep-body", "_HEADER": "base-header",
             "_fragment_source": lambda source: source, "mx": SimpleNamespace(fast=SimpleNamespace(metal_kernel=compile_shader))}
    extracted("src/tensorfold/kernels/qwen/dense/v1/simd_qmm.py", {"Prologue"}, scope)
    scope["_DEFAULT"] = scope["Prologue"]("x", "X[r + j]")
    extracted("src/tensorfold/kernels/qwen/dense/v1/simd_qmm.py", {"_compiled"}, scope)
    scope["made"] = made
    return scope


class PrologueAuthority(unittest.TestCase):
    def test_same_name_changed_load_header_and_input_signature_are_independent(self):
        scope = compilation_scope()
        pro = scope["Prologue"]
        variants = (pro("same", "X[r + j]", ("E", "F"), "header0"),
                    pro("same", "X[r + j + 1]", ("E", "F"), "header0"),
                    pro("same", "X[r + j]", ("E", "F"), "header1"),
                    pro("same", "X[r + j]", ("F", "E"), "header0"))
        shaders = [scope["_compiled"]("scalar", (("K", 128),), False, variant) for variant in variants]
        self.assertEqual(len(scope["made"]), len(variants))
        self.assertEqual(len({shader["name"] for shader in shaders}), len(variants))
        for shader, variant in zip(shaders, variants):
            self.assertIn(variant.load8, shader["source"])
            self.assertEqual(shader["header"], "base-header" + variant.header)
            self.assertEqual(shader["input_names"], ["X", "W", "SC", "BI", "ONE", *variant.inputs])
            self.assertIs(scope["_compiled"]("scalar", (("K", 128),), False,
                                             pro(*variant)), shader)  # equal separately-constructed policy reuses

    def test_dependency_and_output_roles_keep_distinct_shader_identity(self):
        scope = compilation_scope()
        pro = scope["_DEFAULT"]
        ordinary = scope["_compiled"]("scalar", (), False, pro)
        dependent = scope["_compiled"]("scalar", (), True, pro)
        prep = scope["_compiled"]("prep", (), False, pro)
        self.assertEqual(dependent["input_names"][-1], "DEP")
        self.assertEqual(prep["output_names"], ["XF", "XS"])
        self.assertEqual(len({s["name"] for s in (ordinary, dependent, prep)}), 3)

    def test_launch_and_fitted_pipeline_keys_include_full_policy(self):
        scope = compilation_scope()
        pipelines = []
        fitted = {}

        def fit(key, sizes, launch, inputs):
            pipelines.append(key)
            size = max(sizes) if inputs[0] == "wide" else min(sizes)
            fitted[key] = size
            return launch(size)

        def launch(kind, rows, n, dims, group, most=4):
            return (("SGS", most), ("K", dims)), (most * 32, 1, 1), (most * 32, 1, 1), [(rows, n)]

        scope.update(_launch=launch, MMA_SGS=4, _go=lambda kind, plan, dep, pro, inputs: (plan, pro),
                     threads=SimpleNamespace(fit=fit, fitted=fitted.get))
        extracted("src/tensorfold/kernels/qwen/dense/v1/simd_qmm.py", {"_run"}, scope)
        p0 = scope["Prologue"]("same", "X[j]", (), "wide")
        p1 = scope["Prologue"]("same", "X[j]", (), "narrow")
        a = scope["_run"]("mma", 8, 64, 128, 64, False, p0, ["wide"])
        b = scope["_run"]("mma", 8, 64, 128, 64, False, p1, ["narrow"])
        self.assertNotEqual(a[0], b[0])
        self.assertEqual(len(scope["_plans"]), 2)
        self.assertEqual(len(pipelines), 2)
        self.assertNotEqual(pipelines[0], pipelines[1])
        self.assertEqual(scope["_run"]("mma", 8, 64, 128, 64, False, p0, ["wide"]), a)
        self.assertEqual(len(pipelines), 2)


class LaneWarmAuthority(unittest.TestCase):
    def test_warm_group32_and64_both_reach_matching_native_geometry(self):
        import sys
        from unittest.mock import patch

        calls = []
        def native_geometry(x, weight, sbt, *, group=64, **kwargs):
            calls.append({"group": group, **kwargs})
            return object()

        lane = SimpleNamespace(weight_bits=lambda weight, k: 4, lane_matmul=native_geometry)
        package = SimpleNamespace(lane_qmm=lane, stream_gdn=SimpleNamespace())
        attr = "_lane_fuse_groups"
        class Group:
            pass

        groups = []
        for size in (32, 64, 32):
            group = Group()
            group.weight, group.k, group.sk, group.tiled, group.nt = SimpleNamespace(shape=(64, 16)), 128, 1, True, 32
            group.group, group.sbt = size, object()
            module = SimpleNamespace()
            module.__dict__[attr] = {"kv": group}
            groups.append(module)
        model = SimpleNamespace(named_modules=lambda: [(str(i), module) for i, module in enumerate(groups)])
        scope = {"Any": Any, "_Group": Group, "_ATTR": attr,
                 "_group": lambda module, kind, **kwargs: module.__dict__[attr][kind],
                 "mx": SimpleNamespace(array=object, bfloat16="bf16", zeros=lambda *args, **kwargs: object(), eval=lambda *args: None)}
        extracted("src/tensorfold/kernels/qwen/dense/v1/lane_fuse.py", {"warm"}, scope)
        with patch.dict(sys.modules, {"tensorfold.kernels.qwen.dense.v1": package}):
            scope["warm"](model, rows=(1, 17))
        self.assertEqual([call["group"] for call in calls], [32, 32, 64, 64])
        self.assertTrue(all(call["tiled"] and call["nt"] == 32 for call in calls))


if __name__ == "__main__":
    unittest.main()
