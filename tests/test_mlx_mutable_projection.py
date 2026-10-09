"""SDK-free execution of mutable projection authority and private borrow scope.

Fixtures carry expression labels and shapes. These controls establish host
selection/ownership only; native numerical and performance gates are separate.
"""
from __future__ import annotations

import ast
from operator import index
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Sequence
import unittest
from unittest.mock import patch

from test_mlx_lane_gdn_boundary import Array, Core

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "src/tensorfold/kernels/qwen/dense/v1"


def extract(filename, names, scope):
    nodes = [n for n in ast.parse((BASE / filename).read_text()).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    exec(compile(ast.Module(nodes, []), str(BASE / filename), "exec"), scope)
    return scope


def private_operation():
    scope = {"__name__": "sdk_free_projection_operation"}
    exec(compile((BASE / "projection_operation.py").read_text(), str(BASE / "projection_operation.py"), "exec"), scope)
    return SimpleNamespace(**scope)


class Member(dict):
    bits, group_size, mode = 4, 64, "affine"

    def __init__(self, name="0", n=64, k=128):
        super().__init__(weight=Array("W" + name, "uint32", shape=(n, k // 8)),
                         scales=Array("S" + name, "bfloat16", shape=(n, k // 64)),
                         biases=Array("B" + name, "bfloat16", shape=(n, k // 64)))

    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(name)


def lane_scope():
    core, private, calls = Core(), private_operation(), []
    core.uint32 = "uint32"
    def kernel(name):
        def native(**kwargs):
            calls.append((name, [a.payload for a in kwargs["inputs"][:4] if isinstance(a, Array)]))
            return [Array((name, tuple(calls[-1][1])), dtype, shape=shape)
                    for dtype, shape in zip(kwargs["output_dtypes"], kwargs["output_shapes"])]
        return native
    scope = {"Any": Any, "mx": core, "index": index, "BITS": (2, 3, 4, 5, 6, 8), "NT": 32, "MAX_ROWS": 128,
             "ROW_BLOCK": 32, "_mdims": lambda m, mp: Array((m, mp), "int32", shape=(2,)), "_kernel": kernel,
             "pack_scales": lambda s, b: Array((s.payload, b.payload), "bfloat16", shape=(s.shape[1], s.shape[0], 2)),
             "tile_weight": lambda w, nt, group, bits: Array(("tile", w.payload, nt, group, bits), w.dtype, shape=w.shape),
             "_ORIG": lambda m, x: ("original", m["weight"].payload, m["scales"].payload, m["biases"].payload),
             "enabled": True, "max_rows": 128, "NARROW": ("in_proj_z",)}
    extract("lane_stage.py", {"staging_plan"}, scope)
    extract("lane_qmm.py", {"split_k", "weight_bits", "reads", "readable", "supports", "takes", "_layout", "_call", "_admit_launch",
                             "lane_matmul", "install", "uninstall"}, scope)
    return scope, private, calls


def fused_scope(lane):
    values = {"Any": Any, "mx": lane["mx"], "GROUPS": {"gu": ("gate_proj", "up_proj")},
              "_ATTR": "_lane_fuse_groups", "auto_build": True}
    extract("lane_fuse.py", {"_schema", "_shared_transform", "_Group", "_Unfusable", "_build", "_group"}, values)
    return values


class PrivateLifetime(unittest.TestCase):
    def test_generic_retention_nested_reuse_bound_and_exception_retirement(self):
        private = private_operation()
        x, sums = object(), object()
        private.remember(x, sums)
        self.assertIsNone(private.sums_of(x, 64))
        with self.assertRaisesRegex(OSError, "primary"):
            with private.operation():
                private.remember(x, sums)
                self.assertIs(private.sums_of(x, 64), sums)
                self.assertIsNone(private.sums_of(x, 32))
                with private.operation():
                    self.assertIs(private.sums_of(x, 64), sums)
                keep = [object() for _ in range(4)]
                for value in keep:
                    private.remember(value, sums)
                self.assertIsNone(private.sums_of(x, 64))
                raise OSError("primary")
        self.assertIsNone(private.sums_of(keep[-1], 64))
        with private.operation():
            self.assertIsNone(private.sums_of(keep[-1], 64))

    def test_owned_forward_entries_bind_scope_and_do_not_mutate_private_projection_inputs(self):
        for filename, name in (("lane_multi.py", "multi_tree_forward"), ("row_forward.py", "_rows_forward")):
            tree = ast.parse((BASE / filename).read_text())
            node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            self.assertIn("projection_operation.operation()", [ast.unparse(d) for d in node.decorator_list])
            for current in ast.walk(node):
                self.assertNotIsInstance(current, (ast.AsyncFunctionDef, ast.Await))
                if isinstance(current, ast.AugAssign) and isinstance(current.target, ast.Name):
                    self.assertNotIn(current.target.id, ("x", "hidden", "act"))
                if isinstance(current, ast.Assign):
                    for target in current.targets:
                        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                            self.assertNotIn(target.value.id, ("x", "hidden", "act"))

    def test_rotation_owners_signs_and_inputs_bound_to_one_operation(self):
        private = private_operation()
        owner, x, signs, result = object(), object(), object(), object()
        private.remember_rotation(owner, x, signs, result)
        self.assertIsNone(private.rotation_of(owner, x, signs))
        with private.operation():
            private.remember_rotation(owner, x, signs, result)
            with private.operation():
                self.assertIs(private.rotation_of(owner, x, signs), result)
                self.assertIsNone(private.rotation_of(object(), x, signs))
                self.assertIsNone(private.rotation_of(owner, object(), signs))
                self.assertIsNone(private.rotation_of(owner, x, object()))
            for value in (object(), object(), object(), object()):
                private.remember(value, result)
            self.assertIsNone(private.rotation_of(owner, x, signs))
            private.remember_rotation(owner, x, signs, result)
        self.assertIsNone(private.rotation_of(owner, x, signs))
        with private.operation():
            self.assertIsNone(private.rotation_of(owner, x, signs))

    def test_independent_thread_scope_does_not_borrow_parent(self):
        from concurrent.futures import ThreadPoolExecutor
        private = private_operation()
        x, sums = object(), object()
        def worker():
            self.assertIsNone(private.sums_of(x, 64))
            with private.operation():
                private.remember(x, "worker")
                return private.sums_of(x, 64)
        with private.operation():
            private.remember(x, sums)
            with ThreadPoolExecutor(max_workers=1) as pool:
                self.assertEqual(pool.submit(worker).result(), "worker")
            self.assertIs(private.sums_of(x, 64), sums)


class MutableLane(unittest.TestCase):
    def test_raw_scale_pairs_read_current_fields_without_wider_alignment_copy(self):
        scope, private, calls = lane_scope()
        member, x = Member(), Array("X", "bfloat16", shape=(1, 128))
        sbt = Array("offset-view0", "bfloat16", shape=(2, 64, 2))
        scope["mx"].stack = lambda *a, **k: self.fail("scalar pair reads need no alignment copy")
        package = SimpleNamespace(projection_operation=private)
        with patch.dict(sys.modules, {"tensorfold.kernels.qwen.dense.v1": package}):
            scope["lane_matmul"](x, member["weight"], sbt)
            self.assertEqual(calls[-1][1][3], "offset-view0")
            sbt.payload = "offset-view1"
            scope["lane_matmul"](x, member["weight"], sbt)
            self.assertIn("offset-view1", repr(calls[-1]))
            self.assertNotIn("offset-view0", repr(calls[-1]))
        calls.clear()
        with self.assertRaises(ValueError):
            scope["lane_matmul"](x, member["weight"], sbt, sk=33)
        self.assertEqual(calls, [])

    def test_generic_input_descriptor_changes_rederive_sums_and_private_reuses(self):
        scope, private, calls = lane_scope()
        member, x = Member(), Array("X0", "bfloat16", shape=(1, 128))
        package = SimpleNamespace(projection_operation=private)
        with patch.dict(sys.modules, {"tensorfold.kernels.qwen.dense.v1": package}):
            scope["_call"](member, x)
            x.payload = "X1"
            scope["_call"](member, x)
            self.assertEqual([call[1][0] for call in calls if call[0] == "xsum"],
                             [("reshape", "X0"), ("reshape", "X1")])
            calls.clear()
            with private.operation():
                scope["_call"](member, x)
                scope["_call"](member, x)
            self.assertEqual(sum(call[0] == "xsum" for call in calls), 1)
            scope["_call"](member, x)
            self.assertEqual(sum(call[0] == "xsum" for call in calls), 2)

    def test_current_parameter_replacements_and_same_object_updates_reach_kernel(self):
        for key in ("weight", "scales", "biases"):
            for same in (False, True):
                scope, private, calls = lane_scope()
                member, x = Member(), Array("X", "bfloat16", shape=(1, 128))
                member._lane_tile, member._lane_nt = True, 64
                package = SimpleNamespace(projection_operation=private)
                with patch.dict(sys.modules, {"tensorfold.kernels.qwen.dense.v1": package}):
                    scope["_call"](member, x)
                    before = calls[-1]
                    if same:
                        member[key].payload = "current"
                    else:
                        member[key] = Array("current", member[key].dtype, shape=member[key].shape)
                    scope["_call"](member, x)
                self.assertNotEqual(before, calls[-1])
                self.assertIn("current", repr(calls[-1]))
                self.assertFalse(hasattr(member, "_lane_sbt"))
                self.assertEqual(member["weight"].payload, "current" if key == "weight" else "W0")

    def test_raw_launch_arithmetic_threads_and_shared_storage_precede_native_work(self):
        scope, _, calls = lane_scope()
        scope["tile_weight"] = lambda *a, **k: self.fail("unsupported native region must not materialize a layout")
        outside = Member(n=((1 << 31) - 1) // 16 + 1)
        outside._lane_tile = True
        self.assertEqual(scope["_call"](outside, Array("X", "bfloat16", shape=(1, 128)))[0], "original")
        self.assertEqual(calls, [])
        for kwargs, n in (({"tiled": "false"}, 64),
                          ({"sk": 33}, 64),
                          ({"sk": 17, "tiled": True, "nt": 64}, 64),
                          ({"sk": 16}, 134217724),
                          ({"sk": 1, "tiled": True, "nt": 1 << 26}, 1 << 26)):
            scope, _, calls = lane_scope()
            member = Member(n=n)
            sbt = scope["pack_scales"](member["scales"], member["biases"])
            with self.assertRaises(ValueError):
                scope["lane_matmul"](Array("X", "bfloat16", shape=(1, 128)), member["weight"], sbt, **kwargs)
            self.assertEqual(calls, [])
        admit = lane_scope()[0]["_admit_launch"]
        # All deployed NT32/64, SK1..8 policies and grouped widening sources
        # satisfy the host ceiling. Actual pipeline admission is separate.
        for bits in (2, 3, 4, 5, 6, 8):
            for group in (32, 64):
                for sk in range(1, 9):
                    for nt in ((16, 32, 64) if bits == 4 else (32,)):
                        admit(128, 512, 1024, group, nt, sk, bits)
        # Preserve legal explicit policies above split_k's usual largest8.
        admit(16, 64, 128, 64, 64, 9, 4)
        admit(16, 32, 128, 64, 32, 17, 4)

    def test_packed_staging_preserves_selected_slices_and_bounds_owned_partials(self):
        for nt, sk, n in ((32, 8, 128), (64, 8, 128), (64, 9, 128), (32, 17, 64), (128, 8, 128), (512, 1, 512)):
            scope, private, calls = lane_scope()
            member = Member(n=n)
            sbt = scope["pack_scales"](member["scales"], member["biases"])
            trace = []
            original_kernel = scope["_kernel"]
            def kernel(name):
                original = original_kernel(name)
                def run(**kwargs):
                    trace.append((name, dict(kwargs.get("template", ())), kwargs["output_shapes"], kwargs["output_dtypes"]))
                    return original(**kwargs)
                return run
            scope["_kernel"] = kernel
            with patch.dict(sys.modules, {"tensorfold.kernels.qwen.dense.v1": SimpleNamespace(projection_operation=private)}):
                scope["lane_matmul"](Array("X", "bfloat16", shape=(1, 128)), member["weight"], sbt,
                                     tiled=True, nt=nt, sk=sk)
            stages, partials, shared = scope["staging_plan"](nt, sk, coop=nt == 64)
            self.assertLessEqual(shared, 32768)
            projection = trace[1]
            self.assertEqual(projection[1]["SK"], sk)
            self.assertEqual(projection[1]["STAGES"], stages)
            if partials:
                self.assertTrue(projection[0].endswith("_partials"))
                self.assertEqual(projection[2:], ([(sk, 1, n)], ["float32"]))
                self.assertEqual(trace[-1][0], "ordered_reduce")
                self.assertEqual(trace[-1][1]["SK"], sk)
            else:
                self.assertEqual(projection[2:], ([(1, n)], ["bfloat16"]))
            self.assertTrue(calls)

    def test_install_uninstall_preserve_public_arrays_and_changed_format_adapts_policy(self):
        scope, _, _ = lane_scope()
        member = Member()
        original = tuple(member.values())
        nn = SimpleNamespace(QuantizedLinear=Member)
        Member.__call__ = lambda self, x: x
        model = SimpleNamespace(named_modules=lambda: [("proj", member)])
        with patch.dict(sys.modules, {"mlx": SimpleNamespace(nn=nn), "mlx.nn": nn}):
            scope["install"](model, wide=True)
            self.assertEqual(tuple(member.values()), original)
            self.assertEqual(member._lane_nt, 64)
            member.bits = 6
            member["weight"] = Array("sixbit", "uint32", shape=(64, 24))
            layout, tiled, nt = scope["_layout"](member)
            self.assertTrue(tiled)
            self.assertEqual(nt, 32)
            self.assertEqual(layout.payload, ("tile", "sixbit", 32, 64, 6))
            scope["uninstall"]()
            self.assertEqual(member["weight"].payload, "sixbit")

    def test_invalid_native_geometry_is_refused_before_launch(self):
        changes = (lambda x, w, sb: setattr(x, "dtype", "float32"),
                   lambda x, w, sb: setattr(w, "dtype", "float32"),
                   lambda x, w, sb: setattr(sb, "dtype", "float32"),
                   lambda x, w, sb: setattr(sb, "shape", (2, 63, 2)),
                   lambda x, w, sb: setattr(w, "shape", (63, 16)),
                   lambda x, w, sb: setattr(x, "shape", (0, 128)))
        scope, _, calls = lane_scope()
        self.assertEqual(scope["_call"](Member(), Array("empty", "bfloat16", shape=(0, 128)))[0], "original")
        self.assertEqual(calls, [])
        fused = {"Any": Any, "mx": scope["mx"], "enabled": True, "kinds": {"gu"},
                 "separate_rows": {}, "_group": lambda *a: self.fail("empty rows must not prepare a group")}
        extract("lane_fuse.py", {"_project"}, fused)
        package = SimpleNamespace(lane_qmm=SimpleNamespace(enabled=True, max_rows=128))
        with patch.dict(sys.modules, {"tensorfold.kernels.qwen.dense.v1": package}):
            self.assertIsNone(fused["_project"](object(), "gu", Array("empty", "bfloat16", shape=(0, 128))))
        for change in changes:
            scope, _, calls = lane_scope()
            x, member = Array("X", "bfloat16", shape=(1, 128)), Member()
            sbt = scope["pack_scales"](member["scales"], member["biases"])
            change(x, member["weight"], sbt)
            with self.assertRaises(ValueError):
                scope["lane_matmul"](x, member["weight"], sbt)
            self.assertEqual(calls, [])
        scope, _, calls = lane_scope()
        member = Member(n=48)
        with self.assertRaises(ValueError):
            scope["lane_matmul"](Array("X", "bfloat16", shape=(1, 128)), member["weight"],
                                  scope["pack_scales"](member["scales"], member["biases"]), tiled=True, nt=64)
        self.assertEqual(calls, [])


class MutableStacks(unittest.TestCase):
    def test_row_current_values_plan_geometry_and_parent_membership(self):
        scope, _, _ = lane_scope()
        calls = []
        values = {"Any": Any, "Sequence": Sequence, "index": index, "mx": scope["mx"], "GROUPS": {"gu": ("gate_proj", "up_proj")},
                  "_ATTR": "_row_forward_stacks",
                  "BACKEND": lambda x, w, s, b, group, bits: calls.append((w.payload, s.payload, b.payload, group, bits))}
        extract("row_matmul.py", {"Stack", "stack_of", "project_stack"}, values)
        members = (Member("0"), Member("1"))
        stack = values["Stack"](members)
        parent = SimpleNamespace(gate_proj=members[0], up_proj=members[1], _row_forward_stacks={"gu": stack})
        self.assertIs(values["stack_of"](parent, "gu"), stack)
        for key in ("weight", "scales", "biases"):
            for same in (False, True):
                values["project_stack"](stack, object())
                before = calls[-1]
                if same:
                    members[0][key].payload = str(before) + "updated"
                else:
                    old = members[0][key]
                    members[0][key] = Array(str(before) + "replacement", old.dtype, shape=old.shape)
                self.assertTrue(stack.valid())
                values["project_stack"](stack, object())
                self.assertNotEqual(calls[-1], before)
        members[0].bits = 8
        self.assertIsNone(values["stack_of"](parent, "gu"))
        with self.assertRaises(ValueError):
            values["project_stack"](stack, object())
        members[0].bits = 4.5
        with self.assertRaises(ValueError):
            values["project_stack"](stack, object())
        members[0].bits = 4
        biased = Member()
        biased["bias"] = Array("offset", "bfloat16", shape=(64,))
        with self.assertRaises(ValueError):
            values["Stack"]((biased, Member()))
        bad = Member()
        bad["scales"].shape = (64, 3)
        with self.assertRaises(ValueError):
            values["Stack"]((bad, Member()))
        parent.gate_proj = Member("new")
        self.assertIsNone(values["stack_of"](parent, "gu"))
        self.assertEqual(set(values["Stack"].__slots__), {"group_size", "bits", "sizes", "members", "schema"})

    def test_fused_current_values_and_schema_rotation_membership_reselection(self):
        lane, _, _ = lane_scope()
        values = fused_scope(lane)
        members = (Member("0"), Member("1"))
        parent = SimpleNamespace(gate_proj=members[0], up_proj=members[1])
        package = SimpleNamespace(lane_qmm=SimpleNamespace(**lane))
        with patch.dict(sys.modules, {"mlx": SimpleNamespace(nn=SimpleNamespace(QuantizedLinear=Member)),
                                      "mlx.nn": SimpleNamespace(QuantizedLinear=Member),
                                      "tensorfold.kernels.qwen.dense.v1": package}):
            group = values["_group"](parent, "gu")
            self.assertIsNotNone(group)
            for key in ("weight", "scales", "biases"):
                for same in (False, True):
                    before = (group.weight.payload, group.sbt.payload)
                    old = members[0][key]
                    if same:
                        old.payload = repr(before) + "changed"
                    else:
                        members[0][key] = Array(repr(before) + "replacement", old.dtype, shape=old.shape)
                    self.assertTrue(group.valid())
                    self.assertNotEqual((group.weight.payload, group.sbt.payload), before)
            self.assertEqual(group.added, 0)
            members[0]["bias"] = Array("extra", "bfloat16", shape=(64,))
            self.assertIsNone(values["_group"](parent, "gu"))
            del members[0]["bias"]
            self.assertIsNotNone(values["_group"](parent, "gu"))  # negative admission is not retained
            parent.gate_proj = Member("new")
            current = values["_group"](parent, "gu")
            self.assertIsNot(current, group)
            self.assertIs(current.members[0], parent.gate_proj)

    def test_fused_rotate_reads_current_shared_sign_descriptor_and_method(self):
        lane, _, _ = lane_scope()
        values = fused_scope(lane)
        class RotationCache:
            pass
        class Rotated:
            def __init__(self, member, signs):
                self.inner, self.signs = member, signs
                self.rotation = RotationCache()
            def rotate(self, x):
                return (x, self.signs.payload)
        signs = Array("signs0", "bfloat16", shape=(128,))
        parent = SimpleNamespace(gate_proj=Rotated(Member("0"), signs), up_proj=Rotated(Member("1"), signs))
        package = SimpleNamespace(lane_qmm=SimpleNamespace(**lane))
        with patch.dict(sys.modules, {"mlx": SimpleNamespace(nn=SimpleNamespace(QuantizedLinear=Member)),
                                      "mlx.nn": SimpleNamespace(QuantizedLinear=Member),
                                      "tensorfold.kernels.qwen.dense.v1": package,
                                      "tensorfold.families.bonsai.modules": SimpleNamespace(
                                          RotatedLinear=Rotated, RotationCache=RotationCache)}):
            group = values["_group"](parent, "gu")
            signs.payload = "signs1"
            self.assertEqual(group.rotate("X"), ("X", "signs1"))
            parent.gate_proj.rotate = lambda x: (x, "new method")
            self.assertFalse(group.valid())
            self.assertIsNone(values["_group"](parent, "gu"))
            del parent.gate_proj.rotate
            self.assertIsNotNone(values["_group"](parent, "gu"))
            parent.up_proj.signs = Array("unshared", "bfloat16", shape=(128,))
            self.assertIsNone(values["_group"](parent, "gu"))

    def test_fused_cached_rotation_admission_tracks_current_owner_policy(self):
        lane, _, _ = lane_scope()
        values = fused_scope(lane)
        class RotationCache:
            def __call__(self, x, signs):
                return ("declared", x, signs.payload)
        class OtherCache(RotationCache):
            def __call__(self, x, signs):
                return ("custom", x, signs.payload)
        class Rotated:
            def __init__(self, member, signs):
                self.inner, self.signs, self.rotation = member, signs, RotationCache()
            def rotate(self, x):
                return self.rotation(x, self.signs)
        signs = Array("signs", "bfloat16", shape=(128,))
        nn = SimpleNamespace(QuantizedLinear=Member)
        with patch.dict(sys.modules, {"mlx": SimpleNamespace(nn=nn), "mlx.nn": nn,
                                      "tensorfold.kernels.qwen.dense.v1": SimpleNamespace(lane_qmm=SimpleNamespace(**lane)),
                                      "tensorfold.families.bonsai.modules": SimpleNamespace(
                                          RotatedLinear=Rotated, RotationCache=RotationCache)}):
            for field in ("gate_proj", "up_proj"):
                for replacement in (lambda x, signs: ("custom", x, signs.payload), OtherCache()):
                    parent = SimpleNamespace(gate_proj=Rotated(Member("0"), signs),
                                             up_proj=Rotated(Member("1"), signs))
                    group = values["_group"](parent, "gu")
                    self.assertIsNotNone(group)
                    setattr(getattr(parent, field), "rotation", replacement)
                    self.assertFalse(group.valid())
                    self.assertIsNone(values["_group"](parent, "gu"))
                    self.assertIsInstance(values["_build"](parent, "gu"), values["_Unfusable"])
                    self.assertEqual(getattr(parent, field).rotate("X"), ("custom", "X", "signs"))
                    getattr(parent, field).rotation = RotationCache()
                    current = values["_group"](parent, "gu")
                    self.assertIsNotNone(current)
                    self.assertEqual(current.rotate("X"), ("declared", "X", "signs"))
                    signs.payload = "current"
                    self.assertTrue(current.valid())
                    self.assertEqual(current.rotate("X"), ("declared", "X", "current"))
                    signs.payload = "signs"


if __name__ == "__main__":
    unittest.main()
