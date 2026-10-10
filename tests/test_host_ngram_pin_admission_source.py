"""Stdlib resource/caller controls; no SDK, native locking or GPU claims."""
from __future__ import annotations

import ast
import math
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def source_module():
    capacity = ModuleType("tensorfold.cuda.capacity")
    capacity.GIB = 1 << 30
    tree = ast.parse((ROOT / "src/tensorfold/cuda/capacity.py").read_text())
    reserve = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "reserve_bytes")
    namespace = dict(os=os, math=math, GIB=capacity.GIB)
    exec(compile(ast.Module(body=[reserve], type_ignores=[]), "reserve-contract", "exec"), namespace)
    capacity.reserve_bytes = namespace["reserve_bytes"]
    module = ModuleType("tensorfold.cuda.host_pin")
    module.__package__ = "tensorfold.cuda"
    path = ROOT / "src/tensorfold/cuda/host_pin.py"
    with patch.dict(sys.modules, {capacity.__name__: capacity}):
        exec(compile(path.read_bytes(), str(path), "exec"), vars(module))
    return module, capacity


class HostPins(unittest.TestCase):
    def setUp(self):
        self.module, self.capacity = source_module()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.member = self.root / "membership"
        self.member.write_text("0::/leaf\n")
        self.host = self.root / "meminfo"
        self.host.write_text("MemTotal: 67108864 kB\nMemAvailable: 44040192 kB\n")
        self.cgroups = self.root / "groups"
        (self.cgroups / "leaf").mkdir(parents=True)
        self.host_read = self.module._host
        self.group_read = self.module._cgroups
        def arithmetic_host(_path):
            result = self.host_read(self.host)
            if result is not None:
                # Isolate byte arithmetic with zero injected headroom; the
                # actual host-stream reserve policy has its own control below.
                result.update(reserve_bytes=0, room_bytes=result["available_bytes"])
            return result
        self.host_patch = patch.object(self.module, "_host", arithmetic_host)
        self.group_patch = patch.object(self.module, "_cgroups", lambda _member, _root, payload:
                                       self.group_read(self.member, self.cgroups, payload))
        self.host_patch.start()
        self.group_patch.start()
        self.addCleanup(self.host_patch.stop)
        self.addCleanup(self.group_patch.stop)
        self.environment = patch.dict(os.environ, {"TENSORFOLD_MEMORY_RESERVE_GIB": ""})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def group(self, *, maximum, current, anon, kernel=0, file=0, active=0, inactive=0,
              dirty=0, writeback=0, name="leaf"):
        p = self.cgroups / name
        (p / "memory.max").write_text(str(maximum))
        (p / "memory.min").write_text("0")
        (p / "memory.low").write_text("0")
        (p / "memory.current").write_text(str(current))
        (p / "memory.stat").write_text("\n".join(f"{k} {v}" for k, v in dict(
            anon=anon, kernel=kernel, file=file, active_file=active, inactive_file=inactive,
            file_dirty=dirty, file_writeback=writeback).items())+"\n")

    def test_large_vram_never_authorizes_overfull_host_authority(self):
        self.group(maximum=56_000_000_000, current=55_000_000_000,
                   anon=30_949_260_636, file=24_000_000_000, inactive=24_000_000_000)
        p = self.module.admit(26_240_125_952, device_room_bytes=100_000_000_000)
        self.assertTrue(p["device_allows"])
        self.assertFalse(p["cgroups_allow"])
        self.assertFalse(p["admitted"])

    def test_prefetched_clean_pages_are_not_charged_twice(self):
        self.group(maximum=60_000, current=55_000, anon=31_000, file=24_000, inactive=24_000)
        p = self.module.admit(26_000, device_room_bytes=30_000)
        self.assertTrue(p["admitted"])
        self.assertEqual(p["cgroups"][0]["unreclaimable_bytes"], 31_000)
        self.assertEqual(p["cgroups"][0]["room_bytes"], 29_000)

    def test_dirty_and_unevictable_or_shmem_never_become_clean_credit(self):
        # The LRU count excludes 8,000 non-LRU file bytes; dirty/writeback
        # overlap is deliberately subtracted conservatively, not added.
        self.group(maximum=60_000, current=55_000, anon=31_000, file=24_000,
                   inactive=16_000, dirty=3_000, writeback=2_000)
        p = self.module.admit(26_000, device_room_bytes=30_000)
        self.assertEqual(p["cgroups"][0]["clean_file_credit_bytes"], 11_000)
        self.assertFalse(p["admitted"])

    def test_anonymous_and_kernel_floor_survives_inconsistent_sample(self):
        self.group(maximum=60_000, current=40_000, anon=31_000, kernel=5_000,
                   file=30_000, inactive=30_000)
        p = self.module.admit(25_000, device_room_bytes=30_000)
        self.assertEqual(p["cgroups"][0]["unreclaimable_bytes"], 36_000)
        self.assertFalse(p["admitted"])

    def test_credit_never_spends_unselected_or_protected_sibling_pages(self):
        self.group(maximum=60_000, current=59_000, anon=20_000, file=39_000, inactive=39_000)
        p = self.module.admit(1_000, device_room_bytes=30_000, future_bytes=2_000)
        self.assertEqual(p["cgroups"][0]["clean_file_credit_bytes"], 1_000)
        self.assertFalse(p["admitted"])

    def test_descendants_or_ancestry_protection_cannot_authorize_credit(self):
        self.group(maximum=60_000, current=40_000, anon=20_000, file=20_000, inactive=20_000)
        child = self.cgroups / "leaf/child"
        child.mkdir()
        self.assertFalse(self.module.admit(10_000, device_room_bytes=30_000)["admitted"])
        child.rmdir()
        self.group(maximum="max", current=40_000, anon=20_000, name="")
        (self.cgroups / "memory.min").write_text("10000")
        self.assertFalse(self.module.admit(10_000, device_room_bytes=30_000)["admitted"])

    def test_unlimited_leaf_credit_never_becomes_ancestor_sibling_credit(self):
        self.group(maximum="max", current=20_000, anon=19_000, file=1_000, inactive=1_000)
        self.group(maximum=60_000, current=59_000, anon=20_000, file=39_000, inactive=39_000, name="")
        p = self.module.admit(10_000, device_room_bytes=30_000)
        self.assertEqual(p["cgroups"][1]["clean_file_credit_bytes"], 1_000)
        self.assertFalse(p["admitted"])

    def test_every_visible_ancestor_and_explicit_future_room(self):
        self.group(maximum=60_000, current=20_000, anon=20_000)
        self.group(maximum=45_000, current=30_000, anon=30_000, name="")
        p = self.module.admit(10_000, device_room_bytes=30_000, future_bytes=6_000)
        self.assertEqual(len(p["cgroups"]), 2)
        self.assertEqual(p["needed_bytes"], 16_000)
        self.assertFalse(p["admitted"])

    def test_host_unknown_and_existing_device_refusal(self):
        self.assertFalse(self.module.admit(1_000, device_room_bytes=100)["admitted"])
        self.host.unlink()
        p = self.module.admit(1_000, device_room_bytes=10_000)
        self.assertIsNone(p["host"])
        self.assertFalse(p["admitted"])

    def test_existing_host_headroom_applies_to_both_resource_domains(self):
        self.assertEqual(self.host_read(self.host)["reserve_bytes"], 2 << 30)
        self.group(maximum=3 << 30, current=1 << 30, anon=1 << 30)
        with patch.object(self.module, "_host", lambda _path: self.host_read(self.host)):
            p = self.module.admit(1, device_room_bytes=10)
        self.assertEqual(p["cgroups"][0]["pin_room_bytes"], 0)
        self.assertFalse(p["admitted"])

    def test_external_resource_and_type_boundaries(self):
        for value in (True, -1, 1.0, "1"):
            with self.assertRaises(ValueError):
                self.module.admit(value, device_room_bytes=10)
        self.member.write_text("0::/../escape\n")
        with self.assertRaises(ValueError):
            self.module.admit(1, device_room_bytes=10)
        self.member.write_text("0::/leaf\n")
        self.group(maximum=10, current=1, anon=1)
        (self.cgroups / "leaf/memory.stat").unlink()
        with self.assertRaises(FileNotFoundError):
            self.module.admit(1, device_room_bytes=10)

    def test_empty_selection_and_valid_unknown_counter_capability(self):
        with patch.object(self.module, "_host", side_effect=AssertionError("no-op read")):
            self.assertFalse(self.module.admit(0, device_room_bytes=10)["admitted"])
        self.group(maximum=60_000, current=20_000, anon=20_000)
        (self.cgroups / "leaf/memory.stat").write_text("anon 20000\nfile 0\n")
        p = self.module.admit(1_000, device_room_bytes=30_000)
        self.assertFalse(p["admitted"])
        self.assertFalse(p["cgroups"][0]["counter_capability"])
        self.member.write_text("0::/missing\n")
        p = self.module.admit(1_000, device_room_bytes=30_000)
        self.assertFalse(p["admitted"])
        self.assertTrue(p["cgroups"][0]["missing_current_cgroup"])

    def test_unknown_limited_root_finalizes_valid_earlier_leaf(self):
        self.group(maximum=60_000, current=20_000, anon=20_000)
        self.group(maximum=100_000, current=30_000, anon=30_000, name="")
        (self.cgroups / "memory.stat").write_text("anon 30000\nfile 0\n")
        p = self.module.admit(1_000, device_room_bytes=30_000)
        self.assertFalse(p["admitted"])
        self.assertTrue(p["cgroups"][0]["unprotected_childless_current_cgroup"])
        self.assertFalse(p["cgroups"][1]["counter_capability"])

    def test_page_rounded_actual_lock_arrays_only(self):
        arrays = [SimpleNamespace(nbytes=100, ctypes=SimpleNamespace(data=4100)),
                  SimpleNamespace(nbytes=4100, ctypes=SimpleNamespace(data=8192))]
        table = SimpleNamespace(words=arrays, scales=[], biases=[])
        with patch.object(self.module, "PAGESIZE", 4096):
            self.assertEqual(self.module.table_bytes([table]), 12288)
            self.assertEqual(self.module.table_bytes([SimpleNamespace(values=arrays)]), 12288)

    def test_actual_caller_complete_prefetch_and_all_lock_statuses(self):
        path = ROOT / "src/tensorfold/families/qwen4_exp/cuda/engine.py"
        tree = ast.parse(path.read_text())
        owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FlashNextEngine")
        init = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "_initialize")
        node = next(node for node in init.body if isinstance(node, ast.If)
                    and ast.unparse(node.test) == "prefetch and (not ple_on_ssd)")
        events = []
        def table(number, result):
            return SimpleNamespace(words=[], scales=[], biases=[],
                                   prefetch=lambda: events.append(("read", number)),
                                   lock=lambda: events.append(("lock", number)) or result)
        a, b = table(1, False), table(2, True)
        weights = SimpleNamespace(layers=[SimpleNamespace(ple=SimpleNamespace(table=t)) for t in (a, b, a)])
        self.capacity.unified = lambda _torch: False
        for admitted in (False, True):
            events.clear()
            state = SimpleNamespace(capacity_plan=dict(budget_bytes=100, total_bytes_estimate=50,
                                                       serving_peak_bytes_estimate=40))
            namespace = dict(self=state, w=weights, prefetch=True, ple_on_ssd=False,
                             tables_read=False, locked=False, torch=SimpleNamespace())
            with patch.object(self.module, "table_bytes", return_value=10), patch.object(
                    self.module, "admit", return_value=dict(admitted=admitted)), patch.dict(sys.modules, {
                        self.module.__name__: self.module, self.capacity.__name__: self.capacity}):
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
            expected = [("read", 1), ("read", 2)] + ([("lock", 1), ("lock", 2)] if admitted else [])
            self.assertEqual(events, expected)
            self.assertFalse(namespace["locked"])
            self.assertFalse(state.capacity_plan["ngram_pin_admission"]["all_selected_tables_locked"])


if __name__ == "__main__":
    unittest.main()
