"""Pinned TCPStore timeout contract and actual caller AST, stdlib only."""
from __future__ import annotations

import ast
from datetime import timedelta
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from tensorfold.cuda.store_wait import timed_out

ROOT = Path(__file__).resolve().parents[1]


class DistStoreError(RuntimeError):
    pass


class Controls(unittest.TestCase):
    def setUp(self):
        module = ModuleType("torch.distributed")
        module.DistStoreError = DistStoreError
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, {"torch.distributed": module}).start()

    def test_only_exact_native_timeout_key_duration_and_type_retry(self):
        timeout = timedelta(milliseconds=5)
        self.assertTrue(timed_out(DistStoreError("wait timeout after 5ms, keys: /t47/idle"), ["t47/idle"], timeout))
        for error in (DistStoreError("Stop_waiting response is expected"),
                      DistStoreError("wait timeout after 6ms, keys: /t47/idle"),
                      DistStoreError("wait timeout after 5ms, keys: /other"),
                      RuntimeError("wait timeout after 5ms, keys: /t47/idle")):
            self.assertFalse(timed_out(error, ["t47/idle"], timeout))
        class Opaque(DistStoreError):
            def __str__(self):
                raise AssertionError("foreign formatting invoked")
        self.assertFalse(timed_out(Opaque("timeout"), ["t47/idle"], timeout))

    def test_actual_glm_bell_retry_then_consume_and_structural_primary(self):
        tree = ast.parse((ROOT / "src/tensorfold/families/glm5_next/cuda/engine.py").read_bytes())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GlmEngine")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_await_bell")
        values = {}
        exec(compile(ast.Module([method], []), "actual-glm-bell", "exec"), values)
        class Store:
            def __init__(self, errors):
                self.errors, self.deleted, self.calls = list(errors), [], 0
            def wait(self, keys, timeout):
                self.calls += 1
                if self.errors:
                    raise self.errors.pop(0)
            def delete_key(self, key):
                self.deleted.append(key)
        store = Store([DistStoreError("wait timeout after 3600000ms, keys: /tf_glm_request_1")])
        owner = SimpleNamespace(_store=lambda: store)
        values["_await_bell"](owner)
        self.assertEqual((store.calls, store.deleted, owner._bell), (2, ["tf_glm_request_1"], 1))
        primary = DistStoreError("unexpected timeout response")
        store = Store([primary])
        with self.assertRaises(DistStoreError) as caught:
            values["_await_bell"](SimpleNamespace(_store=lambda: store))
        self.assertIs(caught.exception, primary)
        self.assertEqual((store.calls, store.deleted), (1, []))


if __name__ == "__main__":
    unittest.main()
