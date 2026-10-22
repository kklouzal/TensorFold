"""CPU preparation error ownership, using the actual hash/load function body."""

import ast
import hashlib
import sys
import tempfile
import unittest
from functools import lru_cache
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from tensorfold import file_io
from tensorfold.cuda.tensor_file import _regular_stream

SOURCE = Path(__file__).parents[1] / "src/tensorfold/cuda/nvfp4/bank_cpu.py"


class SourcePreparation(unittest.TestCase):
    def exercise(self, *, primary=None, cleanup=None, opening=None):
        events, raw = [], b"owned C++ source bytes\n"
        node = next(node for node in ast.parse(SOURCE.read_bytes()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "_load")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank_cpu.py"
            (path.with_suffix(".cpp")).write_bytes(raw)
            scope = {"__file__": str(path), "hashlib": hashlib, "Path": Path,
                     "lru_cache": lru_cache, "_regular_stream": _regular_stream}
            exec(compile(ast.Module([node], []), str(SOURCE), "exec"), scope)  # noqa: S102 - exact owned source body.

            class Streams:
                retired = False

                def __init__(self, *, max_files):
                    if max_files != 1:
                        raise AssertionError("source read must own exactly one file")

                def open(self, source, flags, mode):
                    events.append("open")
                    if opening is not None:
                        raise opening
                    if source != path.with_suffix(".cpp") or mode != "rb":
                        raise AssertionError("wrong source bytes requested")
                    return SimpleNamespace(stream=SimpleNamespace(read=self.read)), None

                def read(self):
                    events.append("read")
                    if primary is not None:
                        raise primary
                    return raw

                def drain(self):
                    events.append("drain")
                    self.retired = True
                    return [] if cleanup is None else [cleanup]

            module = ModuleType("torch.utils.cpp_extension")
            result = object()

            def load(**kwargs):
                events.append("load")
                self.assertEqual(kwargs["name"], "tf_a4_bank_cpu_" + hashlib.sha256(raw).hexdigest()[:16])
                self.assertEqual(kwargs["sources"], [str(path.with_suffix(".cpp"))])
                self.assertFalse(kwargs["with_cuda"])
                return result

            module.load = load
            with (patch.dict(sys.modules, {"torch.utils.cpp_extension": module}),
                  patch.object(file_io, "FileStreams", Streams)):
                try:
                    actual = scope["_load"]("bank_cpu.cpp")
                except BaseException as error:  # noqa: BLE001 - Exact transported root is the assertion input.
                    return error, events
                self.assertIs(actual, result)
                return None, events

    def test_normal_source_hash_closes_owner_before_same_build_call(self):
        error, events = self.exercise()
        self.assertIsNone(error)
        self.assertEqual(events, ["open", "read", "drain", "load"])

    def test_read_failure_and_close_failure_preserve_exact_roots_and_prevent_build(self):
        primary, cleanup = OSError("read primary"), OSError("close secondary")
        error, events = self.exercise(primary=primary, cleanup=cleanup)
        self.assertIs(error, primary)
        self.assertEqual(events, ["open", "read", "drain"])
        self.assertIn(cleanup, error.__cause__.exceptions)

    def test_read_failure_with_successful_close_stays_the_original(self):
        primary = OSError("read primary")
        error, events = self.exercise(primary=primary)
        self.assertIs(error, primary)
        self.assertEqual(events, ["open", "read", "drain"])

    def test_close_only_failure_prevents_build(self):
        cleanup = OSError("close primary")
        error, events = self.exercise(cleanup=cleanup)
        self.assertIs(error, cleanup)
        self.assertEqual(events, ["open", "read", "drain"])

    def test_failed_open_is_drained_and_no_build_runs(self):
        primary = OSError("open primary")
        error, events = self.exercise(opening=primary)
        self.assertIs(error, primary)
        self.assertEqual(events, ["open", "drain"])


if __name__ == "__main__":
    unittest.main()
