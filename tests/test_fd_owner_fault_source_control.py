"""Stdlib-only checks of the maintained Linux FD worker boundary/protocol.

These checks never load the provider, ctypes, shim or a numerical SDK and do not
stand in for the ROOT-only compiler/native evidence.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "tests/native/fd_owner_fault_native.py"
SHIM = ROOT / "tests/native/fd_close_fault.c"
HEADER = SHIM.with_suffix(".h")


def worker_api():
    """Actual cleanup bodies with labeled stdlib substitutes, no C/SDK load."""
    tree = ast.parse(WORKER.read_text())
    selected = [node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in {"error_name", "raise_grouped", "_transport", "raise_failures", "finish", "identity"}
                or isinstance(node, ast.ClassDef) and node.name == "Controls"]
    namespace = {"os": os, "contextmanager": contextmanager, "unittest": unittest}
    exec(compile(ast.Module(selected, []), str(WORKER), "exec"), namespace)
    return namespace






class Controls(unittest.TestCase):
    def test_fixed_width_native_protocol_matches_independent_python_declaration(self):
        header_names = re.findall(r"int64_t\s+(\w+)\s*;", HEADER.read_text())
        assignment = next(node for node in ast.parse(WORKER.read_text()).body
                          if isinstance(node, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "STATS_FIELDS" for x in node.targets))
        python_names = list(ast.literal_eval(assignment.value))
        self.assertEqual(header_names, python_names)
        self.assertEqual(len(header_names), 14)
        self.assertIn("14 * sizeof(int64_t)", SHIM.read_text())
        self.assertIn("library.tf_fd_fault_stats_size() != ctypes.sizeof(Stats)", WORKER.read_text())

    def test_actual_consumption_and_natural_same_number_reuse_are_required(self):
        source = SHIM.read_text()
        close_at = source.index("result = real_close(descriptor);")
        witness_at = source.index("fcntl(descriptor, F_GETFD)")
        replacement_at = source.index("result = openat(AT_FDCWD, replacement_path")
        self.assertLess(close_at, witness_at)
        self.assertLess(witness_at, replacement_at)
        self.assertIn('dlsym(RTLD_NEXT, "close")', source)
        self.assertIn("else if (result != descriptor)", source)
        self.assertNotRegex(source, r"\bdup[23]?\s*\(")
        self.assertIn("return real_close(descriptor);", source)
        self.assertIn("pthread_equal(arming_thread, pthread_self())", source)
        self.assertIn("target_identity.st_ino", source)

    def test_real_SIGINT_and_all_native_boundaries_are_explicit(self):
        source = WORKER.read_text()
        tree = ast.parse(source)
        self.assertIn("os.kill(os.getpid(), signal.SIGINT)", source)
        self.assertIn("signal.default_int_handler(number, frame)", source)
        self.assertNotRegex(source, r"raise\s+KeyboardInterrupt")
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Controls")
        names = [node.name for node in cls.body if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")]
        self.assertEqual(len(names), 6)
        self.assertTrue(any("pre_C_entry" in name for name in names))
        self.assertTrue(any("post_C_return" in name for name in names))
        self.assertTrue(any("journaled_open_before_result_store" in name for name in names))
        self.assertIn('self.signal_close("c_call")', source)
        self.assertIn('self.signal_close("c_return")', source)
        self.assertIn("monitor.free_tool_id(tool)", source)










    def test_root_guard_precedes_native_load_and_no_SDK_is_imported(self):
        forbidden = {"torch", "numpy", "triton", "mlx", "cupy", "cuda", "jax"}
        for path in (WORKER,):
            tree = ast.parse(path.read_text())
            guards = [node.lineno for node in tree.body if isinstance(node, ast.If)
                      and "TENSORFOLD_ROOT_REMOTE_QUALIFICATION" in ast.unparse(node.test)]
            self.assertEqual(len(guards), 1)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self.assertTrue(all(name.name.split(".")[0] not in forbidden for name in node.names))
                    if any(name.name in {"ctypes", "subprocess"} for name in node.names):
                        self.assertGreater(node.lineno, guards[0])
                elif isinstance(node, ast.ImportFrom):
                    self.assertNotIn((node.module or "").split(".")[0], forbidden)
        self.assertIn('os.environ.get("LD_PRELOAD") != str(shim)', WORKER.read_text())
        self.assertFalse(WORKER.name.startswith("test_"), "explicit Root worker must not enter ordinary pytest discovery")


    def test_worker_fault_releases_real_independent_sentinel_when_marked_cleanup_fails(self):
        api, sentinels = worker_api(), []
        cleanup = OSError("labeled reset failure; no native provider")
        with tempfile.TemporaryDirectory() as directory:
            original, sentinel = Path(directory) / "original", Path(directory) / "sentinel"
            original.write_bytes(b"original")
            sentinel.write_bytes(b"sentinel")
            class LabeledOwner:
                def __init__(self):
                    self.fd, self.closed = -1, True
                def open(self, path, flags):
                    self.fd, self.closed = os.open(path, flags), False
                def fileno(self):
                    return self.fd
                def close(self):
                    os.close(self.fd)
                    self.closed = True
                    # fault() requires a native injected error; this substitute
                    # intentionally fails that outcome assertion before cleanup.
            def opened(path, flags):
                fd = os.open(path, flags)
                sentinels.append(fd)
                return fd
            api["os"] = SimpleNamespace(open=opened, close=os.close, fstat=os.fstat,
                                        O_RDONLY=os.O_RDONLY, O_CLOEXEC=os.O_CLOEXEC)
            owner = api["Controls"]("test_actual_consumed_EIO_with_same_number_reuse_and_no_retry")
            owner.original, owner.sentinel = original, sentinel
            owner.api = SimpleNamespace(OwnedFD=LabeledOwner)
            owner.journal = []
            owner.arm = lambda *args: None
            def reset_failure(*args):
                raise cleanup
            owner.cleanup = reset_failure
            try:
                with self.assertRaises(AssertionError) as caught:
                    owner.fault(5)
                self.assertEqual(len(sentinels), 1)
                with self.assertRaises(OSError) as closed:
                    os.fstat(sentinels[0])
                self.assertEqual(closed.exception.errno, 9)
                self.assertIs(BaseException.__cause__.__get__(caught.exception), cleanup)
            finally:
                for fd in sentinels:
                    try:
                        os.fstat(fd)
                    except OSError:
                        continue
                    os.close(fd)

    def test_worker_finish_attempts_every_restore_without_opaque_hooks(self):
        api, called = worker_api(), []
        class Opaque(RuntimeError):
            def add_note(self, value):
                raise AssertionError("opaque annotation must not run")
            def __str__(self):
                raise AssertionError("opaque formatting must not run")
        primary, first, second, existing = Opaque(), OSError("restore1"), OSError("restore2"), ValueError("prior")
        BaseException.__cause__.__set__(primary, existing)
        def restore(index, error=None):
            called.append(index)
            if error is not None:
                raise error
        with self.assertRaises(Opaque) as caught:
            api["finish"](primary, [lambda: restore(1, first), lambda: restore(2, primary),
                                   lambda: restore(3, second), lambda: restore(4)])
        self.assertIs(caught.exception, primary)
        self.assertEqual(called, [1, 2, 3, 4])
        self.assertEqual(BaseException.__cause__.__get__(primary).exceptions, (existing, first, second))
        self.assertNotIn("primary.add_note(", WORKER.read_text())


    def test_protocol2_slot_is_journaled_before_acquisition_and_close_errors_are_explicit(self):
        api = worker_api()
        events = []
        owner = api["Controls"]("test_actual_consumed_EIO_with_same_number_reuse_and_no_retry")
        owner.journal = []
        class Slot:
            def __init__(self):
                events.append("closed-slot")
                self.closed = True
            def open(self, *args):
                self.assert_published = any(item is self for item in owner.journal)
                if not self.assert_published:
                    raise AssertionError("acquired before operation journal publication")
                events.append("journaled-open")
                self.closed = False
        owner.api = SimpleNamespace(OwnedFD=Slot)
        slot = owner.new_slot()
        self.assertEqual(events, ["closed-slot"])
        self.assertIs(owner.journal[0], slot)
        slot.open("labeled", 0)
        self.assertEqual(events, ["closed-slot", "journaled-open"])
        source = WORKER.read_text()
        self.assertIn("module._ownership_version != 2", source)
        self.assertIn("for code in (0, errno.EIO, errno.EINTR)", source)
        self.assertIn("raise_failures(interrupted.exception, [cleanup.exception])", source)
        self.assertIn("unraisable_operation_credit=False", source)

    def test_existing_authenticated_provider_is_reused_without_a_second_load(self):
        import threading
        node = next(node for node in ast.parse(WORKER.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "load_suppliers")
        expected = "e4f9e18724e1747d8b6ce5a67d8d228fa08fa14f695739acfd0b5bd1629d8db0"
        class Function:
            def __call__(self):
                return 112
        library = SimpleNamespace(tf_fd_fault_arm=Function(), tf_fd_fault_snapshot=Function(),
                                  tf_fd_fault_reset=Function(), tf_fd_fault_stats_size=Function(),
                                  close=object())
        ctypes = SimpleNamespace(CDLL=lambda *a, **k: library, c_int=object(), c_char_p=object(),
                                 c_uint=object(), c_void_p=object(), POINTER=lambda value: value,
                                 sizeof=lambda value: 112, cast=lambda *a: SimpleNamespace(value=1))
        class NoLoad:
            def __getattr__(self, name):
                raise AssertionError("second provider load attempted: " + name)
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            provider, shim, source = [directory / part for part in ("_fd_owner.abi3.so", "shim.so", "_fd_owner.c")]
            for path in (provider, shim, source):
                path.write_bytes(b"labeled source fixture")
            args = SimpleNamespace(provider=provider, shim=shim, source=source,
                                   provider_sha256="provider", shim_sha256="shim", source_sha256=expected)
            module = SimpleNamespace(_ownership_version=2, _limited_api=0x030B0000,
                                     _source_sha256=expected, __file__=str(provider))
            namespace = {"Path": Path, "sha": lambda path: expected if path == source else "provider" if path == provider else "shim",
                         "EXPECTED_SOURCE": expected, "ctypes": ctypes, "Stats": object(),
                         "threading": threading, "sys": sys, "os": os, "importlib": NoLoad()}
            exec(compile(ast.Module([node], []), str(WORKER), "exec"), namespace)
            previous = os.environ.get("LD_PRELOAD")
            try:
                os.environ["LD_PRELOAD"] = str(shim)
                actual, _, metadata = namespace["load_suppliers"](args, provider_module=module)
                self.assertIs(actual, module)
                self.assertTrue(metadata["reused_approved_provider_module"])
                module._ownership_version = 1
                with self.assertRaisesRegex(ValueError, "metadata differs"):
                    namespace["load_suppliers"](args, provider_module=module)
            finally:
                if previous is None:
                    os.environ.pop("LD_PRELOAD", None)
                else:
                    os.environ["LD_PRELOAD"] = previous


    def test_finish_retains_both_prior_roots_before_mutating_foreign_cleanup(self):
        api = worker_api()
        primary = KeyboardInterrupt("operation")
        cause, context, cleanup = ValueError("prior cause"), RuntimeError("prior context"), OSError("cleanup")
        BaseException.__cause__.__set__(primary, cause)
        BaseException.__context__.__set__(primary, context)
        retired = []
        def mutation():
            BaseException.__cause__.__set__(primary, None)
            BaseException.__context__.__set__(primary, None)
            retired.append("mutating callback")
            raise cleanup
        with self.assertRaises(KeyboardInterrupt) as caught:
            api["finish"](primary, [mutation, lambda: retired.append("independent release")])
        self.assertIs(caught.exception, primary)
        self.assertEqual(retired, ["mutating callback", "independent release"])
        self.assertEqual(BaseException.__cause__.__get__(primary).exceptions, (cause, context, cleanup))
        BaseException.__cause__.__set__(primary, cause)
        BaseException.__context__.__set__(primary, context)
        allocation = MemoryError("labeled group publication")
        def failed_group(*args):
            raise allocation
        api["BaseExceptionGroup"] = failed_group
        with self.assertRaises(KeyboardInterrupt) as caught:
            api["finish"](primary, [mutation, lambda: retired.append("second independent release")])
        self.assertIs(caught.exception, primary)
        self.assertIs(BaseException.__cause__.__get__(primary), allocation)
        dictionary = BaseException.__dict__["__dict__"].__get__(primary, type(primary))
        self.assertEqual(dictionary["_tensorfold_retained_failures"], [cause, context, cleanup])

    def test_instruction_callback_arms_requested_integer_errno_without_shadowing(self):
        tree = ast.parse(WORKER.read_text())
        outer = next(node for node in ast.walk(tree)
                     if isinstance(node, ast.FunctionDef) and node.name == "journaled_open_interrupt")
        witness = next(node for node in outer.body
                       if isinstance(node, ast.FunctionDef) and node.name == "witness")
        def acquire():
            return None
        for requested in (0, 5, 4):
            armed, sent, fired, acquired = [], [], [], []
            scope = {"acquire": acquire, "store": 18, "code": requested,
                     "expected_identity": (11, 13), "fired": fired, "acquired": acquired,
                     "descriptors": lambda: {7: (11, 13, 0), 9: (17, 19, 0)},
                     "self": SimpleNamespace(assertEqual=self.assertEqual, arm=lambda fd, code: armed.append((fd, code))),
                     "os": SimpleNamespace(getpid=lambda: 123, kill=lambda pid, signum: sent.append((pid, signum))),
                     "signal": SimpleNamespace(SIGINT=2)}
            exec(compile(ast.Module([witness], []), str(WORKER), "exec"), scope)
            scope["witness"](acquire.__code__, 18)
            self.assertEqual(armed, [(7, requested)])
            self.assertIs(type(armed[0][1]), int)
            self.assertEqual(sent, [(123, 2)])




if __name__ == "__main__":
    unittest.main(verbosity=2)
