"""Execute current loader ownership controls with strict foreign-provider fakes.

No tensor runtime is imported or numerical result claimed. Independent original
arithmetic and packing proofs remain in the archived task evidence; these tests
retain the current failure, interruption, drain and clean-success contracts.
"""

import ast
import contextlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
PATH = REPO / "src/tensorfold/families/qwen3_5/cuda"


class Injected(RuntimeError):
    pass


class DrainFailure(RuntimeError):
    pass


class Value:
    ndim = 2
    dtype = "bf16"
    shape = (2, 2)

    def contiguous(self):
        return self

    def float(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def __add__(self, other):
        return self

    def __truediv__(self, other):
        return self

    def __neg__(self):
        return self

    def __rpow__(self, other):
        return self


class OwnershipControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        (self.home / "config.json").write_text("{}")
        self.closed = 0
        self.failure_mode = None
        self.close_fail = False
        self.extra = False
        self.read_error = None
        self.close_error = None
        self.close_action = None
        self.transport_globals = {}
        outer = self

        class Reader:
            def __init__(inner, *a, **kw):
                inner.names = (
                    {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
                    if outer.loader == "weights.py"
                    else {
                        "model.language_model.embed_tokens.weight",
                        "model.language_model.norm.weight",
                        "lm_head.weight",
                    }
                )
                if outer.extra:
                    inner.names.add("model.unused")

            def __iter__(inner):
                return iter(inner.names)

            def __contains__(inner, name):
                return name in inner.names

            def pop(inner, name):
                if outer.failure_mode == "read":
                    raise outer.read_error if outer.read_error is not None else Injected("read fault")
                if outer.failure_mode == "interrupt":
                    raise KeyboardInterrupt()
                inner.names.discard(name)
                return Value()

            def close(inner):
                outer.closed += 1
                if outer.close_action is not None:
                    outer.close_action()
                if outer.close_error is not None:
                    raise outer.close_error
                if outer.close_fail:
                    raise DrainFailure("drain fault")

        self.reader = Reader
        self.cfg = types.SimpleNamespace(layers=0, rope_dims=4, rope_theta=10000, hidden=2)
        self.modules = {}

        def module(name, **kw):
            value = types.ModuleType(name)
            value.__dict__.update(kw)
            self.modules[name] = value
            return value

        module("tensorfold.families.qwen3_5.cuda.exl3_load", quant_config=lambda _: None, load_exl3=lambda *a: None)
        module("tensorfold.families.qwen3_5.cuda.nvfp4_load", quantized=lambda _: False, load_nvfp4=lambda *a: None)
        module("tensorfold.quantization", resolve_affine=lambda *a: None, validate_shapes=lambda *a: None)
        module("tensorfold.cuda", prompt_precision=types.SimpleNamespace(fp8=lambda: False))
        module(
            "tensorfold.cuda.capacity",
            headers=lambda _: {
                "model.language_model.embed_tokens.weight": {"dtype": "BF16", "shape": [2, 2]},
                "model.language_model.norm.weight": {"dtype": "BF16", "shape": [2]},
                "lm_head.weight": {"dtype": "BF16", "shape": [2, 2]},
            },
        )
        fmt = module("tensorfold.cuda.nvfp4.format", config_block=lambda _: None, scheme=lambda _: "bf16")
        module("tensorfold.cuda.nvfp4", format=fmt)
        module("tensorfold.cuda.nvfp4.linear", Fp4Linear=object, Fp8Linear=object, Staging=lambda: None)

        def plain(*a, **kw):
            if outer.failure_mode == "materialize":
                raise Injected("materialization fault")
            return types.SimpleNamespace()

        module(
            "tensorfold.families.qwen3_5.cuda.weights",
            _Tensors=Reader,
            Config=types.SimpleNamespace(read=lambda _: self.cfg),
            GDN=plain,
            Attention=plain,
            Layer=plain,
            Weights=lambda *a, **kw: types.SimpleNamespace(layers=[]),
            Plain=plain,
        )
        self.torch = types.SimpleNamespace(
            bfloat16="bf16",
            float16="f16",
            float32="f32",
            float64="f64",
            int32="i32",
            uint32="u32",
            arange=lambda *a, **kw: Value(),
            cuda=types.SimpleNamespace(empty_cache=lambda: None),
        )

    def execute(self, name):
        self.loader = name
        path = PATH / name
        tree = ast.parse(path.read_text())
        function = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("load", "load_nvfp4")))
        helper_tree = ast.parse((PATH / "weights.py").read_text())
        helpers = [n for n in helper_tree.body if isinstance(n, ast.FunctionDef)
                   and n.name in ("_close_failed_checkpoint_impl", "_close_failed_checkpoint")]
        module = ast.Module(body=[*helpers, function], type_ignores=[])

        def weights(*a, **kw):
            if self.failure_mode == "materialize":
                raise Injected("materialization fault")
            return types.SimpleNamespace(layers=[])

        scope = {
            "__name__": "tensorfold.families.qwen3_5.cuda.test",
            "__package__": "tensorfold.families.qwen3_5.cuda",
            "Path": Path,
            "json": json,
            "checkpoint_path": lambda root, name: Path(root) / name,
            "read_metadata_json": lambda path: json.loads(path.read_text()),
            "torch": self.torch,
            "_Tensors": self.reader,
            "Config": types.SimpleNamespace(read=lambda _: self.cfg),
            "QLinear": weights,
            "Weights": weights,
            "maths": lambda: ({"nvfp4": False, "fp8": False}, "test"),
            "SUFFIXES": ("weight",),
            "Plain8": weights,
            "Plain": weights,
            "skipped": lambda _: False,
        }
        scope.update(self.transport_globals)
        with patch.dict(sys.modules, self.modules), contextlib.redirect_stdout(__import__("io").StringIO()):
            exec(compile(module, str(path), "exec", flags=__import__("__future__").annotations.compiler_flag), scope)
            self.modules["tensorfold.families.qwen3_5.cuda.weights"]._close_failed_checkpoint = scope["_close_failed_checkpoint"]
            return scope[function.name](self.home, device="cpu")

    def test_group_allocation_failure_retains_primary_native_and_reader_statuses(self):
        for name in ("weights.py", "nvfp4_load.py"):
            with self.subTest(name=name):
                self.closed, self.failure_mode = 0, "read"
                primary, native, context = Injected(), LookupError(), EOFError()
                cleanup, allocation = OSError(), MemoryError("group allocation")
                primary.__cause__, primary.__context__ = native, context
                self.read_error, self.close_error = primary, cleanup
                def erase():
                    primary.__cause__ = primary.__context__ = None
                self.close_action = erase
                def fail(*args):
                    raise allocation
                self.transport_globals = {"BaseExceptionGroup": fail}
                try:
                    self.execute(name)
                except BaseException as actual:
                    self.assertIs(actual, primary)
                    self.assertIs(actual.__cause__, allocation)
                    trace, frames = actual.__traceback__, []
                    while trace is not None:
                        frames.append(trace.tb_frame)
                        trace = trace.tb_next
                    trace = allocation.__traceback__
                    while trace is not None:
                        frames.append(trace.tb_frame)
                        trace = trace.tb_next
                    outer = next(f.f_locals for f in frames if f.f_code.co_name == "_close_failed_checkpoint")
                    self.assertIs(outer["native_cause"], native)
                    self.assertIs(outer["native_context"], context)
                    core = next(f.f_locals for f in frames if f.f_code.co_name == "_close_failed_checkpoint_impl")
                    self.assertEqual(core["others"], [native, context, cleanup])
                else:
                    self.fail("group allocation replaced reader failure")
                self.assertEqual(self.closed, 1)

    def test_post_close_cold_preparation_fault_preserves_primary_native_fields(self):
        tree = ast.parse((PATH / "weights.py").read_text())
        core = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "_close_failed_checkpoint_impl")
        line = next(n.lineno for n in ast.walk(core) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "others" for t in n.targets))
        for name in ("weights.py", "nvfp4_load.py"):
            with self.subTest(name=name):
                self.closed, self.failure_mode = 0, "read"
                primary, native, context = Injected(), LookupError(), EOFError()
                cleanup, allocation = OSError(), MemoryError("post-close preparation")
                primary.__cause__, primary.__context__ = native, context
                self.read_error, self.close_error = primary, cleanup
                def erase():
                    primary.__cause__ = primary.__context__ = None
                self.close_action = erase
                def trace_fault(frame, event, arg):
                    if (event == "line" and frame.f_code.co_name == "_close_failed_checkpoint_impl"
                            and frame.f_lineno == line):
                        raise allocation
                    return trace_fault
                previous_trace = sys.gettrace()
                try:
                    sys.settrace(trace_fault)
                    try:
                        self.execute(name)
                    except BaseException as actual:
                        self.assertIs(actual, primary)
                        self.assertIs(actual.__cause__, allocation)
                        trace = actual.__traceback__
                        while trace is not None and trace.tb_frame.f_code.co_name != "_close_failed_checkpoint":
                            trace = trace.tb_next
                        self.assertIsNotNone(trace)
                        self.assertIs(trace.tb_frame.f_locals["native_cause"], native)
                        self.assertIs(trace.tb_frame.f_locals["native_context"], context)
                        self.assertIs(allocation.__context__, cleanup)
                    else:
                        self.fail("cold preparation replaced primary")
                finally:
                    sys.settrace(previous_trace)
                self.assertEqual(self.closed, 1)

    def test_malformed_notes_preserves_primary_prior_native_cause_and_cleanup(self):
        for name in ("weights.py", "nvfp4_load.py"):
            with self.subTest(name=name):
                self.closed, self.failure_mode = 0, "read"
                primary, native, cleanup = KeyboardInterrupt(), LookupError(), OSError()
                primary.__cause__, primary.__notes__ = native, 123
                self.read_error, self.close_error = primary, cleanup
                try:
                    self.execute(name)
                except BaseException as actual:
                    self.assertIs(actual, primary)
                    self.assertIsInstance(actual.__cause__, BaseExceptionGroup)
                    causes = actual.__cause__.exceptions
                    self.assertIn(native, causes)
                    self.assertIn(cleanup, causes)
                    self.assertTrue(any(type(error) is TypeError for error in causes))
                else:
                    self.fail("loader discarded primary failure")
                self.assertEqual(self.closed, 1)

    def test_same_primary_foreign_close_cannot_erase_captured_native_context(self):
        for name in ("weights.py", "nvfp4_load.py"):
            with self.subTest(name=name):
                self.closed, self.failure_mode = 0, "read"
                primary, native, context = KeyboardInterrupt(), LookupError(), ArithmeticError()
                primary.__cause__, primary.__context__ = native, context
                def erase():
                    primary.__cause__ = primary.__context__ = None
                self.read_error = self.close_error = primary
                self.close_action = erase
                try:
                    self.execute(name)
                except BaseException as actual:
                    self.assertIs(actual, primary)
                    self.assertIsInstance(actual.__cause__, BaseExceptionGroup)
                    self.assertEqual(actual.__cause__.exceptions, (native, context))
                else:
                    self.fail("loader discarded primary failure")
                self.assertEqual(self.closed, 1)

    def test_same_primary_close_failure_never_creates_a_self_cause(self):
        for name in ("weights.py", "nvfp4_load.py"):
            with self.subTest(name=name):
                self.closed, self.failure_mode = 0, "read"
                primary = KeyboardInterrupt()
                self.read_error = self.close_error = primary
                try:
                    self.execute(name)
                except BaseException as actual:
                    self.assertIs(actual, primary)
                    self.assertIsNot(actual.__cause__, primary)
                else:
                    self.fail("loader discarded primary failure")
                self.assertEqual(self.closed, 1)

    def test_native_cause_and_implicit_context_survive_distinct_close_failure(self):
        for name in ("weights.py", "nvfp4_load.py"):
            with self.subTest(name=name):
                self.closed, self.failure_mode = 0, "read"
                primary, native, context, cleanup = Injected(), LookupError(), ArithmeticError(), OSError()
                primary.__cause__, primary.__context__ = native, context
                self.read_error, self.close_error = primary, cleanup
                try:
                    self.execute(name)
                except BaseException as actual:
                    self.assertIs(actual, primary)
                    self.assertIsInstance(actual.__cause__, BaseExceptionGroup)
                    causes = actual.__cause__.exceptions
                    self.assertIn(native, causes)
                    self.assertIn(context, causes)
                    self.assertIn(cleanup, causes)
                else:
                    self.fail("loader discarded primary failure")
                self.assertEqual(self.closed, 1)

    def test_opaque_same_primary_annotation_failure_cannot_mask_the_original(self):
        class Opaque(Injected):
            def __str__(self):
                raise AssertionError("primary diagnostics must not format foreign errors")
            def __getattribute__(self, name):
                if name == "__notes__":
                    raise self
                return super().__getattribute__(name)
        for name in ("weights.py", "nvfp4_load.py"):
            with self.subTest(name=name):
                self.closed, self.failure_mode = 0, "read"
                primary, cleanup = Opaque(), OSError()
                self.read_error, self.close_error = primary, cleanup
                try:
                    self.execute(name)
                except BaseException as actual:
                    self.assertIs(actual, primary)
                    self.assertIsNot(actual.__cause__, primary)
                    self.assertIsNot(actual.__cause__, None)
                else:
                    self.fail("loader discarded primary failure")
                self.assertEqual(self.closed, 1)

    def test_read_failure_drains_once_preserves_error(self):
        for name in ("weights.py", "nvfp4_load.py"):
            with self.subTest(name=name):
                self.closed = 0
                self.failure_mode = "read"
                with self.assertRaises(Injected):
                    self.execute(name)
                self.assertEqual(self.closed, 1)

    def test_materialization_failure_drains_once(self):
        for name in ("weights.py", "nvfp4_load.py"):
            with self.subTest(name=name):
                self.closed = 0
                self.failure_mode = "materialize"
                with self.assertRaises(Injected):
                    self.execute(name)
                self.assertEqual(self.closed, 1)

    def test_cleanup_failure_preserves_primary_and_cause(self):
        for name in ("weights.py", "nvfp4_load.py"):
            with self.subTest(name=name):
                self.closed = 0
                self.failure_mode = "read"
                self.close_fail = True
                with self.assertRaises(Injected) as got:
                    self.execute(name)
                self.assertIsInstance(got.exception.__cause__, DrainFailure)
                self.assertEqual(self.closed, 1)
                self.assertTrue(got.exception.__notes__)

    def test_valid_reader_always_closes_before_unused_validation(self):
        for name in ("weights.py", "nvfp4_load.py"):
            with self.subTest(name=name):
                self.closed = 0
                self.failure_mode = None
                self.extra = True
                with self.assertRaisesRegex(ValueError, "unused checkpoint tensors"):
                    self.execute(name)
                self.assertEqual(self.closed, 1)

    def test_normal_close_failure_is_operation_failure(self):
        for name in ("weights.py", "nvfp4_load.py"):
            with self.subTest(name=name):
                self.closed = 0
                self.failure_mode = None
                self.close_fail = True
                with self.assertRaises(DrainFailure):
                    self.execute(name)
                self.assertEqual(self.closed, 1)

    def test_clean_valid_source_flow_closes_once(self):
        for name in ("weights.py", "nvfp4_load.py"):
            with self.subTest(name=name):
                self.closed = 0
                self.failure_mode = None
                self.extra = False
                result = self.execute(name)
                self.assertTrue(hasattr(result, "inv_freq"))
                self.assertEqual(self.closed, 1)

    def test_interruption_also_drains_reader(self):
        for name in ("weights.py", "nvfp4_load.py"):
            with self.subTest(name=name):
                self.closed = 0
                self.failure_mode = "interrupt"
                with self.assertRaises(KeyboardInterrupt):
                    self.execute(name)
                self.assertEqual(self.closed, 1)


if __name__ == "__main__":
    unittest.main()
