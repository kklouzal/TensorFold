"""CPU fixture/API/source ownership controls, not CUDA graph qualification."""
import ast
import copy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BOUND = ROOT / "tests/cuda/test_flashnext_sm70_bound_graphs.py"
MTP = ROOT / "tests/cuda/test_flashnext_sm70_mtp.py"


def function(path, *names):
    node = ast.parse(path.read_text())
    for name in names:
        node = next(n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    return copy.deepcopy(node)


def compile_function(node, namespace=None):
    node.decorator_list = []
    scope = {} if namespace is None else dict(namespace)
    tree = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(tree, str(BOUND), "exec"), scope)
    return scope[node.name]


class Buffer:
    def __init__(self, name):
        self.name = name

    def __getitem__(self, index):
        return Buffer((self.name, index))

    def clone(self):
        return Buffer((self.name, "snapshot"))


class ObservedConstructor(Exception):
    pass


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "src/tensorfold/families/qwen4_exp/cuda/speculation.py"
        cls.module_name = "_sm70_bound_fixture_geometry"
        specification = importlib.util.spec_from_file_location(cls.module_name, path)
        cls.geometry_module = importlib.util.module_from_spec(specification)
        sys.modules[cls.module_name] = cls.geometry_module
        specification.loader.exec_module(cls.geometry_module)

    @classmethod
    def tearDownClass(cls):
        del sys.modules[cls.module_name]

    def traced(self, forward):
        decoder = NS(_mtp=forward, mbuf=NS(mtp_in=Buffer("inputs"), streams=Buffer("streams")))
        state = NS(mtp_kc=NS(**{name: Buffer(name) for name in ("k", "v", "ks", "vs")}),
                   mtp_len=8, ratio=4, mtp_pooled=Buffer("pooled"))
        frames = []
        install = compile_function(function(BOUND, "_qualify_singleton_draft_protocol", "trace_steps"))
        install(decoder, frames)
        return decoder, [(state, 0, 2)], frames

    def test_wrapper_forwards_default_and_explicit_keywords_and_keeps_maintenance_frame(self):
        calls, output, next_inputs = [], Buffer("logits"), object()
        def forward(segs, *, logits=True, next_inputs=None):
            calls.append((segs, logits, next_inputs))
            return output if logits else None
        decoder, segments, frames = self.traced(forward)
        self.assertIs(decoder._mtp(segments), output)
        self.assertIsNone(decoder._mtp(segments, logits=False, next_inputs=next_inputs))
        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0][0], segments)
        self.assertEqual(calls[0][1:], (True, None))
        self.assertIs(calls[1][2], next_inputs)
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0][8:], (True, False))
        self.assertEqual(frames[1][8:], (False, True))
        self.assertIsInstance(frames[0][7], Buffer)
        self.assertIsNone(frames[1][7])
        # Maintenance retains every cache/input/stream field, not just a flag.
        self.assertTrue(all(isinstance(field, Buffer) for field in frames[1][:7]))

    def test_wrapper_rejects_none_logits_or_unrequested_logits_without_recording_false_frame(self):
        for requested, result in ((True, None), (False, Buffer("unexpected"))):
            decoder, segments, frames = self.traced(lambda *a, **k: result)
            with self.subTest(requested=requested), self.assertRaises(AssertionError):
                decoder._mtp(segments, logits=requested)
            self.assertEqual(frames, [])

    def test_wrapper_preserves_authentic_forward_error_and_unknown_keyword_refusal(self):
        error = RuntimeError("original method failed")
        def fail(*args, **kwargs):
            raise error
        decoder, segments, frames = self.traced(fail)
        with self.assertRaises(RuntimeError) as caught:
            decoder._mtp(segments, logits=False)
        self.assertIs(caught.exception, error)
        self.assertEqual(frames, [])
        with self.assertRaises(TypeError):
            decoder._mtp(segments, unsupported=True)

    def test_frame_comparison_remains_exact_for_all_tensor_fields_and_none_only_maintenance(self):
        helper = function(BOUND, "_qualify_singleton_draft_protocol")
        outer = next(n for n in ast.walk(helper) if isinstance(n, ast.For)
                     and isinstance(n.target, ast.Tuple) and ast.unparse(n.target).startswith("(step,"))
        calls = []
        def close(a, b, **kwargs):
            self.assertEqual(a, b)
            self.assertEqual((kwargs["rtol"], kwargs["atol"]), (0, 0))
            calls.append(kwargs["msg"])
        eager = [(*range(8), True, False), (*range(7), None, False, True)]
        scope = {"torch": NS(testing=NS(assert_close=close)), "actual_frames": eager,
                 "expected_frames": copy.deepcopy(eager), "iteration": 0}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[outer], type_ignores=[])), str(BOUND), "exec"), scope)
        self.assertEqual(len(calls), 15)  # eight projection fields + seven maintenance fields
        self.assertTrue(any("streams" in msg for msg in calls))
        wrong = copy.deepcopy(eager)
        wrong[1] = (*range(7), None, True, True)
        scope["expected_frames"] = wrong
        with self.assertRaises(AssertionError):
            exec(compile(ast.fix_missing_locations(ast.Module(body=[outer], type_ignores=[])), str(BOUND), "exec"), scope)

    def test_growth_fixtures_request_legal_distinct_method_capabilities(self):
        geometry = self.geometry_module.MethodGeometry
        for name, mtp, depth, capacity in (("test_bound_owner_is_invalidated_before_growth_replaces_cache_storage", False, 0, 64),
                                           ("test_bound_floor_prefix_copy_and_growth_cross_the_8192_boundary", True, 2, 16384)):
            weights, observed = NS(mtp=object() if mtp else None), []
            def decoder(w, **kwargs):
                policy = geometry("mtp" if w.mtp is not None else "none", kwargs["depth"])
                observed.append((w, kwargs, policy.target_tail_rows))
                raise ObservedConstructor
            module = ModuleType("test_flashnext_sm70_mtp")
            module._weights_with_stock_mtp_protocol = lambda: weights
            namespace = {"torch": NS(cuda=NS(get_device_capability=lambda: (7, 0))),
                         "pytest": NS(skip=lambda _: self.fail("fixture must not skip")),
                         "_small_weights": lambda **kwargs: weights, "MultiDecoder": decoder}
            operation = compile_function(function(BOUND, name), namespace)
            with patch.dict(sys.modules, {module.__name__: module}), self.assertRaises(ObservedConstructor):
                operation()
            self.assertIs(observed[0][0], weights)
            self.assertEqual(observed[0][1]["depth"], depth)
            self.assertEqual(observed[0][1]["capacity"], capacity)
            self.assertEqual(observed[0][2], 3 if mtp else 1)
            if mtp:
                self.assertLessEqual(2, observed[0][2])  # original two live + third padded row remains legal
        with self.assertRaises(ValueError):
            geometry("none", 2)

    def test_stock_mtp_builder_preserves_target_weight_objects(self):
        builder = function(MTP, "_weights_with_stock_mtp_protocol")
        creation = next(n for n in builder.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                        and n.targets[0].id == "w")
        self.assertEqual(ast.unparse(creation.value), "_small_weights(True)")
        # All target objects come from the original include_ple=True fixture.
        # Dataclasses.replace creates independent companion records. The only
        # attribute store in this builder is attaching the new companion itself.
        stores = [ast.unparse(n) for n in ast.walk(builder) if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)]
        self.assertEqual(stores, ["w.mtp"])
        self.assertFalse(any(isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store) for n in ast.walk(builder)))
        self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                             and n.func.attr.endswith("_") for n in ast.walk(builder)))
        source = ast.parse(MTP.read_text())
        self.assertTrue(any(isinstance(n, ast.ImportFrom) and n.module == "dataclasses"
                            and any(v.name == "replace" for v in n.names) for n in source.body))
        records = ast.parse((ROOT / "src/tensorfold/families/qwen4_exp/cuda/weight_types.py").read_text())
        for name in ("HC", "AttnW", "MoEW", "LayerW", "MTPW"):
            record = next(n for n in records.body if isinstance(n, ast.ClassDef) and n.name == name)
            self.assertIn("dataclass", [ast.unparse(n) for n in record.decorator_list])
            self.assertFalse(any(isinstance(n, ast.FunctionDef) and n.name in ("__init__", "__post_init__", "__setattr__")
                                 for n in record.body))

    def test_companion_dense_constructor_only_borrows_or_copies_target_weight(self):
        path = ROOT / "src/tensorfold/families/qwen4_exp/cuda/sm70_mtp_weights.py"
        tree = ast.parse(path.read_text())
        dense = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Dense")
        constructor = copy.deepcopy(next(n for n in dense.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"))
        half, calls = object(), []
        class ReadOnlyWeight:
            ndim, dtype, shape = 2, half, (3, 4)
            def __setattr__(self, name, value):
                raise AssertionError("constructor mutated target weight")
            def contiguous(self):
                calls.append("contiguous")
                return self
        weight, companion = ReadOnlyWeight(), NS()
        initialize = compile_function(constructor, {"torch": NS(float16=half)})
        initialize(companion, weight)
        self.assertIs(companion.weight, weight)
        self.assertEqual((companion.n, companion.k), (3, 4))
        self.assertEqual(calls, ["contiguous"])


if __name__ == "__main__":
    unittest.main()
