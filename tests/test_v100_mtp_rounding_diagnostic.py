"""CPU data/protocol/lifetime controls, not MTP arithmetic or graph proof."""
import copy
import json
from pathlib import Path
import tempfile
import subprocess
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

from tools import v100_mtp_rounding_diagnostic as diag


def observation(value="a"):
    tensor = {"shape": [2, 3], "dtype": "<f2", "sha256": value * 64, "file": "one.npy"}
    return {name: copy.deepcopy(tensor) for name in diag.LABELS} | {"cache": [{"k": copy.deepcopy(tensor)}]}


class Tensor:
    def __init__(self, value=b"data"):
        self.value = value
    def numel(self):
        return len(self.value)
    def element_size(self):
        return 1
    def is_contiguous(self):
        return True
    def copy_(self, other):
        self.value = other.value
        return self
    def data_ptr(self):
        return id(self)


def state():
    s = NS(compact_indexer=None, image_positions=None, rope_delta=0, pos=9, mtp_len=8, mtp_drafted=0,
           capacity=32, version=0, slot_generation=0, history_generation=0, kv_identity=(("int8", "int8"), "policy"), cur=[0])
    for name in ("rec", "conv", "ple_tail", "pos_dev", "mtp_pos", "history_generations", "kv_status", "mtp_ikc", "mtp_pooled"):
        setattr(s, name, Tensor(name.encode()))
    s.ikc, s.pooled = [Tensor(b"ikc")], [Tensor(b"pooled")]
    def cache():
        return NS(**{n: Tensor(n.encode()) for n in ("k", "v", "ks", "vs")})
    s.kc, s.mtp_kc = [cache()], cache()
    s.kv_check = lambda: None
    return s


class Contracts(unittest.TestCase):
    def test_canonical_inverse_last_rows_and_real_tail_widths(self):
        plan = diag.row_plan([5, 4, 3, 2, 1, 0], [1, 3, 2, 1, 2, 3], width=3)
        self.assertEqual(plan["inverse"], [15, 12, 13, 14, 9, 10, 6, 3, 4, 0, 1, 2])
        self.assertEqual(plan["segment_inverse"], [5, 4, 3, 2, 1, 0])
        self.assertEqual(plan["last"], [2, 4, 6, 10, 14, 15])
        self.assertEqual(len(set(plan["inverse"])), 12)
        self.assertEqual(diag.row_plan(list(range(6)), [1] * 6, width=1)["inverse"], list(range(6)))
        for slots, counts, width in (([0] * 6, [1] * 6, 3), (list(range(6)), [4] * 6, 3),
                                      (list(range(6)), [True] * 6, 1), (list(range(6)), [1] * 6, 2)):
            with self.assertRaises(ValueError):
                diag.row_plan(slots, counts, width=width)

    def test_content_ignores_filenames_but_preserves_bits_shapes_and_dtype(self):
        a, b = observation(), observation()
        b["q"]["file"] = "different-file.npy"
        self.assertIsNone(diag.first_difference(a, b))
        for name in ("shape", "dtype", "sha256"):
            changed = copy.deepcopy(b)
            changed["q"][name] = [3, 2] if name == "shape" else "<f4" if name == "dtype" else "b" * 64
            self.assertEqual(diag.first_difference(a, changed), "q")

    def test_repeated_eager_control_precedes_all_causal_classification(self):
        cells = {name: observation() for name in ("A", "A_repeat", "B", "C", "D")}
        self.assertEqual(diag.classify(cells)["axis"], "no_difference_observed")
        for target, axis in (("B", "context_extent"), ("C", "canonical_padding_order"), ("D", "graph_state_or_capture")):
            changed = copy.deepcopy(cells)
            changed[target]["attention_output"]["sha256"] = "c" * 64
            self.assertEqual(diag.classify(changed)["axis"], axis)
        cells["A_repeat"]["fc_h"]["sha256"] = "b" * 64
        cells["D"]["logits"]["sha256"] = "d" * 64
        self.assertEqual(diag.classify(cells)["axis"], "same_eager_not_reproduced")

    def test_copy_budget_refuses_before_allocation_and_has_no_gpu_credit(self):
        self.assertEqual(diag.copy_plan([2, 3, 4], budget=9)["host_snapshot_bytes"], 9)
        self.assertEqual(diag.copy_plan([2])["project_device_snapshot_bytes"], 0)
        for sizes, budget in (([], 4), ([5], 4), ([True], 4), ([-1], 4), ([2**63], 2**64)):
            with self.assertRaises(ValueError):
                diag.copy_plan(sizes, budget)

    def test_scoped_policy_restores_inherited_instance_bindings_and_errors(self):
        class Parent:
            value = "inherited"
        owner = Parent()
        primary = RuntimeError("original")
        with self.assertRaises(RuntimeError) as caught:
            with diag.attributes([(owner, "value", "temporary")]):
                self.assertEqual(owner.value, "temporary")
                raise primary
        self.assertIs(caught.exception, primary)
        self.assertEqual(owner.value, "inherited")
        self.assertNotIn("value", vars(owner))

    def test_policy_restores_every_binding_even_when_one_restore_fails(self):
        class Broken:
            def __init__(self):
                self.value = "old"
                self.fail = False
            def __setattr__(self, key, value):
                if key == "value" and value == "old" and getattr(self, "fail", False):
                    raise cleanup
                object.__setattr__(self, key, value)
        primary, cleanup = RuntimeError("primary"), OSError("restore")
        first, second = NS(value="first"), Broken()
        with self.assertRaises(RuntimeError) as caught:
            with diag.attributes([(first, "value", "new"), (second, "value", "new")]):
                second.fail = True
                raise primary
        self.assertIs(caught.exception, primary)
        self.assertIs(primary.__cause__, cleanup)
        self.assertEqual(first.value, "first")

    def test_frozen_state_restores_values_but_never_rebinds_or_erases_poison(self):
        s, events = state(), []
        runtime = NS(cuda=NS(synchronize=lambda: events.append("sync")),
                     empty_like=lambda t, device: Tensor(b"0" * t.numel()), is_tensor=lambda t: isinstance(t, Tensor))
        frozen = diag.Frozen([s], runtime)
        pointer, original = s.rec.data_ptr(), s.rec.value
        s.rec.value = b"changed"
        frozen.restore()
        self.assertEqual((s.rec.data_ptr(), s.rec.value), (pointer, original))
        s.slot_generation += 1
        with self.assertRaisesRegex(ValueError, "metadata"):
            frozen.restore()
        s.slot_generation -= 1
        s.rec = Tensor(b"rebound")
        with self.assertRaisesRegex(ValueError, "storage"):
            frozen.restore()
        error = RuntimeError("poison")
        s.kv_check = lambda: (_ for _ in ()).throw(error)
        with self.assertRaises(RuntimeError) as caught:
            frozen.restore()
        self.assertIs(caught.exception, error)
        self.assertTrue(events)

    def test_optional_state_inventory_preserves_none_absent_and_scalar_types(self):
        runtime = NS(is_tensor=lambda t: isinstance(t, Tensor))
        s = state()
        s.history_generations = None  # Actual plain tiny-State construction.
        del s.kv_status
        tensors, structure = diag.state_storage(s, runtime)
        self.assertNotIn("history_generations", tensors)
        self.assertNotIn("kv_status", tensors)
        self.assertEqual(structure["history_generations"], {"kind": "none"})
        self.assertEqual(structure["kv_status"], {"kind": "absent"})
        for value, expected_type in ((False, "bool"), (0, "int"), (-0., "float")):
            s.kv_status = value
            _, scalar = diag.state_storage(s, runtime)
            self.assertEqual(scalar["kv_status"], {"kind": "scalar", "type": expected_type,
                                                   "value": value.hex() if type(value) is float else value})
        s.kv_status = 0.
        self.assertNotEqual(scalar, diag.state_storage(s, runtime)[1])

    def test_optional_state_snapshot_bytes_exclude_non_tensors_and_refuse_structural_mutation(self):
        allocated = []
        def allocate(tensor, *, device):
            allocated.append((tensor, device))
            return Tensor(b"0" * tensor.numel())
        runtime = NS(is_tensor=lambda t: isinstance(t, Tensor), cuda=NS(synchronize=lambda: None), empty_like=allocate)
        s = state()
        s.history_generations, s.kv_status = None, 0
        tensors, structure = diag.state_storage(s, runtime)
        frozen = diag.Frozen([s], runtime)
        expected_bytes = sum(len(value.value) for value in tensors.values())
        self.assertEqual(frozen.bytes, expected_bytes)
        self.assertEqual(diag.copy_plan([value.numel() * value.element_size() for value in tensors.values()])["host_snapshot_bytes"], expected_bytes)
        self.assertEqual(len(allocated), len(tensors))
        self.assertEqual(frozen.structure, [structure])
        self.assertNotIn("history_generations", frozen.saved[0])
        frozen.restore()
        self.assertIsNone(s.history_generations)
        self.assertEqual(s.kv_status, 0)
        for changed in (None, False, 0., Tensor(b"new")):
            s.kv_status = changed
            with self.assertRaisesRegex(ValueError, "structure"):
                frozen.restore()
        s.kv_status = 0
        del s.history_generations
        with self.assertRaisesRegex(ValueError, "structure"):
            frozen.capture()

    def test_unsupported_state_fields_fail_before_any_copy_allocation(self):
        allocations = []
        runtime = NS(is_tensor=lambda t: isinstance(t, Tensor),
                     empty_like=lambda *args, **kwargs: allocations.append(args), cuda=NS(synchronize=lambda: None))
        for field, value in (("history_generations", object()), ("kv_status", []),
                             ("kv_status", float("nan")), ("kv_status", float("inf")),
                             ("rec", None), ("rec", 0)):
            s = state()
            setattr(s, field, value)
            with self.assertRaises(ValueError):
                diag.Frozen([s], runtime)
            self.assertEqual(allocations, [])
        s = state()
        s.ikc.append(None)
        with self.assertRaisesRegex(ValueError, "required initialized"):
            diag.state_storage(s, runtime)

    def test_unsupported_state_domain_refuses(self):
        for field, value in (("compact_indexer", object()), ("image_positions", object()), ("rope_delta", 1)):
            s = state()
            setattr(s, field, value)
            with self.assertRaises(ValueError):
                diag.state_metadata(s)

    def attention_probe_fixture(self, *, error=None, exit_error=None):
        allocations, events, stack = [], [], []
        runtime = NS(float16="half", float32="float", cuda=NS(mem_get_info=lambda: (20800, 1)),
                     empty=lambda shape, **kw: allocations.append((shape, kw)) or object())
        selected = NS(capacity=32, mtp_len=11, mtp_pos=object(),
                      mtp_kc=NS(k=NS(shape=(32, 1, 128), device="cuda"), v=object()))
        probe = diag.AttentionArithmeticProbe(runtime, selected)
        class Mode:
            def __enter__(self):
                stack.append(self)
            def __exit__(self, *exc):
                self_args = stack.pop()
                assert self_args is self
                if exit_error is not None:
                    raise exit_error
        def softmax(*args, **kwargs):
            events.append(("softmax", args, kwargs))
            if error is not None:
                raise error
            return object()
        def mm(*args):
            events.append(("mm", args))
            return object()
        final, output = object(), []
        def attention(*args, **kwargs):
            events.append(("attention", args, kwargs))
            if args[1] is not selected.mtp_kc.k:
                return final
            scores = ops._matmul32("query", "key")
            stack[-1].__torch_dispatch__(softmax, (), (scores,), {"dim": -1, "half_to_float": False})
            ops._matmul32("rounded-P", "value")
            output[:] = [final]
            return output
        ops = NS(attention=attention, _matmul32=mm)
        return probe, ops, Mode, softmax, selected, allocations, events, stack, output

    def test_attention_probe_exact_budget_and_refusal_before_allocation(self):
        probe, _, _, _, _, allocations, *_ = self.attention_probe_fixture()
        probe.allocate()
        self.assertEqual(len(allocations), 9)
        self.assertEqual(sum(shape[0] * (2 if kw["dtype"] == "half" else 4) for shape, kw in allocations), 20800)
        probe, _, _, _, _, allocations, *_ = self.attention_probe_fixture()
        probe.torch.cuda.mem_get_info = lambda: (20799, 1)
        with self.assertRaisesRegex(ValueError, "obligation"):
            probe.allocate()
        self.assertEqual(allocations, [])
        probe, *_ = self.attention_probe_fixture()
        primary, objects = MemoryError("third allocation"), []
        def partial(shape, **kwargs):
            if len(objects) == 2:
                raise primary
            value = object()
            objects.append(value)
            return value
        probe.torch.empty = partial
        with self.assertRaises(MemoryError) as caught:
            probe.allocate()
        self.assertIs(caught.exception, primary)
        self.assertEqual(list(probe.storage.values()), objects)

    def test_attention_probe_records_strided_borrow_without_clone_or_reallocation(self):
        probe, *_ = self.attention_probe_fixture()
        copies = []
        class Array:
            shape, dtype, device = (128, 12), "half", "cuda"
            def numel(self):
                return 1536
        class Storage:
            def __getitem__(self, slc):
                self.extent = slc.stop
                return self
            def view(self, shape):
                self.shape = shape
                return self
            def copy_(self, source):
                copies.append(source)
        source, target = Array(), Storage()
        probe.torch.is_tensor = lambda t: isinstance(t, Array)
        probe.storage = {"key": target}
        probe.begin(12)
        probe.record("key", source)  # Source deliberately has no clone/contiguous operation.
        self.assertEqual((target.extent, target.shape), (1536, (128, 12)))
        self.assertEqual(copies, [source])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            probe.record("key", source)
        probe.begin(12)
        source.shape = (128, 33)
        with self.assertRaisesRegex(ValueError, "geometry"):
            probe.record("key", source)
        self.assertEqual(copies, [source])

    def test_attention_probe_actual_alias_selected_identity_once_and_kwargs(self):
        probe, ops, mode, softmax, s, _, events, stack, output = self.attention_probe_fixture()
        original_attention, original_mm = ops.attention, ops._matmul32
        recorded, out = [], object()
        probe.begin(12)
        with patch.object(probe, "record", side_effect=lambda name, value: recorded.append((name, value))):
            with diag.attributes(probe.bindings(ops, mode_type=mode, softmax=softmax)):
                self.assertIsNotNone(ops.attention(None, object(), None, None, None, 1, 0))
                self.assertEqual(recorded, [])
                args = (object(), s.mtp_kc.k, s.mtp_kc.v, s.mtp_pos, object(), 1, .25)
                result = ops.attention(*args, context=12, out=out, _native_allowed=False)
                self.assertIs(result, output)
                self.assertEqual(events[1], ("attention", args, {"context": 12, "out": out, "_native_allowed": False}))
                self.assertEqual([name for name, _ in recorded], ["query", "key", "QK", "softmax_input", "softmax_output", "P", "value", "PV", "final"])
                self.assertEqual((probe.calls, probe.mm_calls, probe.softmax_calls), (1, 2, 1))
                with self.assertRaisesRegex(ValueError, "identity/call"):
                    ops.attention(*args, context=12)
        self.assertIs(ops.attention, original_attention)
        self.assertIs(ops._matmul32, original_mm)
        self.assertFalse(probe.active)
        self.assertEqual(stack, [])

    def test_attention_probe_delegate_primary_and_mode_cleanup_survive(self):
        primary, cleanup = RuntimeError("actual softmax"), OSError("mode exit")
        probe, ops, mode, softmax, s, _, events, stack, _ = self.attention_probe_fixture(error=primary, exit_error=cleanup)
        old_attention, old_mm = ops.attention, ops._matmul32
        probe.begin(12)
        with patch.object(probe, "record"):
            with self.assertRaises(RuntimeError) as caught:
                with diag.attributes(probe.bindings(ops, mode_type=mode, softmax=softmax)):
                    ops.attention(None, s.mtp_kc.k, s.mtp_kc.v, s.mtp_pos, None, 1, .25, context=12)
        self.assertIs(caught.exception, primary)
        self.assertIs(primary.__cause__, cleanup)
        self.assertEqual(sum(e[0] == "softmax" for e in events), 1)
        self.assertEqual(stack, [])
        self.assertFalse(probe.active)
        self.assertIs(ops.attention, old_attention)
        self.assertIs(ops._matmul32, old_mm)

    def test_attention_dispatch_unknown_ops_once_and_reentry_schema_refusal(self):
        probe, *_ = self.attention_probe_fixture()
        calls, result = [], object()
        def delegate(*args, **kwargs):
            calls.append((args, kwargs))
            return result
        self.assertIs(probe.dispatch(delegate, (1,), {"key": 2}, object()), result)
        self.assertEqual(calls, [((1,), {"key": 2})])
        probe.active, probe.mm_calls = True, 1
        with patch.object(probe, "record"):
            self.assertIs(probe.dispatch(delegate, (object(), -1, False), {}, delegate), result)
            with self.assertRaisesRegex(ValueError, "repeated"):
                probe.dispatch(delegate, (object(), -1, False), {}, delegate)
        self.assertEqual(len(calls), 2)
        probe.softmax_calls = 0
        with self.assertRaisesRegex(ValueError, "schema"):
            probe.dispatch(delegate, (object(), 0, False), {}, delegate)
        self.assertEqual(len(calls), 2)

    def test_selected_linear_observer_calls_real_alias_once_and_restores(self):
        modules = {}
        for name in ("tensorfold", "tensorfold.families", "tensorfold.families.qwen4_exp", "tensorfold.families.qwen4_exp.cuda"):
            modules[name] = ModuleType(name)
            modules[name].__path__ = []
        package = modules["tensorfold.families.qwen4_exp.cuda"]
        calls, captured, output, matrix = [], [], object(), object()
        def real(*args, **kwargs):
            calls.append((args, kwargs))
            return output
        package.mtp, package.forward, package.glue = NS(_mm=real), NS(_mm=real), NS()
        weights = NS(mtp=NS(fc_e=matrix, fc_h=object(), layer=NS(attn=NS(proj=object()))), head=object())
        probe = object.__new__(diag.StageProbe)
        probe.stage = "fc_e"
        probe.record = lambda name, value, **kwargs: captured.append((name, value))
        with patch.dict(sys.modules, modules):
            bindings = probe.bindings(weights)
            with diag.attributes(bindings):
                self.assertIs(package.mtp._mm("x", matrix, "xs", "out", "buffer", extra=7), output)
            self.assertIs(package.mtp._mm, real)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], {"extra": 7})
        self.assertEqual(captured, [("input", "x"), ("output", output)])

    def test_probe_cell_cannot_inherit_a_prior_witness(self):
        probe = object.__new__(diag.StageProbe)
        probe.metadata = {"input": object(), "output": object()}
        probe.begin()
        self.assertEqual(probe.metadata, {})
        with self.assertRaisesRegex(ValueError, "not invoked"):
            probe.flush(None)

    def test_probe_capacity_refuses_before_copy(self):
        probe = object.__new__(diag.StageProbe)
        probe.capacity = 2
        probe.storage = {"input": None}
        with self.assertRaisesRegex(ValueError, "prepared copy"):
            probe.record("input", Tensor(b"larger"))

    def test_selection_probe_checks_flags_before_promoting_ids(self):
        probe = object.__new__(diag.SelectionProbe)
        probe.positions = {slot: 8 for slot in range(6)}
        probe.counts = {slot: 1 for slot in range(6)}
        probe.ratio, probe.budget, probe.cursor = 4, 8, 6
        class Matrix:
            def __init__(self, values):
                self.values = values
            def __getitem__(self, key):
                slot, rows = key
                return NS(cpu=lambda: NS(tolist=lambda: self.values[slot][:rows.stop]))
        class UninitializedIDs:
            def __getitem__(self, key):
                raise AssertionError("dense IDs must not be read")
        probe.flags, probe.nk, probe.ids = Matrix([[0]] * 6), Matrix([[9]] * 6), UninitializedIDs()
        records = probe.flush()
        self.assertTrue(all(row["selected"] == list(range(9)) for row in records))
        probe.flags = Matrix([[1]] * 6)
        with self.assertRaisesRegex(ValueError, "count/flag"):
            probe.flush()

    def test_setup_failure_restores_policies_and_retains_uncertain_owners(self):
        primary, cleanup = RuntimeError("fixture construction"), OSError("device retirement")
        modules = {}
        names = ["tensorfold", "tensorfold.families", "tensorfold.families.qwen4_exp", "tensorfold.families.qwen4_exp.cuda",
                 "tensorfold.cuda", "tensorfold.cuda.sm70", "tensorfold.cuda.sm70.software_binding",
                 "tensorfold.families.qwen4_exp.cuda.multi", "tensorfold.families.qwen4_exp.cuda.multi_graphs",
                 "tensorfold.families.qwen4_exp.cuda.decode", "tensorfold.families.qwen4_exp.cuda.state",
                 "tensorfold.families.qwen4_exp.cuda.forward", "tensorfold.families.qwen4_exp.cuda.mtp",
                 "tensorfold.families.qwen4_exp.cuda.padded_linears", "tensorfold.cuda.sm70.family_ops",
                 "test_flashnext_sm70_mtp", "test_sm70_gguf_prepared", "test_flashnext_sm70_bound_graphs", "test_flashnext_sm70_family"]
        for name in names:
            modules[name] = ModuleType(name)
            modules[name].__path__, modules[name].__file__ = [], __file__
        for name, module in modules.items():
            if "." in name:
                parent, attr = name.rsplit(".", 1)
                setattr(modules[parent], attr, module)
        modules["tensorfold"].__file__ = "/test/site-packages/tensorfold/__init__.py"
        modules["tensorfold.cuda.sm70.software_binding"].issue_source_binding = lambda **kwargs: NS(record=lambda: {})
        family = modules["tensorfold.cuda.sm70.family_ops"]
        family._NATIVE_FAMILY, family._NATIVE_ATTENTION = False, True
        padded = modules["tensorfold.families.qwen4_exp.cuda.padded_linears"]
        padded._ENABLED = False
        def fail():
            raise primary
        modules["test_flashnext_sm70_mtp"]._weights_with_stock_mtp_protocol = fail
        modules["test_sm70_gguf_prepared"].dense_fixture = lambda w: w
        modules["test_flashnext_sm70_bound_graphs"]._qualify_six_lanes = lambda *a: None
        modules["test_flashnext_sm70_bound_graphs"]._qualify_mtp = lambda *a: None
        for module, attributes in (("multi", ("MultiDecoder", "_slot")), ("multi_graphs", ("RoundGraphs",)),
                                   ("decode", ("prefill",)), ("state", ("Buffers",)), ("forward", ("_kv_check",))):
            for attr in attributes:
                setattr(modules["tensorfold.families.qwen4_exp.cuda." + module], attr, lambda *a, **k: None)
        runtime = ModuleType("torch")
        runtime.__version__ = "test-runtime"
        runtime.version = NS(git_version="04cf2557f315a350e7eea7ab064288def65db75a", cuda="12.9")
        runtime.cuda = NS(is_available=lambda: True, get_device_capability=lambda: (7, 0),
                          synchronize=lambda: (_ for _ in ()).throw(cleanup))
        runtime.backends = NS(cuda=NS(matmul=NS(allow_fp16_reduced_precision_reduction=True, allow_fp16_accumulation=False)))
        runtime.is_inference_mode_enabled = lambda: False
        modules["torch"] = runtime
        roots = [None]
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, modules), patch.object(diag, "_ACTIVE", roots):
            args = NS(out=Path(directory), case="reference", image_id="sha256:" + "a" * 64, source_commit="b" * 40)
            with self.assertRaises(RuntimeError) as caught:
                diag.execute(args)
            self.assertIs(caught.exception, primary)
            self.assertIs(primary.__cause__, cleanup)
            self.assertIsNotNone(roots[0])
            self.assertEqual((family._NATIVE_FAMILY, family._NATIVE_ATTENTION, padded._ENABLED), (False, True, False))
            record = json.loads((Path(directory) / "diagnostic.json").read_text())
            self.assertFalse(record["completed"])
            self.assertFalse(record["retirement_verified"])
            self.assertTrue(record["failed"])
            self.assertFalse(record["qualification"])

    def test_unmodified_anchor_runs_before_any_added_capture_and_preserves_failure(self):
        calls, failure = [], AssertionError("actual strict failure")
        def close(actual, expected, **kwargs):
            calls.append("original-comparator")
            raise failure
        torch = NS(testing=NS(assert_close=close), cuda=NS(synchronize=lambda: self.fail("added pre-anchor fence")))
        def _qualify_mtp(weights, cache):
            iteration, counts = 0, [1] * 6
            actual, expected = "borrowed-do-not-read", "borrowed-do-not-read"
            try:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=f"iteration={iteration},counts={counts}")
            finally:
                calls.append("original-finally")
        with patch.object(diag, "Frozen", side_effect=AssertionError("pre-anchor snapshot")), \
             patch.object(diag.Output, "tensor", side_effect=AssertionError("pre-anchor copy")):
            proof, caught = diag.original_anchor(_qualify_mtp, object(), "int8")
        calls.append("added-capture-now-allowed")
        self.assertIs(caught, failure)
        self.assertEqual(calls, ["original-comparator", "original-finally", "added-capture-now-allowed"])
        self.assertTrue(proof["strict_failure_observed"])
        self.assertFalse(proof["tensor_storage_read"])
        self.assertEqual(proof["original_prestate_bytes"], "UNPROVEN")
        self.assertEqual((proof["iteration"], proof["counts"]), (0, [1] * 6))

    def test_anchor_rejects_unreproduced_or_relaxed_comparator(self):
        def _qualify_mtp(weights, cache):
            return None
        with self.assertRaisesRegex(diag.Unsupported, "did not reproduce"):
            diag.original_anchor(_qualify_mtp, None, "int8")
        failure = AssertionError("not original zero tolerance")
        torch = NS(testing=NS(assert_close=lambda *a, **k: (_ for _ in ()).throw(failure)))
        def _qualify_mtp(weights, cache):
            iteration, counts = 0, [1] * 6
            actual, expected = 1, 2
            torch.testing.assert_close(actual, expected, rtol=.01, atol=0, msg=f"iteration={iteration},counts={counts}")
        with self.assertRaises(diag.Unsupported) as caught:
            diag.original_anchor(_qualify_mtp, None, "int8")
        self.assertIs(caught.exception.__cause__, failure)

    def test_instrumented_authority_uses_original_comparator_not_internal_sha_or_signed_zero(self):
        class Scalar:
            def __init__(self, value):
                self.value = value
            def __getitem__(self, index):
                return self
        calls = []
        def close(a, b, **kwargs):
            calls.append(kwargs)
            self.assertEqual((kwargs["rtol"], kwargs["atol"]), (0, 0))
            a = a.value if isinstance(a, Scalar) else a
            b = b.value if isinstance(b, Scalar) else b
            self.assertEqual(a, b)  # +0/-0 compare equal, unlike raw SHA.
        cache = NS(**{n: Scalar(0.) for n in ("k", "v", "ks", "vs")})
        owner = NS(mtp_len=0, ratio=4, mtp_kc=cache, mtp_pooled=Scalar(0.))
        runtime = NS(testing=NS(assert_close=close))
        control, error = diag.strict_instrumented_pair(runtime, -0., 0., NS(streams=Scalar(-0.)),
                               NS(streams=Scalar(0.)), [owner], [owner], [1])
        self.assertFalse(control["strict_failure_observed"])
        self.assertIsNone(error)
        control, error = diag.strict_instrumented_pair(runtime, 1., 2., NS(streams=Scalar(0.)),
                               NS(streams=Scalar(0.)), [owner], [owner], [1])
        self.assertTrue(control["strict_failure_observed"])
        self.assertEqual(control["observable"], "logits")
        self.assertIsInstance(error, AssertionError)
        self.assertTrue(calls)

    def test_execute_has_no_added_snapshot_or_fence_before_original_anchor(self):
        import ast
        tree = ast.parse(Path(diag.__file__).read_text())
        execute = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "execute")
        anchor = next(n for n in ast.walk(execute) if isinstance(n, ast.Call) and ast.unparse(n.func) == "original_anchor")
        before = [ast.unparse(n.func) for n in ast.walk(execute) if isinstance(n, ast.Call) and n.lineno < anchor.lineno]
        forbidden = {"Frozen", "AttentionArithmeticProbe", "copy_plan", "state_tensors", "state_storage", "output.tensor", "torch.cuda.synchronize", "torch.cuda.get_rng_state"}
        self.assertFalse(forbidden.intersection(before))
        self.assertLess(next(n.lineno for n in ast.walk(execute) if isinstance(n, ast.Call) and ast.unparse(n.func) == "_weights_with_stock_mtp_protocol"), anchor.lineno)

    def complete_report(self):
        cells = {name: observation() for name in ("A", "A_repeat", "B", "C", "D")}
        for name in ("B", "C", "D"):
            cells[name]["attention_output"]["sha256"] = "b" * 64
        report = {"schema": "tensorfold.mtp-four-cell-diagnostic-v1", "case": "reference", "image_id_declared": "sha256:" + "a" * 64,
                "source_commit_declared": "b" * 40, "qualification": False, "P3_valid": False, "completed": True,
                "failed": False, "retirement_verified": True, "error_types": [], "instrumented_strict_failure_observed": True,
                "original_anchor": {"strict_failure_observed": True, "tensor_storage_read": False, "source_function": "_qualify_mtp",
                                    "source_sha256": "a" * 64, "line": 1, "expression": "torch.testing.assert_close(actual, expected, rtol=0, atol=0)"},
                "original_input_match": "UNPROVEN", "experiment_status": "CONDITIONAL_MECHANISM_DIAGNOSTIC_ONLY",
                "steps": [{"cells": cells, "discrete_inputs": True, "raw_pair_difference": "attention_output", "strict_control": {"strict_failure_observed": True, "observable": "logits"},
                           "difference": diag.classify(cells), "first_stage_capture": {"stage": "attention_output"}}],
                "native_origin": {"mapped_elfs": [{"name": "tf_sm70_family_actual"}], "missing_required_prefixes": []}}
        step = report["steps"][0]
        step["first_stage_capture"].update(A=[], B=[])
        step["arithmetic_probe"] = {"status": "CONDITIONAL_MECHANISM_DIAGNOSTIC_ONLY",
                                    "logical_row": 1, "slot": 4, "end": 12, "device_copy_obligation_bytes": 20800,
                                    "cells": {cell: {"keys": keys, "retained": copy.deepcopy(cells[cell]), "QSA": [],
                                                     "operators": {name: copy.deepcopy(cells[cell]["q"])
                                                                   for name in diag.AttentionArithmeticProbe.LAYOUT}}
                                              for cell, keys in (("A", 12), ("B", 32))}}
        return report

    def test_arithmetic_witness_rejects_observer_perturbation_and_missing_calls(self):
        report = self.complete_report()
        step = report["steps"][0]
        diag.validate_arithmetic_probe(step)
        for field, value in (("status", "UNSUPPORTED_PERTURBED_WITNESS"), ("slot", 3),
                             ("device_copy_obligation_bytes", 0)):
            altered = copy.deepcopy(step)
            altered["arithmetic_probe"][field] = value
            with self.assertRaises(ValueError):
                diag.validate_arithmetic_probe(altered)
        for mutate in (lambda x: x["retained"]["q"].update(sha256="c" * 64),
                       lambda x: x["QSA"].append({"changed": True}),
                       lambda x: x["operators"].pop("softmax_output")):
            altered = copy.deepcopy(step)
            mutate(altered["arithmetic_probe"]["cells"]["A"])
            with self.assertRaises(ValueError):
                diag.validate_arithmetic_probe(altered)

    def test_pytest_entry_accepts_only_complete_diagnostic_not_qualification(self):
        import runpy
        entry = runpy.run_path(str(Path(__file__).parent / "cuda/test_sm70_mtp_rounding_diagnostic.py"))
        parse = entry["operator_arguments"]
        self.assertIsNone(parse({}))
        environment = dict(zip(entry["ENV"].values(), ("reference", "/evidence/rounding-reference", "sha256:" + "a" * 64, "b" * 40)))
        self.assertEqual(parse(environment)[:2], ["--case", "reference"])
        for key, value in (("TENSORFOLD_MTP_ROUNDING_OUT", "/evidence-other/unsafe"),
                           ("TENSORFOLD_MTP_ROUNDING_IMAGE", "tag:latest"),
                           ("TENSORFOLD_MTP_ROUNDING_CASE", "unknown")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                parse({**environment, key: value})
        with self.assertRaises(ValueError):
            parse({"TENSORFOLD_MTP_ROUNDING_CASE": "reference"})
        report = self.complete_report()
        self.assertEqual(diag.validate_report(report, case="reference", image_id=environment["TENSORFOLD_MTP_ROUNDING_IMAGE"],
                                             source_commit="b" * 40), "CONDITIONAL_MECHANISM_DIAGNOSTIC_ONLY")

    def test_entry_rejects_unreproduced_missing_cells_retirement_and_origin_failure(self):
        for key, value in (("completed", False), ("instrumented_strict_failure_observed", False), ("retirement_verified", False),
                           ("qualification", True), ("P3_valid", True), ("original_input_match", "PROVEN"), ("native_origin", {"mapped_elfs": [], "missing_required_prefixes": []})):
            report = self.complete_report()
            report[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                diag.validate_report(report, case="reference", image_id="sha256:" + "a" * 64, source_commit="b" * 40)
        report = self.complete_report()
        del report["steps"][-1]["cells"]["C"]
        with self.assertRaises(ValueError):
            diag.validate_report(report, case="reference", image_id="sha256:" + "a" * 64, source_commit="b" * 40)
        report = self.complete_report()
        report["steps"][-1]["raw_pair_difference"] = None
        with self.assertRaises(ValueError):
            diag.validate_report(report, case="reference", image_id="sha256:" + "a" * 64, source_commit="b" * 40)

    def test_artifact_entry_checks_payload_stream_and_checksum_without_loading_arrays(self):
        import runpy
        from tools import gb10_io
        entry = runpy.run_path(str(Path(__file__).parent / "cuda/test_sm70_mtp_rounding_diagnostic.py"))
        def magic(stream):
            self.assertEqual(stream.read(2), b"MM")
            return (1, 0)
        def header(stream):
            self.assertEqual(stream.read(2), b"HH")
            return ((2, 2), False, NS(hasobject=False, str="|u1", itemsize=1))
        fake = ModuleType("numpy")
        fake.lib = NS(format=NS(read_magic=magic, read_array_header_1_0=header, read_array_header_2_0=header))
        payload = b"data"
        import hashlib
        record = {"shape": [2, 2], "dtype": "|u1", "file": "raw.npy", "sha256": hashlib.sha256(payload).hexdigest()}
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"numpy": fake, "gb10_io": gb10_io}):
            path = Path(directory) / "raw.npy"
            path.write_bytes(b"MMHH" + payload)
            self.assertEqual(entry["verify_tensor_artifacts"]({"record": record}, Path(directory)), 1)
            path.write_bytes(b"MMHH" + b"fail")
            with self.assertRaisesRegex(ValueError, "SHA"):
                entry["verify_tensor_artifacts"]({"record": record}, Path(directory))
            path.write_bytes(b"MMHH" + payload + b"extra")
            with self.assertRaisesRegex(ValueError, "extent"):
                entry["verify_tensor_artifacts"]({"record": record}, Path(directory))

    def test_import_help_and_spawn_name_have_no_torch_or_filesystem_effects(self):
        script = """import runpy,sys,subprocess,pathlib
subprocess.Popen=lambda *a,**k: (_ for _ in ()).throw(AssertionError('spawn'))
pathlib.Path.mkdir=lambda *a,**k: (_ for _ in ()).throw(AssertionError('mkdir'))
runpy.run_module('tools.v100_mtp_rounding_diagnostic',run_name='__mp_main__')
assert 'torch' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([sys.executable, "-B", "-m", "tools.v100_mtp_rounding_diagnostic", "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
