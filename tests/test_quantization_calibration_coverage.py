"""Pinned-observer protocol controls with synthetic CPU producer state, no ModelOpt."""

import copy
import gc
import weakref
import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from tensorfold.offline_quantization import calibration_coverage as coverage  # noqa: E402
from tensorfold.offline_quantization.plan import NVFP4_RECIPE  # noqa: E402


class Quantizer(torch.nn.Module):
    def __init__(self, kind, *, enabled=True):
        super().__init__()
        self.is_enabled = enabled
        self.num_bits = (2, 1)
        self.block_sizes = {-1: 16, "scale_bits": (4, 3), "type": "static" if kind == "weight" else "dynamic"}
        self._dynamic = False
        self.axis = (0,) if kind == "weight" else None
        if kind == "weight":
            self._original_shape = (64, 64)
            self._block_reshape_size = (-1, 16)
            self._amax_shape_for_export = (64, -1)
        self._use_constant_amax = False
        self._constant_amax = None
        self.register_buffer("amax", torch.zeros(64, 4) if kind == "weight" else torch.tensor(1, dtype=torch.bfloat16))
        if kind == "weight":
            self.register_buffer("global_amax", torch.tensor(0.0))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)
        self.b = torch.nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)
        for module in (self.a, self.b):
            module.input_quantizer = Quantizer("input")
            module.weight_quantizer = Quantizer("weight")
            module.output_quantizer = Quantizer("input", enabled=False)
        self.media_quantizer = Quantizer("input", enabled=False)

    def forward(self, value):
        return self.b(self.a(value))


def plan():
    return {
        "schema": "tensorfold.offline-quantization-plan",
        "schema_version": 1,
        "role": "synthetic-unit-fixture",
        "recipe": {
            "name": NVFP4_RECIPE,
            "numerics": "W4A4",
            "block_size": 16,
            "weight_scale_algorithm": "local_hessian",
            "fp8_scale_sweep": True,
            "batch_amax_fallback": False,
        },
        "coverage": {"selected_projections": 2},
        "tensors": [
            {"name": name + ".weight", "shape": [64, 64], "source_dtype": "BF16", "action": "quantize"}
            for name in ("a", "b")
        ],
    }


@pytest.fixture
def prepared(monkeypatch):
    monkeypatch.setattr(coverage, "_quantizer_type", lambda: Quantizer)
    return Model(), coverage.CalibrationCoverage(plan(), max_input_elements=512, max_batches=2)


def passes(model, gate, *, second_hash="a" * 64, second_rows=2):
    for index in range(2):
        for module in (model.a, model.b):
            module.weight_quantizer.is_enabled = index == 0
        with gate.observe_pass(model):
            with gate.batch("a" * 64 if index == 0 else second_hash):
                model(torch.ones(2 if index == 0 else second_rows, 64, dtype=torch.bfloat16))
    for module in (model.a, model.b):
        module.weight_quantizer.is_enabled = True


def stats(model, rows=2):
    value = torch.eye(16).expand(4, 16, 16).clone()
    value[:, 0, 1] = value[:, 1, 0] = -1  # legitimate negative off-diagonal covariance
    model._local_hessian_accumulators = {
        id(module.weight_quantizer): SimpleNamespace(
            cout=64,
            cin=64,
            block_size=16,
            num_blocks_per_cin=4,
            is_enabled=True,
            input_dtype=torch.float32,
            num_samples=rows,
            hessian_per_block=value.clone(),
            _normalized_hessian=value.clone() / rows,
        )
        for module in (model.a, model.b)
    }


def hooks_removed(model):
    assert not model.a._forward_pre_hooks and not model.b._forward_pre_hooks


def test_two_phase_scope_zero_weight_blocks_bf16_activation_and_bounded_json_receipt(prepared):
    model, gate = prepared
    passes(model, gate)
    stats(model)
    receipt = gate.verify(model)
    hooks_removed(model)
    assert receipt["selected_projections"] == 2 and receipt["retained_activations"] is False
    assert [p["phase"] for p in receipt["passes"]] == ["max", "hessian"]
    assert receipt["statistics"]["a"]["weight_block_amax"]["minimum"] == 0
    assert receipt["statistics"]["a"]["activation_global_amax"]["dtype"] == "torch.bfloat16"
    assert receipt["statistics"]["a"]["hessian"]["minimum"] == -1
    assert receipt["sha256"] == coverage._digest({k: v for k, v in receipt.items() if k != "sha256"})
    assert json.loads(json.dumps(receipt)) == receipt
    assert gate._current is None and not gate._flags and not gate._handles


@pytest.mark.parametrize(
    "mutation", ["extra-enabled", "output-enabled", "dynamic-global", "constant-global", "block32"]
)
def test_actual_quantizer_inventory_and_modes_checked_before_forward(prepared, mutation):
    model, gate = prepared
    if mutation == "extra-enabled":
        model.media_quantizer.is_enabled = True
    elif mutation == "output-enabled":
        model.a.output_quantizer.is_enabled = True
    elif mutation == "dynamic-global":
        model.a.input_quantizer._dynamic = True
    elif mutation == "constant-global":
        model.a.input_quantizer._constant_amax = 1
    else:
        model.a.weight_quantizer.block_sizes[-1] = 32
    with pytest.raises(ValueError):
        with gate.observe_pass(model):
            pytest.fail("bad producer mode reached forward")
    hooks_removed(model)


@pytest.mark.parametrize("mutation", ["empty", "oversize", "nonfinite", "unobserved", "missing-batches"])
def test_per_pass_observation_refuses_missing_invalid_or_unbounded_inputs(prepared, mutation):
    model, gate = prepared
    with pytest.raises((ValueError, RuntimeError)):
        with gate.observe_pass(model):
            if mutation != "missing-batches":
                with gate.batch("a" * 64):
                    value = torch.ones(2, 64, dtype=torch.bfloat16)
                    if mutation == "empty":
                        value = value[:0]
                    if mutation == "oversize":
                        value = torch.ones(9, 64, dtype=torch.bfloat16)
                    if mutation == "nonfinite":
                        value[0, 0] = float("nan")
                    if mutation == "unobserved":
                        model.a(value)
                    else:
                        model(value)
    hooks_removed(model)
    assert not gate._flags and gate._current is None


@pytest.mark.parametrize("change", ["hash", "rows", "one-shot"])
def test_second_pass_must_reiterate_same_declared_inputs_and_counts(prepared, change):
    model, gate = prepared
    if change == "one-shot":
        with gate.observe_pass(model):
            with gate.batch("a" * 64):
                model(torch.ones(2, 64, dtype=torch.bfloat16))
        for module in (model.a, model.b):
            module.weight_quantizer.is_enabled = False
        with pytest.raises(ValueError, match="no completed"):
            with gate.observe_pass(model):
                pass
    else:
        with pytest.raises(ValueError, match="identities/counts"):
            passes(
                model,
                gate,
                second_hash="b" * 64 if change == "hash" else "a" * 64,
                second_rows=3 if change == "rows" else 2,
            )
    hooks_removed(model)


@pytest.mark.parametrize(
    "change", ["missing", "coverage", "samples", "geometry", "dtype", "nonfinite", "input-zero", "negative-weight"]
)
def test_post_calibration_missing_or_fallback_statistics_refused(prepared, change):
    model, gate = prepared
    passes(model, gate)
    stats(model)
    acc = model._local_hessian_accumulators[id(model.a.weight_quantizer)]
    if change == "missing":
        del model._local_hessian_accumulators
    elif change == "coverage":
        model._local_hessian_accumulators.pop(id(model.b.weight_quantizer))
    elif change == "samples":
        acc.num_samples = 0
    elif change == "geometry":
        acc.cin = 128
    elif change == "dtype":
        acc.hessian_per_block = acc.hessian_per_block.double()
    elif change == "nonfinite":
        acc._normalized_hessian[0, 0, 0] = float("inf")
    elif change == "input-zero":
        model.a.input_quantizer.amax.zero_()
    else:
        model.a.weight_quantizer.amax[0, 0] = -1
    with pytest.raises(ValueError):
        gate.verify(model)
    hooks_removed(model)


def test_hooks_attach_to_current_postconversion_children(prepared):
    model, gate = prepared
    old = model.a
    model.a = copy.deepcopy(model.a)  # simulates producer replacement after gate construction
    passes(model, gate)
    stats(model)
    gate.verify(model)
    assert not old._forward_pre_hooks
    hooks_removed(model)


def test_failed_hook_acquisition_retires_earlier_handles(prepared, monkeypatch):
    model, gate = prepared
    primary = RuntimeError("hook acquisition failed")
    monkeypatch.setattr(model.b, "register_forward_pre_hook", lambda *_: (_ for _ in ()).throw(primary))
    with pytest.raises(RuntimeError) as caught:
        with gate.observe_pass(model):
            pass
    assert caught.value is primary
    hooks_removed(model)


def test_cleanup_failure_is_terminal_and_preserves_primary_and_owner(prepared, monkeypatch):
    model, gate = prepared
    primary, cleanup = RuntimeError("calibration failed"), RuntimeError("hook retirement failed")
    original = torch.utils.hooks.RemovableHandle.remove
    chosen = []

    def remove(handle):
        if not chosen:
            chosen.append(handle.id)
        if handle.id == chosen[0]:
            raise cleanup
        return original(handle)

    with monkeypatch.context() as patch:
        patch.setattr(torch.utils.hooks.RemovableHandle, "remove", remove)
        with pytest.raises(RuntimeError) as caught:
            with gate.observe_pass(model):
                with gate.batch("a" * 64):
                    raise primary
        assert caught.value is primary and caught.value.__cause__ is cleanup
        assert gate in primary._tensorfold_retained_owners
        with pytest.raises(RuntimeError, match="terminal"):
            with gate.observe_pass(model):
                pass
    # Controlled test only: retire the injected failure after restoring API.
    for handle in gate._handles:
        original(handle)
    hooks_removed(model)


def test_hook_slot_allocation_failure_acquires_no_hooks(prepared):
    model, gate = prepared

    class FailLength(tuple):
        def __len__(self):
            raise MemoryError("hook slot allocation failed")

    gate.names = FailLength(gate.names)
    with pytest.raises(MemoryError):
        with gate.observe_pass(model):
            pytest.fail("failed hook-slot preparation reached forward")
    hooks_removed(model)


def test_swallowed_batch_failure_cannot_publish_a_calibration_pass(prepared):
    model, gate = prepared
    with pytest.raises(RuntimeError, match="suppressed"):
        with gate.observe_pass(model):
            try:
                with gate.batch("a" * 64):
                    model(torch.ones(2, 64, dtype=torch.bfloat16))
                    raise ValueError("forward pipeline failed after observations")
            except ValueError:
                pass
    assert not gate._passes and gate._poisoned
    hooks_removed(model)


def test_hook_failure_suppressed_inside_batch_still_makes_pass_terminal(prepared):
    model, gate = prepared
    with pytest.raises(RuntimeError, match="suppressed"):
        with gate.observe_pass(model):
            with gate.batch("a" * 64):
                model(torch.ones(2, 64, dtype=torch.bfloat16))
                try:
                    model(torch.ones(0, 64, dtype=torch.bfloat16))
                except ValueError:
                    pass
    assert gate._poisoned and not gate._passes
    hooks_removed(model)


def test_owned_callback_traceback_does_not_retain_input_alias(prepared):
    model, gate = prepared
    primary = None
    with pytest.raises(ValueError) as caught:
        with gate.observe_pass(model):
            with gate.batch("a" * 64):
                callback = next(iter(model.a._forward_pre_hooks.values()))
                invalid = torch.ones(9, 64, dtype=torch.bfloat16)
                held = weakref.ref(invalid)
                try:
                    # Direct invocation excludes PyTorch's independent call
                    # frames: this checks ONLY our two owned hook frames.
                    callback(model.a, (invalid,))
                except ValueError as error:
                    primary = error
                    invalid = None
                    gc.collect()
                    assert held() is None
                    raise
    assert caught.value is primary
    hooks_removed(model)
