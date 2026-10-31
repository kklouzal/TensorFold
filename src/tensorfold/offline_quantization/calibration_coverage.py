"""Observe the pinned two-pass Local-Hessian contract without calibrating a model.

The coordinator supplies a FRESH trusted plan, admitted input/batch bounds and
batch identities computed from its frozen inputs. Declared batch SHA values are
lineage, not authentication of tensor contents. Only statistics hashes below
are computed by this gate. Hooks attach to actual converted modules inside each
forward-loop pass and retire before export. No activations are retained.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import re
import threading
import weakref

import torch

from tensorfold.cleanup import raise_failures, rollback
from .producer_scope import selected_projections


def _quantizer_type():
    # Already-loaded producer dependency, imported only when observation begins.
    # No conversion/calibration algorithm is called here.
    from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer

    return TensorQuantizer


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _statistics(value, *, shape=None, count=None, positive=False, nonnegative=False, fp32=True):
    cpu = None
    try:
        if (
            not isinstance(value, torch.Tensor)
            or not value.is_floating_point()
            or value.device.type == "meta"
            or (fp32 and value.dtype != torch.float32)
            or (shape is not None and tuple(value.shape) != tuple(shape))
            or (count is not None and value.numel() != count)
            or value.numel() == 0
        ):
            raise ValueError("calibration statistic has missing/wrong geometry or precision")
        cpu = value.detach().to("cpu").contiguous()
        if (
            not bool(torch.isfinite(cpu).all())
            or (positive and not bool((cpu > 0).all()))
            or (nonnegative and not bool((cpu >= 0).all()))
        ):
            raise ValueError("calibration statistic is nonfinite or violates its declared range")
        return {
            "shape": list(cpu.shape),
            "dtype": str(cpu.dtype),
            "bytes": cpu.numel() * cpu.element_size(),
            "sha256": hashlib.sha256(memoryview(cpu.reshape(-1).view(torch.uint8).numpy())).hexdigest(),
            "minimum": float(cpu.float().min()),
            "maximum": float(cpu.float().max()),
        }
    finally:
        cpu = value = None


class CalibrationCoverage:
    """Single-thread/exclusive-producer observer; does not own the model.

    Usage: inside the callback given to mtq.quantize, enter observe_pass(model),
    then enter batch(actual_input_sha256) around every forward. After quantize
    returns call verify(model), before export mutates its weights/scales.
    max_input_elements and max_batches MUST come from executor admission.
    """

    def __init__(self, plan, *, max_input_elements, max_batches):
        self.names = selected_projections(plan)
        self.shapes = {
            item["name"].removesuffix(".weight"): tuple(item["shape"])
            for item in plan["tensors"]
            if item.get("action") == "quantize"
        }
        if any(type(n) is not int or n < 1 for n in (max_input_elements, max_batches)):
            raise ValueError("positive admitted calibration input and batch bounds required")
        self.max_input_elements, self.max_batches = max_input_elements, max_batches
        self.role = plan.get("role", "unspecified")
        self.owner = threading.get_ident()
        self._model = None
        self._handles = []
        self._passes = []
        self._current = None
        self._batch = None
        self._flags = {}
        self._poisoned = False
        self._verified = False

    def _check(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError("calibration coverage belongs to its exclusive producer thread")
        if self._poisoned or self._verified:
            raise RuntimeError("calibration coverage is terminal or already verified")

    def _scope(self, model, *, phase):
        module = q = None
        try:
            quantizer_type = _quantizer_type()
            expected = {name + ".input_quantizer" for name in self.names}
            if phase != "hessian":
                expected |= {name + ".weight_quantizer" for name in self.names}
            enabled = {
                name
                for name, module in model.named_modules()
                if isinstance(module, quantizer_type) and module.is_enabled
            }
            if enabled != expected:
                raise ValueError("enabled quantizers differ from the exact calibration phase scope")
            for name in self.names:
                module = model.get_submodule(name)
                if (
                    not isinstance(module, torch.nn.Linear)
                    or tuple(module.weight.shape) != self.shapes[name]
                    or module.weight.dtype != torch.bfloat16
                    or module.bias is not None
                ):
                    raise ValueError(f"{name}: converted projection differs from the planned native shape")
                for which, block_type in (("weight", "static"), ("input", "dynamic")):
                    q = getattr(module, which + "_quantizer")
                    block = q.block_sizes
                    if (
                        q.num_bits != (2, 1)
                        or not isinstance(block, dict)
                        or block.get(-1) != 16
                        or block.get("scale_bits") != (4, 3)
                        or block.get("type", "static") != block_type
                        or q._dynamic
                        or (which == "input" and q.axis is not None)
                        or (which == "weight" and q.axis not in (None, (0,)))
                        or q._use_constant_amax
                        or q._constant_amax is not None
                    ):
                        raise ValueError(f"{name}: actual quantizer global/block calibration mode differs")
        finally:
            module = q = model = None

    def _observe(self, name, _module, arguments):
        value = None
        try:
            self._check()
            if self._current is None or self._batch is None:
                raise RuntimeError("projection input observed outside an identified calibration batch")
            value = arguments[0] if arguments else None
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim < 1
                or value.shape[-1] != self.shapes[name][1]
                or not value.is_floating_point()
                or value.numel() == 0
                or value.numel() > self.max_input_elements
            ):
                raise ValueError("calibration input is empty/invalid or exceeds admitted element bound")
            with torch.no_grad():
                finite = torch.isfinite(value.detach()).all()
                self._flags[name] = finite if name not in self._flags else self._flags[name] & finite
            record = self._current["modules"][name]
            record["calls"] += 1
            record["rows"] += value.numel() // self.shapes[name][1]
            record["max_input_elements"] = max(record["max_input_elements"], value.numel())
        except BaseException:
            self._poisoned = True
            raise
        finally:
            value = arguments = _module = None

    def _input_hook(self, name):
        def observe(module, arguments):
            try:
                self._observe(name, module, arguments)
            finally:
                # Do not add an owned activation/module alias to a retained
                # failure traceback above the observer's own cleared frame.
                module = arguments = None

        return observe

    def _retire(self):
        errors = []
        remaining = []
        for handle in self._handles:
            if handle is None:
                continue
            try:
                handle.remove()
            except BaseException as error:
                errors.append(error)
                remaining.append(handle)
        self._handles = remaining
        if errors:
            self._poisoned = True

            def failed_retirement():
                raise_failures(None, errors)

            rollback(self, errors[0], failed_retirement)

    @contextmanager
    def observe_pass(self, model):
        self._check()
        if self._current is not None or len(self._passes) >= 2:
            raise RuntimeError("exact Local-Hessian observer permits one max and one Hessian pass")
        phase = "max" if not self._passes else "hessian"
        if self._model is not None and self._model() is not model:
            raise ValueError("calibration model identity changed between passes")
        self._model = weakref.ref(model)
        self._current = {
            "phase": phase,
            "batches": [],
            "modules": {name: {"calls": 0, "rows": 0, "max_input_elements": 0} for name in self.names},
        }
        primary = None
        try:
            self._scope(model, phase=phase)
            # Prepare exact builtin slots before acquiring any hook. Indexed
            # publication cannot resize/allocate after registration succeeds.
            self._handles = [None] * len(self.names)
            for index, name in enumerate(self.names):
                self._handles[index] = model.get_submodule(name).register_forward_pre_hook(self._input_hook(name))
            yield self
            if self._poisoned:
                raise RuntimeError("a calibration batch failure was suppressed; pass is terminal")
            if self._batch is not None or not self._current["batches"]:
                raise ValueError("calibration pass has no completed input batches")
            for name in self.names:
                if name not in self._flags or not bool(self._flags[name]):
                    raise ValueError("calibration pass has missing/nonfinite projection observations")
            if self._passes:
                before = self._passes[0]
                if before["batches"] != self._current["batches"] or before["modules"] != self._current["modules"]:
                    raise ValueError("calibration passes changed declared batch identities/counts or projection rows")
            self._passes.append(self._current)
        except BaseException as error:
            primary = error
            self._poisoned = True
        finally:
            self._current = self._batch = None
            self._flags.clear()
            model = None
            if primary is not None:
                # This outer transport also preserves the original failure if
                # retirement bookkeeping itself exhausts before hook removal.
                rollback(self, primary, self._retire)
            try:
                self._retire()
            except BaseException:
                self._poisoned = True
                raise

    @contextmanager
    def batch(self, input_sha256):
        self._check()
        try:
            if self._current is None or self._batch is not None or len(self._current["batches"]) >= self.max_batches:
                raise RuntimeError("calibration batch nesting/state/bound is invalid")
            if not isinstance(input_sha256, str) or re.fullmatch("[a-f0-9]{64}", input_sha256) is None:
                raise ValueError("declared frozen-input SHA256 required")
            before = {name: item["calls"] for name, item in self._current["modules"].items()}
            self._batch = input_sha256
            yield
            if any(self._current["modules"][name]["calls"] <= before[name] for name in self.names):
                raise ValueError("calibration batch did not exercise every selected projection")
            self._current["batches"].append(input_sha256)
        except BaseException:
            self._poisoned = True
            raise
        finally:
            self._batch = None

    def verify(self, model):
        """Read pinned retained statistics; no Hessian/quantization recomputation."""
        module = weightq = inputq = acc = accumulators = None
        try:
            self._check()
            if self._current is not None or len(self._passes) != 2 or self._model() is not model:
                raise ValueError("two completed same-model Local-Hessian passes required before verification")
            self._scope(model, phase="complete")
            accumulators = getattr(model, "_local_hessian_accumulators", None)
            if not isinstance(accumulators, dict):
                raise ValueError("retained Local-Hessian debug accumulators are missing")
            expected = {id(model.get_submodule(name).weight_quantizer) for name in self.names}
            if set(accumulators) != expected:
                raise ValueError("Local-Hessian accumulator coverage differs from selected weight quantizers")
            records = {}
            for name in self.names:
                module = model.get_submodule(name)
                weightq, inputq = module.weight_quantizer, module.input_quantizer
                n, k = self.shapes[name]
                # Static last-axis blocks are flattened internally by the
                # pinned producer; axis(0,) is derived layout, not a changed
                # user per-channel/global quantizer policy.
                if (
                    weightq.axis != (0,)
                    or tuple(weightq._original_shape) != (n, k)
                    or tuple(weightq._block_reshape_size) != (-1, 16)
                    or tuple(weightq._amax_shape_for_export) != (n, -1)
                ):
                    raise ValueError("calibrated static weight block layout differs from producer geometry")
                acc = accumulators[id(weightq)]
                rows = self._passes[1]["modules"][name]["rows"]
                if (
                    acc.cout != n
                    or acc.cin != k
                    or acc.block_size != 16
                    or acc.num_blocks_per_cin != k // 16
                    or acc.is_enabled is not True
                    or acc.input_dtype != torch.float32
                    or type(acc.num_samples) is not int
                    or acc.num_samples <= 0
                    or acc.num_samples != rows
                ):
                    raise ValueError("Local-Hessian geometry/sample count/FP32 input contract differs")
                records[name] = {
                    "samples": acc.num_samples,
                    "hessian": _statistics(acc.hessian_per_block, shape=(k // 16, 16, 16)),
                    "normalized_hessian": _statistics(acc._normalized_hessian, shape=(k // 16, 16, 16)),
                    "weight_block_amax": _statistics(weightq.amax, count=n * k // 16, nonnegative=True),
                    "weight_global_amax": _statistics(weightq.global_amax, count=1, nonnegative=True),
                    # MaxCalibrator preserves activation dtype. BF16->FP32
                    # export is exact; do not invent an FP32 observer requirement.
                    "activation_global_amax": _statistics(inputq.amax, count=1, positive=True, fp32=False),
                }
            receipt = {
                "schema": "tensorfold.local-hessian-coverage",
                "schema_version": 1,
                "role": self.role,
                "status": "calibration-coverage-verified-not-quality-or-export",
                "selected_projections": len(self.names),
                "passes": self._passes,
                "statistics": records,
                "batch_identity_scope": "caller-declared lineage; executor must bind frozen actual inputs",
                "retained_activations": False,
                "max_input_elements": self.max_input_elements,
                "max_batches": self.max_batches,
            }
            receipt["sha256"] = _digest(receipt)
            self._verified = True
            return receipt
        except BaseException:
            self._poisoned = True
            raise
        finally:
            module = weightq = inputq = acc = accumulators = model = None
