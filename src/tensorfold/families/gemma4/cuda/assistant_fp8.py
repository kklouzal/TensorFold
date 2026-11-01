"""Private W8A16 copies of only the official assistant's22 core projections.

This candidate never quantizes the verifier, KV, norms, attention, centroids or
ordered vocabulary head. Real original Linear/Parameter owners remain for native
references and `.weight.device`. Preparation does not mutate the native model;
installation/retirement require its caller's exclusive drained assistant lane.
No runtime wiring selects this module automatically.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import threading
import weakref

import torch
from torch import nn

from tensorfold.cleanup import raise_failures, rollback
from ..assistant_metadata import AssistantConfig


@dataclass(frozen=True)
class FP8Memory:
    original_core_bytes: int
    extra_resident_bytes: int
    extra_loading_peak_bytes: int
    projection_count: int


def projection_shapes(config: AssistantConfig) -> dict[str, tuple[int, int]]:
    """Exact22 ordinary core matrices; the entire masked vocabulary path stays BF16."""
    return {key.removesuffix('.weight'): shape for key, shape in config.shapes().items()
            if len(shape) == 2 and not key.startswith('masked_embedding.')
            and key not in ('model.embed_tokens.weight', 'lm_head.weight')}


def memory_bytes(config: AssistantConfig) -> FP8Memory:
    """Additional bytes while retaining all original BF16 native-reference owners.

    Conservative preparation peak includes every final private copy plus the
    largest existing quantizer/fragment staging reservation. Target/model and
    inference activations are charged separately by the owning engine.
    """
    from tensorfold.cuda.requant import resident_bytes, staging_bytes
    shapes = projection_shapes(config)
    if len(shapes) != 22 or any(k % 64 for _, k in shapes.values()):
        raise ValueError('assistant FP8 core requires its22 supported K/64 projections')
    resident = sum(resident_bytes('fp8', n, k) for n, k in shapes.values())
    staging = max(staging_bytes('fp8', n, k) for n, k in shapes.values())
    original = sum(2 * n * k for n, k in shapes.values())
    return FP8Memory(original, resident, resident + staging, len(shapes))


class _Linear(nn.Module):
    """Native module interface with an actual BF16 weight, not a device-only fake."""
    def __init__(self, native: nn.Linear, operation, owner):
        try:
            super().__init__()
            self._operation = None
            self.native = native
            self._owner = owner
            self._parameter = weakref.ref(native.weight)
            self._version = None if torch.is_inference(native.weight) else native.weight._version
            self.train(native.training)
            # Publish the new packed owner only after every fallible metadata
            # step. A failed wrapper's traceback must not retain that owner.
            self._operation = operation
        finally:
            operation = None

    @property
    def weight(self):
        return self.native.weight

    @property
    def bias(self):
        return self.native.bias

    @property
    def in_features(self):
        return self.native.in_features

    @property
    def out_features(self):
        return self.native.out_features

    def forward(self, value):
        owner = self._owner
        if owner._state != 'installed':
            raise RuntimeError('private assistant projection is not installed or has retired/failed')
        if owner._reference_depth:
            if owner._reference_thread != threading.get_ident():
                raise RuntimeError('assistant native reference belongs to another exclusive lane')
            return self.native(value)
        if (self.weight is not self._parameter() or self.weight.dtype != torch.bfloat16
                or (self._version is not None and self.weight._version != self._version)):
            raise RuntimeError('assistant BF16 source changed after private FP8 preparation')
        if torch.is_grad_enabled():
            raise RuntimeError('private assistant projections are inference-only')
        if value.ndim < 1 or value.shape[-1] != self.in_features:
            raise ValueError('assistant private projection input width differs')
        if value.dtype != torch.bfloat16 or value.device != self.weight.device:
            raise ValueError('assistant W8A16 requires original BF16 activations on its source device')
        shape = (*value.shape[:-1], self.out_features)
        if value.numel() == 0:
            return value.new_empty(shape)
        rows = value.reshape(-1, self.in_features).contiguous()
        # Native provider owns its W8A16 math, row invariance and CUDA boundary.
        return self._operation(rows).reshape(shape)


class AssistantFP8Core:
    """Two-phase candidate preparation/installation, plus an independent native mode.

    Caller reserves `memory_bytes(config)` BEFORE construction and owns exclusive
    access to the native assistant. All BF16 source parameters remain untouched.
    `drain` must retire preparation and every forward consumer before close or
    installation rollback can release private tensor owners.

    This is an inference-only module installation: wrappers register their real
    native child, so state_dict keys include `.native.weight` while installed.
    Native numerical reference mode does not change that hierarchy. Drain/close
    and restore the original modules before checkpoint loading or export; this
    candidate does not provide an installed FP8 save_pretrained format.
    """
    def __init__(self, model: nn.Module, *, drain):
        if not callable(drain):
            raise ValueError('private assistant needs an explicit native-reader drain callback')
        from tensorfold.cuda.requant import requantize
        self._state = 'preparing'
        self._reference_depth, self._reference_thread = 0, None
        self._model, self._drain = model, drain
        self._bindings, self._wrappers = [], {}
        operation = None
        try:
            self.config = AssistantConfig.from_dict(model.config.to_dict())
            self.accounting = memory_bytes(self.config)
            device = None
            for path, shape in projection_shapes(self.config).items():
                native = model.get_submodule(path)
                if (type(native) is not nn.Linear or native.bias is not None
                        or tuple(native.weight.shape) != shape or native.weight.dtype != torch.bfloat16
                        or not native.weight.is_contiguous()):
                    raise ValueError('assistant FP8 source must be original contiguous BF16 Linear: ' + path)
                if device is None:
                    device = native.weight.device
                elif native.weight.device != device:
                    raise ValueError('assistant private projections must share one device')
                parent_path, _, name = path.rpartition('.')
                parent = model.get_submodule(parent_path) if parent_path else model
                self._bindings.append((path, parent, name, native))
            # Validate the entire selection before allocating a private copy.
            with torch.no_grad():
                for path, _parent, _name, native in self._bindings:
                    operation = requantize(native.weight, 'fp8')
                    self._wrappers[path] = _Linear(native, operation, self)
                    operation = None
            actual = sum(wrapper._operation.nbytes() for wrapper in self._wrappers.values())
            if actual > self.accounting.extra_resident_bytes:
                raise RuntimeError('assistant FP8 provider storage exceeds its admitted estimate')
            self._state = 'prepared'
        except BaseException as primary:
            rollback(self, primary, self.close)
        finally:
            operation = None

    def install(self):
        """Publish all22 prepared modules without changing unrelated module owners."""
        if self._state != 'prepared':
            raise RuntimeError('assistant private projections install exactly once from prepared state')
        for _path, parent, name, native in self._bindings:
            if getattr(parent, name) is not native:
                raise RuntimeError('assistant module changed before FP8 installation')
        attempted = []
        self._state = 'installing'
        try:
            for path, parent, name, _native in self._bindings:
                attempted.append((path, parent, name, _native))
                setattr(parent, name, self._wrappers[path])
                if getattr(parent, name) is not self._wrappers[path]:
                    raise RuntimeError('assistant module hook changed the installed projection')
        except BaseException as primary:
            errors = []
            for _path, parent, name, native in reversed(attempted):
                try:
                    setattr(parent, name, native)
                    if getattr(parent, name) is not native:
                        raise RuntimeError('assistant module restoration did not publish its native owner')
                except BaseException as error:
                    errors.append(error)
            self._state = 'failed' if errors else 'prepared'
            raise_failures(primary, errors)
        self._state = 'installed'
        return self._model

    @contextmanager
    def native_reference(self):
        """Compose with the head's other native-reference scopes on its drained lane."""
        if self._state != 'installed':
            raise RuntimeError('assistant native reference requires installed healthy projections')
        thread = threading.get_ident()
        if self._reference_depth and self._reference_thread != thread:
            raise RuntimeError('assistant reference scope is already owned by another lane')
        self._reference_thread = thread
        self._reference_depth += 1
        try:
            yield
        finally:
            self._reference_depth -= 1
            if not self._reference_depth:
                self._reference_thread = None

    def close(self):
        """Drain first; restore native owners before freeing any private buffers."""
        if self._state == 'closed':
            return
        if self._reference_depth:
            raise RuntimeError('cannot retire assistant projections from an active native reference')
        try:
            self._drain()
        except BaseException:
            self._state = 'failed'
            raise
        errors = []
        for path, parent, name, native in reversed(self._bindings):
            try:
                current = getattr(parent, name)
                if current is self._wrappers.get(path):
                    setattr(parent, name, native)
                    if getattr(parent, name) is not native:
                        raise RuntimeError('assistant native projection restoration failed')
                elif current is not native:
                    raise RuntimeError('another owner replaced the assistant projection')
            except BaseException as error:
                errors.append(error)
        if errors:
            self._state = 'failed'
            raise_failures(None, errors)
        self._state = 'closed'
        for wrapper in self._wrappers.values():
            wrapper._operation = None
        self._wrappers.clear()
        self._bindings.clear()
        self._model = None
