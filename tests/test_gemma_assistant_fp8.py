"""CPU ownership/interface controls; real FP8 GPU numerics remain a root gate."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import weakref

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tensorfold.families.gemma4.assistant_metadata import AssistantConfig
from tensorfold.families.gemma4.cuda.assistant_fp8 import AssistantFP8Core, _Linear, memory_bytes, projection_shapes


class ShapeOnlyOperation:
    """No tensor/kernel simulation: metadata-only substitute for lifecycle tests."""
    def __init__(self, weight):
        from tensorfold.cuda.requant import resident_bytes
        self.bytes = resident_bytes('fp8', *weight.shape)
    def nbytes(self):
        return self.bytes
    def __call__(self, _value):
        raise AssertionError('shape-only test operation cannot establish GPU math')


def skeleton():
    raw = json.loads((Path(__file__).parent / 'fixtures/gemma4_assistant_config.json').read_text())
    model = nn.Module()
    model.config = SimpleNamespace(to_dict=lambda: deepcopy(raw))
    for path, (n, k) in projection_shapes(AssistantConfig()).items():
        parts = path.split('.')
        parent = model
        for part in parts[:-1]:
            if not hasattr(parent, part):
                parent.add_module(part, nn.Module())
            parent = getattr(parent, part)
        parent.add_module(parts[-1], nn.Linear(k, n, bias=False, dtype=torch.bfloat16, device='meta'))
    model.masked_embedding = nn.Module()
    model.masked_embedding.centroids = nn.Linear(256, 2048, bias=False, dtype=torch.bfloat16, device='meta')
    model.masked_embedding.register_buffer('token_ordering', torch.empty(262144, dtype=torch.int64, device='meta'))
    model.model.embed_tokens = nn.Embedding(262144, 256, dtype=torch.bfloat16, device='meta')
    model.lm_head = nn.Linear(256, 262144, bias=False, dtype=torch.bfloat16, device='meta')
    model.lm_head.weight = model.model.embed_tokens.weight
    model.model.norm = nn.Identity()
    return model.eval()


def fake_preparation(monkeypatch, model, drain):
    import tensorfold.cuda.requant as requant
    monkeypatch.setattr(requant, 'requantize', lambda weight, kind: ShapeOnlyOperation(weight))
    return AssistantFP8Core(model, drain=drain)


def test_exact22_selection_and_additional_original_owner_accounting():
    shapes = projection_shapes(AssistantConfig())
    assert len(shapes) == 22
    assert not any('masked_embedding' in path or 'embed_tokens' in path or 'lm_head' in path for path in shapes)
    count = memory_bytes(AssistantConfig())
    assert count.original_core_bytes == 21757952
    assert count.extra_resident_bytes == 10984448
    assert count.extra_loading_peak_bytes == 48021504


def test_prepare_does_not_publish_then_install_keeps_real_weights_and_all_unselected_owners(monkeypatch):
    model = skeleton()
    originals = {path:model.get_submodule(path) for path in projection_shapes(AssistantConfig())}
    centroid, embedding, head = model.masked_embedding.centroids, model.model.embed_tokens, model.lm_head
    norm = model.model.norm
    calls = []
    bundle = fake_preparation(monkeypatch, model, lambda: calls.append('drain'))
    assert all(model.get_submodule(path) is original for path, original in originals.items())
    assert bundle.install() is model
    wrapped = model.pre_projection
    assert wrapped.weight is originals['pre_projection'].weight
    assert wrapped.weight.device == originals['pre_projection'].weight.device
    assert wrapped.in_features == 5120 and wrapped.out_features == 256 and wrapped.bias is None
    assert model.masked_embedding.centroids is centroid and model.model.embed_tokens is embedding
    assert model.lm_head is head and model.model.norm is norm and head.weight is embedding.weight
    bundle.close()
    assert calls == ['drain'] and bundle._state == 'closed'
    assert all(model.get_submodule(path) is original for path, original in originals.items())
    assert wrapped._operation is None
    with pytest.raises(RuntimeError, match='retired'):
        wrapped(torch.empty(1, 5120, dtype=torch.bfloat16, device='meta'))
    bundle.close()
    assert calls == ['drain']


def tiny_wrapper():
    owner = AssistantFP8Core.__new__(AssistantFP8Core)
    owner._state, owner._reference_depth, owner._reference_thread = 'installed', 0, None
    native = nn.Linear(64, 8, bias=False, dtype=torch.bfloat16).eval()
    wrapper = _Linear(native, lambda value:F.linear(value, native.weight) * 2, owner)
    return owner, native, wrapper


def test_private_dispatch_and_nested_native_reference_are_distinct_and_exception_safe():
    owner, native, wrapper = tiny_wrapper()
    value = torch.ones(1, 1, 64, dtype=torch.bfloat16)
    original = native.weight.detach().clone()
    with torch.no_grad():
        expected = native(value)
        assert torch.equal(wrapper(value), expected * 2)
        with owner.native_reference():
            assert torch.equal(wrapper(value), expected)
            with owner.native_reference():
                assert torch.equal(wrapper(value), expected)
        with pytest.raises(ValueError):
            with owner.native_reference():
                raise ValueError('oracle failed')
        assert owner._reference_depth == 0 and owner._reference_thread is None
        assert torch.equal(wrapper(value), expected * 2)
    assert torch.equal(native.weight, original)
    with pytest.raises(RuntimeError, match='inference-only'):
        wrapper(value)
    owner._reference_depth, owner._reference_thread = 1, threading.get_ident() + 1
    with pytest.raises(RuntimeError, match='another'):
        wrapper(value)


def test_modified_source_refuses_stale_private_copy():
    owner, native, wrapper = tiny_wrapper()
    with torch.no_grad():
        native.weight.add_(1)
        with pytest.raises(RuntimeError, match='source changed'):
            wrapper(torch.ones(1, 64, dtype=torch.bfloat16))
        with owner.native_reference():
            assert torch.equal(wrapper(torch.ones(1, 64, dtype=torch.bfloat16)), native(torch.ones(1, 64, dtype=torch.bfloat16)))


def test_install_failure_restores_every_attempted_module_and_preserves_primary(monkeypatch):
    model = skeleton()
    originals = {path:model.get_submodule(path) for path in projection_shapes(AssistantConfig())}
    bundle = fake_preparation(monkeypatch, model, lambda: None)
    primary = ValueError('registration callback failed')
    assign = nn.Module.__setattr__
    def fail(module, name, value):
        if module is model and name == 'post_projection' and isinstance(value, _Linear):
            assign(module, name, value)  # failure after publication must roll this attempted assignment back too
            raise primary
        return assign(module, name, value)
    with monkeypatch.context() as patch:
        patch.setattr(nn.Module, '__setattr__', fail)
        with pytest.raises(ValueError) as caught:
            bundle.install()
        assert caught.value is primary
    assert bundle._state == 'prepared'
    assert all(model.get_submodule(path) is original for path, original in originals.items())
    bundle.close()


def test_drain_failure_retains_private_and_installed_owners_until_explicit_retirement(monkeypatch):
    model = skeleton()
    fail = [True]
    error = OSError('reader did not retire')
    def drain():
        if fail[0]:
            raise error
    bundle = fake_preparation(monkeypatch, model, drain)
    bundle.install()
    wrapped = model.pre_projection
    with pytest.raises(OSError) as caught:
        bundle.close()
    assert caught.value is error and bundle._state == 'failed'
    assert model.pre_projection is wrapped and wrapped._operation is not None
    fail[0] = False
    bundle.close()
    assert bundle._state == 'closed' and model.pre_projection is wrapped.native


def test_failed_preparation_clears_unpublished_private_owners_and_never_changes_model(monkeypatch):
    import tensorfold.cuda.requant as requant
    model = skeleton()
    original = model.pre_projection
    weak, calls, drains = [], [], []
    primary = MemoryError('second private copy exhausted')
    def prepare(weight, kind):
        calls.append(kind)
        if len(calls) == 2:
            raise primary
        operation = ShapeOnlyOperation(weight)
        weak.append(weakref.ref(operation))
        return operation
    monkeypatch.setattr(requant, 'requantize', prepare)
    with pytest.raises(MemoryError) as caught:
        AssistantFP8Core(model, drain=lambda: drains.append('drain'))
    assert caught.value is primary and weak[0]() is None
    assert model.pre_projection is original and drains == ['drain']


def test_reject_invalid_source_before_any_private_preparation(monkeypatch):
    import tensorfold.cuda.requant as requant
    model = skeleton()
    model.post_projection = nn.Linear(256, 2560, bias=True, dtype=torch.bfloat16, device='meta')
    calls = []
    monkeypatch.setattr(requant, 'requantize', lambda *args:calls.append('should not run'))
    with pytest.raises(ValueError, match='BF16 Linear'):
        AssistantFP8Core(model, drain=lambda: None)
    assert not calls
    with pytest.raises(ValueError, match='drain'):
        AssistantFP8Core(model, drain=None)


def test_partial_wrapper_metadata_failure_drops_owned_operation_without_stripping_traceback(monkeypatch):
    import tensorfold.cuda.requant as requant
    from tensorfold.families.gemma4.cuda import assistant_fp8
    model = skeleton()
    weak, drains = [], []
    primary = ValueError('wrapper train metadata failed')
    def prepare(weight, kind):
        operation = ShapeOnlyOperation(weight)
        weak.append(weakref.ref(operation))
        return operation
    def fail_train(self, mode=True):
        raise primary
    monkeypatch.setattr(requant, 'requantize', prepare)
    monkeypatch.setattr(assistant_fp8._Linear, 'train', fail_train)
    with pytest.raises(ValueError) as caught:
        AssistantFP8Core(model, drain=lambda: drains.append('drain'))
    assert caught.value is primary and weak[0]() is None and drains == ['drain']
    trace, frames = primary.__traceback__, []
    while trace is not None:
        frames.append(trace.tb_frame)
        trace = trace.tb_next
    constructor = [frame for frame in frames if frame.f_code.co_name == '__init__'
                   and isinstance(frame.f_locals.get('self'), assistant_fp8._Linear)]
    assert constructor and constructor[0].f_locals['operation'] is None
    assert constructor[0].f_locals['self']._operation is None
    assert any(frame.f_code.co_name == 'fail_train' for frame in frames)


def test_linear_subclass_with_custom_forward_is_not_silently_bypassed(monkeypatch):
    import tensorfold.cuda.requant as requant
    model = skeleton()
    class DifferentLinear(nn.Linear):
        def forward(self, value):
            return super().forward(value) * 2
    model.post_projection = DifferentLinear(256, 2560, bias=False, dtype=torch.bfloat16, device='meta')
    calls = []
    monkeypatch.setattr(requant, 'requantize', lambda *args:calls.append('should not run'))
    with pytest.raises(ValueError, match='BF16 Linear'):
        AssistantFP8Core(model, drain=lambda: None)
    assert not calls


def test_partial_wrapper_base_init_failure_does_not_retain_operation_argument(monkeypatch):
    import tensorfold.cuda.requant as requant
    from tensorfold.families.gemma4.cuda import assistant_fp8
    model = skeleton()
    weak = []
    primary = MemoryError('base Module metadata allocation')
    def prepare(weight, kind):
        operation = ShapeOnlyOperation(weight)
        weak.append(weakref.ref(operation))
        return operation
    base = nn.Module.__init__
    def fail_base(self, *args, **kwargs):
        if isinstance(self, assistant_fp8._Linear):
            raise primary
        return base(self, *args, **kwargs)
    monkeypatch.setattr(requant, 'requantize', prepare)
    monkeypatch.setattr(nn.Module, '__init__', fail_base)
    with pytest.raises(MemoryError) as caught:
        AssistantFP8Core(model, drain=lambda: None)
    assert caught.value is primary and weak[0]() is None
    trace, owned = primary.__traceback__, []
    while trace is not None:
        if trace.tb_frame.f_code.co_name == '__init__' and isinstance(trace.tb_frame.f_locals.get('self'), assistant_fp8._Linear):
            owned.append(trace.tb_frame)
        trace = trace.tb_next
    assert owned and owned[0].f_locals['operation'] is None
