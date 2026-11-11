"""Private Half partial publication; numerical target controls live in cuda/."""
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import pytest
import torch

from tensorfold.cuda.sm70 import family_ops as ops, family_native


@pytest.mark.parametrize('rows', [0, 1, 256])
@pytest.mark.parametrize('inference', [False, True])
def test_success_publishes_exactly_one_output_mutation(rows, inference):
    with torch.inference_mode(inference):
        out = torch.empty((256, 2560))
    before = None if inference else out._version
    y = torch.empty((256, 11, 2560), dtype=torch.float16)
    weights, returned, calls = torch.empty((256, 11)), object(), []
    def partial(*args):
        calls.append(args)
        return returned
    with torch.inference_mode(inference), \
            patch.object(torch.Tensor, 'is_cuda', new_callable=PropertyMock, return_value=True), \
            patch.object(ops, '_PRIVATE_HALF_PARTIAL', 'native-half-v1'), \
            patch.object(torch.cuda, 'get_device_capability', return_value=(7, 0)), \
            patch.object(family_native, 'extension', return_value=SimpleNamespace(partial=partial)):
        assert ops.moe_partial(y, weights, out, rows) is returned
    assert calls == [(y, weights, out, rows)]
    if inference:
        assert out.is_inference()
    else:
        assert out._version == before + 1


def test_failure_does_not_publish_successful_mutation():
    out = torch.empty((1, 2560))
    before = out._version
    y = torch.empty((256, 11, 2560), dtype=torch.float16)
    primary = RuntimeError('native refusal')
    def partial(*args):
        raise primary
    with patch.object(torch.Tensor, 'is_cuda', new_callable=PropertyMock, return_value=True), \
            patch.object(ops, '_PRIVATE_HALF_PARTIAL', 'native-half-v1'), \
            patch.object(torch.cuda, 'get_device_capability', return_value=(7, 0)), \
            patch.object(family_native, 'extension', return_value=SimpleNamespace(partial=partial)):
        with pytest.raises(RuntimeError) as found:
            ops.moe_partial(y, torch.empty((256, 11)), out, 1)
    assert found.value is primary
    assert out._version == before


@pytest.mark.parametrize('private,dtype,cuda,capability', [
    ('inherit', torch.float16, True, (7, 0)),
    ('native-half-v1', torch.float32, True, (7, 0)),
    ('native-half-v1', torch.float16, False, (7, 0)),
    ('native-half-v1', torch.float16, True, (8, 0)),
])
def test_unselected_domains_keep_original_reference(private, dtype, cuda, capability):
    y = SimpleNamespace(is_cuda=cuda, dtype=dtype, device='cuda:0')
    args, expected = (y, object(), object(), 3), object()
    with patch.object(ops, '_PRIVATE_HALF_PARTIAL', private), patch.object(ops, '_NATIVE_FAMILY', False), \
            patch.object(torch.cuda, 'get_device_capability', return_value=capability), \
            patch.dict(ops._reference, moe_partial=lambda *got: expected if got == args else None), \
            patch.object(family_native, 'extension', side_effect=AssertionError('unexpected native load')):
        assert ops.moe_partial(*args) is expected


def test_explicit_reference_empty_copy_has_one_version_increment():
    out = torch.empty((1, 2560))
    before = out._version
    ops._reference['moe_partial'](torch.empty((1, 11, 2560)).half(), torch.empty((1, 11)), out, 0)
    assert out._version == before + 1


@pytest.mark.parametrize('domain', ['stride', 'grad', 'subclass', 'negative-view', 'alias-y', 'alias-weights',
    'inference-outside-mode'])
def test_generic_domains_delegate_once_without_loading_native(domain):
    y = torch.empty((2, 11, 2560), dtype=torch.float16)
    weights, out = torch.empty((2, 11)), torch.empty((2, 2560))
    if domain == 'stride':
        weights = torch.empty((2, 22))[:, ::2]
    elif domain == 'grad':
        y.requires_grad_(True)
    elif domain == 'subclass':
        class Other(torch.Tensor):
            pass
        y = y.as_subclass(Other)
    elif domain == 'negative-view':
        y = torch._neg_view(y)
    elif domain == 'alias-y':
        out = y.view(torch.float32).reshape(-1)[:5120].view(2, 2560)
    elif domain == 'alias-weights':
        weights = out.reshape(-1)[:22].view(2, 11)
    else:
        with torch.inference_mode():
            out = torch.empty((2, 2560))
    calls, result = [], object()
    def reference(*args):
        calls.append(args)
        return result
    with patch.object(torch.Tensor, 'is_cuda', new_callable=PropertyMock, return_value=True), \
            patch.object(ops, '_PRIVATE_HALF_PARTIAL', 'native-half-v1'), \
            patch.object(torch.cuda, 'get_device_capability', return_value=(7, 0)), \
            patch.dict(ops._reference, moe_partial=reference), \
            patch.object(family_native, 'extension', side_effect=AssertionError('generic path loaded native')):
        assert ops.moe_partial(y, weights, out, 2) is result
    assert len(calls) == 1 and all(a is b for a, b in zip(calls[0][:3], (y, weights, out)))
    assert calls[0][3] == 2


def test_inherit_preparation_is_inert_without_cuda_or_module_import():
    with patch.object(ops, '_PRIVATE_HALF_PARTIAL', 'inherit'), \
            patch.object(torch.cuda, 'get_device_capability', side_effect=AssertionError('CUDA query')), \
            patch.object(family_native, 'extension', side_effect=AssertionError('module load')):
        assert ops.prepare_half_moe_partial('cuda:0') is None


@pytest.mark.parametrize('device,capability', [('cpu', (7, 0)), ('cuda:0', (8, 0))])
def test_selected_preparation_refuses_unsupported_device(device, capability):
    with patch.object(ops, '_PRIVATE_HALF_PARTIAL', 'native-half-v1'), \
            patch.object(torch.cuda, 'get_device_capability', return_value=capability), \
            patch.object(family_native, 'extension', side_effect=AssertionError('unsupported load')):
        with pytest.raises(ValueError, match='SM70'):
            ops.prepare_half_moe_partial(device)


@pytest.mark.parametrize('failure', [False, True])
def test_preparation_forces_exact_function_under_selected_device(failure):
    from contextlib import contextmanager
    calls, primary = [], RuntimeError('function residency failed')
    @contextmanager
    def device_scope(device):
        calls.append(('enter', str(device)))
        try:
            yield
        finally:
            calls.append(('leave', str(device)))
    def force():
        calls.append('prepare_partial')
        if failure:
            raise primary
    module = SimpleNamespace(prepare_partial=force)
    def extension():
        calls.append('extension')
        return module
    with patch.object(ops, '_PRIVATE_HALF_PARTIAL', 'native-half-v1'), \
            patch.object(torch.cuda, 'get_device_capability', return_value=(7, 0)), \
            patch.object(torch.cuda, 'device', side_effect=device_scope), \
            patch.object(family_native, 'extension', side_effect=extension):
        if failure:
            with pytest.raises(RuntimeError) as found:
                ops.prepare_half_moe_partial('cuda:3')
            assert found.value is primary
        else:
            assert ops.prepare_half_moe_partial('cuda:3') is module
    assert calls == [('enter', 'cuda:3'), 'extension', 'prepare_partial', ('leave', 'cuda:3')]
