"""Operation-owned selection/residency/ownership controls, without CUDA claims."""
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.cuda.sm70 import family_ops as ops
from tensorfold.families.qwen4_exp.cuda import history_owner as history


def test_cold_preparation_resolves_operator_in_selected_device_before_return(monkeypatch):
    from contextlib import contextmanager
    from tensorfold.cuda.sm70 import family_native
    calls = []
    @contextmanager
    def device(value):
        calls.append(('enter', str(value)))
        yield
        calls.append(('exit', str(value)))
    module = NS(prepare_owned_q8=lambda: calls.append('resident'))
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda d: (7, 0))
    monkeypatch.setattr(torch.cuda, 'device', device)
    monkeypatch.setattr(family_native, 'extension', lambda: module)
    assert ops.prepare_owned_q8('cuda:2') is module
    assert calls == [('enter', 'cuda:2'), 'resident', ('exit', 'cuda:2')]


@pytest.mark.parametrize('device,capability', [('cpu', (7, 0)), ('cuda:0', (8, 0))])
def test_cold_region_refusal_never_resolves_native_module(monkeypatch, device, capability):
    from tensorfold.cuda.sm70 import family_native
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda d: capability)
    monkeypatch.setattr(family_native, 'extension', lambda: pytest.fail('native resolution'))
    with pytest.raises(ValueError, match='SM70'):
        ops.prepare_owned_q8(device)


@pytest.mark.parametrize('case', ['prefill', 'mtp', 'multirow', 'old-family', 'graph', 'unprepared', 'unissued'])
def test_unselected_prepare_preserves_original_call_and_return(monkeypatch, case):
    marker = object()
    buffers = NS(prefill=False, owner_kernel_family='sm70-paged-owner-rn-ex2-p2051-v1',
                 owner_graph_family='uncaptured', history_owner=NS(owned_q8_preparation=object()),
                 target_indexer_coordinates=object())
    mtp, end = False, 1
    if case == 'prefill':
        buffers.prefill = True
    elif case == 'mtp':
        mtp = True
    elif case == 'multirow':
        end = 2
    elif case == 'old-family':
        buffers.owner_kernel_family = 'torch-reference'
    elif case == 'graph':
        buffers.owner_graph_family = 'captured'
    elif case == 'unprepared':
        buffers.history_owner.owned_q8_preparation = None
    else:
        buffers.target_indexer_coordinates = None
    args = (object(), object(), object(), object(), buffers, 0, end)
    calls = []
    monkeypatch.setattr(history, '_prepare_reference', lambda *a, **k: (calls.append((a, k)), marker)[1])
    assert history.prepare(*args, mtp=mtp) is marker
    assert calls == [(args, {'mtp': mtp})]


@pytest.mark.parametrize('fault', ['grad', 'inference'])
def test_owned_operator_preserves_unsupported_autograd_and_inference_domain(fault):
    tensor = torch.zeros(1)
    if fault == 'inference':
        with torch.inference_mode():
            tensor = torch.zeros(1)
    cache = NS(k=tensor, v=tensor, ks=tensor, vs=tensor)
    state = NS(kv_status=tensor)
    module = NS(owned_q8_prepare=lambda *a: pytest.fail('native mutation'))
    context = torch.enable_grad() if fault == 'grad' else torch.no_grad()
    with context:
        assert not ops.owned_q8_prepare(module, tensor, None, None, cache, state, None, tensor, tensor,
                                      frequency=None, sections=None, amplitude=None, start=0)


@pytest.mark.parametrize('prefill,family,prepared', [(False, 'sm70-paged-owner-rn-ex2-p2051-v1', True),
                                                  (True, 'sm70-paged-owner-rn-ex2-p2051-v1', False),
                                                  (False, 'sm70-paged-owner-q8-compact-v1', False)])
def test_only_decode_RN_owner_retains_cold_capability_before_adoption(monkeypatch, prefill, family, prepared):
    from tensorfold.families.qwen4_exp.kv_formats import get_pair
    owner = object.__new__(history.OwnerBuffers)
    owner._empty_resources()
    owner.prefill, owner.device, owner.rows = prefill, torch.device('cpu'), 1
    owner.native_pair = get_pair('int8')
    owner.partials = torch.empty((8, 1, 12, 256), dtype=torch.float32)
    owner.partial = torch.empty((1, 12, 256), dtype=torch.float32)
    module = NS(__name__='actual_resident_module')
    calls = []
    monkeypatch.setattr(ops, 'prepare_owned_q8', lambda dev: (calls.append(dev), module)[1])
    owner.configure_native_family(family, 12 if 'rn-ex2' in family else None)
    assert calls == ([torch.device('cpu')] if prepared else [])
    assert owner.owned_q8_preparation is (module if prepared else None)
    assert owner._native_family_configured


def test_cold_failure_keeps_configuration_uncommitted_and_original_exception(monkeypatch):
    from tensorfold.families.qwen4_exp.kv_formats import get_pair
    owner = object.__new__(history.OwnerBuffers)
    owner._empty_resources()
    owner.prefill, owner.device, owner.rows = False, torch.device('cpu'), 1
    owner.native_pair = get_pair('int8')
    owner.partials = torch.empty((8, 1, 12, 256), dtype=torch.float32)
    owner.partial = torch.empty((1, 12, 256), dtype=torch.float32)
    primary = MemoryError('code residency')
    def fail(device):
        raise primary
    monkeypatch.setattr(ops, 'prepare_owned_q8', fail)
    with pytest.raises(MemoryError) as caught:
        owner.configure_native_family('sm70-paged-owner-rn-ex2-p2051-v1', 12)
    assert caught.value is primary
    assert not owner._native_family_configured and owner.owned_q8_preparation is None
