"""Prepared router ownership/control tests; fake CPU native calls are not parity."""
from contextlib import contextmanager
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.cuda import moe
from tensorfold.cuda.sm70 import routing_native as routing
from tensorfold.families.qwen4_exp.cuda import execution_identity
from tensorfold.families.qwen4_exp.cuda import owner_preparation as preparation


def aligned_weight():
    storage = torch.zeros(513*2560+128, dtype=torch.float32)
    offset = (-storage.data_ptr()//4) % 128
    return storage[offset:offset+513*2560].view(513, 2560)


def fixture(rows=40):
    calls = []
    def logits(source, weight, output):
        calls.append((source, weight, output))
        output.numpy().fill(7)  # Raw write, without an ATen mutation publication.
    module = NS(__name__='owned_test_module', __file__='/test/owned.so', prepared_logits=logits)
    cap = routing.PreparedSerialRouter(module, 'cpu', {'decode': rows, 'mtp': rows, 'prefill': 256})
    target, draft = aligned_weight(), aligned_weight()
    weights = NS(device=torch.device('cpu'), cfg=NS(hidden=2560, experts=512, top_k=10, execution_dtype='float16'),
        layers=[NS(moe=NS(router=target))], mtp=NS(layer=NS(moe=NS(router=draft))))
    cap.bind_weights(weights)
    roles = {}
    for role in ('decode', 'mtp'):
        buffers = NS(owner_role=role, rows=rows, prefill=False,
            mixed=torch.zeros(rows, 2560, dtype=torch.float16),
            moe=NS(logits=torch.zeros(rows, 513, dtype=torch.float32)))
        buffers.router_logits = cap.bind_buffers(buffers)
        roles[role] = buffers
    weights._private_prepared_router = cap
    return cap, weights, roles, calls


def spec(**extra):
    return {'kernel_family': preparation.RN_KERNEL_FAMILY, 'rn_heads_per_block': 12,
        'graph_family': preparation.UNCAPTURED, 'graph_budget_bytes': 0, 'trial': True, **extra}


def test_policy_is_explicit_and_separate_from_existing_global_v1():
    before = routing._NATIVE_LOGITS, routing._PRIVATE_SERIAL_LOGITS
    assert 'prepared_router' not in preparation.parse_preparation(spec())
    value = spec(prepared_router=routing.PREPARED_LOGITS_POLICY)
    owner = preparation.OwnerPreparation(value, role_rows={'decode': 40, 'mtp': 40, 'prefill': 256},
        manifest_sha256='a'*64)
    identity = owner.numerical_selection['prepared_router']
    assert identity['reference_bit_identity'] is False
    assert identity['one_row'] == 'existing-serial-reference'
    assert identity['selection_and_expert_weights'] == 'existing-selected-provider-unchanged'
    assert owner.prepared_router is None
    for bad in (False, True, 1, 'native-serial-v1', 'owned-serial-router-v2'):
        with pytest.raises(ValueError, match='serial router'):
            preparation.parse_preparation(spec(prepared_router=bad))
    with pytest.raises(ValueError, match='trial'):
        preparation.parse_preparation({**value, 'trial': False})
    assert (routing._NATIVE_LOGITS, routing._PRIVATE_SERIAL_LOGITS) == before


@pytest.mark.parametrize('attribute,value', [('_owned_expert_requested', False),
    ('_owned_expert_admin_requested', True), ('sm70_profile', 'optimized'),
    ('expert_partition', NS(etp=8))])
def test_region_rejection_precedes_native_residency_and_state(monkeypatch, attribute, value):
    from tensorfold.cuda.sm70 import foundation_headers
    owner = preparation.OwnerPreparation(spec(prepared_router=routing.PREPARED_LOGITS_POLICY),
        role_rows={'decode': 40, 'mtp': 40, 'prefill': 256}, manifest_sha256='a'*64)
    monkeypatch.setattr(foundation_headers, 'selected_identity', lambda: pytest.fail('native source reached'))
    monkeypatch.setattr(execution_identity, '_retire_store', lambda *a, **kw: None)
    owner._phase = lambda engine, name, operation, **kw: operation()
    engine = NS(comm=None, tp=8, position_profile=object(), method_geometry=NS(method='mtp'),
        _owned_expert_requested=True, _owned_expert_admin_requested=False,
        sm70_profile='reference', expert_partition=NS(etp=4))
    setattr(engine, attribute, value)
    with pytest.raises(ValueError, match='private TP8/ETP4'):
        owner.prepare(engine, {'num_attention_heads': 24, 'num_key_value_heads': 2,
            'hidden_size': 2560, 'head_dim': 256}, NS(identity='q8'), page_tokens=128, device='cpu')
    assert owner.before is owner.prepared_router is None


def test_preparation_resolves_and_warms_selected_function_before_return(monkeypatch):
    calls = []
    module = NS(prepared_logits=lambda *a: pytest.fail('cold numerical call'),
                prepare_logits=lambda: calls.append('resident'))
    @contextmanager
    def device(value):
        calls.append(('enter', str(value)))
        yield
        calls.append('exit')
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda d: (7, 0))
    monkeypatch.setattr(torch.cuda, 'device', device)
    monkeypatch.setattr(routing, 'extension', lambda: (calls.append('module'), module)[1])
    prepared = routing.prepare_serial_router('cuda:2', {'decode': 256, 'mtp': 40})
    assert prepared.module is module
    assert calls == [('enter', 'cuda:2'), 'module', 'resident', 'exit']
    module.prepare_logits = None
    with pytest.raises(ValueError, match='built native entry'):
        routing.prepare_serial_router('cuda:2', {'decode': 40})


@pytest.mark.parametrize('role', ['decode', 'mtp'])
@pytest.mark.parametrize('rows', [0, 1, 2, 33, 40])
@torch.no_grad()
def test_prepared_call_uses_actual_role_owners_and_reference_row_publication(monkeypatch, role, rows):
    cap, weights, buffers, calls = fixture()
    b = buffers[role]
    weight = weights.layers[0].moe.router if role == 'decode' else weights.mtp.layer.moe.router
    supports = routing.supports_logits
    monkeypatch.setattr(routing, 'extension', lambda: pytest.fail('hot loader'))
    monkeypatch.setattr(routing, 'supports_logits', lambda *a: pytest.fail('hot support query'))
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda *a: pytest.fail('hot hardware query'))
    version = b.moe.logits._version
    selected = b.router_logits.run(b.mixed[:rows], weight, b.moe.logits[:rows])
    assert selected is (rows >= 2)
    assert len(calls) == (1 if rows >= 2 else 0)
    assert b.moe.logits._version-version == (rows if rows >= 2 else 0)
    if selected:
        assert calls[0][1] is weight and calls[0][0].data_ptr() == b.mixed.data_ptr()
    monkeypatch.setattr(routing, 'supports_logits', supports)
    reference = torch.empty(rows, 513)
    before = reference._version
    moe.router(b.mixed[:rows], weight, reference, independent_rows=True)
    assert reference._version-before == rows
    assert cap.record()['invocation_evidence'] is False
    cap.close()


@torch.no_grad()
def test_foreign_weight_pointer_and_lifetime_cannot_invoke_prepared_code():
    cap, weights, buffers, calls = fixture()
    b = buffers['decode']
    with pytest.raises(ValueError, match='ownership binding'):
        b.router_logits.run(b.mixed, weights.mtp.layer.moe.router, b.moe.logits)
    b.mixed = torch.zeros_like(b.mixed)
    with pytest.raises(ValueError, match='ownership binding'):
        b.router_logits.run(b.mixed, weights.layers[0].moe.router, b.moe.logits)
    cap.close()
    with pytest.raises(RuntimeError, match='closed'):
        b.router_logits.run(b.mixed[:1], weights.layers[0].moe.router, b.moe.logits[:1])
    assert calls == []


def test_autograd_context_preserves_reference_fallback():
    cap, weights, buffers, calls = fixture()
    b = buffers['decode']
    with torch.enable_grad():
        assert not b.router_logits.run(b.mixed, weights.layers[0].moe.router, b.moe.logits)
    assert calls == []
    cap.close()


@pytest.mark.parametrize('fault', ['half', 'shape', 'alignment', 'gradient', 'mtp-missing', 'execution-dtype'])
def test_explicit_unavailable_weight_request_is_not_published_or_substituted(fault):
    module = NS(prepared_logits=lambda *a: pytest.fail('cold numerical call'))
    cap = routing.PreparedSerialRouter(module, 'cpu', {'decode': 40, 'mtp': 40})
    weight = aligned_weight()
    target = NS(moe=NS(router=weight))
    weights = NS(device=torch.device('cpu'), cfg=NS(hidden=2560, experts=512, top_k=10, execution_dtype='float16'),
        layers=[target], mtp=NS(layer=NS(moe=NS(router=weight))))
    if fault == 'half':
        target.moe.router = weight.half()
    elif fault == 'shape':
        target.moe.router = weight[:512]
    elif fault == 'alignment':
        storage = torch.empty(513*2560+129, dtype=torch.float32)
        offset = (-storage.data_ptr()//4) % 128+1
        target.moe.router = storage[offset:offset+513*2560].reshape(513, 2560)
    elif fault == 'gradient':
        weight.requires_grad_(True)
    elif fault == 'mtp-missing':
        weights.mtp = None
    else:
        weights.cfg.execution_dtype = 'bfloat16'
    with pytest.raises(ValueError, match='serial router'):
        cap.bind_weights(weights)
    assert cap.weights is cap.weight_owner is None and cap.buffers == {}


def test_pre_state_weight_phase_retires_and_adoption_checks_actual_buffer_binding(monkeypatch):
    original, weights, buffers, _calls = fixture()
    cap = routing.PreparedSerialRouter(original.module, 'cpu', {'decode': 40, 'mtp': 40})
    original.close()
    weights._private_prepared_router = cap
    owner = preparation.OwnerPreparation(spec(prepared_router=routing.PREPARED_LOGITS_POLICY),
        role_rows={'decode': 40, 'mtp': 40, 'prefill': 256}, manifest_sha256='a'*64)
    owner.prepared_router, owner.prepared = cap, True
    events = []
    owner._phase = lambda engine, stage, operation, **kw: (events.append(stage), operation())[1]
    monkeypatch.setattr(execution_identity, '_retire_store', lambda e, c, *, stages: events.append(stages))
    owner.bind_router_weights(object(), weights)
    assert events == ['serial-router-weight-binding', ('serial-router-weight-binding',)]
    for b in buffers.values():
        b.router_logits = cap.bind_buffers(b)
    decoder = NS(buf=buffers['decode'], mbuf=buffers['mtp'], pbuf=NS(router_logits=None))
    cap.validate_adoption(weights, decoder)
    foreign = NS(**vars(weights))
    with pytest.raises(ValueError, match='Weights owner'):
        cap.validate_adoption(foreign, decoder)
    decoder.mbuf = NS(router_logits=None)
    with pytest.raises(ValueError, match='Buffers owner'):
        cap.validate_adoption(weights, decoder)
    cap.close()
