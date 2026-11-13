"""Owned family selection is explicit, isolated and preserves native ABI arguments."""
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from tensorfold.cuda.sm70 import family_ops, owned_family
from tensorfold.families.qwen4_exp.cuda import execution_identity, owner_preparation as preparation


def specification(**extra):
    return dict(kernel_family=preparation.RN_KERNEL_FAMILY, rn_heads_per_block=12,
                graph_family=preparation.UNCAPTURED, graph_budget_bytes=0, trial=True,
                **extra)


def test_explicit_policy_is_separate_numerical_identity_and_default_off():
    value = specification()
    assert 'native_family' not in preparation.parse_preparation(value)
    value['native_family'] = owned_family.POLICY
    owner = preparation.OwnerPreparation(value, role_rows={'decode': 120, 'mtp': 120, 'prefill': 256},
                                         manifest_sha256='a' * 64)
    identity = owner.numerical_selection['native_family']
    assert identity['reference_bit_identity'] is False
    assert identity['expert_kernels_and_linears'] == 'existing-selected-providers'
    assert owner.native_family is None
    # Preparing a policy does not mutate any other engine's dispatch.
    assert family_ops._NATIVE_FAMILY is False
    for invalid in (False, True, 'optimized', 'native', 1):
        with pytest.raises(ValueError, match='explicit RN'):
            preparation.parse_preparation({**value, 'native_family': invalid})
    for changed in ({'trial': False}, {'kernel_family': preparation.KERNEL_FAMILY},
                    {'reference_arithmetic': {'policy': 'reference-m32-v1', 'graph_budget_bytes': 1}}):
        with pytest.raises(ValueError):
            preparation.parse_preparation({**value, **changed})


@pytest.mark.parametrize('attribute,value', [('_owned_expert_requested', False),
    ('_owned_expert_admin_requested', True), ('sm70_profile', 'optimized'),
    ('expert_partition', NS(etp=8))])
def test_unsupported_engine_fails_before_native_or_state_allocation(monkeypatch, attribute, value):
    from tensorfold.cuda.sm70 import foundation_headers
    owner = preparation.OwnerPreparation(specification(native_family=owned_family.POLICY),
        role_rows={'decode': 120, 'mtp': 120, 'prefill': 256}, manifest_sha256='a' * 64)
    monkeypatch.setattr(foundation_headers, 'selected_identity', lambda: pytest.fail('reached native backend'))
    monkeypatch.setattr(execution_identity, '_retire_store', lambda *a, **k: None)
    owner._phase = lambda engine, name, operation, **kwargs: operation()
    engine = NS(comm=None, tp=8, position_profile=object(), method_geometry=NS(method='mtp'),
                _owned_expert_requested=True, _owned_expert_admin_requested=False,
                sm70_profile='reference', expert_partition=NS(etp=4))
    setattr(engine, attribute, value)
    text = dict(num_attention_heads=24, num_key_value_heads=2, hidden_size=2560, head_dim=256)
    with pytest.raises(ValueError, match='private TP8/ETP4'):
        owner.prepare(engine, text, NS(identity='q8'), page_tokens=128, device='cpu')
    assert owner.before is None and owner.native_family is None and owner.workspaces == {}


def test_per_buffer_dispatch_preserves_default_and_other_engine():
    from tensorfold.families.qwen4_exp.cuda.forward import _family
    from tensorfold.families.qwen4_exp.cuda import glue
    selected = object()
    assert _family(NS(native_family=selected)) is selected
    assert _family(NS(native_family=None)) is glue
    assert _family(NS()) is glue


def test_ple_live_rows_independence_and_mutation_metadata_reach_existing_abi():
    module = NS(ple_gate=Mock())
    prepared = owned_family.PreparedFamily(module, 'cpu')
    keys, values, h = torch.zeros((3, 32)), torch.zeros((3, 8)), torch.zeros((3, 32))
    norm, gated, pss, live = torch.ones(32), torch.empty_like(h), torch.empty((3, 4)), torch.tensor([2])
    versions = gated._version, pss._version
    prepared.ple_gate(keys, values, h, norm, norm, gated, pss, 1e-6, 4,
                      valid_rows=live, independent_rows=True)
    module.ple_gate.assert_called_once_with(keys, values, h, norm, norm, gated, pss, 4,
                                             1e-6, live, True)
    assert (gated._version, pss._version) == (versions[0] + 3, versions[1] + 3)


def test_raw_writeback_publishes_exact_residual_alias_once_and_failures_propagate():
    module = NS(writeback=Mock())
    prepared = owned_family.PreparedFamily(module, 'cpu')
    h, pss = torch.zeros((1, 10240)), torch.empty((1, 10, 4))
    version = h._version
    prepared.hc_writeback(h, h, pss, 4, 0)
    assert h._version == version + 1 and pss._version == 1
    module.writeback.side_effect = RuntimeError('native launch failed')
    with pytest.raises(RuntimeError, match='native launch failed'):
        prepared.hc_writeback(h, h, pss, 4, 0)
    assert h._version == version + 1


def test_mtp_norm_add_and_shift_use_caller_owned_storage():
    module = NS(rms=Mock(), add=Mock(), shift=Mock())
    prepared = owned_family.PreparedFamily(module, 'cpu')
    x, scale, out, xs = torch.zeros((1, 32)), torch.ones(32), torch.empty((1, 32)), torch.empty((1, 1))
    result = prepared.rmsnorm(x, scale, 1e-6, out=out, xs=xs)
    assert result[0] is out and result[1] is xs
    module.rms.assert_called_once_with(x, scale, out, xs, 32, 1e-6)
    with pytest.raises(ValueError, match='caller-owned'):
        prepared.rmsnorm(x, scale, 1e-6)
    hidden, output = torch.zeros((4, 32)), torch.empty((1, 128))
    prepared.add_streams(x, hidden, output, 4)
    module.add.assert_called_once_with(x, hidden, output, 4)
    old, new = torch.zeros((1, 3, 32)), torch.ones((1, 1, 32))
    prepared.shift_windows(old, new, 1, 32)
    module.shift.assert_called_once_with(old, new, 1, 32)
    assert old._version == 1


def test_no_inference_version_counter_is_required():
    with torch.inference_mode():
        value = torch.zeros((1, 32))
        owned_family.PreparedFamily(NS(add=Mock()), 'cpu').add_streams(value, value, value, 1)


def test_adoption_rejects_unbound_or_foreign_family_capability(monkeypatch):
    from tensorfold.cuda.sm70 import foundation_headers
    owner = preparation.OwnerPreparation(specification(native_family=owned_family.POLICY),
        role_rows={'decode': 120, 'mtp': 120, 'prefill': 256}, manifest_sha256='a' * 64)
    owner.prepared = True
    owner.source_foundation = {'id': 'fixture'}
    owner.native_family = object()
    owner.workspaces = {'decode': NS(native_kernel_family=preparation.RN_KERNEL_FAMILY,
        native_heads_per_block=12, bind_native_states=Mock())}
    monkeypatch.setattr(foundation_headers, 'selected_identity', lambda: owner.source_foundation)
    buffers = NS(history_owner=owner.workspaces['decode'], native_family=object())
    decoder = NS(states=[], buf=buffers, mbuf=None, pbuf=None)
    weights = NS(meta={'owner_workspaces': owner.workspaces})
    with pytest.raises(RuntimeError, match='prepared native family capability'):
        owner.adopt(weights, decoder)


def test_qualification_receipt_distinguishes_capability_from_invocation():
    from tools.v100_owned_qualification import _native_family_receipt
    record = {'policy': owned_family.POLICY}
    capability = NS(record=lambda: record)
    weights = NS(_private_native_family=capability, meta={'private_native_family_policy': record})
    engine = NS(w=weights, _owner_preparation=NS(native_family=capability,
                specification={'native_family': owned_family.POLICY}),
                multi=NS(buf=NS(rows=120, native_family=capability),
                         mbuf=NS(rows=120, native_family=capability),
                         pbuf=NS(rows=256, native_family=capability)))
    receipt = _native_family_receipt(engine)
    assert receipt['roles'] == {'decode': {'rows': 120, 'bound': True},
        'mtp': {'rows': 120, 'bound': True}, 'prefill': {'rows': 256, 'bound': True}}
    assert receipt['invocation_evidence'] is False and receipt['P3_valid'] is False
    engine.multi.mbuf.native_family = None
    with pytest.raises(ValueError, match='role did not adopt'):
        _native_family_receipt(engine)
