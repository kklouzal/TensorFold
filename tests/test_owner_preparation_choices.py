"""Opt-in numerical choices fail before allocation and bind their versioned contract."""
from types import SimpleNamespace as NS

import pytest

from tensorfold.families.qwen4_exp.cuda import execution_identity, owner_preparation as preparation


def spec(**extra):
    return dict(kernel_family=preparation.KERNEL_FAMILY, graph_family=preparation.UNCAPTURED,
                graph_budget_bytes=0, trial=True, **extra)


def owner(**extra):
    return preparation.OwnerPreparation(spec(**extra), role_rows={'decode': 4, 'prefill': 8},
                                        manifest_sha256='a' * 64)


def test_native_gdn_capture_is_opt_in_and_false_preserves_old_identity():
    assert preparation.parse_preparation(spec(native_gdn_capture=False)) == preparation.parse_preparation(spec())
    assert preparation.parse_preparation(spec(native_gdn_capture=True))['native_gdn_capture'] is True
    for value in (0, 1, 'true', None):
        with pytest.raises(ValueError, match='boolean'):
            preparation.parse_preparation(spec(native_gdn_capture=value))
    with pytest.raises(ValueError, match='trial'):
        preparation.parse_preparation({**spec(native_gdn_capture=True), 'trial': False})


def dense_spec():
    return dict(backend='cublas', geometry='fixed-m32', roles=['decode'], panel_rows=32)


def test_dense_numerics_publish_versioned_policy_and_unchanged_mtp():
    assert owner().numerical_selection is None
    identity = owner(dense_projections=dense_spec()).numerical_selection['dense_projections']
    assert isinstance(identity['policy'], str) and identity['policy']
    assert identity['MTP_dense'] == 'existing-reference-provider'
    assert identity['HC_decode_partials'] == 'float32'
    assert identity['roles'] == ['decode']


@pytest.mark.parametrize('position,method', [(None, 'none'), (None, 'mtp'),
                                           (object(), 'dspark'), (object(), 'dflash2')])
def test_new_numerics_require_bound_none_or_mtp_before_backend_or_allocation(monkeypatch, position, method):
    from tensorfold.cuda.sm70 import foundation_headers
    from tensorfold.families.qwen4_exp.cuda import dense_projections
    prepared = owner(dense_projections=dense_spec())
    monkeypatch.setattr(dense_projections, 'forecast', lambda *a, **k: {})
    monkeypatch.setattr(foundation_headers, 'selected_identity', lambda: pytest.fail('unsupported choice reached backend'))
    monkeypatch.setattr(execution_identity, '_retire_store', lambda *a, **k: None)
    prepared._phase = lambda engine, name, operation, **kwargs: operation()
    engine = NS(comm=None, tp=8, position_profile=position, method_geometry=NS(method=method))
    text = dict(num_attention_heads=24, num_key_value_heads=2, hidden_size=2560, head_dim=256)
    with pytest.raises(ValueError, match='bound NONE/MTP'):
        prepared.prepare(engine, text, NS(identity='q8'), page_tokens=128, device='cpu')
    assert prepared.before is None and prepared.workspaces == {} and prepared.dense_owner is None


def test_unavailable_requested_native_capture_cannot_silently_select_scalar(monkeypatch):
    from tensorfold.cuda.sm70 import foundation_headers
    prepared = owner(native_gdn_capture=True)
    monkeypatch.setattr(foundation_headers, 'selected_identity', lambda: pytest.fail('unavailable capture reached backend'))
    monkeypatch.setattr(execution_identity, '_retire_store', lambda *a, **k: None)
    prepared._phase = lambda engine, name, operation, **kwargs: operation()
    engine = NS(comm=None, tp=8, _native_gdn_capture_enabled=False)
    text = dict(num_attention_heads=24, num_key_value_heads=2, hidden_size=2560, head_dim=256)
    with pytest.raises(ValueError, match='private exact graph'):
        prepared.prepare(engine, text, NS(identity='q8'), page_tokens=128, device='cpu')
    assert prepared.before is None and prepared.workspaces == {}


@pytest.mark.parametrize('containment_fails', [False, True])
def test_late_binding_retirement_failure_poisoning_preserves_primary(monkeypatch, containment_fails):
    prepared = owner()
    prepared.prepared, prepared.outside_active_bytes = True, 0
    prepared.dense_owner = NS(bind_weights=lambda weights: {'binding_memory': {
        'committed_delta_bytes': 12, 'outside_active_delta_bytes': 12}})
    prepared._phase = lambda engine, name, operation, **kwargs: operation()
    primary, secondary = RuntimeError('retirement failed'), RuntimeError('containment failed')
    def retire(*args, **kwargs):
        raise primary
    calls = []
    def fatal(engine, error):
        calls.append(error)
        if containment_fails:
            raise secondary
    monkeypatch.setattr(execution_identity, '_retire_store', retire)
    monkeypatch.setattr(execution_identity, '_fatal', fatal)
    with pytest.raises(RuntimeError) as caught:
        prepared.bind_dense_weights(NS(), NS())
    assert caught.value is primary and calls == [primary]
    if containment_fails:
        cause = primary.__cause__
        assert cause is secondary or isinstance(cause, BaseExceptionGroup) and secondary in cause.exceptions


def test_exact_graph_subset_is_explicit_and_preserves_default_preparation():
    from test_exact_graph_policy import policy
    assert preparation.parse_preparation(spec(exact_graph_policy=None)) == preparation.parse_preparation(spec())
    selected = preparation.parse_preparation(spec(exact_graph_policy=policy()))
    assert selected['exact_graph_policy']['keys'] == [['mtp', 1, 1], ['verify', 5, 1]]
    with pytest.raises(ValueError, match='trial'):
        preparation.parse_preparation({**spec(exact_graph_policy=policy()), 'trial': False})


def test_unavailable_subset_fails_coordinated_contract_before_backend_allocation(monkeypatch):
    from tensorfold.cuda.sm70 import foundation_headers
    from test_exact_graph_policy import policy
    prepared = owner(exact_graph_policy=policy())
    monkeypatch.setattr(foundation_headers, 'selected_identity', lambda: pytest.fail('unsupported subset reached backend'))
    monkeypatch.setattr(execution_identity, '_retire_store', lambda *a, **k: None)
    prepared._phase = lambda engine, name, operation, **kwargs: operation()
    engine = NS(comm=None, tp=8, _exact_graph_subset_enabled=False)
    text = dict(num_attention_heads=24, num_key_value_heads=2, hidden_size=2560, head_dim=256)
    with pytest.raises(ValueError, match='private exact graph'):
        prepared.prepare(engine, text, NS(identity='q8'), page_tokens=128, device='cpu')
    assert prepared.before is None and prepared.workspaces == {}
