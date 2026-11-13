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


def test_native_hc_writeback_capture_is_opt_in_with_unchanged_false_identity():
    assert preparation.parse_preparation(spec(native_hc_writeback_capture=False)) == preparation.parse_preparation(spec())
    assert preparation.parse_preparation(spec(native_hc_writeback_capture=True))['native_hc_writeback_capture'] is True
    for value in (0, 1, 'true', None):
        with pytest.raises(ValueError, match='boolean'):
            preparation.parse_preparation(spec(native_hc_writeback_capture=value))
    with pytest.raises(ValueError, match='trial'):
        preparation.parse_preparation({**spec(native_hc_writeback_capture=True), 'trial': False})


def test_unavailable_native_hc_capture_stops_before_backend_or_allocation(monkeypatch):
    from tensorfold.cuda.sm70 import foundation_headers
    prepared = owner(native_hc_writeback_capture=True)
    monkeypatch.setattr(foundation_headers, 'selected_identity', lambda: pytest.fail('unsupported HC capture reached backend'))
    monkeypatch.setattr(execution_identity, '_retire_store', lambda *a, **k: None)
    prepared._phase = lambda engine, name, operation, **kwargs: operation()
    engine = NS(comm=None, tp=8, _native_hc_writeback_capture_enabled=False)
    text = dict(num_attention_heads=24, num_key_value_heads=2, hidden_size=2560, head_dim=256)
    with pytest.raises(ValueError, match='private exact graph'):
        prepared.prepare(engine, text, NS(identity='q8'), page_tokens=128, device='cpu')
    assert prepared.before is None and prepared.workspaces == {}


def test_native_target_eager_hc_is_default_off_and_strict_trial():
    name='native_hc_writeback_target_eager'
    assert preparation.parse_preparation(spec(**{name:False}))==preparation.parse_preparation(spec())
    assert preparation.parse_preparation(spec(**{name:True}))[name]is True
    both=preparation.parse_preparation(spec(native_hc_writeback_capture=True,**{name:True}))
    assert both[name]and both['native_hc_writeback_capture']
    for value in (0,1,'true',None):
        with pytest.raises(ValueError,match='explicit boolean'):
            preparation.parse_preparation(spec(**{name:value}))
    with pytest.raises(ValueError,match='trial'):
        preparation.parse_preparation({**spec(**{name:True}),'trial':False})


def test_unavailable_target_eager_hc_fails_before_backend_or_state_allocation(monkeypatch):
    from tensorfold.cuda.sm70 import foundation_headers
    selected=owner(native_hc_writeback_target_eager=True)
    monkeypatch.setattr(foundation_headers,'selected_identity',lambda:pytest.fail('unavailable trial reached backend'))
    monkeypatch.setattr(execution_identity,'_retire_store',lambda *a,**k:None)
    selected._phase=lambda engine,name,operation,**kwargs:operation()
    engine=NS(comm=None,tp=8,_native_hc_writeback_target_eager_enabled=False)
    text=dict(num_attention_heads=24,num_key_value_heads=2,hidden_size=2560,head_dim=256)
    with pytest.raises(ValueError,match='graph-enabled region'):
        selected.prepare(engine,text,NS(identity='q8'),page_tokens=128,device='cpu')
    assert selected.before is None and selected.workspaces=={}


def hc_spec():
    return {'backend':'bitblas-hc','manifest':'/sealed/hc/manifest.json','manifest_sha256':'b'*64,
            'roles':['decode','prefill']}


def test_hc_backend_is_explicit_and_keeps_other_numerics_and_flags():
    selected=owner(dense_projections=hc_spec(),native_gdn_capture=True,native_hc_writeback_capture=True)
    identity=selected.numerical_selection['dense_projections']
    assert identity['scope']=='target-HC-only' and identity['other_linears']=='existing-reference-provider'
    assert identity['MTP']=='existing-reference-provider'
    assert identity['sealed_libraries_sha256']=='b'*64
    assert selected.specification['native_gdn_capture'] and selected.specification['native_hc_writeback_capture']
    assert selected.identity()['specification']['dense_projections']==hc_spec()
    from tensorfold.engine.model_identity import canonical_runtime_json
    assert canonical_runtime_json(selected.identity())


@pytest.mark.parametrize('backend',['cublas','cublaslt','cutlass'])
def test_existing_dense_backends_keep_their_original_factory_and_identity(backend):
    from tensorfold.families.qwen4_exp.cuda import dense_projections
    value={**dense_spec(),'backend':backend}
    module,selection=preparation._dense_backend(value)
    assert module is dense_projections and selection.as_dict()==value
    assert owner(dense_projections=value).numerical_selection['dense_projections']==selection.numerical_identity


def test_hc_factory_uses_custom_manifest_schema_and_zero_additional_device_temporary():
    from tensorfold.families.qwen4_exp.cuda import bitblas_hc
    prepared=owner(dense_projections=hc_spec())
    backend,selection=preparation._dense_backend(hc_spec())
    assert backend is bitblas_hc
    prepared.dense_forecast=backend.forecast(selection,{'hidden_size':2560,'num_hidden_layers':48},prepared.role_rows)
    prepared.aten_temporary_extra_bytes=17
    assert prepared.additional_temporary_bytes==17
    prepared.dense_binding_memory={'committed_delta_bytes':0}
    assert prepared.additional_temporary_bytes==17
    for changes in ({'roles':['mtp']},{'geometry':'fixed-m32'},{'manifest_sha256':'wrong'}):
        with pytest.raises(ValueError):
            owner(dense_projections={**hc_spec(),**changes})
    with pytest.raises(ValueError,match='unqualified trial'):
        preparation.parse_preparation({**spec(dense_projections=hc_spec()),'trial':False})


def test_actual_prepare_phase_dispatches_hc_factory_and_journals_before_later_work(monkeypatch):
    import torch

    from tensorfold.cuda.sm70 import foundation_headers
    from tensorfold.families.qwen4_exp.cuda import bitblas_hc
    prepared=owner(dense_projections=hc_spec())
    calls=[];native=NS(record=lambda:{'HC-only':True})
    def create(selection,text,role_rows,device,*,world,register):
        calls.append((selection.as_dict(),dict(role_rows),world));register(native);return native
    monkeypatch.setattr(bitblas_hc.Owner,'prepare',create)
    monkeypatch.setattr(foundation_headers,'selected_identity',dict)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda *a:None)
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    monkeypatch.setattr(execution_identity,'_retire_store',lambda *a,**k:None)
    prepared._memory=lambda *a:{'physical_free_bytes':1000000000,'torch_allocated_bytes':0,'torch_reserved_bytes':0}
    def phase(engine,name,operation,**kwargs):
        result=operation()
        if name=='contract':return [result]*8
        if name=='dense-projections-prepare':raise ValueError('stop after journalled HC phase')
        pytest.fail('unrelated native workspace work reached')
    prepared._phase=phase
    engine=NS(comm=None,tp=8,position_profile=object(),method_geometry=NS(method='mtp'))
    text={'num_attention_heads':24,'num_key_value_heads':2,'hidden_size':2560,'head_dim':256,'num_hidden_layers':48}
    with pytest.raises(ValueError,match='journalled HC'):
        prepared.prepare(engine,text,NS(identity='q8'),page_tokens=128,device='cpu')
    assert calls==[(hc_spec(),prepared.role_rows,8)] and prepared.dense_owner is native
    assert prepared.workspaces=={}


@pytest.mark.parametrize('valid',[False,True])
def test_hc_binding_memory_proof_is_required_and_charged_once_before_states(monkeypatch,valid):
    prepared=owner(dense_projections=hc_spec())
    memory={'committed_delta_bytes':7 if valid else None,'outside_active_delta_bytes':7 if valid else None}
    prepared.prepared=True;prepared.outside_active_bytes=11
    prepared.dense_owner=NS(bind_weights=lambda weights:{'binding_memory':memory})
    prepared._phase=lambda engine,name,operation,**kwargs:operation()
    monkeypatch.setattr(execution_identity,'_retire_store',lambda *a,**k:None)
    if valid:
        prepared.bind_dense_weights(NS(),NS())
        assert prepared.outside_active_bytes==18 and prepared.dense_binding_memory==memory
        with pytest.raises(RuntimeError,match='one live prepared'):
            prepared.bind_dense_weights(NS(),NS())
    else:
        with pytest.raises(ValueError,match='unproven'):
            prepared.bind_dense_weights(NS(),NS())
        assert prepared.outside_active_bytes==11 and prepared.dense_binding_memory is None


@pytest.mark.parametrize('position,method',[(None,'none'),(None,'mtp'),(object(),'dspark'),(object(),'dflash2')])
def test_hc_numerics_share_required_bound_none_mtp_guard_before_backend(monkeypatch,position,method):
    from tensorfold.cuda.sm70 import foundation_headers
    from tensorfold.families.qwen4_exp.cuda import bitblas_hc
    prepared=owner(dense_projections=hc_spec())
    monkeypatch.setattr(bitblas_hc,'forecast',lambda *a,**k:{})
    monkeypatch.setattr(foundation_headers,'selected_identity',lambda:pytest.fail('HC unsupported region reached backend'))
    monkeypatch.setattr(execution_identity,'_retire_store',lambda *a,**k:None)
    prepared._phase=lambda engine,name,operation,**kwargs:operation()
    engine=NS(comm=None,tp=8,position_profile=position,method_geometry=NS(method=method))
    text={'num_attention_heads':24,'num_key_value_heads':2,'hidden_size':2560,'head_dim':256}
    with pytest.raises(ValueError,match='bound NONE/MTP'):
        prepared.prepare(engine,text,NS(identity='q8'),page_tokens=128,device='cpu')
    assert prepared.before is None and prepared.dense_owner is None and prepared.workspaces=={}


def test_hc_reuses_dense_owner_adoption_handoff_and_close_journal(monkeypatch):
    from tensorfold.cuda.sm70 import foundation_headers
    prepared=owner(dense_projections=hc_spec())
    prepared.prepared=True
    calls=[]
    for role in ('decode','prefill'):
        prepared.workspaces[role]=NS(native_kernel_family=preparation.KERNEL_FAMILY,native_heads_per_block=None,
            bind_native_states=lambda states,role=role:calls.append(('states',role)),
            close=lambda role=role:calls.append(('workspace-close',role)))
    prepared.dense_owner=NS(bound_weights=True,adopt=lambda w,d:calls.append(('HC-adopt',w,d)),
        handoff_to_current_thread=lambda:calls.append('HC-handoff'),close=lambda:calls.append('HC-close'))
    monkeypatch.setattr(foundation_headers,'selected_identity',lambda:None)
    weights=NS(meta={'owner_workspaces':prepared.workspaces})
    decoder=NS(buf=NS(history_owner=prepared.workspaces['decode'],rows=4),mbuf=None,
        pbuf=NS(history_owner=prepared.workspaces['prefill'],rows=8),states=[],
        controller=NS(configure_execution_families=lambda **kwargs:calls.append(('families',kwargs))),
        apply_execution_family=lambda:calls.append('apply-family'))
    prepared.adopt(weights,decoder)
    assert calls[-1]==('HC-adopt',weights,decoder)
    assert decoder.buf.owner_graphs is None and decoder.pbuf.owner_graphs is None
    prepared.handoff_to_current_thread()
    assert calls[-1]=='HC-handoff'
    prepared.close();prepared.close()
    assert calls.count('HC-close')==1 and prepared.closed


def test_hc_cannot_attach_before_the_shared_prestate_binding_proof(monkeypatch):
    from tensorfold.cuda.sm70 import foundation_headers
    prepared=owner(dense_projections=hc_spec());prepared.prepared=True
    prepared.dense_owner=NS(bound_weights=False,adopt=lambda *args:pytest.fail('unbound HC reached serving'))
    for role in ('decode','prefill'):
        prepared.workspaces[role]=NS(native_kernel_family=preparation.KERNEL_FAMILY,native_heads_per_block=None,
            bind_native_states=lambda states:None)
    weights=NS(meta={'owner_workspaces':prepared.workspaces})
    decoder=NS(buf=NS(history_owner=prepared.workspaces['decode'],rows=4),mbuf=None,
        pbuf=NS(history_owner=prepared.workspaces['prefill'],rows=8),states=[],
        controller=NS(configure_execution_families=lambda **kwargs:None),apply_execution_family=lambda:None)
    monkeypatch.setattr(foundation_headers,'selected_identity',lambda:None)
    with pytest.raises(RuntimeError,match='before State construction'):
        prepared.adopt(weights,decoder)


@pytest.mark.parametrize('backend', ['cublas', 'bitblas-hc'])
def test_local_hc_readout_cannot_silently_bypass_selected_dense_numerics(monkeypatch, backend):
    from tensorfold.cuda.sm70 import foundation_headers
    from tensorfold.families.qwen4_exp.cuda import dense_projections, bitblas_hc
    selection = (dense_spec() if backend == 'cublas' else
                 {'backend': backend, 'manifest': '/operator/bitblas-hc/manifest.json',
                  'manifest_sha256': 'a' * 64, 'roles': ['decode']})
    prepared = owner(dense_projections=selection)
    module = dense_projections if backend == 'cublas' else bitblas_hc
    monkeypatch.setattr(module, 'forecast', lambda *a, **k: {})
    monkeypatch.setattr(foundation_headers, 'selected_identity', lambda: pytest.fail('unsupported local readout reached backend'))
    monkeypatch.setattr(execution_identity, '_retire_store', lambda *a, **k: None)
    prepared._phase = lambda engine, name, operation, **kwargs: operation()
    engine = NS(comm=None, tp=8, position_profile=object(), method_geometry=NS(method='none'),
                _local_hc_readout_requested=True)
    text = dict(num_attention_heads=96, num_key_value_heads=2, hidden_size=2560, head_dim=256)
    with pytest.raises(ValueError, match='local HCReadoutGraphs'):
        prepared.prepare(engine, text, NS(identity='q8'), page_tokens=128, device='cpu')
    assert prepared.before is None and prepared.dense_owner is None


def test_value8_capture_is_explicit_and_requires_native_frame_policy():
    assert preparation.parse_preparation(spec(gdn_value8_capture=False)) == preparation.parse_preparation(spec())
    enabled = preparation.parse_preparation(spec(native_gdn_capture=True, gdn_value8_capture=True))
    assert enabled['gdn_value8_capture'] is True
    for value in (0, 1, 'true', None):
        with pytest.raises(ValueError, match='Value8'):
            preparation.parse_preparation(spec(native_gdn_capture=True, gdn_value8_capture=value))
    with pytest.raises(ValueError, match='Value8'):
        preparation.parse_preparation(spec(gdn_value8_capture=True))
    with pytest.raises(ValueError):
        preparation.parse_preparation({**spec(native_gdn_capture=True, gdn_value8_capture=True), 'trial': False})


def test_unavailable_value8_capture_refuses_before_backend_or_allocation(monkeypatch):
    from tensorfold.cuda.sm70 import foundation_headers
    prepared = owner(native_gdn_capture=True, gdn_value8_capture=True)
    monkeypatch.setattr(foundation_headers, 'selected_identity', lambda: pytest.fail('unsupported value8 reached backend'))
    monkeypatch.setattr(execution_identity, '_retire_store', lambda *a, **k: None)
    prepared._phase = lambda engine, name, operation, **kwargs: operation()
    engine = NS(comm=None, tp=8, _native_gdn_capture_enabled=True, _gdn_value8_capture_enabled=False)
    text = dict(num_attention_heads=96, num_key_value_heads=2, hidden_size=2560, head_dim=256)
    with pytest.raises(ValueError, match='Value8'):
        prepared.prepare(engine, text, NS(identity='q8'), page_tokens=128, device='cpu')
    assert prepared.before is None and prepared.workspaces == {}


@pytest.mark.parametrize('mask', range(8))
def test_native_family_q8_windows_and_prepared_panels_select_independently(mask):
    from tensorfold.cuda.sm70 import owned_family
    from tensorfold.families.qwen4_exp.cuda import padded_linears

    value = dict(kernel_family=preparation.RN_KERNEL_FAMILY, rn_heads_per_block=12,
                 graph_family=preparation.UNCAPTURED, graph_budget_bytes=0, trial=True)
    expected = set()
    if mask & 1:
        value['native_family'] = owned_family.POLICY
        expected.add('native_family')
    if mask & 2:
        value['owned_q8_window_prepare'] = True
        expected.add('owned_q8_window_preparation')
    if mask & 4:
        value['prepared_linears'] = padded_linears.OWNED_POLICY
        expected.add('prepared_linears')
    selected = preparation.OwnerPreparation(value,
        role_rows={'decode': 120, 'mtp': 120, 'prefill': 256}, manifest_sha256='a' * 64)
    assert set(selected.numerical_selection or {}) == expected
    assert preparation.parse_preparation(selected.specification) == selected.specification


def composed_spec(backend='cublas', roles=None):
    from tensorfold.families.qwen4_exp.cuda.padded_linears import OWNED_POLICY
    return {**spec(), 'kernel_family': preparation.RN_KERNEL_FAMILY, 'rn_heads_per_block': 12,
        'prepared_linears': OWNED_POLICY,
        'dense_projections': {'backend': backend, 'geometry': 'fixed-m32',
                              'roles': ['prefill'] if roles is None else roles, 'panel_rows': 32}}


@pytest.mark.parametrize('backend', ['cublas', 'cublaslt', 'cutlass'])
def test_only_nonoverlap_vendor_prefill_can_compose_with_prepared_m32(backend):
    selected = preparation.OwnerPreparation(composed_spec(backend),
        role_rows={'decode': 120, 'mtp': 120, 'prefill': 256}, manifest_sha256='a' * 64)
    numeric = selected.numerical_selection
    assert numeric['dense_projections']['roles'] == ['prefill']
    assert numeric['prepared_linears']['roles'] == ['decode', 'mtp']
    assert numeric['projection_role_composition']['reference_bit_identity'] is False
    for roles in (['decode'], ['decode', 'prefill']):
        with pytest.raises(ValueError, match='competing dense'):
            preparation.parse_preparation(composed_spec(backend, roles))
    value = composed_spec()
    value['dense_projections'] = {**hc_spec(), 'roles': ['prefill']}
    with pytest.raises(ValueError, match='competing dense'):
        preparation.parse_preparation(value)


@pytest.mark.parametrize('second_unproven', [False, True, 'gap'])
def test_composed_owners_bind_serially_and_commit_each_opaque_delta_once(monkeypatch, second_unproven):
    selected = preparation.OwnerPreparation(composed_spec(),
        role_rows={'decode': 120, 'mtp': 120, 'prefill': 256}, manifest_sha256='a' * 64)
    selected.prepared, selected.outside_active_bytes = True, 7
    selected.before = {'torch_allocated_bytes': 0, 'torch_reserved_bytes': 0, 'physical_free_bytes': 1000}
    selected.after = {'torch_allocated_bytes': 0, 'torch_reserved_bytes': 0, 'physical_free_bytes': 993}
    calls = []
    def bind(name, committed, outside, scratch):
        def run(weights):
            calls.append(name)
            before = ({'torch_allocated_bytes': 0, 'torch_reserved_bytes': 0, 'physical_free_bytes': 993}
                      if name == 'prefill' else {'allocated': 10, 'reserved': 10,
                                                'free': 980 if second_unproven == 'gap' else 981})
            after = ({'torch_allocated_bytes': 10, 'torch_reserved_bytes': 10, 'physical_free_bytes': 981}
                     if name == 'prefill' else {'allocated': 27, 'reserved': 27,
                                               'free': 960 if second_unproven == 'gap' else 961})
            return {'binding_memory': {'committed_delta_bytes': committed,
                'outside_active_delta_bytes': outside, 'cold_temporary_bytes': scratch,
                'before': before, 'after': after}}
        return NS(bind_weights=run)
    selected.dense_owner = bind('prefill', 12, 2, 100)
    selected.prepared_m32 = bind('decode+mtp', None if second_unproven is True else 20, 3, 200)
    selected._phase = lambda engine, name, operation, **kwargs: operation()
    monkeypatch.setattr(execution_identity, '_retire_store', lambda *a, **k: None)
    if second_unproven:
        with pytest.raises(ValueError, match='unproven'):
            selected.bind_dense_weights(NS(), NS())
        assert selected.dense_binding_memory is None and selected.committed_bytes() is None
        assert selected.outside_active_bytes == 7
    else:
        selected.bind_dense_weights(NS(), NS())
        assert selected.dense_binding_memory['committed_delta_bytes'] == 32
        assert selected.dense_binding_memory['outside_active_delta_bytes'] == 5
        assert selected.dense_binding_memory['cold_temporary_bytes'] == 200
        assert selected.outside_active_bytes == 12 and selected.committed_bytes() == 39
        assert set(selected.dense_binding_memory['owners']) == {'dense_projections', 'prepared_linears'}
    assert calls == ['prefill', 'decode+mtp']
    with pytest.raises(RuntimeError, match='one live prepared'):
        selected.bind_dense_weights(NS(), NS())


def test_coordinate_host_stage_requires_exact_private_policy_and_binds_identity():
    from tensorfold.families.qwen4_exp.cuda.compact_indexer import HOST_COORDINATE_POLICY
    chosen = owner(coordinate_staging=HOST_COORDINATE_POLICY)
    assert chosen.numerical_selection['coordinate_staging']['policy'] == HOST_COORDINATE_POLICY
    assert 'coordinate_staging' not in preparation.parse_preparation(spec())
    for value in [True, 'unknown']:
        with pytest.raises(ValueError,match='coordinate staging'):
            owner(coordinate_staging=value)
    with pytest.raises(ValueError,match='coordinate staging'):
        preparation.parse_preparation({**spec(coordinate_staging=HOST_COORDINATE_POLICY),'trial':False})
