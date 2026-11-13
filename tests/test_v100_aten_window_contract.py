"""Finite ATen C8 DATA/identity controls; no model, native or quality claim."""
import copy
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tests.test_v100_target_window_contract import current_inputs
from tools import v100_owned_model_contract as model
from tools import v100_target_window_teacher as window


def inputs(layout=None, *, reference=False):
    operation, plan, client, profile = current_inputs()
    preparation = operation['prepared_execution']
    if not reference:
        preparation.pop('rn_heads_per_block')
        preparation['kernel_family'] = window.ATEN_FAMILY
        preparation['aten_tiles'] = {'panel_rows':16,'tensor_budget_bytes':1 << 30}
        if layout is not None:
            preparation['aten_tiles']['query_layout'] = layout
    for value in (operation, client):
        value['preparation_contract'] = window.ATEN_PREPARATION_CONTRACT
        value['teacher_scope'] = window.ATEN_SCOPE
    client['prepared_execution'] = copy.deepcopy(preparation)
    client['expected_service'] = window.expected_service(client)
    profile['preparation'] = copy.deepcopy(preparation)
    return operation, plan, client, profile


@pytest.mark.parametrize('layout', [None, *window.ATEN_LAYOUTS])
def test_actual_owner_canonical_attention_numerics_and_service_are_explicit(layout):
    operation, plan, client, profile = inputs(layout)
    window.validate_operation(operation)
    window.validate_teacher_client(client)
    window.validate_teacher_operation_plan(operation,plan,client,profile=profile)
    assert model.teacher_operation_contract(operation) is window
    assert client['expected_service']['kernel_family'] == window.ATEN_FAMILY
    actual = window._preparation_owner(operation)
    assert actual.specification == operation['prepared_execution']
    attention = actual.numerical_selection['attention']
    assert attention['family'] == window.ATEN_FAMILY
    assert ('query_layout' in attention) == (layout is not None)
    assert attention.get('query_layout') == layout
    window._validate_preparation_numerics({'owner_preparation_numerics':actual.numerical_selection},operation)
    with pytest.raises(ValueError,match='numerical registration'):
        window._validate_preparation_numerics({'owner_preparation_numerics':{}},operation)
    assert operation['numerical_trial'] == dict.fromkeys(window.numerical.AXES,False)


def test_rn_reference_uses_new_scope_but_keeps_actual_original_family_and_identity():
    operation,plan,client,profile = inputs(reference=True)
    window.validate_teacher_operation_plan(operation,plan,client,profile=profile)
    assert client['expected_service'] == window.SERVICE
    assert operation['teacher_scope'] != window.SCOPE
    assert window._expected_preparation_numerics(operation) is None
    window._validate_preparation_numerics({},operation)


@pytest.mark.parametrize('axis', list(window.numerical.AXES))
def test_attention_contract_cannot_relabel_three_rn_axes_as_family_booleans(axis):
    operation,*_ = inputs()
    operation['numerical_trial'][axis] = True
    with pytest.raises(ValueError,match='axes off'):
        window.validate_operation(operation)


@pytest.mark.parametrize('field,value', [('rn_heads_per_block',12),('native_family','owned-native-family-v1'),
    ('prepared_linears','owned-prepared-m32-v1'),('flash_tiles',{}),('dense_projections',{'backend':'bitblas-hc'}),
    ('unknown_external_bundle',True),('coordinate_staging','owned-host-int64-coordinates-v1')])
def test_attention_contract_refuses_unrelated_or_external_preparation(field,value):
    operation,*_ = inputs()
    operation['prepared_execution'][field] = value
    with pytest.raises(ValueError):
        window.validate_operation(operation)


@pytest.mark.parametrize('tiles', [{'panel_rows':True,'tensor_budget_bytes':1 << 30},
    {'panel_rows':32,'tensor_budget_bytes':1 << 30}, {'panel_rows':16,'tensor_budget_bytes':True},
    {'panel_rows':16,'tensor_budget_bytes':1 << 30,'query_layout':None},
    {'panel_rows':16,'tensor_budget_bytes':1 << 30,'query_layout':'unknown'}])
def test_attention_profile_has_one_paid_finite_panel_and_layout_domain(tiles):
    operation,*_ = inputs()
    operation['prepared_execution']['aten_tiles'] = tiles
    with pytest.raises(ValueError):
        window.validate_operation(operation)


def test_scope_service_and_canonical_manifest_are_not_normalized_into_rn12():
    operation,plan,client,profile = inputs(window.ATEN_LAYOUTS[1])
    for contract in (None,window.PREPARATION_CONTRACT):
        with pytest.raises(ValueError):
            window.validate_preparation(operation['prepared_execution'],operation['numerical_trial'],contract)
    for key in ('row_buckets','collective_family'):
        bad = copy.deepcopy(operation)
        bad['prepared_execution'].pop(key)
        with pytest.raises(ValueError):
            window.validate_operation(bad)
    bad = copy.deepcopy(operation)
    bad['teacher_scope'] = window.SCOPE
    with pytest.raises(ValueError):
        window.validate_operation(bad)
    client['expected_service'] = dict(window.SERVICE)
    with pytest.raises(ValueError):
        window.validate_teacher_client(client)
    client['expected_service'] = window.expected_service(client)
    client['prepared_execution']['aten_tiles'].pop('query_layout')
    with pytest.raises(ValueError):
        window.validate_teacher_operation_plan(operation,plan,client,profile=profile)


def test_finite_expert21_and_raw_corpus_window_grants_remain_unchanged():
    operation,plan,client,profile = inputs()
    assert len(window.selectors()) == 21
    for selector in window.selectors():
        operation['selector'] = client['selector'] = selector
        window.validate_teacher_operation_plan(operation,plan,client,profile=profile)
    assert plan['vocab'] == 248320 and plan['target_rows'] == 120 and plan['prefill_rows'] == 256
    assert plan['resident_states'] == 24 and plan['context'] == 262144
    changed = copy.deepcopy(plan)
    changed['identity']['profile_sha256'] = 'f'*64
    with pytest.raises(ValueError,match='profile'):
        window.validate_teacher_operation_plan(operation,changed,client,profile=profile)


@pytest.mark.parametrize('start,end', [(None,'native-gqa-v1'),(None,'fixed12-head-as-query-lse-v1'),
                                    ('native-gqa-v1','fixed12-head-as-query-lse-v1')])
def test_only_actual_layout_field_is_variable_within_aten(start,end):
    reference,*_ = inputs(start)
    candidate,*_ = inputs(end)
    changes = window.preparation_changes(reference,candidate)
    assert set(changes) == {'aten_tiles'} and changes['aten_tiles']['classification'] == 'numerical-layout'
    assert window._attention_preparation_projection(reference['prepared_execution'],changes) == \
           window._attention_preparation_projection(candidate['prepared_execution'],changes)
    candidate['prepared_execution']['aten_tiles']['tensor_budget_bytes'] += 1
    with pytest.raises(ValueError,match='unrelated'):
        window.preparation_changes(reference,candidate)


def test_family_projection_is_explicit_numeric_and_preserves_graph_authority():
    reference,*_ = inputs(reference=True)
    candidate,*_ = inputs('fixed12-head-as-query-lse-v1')
    changes = window.preparation_changes(reference,candidate)
    assert set(changes) == {'kernel_family','rn_heads_per_block','aten_tiles'}
    assert {entry['classification'] for entry in changes.values()} == {'numerical-family'}
    candidate['prepared_execution']['exact_graph_policy']['evidence_sha256'] = 'f'*64
    with pytest.raises(ValueError,match='unrelated'):
        window.preparation_changes(reference,candidate)


def profile(operation):
    """Controlled boundary DATA, not a genuine model/provider observation.

    Operator and tile metadata come from actual owning implementations. Library
    hashes are inert fixture identities; the separate VMA tests require them to
    match a closed observed backing before this DATA could qualify anything.
    """
    import torch

    from tensorfold.cuda.sm70 import attention_backends as backend
    from tensorfold.cuda.sm70.paged_attention import rn_preparation_identity
    from tensorfold.families.qwen4_exp.cuda.execution_identity import digest
    actual = window._preparation_owner(operation)
    operator = backend.operator_identity()
    operator['torch'] = str(operator['torch'])
    libraries = {'torch_C':{'path':str(Path(torch._C.__file__).resolve()),'bytes':15472,'sha256':'1'*64},
        'torch_cuda':{'path':str(Path(torch.__file__).resolve().parent/'lib/libtorch_cuda.so'),'bytes':227388872,'sha256':'2'*64}}
    contents = {'library/'+row['path']:{'size':row['bytes'],'sha256':row['sha256']}for row in libraries.values()}
    roles = {}
    aten = operation['prepared_execution']['kernel_family'] == window.ATEN_FAMILY
    if aten:
        for role,requirements in actual.aten_requirements.items():
            tile = object.__new__(backend.OwnerTiles)
            tile.requirements,tile.panel_rows,tile.closed = requirements,16,False
            tile.query_layout = operation['prepared_execution']['aten_tiles'].get('query_layout')
            tile.primitive = NS(identity=operator)
            roles[role] = tile.record()
    allocations = {}
    for role,rows in [('decode',120),('mtp',120),('prefill',256)]:
        paid = actual.aten_requirements[role]['persistent_bytes'] if aten else 0
        allocations[role] = [{'rank':rank,'rows':rows,'device_tensor_bytes':200+paid,'native_tensor_bytes':100+paid,
                             'communication_tensor_bytes':100,'unrelated':{'logical_context':262144}}for rank in range(8)]
    retained = {'specification':{k:v for k,v in operation['prepared_execution'].items()if k!='trial'},
        'roles':{'decode':120,'mtp':120,'prefill':256},'allocations':allocations,'unrelated_owner_field':'retain'}
    if aten:
        retained.update(aten_operator_identity=operator,aten_tensor_requirements=actual.aten_requirements)
    else:
        retained['rn_numerical_identity'] = rn_preparation_identity()
    raw = {'owner_preparation':retained,'unrelated_retention':{'context':262144}}
    return {'attention_provider':{'operator':operator,'libraries':libraries,'roles':roles},
        'numerical_trial_retention':raw,'teacher_execution_registration':{'contents':contents,
            'binding':{'identity':{'retention':digest(raw)}}}}


@pytest.mark.parametrize('reference_mode', ['rn','default','native-gqa-v1'])
def test_raw_retention_and_paid_rank_lists_preserved_for_declared_numeric_changes(reference_mode):
    reference,*_ = inputs(None if reference_mode in ('rn','default') else reference_mode,reference=reference_mode=='rn')
    candidate,*_ = inputs('fixed12-head-as-query-lse-v1')
    changes = window.preparation_changes(reference,candidate)
    a,b = {'teacher_profile':profile(reference)},{'teacher_profile':profile(candidate)}
    before = copy.deepcopy(b)
    assert window._comparison_retention(a,reference,changes) == window._comparison_retention(b,candidate,changes)
    assert window._attention_provider_comparison(a['teacher_profile'],reference,changes) == \
           window._attention_provider_comparison(b['teacher_profile'],candidate,changes)
    assert b == before
    from tensorfold.families.qwen4_exp.cuda.execution_identity import digest
    raw = b['teacher_profile']['numerical_trial_retention']
    raw['owner_preparation']['allocations']['mtp'][3]['unrelated']['logical_context'] -= 1
    b['teacher_profile']['teacher_execution_registration']['binding']['identity']['retention'] = digest(raw)
    assert window._comparison_retention(a,reference,changes) != window._comparison_retention(b,candidate,changes)
    raw['owner_preparation']['allocations']['decode'] = {'rank':0}
    b['teacher_profile']['teacher_execution_registration']['binding']['identity']['retention'] = digest(raw)
    with pytest.raises(ValueError,match='eight-rank'):
        window._comparison_retention(b,candidate,changes)


@pytest.mark.parametrize('fault', ['wrong_layout','wrong_sdk','missing_library','unissued_library','wrong_shape'])
def test_sdk_role_and_source_registration_mismatches_fail_closed(fault):
    operation,*_ = inputs('fixed12-head-as-query-lse-v1')
    actual = profile(operation)
    if fault == 'wrong_layout':actual['attention_provider']['roles']['mtp'].pop('query_layout')
    elif fault == 'wrong_sdk':actual['attention_provider']['operator']['torch_git'] = '0'*40
    elif fault == 'missing_library':actual['attention_provider']['libraries'].pop('torch_cuda')
    elif fault == 'unissued_library':actual['teacher_execution_registration']['contents'].clear()
    else:actual['attention_provider']['roles']['decode']['requirements'] = {}
    with pytest.raises(ValueError):window._validate_attention_provider(actual,operation)


def test_actual_closed_torch_vma_linkage_is_required_not_a_provider_boolean():
    operation,*_ = inputs('fixed12-head-as-query-lse-v1')
    actual = profile(operation)
    libraries = actual['attention_provider']['libraries']
    native = {'required_provider_paths':[row['path']for row in libraries.values()],
        'missing_required_provider_paths':[], 'mapped_elfs':[{**row,'vmas':[{'start':1,'end':2}]}for row in libraries.values()]}
    window._validate_attention_vmas(actual,operation,native)
    for changed in [dict(native,required_provider_paths=[]),dict(native,mapped_elfs=[]),
                    dict(native,missing_required_provider_paths=['missing'])]:
        with pytest.raises(ValueError):window._validate_attention_vmas(actual,operation,changed)
    bad = copy.deepcopy(native)
    bad['mapped_elfs'][0]['sha256'] = 'f'*64
    with pytest.raises(ValueError):window._validate_attention_vmas(actual,operation,bad)


def test_unrelated_attention_role_metadata_is_rejected_before_family_projection():
    reference,*_ = inputs(reference=True)
    candidate,*_ = inputs('fixed12-head-as-query-lse-v1')
    changes = window.preparation_changes(reference,candidate)
    actual = profile(candidate)
    actual['attention_provider']['roles']['decode']['unrelated_retention_field'] = {'context':1}
    with pytest.raises(ValueError,match='paid ATen roles'):
        window._attention_provider_comparison(actual,candidate,changes)


def test_actual_qualifier_routes_new_scope_through_window_preflight(monkeypatch):
    from tools import v100_owned_qualification as qualifier
    operation,*_ = inputs()
    events = []
    monkeypatch.setattr(window,'preflight',lambda *a,**k:events.append(k['operation']) or 'window-owned')
    assert qualifier._teacher_preflight(NS(),{},NS(),prepared_operation=operation,selected=NS()) == 'window-owned'
    assert events == [operation]


def test_runtime_provider_derives_real_sdk_paths_and_typed_retained_panels():
    import torch

    from tensorfold.cuda.sm70 import attention_backends as backend

    operation,*_ = inputs('fixed12-head-as-query-lse-v1')
    actual_profile = profile(operation)
    actual = window._preparation_owner(operation)
    workspaces = {}
    for role,requirements in actual.aten_requirements.items():
        tile = object.__new__(backend.OwnerTiles)
        tile.requirements,tile.panel_rows,tile.closed = requirements,16,False
        tile.query_layout = operation['prepared_execution']['aten_tiles']['query_layout']
        tile.primitive = NS(identity=actual_profile['attention_provider']['operator'])
        tile.panels = (None,*[backend.Prepared(backend.TileSpec(count,1,2051,12,1,.0625,
            masked=True,query_layout=tile.query_layout),torch.device('cpu'),1 << 30,tile.primitive.identity)
            for count in range(1,17)])
        workspaces[role] = NS(native_kernel_family=window.ATEN_FAMILY,aten_tiles=tile)
    actual.workspaces = workspaces
    engine = NS(_owner_preparation=actual)
    result = window._runtime_attention_provider(engine,operation,
        actual_profile['teacher_execution_registration']['contents'])
    assert result == actual_profile['attention_provider']
    assert engine._teacher_attention_provider_paths == tuple(row['path']for row in result['libraries'].values())
    workspaces['mtp'].aten_tiles.query_layout = None
    with pytest.raises(ValueError,match='layout capability'):
        window._runtime_attention_provider(engine,operation,actual_profile['teacher_execution_registration']['contents'])


def test_paid_allocation_charge_is_checked_without_normalizing_raw_records():
    from tensorfold.families.qwen4_exp.cuda.execution_identity import digest
    operation,*_ = inputs()
    value = profile(operation)
    raw = value['numerical_trial_retention']
    raw['owner_preparation']['allocations']['decode'][0]['device_tensor_bytes'] = 0
    value['teacher_execution_registration']['binding']['identity']['retention'] = digest(raw)
    with pytest.raises(ValueError,match='paid ATen attention payload'):
        window._comparison_retention({'teacher_profile':value},operation,{})
