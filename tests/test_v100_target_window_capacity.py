"""Source37-shaped cold capacity contracts; these fixtures are not CUDA evidence."""
import copy
from types import SimpleNamespace as NS

import pytest

from tensorfold.families.qwen4_exp.cuda.execution_identity import digest
from tensorfold.families.qwen4_exp.kv_formats import get_pair
from tools import v100_target_window_teacher as window

GIB = 1 << 30


def proof(bits=8):
    slots = 24 if bits == 8 else 32
    physical = 262149
    retention = {'keep': 8, 'recovery_points_per_piece': 2}
    snapshot = 64 * (8 + slots * 2)
    future = {'growth_bytes': 0, 'copy_overlap_bytes': 128, 'snapshot_bytes_each': 64,
              'snapshot_bytes': snapshot, 'transient_bytes': 4096, 'workspace_bytes': 2*GIB,
              'future_bytes': 2*GIB + snapshot + 128}
    ranks, windows = [], []
    for rank in range(8):
        free = (30 if rank < 7 else 29)*GIB
        ranks.append({'rank': rank, 'budget_bytes': free, 'resident_weight_bytes': 12*GIB,
            'loading_overlap_bytes': GIB, 'session_execution_bytes': 12*GIB,
            'resource_ledger_bytes': {'serving_workspace': 2*GIB},
            'allocation_baseline': {'physical_free_bytes': free, 'device_total_bytes': 32*GIB,
                'preallocated_project_bytes': GIB, 'effective_device_floor_bytes': 0,
                'absolute_grant_cap_bytes': None, 'remaining_absolute_grant_bytes': None,
                'usable_serving_budget_bytes': free}})
        windows.append({'rank': rank, 'materialized': True, 'startup_capacities': [physical]*slots,
                        'allocated_cache_bytes': 5*GIB, 'torch_allocated_bytes': 20*GIB,
                        'torch_reserved_bytes': 22*GIB})
    keys = {(role,count*rows,count) for count in (1,2,4,8,16,24)
            for rows in (range(1,6) if count == 1 else (1,5)) for role in ('verify','mtp')}
    keys.update({('verify',28,8), ('mtp',28,8), *(('mtp',n,n) for n in (3,5,6,7))})
    if slots == 32: keys.update((role,rows,32) for role in ('verify','mtp') for rows in (32,160))
    keys = [list(key) for key in sorted(keys)]
    admission = {'policy': window.CAPACITY_CONTRACT, 'agreed_graph_bytes': 5*GIB,
        'admitted_project_budget_bytes': 29*GIB, 'live_reserve_bytes': 0,
        'full_slots': slots, 'capacity': physical, 'future_service': future,
        'private_spare_credit_bytes': 0, 'header_room_bytes': GIB,
        'graph_outside_allocator_bytes': 0,
        'observations': [{'rank': rank, 'graph_room_bytes': (6 if rank < 7 else 5)*GIB,
            'live_after_floor_bytes': (6 if rank < 7 else 5)*GIB+future['future_bytes'],
            'future_service_bytes': future['future_bytes']} for rank in range(8)]}
    measured = []
    for index, key in enumerate(keys):
        measured.append({'bytes': 1 << 20, 'pieces': 15 if key[0] == 'verify' else 2,
            'accounting': {'schema': 'tensorfold.exact-graph-key-v1', 'rank': 0,
                'role': key[0], 'rows': key[1], 'participants': key[2], 'admitted': True,
                'prior_keys': index, 'prior_bytes': index*(1 << 20), 'physical_delta_bytes': 1 << 20,
                'budget_bytes': 5*GIB, 'pieces': 15 if key[0] == 'verify' else 2}})
    graph = {'automatic': True, 'captured': True, 'reason': None, 'stopped': None, 'rejected': None,
        'planned_bytes': 5*GIB, 'budget_bytes': 5*GIB, 'future_service': future,
        'observed_graph_admission': copy.deepcopy(admission),
        'exact_graph_catalogue': {'policy': 'observed-demand-v1', 'evidence_sha256': 'd'*64,
            'requested_keys': keys, 'captured_keys': copy.deepcopy(keys), 'declined_keys': []},
        'bytes': len(keys)*(1 << 20), 'settled_bytes': len(keys)*(1 << 20), 'room_bytes': len(keys)*(1 << 20), 'keys': measured,
        'pieces': sum(row['pieces'] for row in measured), 'verify_keys': sum(key[0]=='verify' for key in keys),
        'mtp_keys': sum(key[0]=='mtp' for key in keys), 'capture_during_serving': False,
        'allocator': {'backend': 'native', 'accounting': 'native-pool-inactive',
                      'private_pool_spare_bytes': 128 << 20}}
    final_admission = {**copy.deepcopy(admission), 'observed_serving_base_bytes': 24*GIB,
                       'original_header': {'budget_bytes': 25*GIB}, 'basis': 'source-owned refinement'}
    cap = {'native_window': 262144, 'requested_context': 262144, 'context_window': 262144,
        'cache_slots': physical, 'budget_bytes': 29*GIB, 'startup_peak_bytes_estimate': 13*GIB,
        'serving_peak_bytes_estimate': 29*GIB, 'total_bytes_estimate': 29*GIB,
        'full_window_headroom': {'policy': window.CAPACITY_CONTRACT, 'required_sessions': slots,
                                'logical_context': 262144, 'unused_device_floor_bytes': 0},
        'session_capacity': {'context_tokens': 262144, 'configured_ceiling': slots,
            'resident_capacity': slots, 'full_context_commitments': slots, 'ranks': ranks},
        'kv_full_windows': {'requested': slots, 'reserved': slots, 'stream_slots': slots,
            'context_window': 262144, 'cache_slots_each': physical, 'workspace_bytes': 2*GIB,
            'materialized': True, 'ranks': windows},
        'execution_envelope': {'method': 'mtp', 'max_proposals': 4, 'target_rows': slots*5, 'prefill_tokens': 256},
        'observed_graph_admission': final_admission, 'multi_graphs': graph,
        'resource_ledger_bytes': {'serving_workspace': 2*GIB}}
    value = {'schema': window.CAPACITY_PROOF_SCHEMA, 'contract': window.CAPACITY_CONTRACT,
        'rank': 0, 'slots': slots, 'bits': bits, 'physical_capacity': physical, 'capacity': cap,
        'states': [{'slot': i, 'logical_context': 262144, 'capacity': physical, 'limit': physical,
                    'pos': 0, 'mtp_len': 0} for i in range(slots)],
        'owner': {'admitted': True, 'bound_closed': False, 'pending': 0,
            'failed_owner': False, 'capture_failure': False, 'preparation_closed': False, 'prepared': True},
        'sample': {'physical_free_bytes': 4*GIB, 'device_total_bytes': 32*GIB,
            'allocated_bytes': 24*GIB, 'reserved_bytes': 26*GIB, 'live_bytes': 5*GIB,
            'observed_at_unix_ns': 1, 'interval_monotonic_ns': [1, 2],
            'ownership': {'prepared_private_spare_bytes': 128 << 20,
                'prepared_outside_active_bytes': 32 << 20, 'mapped_bytes': 0,
                'absolute_cap_bytes': None, 'graph_outside_allocator_bytes': 0}},
        'retention_sha256': digest(retention)}
    pair = get_pair('int8' if bits == 8 else 'int4')
    value['kv_pair'] = {'key_dtype':pair.key_dtype,'value_dtype':pair.value_dtype,'identity':pair.identity}
    return value, retention


@pytest.mark.parametrize('bits', [8, 4])
def test_complete_full_native_regions_and_original_signed_settlement(bits):
    value, retention = proof(bits)
    assert window.validate_capacity_proof(value, retention=retention, slots=value['slots'], bits=bits) is value
    value['capacity']['multi_graphs']['settled_bytes'] = -1
    assert window.validate_capacity_proof(value, retention=retention, slots=value['slots'], bits=bits) is value


@pytest.mark.parametrize('fault', [
    'policy', 'floor', 'logical', 'ceiling', 'one_rank', 'duplicate_rank', 'tail', 'partial_window',
    'missing_state', 'state_tail', 'state_dirty', 'workspace', 'future', 'snapshot', 'minimum',
    'rank_budget', 'absolute', 'preallocated', 'private', 'retired', 'pending', 'unadmitted',
    'catalogue', 'key_accounting', 'key_rank', 'pool_overbudget', 'sample_live', 'private_credit',
    'sample_device', 'bool_integer', 'bool_owner',
])
def test_unproven_or_borrowed_full_capacity_is_never_accepted(fault):
    value, retention = proof()
    cap = value['capacity']; ranks = cap['session_capacity']['ranks']; graph = cap['multi_graphs']
    if fault == 'policy': cap['full_window_headroom']['policy'] = 'arbitrary-zero-floor'
    elif fault == 'floor': ranks[3]['allocation_baseline']['effective_device_floor_bytes'] = 4*GIB
    elif fault == 'logical': cap['native_window'] -= 1
    elif fault == 'ceiling': cap['session_capacity']['configured_ceiling'] = 32
    elif fault == 'one_rank': ranks.pop()
    elif fault == 'duplicate_rank': ranks[7]['rank'] = 0
    elif fault == 'tail': cap['cache_slots'] = value['physical_capacity'] = 262144
    elif fault == 'partial_window': cap['kv_full_windows']['ranks'][7]['startup_capacities'][23] = 262144
    elif fault == 'missing_state': value['states'].pop()
    elif fault == 'state_tail': value['states'][23]['limit'] = 262144
    elif fault == 'state_dirty': value['states'][23]['pos'] = 1
    elif fault == 'workspace': cap['resource_ledger_bytes']['serving_workspace'] = GIB
    elif fault == 'future': cap['observed_graph_admission']['future_service']['growth_bytes'] = 1
    elif fault == 'snapshot': retention['keep'] += 1; value['retention_sha256'] = digest(retention)
    elif fault == 'minimum': cap['observed_graph_admission']['agreed_graph_bytes'] = 6*GIB
    elif fault == 'rank_budget': ranks[7]['budget_bytes'] += 1
    elif fault == 'absolute': ranks[7]['allocation_baseline']['absolute_grant_cap_bytes'] = 30*GIB
    elif fault == 'preallocated': ranks[7]['allocation_baseline']['preallocated_project_bytes'] = -1
    elif fault == 'private': graph['allocator']['private_pool_spare_bytes'] = None
    elif fault == 'retired': value['owner']['bound_closed'] = True
    elif fault == 'pending': value['owner']['pending'] = 1
    elif fault == 'unadmitted': value['owner']['admitted'] = False
    elif fault == 'catalogue': graph['exact_graph_catalogue']['captured_keys'].pop()
    elif fault == 'key_accounting': graph['keys'][2]['accounting']['prior_bytes'] -= 1
    elif fault == 'key_rank': graph['keys'][2]['accounting']['rank'] = 7
    elif fault == 'pool_overbudget': graph['room_bytes'] = graph['settled_bytes'] = 6*GIB
    elif fault == 'sample_live': value['sample']['live_bytes'] = 0
    elif fault == 'private_credit': value['sample']['live_bytes'] = 6*GIB
    elif fault == 'sample_device': value['sample']['device_total_bytes'] = 40*GIB
    elif fault == 'bool_integer': cap['session_capacity']['resident_capacity'] = True
    elif fault == 'bool_owner': value['owner']['admitted'] = 1
    with pytest.raises(ValueError):
        window.validate_capacity_proof(value, retention=retention)


def test_32_q4_requires_all_32_physical_owners_and_e160_graphs():
    value, retention = proof(4)
    value['states'] = value['states'][:8]
    with pytest.raises(ValueError): window.validate_capacity_proof(value, retention=retention, slots=32, bits=4)
    value, retention = proof(4)
    graph = value['capacity']['multi_graphs']; catalogue = graph['exact_graph_catalogue']
    index = catalogue['requested_keys'].index(['verify',160,32])
    catalogue['requested_keys'].pop(index); catalogue['captured_keys'].pop(index); graph['keys'].pop(index)
    for position, row in enumerate(graph['keys']):
        row['accounting']['prior_keys'] = position; row['accounting']['prior_bytes'] = position*(1 << 20)
    graph['bytes'] = graph['settled_bytes'] = graph['room_bytes'] = len(graph['keys'])*(1 << 20)
    graph['pieces'] = sum(row['pieces'] for row in graph['keys']); graph['verify_keys'] -= 1
    with pytest.raises(ValueError, match='full E160'):
        window.validate_capacity_proof(value, retention=retention, slots=32, bits=4)


def test_explicit_contract_is_closed_and_bound_across_actual_client_operation_plan():
    from tests.test_v100_target_window_contract import inputs
    operation, plan, client, profile = inputs()
    # The old contract is deliberately still admissible on its historical runtime.
    window.validate_teacher_operation_plan(operation, plan, client, profile=profile)
    operation['capacity_contract'] = window.CAPACITY_CONTRACT
    with pytest.raises(ValueError, match='capacity client/operation'):
        window.validate_teacher_operation_plan(operation, plan, client, profile=profile)
    client['capacity_contract'] = window.CAPACITY_CONTRACT
    window.validate_teacher_operation_plan(operation, plan, client, profile=profile)
    for bad in (None, False, 'full-window-v2'):
        invalid = {**operation, 'capacity_contract': bad}
        with pytest.raises(ValueError): window.validate_operation(invalid)


def test_startup_capacity_is_bound_to_raw_retention_and_cannot_drop_catalogue():
    value, retention = proof(); cap = value['capacity']
    operation = {'capacity_contract': window.CAPACITY_CONTRACT, 'prepared_execution': {
        'exact_graph_policy': {'policy': 'observed-demand-v1', 'evidence_sha256': 'd'*64,
                              'keys': cap['multi_graphs']['exact_graph_catalogue']['requested_keys']}}}
    profile = {'rank': 0, 'full_window_capacity': value, 'numerical_trial_retention': retention,
        'teacher_execution_registration': {'binding': {'identity': {'retention': digest(retention), 'codec':digest({'pair':value['kv_pair']['identity']})}},
            'capacity_projection': {'schema': window.CAPACITY_PROOF_SCHEMA,
                                   'capacity_sha256': digest(value), 'retention_sha256': digest(retention)}}}
    window._validate_startup_capacity(profile, operation)
    changed = copy.deepcopy(profile); changed['full_window_capacity']['sample']['live_bytes'] -= 1
    with pytest.raises(ValueError, match='bound'): window._validate_startup_capacity(changed, operation)
    changed = copy.deepcopy(operation); changed['prepared_execution']['exact_graph_policy']['keys'].append(['verify',3,2])
    with pytest.raises(ValueError, match='dropped'): window._validate_startup_capacity(profile, changed)
    with pytest.raises(ValueError): window._validate_startup_capacity({}, operation)
    with pytest.raises(ValueError): window._validate_startup_capacity(profile, {})


@pytest.mark.parametrize('bits', [8,4])
def test_actual_cold_projection_uses_installed_runtime_boundaries_without_gpu(monkeypatch,bits):
    import torch

    from tensorfold.cuda import capacity, kv_arena
    value, retention = proof(bits); cap = value['capacity']; graph = cap['multi_graphs']
    states = [NS(**{k:v for k,v in state.items()if k != 'slot'}) for state in value['states']]
    graphs = NS(admitted=True, bound=NS(closed=False), pending=[], failure=None,
                room_bytes=graph['room_bytes'], capture_reserved_delta_bytes=graph['room_bytes'])
    engine = NS(rank=0, kv_pair=get_pair('int8' if bits == 8 else 'int4'), capacity_plan=cap, w=NS(device=torch.device('cpu')),
        multi=NS(states=states, capacity=262149, graphs=graphs, _failed_graphs=None, memory_gate=NS(live=lambda:5*GIB)),
        _owner_preparation=NS(closed=False, prepared=True, private_spare_bytes=128<<20, outside_active_bytes=32<<20))
    graphs.dec = engine.multi; graphs.bound.owner = graphs; graphs.bound.dec = engine.multi
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda *_:(4*GIB,32*GIB))
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda *_:24*GIB)
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda *_:26*GIB)
    monkeypatch.setattr(capacity, 'cuda_limit_bytes', lambda:None)
    monkeypatch.setattr(kv_arena, 'mapped_bytes', lambda:0)
    operation = {'capacity_contract':window.CAPACITY_CONTRACT, 'prepared_execution':{'exact_graph_policy':{
        'policy':'observed-demand-v1','evidence_sha256':'d'*64,
        'keys':graph['exact_graph_catalogue']['requested_keys']}}}
    projected = window._runtime_capacity_proof(engine, operation, retention, slots=value['slots'], bits=bits)
    assert projected['states'] == value['states'] and projected['capacity'] == cap
    assert not torch.cuda.is_initialized()
    with pytest.raises(ValueError, match='explicit capacity contract'):
        window._runtime_capacity_proof(engine, {}, retention)
    legacy = NS(capacity_plan={'session_capacity':{'ranks':[{'allocation_baseline':{'effective_device_floor_bytes':4*GIB}}]}})
    assert window._runtime_capacity_proof(legacy, {}, retention) is None


def test_finite_absolute_grants_and_unknown_private_storage_have_distinct_domains():
    for absolute in (31*GIB, 1 << 80):
        value, retention = proof()
        for row in value['capacity']['session_capacity']['ranks']:
            base = row['allocation_baseline']; base['absolute_grant_cap_bytes'] = absolute
            base['remaining_absolute_grant_bytes'] = absolute-base['preallocated_project_bytes']
        value['sample']['ownership']['absolute_cap_bytes'] = absolute
        window.validate_capacity_proof(value, retention=retention)
    value, retention = proof()
    value['capacity']['multi_graphs']['allocator'].update(private_pool_spare_bytes=None, accounting='physical-only')
    value['sample']['live_bytes'] = 3*GIB
    window.validate_capacity_proof(value, retention=retention)
    value['sample']['live_bytes'] = 5*GIB
    with pytest.raises(ValueError): window.validate_capacity_proof(value, retention=retention)


def test_public_service_cannot_treat_undeclared_zero_floor_as_legacy():
    window._legacy_capacity_plan({'session_capacity':{'ranks':[{'allocation_baseline':{'effective_device_floor_bytes':4*GIB}}]}})
    for bad in ({'full_window_headroom':{'policy':window.CAPACITY_CONTRACT}},
                {'session_capacity':{'ranks':[{'allocation_baseline':{'effective_device_floor_bytes':0}}]}}):
        with pytest.raises(ValueError, match='explicit capacity contract'):window._legacy_capacity_plan(bad)


def test_actual_http_service_consumer_requires_the_scoped_complete_capacity_plan():
    from tensorfold.families.qwen4_exp.cuda.owner_preparation import parse_preparation
    from tests.test_v100_target_window_contract import inputs
    _, _, client, _ = inputs()
    value, _ = proof(); cap = value['capacity']; catalogue = cap['multi_graphs']['exact_graph_catalogue']
    client.update(capacity_contract=window.CAPACITY_CONTRACT, preparation_contract=window.PREPARATION_CONTRACT)
    client['prepared_execution']['exact_graph_policy'] = {
        'policy':catalogue['policy'], 'evidence_sha256':catalogue['evidence_sha256'], 'keys':catalogue['requested_keys']}
    client['prepared_execution'] = parse_preparation(client['prepared_execution'])
    expected = client['expected_service']
    cap['expert_parallel'] = {k:expected[k] for k in ('tp','pp','etp','ep')}
    families = {k:expected[k] for k in ('kernel_family','graph_family')}
    binding = {'manifest_sha256':'c'*64,'verified_input_files':7,'verified_input_bytes':10,
        'content_passes':1,'hashes_reused_this_binding':0,'shared_immutable_same_host':True}
    stats = {'object':'tensorfold.stats','capacity_plan':cap,
        'model':{'id':client['model'],'context_length':262144,'kv_dtype':'int8'},
        'server':{'instance_id':'fixture-instance'},
        'load':{'requests_in_flight':0,'requests_running':0,'requests_waiting':0},
        'runtime':{'execution':{'execution_families':{'effective':families,'trial':families,
            'native_prepared':True,'graph_rows':0}},'capacity_plan':{'execution_profile_binding':binding}}}
    prepared = {'manifest_canonical_sha256':'c'*64,'manifest_artifact_files':7,'manifest_artifact_bytes':10}
    assert window.validate_service(stats,client,'fixture-instance',prepared_manifest=prepared)['resident_capacity'] == 24
    old = {k:v for k,v in client.items()if k != 'capacity_contract'}
    with pytest.raises(ValueError,match='explicit capacity contract'):
        window.validate_service(stats,old,'fixture-instance',prepared_manifest=prepared)
    broken = copy.deepcopy(stats);broken['capacity_plan']['session_capacity']['ranks'].pop()
    with pytest.raises(ValueError,match='eight allocation'):
        window.validate_service(broken,client,'fixture-instance',prepared_manifest=prepared)


def test_actual_source37_requested_membership_is_not_capture_sequence():
    import json
    from pathlib import Path
    real = json.loads((Path(__file__).parent/'fixtures/c8-source37-real-c36-order.json').read_text())
    value, retention = proof(); graph = value['capacity']['multi_graphs']
    assert real['requested_keys'] != real['captured_keys']
    assert set(map(tuple,real['requested_keys'])) == set(map(tuple,real['captured_keys']))
    # Preserve the actual native sequence and every actual byte-chain field.
    graph['exact_graph_catalogue']['requested_keys'] = real['requested_keys']
    graph['exact_graph_catalogue']['captured_keys'] = real['captured_keys']
    budget = real['accounting'][0]['budget_bytes']
    graph['planned_bytes'] = graph['budget_bytes'] = budget
    graph['keys'] = [{'bytes':max(0,row['physical_delta_bytes']),'pieces':row['pieces'],'accounting':row}
                     for row in real['accounting']]
    graph['bytes'] = graph['settled_bytes'] = graph['room_bytes'] = sum(row['bytes']for row in graph['keys'])
    graph['pieces'] = sum(row['pieces']for row in graph['keys'])
    admission = value['capacity']['observed_graph_admission'];admission['agreed_graph_bytes'] = budget
    for row in admission['observations']:
        row['graph_room_bytes'] = budget;row['live_after_floor_bytes'] = budget+row['future_service_bytes']
    graph['observed_graph_admission'] = {k:v for k,v in admission.items()
        if k not in ('original_header','observed_serving_base_bytes','basis')}
    admission['observed_serving_base_bytes'] = value['capacity']['budget_bytes']-budget
    # Only unpublished surrounding fields use the explicit CPU fixture. This
    # proves the offending predicate and actual captured chain, not hardware fit.
    window.validate_capacity_proof(value,retention=retention)
    for fault in ('duplicate','missing','measured_reorder','captured_reorder','prior_chain'):
        changed = copy.deepcopy(value);record = changed['capacity']['multi_graphs'];cat = record['exact_graph_catalogue']
        if fault=='duplicate':cat['captured_keys'][-1] = cat['captured_keys'][0]
        elif fault=='missing':cat['captured_keys'].pop()
        elif fault=='measured_reorder':record['keys'][0],record['keys'][1] = record['keys'][1],record['keys'][0]
        elif fault=='captured_reorder':cat['captured_keys'][0],cat['captured_keys'][1] = cat['captured_keys'][1],cat['captured_keys'][0]
        else:record['keys'][1]['accounting']['prior_bytes'] += 1
        with pytest.raises(ValueError):window.validate_capacity_proof(changed,retention=retention)


@pytest.mark.parametrize('rank',[0,7])
@pytest.mark.parametrize('capacity',[False,True])
def test_shared_content_reader_uses_actual_producer_registration_shape(rank,capacity):
    operation = {'capacity_contract':window.CAPACITY_CONTRACT} if capacity else {}
    binding = {'manifest_sha256':'a'*64,'bytes_read_this_binding':4096}
    projection = {'schema':'tensorfold.shared-execution-content-projection-v1','issuer_rank':0,
        'local_authority_owned':rank==0,'cohort_content_passes':1,
        'local_payload_bytes_read_this_binding':4096 if rank==0 else 0,
        'cohort_hashes_reused_this_binding':0,'manifest_sha256':'a'*64,'retained_host_bytes':512}
    registration = {'binding':binding,'numerics':{},'contents':{},'content_projection':projection}
    if capacity:registration['capacity_projection']={}
    window._validate_content_projection(registration,operation,rank)
    for key,values in {'cohort_content_passes':[True,False,1.0,None],
                      'cohort_hashes_reused_this_binding':[False,True,0.0,None],
                      'retained_host_bytes':[True,False,0,-1,512.0,None],
                      'local_payload_bytes_read_this_binding':[False,True,0.0,None],
                      'issuer_rank':[False,True,0.0,None]}.items():
        for invalid in values:
            altered=copy.deepcopy(registration);altered['content_projection'][key]=invalid
            with pytest.raises(ValueError):window._validate_content_projection(altered,operation,rank)
    for key in ('content_projection', 'capacity_projection' if capacity else 'numerics'):
        altered=copy.deepcopy(registration);altered.pop(key)
        with pytest.raises(ValueError):window._validate_content_projection(altered,operation,rank)
    altered=copy.deepcopy(registration);altered['unknown_projection']={}
    with pytest.raises(ValueError):window._validate_content_projection(altered,operation,rank)
