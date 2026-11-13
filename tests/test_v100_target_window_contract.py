"""Pure versioned operation controls; fixtures are not hardware/quality evidence."""
import copy
import itertools
import json
from pathlib import Path

import pytest

from tests.test_v100_prepared_teacher_contract import data
from tests.test_v100_target_window_teacher import plan as window_plan
from tools import v100_owned_model_contract as model
from tools import v100_owned_qualification as qualifier
from tools import v100_target_window_teacher as window


def inputs(flags=(False, False, False)):
    operation, _, _, _, client = data()
    operation.pop('case_authority')
    trial = dict(zip(window.numerical.AXES, flags))
    prep = {**operation['prepared_execution'], 'native_gdn_capture': True, 'native_hc_writeback_capture': True,
        'exact_graph_policy': {'policy': 'paid-subset-v1', 'keys': [['verify', 8, 8], ['verify', 40, 8], ['mtp', 8, 8]],
                               'evidence_sha256': 'd'*64},
        **{k: v for k, v in window.numerical.AXES.items() if trial[k]}}
    numa = {'schema': window.numa.NUMA_SCHEMA, 'placement_source': {'bytes': 1, 'sha256': '1'*64},
        **{k: {'path': '/operator/'+name, 'bytes': 1, 'sha256': '2'*64} for k, (name, _) in window.FILES.items()}}
    mtp = {'path': '/models/hf/hub/fixture/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf',
        'bytes': 2786568256, 'sha256': '5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6'}
    plan = window_plan(4)
    plan['identity']['profile_sha256'] = operation['execution_manifest']['sha256']
    resource = window.resources(operation['teacher_plan']['bytes'], plan)
    operation.update(teacher_scope=window.SCOPE, prepared_execution=prep, numerical_trial=trial,
        mtp_artifact=mtp, numa=numa, mtp_confidence=0.7,
        output_grants={'frame_bytes': resource['max_total_frame_file_bytes'],
            'metadata_bytes': resource['metadata_output_bytes_bound'],
            'disk_bytes': resource['max_total_frame_file_bytes']+resource['metadata_output_bytes_bound'],
            'whole_timeout_s': client['whole_timeout_s']})
    client.update(schema=window.CLIENT_SCHEMA, teacher_scope=window.SCOPE, prepared_execution=copy.deepcopy(prep),
        numerical_trial=copy.deepcopy(trial), numa=copy.deepcopy(numa), mtp_confidence=0.7)
    profile = {'schema': 'tensorfold.execution-profile-manifest', 'version': 1, 'preparation': copy.deepcopy(prep),
        'artifacts': {**{'target/'+Path(item['path']).name: {'size': item['bytes'], 'sha256': item['sha256']}
                         for item in operation['model_assets']},
                      **{'metadata/'+Path(item['path']).name: {'size': item['bytes'], 'sha256': item['sha256']}
                         for item in operation['frontend_assets'] if Path(item['path']).name != 'ready.json'},
                      'draft/'+Path(mtp['path']).name: {'size': mtp['bytes'], 'sha256': mtp['sha256']}}}
    return operation, plan, client, profile


def test_versioned_numerical_combinations_keep_the_existing_expert_axis():
    for flags in itertools.product((False, True), repeat=3):
        operation, plan, client, profile = inputs(flags)
        for selector in window.selectors():
            operation['selector'] = client['selector'] = selector
            window.validate_operation(operation)
            window.validate_teacher_client(client)
            assert window.validate_teacher_operation_plan(operation, plan, client, profile=profile)['mode'] == 'forced-window'
        for old in (window.numerical, window.numa, window.numerical.base):
            with pytest.raises(ValueError):
                old.validate_operation(operation)


def test_v4_dispatch_and_fixed_mtp_full_capacity_cli():
    operation, _, client, _ = inputs((True, True, True))
    assert model.teacher_contract({'schema': window.SCHEMA}) is window
    assert model.teacher_operation_contract(operation) is window
    ids = [f'GPU-00000000-0000-0000-0000-{n:012x}' for n in range(8)]
    args = window.qualifier_argv(operation, client, ids, '/evidence/teacher')
    split = args.index('--')
    assert '--no-graphs' not in args
    assert args[args.index('--draft-method')+1] == 'mtp'
    assert args[args.index('--mtp-drafts')+1] == '4'
    assert args[args.index('--drafter')+1] == operation['mtp_artifact']['path']
    assert args[args.index('--max-tokens')+1] == '128'
    assert args.index('--numa-plan') < split and args.index('--execution-rows') < split
    assert args[args.index('--execution-rows')+1] == '120'
    assert args[args.index('--parallel')+1] == '24' and args[args.index('--context')+1] == '262144'


@pytest.mark.parametrize('field,bad', [('bytes', 1), ('sha256', '0'*64), ('path', '/models/other.gguf')])
def test_unmatched_mtp_artifact_is_not_just_a_drafter_flag(field, bad):
    operation, *_ = inputs()
    operation['mtp_artifact'][field] = bad
    with pytest.raises(ValueError, match='MTP'):
        window.validate_operation(operation)


def test_paid_catalogue_without_c8_keys_cannot_claim_graph_coverage():
    operation, *_ = inputs()
    operation['prepared_execution']['exact_graph_policy']['keys'] = [['verify', 1, 1], ['mtp', 1, 1]]
    with pytest.raises(ValueError, match='eight-participant'):
        window.validate_operation(operation)


def test_raw_manifest_identity_and_complete_draft_content_remain_binding():
    operation, plan, client, profile = inputs()
    changed = copy.deepcopy(plan)
    changed['identity']['profile_sha256'] = 'f'*64
    with pytest.raises(ValueError, match='profile'):
        window.validate_teacher_operation_plan(operation, changed, client, profile=profile)
    profile['artifacts'].clear()
    with pytest.raises(ValueError, match='manifest artifacts'):
        window.validate_teacher_operation_plan(operation, plan, client, profile=profile)


def test_complete_frame_disk_and_time_grants_are_not_silently_reduced():
    for field in ('frame_bytes', 'metadata_bytes', 'whole_timeout_s'):
        operation, plan, client, profile = inputs()
        operation['output_grants'][field] -= 1
        with pytest.raises(ValueError):
            window.validate_teacher_operation_plan(operation, plan, client, profile=profile)


@pytest.mark.parametrize('fault', [None, 'plan_extent', 'plan_hash'])
def test_actual_parent_client_input_uses_v4_plan_bounds_without_legacy_max_frames(tmp_path, fault):
    _, plan, client, _ = inputs()
    teacher = {'data': plan, 'input_sha256': client['teacher_plan_sha256']}
    path = tmp_path/'client.json'
    path.write_text(json.dumps(client))
    if fault == 'plan_extent':
        plan['sequences'][0]['forced_ids'] *= 33
    elif fault == 'plan_hash':
        teacher['input_sha256'] = '0'*64
    if fault:
        with pytest.raises(ValueError):
            qualifier._teacher_client_input(path, teacher, prepared=True, window=True)
    else:
        assert 'max_frames' not in plan
        assert qualifier._teacher_client_input(path, teacher, prepared=True, window=True)['data'] == client


def test_actual_parent_client_input_retains_legacy_max64_bound(tmp_path):
    _, _, _, plan, client = data()
    path = tmp_path/'client.json'
    path.write_text(json.dumps(client))
    teacher = {'data': plan, 'input_sha256': client['teacher_plan_sha256']}
    for frames in (0, 65):
        plan['max_frames'] = frames
        with pytest.raises(ValueError, match='MAX64'):
            qualifier._teacher_client_input(path, teacher, prepared=True)


def current_inputs(flags=(False, False, False)):
    from tensorfold.families.qwen4_exp.cuda.exact_graph_policy import parse_policy
    operation, plan, client, profile = inputs(flags)
    prep = operation['prepared_execution']
    prep['exact_graph_policy'] = parse_policy(prep['exact_graph_policy'])
    for value in (operation, client):
        value['preparation_contract'] = window.PREPARATION_CONTRACT
    client['prepared_execution'] = copy.deepcopy(prep)
    profile['preparation'] = copy.deepcopy(prep)
    return operation, plan, client, profile


def test_observed_32_key_profile_matches_actual_startup_catalogue_and_keeps_legacy_closed():
    from types import SimpleNamespace

    from tensorfold.families.qwen4_exp.cuda.exact_graphs import ExactStateGraphs
    operation, _, client, _ = current_inputs()
    graph = object.__new__(ExactStateGraphs)
    graph.dec = SimpleNamespace(depth=4, buf=SimpleNamespace(rows=120), mbuf=SimpleNamespace(rows=120))
    graph.requested_policy = None
    graph._catalogue(list(range(24)))
    keys = sorted({*graph.available_keys, ('verify', 28, 8), ('mtp', 28, 8)})
    assert len(keys) == 32
    policy = {'policy':'observed-demand-v1', 'keys':[list(k) for k in keys], 'evidence_sha256':'a'*64}
    operation['prepared_execution']['exact_graph_policy'] = policy
    window.validate_operation(operation)
    graph.requested_policy = {**policy, 'keys':tuple(tuple(key) for key in policy['keys'])}
    graph._catalogue(list(range(24)))
    assert set(graph.requested_keys) == set(keys)
    with pytest.raises(ValueError):
        window.validate_preparation(operation['prepared_execution'], operation['numerical_trial'])
    client['preparation_contract'] = 'unknown-version'
    with pytest.raises(ValueError, match='contract'):
        window.validate_teacher_client(client)


@pytest.mark.parametrize('key,value', [*window.TRANSPORT_OPTIONS.items(),
    ('native_hc_writeback_target_eager', True), ('gdn_value8_capture', True)])
def test_explicit_candidate_options_bind_actual_runtime_owner_numerics(key, value):
    operation, *_ = current_inputs((False, False, key == 'owned_q8_window_dispatch'))
    operation['prepared_execution'][key] = value
    window.validate_operation(operation)
    expected = window._expected_preparation_numerics(operation)
    raw = {} if expected is None else {'owner_preparation_numerics':expected}
    window._validate_preparation_numerics(raw, operation)
    if expected is not None:
        with pytest.raises(ValueError, match='numerical registration'):
            window._validate_preparation_numerics({'owner_preparation_numerics':{}}, operation)
    bad = copy.deepcopy(operation)
    bad['prepared_execution'][key] = 'unsupported'
    with pytest.raises(ValueError):
        window.validate_operation(bad)


@pytest.mark.parametrize('backend', ['cublas', 'cublaslt', 'cutlass'])
def test_vendor_prefill_and_m32_compose_only_with_disjoint_actual_runtime_roles(backend):
    operation, *_ = current_inputs((False, True, False))
    dense = {'backend':backend,'geometry':'fixed-m32','roles':['prefill'],'panel_rows':32}
    operation['prepared_execution']['dense_projections'] = dense
    window.validate_operation(operation)
    expected = window._expected_preparation_numerics(operation)
    assert expected['dense_projections']['backend'] == backend
    assert expected['projection_role_composition']['reference_bit_identity'] is False
    dense['roles'] = ['decode']
    with pytest.raises(ValueError, match='disjoint'):
        window.validate_operation(operation)


@pytest.mark.parametrize('key', [['verify',41,8], ['mtp',121,24], ['verify',25,25], ['mtp',True,1]])
def test_extended_graph_authority_refuses_outside_actual_state_row_depth_envelope(key):
    operation, *_ = current_inputs()
    policy = operation['prepared_execution']['exact_graph_policy']
    policy.update(policy='observed-demand-v1', keys=sorted([*policy['keys'], key]))
    with pytest.raises(ValueError):
        window.validate_operation(operation)


@pytest.mark.parametrize('options', [('descriptor_upload',), ('row_sampling_scratch',),
                                    ('descriptor_upload', 'row_sampling_scratch')])
def test_preparation_comparison_keeps_actual_rank_lists_and_unrelated_retention_fields(options):
    from tensorfold.families.qwen4_exp.cuda.execution_identity import digest
    reference, *_ = current_inputs()
    candidate = copy.deepcopy(reference)
    for key in options:
        candidate['prepared_execution'][key] = window.TRANSPORT_OPTIONS[key]
    changes = window.preparation_changes(reference, candidate)
    assert all(value['classification'] == 'transport-contract' for value in changes.values())
    def row(operation):
        from types import SimpleNamespace

        from tensorfold.families.qwen4_exp.cuda.history_owner import OwnerBuffers
        from tensorfold.families.qwen4_exp.cuda.row_sampling import RowSamplingScratch
        from tensorfold.families.qwen4_exp.kv_formats import get_pair
        from tests.test_native_owner_adapter import _weights

        scratch = 'row_sampling_scratch' in operation['prepared_execution']
        allocations, scratch_records = {}, {}
        for role, rows in [('decode',120),('mtp',120),('prefill',256)]:
            records = []
            for rank in range(8):
                owner = OwnerBuffers(_weights(rank), rows, SimpleNamespace(idw=2052), prefill=role=='prefill')
                sample = None
                try:
                    owner.prepare_native(pair=get_pair('int8'), descriptor_upload_policy=
                        operation['prepared_execution'].get('descriptor_upload'))
                    record = owner.allocation_record()
                    if scratch:
                        sample = RowSamplingScratch(31040, rows, role, 'cpu')
                        record['row_sampling_scratch'] = sample.record()
                        scratch_records[role] = sample.record()
                    records.append(record)
                finally:
                    if sample is not None:
                        sample.close()
                    owner.close()
            allocations[role] = records
        owner = {'specification':{k:v for k,v in operation['prepared_execution'].items() if k!='trial'},
                 'allocations':allocations, 'roles':{'decode':120,'mtp':120,'prefill':256}}
        if scratch:
            owner['row_sampling_scratch'] = scratch_records
        raw = {'owner_preparation':owner, 'indexer_history':'gpu'}
        return {'teacher_profile':{'numerical_trial_retention':raw,
            'teacher_execution_registration':{'binding':{'identity':{'retention':digest(raw)}}}}}
    a, b = row(reference), row(candidate)
    if 'descriptor_upload' in options:
        old_record = a['teacher_profile']['numerical_trial_retention']['owner_preparation']['allocations']['decode'][0]
        new_record = b['teacher_profile']['numerical_trial_retention']['owner_preparation']['allocations']['decode'][0]
        assert old_record['host_metadata_bytes'] != new_record['host_metadata_bytes']
    before = copy.deepcopy(b)
    assert window._comparison_retention(a, reference, changes) == window._comparison_retention(b, candidate, changes)
    assert b == before  # The actual paid/raw lists are not rewritten by comparison.
    raw = b['teacher_profile']['numerical_trial_retention']
    raw['owner_preparation']['allocations']['decode'][0]['heads'] += 1
    b['teacher_profile']['teacher_execution_registration']['binding']['identity']['retention'] = digest(raw)
    assert window._comparison_retention(a, reference, changes) != window._comparison_retention(b, candidate, changes)
    raw['owner_preparation']['allocations']['decode'] = {'rank':0,'rows':120}
    b['teacher_profile']['teacher_execution_registration']['binding']['identity']['retention'] = digest(raw)
    with pytest.raises(ValueError, match='eight-rank'):
        window._comparison_retention(b, candidate, changes)
    candidate['prepared_execution']['exact_graph_policy']['evidence_sha256'] = 'f'*64
    with pytest.raises(ValueError, match='unrelated'):
        window.preparation_changes(reference, candidate)


def test_extended_contract_is_explicit_and_does_not_admit_external_or_unknown_profiles():
    for key, value in [('unknown_axis', True), ('flash_tiles', {}),
                       ('dense_projections', {'backend':'bitblas-hc'}), ('reference_arithmetic', {})]:
        operation, *_ = current_inputs()
        operation['prepared_execution'][key] = value
        with pytest.raises(ValueError):
            window.validate_operation(operation)


def test_current_preparation_contract_is_bound_to_both_actual_operation_and_client():
    operation, plan, client, profile = current_inputs()
    window.validate_teacher_operation_plan(operation, plan, client, profile=profile)
    client.pop('preparation_contract')
    with pytest.raises(ValueError, match='contract'):
        window.validate_teacher_operation_plan(operation, plan, client, profile=profile)


def test_real_source30_raw_profile_requires_a_new_canonical_manifest_identity():
    import hashlib

    from tensorfold.families.qwen4_exp.cuda.owner_preparation import parse_preparation

    # Exact published Source30 native-gate DATA, retained as provenance only.
    raw = (Path(__file__).parent/'fixtures/source30-ragged-profile-raw.json').read_bytes()
    assert len(raw) == 4013
    assert hashlib.sha256(raw).hexdigest() == '51ccd28928968520bef95c3de3cd5b5694230bd54a047a35c5b335207f86d4b6'
    profile = json.loads(raw)
    original = copy.deepcopy(profile)
    assert 'row_buckets' not in profile['preparation']
    operation, plan, client, held_profile = current_inputs((True, True, True))
    operation['prepared_execution'] = profile['preparation']
    with pytest.raises(ValueError, match='canonical manifest'):
        window.validate_operation(operation)
    canonical = parse_preparation(profile['preparation'])
    assert canonical['row_buckets'] == {}
    assert profile == original
    # The derived manifest must also bind this test operation's held assets;
    # the immutable historical profile above remains provenance, not live DATA.
    profile = {**profile, 'preparation':canonical, 'artifacts':held_profile['artifacts']}
    encoded = (json.dumps(profile, indent=2)+'\n').encode()
    assert hashlib.sha256(encoded).hexdigest() != hashlib.sha256(raw).hexdigest()
    operation['prepared_execution'] = canonical
    operation['execution_manifest'] = {**operation['execution_manifest'], 'bytes':len(encoded),
                                       'sha256':hashlib.sha256(encoded).hexdigest()}
    window.validate_operation(operation)
    actual = window._expected_preparation_numerics(operation)
    assert actual['owned_q8_window_preparation']['policy'] == 'owned-q8-window-grid-preparation-v1'
    plan['identity']['profile_sha256'] = operation['execution_manifest']['sha256']
    client['prepared_execution'] = canonical
    client['execution_manifest_sha256'] = operation['execution_manifest']['sha256']
    window.validate_teacher_operation_plan(operation, plan, client, profile=profile)
    client['execution_manifest_sha256'] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError, match='identity'):
        window.validate_teacher_operation_plan(operation, plan, client, profile=profile)


@pytest.mark.parametrize('limits,thread_stack,valid', [
    ((8 << 20, 64 << 20), 0, True),  # Actual Source32 CAP40 interpreter shape.
    ((64 << 20, 64 << 20), 0, True),
    ((0, 64 << 20), 0, False),
    ((-1, 64 << 20), 0, False),
    ((8 << 20, -1), 0, False),
    (((64 << 20)+1, 64 << 20), 0, False),
    ((64 << 20, (64 << 20)+1), 0, False),
    ((8 << 20, 64 << 20), 1 << 20, False),
])
def test_http_stack_effective_soft_limit_is_bounded_by_original_admission(monkeypatch, tmp_path, limits, thread_stack, valid):
    import resource
    import threading

    monkeypatch.setattr(resource, 'getrlimit', lambda kind: limits)
    monkeypatch.setattr(threading, 'stack_size', lambda: thread_stack)
    if valid:
        client = window.WindowClient(None, None, tmp_path)
        assert client.result is None and client.post_attempts == 0
    else:
        with pytest.raises(ValueError, match='stack policy exceeds'):
            window.WindowClient(None, None, tmp_path)


def test_http_stack_payload_keeps_all_eight_original_64MiB_bounds():
    from tests.test_v100_target_window_teacher import plan

    teacher = {'resources':window.resources(1, plan(4))}
    window.client_credit(teacher, 1)
    assert teacher['resources']['client_maximum_threads'] == 8
    assert teacher['resources']['client_thread_stack_per_thread_bound'] == 64 << 20
    assert teacher['resources']['client_thread_stack_virtual_bound'] == 8*(64 << 20)


@pytest.mark.parametrize('prefix', ['target/', 'draft/', 'metadata/'])
@pytest.mark.parametrize('fault', ['size', 'sha256', 'missing', 'extra', 'record-extra', 'bool-size'])
def test_complete_runtime_artifact_correspondence_rejects_stale_or_partial_profiles(prefix, fault):
    operation, plan, client, profile = inputs()
    name = next(key for key in profile['artifacts'] if key.startswith(prefix))
    if fault == 'size':
        profile['artifacts'][name]['size'] += 1
    elif fault == 'sha256':
        profile['artifacts'][name]['sha256'] = '0'*64
    elif fault == 'missing':
        del profile['artifacts'][name]
    elif fault == 'extra':
        profile['artifacts'][prefix+'unconsumed.json'] = {'size': 1, 'sha256': '0'*64}
    elif fault == 'record-extra':
        profile['artifacts'][name]['path'] = '/other'
    else:
        # Even a numerically equal JSON bool cannot replace a byte extent.
        if prefix == 'metadata/':
            item = next(item for item in operation['frontend_assets'] if Path(item['path']).name == name.split('/', 1)[1])
            item['bytes'] = profile['artifacts'][name]['size'] = 1
        profile['artifacts'][name]['size'] = True
    with pytest.raises(ValueError, match='manifest artifacts'):
        window.validate_teacher_operation_plan(operation, plan, client, profile=profile)


def test_runtime_frontend_inventory_excludes_preparation_ready_marker():
    operation, plan, client, profile = inputs()
    expected = {'metadata/config.json', 'metadata/tokenizer.json', 'metadata/tokenizer_config.json',
                'metadata/generation_config.json'}
    assert {name for name in profile['artifacts'] if name.startswith('metadata/')} == expected
    assert window.validate_teacher_operation_plan(operation, plan, client, profile=profile)['mode'] == 'forced-window'
    ready = next(item for item in operation['frontend_assets'] if Path(item['path']).name == 'ready.json')
    profile['artifacts']['metadata/ready.json'] = {'size': ready['bytes'], 'sha256': ready['sha256']}
    with pytest.raises(ValueError, match='manifest artifacts'):
        window.validate_teacher_operation_plan(operation, plan, client, profile=profile)
