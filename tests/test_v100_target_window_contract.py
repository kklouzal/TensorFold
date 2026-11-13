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
        'artifacts': {'draft/'+Path(mtp['path']).name: {'size': mtp['bytes'], 'sha256': mtp['sha256']}}}
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
    with pytest.raises(ValueError, match='MTP content'):
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
