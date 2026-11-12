"""Full-shape synthetic closed records; no fixture is native/model evidence."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tensorfold.families.qwen4_exp.cuda.execution_identity import digest
from tests.test_v100_owned_teacher_comparison import _write
from tests.test_v100_prepared_teacher_contract import data, result
from tools import v100_onecat_combinations as prepared
from tools import v100_rotorquant_quality as quality


def plan(selector):
    value = ['owned-etp4-native-slots-v1', 8, 4, 512, 640, 2560, 11, 32, 256, False, 'grouped-v1']
    if selector['gu'] == 'incumbent':
        return value + ['grouped-gate-up', 4]
    selection = ['owned-onecat-gu-current-down-v1', selector['gu'], 117976064, 67108864, 'reference-grouped4']
    for key, marker, default in [('down', 'down', 'tensorfold'), ('iq2_source', 'iq2-source', 'active-frame'),
                                 ('prefill_method', 'prefill', 'tensorfold')]:
        if selector[key] != default:
            selection += [marker, selector[key]]
    return value + ['owned-onecat-active-frame-current-down-v1', selection]


def side(root, mode, selector, reference=None, *, fault=None):
    root.mkdir()
    operation, *_ = data()
    operation['selector'] = selector
    reference_sha = None
    if reference is not None:
        source, original = reference
        reference_sha = original['sha256']
        captured = json.loads((Path(source['root']) / original['file']).read_text())
        fixed = {'teacher/teacher-authority.json': {key: original[key] for key in ('bytes', 'sha256')}}
        for collection in ('rank_receipts', 'startup_receipts'):
            fixed.update({'teacher/' + row['file']: {key: row[key] for key in ('bytes', 'sha256')}
                          for row in captured[collection]})
        fixed.update({'teacher/' + row['vector']['file']: {key: row['vector'][key] for key in ('bytes', 'sha256')}
                      for row in captured['frames']})
        operation['reference'] = {'root': '/reference', 'authority': {'path': '/reference/teacher/teacher-authority.json',
            **{key: original[key] for key in ('bytes', 'sha256')}}, 'files': fixed,
            'host_result': {'path': '/reference/host-result.json',
                            **{key: source['host_result'][key] for key in ('bytes', 'sha256')}},
            'completion': {'path': '/operator/reference-capture-completion.json',
                           **{key: source['completion'][key] for key in ('bytes', 'sha256')}}}
    activation = ('8' if mode == 'capture' else '9') * 64
    family = operation['prepared_execution']
    native = {'tf_gguf_fixture.so': '7'*64}
    if selector['gu'] not in ('incumbent', 'tensorfold'):
        native.update({'tf_onecat_gu_fixture.so': '5'*64, 'tf_onecat_pack_fixture.so': '6'*64})
    if selector['down'] != 'tensorfold':
        native['tf_onecat_down_fixture.so'] = '4'*64
    # Hundreds of real-shaped source entries exercise complete registration
    # metadata and the existing prepaid JSON envelope, not a two-field toy.
    contents = {f'source/tensorfold/families/fixture/module_{i:04d}.py': {
        'size': 2048+i, 'sha256': hashlib.sha256(str(i).encode()).hexdigest()} for i in range(512)}
    contents.update({'library//native/'+name: {'size': 1, 'sha256': sha} for name, sha in native.items()})
    numbers = [{'execution_dtype': 'float16', 'qsa_selection': 'ordered-fixture', 'torch_deterministic': False,
        'float32_matmul_precision': 'highest', 'allow_tf32': False,
        'runtime_policy': {'torch': 'source-owned-fixture', 'cuda': '12.9', 'capability': [7, 0],
                           'rounding': {'attention': 'float32', 'stored': 'float16'}},
        'rope': {'theta': 10000., 'mode': 'original'}, 'eos': [248046], 'method': {'method': 'none', 'depth': 0},
        'mtp_import': None, 'expert_partition': {'rank': r, 'world': 8, 'etp': 4, 'ep': 2},
        'proposal_confidence': .5, 'owned_expert_native': plan(selector), 'draft_vocabulary': None} for r in range(8)]
    if fault:
        fault('numbers', None, numbers)
    identity = {key: 'a'*64 for key in ('target', 'draft', 'tokenizer', 'codec', 'runtime', 'topology', 'placement',
                                       'retention', 'target_weight_encoding')}
    identity.update(source=digest({'files': {k: v for k, v in contents.items() if k.startswith('source/')}}),
                    kernel_build=digest({'libraries': {k: v for k, v in contents.items() if k.startswith('library/')}}),
                    numerics=digest({'ranks': numbers}), context_tokens=262144, resident_sessions=24,
                    history_layout='unique', method='none', draft='none')
    binding = {'identity': identity, 'manifest_sha256': 'b'*64, 'verified_input_files': 7,
               'verified_input_bytes': 100, 'content_passes': 1, 'hashes_reused_this_binding': 0,
               'shared_immutable_same_host': True}
    kv = ['basis', ['coordinate-profile-v1', activation], ['logical-context-v1', 262144],
          ['owned-expert-native', plan(selector)]]
    vector = root/'teacher/teacher-rank0/frame-000000.npy'
    vector.parent.mkdir(parents=True)
    logits = np.zeros(248320, dtype='<f4')
    logits[1] = 4
    np.save(vector, logits, allow_pickle=False)
    raw = vector.read_bytes()
    frames = [{'step': 0, 'position': 3, 'teacher_id': 1, 'selected_original': 1, 'full_vocab': 248320,
               'role': 'prefill', 'execution_family': family, 'kv_identity': kv,
               'vector': {'file': 'teacher-rank0/frame-000000.npy', 'bytes': len(raw),
                          'sha256': hashlib.sha256(raw).hexdigest(), 'dtype': '<f4', 'shape': [248320]}}]
    groups = {name: [] for name in ('startup_receipts', 'rank_receipts', 'child_native_records')}
    profiles = []
    sources = {key: 'a'*64 for key in ('engine', 'multi', 'decode', 'sampling', 'teacher')}
    for r in range(8):
        registration = {'binding': copy.deepcopy(binding), 'numerics': copy.deepcopy(numbers[r]), 'contents': contents}
        if fault:
            fault('registration', r, registration)
        profile = {'schema': 'tensorfold.private-owned-qualification-v1', 'policy': 'owned-etp4-native-slots-v1',
            'rank': r, 'world': 8, 'target_banks': 48, 'mtp_banks': 0, 'capability': [7, 0],
            'plan_identity': plan(selector), 'coordinate_profile_identity': activation,
            'teacher_execution_registration': registration, 'teacher_semantics': {'state_kv_identity': kv,
                'context': 262144, 'cache_slots': 262144, 'prefill_rows': 256, 'target_rows': 24,
                'vocab_widths': [31040]*8, 'sm70_profile': 'reference', 'prepared_execution': family}}
        profiles.append(profile)
        startup = {'rank': r, 'world': 8, 'plan_identity': plan(selector), 'state_count': 24, 'mtp_banks': 0,
            'teacher_profile': profile, 'coordinate_profile_identity': activation, 'teacher_worker_PID': 100+r,
            'teacher_server_instance_id': 'instance', 'teacher_observer': {
                'path': '/canonical/tools/v100_native_contract_driver.py', 'bytes': 3, 'sha256': 'd'*64},
            'teacher_windows': {'full_resident': True, 'slots': 24, 'capacity_min': 262145, 'capacity_max': 262145,
                'limit_min': 262145, 'limit_max': 262145, 'logical_min': 262144, 'logical_max': 262144}}
        if selector['gu'] == 'incumbent':
            startup['grouped_gate_up'] = {'requested_rows': 4, 'actual_rows': 4}
        else:
            gu = selector['gu'] != 'tensorfold'
            p = {'policy': 'owned-onecat-active-frame-current-down-v1', 'method': selector['gu'],
                'down_method': selector['down'], 'iq2_source': selector['iq2_source'], 'prefill_method': selector['prefill_method'],
                'banks': 48, 'frame_slab_bytes': 117964800 if gu else 0, 'shared_lookup_bytes': 11264 if gu else 0,
                'additional_device_bytes': 117976064 if gu else 0, 'original_raw_retained': True,
                'baseline_grouped_gate_up_rows': 4, 'packer_workspace_bytes': 0, 'mtp_provider': 'tensorfold-fp32-mtp',
                'down_format_policy': {'20': selector['down'], '42': 'tensorfold'}}
            origins = {name: {'path': '/native/tf_gguf_fixture.so'} for name in ('gate_up', 'down')}
            if gu:
                origins.update(gate_up={'path': '/native/tf_onecat_gu_fixture.so'}, packer={'path': '/native/tf_onecat_pack_fixture.so'})
            if selector['down'] != 'tensorfold':
                origins['q8_down'] = {'path': '/native/tf_onecat_down_fixture.so'}
            if selector['iq2_source'] == 'raw':
                origins['raw_iq2_gate_up'] = origins['gate_up']
            startup['onecat_experts'] = {'schema': 'tensorfold.private-onecat-frame-binding-v1', 'rank': r, 'world': 8,
                'status': 'prepared-execution-eligible', 'record': {'schema': 'tensorfold.onecat-active-frame-owner-v1',
                    'enabled': True, 'closed': False, 'profile': p, 'raw_bindings': [{}]*48}, 'origins': origins}
        summary = {'schema': 'tensorfold.owned-teacher-result-v1', 'rank': r, 'mode': mode, 'closed': True, 'finished': True, 'failed': False,
                   'teacher_ids': [1], 'runtime_sources': sources, 'actual_owned_profile': profile, 'frames': frames}
        observed = {'schema': 'tensorfold.native-elf-observation-v3', 'identity_method': 'kernel-vma-backed-fd-v1',
            'rank': r, 'tag': 'teacher-closed', 'PID': 100+r, 'operation_limits': prepared.MODEL_OBSERVER_LIMITS,
            'missing_required_prefixes': [], 'observer': {'bytes': 3, 'sha256': 'd'*64},
            'access_precondition': {'uids': [1000]*4, 'gids': [1000]*4, 'NoNewPrivs': '0',
                'caps': {k: '0000010000000000' for k in ('CapInh', 'CapPrm', 'CapEff', 'CapBnd', 'CapAmb')}},
            'mapped_elfs': [{'name': name, 'path': '/native/'+name, 'bytes': 1, 'sha256': sha,
                'vmas': [{'start': 1, 'end': 2, 'permissions': 'r-xp', 'offset': 0}]} for name, sha in native.items()]}
        for name, filename, record in [('startup_receipts', f'teacher/rank-{r}.json', startup),
              ('rank_receipts', f'teacher/teacher-rank{r}/summary.json', summary),
              ('child_native_records', f'child-native/teacher-closed-rank-{r}-pid-{100+r}.json', observed)]:
            if fault:
                fault(name, r, record)
            groups[name].append({**_write(root, filename, record), 'rank': r})
    authority = {'schema': 'tensorfold.owned-teacher-authority-v1', 'mode': mode, 'method': 'none',
        'closed': True, 'finished': True, 'failed': False, 'cohort_complete': True, 'launch_exit': 0,
        'input_sha256': operation['teacher_plan']['sha256'], 'reference_sha256': reference_sha,
        'input_ids': [1, 2, 3], 'vocab': 248320, 'widths': [31040]*8, 'teacher_ids': [1], 'frames': frames,
        'runtime_sources': sources, 'owner_family': family, 'declared_identity': {'case_id': 'explicit_fixture',
            'model_sha256': 'a'*64, 'frontend_sha256': 'b'*64, 'profile_sha256': 'c'*64}, 'actual_owned_profile': profiles[0],
        **{key: [{**row, 'file': row['file'].removeprefix('teacher/')} for row in groups[key]]
           for key in ('rank_receipts', 'startup_receipts')}}
    authority_ref = _write(root, 'teacher/teacher-authority.json', authority)
    client = {'schema': 'tensorfold.owned-teacher-client-report-v1', 'complete': True,
        'quality_acceptance': False, 'P3_valid': False, 'completion_tokens': 1, 'prompt_tokens': 3,
        'client_protocol_passed': True, 'post_attempts': 1, 'mode': mode, 'token_ids': [1],
        'teacher_plan_sha256': operation['teacher_plan']['sha256'], 'teacher_client_sha256': operation['teacher_client']['sha256'],
        'cached_tokens': 0, 'drafts': False, 'drafted': 0, 'accepted': 0, 'server_instance_id': 'instance'}
    model = {'schema': prepared.RESULT_SCHEMA, 'teacher_plan_sha256': operation['teacher_plan']['sha256'],
        'teacher_client_sha256': operation['teacher_client']['sha256'], 'launch_exit': 0,
        'teacher_authority': authority_ref, 'client_report': _write(root, 'teacher/teacher-client-report.json', client), **groups}
    host = result(operation)
    host['inner'].update(model_result=model, content_binding=_write(root, 'model-content-binding.json', {
        'completed': True, 'assets': [{'identity': item} for item in prepared.operation_assets(operation)]}))
    _write(root, 'owned-teacher-operation-result.json', host['inner'])
    host['source_after'] = {'source_manifest': '9'*64, 'source_fixture': {'base': 'same'},
        'source': {'tools/v100_native_contract_driver.py': {'bytes': 3, 'sha256': 'd'*64}},
        'image_id': host['inner']['image_id_declared']}
    host_ref = _write(root, 'host-result.json', host)
    completion = {'schema': 'tensorfold.owned-teacher-completion-v1', 'host_result_sha256': host_ref['sha256'],
        'authority_sha256': authority_ref['sha256'], 'host_cli_exit': 0,
        **{k: 'e'*64 for k in ('operation_gate_sha256', 'review_sha256', 'terminal_observation_sha256')}}
    completion_ref = _write(root, 'completion.json', completion)
    return {'root': str(root), 'host_result': host_ref, 'completion': {'path': str(root/'completion.json'),
            **{k: completion_ref[k] for k in ('bytes', 'sha256')}}}, authority_ref, profiles[0]


def test_full_registration_closed_pair_uses_full_vectors_and_original_metric_arithmetic(tmp_path):
    first, authority, profile = side(tmp_path/'reference', 'capture', prepared.selectors()[0])
    selected = prepared.selectors()[-1]
    second, _, _ = side(tmp_path/'candidate', 'replay', selected, (first, authority))
    report = quality.compare_owned_candidate_teacher(first, second, method=selected['gu'], down_method=selected['down'],
        iq2_source=selected['iq2_source'], prefill_method=selected['prefill_method'], read_timeout_s=30,
        teacher_scope=prepared.SCOPE)
    assert report['complete'] and report['quality_metrics']['mean_kl'] == 0
    assert report['quality_metrics']['ppl_ratio'] == 1 and not report['quality_acceptance']
    assert report['coverage']['decoder_target_rows'] == 24 and report['coverage']['expert_prepared_rows'] == 32
    assert len(profile['teacher_execution_registration']['contents']) >= 512
    # Compare the actual serialized profile footprint with the issuer's normal
    # per-rank receipt envelope, rather than waiving the existing bound.
    resources = quality.owned_teacher_resources(1024, 0, 248320, 64)
    assert len(json.dumps(profile).encode()) < resources['receipt_bytes_limit']


@pytest.mark.parametrize('fault', ['rank', 'native', 'source', 'reuse', 'numeric-policy'])
def test_all_rank_native_source_and_numeric_drift_refuse_before_vector_arithmetic(tmp_path, monkeypatch, fault):
    first, authority, _ = side(tmp_path/'reference', 'capture', prepared.selectors()[0])
    def alter(kind, rank, record):
        if fault == 'rank' and kind == 'startup_receipts' and rank == 5:
            record['teacher_worker_PID'] += 1
        elif fault == 'native' and kind == 'child_native_records' and rank == 7:
            record['mapped_elfs'][0]['sha256'] = 'f'*64
        elif fault == 'source' and kind == 'child_native_records' and rank == 3:
            record['observer']['sha256'] = 'f'*64
        elif fault == 'reuse' and kind == 'registration':
            record['binding']['hashes_reused_this_binding'] = 7
        elif fault == 'numeric-policy' and kind == 'numbers':
            record[4]['allow_tf32'] = True
    selected = prepared.selectors()[-1]
    second, _, _ = side(tmp_path/'candidate', 'replay', selected, (first, authority), fault=alter)
    monkeypatch.setattr(quality, '_candidate_teacher_metrics', lambda *args: pytest.fail('invalid metadata reached vectors'))
    with pytest.raises(ValueError):
        quality.compare_owned_candidate_teacher(first, second, method=selected['gu'], down_method=selected['down'],
            iq2_source=selected['iq2_source'], prefill_method=selected['prefill_method'], read_timeout_s=30,
            teacher_scope=prepared.SCOPE)
