"""Offline matched-control boundary/metric tests; no model or quality margins."""
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from tools import v100_rotorquant_quality as quality


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, allow_nan=False) + '\n').encode()
    path.write_bytes(raw)
    return {'file': name, 'bytes': len(raw), 'sha256': _sha(raw)}


def _side(root, mode, reference=None, *, arrays=None, alter=None, host_fault=None, completion_fault=None, native_fault=None, image=None, activation=None, record_fault=None, state_change=None):
    root.mkdir()
    teacher_ids = [1, 1, 1, 3]
    activation = (('8' if mode == 'capture' else '9') * 64) if activation is None else activation
    state_kv = ['basis-v1', ['position-profile-v1', 'a' * 64, 0],
                ['coordinate-profile-v1', activation], ['logical-context-v1', 262144]]
    if state_change:
        state_change(state_kv)
    profile = {key: 'stable' for key in ('plan_identity', 'policy', 'target_banks', 'mtp_banks', 'state_count', 'capability')}
    profile.update(coordinate_profile_identity=activation, teacher_semantics={
        'state_kv_identity': state_kv, 'context': 262144, 'other_stable': 'unchanged'})
    sources = {key: 'a' * 64 for key in ('engine', 'multi', 'decode', 'sampling')}
    sources['teacher'] = ('b' if mode == 'capture' else 'c') * 64
    frames = []
    for step in range(4):
        path = root / f'teacher/teacher-rank0/frame-{step:06d}.npy'
        path.parent.mkdir(parents=True, exist_ok=True)
        value = np.array([0., 1., 1., 0.], dtype='<f4') if arrays is None else arrays[step]
        np.save(path, value, allow_pickle=False)
        raw = path.read_bytes()
        frames.append({'step': step, 'position': 2 + step, 'teacher_id': teacher_ids[step], 'full_vocab': 4,
                       'selected_original': teacher_ids[step], 'role': 'prefill' if step == 0 else 'decode',
                       'execution_family': {'kernel_family': 'torch-reference', 'graph_family': 'uncaptured'},
                       'kv_identity': copy.deepcopy(state_kv), 'vector': {
                           'file': f'teacher-rank0/frame-{step:06d}.npy', 'bytes': len(raw),
                           'sha256': _sha(raw), 'dtype': '<f4', 'shape': [4]}})
    groups = {}
    authority_groups = {}
    for group in ('rank_receipts', 'startup_receipts', 'child_native_records'):
        groups[group], authority_groups[group] = [], []
        for rank in range(8):
            name = (f'teacher/teacher-rank{rank}/summary.json' if group == 'rank_receipts' else
                    f'teacher/rank-{rank}.json' if group == 'startup_receipts' else
                    f'child-native/teacher-closed-rank-{rank}-pid-{100+rank}.json')
            record = {'rank': rank, 'closed': True, 'finished': True, 'failed': False,
                      'mode': mode, 'teacher_ids': teacher_ids, 'runtime_sources': sources}
            if group == 'rank_receipts':
                record['actual_owned_profile'] = copy.deepcopy(profile)
                record['frames'] = copy.deepcopy(frames)
            elif group == 'startup_receipts':
                record['teacher_profile'] = copy.deepcopy(profile)
                record['coordinate_profile_identity'] = activation
            if group == 'child_native_records':
                record['mapped_elfs'] = [{'name': 'tf_gguf_fixture', 'sha256': '7' * 64}]
                if native_fault:
                    native_fault(record)
            if record_fault:
                record_fault(group, rank, record)
            item = {**_write(root, name, record), 'rank': rank}
            groups[group].append(item)
            authority_groups[group].append({**item, 'file': name.removeprefix('teacher/')})
    authority = {'schema': 'tensorfold.owned-teacher-authority-v1', 'mode': mode, 'method': 'none',
                 'closed': True, 'finished': True, 'failed': False, 'cohort_complete': True, 'launch_exit': 0,
                 'input_sha256': 'd' * 64, 'reference_sha256': reference, 'input_ids': [5, 6], 'vocab': 4,
                 'widths': [1] * 4, 'teacher_ids': teacher_ids, 'frames': frames, 'runtime_sources': sources,
                 'owner_family': {'kernel_family': 'torch-reference', 'graph_family': 'uncaptured'},
                 'declared_identity': {key: 'e' * 64 for key in ('case_id', 'model_sha256', 'frontend_sha256', 'profile_sha256')},
                 'actual_owned_profile': copy.deepcopy(profile),
                 'rank_receipts': authority_groups['rank_receipts'], 'startup_receipts': authority_groups['startup_receipts']}
    if alter:
        alter(authority)
    authority_ref = _write(root, 'teacher/teacher-authority.json', authority)
    client = {'complete': True, 'client_protocol_passed': True, 'post_attempts': 1, 'mode': mode,
              'token_ids': authority['teacher_ids'], 'teacher_plan_sha256': 'd' * 64, 'teacher_client_sha256': 'f' * 64,
              'cached_tokens': 0, 'drafts': False, 'drafted': 0, 'accepted': 0}
    model = {'schema': 'tensorfold.owned-teacher-model-result-v1', 'teacher_plan_sha256': 'd' * 64,
             'teacher_client_sha256': 'f' * 64, 'launch_exit': 0, 'teacher_authority': authority_ref,
             'client_report': _write(root, 'teacher/teacher-client-report.json', client), **groups}
    _write(root, 'teacher/teacher-model-result.json', model)
    operation = {'teacher_plan': {'sha256': 'd' * 64}, 'teacher_client': {'sha256': 'f' * 64},
                 'model_assets': ['same-actual-checkpoint'], 'frontend_assets': ['same-actual-frontend'],
                 'frontend_root': '/frontend/same', 'model_entrypoint': '/models/same', 'execution_manifest': None}
    inner = {'schema': 'tensorfold.owned-teacher-operation-result-v1', 'mode': 'owned-teacher-model-v1',
             'source_commit': ('1' if mode == 'capture' else '2') * 40, 'image_id_declared': image or 'sha256:' + '3' * 64,
             'operation': operation, 'source_after_match': True, 'content_binding_complete': True,
             'content_binding': _write(root, 'model-content-binding.json', {'completed': True}),
             'parent_cuda_uninitialized': True, 'model_result': model, 'primary_error': None, 'post_errors': [],
             'scope': 'synthetic CPU boundary fixture'}
    _write(root, 'owned-teacher-operation-result.json', inner)
    host = {'schema': 'tensorfold.native-contract-host-result-v1', 'report_complete': True, 'source_match': True,
            'health_match': True, 'primary_error': None, 'cleanup_errors': [], 'attach_exit': 0,
            'container': {'Image': inner['image_id_declared'], 'State': {'Status': 'exited', 'Running': False, 'Pid': 0, 'ExitCode': 0, 'OOMKilled': False}},
            'inner': inner}
    if host_fault:
        host_fault(host)
    host_ref = _write(root, 'host-result.json', host)
    completion = {'schema': 'tensorfold.owned-teacher-completion-v1', 'host_result_sha256': host_ref['sha256'],
                  'authority_sha256': authority_ref['sha256'], 'host_cli_exit': 0,
                  'operation_gate_sha256': '4' * 64, 'review_sha256': '5' * 64, 'terminal_observation_sha256': '6' * 64}
    if completion_fault:
        completion_fault(completion)
    item = _write(root, 'external-completion.json', completion)
    return {'root': str(root), 'host_result': host_ref, 'completion': {
        'path': str(root / item['file']), 'bytes': item['bytes'], 'sha256': item['sha256']}}, authority_ref['sha256']


def _pair(tmp_path, **replay_kwargs):
    first, sha = _side(tmp_path / 'reference', 'capture')
    second, _ = _side(tmp_path / 'replay', 'replay', sha, **replay_kwargs)
    return first, second


def test_matched_control_reports_bits_raw_argmax_ties_and_actual_vs_forced(tmp_path):
    first, second = _pair(tmp_path)
    report = quality.compare_owned_teacher(first, second, read_timeout_s=10)
    assert report['status'] == 'EXACT_MATCHED_CONTROL'
    assert len(report['frames']) == 4 and report['quality_acceptance'] is False and report['P3_valid'] is False
    row = report['frames'][0]
    assert row['exact_uint32_bits'] is True and row['different_uint32_values'] == 0 and row['finite'] is True
    assert row['base_top1'] == row['candidate_top1'] == 1
    assert row['reference_raw_top1_ties'] == row['replay_raw_top1_ties'] == 2
    assert row['reference_original_selected'] == row['replay_original_selected'] == 1
    assert row['reference_forced_id'] == row['replay_forced_id'] == 1
    assert row['kl'] == row['nll_delta'] == row['max_abs_logit_delta'] == 0
    assert report['teacher_sources'][0] != report['teacher_sources'][1]
    assert report['operator_commits'][0] != report['operator_commits'][1]
    assert 'passed' not in report and 'gate' not in report


def test_signed_zero_is_bit_difference_without_numerical_margin(tmp_path):
    arrays = [np.array([-0., 1., 1., 0.], dtype='<f4') for _ in range(4)]
    first, second = _pair(tmp_path, arrays=arrays)
    report = quality.compare_owned_teacher(first, second, read_timeout_s=10)
    assert report['status'] == 'NONEXACT_MATCHED_CONTROL_VARIABILITY_REQUIRES_ANALYSIS'
    assert all(row['different_uint32_values'] == 1 and row['max_abs_logit_delta'] == 0 for row in report['frames'])


@pytest.mark.parametrize('field', ['reference_sha256', 'input_ids', 'vocab', 'declared_identity', 'actual_owned_profile', 'runtime_sources', 'frames'])
def test_pair_identity_mismatches_refuse_before_arithmetic(tmp_path, monkeypatch, field):
    def alter(authority):
        if field == 'reference_sha256':
            authority[field] = '0' * 64
        elif field == 'input_ids':
            authority[field] = [7, 8]
        elif field == 'vocab':
            authority[field] = 5
        elif field == 'declared_identity':
            authority[field]['profile_sha256'] = '0' * 64
        elif field == 'actual_owned_profile':
            authority[field]['teacher_semantics'] = 'different'
        elif field == 'runtime_sources':
            authority[field]['decode'] = '0' * 64
        else:
            authority[field][1]['position'] += 1
    first, second = _pair(tmp_path, alter=alter)
    monkeypatch.setattr(quality, '_owned_comparison_vector', lambda *args, **kwargs: pytest.fail('numeric payload before identity validation'))
    with pytest.raises(ValueError):
        quality.compare_owned_teacher(first, second, read_timeout_s=10)


@pytest.mark.parametrize('fault', ['exit', 'host-sha', 'authority-sha'])
def test_external_cli_completion_is_independent_of_attach_success(tmp_path, fault):
    def alter(completion):
        completion[{'exit': 'host_cli_exit', 'host-sha': 'host_result_sha256', 'authority-sha': 'authority_sha256'}[fault]] = 1 if fault == 'exit' else '0' * 64
    first, second = _pair(tmp_path, completion_fault=alter)
    with pytest.raises(ValueError, match='external Host CLI'):
        quality.compare_owned_teacher(first, second, read_timeout_s=10)


@pytest.mark.parametrize('fault', ['container', 'cleanup', 'parent'])
def test_failed_host_or_retirement_never_becomes_comparison_authority(tmp_path, fault):
    def alter(host):
        if fault == 'container':
            host['container']['State']['ExitCode'] = 1
        elif fault == 'cleanup':
            host['cleanup_errors'] = ['retire failed']
        else:
            host['inner']['parent_cuda_uninitialized'] = False
    first, second = _pair(tmp_path, host_fault=alter)
    with pytest.raises(ValueError):
        quality.compare_owned_teacher(first, second, read_timeout_s=10)


def test_changed_reference_vector_and_symlink_escape_are_refused(tmp_path):
    first, second = _pair(tmp_path)
    vector = Path(second['root']) / 'teacher/teacher-rank0/frame-000000.npy'
    vector.write_bytes(vector.read_bytes() + b'changed')
    with pytest.raises(ValueError):
        quality.compare_owned_teacher(first, second, read_timeout_s=10)
    vector.unlink()
    vector.symlink_to(Path(first['root']) / 'teacher/teacher-rank0/frame-000000.npy')
    with pytest.raises(ValueError, match='escapes'):
        quality.compare_owned_teacher(first, second, read_timeout_s=10)


def test_nonfinite_vectors_are_not_summarized_as_quality(tmp_path):
    arrays = [np.array([np.inf, 1., 1., 0.], dtype='<f4') for _ in range(4)]
    first, second = _pair(tmp_path, arrays=arrays)
    with pytest.raises(ValueError, match='finite'):
        quality.compare_owned_teacher(first, second, read_timeout_s=10)


def test_comparison_cli_binds_input_and_preserves_diagnostic_status(tmp_path):
    first, second = _pair(tmp_path)
    data = {'schema': 'tensorfold.owned-teacher-comparison-input-v1', 'reference': first, 'replay': second, 'read_timeout_s': 10}
    path = tmp_path / 'input.json'
    path.write_text(json.dumps(data))
    out = tmp_path / 'result'
    assert quality.main(['--compare-owned-teacher', '--comparison-input', str(path), '--expected-input-sha', _sha(path.read_bytes()), '--output-dir', str(out)]) == 0
    assert json.loads((out / 'comparison.json').read_text())['status'] == 'EXACT_MATCHED_CONTROL'
    with pytest.raises(ValueError, match='SHA'):
        quality.main(['--compare-owned-teacher', '--comparison-input', str(path), '--expected-input-sha', '0' * 64, '--output-dir', str(tmp_path / 'bad')])
    assert not (tmp_path / 'bad').exists()


def test_original_selection_divergence_is_control_variability_even_with_exact_logits(tmp_path):
    def alter(authority):
        authority['frames'][0]['selected_original'] = 2
    first, second = _pair(tmp_path, alter=alter)
    report = quality.compare_owned_teacher(first, second, read_timeout_s=10)
    assert report['exact_uint32_all_frames'] is True
    assert report['original_selection_agreement_all_frames'] is False
    assert report['status'] == 'NONEXACT_MATCHED_CONTROL_VARIABILITY_REQUIRES_ANALYSIS'
    assert report['frames'][0]['replay_original_selected'] == 2
    assert report['frames'][0]['replay_forced_id'] == 1


def test_matched_control_rejects_changed_immutable_image(tmp_path, monkeypatch):
    first, second = _pair(tmp_path, image='sha256:' + '9' * 64)
    monkeypatch.setattr(quality, '_owned_comparison_vector', lambda *args, **kwargs: pytest.fail('payload before image proof'))
    with pytest.raises(ValueError, match='same immutable image'):
        quality.compare_owned_teacher(first, second, read_timeout_s=10)


@pytest.mark.parametrize('fault', ['hash', 'name', 'missing'])
def test_matched_control_rejects_changed_or_missing_actual_native_identity(tmp_path, monkeypatch, fault):
    def alter(record):
        if record['rank'] == 7:
            if fault == 'missing':
                record['mapped_elfs'] = []
            else:
                record['mapped_elfs'][0][{'hash': 'sha256', 'name': 'name'}[fault]] = '9' * 64
    first, second = _pair(tmp_path, native_fault=alter)
    monkeypatch.setattr(quality, '_owned_comparison_vector', lambda *args, **kwargs: pytest.fail('payload before native proof'))
    with pytest.raises(ValueError, match='native'):
        quality.compare_owned_teacher(first, second, read_timeout_s=10)


@pytest.mark.parametrize('fault', ['huge-shape', 'wrong-dtype', 'fortran', 'truncated', 'trailing', 'huge-header'])
def test_actual_npy_preflight_refuses_before_numpy_allocation(tmp_path, monkeypatch, fault):
    import io
    import struct
    stream = io.BytesIO()
    header = {'descr': '<f4', 'fortran_order': False, 'shape': (4,)}
    if fault == 'huge-shape':
        header['shape'] = (2**60,)
    elif fault == 'wrong-dtype':
        header['descr'] = '<f8'
    elif fault == 'fortran':
        header['fortran_order'] = True
    if fault == 'huge-header':
        raw = np.lib.format.magic(2, 0) + struct.pack('<I', 2**31)
    else:
        np.lib.format.write_array_header_1_0(stream, header)
        raw = stream.getvalue() + bytes(16)
        if fault == 'truncated':
            raw = raw[:-1]
        elif fault == 'trailing':
            raw += b'extra'
    path = tmp_path / 'adversarial.npy'
    path.write_bytes(raw)
    vector = {'bytes': len(raw), 'sha256': _sha(raw)}
    monkeypatch.setattr(np, 'load', lambda *args, **kwargs: pytest.fail('allocation before verified actual header/extent'))
    with pytest.raises(ValueError):
        quality._owned_comparison_vector(path, vector, 4, quality._teacher_read_control(10))


def test_numpy_version2_header_and_same_held_payload_are_supported(tmp_path, monkeypatch):
    import io
    stream = io.BytesIO()
    values = np.array([0., -0., 2., 3.], dtype='<f4')
    np.lib.format.write_array_header_2_0(stream, {'descr': '<f4', 'fortran_order': False, 'shape': (4,)})
    raw = stream.getvalue() + values.tobytes()
    path = tmp_path / 'version2.npy'
    path.write_bytes(raw)
    original = np.load
    def load(held, **kwargs):
        # A subsequent on-disk mutation cannot change the already verified,
        # preflighted byte owner passed to NumPy allocation/decoding.
        path.write_bytes(b'changed after owned read')
        return original(held, **kwargs)
    monkeypatch.setattr(np, 'load', load)
    actual = quality._owned_comparison_vector(path, {'bytes': len(raw), 'sha256': _sha(raw)}, 4,
                                              quality._teacher_read_control(10))
    assert np.array_equal(actual.view('<u4'), values.view('<u4'))


@pytest.mark.parametrize('seconds', [True, False, 0, -1, float('inf'), float('-inf'), float('nan'), '10', None, sys.maxsize + 1, 10**400])
def test_comparison_deadline_domain_precedes_clock_and_io(monkeypatch, seconds):
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == 'numpy':
            pytest.fail('NumPy import before deadline validation')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    monkeypatch.setattr(quality, '_teacher_read_control', lambda value: pytest.fail('deadline arithmetic before validation'))
    monkeypatch.setattr(quality, '_owned_comparison_side', lambda *args: pytest.fail('IO before deadline validation'))
    with pytest.raises(ValueError, match='positive finite'):
        quality.compare_owned_teacher({}, {}, read_timeout_s=seconds)


def test_distinct_resident_activations_normalize_only_after_complete_within_run_linkage(tmp_path):
    first, second = _pair(tmp_path)
    report = quality.compare_owned_teacher(first, second, read_timeout_s=10)
    assert report['status'] == 'EXACT_MATCHED_CONTROL'
    assert report['coordinate_activations'] == ['8' * 64, '9' * 64]
    assert 'typed resident-activation' in report['coordinate_scope']


@pytest.mark.parametrize('location', ['summary-profile', 'startup-top', 'startup-profile', 'summary-frame', 'authority-profile', 'authority-frame'])
def test_mismatched_activation_within_one_run_is_refused_before_numeric_payload(tmp_path, monkeypatch, location):
    def records(group, rank, record):
        if rank != 7:
            return
        if location == 'summary-profile' and group == 'rank_receipts':
            record['actual_owned_profile']['coordinate_profile_identity'] = '0' * 64
        elif location == 'startup-top' and group == 'startup_receipts':
            record['coordinate_profile_identity'] = '0' * 64
        elif location == 'startup-profile' and group == 'startup_receipts':
            record['teacher_profile']['teacher_semantics']['state_kv_identity'][2][1] = '0' * 64
        elif location == 'summary-frame' and group == 'rank_receipts':
            record['frames'][0]['kv_identity'][2][1] = '0' * 64
    def authority(value):
        if location == 'authority-profile':
            value['actual_owned_profile']['coordinate_profile_identity'] = '0' * 64
        elif location == 'authority-frame':
            value['frames'][0]['kv_identity'][2][1] = '0' * 64
    first, second = _pair(tmp_path, alter=authority, record_fault=records)
    monkeypatch.setattr(quality, '_owned_comparison_vector', lambda *args: pytest.fail('numeric payload before activation proof'))
    with pytest.raises(ValueError, match='activation|coordinate|State'):
        quality.compare_owned_teacher(first, second, read_timeout_s=10)


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'malformed', 'wrong-hash', 'position', 'context', 'other-marker'])
def test_only_one_typed_activation_marker_is_normalized_and_other_kv_fields_stay_exact(tmp_path, monkeypatch, fault):
    def change(identity):
        if fault == 'missing':
            identity.pop(2)
        elif fault == 'duplicate':
            identity.append(copy.deepcopy(identity[2]))
        elif fault == 'malformed':
            identity[2].append('extra')
        elif fault == 'wrong-hash':
            identity[2][1] = '0' * 64
        elif fault == 'position':
            identity[1][2] = 1
        elif fault == 'context':
            identity[3][1] = 131072
        else:
            identity.append(['another-stable-marker', '9' * 64])
    first, second = _pair(tmp_path, state_change=change)
    monkeypatch.setattr(quality, '_owned_comparison_vector', lambda *args: pytest.fail('numeric payload before stable KV proof'))
    with pytest.raises(ValueError, match='marker|State|activation'):
        quality.compare_owned_teacher(first, second, read_timeout_s=10)


@pytest.mark.parametrize('activation', ['not-a-sha', False, 123])
def test_actual_activation_identity_requires_sha256_before_normalization(tmp_path, activation):
    first, second = _pair(tmp_path, activation=activation)
    with pytest.raises(ValueError, match='actual activation SHA'):
        quality.compare_owned_teacher(first, second, read_timeout_s=10)
