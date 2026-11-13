"""Synthetic sharded-row controls; fixtures are never native/model evidence."""
import copy
import hashlib
import json
import os
import weakref

import numpy as np
import pytest

from tests.test_v100_target_window_teacher import plan as forced_plan
from tools import v100_rotorquant_quality as quality
from tools import v100_target_window_evidence as evidence
from tools import v100_target_window_teacher as window


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=1)+'\n').encode()
    path.write_bytes(raw)
    return {'file': path.name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def _vector(path, score, *, shape=None):
    values = np.zeros(248320, dtype='<f4')
    values[7] = score
    if shape:
        values = values.reshape(shape)
    np.save(path, values, allow_pickle=False)
    raw = path.read_bytes()
    return {'file': path.name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
            'dtype': '<f4', 'shape': [248320]}


def fixture(root, *, score=1, reverse=False, profile='3', confidence=.7, slot_offset=0, rope_delta=0, raw_probability=.25):
    # Four forced outputs allow a real cap1 C8 refresh after the first keep;
    # the old three-output fixture had cap0 and could not claim that graph hit.
    plan = forced_plan(4)
    plan['proposal_policy'] = 'forced-gold-cap-aware-full-depth-v1'
    plan['mtp_confidence'] = confidence
    plan['identity']['profile_sha256'] = profile*64
    raw = json.dumps(plan).encode()
    prepared = {'data': plan, 'input_sha256': hashlib.sha256(raw).hexdigest(), 'resources': window.resources(len(raw), plan)}
    limits, input_sha = prepared['resources'], prepared['input_sha256']
    root.mkdir()
    sources = {key: hashlib.sha256(key.encode()).hexdigest() for key in evidence.SOURCE_KEYS}
    cases = {case['case_id']: case for case in plan['sequences']}
    case_order = list(cases)
    packed = list(reversed(case_order)) if reverse else case_order
    def snapshot(name, progress, *, admitted=False):
        case = cases[name]
        index, prompt = case_order.index(name), len(case['prompt_ids'])
        cap = max(0, min(4, len(case['forced_ids'])-progress-1))
        tentative = max(cap-1, 0) if progress else 0
        mtp_len = (prompt-1 if progress == 0 else prompt+progress-1+tentative)
        if progress == len(case['forced_ids']):
            mtp_len = prompt+progress-2  # The final window never refreshes MTP.
        return {'case_id': name, 'sid': index+1, 'history_slot': index+slot_offset, 'slot_generation': 1,
                'state_pos': 0 if admitted else prompt+max(0, progress-1),
                'mtp_len': 0 if admitted else mtp_len, 'mtp_drafted': 0 if admitted else tentative,
                'output_tokens': progress,
                'draft_tokens': [] if admitted or progress == 0 or progress == len(case['forced_ids'])
                                else case['forced_ids'][progress:progress+cap],
                'done': progress == len(case['forced_ids']),
                'kv_identity': [['logical-context-v1', 262144], ['coordinate-profile-v1', 'a'*64]]}
    def frame(name, number, index, local, rows, progress, packed_row, batch_rows, rank):
        case = cases[name]
        prefill = index is None
        position = len(case['prompt_ids'])+(0 if prefill else progress+local)
        row = {**snapshot(name, 0 if prefill else progress), 'frame': number, 'window': index,
            'role': 'prefill' if prefill else 'target-verify', 'absolute_position': position,
            'gold_token': case['forced_ids'][position-len(case['prompt_ids'])], 'window_row': local,
            'target_window_rows': rows, 'active_states': 1 if prefill else 8, 'physical_target_rows': 120,
            'vocab': 248320, 'vector': None, 'method': 'mtp', 'depth': 4, 'teacher_forced': True,
            'original_sampled': 7, 'packed_row': packed_row, 'batch_target_rows': batch_rows,
            'input_token': case['prompt_ids'][-1] if prefill else case['forced_ids'][progress-1+local],
            'input_position': position-1, 'rope_delta': rope_delta, 'input_coordinate': position-1+rope_delta}
        if not prefill:
            row['draft_tokens'] = list(case['forced_ids'][progress:progress+rows-1])
        if rank == 0:
            row['vector'] = _vector(root/f'window-teacher-rank{rank}'/f'frame-{number:06d}.npy',
                                    score+(0 if prefill else 2*index+local+1))
        return row
    for rank in range(8):
        directory = root/f'window-teacher-rank{rank}'
        directory.mkdir()
        pid = 1000+rank
        stages, picks = [], []
        def stage(phase, kind, lanes, *, stages=stages, rank=rank, pid=pid, directory=directory):
            at, values = 0, []
            for name, start, count in lanes:
                seq = cases[name]['prompt_ids']+cases[name]['forced_ids']
                state = snapshot(name, 0, admitted=True)
                values.append({'case_id': name, **{key: state[key] for key in ('sid', 'history_slot', 'slot_generation')},
                    'cache_start': start-1, 'coordinate_start': start-1+rope_delta, 'rope_delta': rope_delta,
                    'token_position_start': start, 'token_ids': seq[start:start+count],
                    'packed_first': at, 'packed_last': at+count})
                at += count
            index = len(stages)
            value = {'schema': 'tensorfold.target-window-teacher-mtp-stage-v1', 'rank': rank,
                'worker_PID': pid, 'input_sha256': input_sha, 'stage_index': index, 'window': phase,
                'participants': len(values), 'rows': at, 'lanes': values, 'kind': kind}
            stages.append(_write(directory/f'mtp-stage-{index:06d}.json', value))
            return index, value
        def pick(phase, supplying, cap, proposal_index, *, picks=picks, rank=rank, pid=pid, directory=directory):
            stage_index, source = supplying
            index, rows = len(picks), []
            for lane in source['lanes']:
                position = lane['token_position_start']+len(lane['token_ids'])
                case = cases[lane['case_id']]
                gold = case['forced_ids'][position-len(case['prompt_ids'])]
                rows.append({key: lane[key] for key in ('case_id', 'sid', 'history_slot', 'slot_generation')} | {
                    'absolute_position': position, 'gold_token': gold, 'original_token': 7 if score == 1 else 8,
                    'original_probability': None if phase is None and confidence == 0 else raw_probability,
                    'forced_token': gold, 'forced_probability': 1.0, 'continuation_policy': plan['proposal_policy'],
                    'proposal_cap': cap, 'proposal_index': proposal_index})
            value = {'schema': 'tensorfold.target-window-teacher-mtp-pick-v1', 'rank': rank,
                'worker_PID': pid, 'input_sha256': input_sha, 'pick_index': index, 'window': phase,
                'branch': ('serial-sample' if confidence == 0 else 'serial-sample-draft') if phase is None else 'multi-picks',
                'source_kind': 'prefill' if phase is None else 'mtp', 'stage_index': stage_index, 'rows': rows}
            picks.append(_write(directory/f'mtp-pick-{index:06d}.json', value))
        profile_record = {'rank': rank, 'world': 8, 'target_banks': 48, 'mtp_banks': 1, 'capability': [7, 0],
                          'fixture_only': True, 'profile_sha256': plan['identity']['profile_sha256']}
        _write(root/f'rank-{rank}.json', {'schema': 'tensorfold.private-owned-qualification-v1',
            'rank': rank, 'world': 8, 'target_banks': 48, 'mtp_banks': 1, 'teacher_worker_PID': pid,
            'teacher_profile': profile_record, 'history_host_obligation': {
                'global_bytes': limits['global_host_obligation_bytes'], 'input_sha256': input_sha}})
        _write(directory/'setup.json', {'schema': 'tensorfold.target-window-teacher-setup-v1',
            'input_sha256': input_sha, 'actual_profile': profile_record, 'resources': limits,
            'writer': 'existing-owned-full-vocabulary-npy-writer', 'qualification': False,
            'P3_valid': False, 'runtime_sources': sources})
        prefills, windows = [], []
        for index, name in enumerate(case_order):
            _write(directory/f'admission-{index}.json', {'schema': 'tensorfold.target-window-teacher-admission-v1',
                'index': index, 'case_id': name, 'worker_PID': pid, 'rank': rank,
                'input_sha256': input_sha, 'state': snapshot(name, 0, admitted=True)})
            prompt = len(cases[name]['prompt_ids'])
            stage(None, 'prefill', [(name, 1, prompt-1)])
            prefills.append(_write(directory/f'prefill-{index}.json', frame(name, index, None, 0, 1, 0, 0, 1, rank)))
            # Real initial cap2: pending-anchor refresh, then one forced proposal input.
            for proposal_index in range(2):
                supplying = stage(None, 'mtp', [(name, prompt+proposal_index, 1)])
                pick(None, supplying, 2, proposal_index)
        progress = dict.fromkeys(case_order, 1)
        counts = {'serving_hits': {'verify': 0, 'mtp': 0}, 'serving_misses': {'verify': 0, 'mtp': 0}}
        initial = copy.deepcopy(counts)
        number = 8
        for index, schedule in enumerate(plan['rounds']):
            before = {name: snapshot(name, progress[name]) for name in case_order}
            lanes = {row['case_id']: row for row in schedule['lanes']}
            rows, acceptance = [], []
            batch_rows = sum(row['rows'] for row in schedule['lanes'])
            offset = 0
            for name in packed:
                lane = lanes[name]
                for local in range(lane['rows']):
                    rows.append(frame(name, number, index, local, lane['rows'], progress[name], offset, batch_rows, rank))
                    number += 1
                    offset += 1
                acceptance.append({'case_id': name, 'original_path': list(range(lane['rows'])), 'original_terminal': 7,
                    'forced_path': list(range(lane['keep'])), 'forced_terminal': 7,
                    'role': 'diagnostic-forced-keep-not-serving-acceptance',
                    'original_scope': 'counterfactual-forced-proposals-and-prefix-v1'})
            before_counts = copy.deepcopy(counts)
            slots = [before[name]['history_slot'] for name in packed]
            generations = [before[name]['slot_generation'] for name in packed]
            final = index == len(plan['rounds'])-1
            calls = [{'role': 'verify', 'participants': 8, 'rows': batch_rows,
                'serving_hits_delta': int(final), 'serving_misses_delta': int(not final),
                'history_slots': slots, 'slot_generations': generations}]
            counts['serving_hits' if final else 'serving_misses']['verify'] += 1
            if not final:
                supplying = stage(index, 'mtp', [(name, len(cases[name]['prompt_ids'])+progress[name], 1) for name in packed])
                if index == 0:
                    pick(index, supplying, 1, 0)
                    calls.append({'role': 'mtp', 'participants': 8, 'rows': 8,
                        'serving_hits_delta': 1, 'serving_misses_delta': 0,
                        'history_slots': slots, 'slot_generations': generations})
                    counts['serving_hits']['mtp'] += 1
            for lane in schedule['lanes']:
                progress[lane['case_id']] += lane['keep']
            after = {name: snapshot(name, progress[name]) for name in case_order}
            delta = {kind: {role: counts[kind][role]-before_counts[kind][role] for role in ('verify', 'mtp')}
                     for kind in counts}
            value = {'schema': 'tensorfold.target-window-teacher-window-v1', 'rank': rank, 'worker_PID': pid,
                'window': index, 'input_sha256': input_sha, 'schedule': schedule, 'before': before, 'after': after,
                'frames': rows, 'acceptance': acceptance, 'graph_counts_before': before_counts,
                'graph_counts_after': copy.deepcopy(counts), 'graph_deltas': delta, 'graph_calls': calls,
                'method': 'mtp', 'depth': 4, 'active_states': 8, 'role': 'forced-target-window',
                'qualification': False, 'P3_valid': False}
            windows.append(_write(directory/f'window-{index:06d}.json', value))
        _write(directory/'summary.json', {'schema': 'tensorfold.target-window-teacher-rank-result-v1',
            'rank': rank, 'worker_PID': pid, 'input_sha256': input_sha, 'closed': True,
            'finished': sorted(cases), 'failed': False, 'profile': profile_record, 'prefill': prefills,
            'windows': windows, 'frame_count': number, 'initial_graph_counts': initial,
            'final_graph_counts': counts, 'resources': limits, 'quality_scope': plan['quality_scope'],
            'metrics': plan['metrics'], 'qualification': False, 'P3_valid': False, 'runtime_sources': sources,
            'mtp_stages': stages, 'mtp_picks': picks})
    return prepared


def test_sharded_authority_and_metrics_keep_overlap_without_corpus_double_weight(tmp_path, monkeypatch):
    left, right = tmp_path/'left', tmp_path/'right'
    prepared_left = fixture(left)
    prepared_right = fixture(right, score=2, reverse=True, profile='4')
    live, fd_counts = [], []
    original = quality._owned_comparison_vector
    def observed(*args, **kwargs):
        value = original(*args, **kwargs)
        live.append(weakref.ref(value))
        assert sum(ref() is not None for ref in live) <= 3
        fd_counts.append(len(os.listdir('/proc/self/fd')))
        return value
    monkeypatch.setattr(quality, '_owned_comparison_vector', observed)
    first = evidence.seal(left, prepared_left, 0)
    second = evidence.seal(right, prepared_right, 0)
    report = evidence.compare_rows(left, right, first, second, control=quality._teacher_read_control(60),
                                   output=tmp_path/'metrics')
    assert report['raw_row_count'] == 56 and report['unique_corpus_positions'] == 32
    assert report['corpus']['rows'] == 32 and report['all_window_rows']['rows'] == 56
    assert report['prefill_singleton_windows']['rows'] == 8
    assert report['mtp_forced_input_comparison']['actual_input_coordinate_batches'] == 26
    assert report['mtp_forced_input_comparison']['original_sample_observations'] == 24
    assert report['mtp_forced_input_comparison']['original_token_differences'] == 24
    assert report['corpus']['ppl_ratio'] < 1 and not report['quality_acceptance'] and not report['P3_valid']
    last = json.loads((tmp_path/'metrics/window-000001-metrics.json').read_text())
    assert all(not row['first_corpus_observation'] for row in last['rows'])
    assert all('overlap_first_observation_comparison' in row for row in last['rows'])
    assert any(row['reference_row']['packed_row'] != row['candidate_row']['packed_row'] for row in last['rows'])
    assert max(fd_counts)-min(fd_counts) <= 1  # Reused readers close each descriptor before advancing.
    assert sum(ref() is not None for ref in live) == 0
    assert not (left/'window-teacher-rank0').stat().st_mode & 0o222


def _rewrite_window(root, rank, index, change):
    directory = root/f'window-teacher-rank{rank}'
    path = directory/f'window-{index:06d}.json'
    row = json.loads(path.read_text())
    change(row)
    ref = _write(path, row)
    summary_path = directory/'summary.json'
    summary = json.loads(summary_path.read_text())
    summary['windows'][index] = ref
    _write(summary_path, summary)


def _rewrite_mtp(root, rank, kind, index, change):
    directory = root/f'window-teacher-rank{rank}'
    path = directory/f'mtp-{kind}-{index:06d}.json'
    row = json.loads(path.read_text())
    change(row)
    ref = _write(path, row)
    path = directory/'summary.json'
    summary = json.loads(path.read_text())
    summary['mtp_stages' if kind == 'stage' else 'mtp_picks'][index] = ref
    _write(path, summary)


@pytest.mark.parametrize('fault', ['state_generation', 'missing_tail', 'c1_mtp_graph', 'source_hash',
                                  'unknown_file', 'duplicate_pid', 'header', 'frame_boolean', 'forced_path_boolean', 'original_acceptance',
                                  'target_input_token', 'mtp_stage_token', 'mtp_stage_coordinate', 'mtp_pick_forced',
                                  'mtp_pick_probability', 'mtp_pick_cap', 'mtp_pick_stage', 'mtp_continuation', 'mtp_truncated'])
def test_seal_refuses_partial_forged_or_ambiguous_shards_before_authority(tmp_path, fault):
    root = tmp_path/'teacher'
    prepared = fixture(root)
    if fault == 'state_generation':
        _rewrite_window(root, 0, 1, lambda row: row['before']['case_0'].__setitem__('slot_generation', 2))
    elif fault == 'frame_boolean':
        _rewrite_window(root, 0, 0, lambda row: row['frames'][0].__setitem__('window', False))
    elif fault == 'forced_path_boolean':
        _rewrite_window(root, 0, 0, lambda row: row['acceptance'][0].__setitem__('forced_path', [False]))
    elif fault == 'original_acceptance':
        _rewrite_window(root, 0, 0, lambda row: row['acceptance'][0].__setitem__('original_path', [0]))
    elif fault == 'target_input_token':
        _rewrite_window(root, 0, 0, lambda row: row['frames'][0].__setitem__('input_token', 8))
    elif fault.startswith('mtp_'):
        if fault == 'mtp_stage_token':
            _rewrite_mtp(root, 0, 'stage', 0, lambda row: row['lanes'][0]['token_ids'].__setitem__(0, 9))
        elif fault == 'mtp_stage_coordinate':
            _rewrite_mtp(root, 0, 'stage', 0, lambda row: row['lanes'][0].__setitem__('coordinate_start', 1))
        elif fault == 'mtp_pick_forced':
            _rewrite_mtp(root, 0, 'pick', 0, lambda row: row['rows'][0].__setitem__('forced_token', 8))
        elif fault == 'mtp_pick_probability':
            _rewrite_mtp(root, 0, 'pick', 0, lambda row: row['rows'][0].__setitem__('original_probability', None))
        elif fault == 'mtp_pick_cap':
            _rewrite_mtp(root, 0, 'pick', 0, lambda row: row['rows'][0].__setitem__('proposal_cap', 1))
        elif fault == 'mtp_pick_stage':
            _rewrite_mtp(root, 0, 'pick', 0, lambda row: row.__setitem__('stage_index', 0))
        elif fault == 'mtp_continuation':
            _rewrite_mtp(root, 0, 'pick', 0, lambda row: row['rows'][0].__setitem__('continuation_policy', 'ordinary'))
        elif fault == 'mtp_truncated':
            path = root/'window-teacher-rank0/summary.json'
            summary = json.loads(path.read_text())
            summary['mtp_picks'].pop()
            _write(path, summary)
    elif fault == 'missing_tail':
        _rewrite_window(root, 0, 0, lambda row: row['frames'].pop())
    elif fault == 'c1_mtp_graph':
        def change(row):
            call = row['graph_calls'][1]
            call.update(participants=1, rows=1, history_slots=[0], slot_generations=[1])
        _rewrite_window(root, 0, 0, change)
    elif fault in ('source_hash', 'duplicate_pid'):
        path = root/'window-teacher-rank1/summary.json'
        summary = json.loads(path.read_text())
        if fault == 'source_hash':
            summary['runtime_sources']['window_teacher'] = '0'*64
        else:
            summary['worker_PID'] = 1000
        _write(path, summary)
    elif fault == 'unknown_file':
        (root/'window-teacher-rank0/unbound.txt').write_text('not part of the observed closure')
    elif fault == 'header':
        directory = root/'window-teacher-rank0'
        vector = _vector(directory/'frame-000000.npy', 1, shape=(124160, 2))
        path = directory/'prefill-0.json'
        row = json.loads(path.read_text())
        row['vector'] = vector
        ref = _write(path, row)
        path = directory/'summary.json'
        summary = json.loads(path.read_text())
        summary['prefill'][0] = ref
        _write(path, summary)
    with pytest.raises(ValueError):
        evidence.seal(root, prepared, 0)
    assert not (root/'teacher-authority.json').exists()


def test_sharded_model_binding_rechecks_actual_vectors_after_publication(tmp_path):
    outer = tmp_path/'evidence'
    outer.mkdir()
    root = outer/'teacher'
    prepared = fixture(root)
    authority = evidence.seal(root, prepared, 0)
    raw = (root/'teacher-authority.json').read_bytes()
    model = {'teacher_plan_sha256': prepared['input_sha256'], 'teacher_client_sha256': 'b'*64,
        'teacher_authority': {'file': 'teacher/teacher-authority.json', 'bytes': len(raw),
                              'sha256': hashlib.sha256(raw).hexdigest()},
        **{key: [{**ref, 'file': 'teacher/'+ref['file']} for ref in authority[key]]
           for key in ('rank_receipts', 'startup_receipts')}}
    operation = {'teacher_plan': {'sha256': prepared['input_sha256']}, 'teacher_client': {'sha256': 'b'*64}}
    assert evidence.validate_artifacts(outer, model, operation, plan=prepared['data']) == authority
    path = root/'window-teacher-rank0/frame-000000.npy'
    path.chmod(0o644)
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    with pytest.raises(ValueError, match='identity'):
        evidence.validate_artifacts(outer, model, operation, plan=prepared['data'])


def test_failed_original_launcher_cannot_publish_a_sharded_authority(tmp_path):
    root = tmp_path/'teacher'
    prepared = fixture(root)
    with pytest.raises(ValueError, match='launcher'):
        evidence.seal(root, prepared, 1)
    assert not (root/'teacher-authority.json').exists()


@pytest.mark.parametrize('difference', ['slot', 'coordinate'])
def test_different_actual_mtp_input_coordinates_or_slot_prefix_cannot_be_paired(tmp_path, difference):
    left, right = tmp_path/'reference', tmp_path/'candidate'
    a = fixture(left)
    b = fixture(right, slot_offset=8 if difference == 'slot' else 0,
                rope_delta=1 if difference == 'coordinate' else 0)
    first, second = evidence.seal(left, a, 0), evidence.seal(right, b, 0)
    with pytest.raises(ValueError, match='MTP input'):
        evidence.compare_rows(left, right, first, second, control=quality._teacher_read_control(60),
                              output=tmp_path/'comparison')


def test_zero_confidence_records_probability_free_serial_branch_without_changing_gold_continuation(tmp_path):
    root = tmp_path/'teacher'
    prepared = fixture(root, confidence=0)
    authority = evidence.seal(root, prepared, 0)
    summary = json.loads((root/authority['rank_receipts'][0]['file']).read_text())
    serial = json.loads((root/'window-teacher-rank0'/summary['mtp_picks'][0]['file']).read_text())
    assert serial['branch'] == 'serial-sample'
    assert serial['rows'][0]['original_probability'] is None
    assert serial['rows'][0]['forced_probability'] == 1.0
    assert authority['mtp_records_per_rank'] == {'stages': 26, 'picks': 17}


def test_raw_original_probability_signal_is_preserved_without_clamp_or_quality_admission(tmp_path):
    left, right = tmp_path/'reference', tmp_path/'candidate'
    a = fixture(left)
    b = fixture(right, raw_probability=1.25)
    first, second = evidence.seal(left, a, 0), evidence.seal(right, b, 0)
    report = evidence.compare_rows(left, right, first, second, control=quality._teacher_read_control(60),
                                   output=tmp_path/'comparison')
    diagnostic = report['mtp_forced_input_comparison']
    assert diagnostic['original_probability_outside_unit_interval'] == {'reference': 0, 'candidate': 24}
    assert diagnostic['original_probability_status'] == 'OUTSIDE_UNIT_INTERVAL_OBSERVED'
    assert report['quality_acceptance'] is False and report['P3_valid'] is False
    shard = json.loads((tmp_path/'comparison'/diagnostic['pick_shards'][0]['file']).read_text())
    assert shard['rows'][0]['candidate_original_probability'] == 1.25
