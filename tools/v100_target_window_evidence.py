"""Bounded sharded evidence for the production C8/MTP forced-window teacher.

The owning v4 contract must separately validate Host/CLI retirement, immutable
source/image/SDK, retained numerical providers and kernel VMAs, NUMA/CAP40 and
profile selection. This module binds the actual row/shard closure and reuses
the established full-vocabulary reader and F64 arithmetic. It does not grant
quality acceptance, prove held-out accuracy, or supply performance evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import sys
from pathlib import Path

if __package__:
    from . import v100_rotorquant_quality as quality
else:
    import v100_rotorquant_quality as quality

AUTHORITY_SCHEMA = 'tensorfold.target-window-teacher-authority-v1'
SHARD_LIMIT = 8 << 20
ROW_LIMIT = 4096
METRIC_SHARD_LIMIT = 1 << 20
SOURCE_KEYS = {'engine', 'multi', 'decode', 'sampling', 'teacher', 'window_teacher'}
SNAPSHOT_KEYS = {'case_id', 'sid', 'history_slot', 'slot_generation', 'state_pos', 'mtp_len',
                 'mtp_drafted', 'output_tokens', 'kv_identity', 'draft_tokens', 'done'}
FRAME_KEYS = SNAPSHOT_KEYS | {'frame', 'window', 'role', 'absolute_position', 'gold_token', 'window_row',
    'target_window_rows', 'active_states', 'physical_target_rows', 'vocab', 'vector', 'method', 'depth',
    'teacher_forced', 'original_sampled', 'packed_row', 'batch_target_rows',
    'input_token', 'input_position', 'rope_delta', 'input_coordinate'}
SUMMARY_KEYS = {'schema', 'rank', 'worker_PID', 'input_sha256', 'closed', 'finished', 'failed', 'profile',
    'prefill', 'windows', 'frame_count', 'initial_graph_counts', 'final_graph_counts', 'resources',
    'quality_scope', 'metrics', 'qualification', 'P3_valid', 'runtime_sources', 'mtp_stages', 'mtp_picks'}
WINDOW_KEYS = {'schema', 'rank', 'worker_PID', 'window', 'input_sha256', 'schedule', 'before', 'after',
    'frames', 'acceptance', 'graph_counts_before', 'graph_counts_after', 'graph_deltas', 'graph_calls',
    'method', 'depth', 'active_states', 'role', 'qualification', 'P3_valid'}


def _window():
    if __package__:
        from . import v100_target_window_teacher
    else:
        import v100_target_window_teacher
    return v100_target_window_teacher


def _require(value, message):
    if not value:
        raise ValueError(message)


def _sha(value):
    return type(value) is str and quality.SHA.fullmatch(value) is not None


def _integer(value, *, minimum=0):
    return type(value) is int and minimum <= value < 2**63


def _signed(value):
    return type(value) is int and -(2**63) <= value < 2**63


def _path(root, name):
    _require(type(name) is str and name and not Path(name).is_absolute()
             and '..' not in Path(name).parts and str(Path(name)) == name,
             'ordinary relative evidence path required')
    root = Path(root)
    path = root/name
    _require(not root.is_symlink() and path.resolve(strict=True).is_relative_to(root.resolve(strict=True)),
             'evidence path escapes its actual root')
    for parent in path.parents:
        if parent == root:
            break
        _require(not parent.is_symlink(), 'evidence directory symlink refused')
    return path


def _ref(root, path, maximum, control, *, rank=None, freeze=False):
    path = Path(path)
    name = str(path.relative_to(root))
    _require(_path(root, name) == path and not path.is_symlink(), 'ordinary evidence file required')
    info = path.stat(follow_symlinks=False)
    _require(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= maximum, 'evidence extent/type exceeds its bound')
    if freeze:
        # All producer processes have positively retired. Only these owned
        # evidence files are made read-only; no model/source assets are changed.
        _require(info.st_uid == os.getuid(), 'sealing requires actual evidence ownership')
        path.chmod(stat.S_IMODE(info.st_mode) & ~0o222)
    result = {'file': name, 'bytes': info.st_size,
              'sha256': quality._owned_teacher_hash(path, info.st_size, maximum, control)}
    if rank is not None:
        result['rank'] = rank
    return result


def _read(root, ref, maximum, control):
    _path(root, ref.get('file'))
    return quality._owned_comparison_file(Path(root), ref, control, maximum=maximum)


def _local_ref(root, directory, ref, maximum, control, *, expected, freeze=False):
    _require(type(ref) is dict and set(ref) == {'file', 'bytes', 'sha256'} and ref['file'] == expected
             and _integer(ref['bytes'], minimum=1) and _sha(ref['sha256']),
             'fixed shard reference differs')
    path = _path(root, str(Path(directory)/expected))
    actual = _ref(root, path, maximum, control, freeze=freeze)
    _require({k: actual[k] for k in ('bytes', 'sha256')} == {k: ref[k] for k in ('bytes', 'sha256')},
             'shard identity changed')
    return actual


def _snapshot(row, case, *, progress=None, identity=None):
    _require(type(row) is dict and set(row) == SNAPSHOT_KEYS and row['case_id'] == case['case_id']
             and all(_integer(row[k]) for k in SNAPSHOT_KEYS - {'case_id', 'kv_identity', 'draft_tokens', 'done'})
             and type(row['kv_identity']) is list and row['kv_identity']
             and type(row['draft_tokens']) is list and len(row['draft_tokens']) <= 4
             and all(_integer(token) and token < 248320 for token in row['draft_tokens'])
             and type(row['done']) is bool, 'complete actual State snapshot required')
    if progress is not None:
        _require(row['output_tokens'] == progress and row['state_pos'] == len(case['prompt_ids'])+progress-1,
                 'State snapshot differs from the exact committed forced prefix')
    if identity is not None:
        _require(all(row[k] == identity[k] for k in ('sid', 'history_slot', 'slot_generation', 'kv_identity')),
                 'State ownership/generation changed across a forced transaction')
    return row


def _counts(value):
    _require(type(value) is dict and set(value) == {'serving_hits', 'serving_misses'}, 'closed graph-count schema required')
    for row in value.values():
        _require(type(row) is dict and set(row) == {'verify', 'mtp'} and all(_integer(v) for v in row.values()),
                 'complete nonnegative per-role graph counts required')
    return value


def _frame(row, case, *, frame, window, local_row, rows, progress, rank, before, batch_rows):
    _require(type(row) is dict and set(row) == FRAME_KEYS and len(json.dumps(row, ensure_ascii=True).encode()) <= ROW_LIMIT,
             'bounded complete raw row metadata required')
    prefill = window is None
    position = len(case['prompt_ids']) + (0 if prefill else progress+local_row)
    _require(row['frame'] == frame and type(row['frame']) is int and row['window'] == window
             and (row['window'] is None if prefill else type(row['window']) is int)
             and row['role'] == ('prefill' if prefill else 'target-verify')
             and row['absolute_position'] == position and type(row['absolute_position']) is int
             and row['gold_token'] == case['forced_ids'][position-len(case['prompt_ids'])]
             and type(row['gold_token']) is int and row['window_row'] == local_row
             and type(row['window_row']) is int and row['target_window_rows'] == rows
             and type(row['target_window_rows']) is int and row['active_states'] == (1 if prefill else 8)
             and type(row['active_states']) is int and row['physical_target_rows'] == 120
             and type(row['physical_target_rows']) is int and row['vocab'] == 248320
             and type(row['vocab']) is int and row['method'] == 'mtp' and row['depth'] == 4
             and type(row['depth']) is int and row['teacher_forced'] is True and row['done'] is False
             and _integer(row['original_sampled']) and row['original_sampled'] < 248320
             and _integer(row['packed_row']) and row['packed_row'] < batch_rows
             and row['batch_target_rows'] == batch_rows and type(row['batch_target_rows']) is int
             and _integer(row['input_position']) and row['input_position'] == position-1
             and _signed(row['rope_delta']) and _signed(row['input_coordinate'])
             and row['input_coordinate'] == row['input_position']+row['rope_delta']
             and _integer(row['input_token'])
             and row['input_token'] == (case['prompt_ids'][-1] if prefill else case['forced_ids'][progress-1+local_row]),
             'raw row does not bind its declared full-vocabulary forced window')
    snap = {key: row[key] for key in SNAPSHOT_KEYS}
    if prefill:
        _snapshot(snap, case, identity=before)
        _require(snap['output_tokens'] == 0 and snap['state_pos'] == len(case['prompt_ids'])
                 and row['packed_row'] == 0 and not row['draft_tokens'], 'prefill frame differs from its original pending token')
    else:
        _snapshot(snap, case, progress=progress, identity=before)
        _require(row['draft_tokens'] == case['forced_ids'][progress:progress+rows-1],
                 'scored target row lost its exact forced proposal inputs')
    vector = row['vector']
    if rank == 0:
        _require(type(vector) is dict and set(vector) == {'file', 'bytes', 'sha256', 'dtype', 'shape'}
                 and vector['file'] == f'frame-{frame:06d}.npy' and vector['dtype'] == '<f4'
                 and type(vector['shape']) is list and vector['shape'] == [248320]
                 and type(vector['shape'][0]) is int and type(vector['bytes']) is int
                 and 4*248320 < vector['bytes'] <= 4*248320+4096 and _sha(vector['sha256']),
                 'rank0 complete raw Float32 vocabulary vector required')
    else:
        _require(vector is None, 'only rank0 publishes the gathered vocabulary vector')
    return row


def _vector(root, directory, row, control, *, freeze=False):
    vector = row['vector']
    ref = _local_ref(root, directory, {k: vector[k] for k in ('file', 'bytes', 'sha256')},
                     4*248320+4096, control, expected=f"frame-{row['frame']:06d}.npy", freeze=freeze)
    # NumPy owns its grammar; this existing reader validates the actual header,
    # complete finite F32 payload and extent. The array is not retained.
    array = quality._owned_comparison_vector(_path(root, ref['file']), vector, 248320, control)
    del array
    return ref


def _canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _window_record(value, plan, rank, pid, input_sha, progress, identities, previous, first_frame):
    _require(type(value) is dict and set(value) == WINDOW_KEYS
             and value['schema'] == 'tensorfold.target-window-teacher-window-v1'
             and value['rank'] == rank and type(value['rank']) is int
             and value['worker_PID'] == pid and type(value['worker_PID']) is int
             and value['input_sha256'] == input_sha and type(value['window']) is int
             and 0 <= value['window'] < len(plan['rounds'])
             and value['schedule'] == plan['rounds'][value['window']]
             and value['method'] == 'mtp' and value['depth'] == 4 and type(value['depth']) is int
             and value['active_states'] == 8 and type(value['active_states']) is int
             and value['role'] == 'forced-target-window' and value['qualification'] is False and value['P3_valid'] is False,
             'actual complete target-window record required')
    cases = {case['case_id']: case for case in plan['sequences']}
    lanes = {lane['case_id']: lane for lane in value['schedule']['lanes']}
    _require(type(value['before']) is dict and type(value['after']) is dict
             and set(value['before']) == set(cases) == set(value['after']), 'all eight State transactions required')
    phases = _phase_specs(plan)
    for name, case in cases.items():
        before = _snapshot(value['before'][name], case, progress=progress[name], identity=identities[name])
        after = _snapshot(value['after'][name], case, progress=progress[name]+lanes[name]['keep'], identity=before)
        _require(before['done'] is False and after['done'] == (value['window'] == len(plan['rounds'])-1),
                 'forced State ended outside the declared common final window')
        if previous is not None:
            _require(before == previous[name], 'State transaction drifted between adjacent windows')
        before_cap = phases[None if value['window'] == 0 else value['window']-1][name][2]
        _require(before['draft_tokens'] == case['forced_ids'][progress[name]:progress[name]+before_cap],
                 'before-window MTP proposals are not the complete cap-aware forced gold prefix')
        before_tentative = max(before_cap-1, 0)
        _require(before['mtp_drafted'] == before_tentative
                 and before['mtp_len'] == len(case['prompt_ids'])+progress[name]-1+before_tentative,
                 'before-window MTP cache prefix/tentative counters differ from the forced production chain')
        after_progress = progress[name]+lanes[name]['keep']
        expected_after = (case['forced_ids'][progress[name]:progress[name]+lanes[name]['rows']-1]
                          if after['done'] else case['forced_ids'][after_progress:after_progress+phases[value['window']][name][2]])
        _require(after['draft_tokens'] == expected_after, 'after-window MTP proposal list is not the declared forced prefix')
        if after['done']:
            _require(after['mtp_len'] == before['mtp_len'] and after['mtp_drafted'] == before['mtp_drafted'],
                     'final target window unexpectedly changed retired MTP conditioning')
        else:
            after_tentative = max(phases[value['window']][name][2]-1, 0)
            _require(after['mtp_drafted'] == after_tentative
                     and after['mtp_len'] == len(case['prompt_ids'])+after_progress-1+after_tentative,
                     'after-window MTP refresh/proposal cache prefix is incomplete')
    rows = value['frames']
    total = sum(lane['rows'] for lane in lanes.values())
    _require(type(rows) is list and len(rows) == total, 'every staged raw row must survive, including overlap/tails')
    local, packed = {name: 0 for name in cases}, set()
    order = []
    for index, row in enumerate(rows):
        _require(type(row) is dict and type(row.get('case_id')) is str and row['case_id'] in cases, 'raw frame names an unknown forced case')
        name = row['case_id']
        if name not in order:
            order.append(name)
        _require(order[-1] == name, 'a case window must occupy one contiguous packed segment')
        _frame(row, cases[name], frame=first_frame+index, window=value['window'], local_row=local[name],
               rows=lanes[name]['rows'], progress=progress[name], rank=rank, before=value['before'][name], batch_rows=total)
        _require(row['packed_row'] == index and index not in packed, 'packed target row offset differs')
        packed.add(index)
        local[name] += 1
    _require(all(local[name] == lane['rows'] for name, lane in lanes.items()), 'target row catalogue incomplete')
    acceptance = value['acceptance']
    _require(type(acceptance) is list and len(acceptance) == 8, 'all eight original/forced acceptance records required')
    from tensorfold.cuda.streams import accept
    for name, row in zip(order, acceptance, strict=True):
        lane, case = lanes[name], cases[name]
        _require(type(row) is dict and set(row) == {'case_id', 'original_path', 'original_terminal', 'forced_path',
            'forced_terminal', 'role', 'original_scope'} and row['case_id'] == name
            and type(row['forced_path']) is list and row['forced_path'] == list(range(lane['keep']))
            and all(type(i) is int for i in row['forced_path'])
            and row['forced_terminal'] == case['forced_ids'][progress[name]+lane['keep']-1]
            and type(row['forced_terminal']) is int
            and row['role'] == 'diagnostic-forced-keep-not-serving-acceptance'
            and row['original_scope'] == 'counterfactual-forced-proposals-and-prefix-v1'
            and type(row['original_path']) is list and row['original_path']
            and row['original_path'] == list(range(len(row['original_path'])))
            and all(type(i) is int for i in row['original_path'])
            and len(row['original_path']) <= lane['rows'] and _integer(row['original_terminal'])
            and row['original_terminal'] < 248320, 'original versus diagnostic acceptance linkage differs')
        source_rows = [frame for frame in rows if frame['case_id'] == name]
        tokens = case['forced_ids'][progress[name]-1:progress[name]-1+lane['rows']]
        path, terminal = accept(tokens, list(range(-1, lane['rows']-1)),
                                [frame['original_sampled'] for frame in source_rows],
                                len(case['forced_ids'])-progress[name])
        _require(row['original_path'] == path and row['original_terminal'] == terminal,
                 'counterfactual original target acceptance differs on the forced proposal path')
    before_counts, after_counts = _counts(value['graph_counts_before']), _counts(value['graph_counts_after'])
    delta = {kind: {role: after_counts[kind][role]-before_counts[kind][role] for role in ('verify', 'mtp')}
             for kind in before_counts}
    _require(all(n >= 0 for counts in delta.values() for n in counts.values()) and value['graph_deltas'] == delta,
             'actual graph count deltas differ')
    calls, coverage = value['graph_calls'], {'verify': False, 'mtp': False}
    _require(type(calls) is list and 1 <= len(calls) <= 5, 'bounded actual target/MTP graph call records required')
    totals = {kind: dict.fromkeys(('verify', 'mtp'), 0) for kind in before_counts}
    for call in calls:
        _require(type(call) is dict and set(call) == {'role', 'participants', 'rows', 'serving_hits_delta',
                'serving_misses_delta', 'history_slots', 'slot_generations'} and type(call['role']) is str and call['role'] in coverage
                and type(call['participants']) is int and 1 <= call['participants'] <= 8
                and type(call['rows']) is int and call['participants'] <= call['rows'] <= 40
                and type(call['serving_hits_delta']) is int and call['serving_hits_delta'] in (0, 1)
                and type(call['serving_misses_delta']) is int and call['serving_misses_delta'] in (0, 1)
                and call['serving_hits_delta']+call['serving_misses_delta'] == 1,
                'actual graph call geometry/counter linkage required')
        _require(type(call['history_slots']) is list and type(call['slot_generations']) is list
                and len(call['history_slots']) == len(call['slot_generations']) == call['participants']
                and all(_integer(v) for v in (*call['history_slots'], *call['slot_generations'])),
                'graph call actual slot/generation evidence required')
        totals['serving_hits'][call['role']] += call['serving_hits_delta']
        totals['serving_misses'][call['role']] += call['serving_misses_delta']
        coverage[call['role']] |= call['participants'] == 8 and call['serving_hits_delta'] == 1
    _require(totals == delta, 'graph call records do not exhaust actual per-window counters')
    verify = [call for call in calls if call['role'] == 'verify']
    _require(len(verify) == 1 and verify[0]['participants'] == 8 and verify[0]['rows'] == total
             and verify[0]['history_slots'] == [value['before'][name]['history_slot'] for name in order]
             and verify[0]['slot_generations'] == [value['before'][name]['slot_generation'] for name in order],
             'actual target graph call differs from the staged eight-State layout')
    return coverage



def _gold(case, start, count):
    end = len(case['prompt_ids'])+len(case['forced_ids'])
    _require(_integer(start) and _integer(count, minimum=1) and start+count <= end,
             'staged/picked MTP token interval exceeds the explicit immutable gold sequence')
    prompt = len(case['prompt_ids'])
    return [case['prompt_ids'][at] if at < prompt else case['forced_ids'][at-prompt]
            for at in range(start, start+count)]


def _phase_specs(plan):
    # Reuse the production method's cap contract. This metadata projection
    # does not implement target/MTP execution or sampling.
    from tensorfold.families.qwen4_exp.cuda.speculation import MethodGeometry
    method = MethodGeometry('mtp', 4)
    def cap(case, progress):
        return method.cap_proposals(4, reply_tokens=len(case['forced_ids'])-progress,
            context_tokens=plan['context']-len(case['prompt_ids'])-progress, target_rows=plan['target_rows'])
    cases = {case['case_id']: case for case in plan['sequences']}
    phases = {None: {name: (1, 1, cap(case, 1)) for name, case in cases.items()}}
    progress = dict.fromkeys(cases, 1)
    for index, schedule in enumerate(plan['rounds']):
        phase = {}
        for lane in schedule['lanes']:
            name = lane['case_id']
            before = progress[name]
            after = before+lane['keep']
            phase[name] = (before, after, cap(cases[name], after))
            progress[name] = after
        phases[index] = phase
    return phases


def _mtp_header(value, schema, rank, pid, input_sha, index_key, index, plan):
    _require(value.get('schema') == schema and value.get('rank') == rank and type(value.get('rank')) is int
             and value.get('worker_PID') == pid and type(value.get('worker_PID')) is int
             and value.get('input_sha256') == input_sha and value.get(index_key) == index
             and type(value.get(index_key)) is int
             and (value.get('window') is None or type(value.get('window')) is int
                  and 0 <= value['window'] < len(plan['rounds'])), 'actual indexed MTP callback binding differs')


def _mtp_stage_record(value, plan, rank, pid, input_sha, index, identities):
    _require(type(value) is dict and set(value) == {'schema', 'rank', 'worker_PID', 'input_sha256', 'stage_index',
                 'window', 'participants', 'rows', 'lanes', 'kind'}, 'closed low-level MTP stage record required')
    _mtp_header(value, 'tensorfold.target-window-teacher-mtp-stage-v1', rank, pid, input_sha, 'stage_index', index, plan)
    _require(value['kind'] in ('prefill', 'mtp') and type(value['lanes']) is list
             and type(value['participants']) is int and 1 <= value['participants'] <= 8
             and value['participants'] == len(value['lanes']) and _integer(value['rows'], minimum=1)
             and value['rows'] <= (256 if value['kind'] == 'prefill' else 40)
             and (value['kind'] != 'prefill' or value['window'] is None),
             'actual MTP staging buffer role/geometry differs')
    cases = {case['case_id']: case for case in plan['sequences']}
    packed, names = 0, set()
    for lane in value['lanes']:
        _require(type(lane) is dict and set(lane) == {'case_id', 'sid', 'history_slot', 'slot_generation',
             'cache_start', 'coordinate_start', 'rope_delta', 'token_position_start', 'token_ids',
             'packed_first', 'packed_last'} and type(lane['case_id']) is str and lane['case_id'] in cases
             and lane['case_id'] not in names, 'one actual MTP input lane per State required')
        name = lane['case_id']
        _require(all(_integer(lane[key]) for key in ('sid', 'history_slot', 'slot_generation', 'cache_start',
                     'token_position_start', 'packed_first', 'packed_last'))
                 and all(lane[key] == identities[name][key] for key in ('sid', 'history_slot', 'slot_generation'))
                 and _signed(lane['coordinate_start']) and _signed(lane['rope_delta'])
                 and lane['token_position_start'] == lane['cache_start']+1
                 and lane['coordinate_start'] == lane['cache_start']+lane['rope_delta']
                 and type(lane['token_ids']) is list and lane['token_ids']
                 and len(lane['token_ids']) <= value['rows']
                 and all(_integer(token) and token < 248320 for token in lane['token_ids'])
                 and lane['token_ids'] == _gold(cases[name], lane['token_position_start'], len(lane['token_ids']))
                 and lane['packed_first'] == packed and lane['packed_last'] == packed+len(lane['token_ids']),
                 'actual staged MTP token IDs/cache/coordinate/slot prefix differs from immutable gold')
        packed = lane['packed_last']
        names.add(name)
    _require(packed == value['rows'], 'MTP staged lanes do not exhaust the declared packed input rows')
    return value


def _mtp_records(root, directory, summary, plan, rank, pid, input_sha, identities, control, *, freeze=False):
    """Validate every actual staging/pick callback with bounded reference state."""
    limits = summary['resources']
    stages, picks = summary['mtp_stages'], summary['mtp_picks']
    cases = {case['case_id']: case for case in plan['sequences']}
    phases = _phase_specs(plan)
    _require(type(stages) is list and 8 <= len(stages) <= limits['maximum_mtp_stages']
             and type(picks) is list and len(picks) <= limits['maximum_mtp_picks'],
             'bounded complete MTP callback reference indices required')
    stage_metadata, stage_digests, pick_digests = [], [], []
    prompt_progress = dict.fromkeys(cases, 1)
    serial_stages = dict.fromkeys(cases, 0)
    window_stages = dict.fromkeys(range(len(plan['rounds'])), 0)
    ropes = {}
    expected_files, byte_count = set(), 0
    last_window = -1
    for index, declared in enumerate(stages):
        name = f'mtp-stage-{index:06d}.json'
        ref = _local_ref(root, directory, declared, 16384, control, expected=name, freeze=freeze)
        value = _read(root, ref, 16384, control)
        _mtp_stage_record(value, plan, rank, pid, input_sha, index, identities)
        phase = -1 if value['window'] is None else value['window']
        _require(phase >= last_window, 'MTP stage callback phases regressed')
        last_window = phase
        ordinal = None
        if value['window'] is None:
            for lane in value['lanes']:
                case = cases[lane['case_id']]
                case_id, start, count = lane['case_id'], lane['token_position_start'], len(lane['token_ids'])
                if value['kind'] == 'prefill':
                    _require(not serial_stages[case_id] and start == prompt_progress[case_id]
                             and start+count <= len(case['prompt_ids']), 'prompt-conditioning MTP prefix has a gap/overlap')
                    prompt_progress[case_id] += count
                else:
                    cap = phases[None][case_id][2]
                    ordinal = serial_stages[case_id]
                    _require(value['participants'] == 1 and prompt_progress[case_id] == len(case['prompt_ids'])
                             and ordinal < max(1, cap) and count == 1
                             and start == len(case['prompt_ids'])+ordinal,
                             'serial initial MTP conditioning/proposal chain differs from its production cap')
                    serial_stages[case_id] += 1
        else:
            window = value['window']
            _require(all(prompt_progress[name] == len(case['prompt_ids'])
                         and serial_stages[name] == max(1, phases[None][name][2]) for name, case in cases.items()),
                     'batched MTP began without the complete initial conditioning/proposal prefix')
            _require(value['kind'] == 'mtp' and window < len(plan['rounds'])-1,
                     'unplanned MTP stage after the common final target window')
            ordinal = window_stages[window]
            phase_spec = phases[window]
            eligible = set(cases) if ordinal == 0 else {name for name, (_before, _after, cap) in phase_spec.items() if cap > ordinal}
            _require({lane['case_id'] for lane in value['lanes']} == eligible and eligible,
                     'batched MTP stage omitted or added a cap-eligible State')
            for lane in value['lanes']:
                case_id = lane['case_id']
                before, after, _cap = phase_spec[case_id]
                start = len(cases[case_id]['prompt_ids'])+(before if ordinal == 0 else after+ordinal-1)
                count = after-before if ordinal == 0 else 1
                _require(lane['token_position_start'] == start and len(lane['token_ids']) == count,
                         'MTP refresh/proposal inputs changed the exact committed gold prefix')
            window_stages[window] += 1
        for lane in value['lanes']:
            case_id = lane['case_id']
            if case_id not in ropes:
                ropes[case_id] = lane['rope_delta']
            _require(lane['rope_delta'] == ropes[case_id], 'MTP coordinate translation changed during the fixed text protocol')
        stage_metadata.append((value['window'], value['kind'], ordinal))
        stage_digests.append(_canonical({key: item for key, item in value.items() if key not in ('rank', 'worker_PID')}))
        expected_files.add(name)
        byte_count += ref['bytes']
    _require(all(prompt_progress[name] == len(case['prompt_ids'])
                 and serial_stages[name] == max(1, phases[None][name][2]) for name, case in cases.items()),
             'initial MTP prompt/serial chain evidence is incomplete')
    for index, spec in phases.items():
        if index is None:
            continue
        expected = 0 if index == len(plan['rounds'])-1 else max(1, max(cap for _before, _after, cap in spec.values()))
        _require(window_stages[index] == expected, 'batched MTP stage chain was truncated or duplicated')
    serial_picks = dict.fromkeys(cases, 0)
    batched_picks = dict.fromkeys(range(len(plan['rounds'])), 0)
    last_window = -1
    for index, declared in enumerate(picks):
        name = f'mtp-pick-{index:06d}.json'
        ref = _local_ref(root, directory, declared, 16384, control, expected=name, freeze=freeze)
        value = _read(root, ref, 16384, control)
        _require(type(value) is dict and set(value) == {'schema', 'rank', 'worker_PID', 'input_sha256', 'pick_index',
                 'window', 'branch', 'source_kind', 'stage_index', 'rows'}, 'closed original/forced MTP pick record required')
        _mtp_header(value, 'tensorfold.target-window-teacher-mtp-pick-v1', rank, pid, input_sha, 'pick_index', index, plan)
        _require(_integer(value['stage_index']) and value['stage_index'] < len(stages)
                 and type(value['rows']) is list and 1 <= len(value['rows']) <= 8,
                 'MTP pick must bind its actual supplying stage and bounded source rows')
        stage_ref = _local_ref(root, directory, stages[value['stage_index']], 16384, control,
                               expected=f"mtp-stage-{value['stage_index']:06d}.json")
        stage = _read(root, stage_ref, 16384, control)
        phase, kind, ordinal = stage_metadata[value['stage_index']]
        _require(phase == value['window'] and kind == 'mtp' and stage['kind'] == 'mtp',
                 'MTP pick points to a different phase or a prompt-conditioning-only buffer')
        phase_number = -1 if phase is None else phase
        _require(phase_number >= last_window, 'MTP pick phases regressed')
        last_window = phase_number
        if phase is None:
            branch = 'serial-sample' if plan['mtp_confidence'] == 0 else 'serial-sample-draft'
            _require(value['branch'] == branch and value['source_kind'] == 'prefill' and len(value['rows']) == 1,
                     'initial MTP pick branch differs from the explicit confidence policy')
            first_row = value['rows'][0]
            _require(type(first_row) is dict and type(first_row.get('case_id')) is str
                     and first_row['case_id'] in cases, 'serial MTP pick names an unknown case')
            proposal_index = serial_picks[first_row['case_id']]
        else:
            _require(value['branch'] == 'multi-picks' and value['source_kind'] == 'mtp'
                     and phase < len(plan['rounds'])-1, 'batched MTP pick branch/phase differs')
            proposal_index = batched_picks[phase]
        expected_cases = {lane['case_id'] for lane in stage['lanes'] if phases[phase][lane['case_id']][2] > proposal_index}
        observed = set()
        for row in value['rows']:
            _require(type(row) is dict and set(row) == {'case_id', 'sid', 'history_slot', 'slot_generation',
                'absolute_position', 'gold_token', 'original_token', 'original_probability', 'forced_token',
                'forced_probability', 'continuation_policy', 'proposal_cap', 'proposal_index'}
                and type(row['case_id']) is str and row['case_id'] in expected_cases and row['case_id'] not in observed,
                'one actual original/forced MTP pick per eligible State required')
            case_id, case = row['case_id'], cases[row['case_id']]
            _before, after, cap = phases[phase][case_id]
            supplying = next(lane for lane in stage['lanes'] if lane['case_id'] == case_id)
            _require(all(row[key] == supplying[key] and _integer(row[key]) for key in ('sid', 'history_slot', 'slot_generation'))
                     and row['proposal_cap'] == cap and type(row['proposal_cap']) is int and 1 <= cap <= 4
                     and row['proposal_index'] == proposal_index and type(row['proposal_index']) is int
                     and 0 <= proposal_index < cap and ordinal == proposal_index
                     and _integer(row['absolute_position'])
                     and row['absolute_position'] == supplying['token_position_start']+len(supplying['token_ids'])
                     and row['absolute_position'] == len(case['prompt_ids'])+after+proposal_index
                     and row['gold_token'] == _gold(case, row['absolute_position'], 1)[0]
                     and _integer(row['gold_token']) and row['forced_token'] == row['gold_token']
                     and type(row['forced_token']) is int and _integer(row['original_token']) and row['original_token'] < 248320
                     and type(row['forced_probability']) is float and row['forced_probability'] == 1.0
                     and row['continuation_policy'] == plan['proposal_policy'],
                     'actual MTP sample/forced continuation is outside its staged gold prefix/cap')
            probability = row['original_probability']
            _require((probability is None and value['branch'] == 'serial-sample' and plan['mtp_confidence'] == 0)
                     or type(probability) in (int, float) and math.isfinite(probability) and probability >= 0
                        and value['branch'] != 'serial-sample', 'original MTP probability/null branch contract differs')
            observed.add(case_id)
            if phase is None:
                serial_picks[case_id] += 1
        _require(observed == expected_cases, 'MTP pick omitted or added an eligible State')
        if phase is not None:
            batched_picks[phase] += 1
        pick_digests.append(_canonical({key: item for key, item in value.items() if key not in ('rank', 'worker_PID')}))
        expected_files.add(name)
        byte_count += ref['bytes']
    _require(all(serial_picks[name] == spec[2] for name, spec in phases[None].items()),
             'initial MTP proposal samples did not exhaust their actual production caps')
    for index, spec in phases.items():
        if index is None:
            continue
        expected = 0 if index == len(plan['rounds'])-1 else max(cap for _before, _after, cap in spec.values())
        _require(batched_picks[index] == expected, 'batched MTP proposal samples did not exhaust their production caps')
    return {'files': expected_files, 'bytes': byte_count, 'stage_digests': stage_digests,
            'pick_digests': pick_digests, 'ropes': ropes, 'phases': phases}

def _limits(plan, limits):
    window = _window()
    window.validate_plan(plan)
    total = window.ACTIVE + sum(sum(lane['rows'] for lane in row['lanes']) for row in plan['rounds'])
    maximum = max(sum(lane['rows'] for lane in row['lanes']) for row in plan['rounds'])
    _require(type(limits) is dict and limits.get('maximum_total_frames') == total
             and type(limits.get('maximum_total_frames')) is int and limits.get('maximum_windows') == len(plan['rounds'])
             and type(limits.get('maximum_windows')) is int
             and limits.get('maximum_window_rows') == maximum and type(limits.get('maximum_window_rows')) is int
             and limits.get('max_total_frame_file_bytes') == total*(4*window.VOCAB+4096)
             and type(limits.get('max_total_frame_file_bytes')) is int
             and limits.get('additional_project_device_bytes') == 4*window.VOCAB
             and type(limits.get('additional_project_device_bytes')) is int
             and _integer(limits.get('global_host_obligation_bytes'), minimum=1),
             'declared streaming geometry/device/file extent differs from the exact plan')
    expected_stages = math.ceil(sum(len(case['prompt_ids']) for case in plan['sequences'])/256)+8+8*4+4*len(plan['rounds'])
    expected_picks = 8*4+4*len(plan['rounds'])
    _require(plan.get('proposal_policy') == 'forced-gold-cap-aware-full-depth-v1'
             and limits.get('maximum_mtp_stages') == expected_stages and type(limits.get('maximum_mtp_stages')) is int
             and limits.get('maximum_mtp_picks') == expected_picks and type(limits.get('maximum_mtp_picks')) is int
             and limits.get('mtp_record_bytes_limit') == 16384 and type(limits.get('mtp_record_bytes_limit')) is int,
             'complete forced MTP callback record allowances differ from the plan')
    for name in ('receipt_bytes_limit', 'authority_bytes_limit', 'max_frame_file_bytes'):
        _require(_integer(limits.get(name), minimum=1) and limits[name] <= SHARD_LIMIT,
                 'bounded streaming metadata/vector allowance required')
    _require(limits['max_frame_file_bytes'] == 4*window.VOCAB+4096,
             'complete raw vector bound must remain explicit')
    return total


def _positive_summary(summary, rank, input_sha, plan, limits):
    _require(type(summary) is dict and set(summary) == SUMMARY_KEYS
             and summary['schema'] == 'tensorfold.target-window-teacher-rank-result-v1'
             and summary['rank'] == rank and type(summary['rank']) is int
             and _integer(summary['worker_PID'], minimum=2) and summary['input_sha256'] == input_sha
             and summary['closed'] is True and summary['failed'] is False
             and summary['finished'] == sorted(case['case_id'] for case in plan['sequences'])
             and summary['frame_count'] == limits['maximum_total_frames'] and type(summary['frame_count']) is int
             and summary['resources'] == limits and summary['quality_scope'] == plan['quality_scope']
             and summary['metrics'] == plan['metrics']
             and summary['qualification'] is False and summary['P3_valid'] is False,
             'rank lacks positive complete forced-window retirement')
    sources = summary['runtime_sources']
    _require(type(sources) is dict and set(sources) == SOURCE_KEYS and all(_sha(value) for value in sources.values()),
             'complete actual installed/tool source hashes required')
    profile = summary['profile']
    _require(type(profile) is dict and profile.get('rank') == rank and type(profile.get('rank')) is int
             and profile.get('world') == 8 and type(profile.get('world')) is int
             and profile.get('target_banks') == 48 and type(profile.get('target_banks')) is int
             and profile.get('mtp_banks') == 1 and type(profile.get('mtp_banks')) is int
             and profile.get('capability') == [7, 0]
             and all(type(n) is int for n in profile['capability']), 'actual SC117 TP8 MTP profile binding required')
    _require(type(summary['prefill']) is list and len(summary['prefill']) == 8
             and type(summary['windows']) is list and len(summary['windows']) == len(plan['rounds']),
             'complete bounded shard reference indices required')
    _counts(summary['initial_graph_counts'])
    _counts(summary['final_graph_counts'])


def _cohort(root, plan, input_sha, limits, control, *, freeze=False):
    """Validate one current shard/vector at a time, retaining only bounded indices."""
    root = Path(root)
    expected_frames = _limits(plan, limits)
    _require(_sha(input_sha), 'actual forced-plan SHA required')
    cases = {case['case_id']: case for case in plan['sequences']}
    receipts, startups, admission_receipts, pids = [], [], [], []
    cohort_sources = None
    reference_prefill, reference_windows, reference_admissions = [], [], []
    reference_stages, reference_picks = None, None
    byte_total, file_total = 0, 0
    for rank in range(8):
        control.check()
        directory = f'window-teacher-rank{rank}'
        summary_ref = _ref(root, _path(root, directory+'/summary.json'), limits['receipt_bytes_limit'], control,
                           rank=rank, freeze=freeze)
        summary = _read(root, summary_ref, limits['receipt_bytes_limit'], control)
        _positive_summary(summary, rank, input_sha, plan, limits)
        pid = summary['worker_PID']
        _require(pid not in pids, 'each rank must have its own actual worker process')
        pids.append(pid)
        if cohort_sources is None:
            cohort_sources = summary['runtime_sources']
        _require(cohort_sources == summary['runtime_sources'], 'actual runtime/tool sources differ across ranks')
        startup_ref = _ref(root, _path(root, f'rank-{rank}.json'), SHARD_LIMIT, control, rank=rank, freeze=freeze)
        startup = _read(root, startup_ref, SHARD_LIMIT, control)
        _require(startup.get('schema') == 'tensorfold.private-owned-qualification-v1'
                 and startup.get('rank') == rank and type(startup.get('rank')) is int
                 and startup.get('world') == 8 and type(startup.get('world')) is int
                 and startup.get('target_banks') == 48 and type(startup.get('target_banks')) is int
                 and startup.get('mtp_banks') == 1 and type(startup.get('mtp_banks')) is int
                 and startup.get('teacher_worker_PID') == pid and type(startup.get('teacher_worker_PID')) is int and startup.get('teacher_profile') == summary['profile']
                 and startup.get('history_host_obligation') == {
                     'global_bytes': limits['global_host_obligation_bytes'], 'input_sha256': input_sha},
                 'rank summary does not bind its actual source-owned startup/host credit')
        setup_ref = _ref(root, _path(root, directory+'/setup.json'), SHARD_LIMIT, control, freeze=freeze)
        setup = _read(root, setup_ref, SHARD_LIMIT, control)
        _require(type(setup) is dict and set(setup) == {'schema', 'input_sha256', 'actual_profile', 'resources',
                    'writer', 'qualification', 'P3_valid', 'runtime_sources'}
                 and setup['schema'] == 'tensorfold.target-window-teacher-setup-v1'
                 and setup['input_sha256'] == input_sha and setup['actual_profile'] == summary['profile']
                 and setup['resources'] == limits and setup['runtime_sources'] == summary['runtime_sources']
                 and setup['writer'] == 'existing-owned-full-vocabulary-npy-writer'
                 and setup['qualification'] is False and setup['P3_valid'] is False,
                 'actual teacher setup/profile/source/resource linkage differs')
        byte_total += sum(ref['bytes'] for ref in (summary_ref, startup_ref, setup_ref))
        file_total += 3
        expected_files = {'summary.json', 'setup.json'}
        identities, admissions, sids, slots = {}, [], set(), set()
        for index, case in enumerate(plan['sequences']):
            name = f'admission-{index}.json'
            ref = _ref(root, _path(root, directory+'/'+name), 1 << 20, control, freeze=freeze)
            admission = _read(root, ref, 1 << 20, control)
            _require(type(admission) is dict and set(admission) == {'schema', 'index', 'case_id', 'worker_PID',
                     'rank', 'input_sha256', 'state'}
                     and admission['schema'] == 'tensorfold.target-window-teacher-admission-v1'
                     and admission['index'] == index and type(admission['index']) is int
                     and admission['case_id'] == case['case_id'] and admission['rank'] == rank
                     and type(admission['rank']) is int and admission['worker_PID'] == pid
                     and type(admission['worker_PID']) is int
                     and admission['input_sha256'] == input_sha, 'actual ordered admission proof differs')
            state = _snapshot(admission['state'], case)
            _require(state['output_tokens'] == 0 and state['done'] is False and state['sid'] not in sids and state['history_slot'] not in slots,
                     'each cold case requires a unique actual admitted State/slot')
            sids.add(state['sid'])
            slots.add(state['history_slot'])
            identities[case['case_id']] = state
            digest = _canonical(state)
            if rank == 0:
                reference_admissions.append(digest)
            _require(digest == reference_admissions[index], 'admitted State ownership differs across ranks')
            admissions.append(ref)
            expected_files.add(name)
            byte_total += ref['bytes']
            file_total += 1
        admission_receipts.append(admissions)
        mtp = _mtp_records(root, directory, summary, plan, rank, pid, input_sha, identities, control, freeze=freeze)
        if rank == 0:
            reference_stages, reference_picks = mtp['stage_digests'], mtp['pick_digests']
        _require(mtp['stage_digests'] == reference_stages and mtp['pick_digests'] == reference_picks,
                 'actual MTP gold inputs, coordinates, original samples or forced continuation differ across TP ranks')
        expected_files.update(mtp['files'])
        byte_total += mtp['bytes']
        file_total += len(mtp['files'])
        observed_cases = set()
        for index, declared in enumerate(summary['prefill']):
            name = f'prefill-{index}.json'
            ref = _local_ref(root, directory, declared, 1 << 20, control, expected=name, freeze=freeze)
            row = _read(root, ref, 1 << 20, control)
            _require(type(row) is dict and type(row.get('case_id')) is str and row['case_id'] in cases and row['case_id'] not in observed_cases,
                     'each case must publish its single original prefill frame')
            case = cases[row['case_id']]
            _frame(row, case, frame=index, window=None, local_row=0, rows=1, progress=0, rank=rank,
                   before=identities[case['case_id']], batch_rows=1)
            _require(row['rope_delta'] == mtp['ropes'][case['case_id']], 'target prefill and MTP coordinates differ')
            observed_cases.add(case['case_id'])
            digest = _canonical({key: value for key, value in row.items() if key != 'vector'})
            if rank == 0:
                reference_prefill.append(digest)
                vector_ref = _vector(root, directory, row, control, freeze=freeze)
                expected_files.add(Path(vector_ref['file']).name)
                byte_total += vector_ref['bytes']
                file_total += 1
            _require(digest == reference_prefill[index], 'prefill row differs across actual TP ranks')
            expected_files.add(name)
            byte_total += ref['bytes']
            file_total += 1
        progress = dict.fromkeys(cases, 1)
        previous, frame_index = None, 8
        previous_counts = summary['initial_graph_counts']
        coverage = {'verify': False, 'mtp': False}
        for index, declared in enumerate(summary['windows']):
            control.check()
            name = f'window-{index:06d}.json'
            ref = _local_ref(root, directory, declared, limits['receipt_bytes_limit'], control,
                             expected=name, freeze=freeze)
            value = _read(root, ref, limits['receipt_bytes_limit'], control)
            _require(value.get('window') == index, 'window index differs from its immutable shard name')
            proof = _window_record(value, plan, rank, pid, input_sha, progress, identities, previous, frame_index)
            _require(all(row['rope_delta'] == mtp['ropes'][row['case_id']] for row in value['frames']),
                     'target verification and actual MTP input coordinate translations differ')
            if index:
                _require(value['graph_counts_before'] == previous_counts, 'unexplained graph work appeared between windows')
            else:
                _require(all(value['graph_counts_before'][kind][role] >= previous_counts[kind][role]
                             for kind in previous_counts for role in ('verify', 'mtp')),
                         'initial graph counts regressed before the first window')
            previous_counts = value['graph_counts_after']
            for role in coverage:
                coverage[role] |= proof[role]
            normalized = {key: item for key, item in value.items() if key not in ('rank', 'worker_PID', 'frames')}
            normalized['frames'] = [{key: item for key, item in row.items() if key != 'vector'} for row in value['frames']]
            digest = _canonical(normalized)
            if rank == 0:
                reference_windows.append(digest)
                for row in value['frames']:
                    vector_ref = _vector(root, directory, row, control, freeze=freeze)
                    expected_files.add(Path(vector_ref['file']).name)
                    byte_total += vector_ref['bytes']
                    file_total += 1
            _require(digest == reference_windows[index], 'actual State/window/row/graph evidence differs across TP ranks')
            frame_index += len(value['frames'])
            for lane in plan['rounds'][index]['lanes']:
                progress[lane['case_id']] += lane['keep']
            previous = value['after']
            expected_files.add(name)
            byte_total += ref['bytes']
            file_total += 1
            del value
        _require(frame_index == expected_frames and all(coverage.values())
                 and summary['final_graph_counts'] == previous_counts
                 and all(progress[name] == len(case['forced_ids']) for name, case in cases.items()),
                 'complete forced corpus/windows and actual eight-State target/MTP graph replay required')
        directory_path = _path(root, directory)
        _require({path.name for path in directory_path.iterdir()} == expected_files
                 and all(path.is_file() and not path.is_symlink() for path in directory_path.iterdir()),
                 'rank directory contains missing or unbound artifacts')
        if freeze:
            directory_path.chmod(stat.S_IMODE(directory_path.stat().st_mode) & ~0o222)
        receipts.append(summary_ref)
        startups.append(startup_ref)
        del summary, startup, setup
    return {'schema': AUTHORITY_SCHEMA, 'launch_exit': 0, 'cohort_complete': True, 'closed': True,
            'finished': True, 'failed': False, 'teacher_plan_sha256': input_sha, 'plan': plan, 'resources': limits,
            'frame_count': expected_frames, 'rank_PIDs': pids, 'rank_receipts': receipts,
            'startup_receipts': startups, 'admission_receipts': admission_receipts,
            'runtime_sources': cohort_sources, 'data_file_count': file_total, 'data_bytes': byte_total,
            'mtp_records_per_rank': {'stages': len(reference_stages), 'picks': len(reference_picks)},
            'quality_scope': plan['quality_scope'], 'metrics': plan['metrics'],
            'qualification': False, 'quality_acceptance': False, 'P3_valid': False,
            'scope': 'complete sharded forced-window rows; Host/provider/NUMA/quality acceptance remain owning-contract duties'}


def seal(output_teacher_root, prepared, launch_exit):
    """Seal after all producers retired; no producer or GPU work is started here."""
    _require(type(launch_exit) is int and launch_exit == 0, 'positive original launcher retirement required')
    root, plan, limits = Path(output_teacher_root), prepared['data'], prepared['resources']
    _limits(plan, limits)
    control = quality._teacher_read_control(plan['read_timeout_s'])
    # Establish every producer's positive close before making any artifact
    # read-only. The full pass then validates/fixes each ordinary file in turn.
    for rank in range(8):
        path = _path(root, f'window-teacher-rank{rank}/summary.json')
        raw = quality._owned_teacher_read(path, limits['receipt_bytes_limit'])
        summary = quality._owned_teacher_object(raw)
        _positive_summary(summary, rank, prepared['input_sha256'], plan, limits)
        control.check()
    authority = _cohort(root, plan, prepared['input_sha256'], limits, control, freeze=True)
    path = root/'teacher-authority.json'
    quality._owned_teacher_json(path, authority, limits['authority_bytes_limit'], control)
    path.chmod(0o444)
    return authority


def validate_artifacts(root_evidence, model_result, operation, *, plan):
    """Replay the bound row closure; caller owns full Host/provider/NUMA proof."""
    root, teacher = Path(root_evidence), Path(root_evidence)/'teacher'
    control = quality._teacher_read_control(plan['read_timeout_s'])
    _require(model_result.get('teacher_plan_sha256') == operation['teacher_plan']['sha256']
             and model_result.get('teacher_client_sha256') == operation['teacher_client']['sha256'],
             'model result differs from its original forced plan/client')
    authority_ref = model_result['teacher_authority']
    _require(authority_ref.get('file') == 'teacher/teacher-authority.json', 'fixed sharded authority path required')
    authority = _read(root, authority_ref, SHARD_LIMIT, control)
    _require(authority.get('schema') == AUTHORITY_SCHEMA and authority.get('plan') == plan
             and authority.get('teacher_plan_sha256') == operation['teacher_plan']['sha256'],
             'actual authority/forced-plan binding differs')
    actual = _cohort(teacher, plan, operation['teacher_plan']['sha256'], authority['resources'], control)
    _require(actual == authority, 'actual sharded closure changed after positive publication')
    for key in ('rank_receipts', 'startup_receipts'):
        expected = [{**ref, 'file': 'teacher/'+ref['file']} for ref in authority[key]]
        _require(model_result.get(key) == expected, 'model result does not bind the original rank/startup closure')
    return authority


def _groups(root, authority, control):
    """Yield rank0 rows only after their actual immutable shard is rechecked."""
    rank_ref = authority['rank_receipts'][0]
    summary = _read(root, rank_ref, authority['resources']['receipt_bytes_limit'], control)
    _positive_summary(summary, 0, authority['teacher_plan_sha256'], authority['plan'], authority['resources'])
    directory = 'window-teacher-rank0'
    prefills = []
    for index, declared in enumerate(summary['prefill']):
        ref = _local_ref(root, directory, declared, 1 << 20, control, expected=f'prefill-{index}.json')
        prefills.append(_read(root, ref, 1 << 20, control))
    yield 'prefill', None, prefills, {'role': 'prefill-singleton-windows'}
    del prefills
    for index, declared in enumerate(summary['windows']):
        ref = _local_ref(root, directory, declared, authority['resources']['receipt_bytes_limit'], control,
                         expected=f'window-{index:06d}.json')
        window = _read(root, ref, authority['resources']['receipt_bytes_limit'], control)
        context = {key: window[key] for key in ('schedule', 'before', 'after', 'acceptance',
            'graph_calls', 'graph_counts_before', 'graph_counts_after', 'graph_deltas')}
        yield 'target-verify', index, window['frames'], context
        del window


def _pair_key(row):
    return (row['window'], row['case_id'], row['window_row'], row['absolute_position'], row['gold_token'])


def _corpus_key(row):
    return row['case_id'], row['absolute_position'], row['gold_token']


def _raw_vector(root, ref, control):
    path = quality._owned_comparison_file(Path(root), ref, control, maximum=4*248320+4096, parsed=False)
    return quality._owned_comparison_vector(path, ref, 248320, control)


def _row_vector(root, row, control):
    vector = row['vector']
    ref = {key: vector[key] for key in ('bytes', 'sha256')}
    ref['file'] = 'window-teacher-rank0/'+vector['file']
    return _raw_vector(root, ref, control), ref


def _metric_rows(root, refs, control, *, unique=False, prefill=False):
    for ref in refs:
        shard = _read(root, ref, METRIC_SHARD_LIMIT, control)
        for row in shard['rows']:
            if (not unique or row['first_corpus_observation']) and (not prefill or row['role'] == 'prefill'):
                yield row


def _aggregate(root, refs, control, *, unique=False, prefill=False):
    """Use math.fsum over reopened bounded metric shards, never retain all rows."""
    count, exact, agreement = 0, 0, 0
    for row in _metric_rows(root, refs, control, unique=unique, prefill=prefill):
        count += 1
        exact += row['exact_uint32_bits']
        agreement += row['base_top1'] == row['candidate_top1']
    _require(count > 0, 'nonempty declared metric population required')
    means = {}
    for key in ('kl', 'nll_delta', 'base_nll', 'candidate_nll'):
        means[key] = math.fsum(row[key] for row in _metric_rows(root, refs, control,
                          unique=unique, prefill=prefill))/count
    def perplexity(value):
        return math.exp(value) if value <= math.log(sys.float_info.max) else None
    return {'rows': count, 'mean_kl': means['kl'], 'mean_nll_delta': means['nll_delta'],
            'reference_mean_nll': means['base_nll'], 'candidate_mean_nll': means['candidate_nll'],
            'reference_perplexity': perplexity(means['base_nll']),
            'candidate_perplexity': perplexity(means['candidate_nll']), 'ppl_ratio': perplexity(means['nll_delta']),
            'top1_agreement': agreement/count, 'exact_frames': exact}


def _chain_observations(root, refs, control):
    result = {side: {'maximum_observed_forced_proposal_list': 0,
                    'maximum_recorded_mtp_graph_calls_per_window': 0,
                    'eight_state_graph_hits_by_rows': {'verify': {}, 'mtp': {}}}
              for side in ('reference', 'candidate')}
    for ref in refs:
        shard = _read(root, ref, METRIC_SHARD_LIMIT, control)
        if shard['role'] == 'prefill':
            continue
        for side, state in result.items():
            context = shard[side+'_window']
            for snapshots in (context['before'], context['after']):
                state['maximum_observed_forced_proposal_list'] = max(
                    state['maximum_observed_forced_proposal_list'],
                    max(len(row['draft_tokens']) for row in snapshots.values()))
            state['maximum_recorded_mtp_graph_calls_per_window'] = max(
                state['maximum_recorded_mtp_graph_calls_per_window'],
                sum(call['role'] == 'mtp' for call in context['graph_calls']))
            for call in context['graph_calls']:
                if call['participants'] == 8 and call['serving_hits_delta']:
                    role = state['eight_state_graph_hits_by_rows'][call['role']]
                    key = str(call['rows'])
                    role[key] = role.get(key, 0)+call['serving_hits_delta']
    return result



def _stage_view(value):
    # Comparison keys never replace the artifact SHA or either run's identity.
    # Packed offsets/order are preserved in the raw records but logical batch
    # equality is independent of the controller's physical case ordering.
    return {key: value[key] for key in ('window', 'participants', 'rows', 'kind')} | {
        'lanes': sorted(({key: item for key, item in row.items() if key not in ('packed_first', 'packed_last')}
                         for row in value['lanes']), key=lambda row: row['case_id'])}


def _pick_view(value, stage_key):
    return {key: value[key] for key in ('window', 'branch', 'source_kind')} | {'supplying_stage': stage_key,
        'rows': sorted(({key: item for key, item in row.items() if key not in ('original_token', 'original_probability')}
                        for row in value['rows']), key=lambda row: row['case_id'])}


def _mtp_indices(root, authority, control):
    summary = _read(root, authority['rank_receipts'][0], authority['resources']['receipt_bytes_limit'], control)
    _positive_summary(summary, 0, authority['teacher_plan_sha256'], authority['plan'], authority['resources'])
    stages, picks, stage_keys = {}, {}, []
    for index, declared in enumerate(summary['mtp_stages']):
        ref = _local_ref(root, 'window-teacher-rank0', declared, 16384, control, expected=f'mtp-stage-{index:06d}.json')
        value = _read(root, ref, 16384, control)
        key = _canonical(_stage_view(value))
        _require(key not in stages, 'duplicate actual MTP staging batch/prefix in the fixed protocol')
        stages[key] = ref
        stage_keys.append(key)
    for index, declared in enumerate(summary['mtp_picks']):
        ref = _local_ref(root, 'window-teacher-rank0', declared, 16384, control, expected=f'mtp-pick-{index:06d}.json')
        value = _read(root, ref, 16384, control)
        _require(_integer(value.get('stage_index')) and value['stage_index'] < len(stage_keys),
                 'MTP pick lost its actual supplying stage before comparison')
        key = _canonical(_pick_view(value, stage_keys[value['stage_index']]))
        _require(key not in picks, 'duplicate MTP proposal-step callback in the fixed protocol')
        picks[key] = (ref, stage_keys[value['stage_index']])
    return stages, picks


def _compare_mtp(roots, authorities, control, output):
    left_stages, left_picks = _mtp_indices(roots[0], authorities[0], control)
    right_stages, right_picks = _mtp_indices(roots[1], authorities[1], control)
    _require(set(left_stages) == set(right_stages),
             'actual MTP input IDs/cache/coordinates/slot prefixes or callback row geometry differ between profiles')
    _require(set(left_picks) == set(right_picks),
             'actual forced MTP proposal positions/caps/continuations differ between profiles')
    stage_refs, pick_refs = [], []
    original_rows = original_differences = 0
    outside_unit = {'reference': 0, 'candidate': 0}
    for index, (key, left_ref) in enumerate(left_stages.items()):
        right_ref = right_stages[key]
        left, right = _read(roots[0], left_ref, 16384, control), _read(roots[1], right_ref, 16384, control)
        _require(_stage_view(left) == _stage_view(right), 'MTP staging comparison key did not bind exact raw inputs')
        path = output/f'mtp-stage-{index:06d}-comparison.json'
        quality._owned_teacher_json(path, {'schema': 'tensorfold.target-window-teacher-mtp-stage-comparison-v1',
            'reference': left_ref, 'candidate': right_ref, 'reference_record': left, 'candidate_record': right,
            'actual_gold_input_coordinates_equal': True, 'qualification': False, 'P3_valid': False}, METRIC_SHARD_LIMIT, control)
        path.chmod(0o444)
        stage_refs.append(_ref(output, path, METRIC_SHARD_LIMIT, control))
    for index, (key, (left_ref, left_stage)) in enumerate(left_picks.items()):
        right_ref, right_stage = right_picks[key]
        left, right = _read(roots[0], left_ref, 16384, control), _read(roots[1], right_ref, 16384, control)
        _require(left_stage == right_stage and _pick_view(left, left_stage) == _pick_view(right, right_stage),
                 'MTP pick comparison key did not bind exact forced continuation')
        right_rows = {row['case_id']: row for row in right['rows']}
        rows = []
        for old in left['rows']:
            new = right_rows[old['case_id']]
            a, b = old['original_probability'], new['original_probability']
            _require((a is None) == (b is None), 'paired original MTP probability branches differ')
            different = old['original_token'] != new['original_token']
            outside_unit['reference'] += a is not None and not 0 <= a <= 1
            outside_unit['candidate'] += b is not None and not 0 <= b <= 1
            rows.append({'case_id': old['case_id'], 'absolute_position': old['absolute_position'],
                'proposal_cap': old['proposal_cap'], 'proposal_index': old['proposal_index'],
                'reference_original_token': old['original_token'], 'candidate_original_token': new['original_token'],
                'original_token_equal': not different, 'reference_original_probability': a,
                'candidate_original_probability': b, 'original_probability_delta': None if a is None else b-a,
                'reference_original_probability_outside_unit_interval': a is not None and not 0 <= a <= 1,
                'candidate_original_probability_outside_unit_interval': b is not None and not 0 <= b <= 1,
                'forced_token': old['forced_token'], 'forced_probability': old['forced_probability']})
            original_rows += 1
            original_differences += different
        path = output/f'mtp-pick-{index:06d}-comparison.json'
        quality._owned_teacher_json(path, {'schema': 'tensorfold.target-window-teacher-mtp-pick-comparison-v1',
            'reference': left_ref, 'candidate': right_ref, 'reference_record': left, 'candidate_record': right,
            'rows': rows, 'qualification': False, 'P3_valid': False}, METRIC_SHARD_LIMIT, control)
        path.chmod(0o444)
        pick_refs.append(_ref(output, path, METRIC_SHARD_LIMIT, control))
    return {'policy': authorities[0]['plan']['proposal_policy'], 'stage_shards': stage_refs, 'pick_shards': pick_refs,
            'actual_input_coordinate_batches': len(stage_refs), 'actual_pick_callbacks': len(pick_refs),
            'original_sample_observations': original_rows, 'original_token_differences': original_differences,
            'original_probability_outside_unit_interval': outside_unit,
            'original_probability_status': ('OUTSIDE_UNIT_INTERVAL_OBSERVED' if any(outside_unit.values())
                                            else 'NO_OUTSIDE_UNIT_INTERVAL_OBSERVED'),
            'scope': 'original MTP samples under matched forced gold inputs and forced probability1 continuation; '
                     'dependent proposal observations, not MTP-head KL or ordinary generated trajectories'}

def compare_rows(left_teacher_root, right_teacher_root, left_authority, right_authority, *, control, output):
    """Compare owning-contract-admitted rows without retaining vectors or FDs.

    Call only after the outer v4 comparison has verified both complete Host,
    source/image/model/frontend/SDK/native/NUMA/profile closures. Those duties
    are intentionally not replaced by this lower-level numeric reader.
    """
    import numpy as np
    roots = (Path(left_teacher_root), Path(right_teacher_root))
    authorities = (left_authority, right_authority)
    actual_refs = []
    for root, authority in zip(roots, authorities, strict=True):
        _require(type(authority) is dict and authority.get('schema') == AUTHORITY_SCHEMA,
                 'versioned positively closed sharded authority required')
        ref = _ref(root, _path(root, 'teacher-authority.json'), SHARD_LIMIT, control)
        _require(_read(root, ref, SHARD_LIMIT, control) == authority, 'comparison authority changed before use')
        _limits(authority['plan'], authority['resources'])
        actual_refs.append(ref)
    a, b = (authority['plan'] for authority in authorities)
    _require(set(a) == set(b) and all(a[key] == b[key] for key in a if key != 'identity'),
             'candidate changed the explicit corpus, windows, keep schedule or metric policy')
    _require(all(a['identity'][key] == b['identity'][key] for key in ('corpus_id', 'model_sha256', 'frontend_sha256')),
             'candidate changed its declared corpus/model/frontend identity')
    _require(left_authority['frame_count'] == right_authority['frame_count'], 'paired raw row counts differ')
    output = Path(output)
    _require(output.is_absolute() and not output.exists() and not output.is_symlink(), 'fresh absolute metric output required')
    output.mkdir(mode=0o700)
    mtp_comparison = _compare_mtp(roots, authorities, control, output)
    refs, first = [], {}
    observed_rows = 0
    for left_group, right_group in zip(_groups(roots[0], left_authority, control),
                                       _groups(roots[1], right_authority, control), strict=True):
        role, index, left_rows, left_context = left_group
        _require((role, index) == right_group[:2], 'paired immutable window sequence differs')
        right_rows = right_group[2]
        old = {_pair_key(row): row for row in left_rows}
        new = {_pair_key(row): row for row in right_rows}
        _require(len(old) == len(left_rows) and len(new) == len(right_rows) and set(old) == set(new),
                 'paired logical case/position/gold/window-row catalogue differs')
        rows = []
        for key, left in old.items():
            control.check()
            right = new[key]
            _require(all(left[name] == right[name] for name in ('role', 'target_window_rows', 'batch_target_rows',
                     'active_states', 'physical_target_rows', 'method', 'depth', 'vocab', 'teacher_forced',
                     'input_token', 'input_position', 'rope_delta', 'input_coordinate')),
                     'paired C8 execution region/row shape differs')
            x, x_ref = _row_vector(roots[0], left, control)
            y, y_ref = _row_vector(roots[1], right, control)
            unequal = x.view('<u4') != y.view('<u4')
            corpus_key = _corpus_key(left)
            initial = corpus_key not in first
            row = {'case_id': left['case_id'], 'absolute_position': left['absolute_position'],
                   'gold_token': left['gold_token'], 'role': role, 'window': index, 'window_row': left['window_row'],
                   'first_corpus_observation': initial, 'reference_row': left, 'candidate_row': right,
                   'exact_uint32_bits': not bool(unequal.any()), 'different_uint32_values': int(unequal.sum()),
                   'reference_raw_top1_ties': int(np.count_nonzero(x == x.max())),
                   'candidate_raw_top1_ties': int(np.count_nonzero(y == y.max())), 'finite': True,
                   **quality.compare_logits(x, y, left['gold_token'])}
            del unequal
            if initial:
                first[corpus_key] = (x_ref, y_ref)
            else:
                repeats = {}
                for side, current, ref in (('reference', x, first[corpus_key][0]),
                                           ('candidate', y, first[corpus_key][1])):
                    original = _raw_vector(roots[0 if side == 'reference' else 1], ref, control)
                    repeats[side] = {'first_vector': ref,
                        'different_uint32_values': int(np.count_nonzero(original.view('<u4') != current.view('<u4'))),
                        **quality.compare_logits(original, current, left['gold_token'])}
                    del original
                row['overlap_first_observation_comparison'] = repeats
                del current
            del x, y
            rows.append(row)
            observed_rows += 1
        name = 'prefill-metrics.json' if index is None else f'window-{index:06d}-metrics.json'
        path = output/name
        quality._owned_teacher_json(path, {'schema': 'tensorfold.target-window-teacher-metric-shard-v1',
            'role': role, 'window': index, 'rows': rows,
            'reference_window': left_context, 'candidate_window': right_group[3],
            'qualification': False, 'P3_valid': False}, METRIC_SHARD_LIMIT, control)
        path.chmod(0o444)
        refs.append(_ref(output, path, METRIC_SHARD_LIMIT, control))
        del rows, old, new, left_rows, right_rows
    _require(observed_rows == left_authority['frame_count']
             and len(first) == sum(len(case['forced_ids']) for case in a['sequences']),
             'complete raw rows or unique forced corpus positions are missing')
    corpus = _aggregate(output, refs, control, unique=True)
    windows = _aggregate(output, refs, control)
    prefill = _aggregate(output, refs, control, prefill=True)
    report = {'schema': 'tensorfold.target-window-teacher-row-comparison-v1', 'complete': True,
        'quality_acceptance': False, 'qualification': False, 'P3_valid': False,
        'authority_inputs': [{'root': str(root), **ref} for root, ref in zip(roots, actual_refs, strict=True)],
        'plan_identities': [a['identity'], b['identity']], 'metric_policy': a['metrics'],
        'corpus': corpus, 'all_window_rows': windows, 'prefill_singleton_windows': prefill,
        'actual_chain_observations': _chain_observations(output, refs, control),
        'mtp_forced_input_comparison': mtp_comparison,
        'original_acceptance_scope': 'counterfactual-forced-proposals-and-prefix-v1',
        'raw_row_count': observed_rows, 'unique_corpus_positions': len(first), 'metric_shards': refs,
        'coverage': {'method': 'mtp', 'configured_maximum_depth': 4, 'active_target_states': 8, 'vocab': 248320,
                     'resident_states': 24, 'context': 262144, 'target_rows': 120, 'prefill_rows': 256},
        'scope': 'conditional forced-prefix metrics, not held-out accuracy/free-generation acceptance; '
                 'all-window rows include prefill singleton windows and dependent overlap/rollback observations; '
                 'configured depth is a maximum, actual proposal lists/graph calls are separate; no MTP-head KL'}
    quality._owned_teacher_json(output/'comparison.json', report, SHARD_LIMIT, control)
    (output/'comparison.json').chmod(0o444)
    return report
