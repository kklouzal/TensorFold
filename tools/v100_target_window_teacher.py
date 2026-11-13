"""Forced C8/MTP4 target-window DATA and production-decoder observation.

Forced windows and keep paths are diagnostic inputs. They are deliberately
separate from ordinary speculative acceptance and are never serving defaults.
Complete raw rows, including overlap/rejected tails, are retained.
"""
from __future__ import annotations

import hashlib
import math
import re
from itertools import chain
from pathlib import Path

if __package__:
    from . import v100_rotorquant_quality as frames
else:
    import v100_rotorquant_quality as frames

PLAN_SCHEMA = 'tensorfold.target-window-teacher-plan-v1'
MAX_PLAN_BYTES = 1 << 20
MAX_ROUNDS = 127
MAX_SEQUENCE_TOKENS = 128
MTP_RECORD_BYTES = 16384
PROPOSAL_POLICY = 'forced-gold-cap-aware-full-depth-v1'
VOCAB, WORLD, ACTIVE, DEPTH = 248320, 8, 8, 4
ROW_WIDTH = VOCAB // WORLD
METRICS = {'corpus': 'unique-case-position-first-observation-v1',
           'windows': 'all-window-rows-v1'}

if __package__:
    from . import v100_numerical_teacher as numerical
else:
    import v100_numerical_teacher as numerical
numa, legacy = numerical.numa, numerical.legacy
SCHEMA, MODE, SCOPE = legacy.WINDOW_TEACHER_SCHEMA, legacy.WINDOW_TEACHER_MODE, legacy.WINDOW_TEACHER_SCOPE
HOST_RESULT_SCHEMA = 'tensorfold.owned-target-window-teacher-operation-result-v4'
RESULT_SCHEMA = 'tensorfold.target-window-teacher-model-result-v4'
CLIENT_SCHEMA = 'tensorfold.target-window-teacher-client-v4'
CLIENT_REPORT_SCHEMA = 'tensorfold.target-window-teacher-client-report-v4'
FILES, PLACEMENT_MEMBER, RECEIPT_DIR = numa.FILES, numa.PLACEMENT_MEMBER, numa.RECEIPT_DIR
REQUIRED_PREFIXES, MODEL_OBSERVER_LIMITS = legacy.REQUIRED_PREFIXES, legacy.MODEL_OBSERVER_LIMITS
SERVICE = dict(numa.SERVICE)
identity, absolute, selectors = legacy.identity, legacy.absolute, numerical.selectors
validate_numa, frozen_host_inputs, load_inputs = numa.validate_numa, numa.frozen_host_inputs, numa.load_inputs
publish_inputs, collect, evidence = numa.publish_inputs, numa.collect, numa.evidence
validate_teacher_completion = legacy.validate_teacher_completion


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_plan(value):
    required = {'schema', 'identity', 'vocab', 'context', 'resident_states', 'target_rows', 'prefill_rows',
                'method', 'depth', 'mtp_confidence', 'proposal_policy', 'sequences', 'rounds', 'metrics', 'read_timeout_s', 'quality_scope'}
    require(type(value) is dict and set(value) == required and value['schema'] == PLAN_SCHEMA,
            'closed versioned target-window teacher plan required')
    for key, expected in (('vocab', VOCAB), ('context', 262144), ('resident_states', 24),
                          ('target_rows', 120), ('prefill_rows', 256), ('depth', DEPTH)):
        require(type(value[key]) is int and value[key] == expected, 'target-window profile differs: ' + key)
    require(type(value['mtp_confidence']) in (int, float) and math.isfinite(value['mtp_confidence'])
            and 0 <= value['mtp_confidence'] <= 1, 'explicit valid MTP confidence required')
    require(value['proposal_policy'] == PROPOSAL_POLICY, 'explicit gold proposal/continuation policy required')
    require(value['method'] == 'mtp' and value['metrics'] == METRICS
            and value['quality_scope'] == 'diagnostic-forced-prefix-not-heldout',
            'explicit MTP forced-prefix metric scope required')
    require(type(value['read_timeout_s']) in (int, float) and math.isfinite(value['read_timeout_s'])
            and 0 < value['read_timeout_s'] <= 7500, 'bounded target-window read deadline required')
    declared = value['identity']
    require(type(declared) is dict and set(declared) == {'run_id', 'corpus_id', 'model_sha256', 'frontend_sha256', 'profile_sha256'}
            and all(type(declared[k]) is str and re.fullmatch(r'[A-Za-z0-9_-]{1,64}', declared[k])
                    for k in ('run_id', 'corpus_id'))
            and all(type(declared[k]) is str and re.fullmatch(r'[0-9a-f]{64}', declared[k])
                    for k in ('model_sha256', 'frontend_sha256', 'profile_sha256')),
            'actual corpus/model/frontend/profile declarations required')
    sequences = value['sequences']
    require(type(sequences) is list and len(sequences) == ACTIVE, 'exactly eight forced sequences required')
    names, progress = [], {}
    for row in sequences:
        require(type(row) is dict and set(row) == {'case_id', 'prompt_ids', 'forced_ids'}
                and type(row['case_id']) is str and re.fullmatch(r'[A-Za-z0-9_-]{1,64}', row['case_id'])
                and row['case_id'] not in progress, 'unique explicit forced-sequence case required')
        prompt, forced = row['prompt_ids'], row['forced_ids']
        require(type(prompt) is list and prompt and type(forced) is list and 2 <= len(forced) <= MAX_SEQUENCE_TOKENS
                and len(prompt) + len(forced) <= value['context']
                and all(type(token) is int and 0 <= token < VOCAB for token in chain(prompt, forced)),
                'bounded explicit prompt/forced tokens required')
        names.append(row['case_id'])
        progress[row['case_id']] = 1  # The first forced output is the prefill frame.
    rounds = value['rounds']
    require(type(rounds) is list and 1 <= len(rounds) <= MAX_ROUNDS, 'bounded explicit keep schedule required')
    lengths = {row['case_id']: len(row['forced_ids']) for row in sequences}
    for step, row in enumerate(rounds):
        require(type(row) is dict and set(row) == {'lanes'} and type(row['lanes']) is list
                and len(row['lanes']) == ACTIVE, 'eight explicit lanes per target window required')
        observed, total = [], 0
        for lane in row['lanes']:
            require(type(lane) is dict and set(lane) == {'case_id', 'rows', 'keep'}
                    and type(lane['case_id']) is str and lane['case_id'] in progress and lane['case_id'] not in observed
                    and type(lane['rows']) is int and 1 <= lane['rows'] <= DEPTH + 1
                    and type(lane['keep']) is int and 1 <= lane['keep'] <= lane['rows'],
                    'one bounded target row/keep interval per case required')
            case = lane['case_id']
            require(progress[case] + lane['rows'] <= lengths[case], 'target row has no explicit next-token gold label')
            progress[case] += lane['keep']
            require((progress[case] == lengths[case]) == (step == len(rounds)-1),
                    'all eight cases must remain active through the same final window')
            observed.append(case)
            total += lane['rows']
        require(observed == names and total <= ACTIVE*(DEPTH+1), 'declared logical lane order/target row total differs')
    return value


def load_plan(path, *, expected_sha256, expected_bytes):
    require(type(expected_bytes) is int and 0 < expected_bytes <= MAX_PLAN_BYTES,
            'bounded exact target-window plan extent required')
    raw = frames._owned_teacher_read(path, expected_bytes)
    require(len(raw) == expected_bytes and hashlib.sha256(raw).hexdigest() == expected_sha256,
            'target-window plan bytes changed')
    return validate_plan(frames._owned_teacher_object(raw))


def resources(input_bytes, plan):
    validate_plan(plan)
    require(type(input_bytes) is int and 0 < input_bytes <= MAX_PLAN_BYTES, 'bounded charged plan required')
    maximum_rows = max(sum(lane['rows'] for lane in row['lanes']) for row in plan['rounds'])
    total_frames = ACTIVE + sum(sum(lane['rows'] for lane in row['lanes']) for row in plan['rounds'])
    # Reuse the existing metadata/copy/writer projection for one streaming
    # window. The new protocol retains bounded references, not all raw trees.
    projection = frames.owned_teacher_resources(input_bytes, 0, VOCAB, maximum_rows)
    from tensorfold.families.qwen4_exp.cuda.retention_resources import _json_payload_policy, _json_tree_payload
    prompt_calls = (sum(len(case['prompt_ids']) for case in plan['sequences'])+255)//256 + ACTIVE
    mtp_picks = ACTIVE*DEPTH + DEPTH*len(plan['rounds'])
    mtp_stages = prompt_calls + mtp_picks
    index_bytes = (len(plan['rounds']) + ACTIVE + mtp_picks + mtp_stages) * 256
    index_payload = _json_tree_payload(index_bytes, _json_payload_policy())
    projection['per_rank_payload_bytes'] += index_payload
    projection['global_host_obligation_bytes'] += WORLD * index_payload
    from tensorfold.families.qwen4_exp.cuda.retention_resources import _json_parse_payload
    policy = _json_payload_policy()
    # Sealer retains one summary/index, startup, setup and current window plus
    # its normalized tree. Fund their source-declared read ceilings explicitly.
    parent_extra = sum(_json_parse_payload(size, policy)['total_bytes'] + _json_tree_payload(size, policy)
        for size in (8 << 20, 8 << 20, projection['receipt_bytes_limit'], projection['receipt_bytes_limit']))
    parent_extra += _json_tree_payload(projection['receipt_bytes_limit'], policy)
    from tensorfold.families.qwen4_exp.cuda.retention_resources import _python_layout, _streamed_json_encoding_payload
    parent_extra += _json_tree_payload(8 << 20, policy)
    parent_extra += _streamed_json_encoding_payload(8 << 20, _python_layout())['total_bytes']
    projection['global_host_obligation_bytes'] += parent_extra
    projection['parent_payload_bytes'] += parent_extra
    projection['authority_bytes_limit'] = 8 << 20
    metadata_bound = WORLD*((len(plan['rounds'])+2)*projection['receipt_bytes_limit'] + (16+2*ACTIVE)*(1 << 20))
    metadata_bound += projection['authority_bytes_limit'] + (48 << 20)
    metadata_bound += WORLD*(mtp_stages+mtp_picks)*MTP_RECORD_BYTES
    return {**projection, 'retained_index_raw_bound': index_bytes, 'retained_index_payload_bytes': index_payload, 'schema': 'tensorfold.target-window-teacher-resources-v1',
            'metadata_output_bytes_bound': metadata_bound, 'maximum_mtp_stages': mtp_stages,
            'maximum_mtp_picks': mtp_picks, 'mtp_record_bytes_limit': MTP_RECORD_BYTES, 'parent_seal_extra_payload_bytes': parent_extra,
            'maximum_window_rows': maximum_rows, 'maximum_windows': len(plan['rounds']),
            'maximum_total_frames': total_frames,
            'max_total_frame_file_bytes': total_frames * (4*VOCAB + 4096),
            'additional_project_device_bytes': 4*VOCAB,
            'device_scratch': {'shape': [WORLD, ROW_WIDTH], 'dtype': 'float32', 'payload_bytes': 4*VOCAB,
                'ownership': 'dedicated pre-admitted gather destination; live target rows never borrowed'},
            'scope': 'streamed per-window payload; allocator/native/RSS measurements remain separate; no P3 claim'}


CAPACITY_CONTRACT = 'full-window-physical-headroom-v1'
CAPACITY_PROOF_SCHEMA = 'tensorfold.window-full-capacity-proof-v1'
_CAPACITY_FIELDS = ('native_window', 'requested_context', 'context_window', 'cache_slots', 'budget_bytes',
    'startup_peak_bytes_estimate', 'serving_peak_bytes_estimate', 'total_bytes_estimate',
    'full_window_headroom', 'session_capacity', 'kv_full_windows', 'execution_envelope',
    'observed_graph_admission', 'multi_graphs', 'resource_ledger_bytes')


def _capacity_contract(operation):
    value = operation.get('capacity_contract')
    require('capacity_contract' not in operation or value == CAPACITY_CONTRACT, 'unknown window capacity contract')
    return value


def _capacity_integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, 'invalid capacity integer: '+name)
    return value


def _validate_capacity_plan(cap, *, slots, bits, local_rank=0):
    from tensorfold.families.qwen4_exp.cuda.service_memory import full_window_graph_budget
    require(type(slots) is int and type(bits) is int and (slots, bits) in ((24, 8), (32, 4)), 'unsupported full-window capacity region')
    number = _capacity_integer
    require(type(cap) is dict and set(cap) == set(_CAPACITY_FIELDS), 'complete capacity projection required')
    physical = number(cap['cache_slots'], 'physical capacity', 262144+DEPTH+1)
    require(all(type(cap[k]) is int and cap[k] == 262144 for k in
                ('native_window', 'requested_context', 'context_window'))
            and cap['cache_slots'] == physical, 'full native context and speculative physical tail required')
    marker, sessions, windows = (cap[k] for k in ('full_window_headroom', 'session_capacity', 'kv_full_windows'))
    require(type(marker) is dict and marker.get('policy') == CAPACITY_CONTRACT
            and type(marker.get('required_sessions')) is int and marker['required_sessions'] == slots
            and marker.get('logical_context') == 262144 and type(marker.get('logical_context')) is int
            and type(marker.get('unused_device_floor_bytes')) is int and marker['unused_device_floor_bytes'] == 0,
            'zero floor requires its scoped full-window policy')
    require(type(sessions) is dict and type(sessions.get('context_tokens')) is int and sessions['context_tokens'] == 262144
            and all(type(sessions.get(k)) is int and sessions[k] == slots for k in
                    ('configured_ceiling', 'resident_capacity', 'full_context_commitments')),
            'explicit full resident capacity and ceiling required')
    ranks = sessions.get('ranks')
    require(type(ranks) is list and len(ranks) == WORLD, 'all eight allocation baselines required')
    caps = []
    for rank, row in enumerate(ranks):
        require(type(row) is dict and type(row.get('rank')) is int and row['rank'] == rank
                and type(row.get('allocation_baseline')) is dict, 'ordered actual rank allocation baseline required')
        baseline = row['allocation_baseline']
        free = number(baseline.get('physical_free_bytes'), 'baseline physical free')
        total = number(baseline.get('device_total_bytes'), 'device total', 1)
        preallocated = number(baseline.get('preallocated_project_bytes'), 'preallocated project')
        require(free <= total and type(baseline.get('effective_device_floor_bytes')) is int
                and baseline['effective_device_floor_bytes'] == 0, 'scoped zero-floor baseline required')
        absolute, remaining = baseline.get('absolute_grant_cap_bytes'), baseline.get('remaining_absolute_grant_bytes')
        if absolute is None:
            require(remaining is None, 'absent absolute grant cannot create remaining credit')
        else:
            number(absolute, 'absolute grant', 1)
            require(absolute >= preallocated and type(remaining) is int and remaining == absolute-preallocated,
                    'absolute grant must deduct preallocated owners exactly once')
        usable = free if remaining is None else min(free, remaining)
        require(type(baseline.get('usable_serving_budget_bytes')) is int
                and baseline['usable_serving_budget_bytes'] == usable
                and type(row.get('budget_bytes')) is int and row['budget_bytes'] == usable,
                'rank physical/absolute startup budgets differ')
        ledger = row.get('resource_ledger_bytes')
        require(type(ledger) is dict and type(ledger.get('serving_workspace')) is int,
                'actual rank workspace ledger required')
        caps.append(absolute)
        require(all(type(v) is int and v >= 0 for v in ledger.values()), 'invalid rank resource ledger bytes')
        resident = number(row.get('resident_weight_bytes'), 'resident weight')
        loading = number(row.get('loading_overlap_bytes'), 'loading overlap')
        execution = number(row.get('session_execution_bytes'), 'session execution')
        require(resident + max(loading, execution) <= usable, 'rank loading/complete service exceeds its grant')
    budget = full_window_graph_budget(cap)
    require(number(cap['budget_bytes'], 'project budget') == budget,
            'project ceiling must equal the worst physical/absolute rank grant')
    for key in ('startup_peak_bytes_estimate', 'serving_peak_bytes_estimate', 'total_bytes_estimate'):
        number(cap[key], key)
    require(cap['total_bytes_estimate'] == max(cap['startup_peak_bytes_estimate'], cap['serving_peak_bytes_estimate'])
            and cap['total_bytes_estimate'] <= budget, 'complete startup/serving peak exceeds project budget')
    require(type(windows) is dict and windows.get('materialized') is True
            and all(type(windows.get(k)) is int and windows[k] == slots for k in ('requested', 'reserved', 'stream_slots'))
            and type(windows.get('context_window')) is int and windows['context_window'] == 262144
            and type(windows.get('cache_slots_each')) is int and windows['cache_slots_each'] == physical,
            'every complete physical full window must be materialized')
    workspace = number(windows.get('workspace_bytes'), 'workspace', 2 << 30)
    require(type(cap['resource_ledger_bytes']) is dict
            and cap['resource_ledger_bytes'].get('serving_workspace') == workspace
            and all(row['resource_ledger_bytes']['serving_workspace'] == workspace for row in ranks),
            'workspace differs from complete rank service promises')
    allocated = windows.get('ranks')
    require(type(allocated) is list and len(allocated) == WORLD, 'all eight materialized rank receipts required')
    for rank, row in enumerate(allocated):
        require(type(row) is dict and type(row.get('rank')) is int and row['rank'] == rank
                and row.get('materialized') is True and type(row.get('startup_capacities')) is list
                and len(row['startup_capacities']) == slots
                and all(type(n) is int and n >= physical for n in row['startup_capacities']),
                'rank does not own all full physical windows')
        cache = number(row.get('allocated_cache_bytes'), 'owned cache', 1)
        active = number(row.get('torch_allocated_bytes'), 'rank active')
        reserved = number(row.get('torch_reserved_bytes'), 'rank reserved')
        require(cache <= active <= reserved, 'physical allocator cannot cover owned windows')
    envelope = cap['execution_envelope']
    require(type(envelope) is dict and envelope.get('method') == 'mtp'
            and type(envelope.get('max_proposals')) is int and envelope['max_proposals'] == DEPTH
            and type(envelope.get('target_rows')) is int and envelope['target_rows'] == slots*(DEPTH+1)
            and type(envelope.get('prefill_tokens')) is int and envelope['prefill_tokens'] == 256,
            'full MTP4 row/prefill envelope required')
    graph, admission = cap['multi_graphs'], cap['observed_graph_admission']
    require(type(graph) is dict and graph.get('automatic') is True and graph.get('captured') is True
            and graph.get('reason') is None and graph.get('stopped') is None and graph.get('rejected') is None
            and type(admission) is dict and admission.get('policy') == CAPACITY_CONTRACT
            and type(graph.get('observed_graph_admission')) is dict
            and {k:v for k,v in graph['observed_graph_admission'].items() if k != 'basis'} == {k:v for k,v in admission.items()
                if k not in ('original_header', 'observed_serving_base_bytes', 'basis')}
            and type(admission.get('admitted_project_budget_bytes')) is int
            and admission['admitted_project_budget_bytes'] == budget
            and type(admission.get('live_reserve_bytes')) is int and admission['live_reserve_bytes'] == 0,
            'current captured graph must retain its actual scoped admission')
    observations = admission.get('observations')
    require(type(observations) is list and len(observations) == WORLD, 'all-rank graph headroom observations required')
    rooms = []
    for rank, observation in enumerate(observations):
        require(type(observation) is dict and type(observation.get('rank')) is int and observation['rank'] == rank,
                'ordered graph observation required')
        live = number(observation.get('live_after_floor_bytes'), 'observed live')
        future = number(observation.get('future_service_bytes'), 'observed future', 2 << 30)
        room = number(observation.get('graph_room_bytes'), 'observed graph room')
        require(live >= future and room == min(live-future, budget), 'graph observation borrowed promised future storage')
        rooms.append(room)
    agreed = number(admission.get('agreed_graph_bytes'), 'agreed graph budget', 1)
    require(agreed == min(rooms) and type(graph.get('planned_bytes')) is int and type(graph.get('budget_bytes')) is int
            and graph['planned_bytes'] == graph['budget_bytes'] == agreed,
            'graph budget must be the all-rank minimum')
    future = admission.get('future_service')
    require(type(future) is dict and graph.get('future_service') == future
            and all(type(future.get(k)) is int and future[k] >= 0 for k in
                    ('growth_bytes', 'copy_overlap_bytes', 'snapshot_bytes_each', 'snapshot_bytes', 'transient_bytes',
                     'workspace_bytes', 'future_bytes'))
            and future['growth_bytes'] == 0 and future['workspace_bytes'] == workspace
            and future['future_bytes'] == sum(future[k] for k in
                    ('growth_bytes', 'copy_overlap_bytes', 'snapshot_bytes', 'workspace_bytes'))
            and all(row['future_service_bytes'] == future['future_bytes'] for row in observations),
            'unchanged complete future service/workspace proof required')
    require(type(admission.get('observed_serving_base_bytes')) is int
            and admission['observed_serving_base_bytes'] == budget-agreed
            and cap['serving_peak_bytes_estimate'] == admission['observed_serving_base_bytes']+agreed,
            'graph reservation differs from refined serving envelope')
    catalogue = graph.get('exact_graph_catalogue', {})
    require(type(catalogue) is dict and type(catalogue.get('requested_keys')) is list
            and catalogue['requested_keys'] and type(catalogue.get('captured_keys')) is list
            and catalogue.get('declined_keys') == [], 'complete requested graph catalogue required')
    keys = catalogue['requested_keys']
    require(all(type(key) is list and len(key) == 3 and key[0] in ('verify','mtp')
            and type(key[1]) is int and type(key[2]) is int and 0 < key[2] <= slots
            and key[2] <= key[1] <= slots*(DEPTH+1) for key in [*keys,*catalogue['captured_keys']])
            and len({tuple(key) for key in keys}) == len(keys), 'invalid/duplicate full-window graph key')
    captured = catalogue['captured_keys']
    require(len(captured) == len(keys) and len({tuple(key) for key in captured}) == len(captured)
            and {tuple(key) for key in captured} == {tuple(key) for key in keys},
            'complete unique requested graph catalogue required')
    measured = graph.get('keys')
    require(type(measured) is list and len(measured) == len(keys), 'every requested key needs its actual capture accounting')
    charged, pieces = 0, 0
    for index, item in enumerate(measured):
        require(type(item) is dict and type(item.get('accounting')) is dict, 'actual per-key capture accounting required')
        row = item['accounting']
        key = [row.get('role'), row.get('rows'), row.get('participants')]
        require(key == catalogue['captured_keys'][index] and row.get('schema') == 'tensorfold.exact-graph-key-v1'
                and row.get('admitted') is True and type(row.get('rank')) is int and row['rank'] == local_rank
                and type(row.get('prior_keys')) is int and row['prior_keys'] == index
                and type(row.get('prior_bytes')) is int and row['prior_bytes'] == charged
                and type(row.get('physical_delta_bytes')) is int and type(row.get('budget_bytes')) is int
                and row['budget_bytes'] == agreed and type(row.get('pieces')) is int and row['pieces'] > 0
                and type(item.get('bytes')) is int and item['bytes'] == max(0,row['physical_delta_bytes'])
                and type(item.get('pieces')) is int and item['pieces'] == row['pieces'],
                'graph key memory/ownership differs from its settled capture')
        charged += item['bytes']
        pieces += item['pieces']
    require(graph.get('bytes') == charged and graph.get('pieces') == pieces
            and graph.get('verify_keys') == sum(key[0] == 'verify' for key in keys)
            and graph.get('mtp_keys') == sum(key[0] == 'mtp' for key in keys)
            and graph.get('capture_during_serving') is False, 'graph capture totals/catalogue differ')
    required = {(role,count*rows,count) for count in (1,2,4,8,16,24)
                for rows in (range(1,6) if count == 1 else (1,5)) for role in ('verify','mtp')}
    required.update({('verify',28,8), ('mtp',28,8), *(('mtp',n,n) for n in (3,5,6,7))})
    require(required <= {tuple(key) for key in keys}, 'full-window proof dropped required C36 keys')
    if slots == 32:
        require(all(key in catalogue['requested_keys'] for key in
                    (['verify',32,32], ['mtp',32,32], ['verify',160,32], ['mtp',160,32])),
                '32-slot capacity proof lacks full E160 graph keys')
    require(type(graph.get('settled_bytes')) is int, 'actual signed capture settlement required')
    room = number(graph.get('room_bytes'), 'retained graph room')
    require(room == max(number(graph.get('bytes'), 'graph bytes'), graph.get('settled_bytes', 0))
            and room <= agreed, 'settled private graph retention exceeds admission')
    allocator = graph.get('allocator')
    require(type(allocator) is dict, 'actual graph allocator ownership required')
    private = allocator.get('private_pool_spare_bytes')
    if private is None:
        require(allocator.get('accounting') == 'physical-only' and all(cap is None for cap in caps),
                'unmeasured private storage cannot support absolute-grant headroom')
    else:
        number(private, 'private pool spare')
        require(allocator.get('backend') == 'native' and allocator.get('accounting') == 'native-pool-inactive',
                'private spare must belong to the measured native pool')
    return {'physical':physical, 'future':future, 'absolute_caps':caps}


def validate_capacity_proof(proof, *, retention, slots=24, bits=8):
    """The two user-required full native regions; no eight-active capacity inference."""
    require(type(slots) is int and type(bits) is int and (slots, bits) in ((24, 8), (32, 4)), 'unsupported full-window capacity region')
    fields = {'schema', 'contract', 'rank', 'slots', 'bits', 'physical_capacity', 'capacity',
              'states', 'owner', 'sample', 'retention_sha256', 'kv_pair'}
    require(type(proof) is dict and set(proof) == fields and proof['schema'] == CAPACITY_PROOF_SCHEMA
            and proof['contract'] == CAPACITY_CONTRACT and type(proof['rank']) is int
            and 0 <= proof['rank'] < WORLD and type(proof['slots']) is int and proof['slots'] == slots
            and type(proof['bits']) is int and proof['bits'] == bits
            and legacy.sha(proof['retention_sha256']), 'closed full-window capacity proof required')
    from tensorfold.families.qwen4_exp.cuda.execution_identity import digest
    require(type(retention) is dict and proof['retention_sha256'] == digest(retention)
            and type(retention.get('keep')) is int and retention['keep'] >= 0
            and type(retention.get('recovery_points_per_piece')) is int and retention['recovery_points_per_piece'] >= 0,
            'capacity proof requires its actual issued recovery retention')
    from tensorfold.families.qwen4_exp.kv_formats import get_pair
    kv = proof['kv_pair']
    require(type(kv) is dict and set(kv) == {'key_dtype','value_dtype','identity'}
            and all(type(v) is str for v in kv.values()), 'actual typed KV formats required')
    pair = get_pair('bf16', kv['key_dtype'], kv['value_dtype'])
    require(pair.identity == kv['identity'] and pair.key.bits == pair.value.bits == bits,
            'capacity proof changed actual KV storage/codec identity')
    cap = proof['capacity']
    validated = _validate_capacity_plan(cap, slots=slots, bits=bits, local_rank=proof['rank'])
    physical, future = validated['physical'], validated['future']
    require(future['snapshot_bytes'] == future['snapshot_bytes_each']*(
                retention['keep'] + slots*retention['recovery_points_per_piece']),
            'future service snapshot promise differs from actual retained recovery owners')
    allocated = cap['kv_full_windows']['ranks']
    number = _capacity_integer
    require(type(proof['physical_capacity']) is int and proof['physical_capacity'] == physical,
            'local physical capacity differs from the complete admission')
    states = proof['states']
    require(type(states) is list and len(states) == slots, 'all actual State owners required')
    for index, state in enumerate(states):
        require(type(state) is dict and set(state) == {'slot', 'logical_context', 'capacity', 'limit', 'pos', 'mtp_len'}
                and all(type(n) is int for n in state.values()) and state['slot'] == index
                and state['logical_context'] == 262144 and state['capacity'] >= physical and state['limit'] >= physical
                and state['pos'] == state['mtp_len'] == 0
                and state['capacity'] == allocated[proof['rank']]['startup_capacities'][index],
                'actual idle State differs from all-rank materialized physical ownership')
    require(type(proof['owner']) is dict and all(type(proof['owner'].get(k)) is bool for k in
                ('admitted','bound_closed','failed_owner','capture_failure','preparation_closed','prepared'))
            and type(proof['owner'].get('pending')) is int
            and proof['owner'] == {'admitted':True, 'bound_closed':False, 'pending':0,
                'failed_owner':False, 'capture_failure':False, 'preparation_closed':False, 'prepared':True},
            'failed/pending/retired private owners cannot create startup headroom')
    sample = proof['sample']
    require(type(sample) is dict and set(sample) == {'physical_free_bytes','device_total_bytes','allocated_bytes',
                'reserved_bytes','live_bytes','observed_at_unix_ns','interval_monotonic_ns','ownership'}, 'fresh physical sample required')
    require(number(sample['physical_free_bytes'], 'sample free') <= number(sample['device_total_bytes'], 'sample total', 1)
            and number(sample['allocated_bytes'], 'sample active') <= number(sample['reserved_bytes'], 'sample reserved')
            and number(sample['live_bytes'], 'current usable live') >= future['future_bytes'],
            'current graph/private ownership cannot preserve future service')
    require(sample['device_total_bytes'] == cap['session_capacity']['ranks'][proof['rank']]['allocation_baseline']['device_total_bytes'],
            'fresh sample device differs from rank baseline')
    ownership = sample['ownership']
    require(type(ownership) is dict and set(ownership) == {'prepared_private_spare_bytes',
        'prepared_outside_active_bytes','mapped_bytes','absolute_cap_bytes','graph_outside_allocator_bytes'},
        'actual private/mapped/driver ownership required')
    prepared_private = ownership['prepared_private_spare_bytes']
    outside = ownership['prepared_outside_active_bytes']
    for value, label in ((prepared_private,'prepared private spare'), (outside,'prepared outside active')):
        require(value is None or type(value) is int and value >= 0, 'unmeasured '+label+' cannot be inferred zero')
    mapped = number(ownership['mapped_bytes'], 'mapped bytes')
    graph_outside = number(ownership['graph_outside_allocator_bytes'], 'graph outside allocator')
    private = cap['multi_graphs']['allocator']['private_pool_spare_bytes']
    ordinary = (max(0,sample['reserved_bytes']-sample['allocated_bytes']-prepared_private-private)
                if prepared_private is not None and private is not None else 0)
    upper = sample['physical_free_bytes'] + ordinary if private is not None else sample['physical_free_bytes']
    absolute = ownership['absolute_cap_bytes']
    require(absolute == validated['absolute_caps'][proof['rank']], 'current absolute grant differs from startup')
    if absolute is not None:
        require(outside is not None and private is not None, 'unknown outside/private storage cannot support an absolute grant')
        upper = max(0,min(upper,absolute-sample['allocated_bytes']-mapped-outside-graph_outside-private))
    require(sample['live_bytes'] <= upper, 'live room credited private/mapped/outside ownership or exceeded physical grant')
    number(sample['observed_at_unix_ns'], 'sample time')
    interval = sample['interval_monotonic_ns']
    require(type(interval) is list and len(interval) == 2 and all(type(n) is int and n >= 0 for n in interval)
            and interval[0] <= interval[1], 'actual physical sample interval required')
    return proof


def _legacy_capacity_plan(cap):
    rows = cap.get('session_capacity', {}).get('ranks', [])
    require(type(rows) is list and all(type(row) is dict for row in rows), 'invalid legacy rank capacity metadata')
    require(cap.get('full_window_headroom') is None and all(
        type(row.get('allocation_baseline', {})) is dict
        and row.get('allocation_baseline', {}).get('effective_device_floor_bytes') != 0 for row in rows),
        'scoped zero-floor engine requires an explicit capacity contract')


def _runtime_capacity_proof(engine, operation, retention, *, slots=24, bits=8):
    from copy import deepcopy
    contract = _capacity_contract(operation)
    cap = engine.capacity_plan
    if contract is None:
        _legacy_capacity_plan(cap)
        return None
    import time

    import torch

    from tensorfold.cuda.capacity import cuda_limit_bytes
    from tensorfold.cuda.kv_arena import mapped_bytes
    from tensorfold.families.qwen4_exp.cuda.execution_identity import digest
    graphs, prepared = engine.multi.graphs, engine._owner_preparation
    require(graphs is not None and graphs.dec is engine.multi and graphs.bound.owner is graphs
            and graphs.bound.dec is engine.multi and len(engine.multi.states) == slots and len({id(state) for state in engine.multi.states}) == slots,
            'actual distinct full-window owners required')
    start, wall = time.monotonic_ns(), time.time_ns()
    free, total = map(int, torch.cuda.mem_get_info(engine.w.device))
    sample = {'physical_free_bytes':free, 'device_total_bytes':total,
        'allocated_bytes':int(torch.cuda.memory_allocated(engine.w.device)),
        'reserved_bytes':int(torch.cuda.memory_reserved(engine.w.device)),
        'live_bytes':engine.multi.memory_gate.live(), 'observed_at_unix_ns':wall,
        'interval_monotonic_ns':[start,time.monotonic_ns()],
        'ownership':{'prepared_private_spare_bytes':prepared.private_spare_bytes,
            'prepared_outside_active_bytes':prepared.outside_active_bytes, 'mapped_bytes':mapped_bytes(),
            'absolute_cap_bytes':cuda_limit_bytes(),
            'graph_outside_allocator_bytes':max(0,graphs.room_bytes-graphs.capture_reserved_delta_bytes)}}
    proof = {'schema':CAPACITY_PROOF_SCHEMA, 'contract':contract, 'rank':engine.rank, 'slots':slots, 'bits':bits,
        'physical_capacity':engine.multi.capacity,
        'kv_pair':{'key_dtype':engine.kv_pair.key_dtype, 'value_dtype':engine.kv_pair.value_dtype, 'identity':engine.kv_pair.identity},
        'capacity':deepcopy({k:cap[k] for k in _CAPACITY_FIELDS}),
        'states':[{'slot':i, **{k:getattr(state,k) for k in ('logical_context','capacity','limit','pos','mtp_len')}}
                  for i,state in enumerate(engine.multi.states)],
        'owner':{'admitted':graphs.admitted, 'bound_closed':graphs.bound.closed, 'pending':len(graphs.pending),
            'failed_owner':engine.multi._failed_graphs is not None, 'capture_failure':graphs.failure is not None,
            'preparation_closed':prepared.closed, 'prepared':prepared.prepared}, 'sample':sample, 'retention_sha256':digest(retention)}
    validate_capacity_proof(proof, retention=retention, slots=slots, bits=bits)
    _validate_capacity_catalogue(proof['capacity'], operation)
    return proof


def _validate_capacity_catalogue(capacity, operation):
    catalogue = capacity['multi_graphs']['exact_graph_catalogue']
    policy = operation['prepared_execution'].get('exact_graph_policy')
    if policy is None:
        require(catalogue.get('policy') == 'full-original-v1'
                and catalogue.get('requested_keys') == catalogue.get('available_keys'),
                'full original graph catalogue was reduced')
    else:
        require(catalogue.get('policy') == policy['policy']
                and catalogue.get('evidence_sha256') == policy['evidence_sha256']
                and {tuple(key) for key in catalogue['requested_keys']} == {tuple(key) for key in policy['keys']},
                'capacity proof dropped or changed requested graph keys')


def _validate_startup_capacity(profile, operation):
    contract = _capacity_contract(operation)
    require(('full_window_capacity' in profile) == (contract is not None), 'capacity contract/proof presence differs')
    if contract is None:
        return
    from tensorfold.families.qwen4_exp.cuda.execution_identity import digest
    proof = validate_capacity_proof(profile['full_window_capacity'], retention=profile.get('numerical_trial_retention'))
    _validate_capacity_catalogue(proof['capacity'], operation)
    from tensorfold.families.qwen4_exp.kv_formats import get_pair
    require(proof['kv_pair'] == {'key_dtype':'int8','value_dtype':'int8','identity':get_pair('int8').identity},
            'C8 teacher capacity requires its actual ordinary int8 KV pair')
    registration = profile['teacher_execution_registration']
    require(proof['rank'] == profile['rank'] and proof['retention_sha256'] == registration['binding']['identity']['retention']
            and registration['binding']['identity'].get('codec') == digest({'pair':proof['kv_pair']['identity']})
            and registration.get('capacity_projection') == {'schema':CAPACITY_PROOF_SCHEMA,
                'capacity_sha256':digest(proof), 'retention_sha256':proof['retention_sha256']},
            'capacity proof is not bound to actual startup retention')


def graph_counts(graphs):
    from tensorfold.families.qwen4_exp.cuda.exact_graphs import ExactStateGraphs
    require(graphs is not None and type(getattr(graphs, 'bound', None)) is ExactStateGraphs,
            'target-window teacher requires its actual exact-graph owner')
    raw = graphs.record()
    result = {}
    for role in ('serving_hits', 'serving_misses'):
        counts = raw.get(role)
        require(type(counts) is dict and set(counts) == {'verify', 'mtp'}
                and all(type(n) is int and n >= 0 for n in counts.values()),
                'actual per-role serving graph counts required')
        result[role] = dict(counts)
    return result


class WindowTeacher:
    """Observe and force explicit windows on the production MultiDecoder.

    The sampler still runs and its original choices are recorded. Only this
    private operation replaces sampler-returned rows and the accepted path
    with explicit gold tokens/keep schedule, including MTP refresh conditioning. Production stage, target/MTP computation, commit and State owners
    remain the implementation. No forced choice is advertised as acceptance.
    """
    write_vector = frames.OwnedTeacherFrames.write_vector
    abort = frames.OwnedTeacherFrames.abort
    _transport_status = frames.OwnedTeacherFrames._transport_status

    def __init__(self, engine, prepared, output, *, fail, profile):
        import threading

        import torch

        from tensorfold.cleanup import raise_failures
        self.engine, self.decoder, self.prepared = engine, engine.multi, prepared
        self.fail, self.raise_failures, self.profile, self.torch = fail, raise_failures, profile, torch
        self.directory = Path(output)/f'window-teacher-rank{engine.rank}'
        self.failed, self.abort_status = None, [None, None]
        self.thread_id, self.thread = threading.get_ident, None
        self.delegating = self.closed = self.close_attempted = False
        self.round_index = self.frame_index = 0
        self.streams, self.by_state, self.originals, self.restores = {}, {}, {}, []
        self.window_refs, self.prefill_refs, self.mtp_stage_refs, self.mtp_pick_refs = [], [], [], []
        self.last_mtp_stage, self.mtp_pick_step = None, 0
        self.current, self.staged, self.sampled, self.accepted = None, [], [], []
        self.graph_events = []
        self.stream = self.gathered = self.cpu = None
        self.finished = set()
        data, budget = validate_plan(prepared['data']), prepared['resources']
        require(engine.tp == WORLD and engine.sm70_profile == 'reference' and engine.w.mtp is not None
                and self.decoder.speculation.method == 'mtp' and self.decoder.depth == DEPTH
                and self.decoder.confidence == data['mtp_confidence']
                and len(self.decoder.states) == 24 and self.decoder.full_resident
                and self.decoder.buf.rows == 120 and self.decoder.prefill_rows == 256
                and self.decoder.mbuf is not None and self.decoder.share == 0 and engine.context_window == data['context'],
                'actual C8 MTP4/full24/E120/P256 owner required')
        require(engine._owned_qualification_host == (budget['global_host_obligation_bytes'], prepared['input_sha256'])
                and getattr(engine, '_owned_qualification_device', None) ==
                (budget['additional_project_device_bytes'], prepared['input_sha256']),
                'target-window staging must be funded before original capacity admission')
        widths = list(self.decoder._sample_plan.widths)
        require(widths == [ROW_WIDTH]*WORLD and self.decoder._sample_plan.mapping is None
                and engine.w.meta['vocab_offset'] == engine.rank*ROW_WIDTH,
                'actual contiguous complete TP8 vocabulary required')
        self.initial_graphs = graph_counts(self.decoder.graphs)
        require(not self.decoder.streams and not self.decoder.filling, 'cold empty service required before teacher installation')
        self.widths = widths
        self.runtime_sources = {**frames._owned_runtime_sources(engine),
            'window_teacher': frames._owned_teacher_hash(Path(__file__), Path(__file__).stat().st_size, 8 << 20)}
        self.directory.mkdir(mode=0o700)
        self.gathered = torch.empty((VOCAB,), dtype=torch.float32, device=engine.w.device)
        require(self.gathered.numel()*self.gathered.element_size() == budget['additional_project_device_bytes'],
                'actual dedicated gather payload differs from its admission')
        if engine.rank == 0:
            self.cpu = torch.empty((VOCAB,), dtype=torch.float32, device='cpu')
        for name in ('admit', 'round', 'finish', '_runnable', '_picks'):
            self.originals[name] = getattr(self.decoder, name)
        self.old_close = engine.close

    def _guard(self, call, *args, **kwargs):
        try:
            return call(*args, **kwargs)
        except BaseException as primary:
            self.abort(primary)
            raise

    def _snapshot(self, stream):
        state = stream.st
        return {'case_id': self.streams[stream.sid][0]['case_id'], 'sid': stream.sid,
                'history_slot': state.history_slot, 'slot_generation': state.slot_generation,
                'state_pos': state.pos, 'mtp_len': state.mtp_len, 'mtp_drafted': state.mtp_drafted,
                'output_tokens': len(stream.out), 'kv_identity': state.kv_identity,
                'draft_tokens': list(stream.drafts), 'done': stream.done}

    def _capture(self, stream, source, position, *, role, row, rows, packed_row, batch_rows):
        torch = self.torch
        case = self.streams[stream.sid][0]
        offset = position - len(case['prompt_ids'])
        require(0 <= offset < len(case['forced_ids']) and tuple(source.shape) == (ROW_WIDTH,)
                and source.dtype == torch.float32 and source.is_cuda and source.is_contiguous()
                and not source.requires_grad and source.device == self.gathered.device,
                'complete raw target vocabulary shard and explicit gold position required')
        start, end = source.data_ptr(), source.data_ptr()+4*source.numel()
        other = self.gathered.data_ptr()
        require(end <= other or start >= other+4*VOCAB, 'live target source aliases the gather scratch')
        require(self.frame_index < self.prepared['resources']['maximum_total_frames'],
                'teacher frame count exhausted before any new output write')
        self.engine.comm.all_gather(source, self.gathered)
        if self.cpu is not None:
            self.cpu.copy_(self.gathered)
        torch.cuda.current_stream(source.device).synchronize()
        self.stream = stream  # The shared bounded writer uses this cancellation owner.
        vector = self.write_vector(self.frame_index) if self.engine.rank == 0 else None
        record = {**self._snapshot(stream), 'frame': self.frame_index, 'window': self.round_index if role != 'prefill' else None,
                  'role': role, 'absolute_position': position, 'gold_token': case['forced_ids'][offset],
                  'input_token': case['prompt_ids'][-1] if role == 'prefill' else case['forced_ids'][offset-1],
                  'input_position': position-1, 'rope_delta': stream.st.rope_delta,
                  'input_coordinate': position-1+stream.st.rope_delta,
                  'window_row': row, 'target_window_rows': rows, 'packed_row': packed_row,
                  'batch_target_rows': batch_rows, 'active_states': 8 if role != 'prefill' else 1,
                  'physical_target_rows': self.decoder.buf.rows, 'vocab': VOCAB, 'vector': vector,
                  'method': 'mtp', 'depth': DEPTH, 'teacher_forced': True}
        self.frame_index += 1
        require(self.frame_index <= self.prepared['resources']['maximum_total_frames'], 'teacher frame extent exceeded')
        return record

    def admitted(self, stream):
        index = len(self.streams)
        require(index < ACTIVE and stream.sid not in self.streams, 'exactly eight fresh request admissions required')
        case = self.prepared['data']['sequences'][index]
        require(list(stream.prompt) == case['prompt_ids'] and stream.count == len(case['forced_ids'])
                and stream.draft is True and stream.stop_eos is False
                and (stream.sampling is None or stream.sampling.temperature == 0)
                and stream.vision is None and stream.constraint is None and stream.penalties is None
                and stream.probabilities is None and stream.carry is None and not stream.owed
                and not stream.background and stream.copies is None and not stream.stops
                and getattr(stream, 'retention_request', None) is None,
                'ordered cold greedy MTP request differs from its explicit forced case')
        result = self.originals['admit'](stream)
        require(stream.cached == 0 and stream.st is not None and id(stream.st) not in self.by_state,
                'forced windows require unique fresh States with no prefix reuse')
        self.streams[stream.sid] = (case, stream)
        self.by_state[id(stream.st)] = stream
        frames._owned_teacher_json(self.directory/f'admission-{index}.json', {
            'schema': 'tensorfold.target-window-teacher-admission-v1', 'index': index,
            'case_id': case['case_id'], 'worker_PID': frames.os.getpid(), 'rank': self.engine.rank,
            'input_sha256': self.prepared['input_sha256'], 'state': self._snapshot(stream)}, 1 << 20)
        return result

    def runnable(self, stream):
        ready = len(self.streams) == ACTIVE and not self.decoder.filling and all(
            current.sid in self.decoder.streams and current.out for _, current in self.streams.values())
        return getattr(self, 'window_enabled', False) and ready and self.originals['_runnable'](stream)

    def _first_sample(self, original, slot, logits, positions, sampling, **kwargs):
        if slot.w is not self.engine.w or id(slot.st) not in self.by_state:
            return original(slot, logits, positions, sampling, **kwargs)
        if kwargs.get('draft', False):
            result = original(slot, logits, positions, sampling, **kwargs)
            require(len(positions) == 1 and type(result) is list and len(result) == 1, 'one serial proposal pick required')
            token = self._record_pick('serial-sample', [self.by_state[id(slot.st)]], list(positions), [(result[0], None)])[0]
            return [token]
        stream = self.by_state[id(slot.st)]
        case = self.streams[stream.sid][0]
        require(not stream.out and list(positions) == [len(case['prompt_ids'])] and tuple(logits.shape) == (1, ROW_WIDTH),
                'exact first target row required before forced-window admission')
        record = self._capture(stream, logits[0], positions[0], role='prefill', row=0, rows=1, packed_row=0, batch_rows=1)
        result = original(slot, logits, positions, sampling, **kwargs)
        require(type(result) is list and len(result) == 1 and type(result[0]) is int, 'original prefill sampler outcome required')
        record['original_sampled'] = result[0]
        require(len(frames.json.dumps(record, allow_nan=False).encode()) <= 4096,
                'completed prefill frame metadata exceeds its declared envelope')
        path = self.directory/f'prefill-{len(self.prefill_refs)}.json'
        frames._owned_teacher_json(path, record, 1 << 20)
        self.prefill_refs.append({'file': path.name, 'bytes': path.stat().st_size,
                                 'sha256': frames._owned_teacher_hash(path, path.stat().st_size, 1 << 20)})
        return [case['forced_ids'][0]]

    def _stage(self, original, weights, buffers, windows, **kwargs):
        if weights is not self.engine.w or (buffers is not self.decoder.buf and buffers is not self.decoder.pbuf):
            return original(weights, buffers, windows, **kwargs)
        prefill = buffers is self.decoder.pbuf
        self._fixed_policy()
        require(windows and len(windows) <= ACTIVE, 'bounded actual target staging cohort required')
        if not prefill:
            require(self.current is not None and not self.staged and len(windows) == ACTIVE,
                    'one actual eight-segment production target stage required')
        expected, total = [], 0
        for state, tokens in windows:
            require(id(state) in self.by_state and state.image_positions is None,
                    'target stage must borrow an actual admitted text State')
            stream = self.by_state[id(state)]
            case = self.streams[stream.sid][0]
            require(list(tokens) == self._gold_span(case, state.pos, len(tokens))
                    and (not prefill or state.pos+len(tokens) <= len(case['prompt_ids'])),
                    'target compute would consume a non-gold prefix')
            expected.extend(tokens)
            total += len(tokens)
        result = original(weights, buffers, windows, **kwargs)
        require(len(result) == len(windows), 'actual target stage participant count differs')
        buffers.staged.synchronize()
        require(buffers.ids_host[:total].tolist() == expected, 'actual host-staged target IDs differ from declared gold')
        by_case = {} if prefill else {lane['case_id']: lane for lane in self.current['lanes']}
        for (state, begin, end), (actual_state, tokens) in zip(result, windows):
            require(state is actual_state and end-begin == len(tokens), 'staged State/row identity differs')
            if not prefill:
                stream = self.by_state[id(state)]
                case = self.streams[stream.sid][0]
                lane = by_case[case['case_id']]
                require(end-begin == lane['rows'], 'actual target shape differs from frozen window')
                self.staged.append((stream, begin, end, lane))
        if not prefill:
            require(len({stream.sid for stream, *_ in self.staged}) == ACTIVE, 'target stage omitted an active case')
        return result

    def _sample(self, original, args, kwargs, *, gathered):
        plan = kwargs.get('plan')
        if plan is not self.decoder._sample_plan or self.delegating:
            return original(*args, **kwargs)
        require(self.current is not None and len(self.staged) == ACTIVE and not self.sampled,
                'one complete eight-case target sampler required')
        logits, positions = (kwargs.get('logits'), args[3]) if gathered else (args[1], args[2])
        total = sum(end-begin for _, begin, end, _ in self.staged)
        require(tuple(logits.shape) == (total, ROW_WIDTH) and logits.data_ptr() == self.decoder.buf.logits.data_ptr(),
                'sampler source must be the complete original production target window')
        expected_positions, records = [], []
        for stream, begin, end, _ in self.staged:
            case = self.streams[stream.sid][0]
            at = len(stream.out)
            for row in range(end-begin):
                position = len(case['prompt_ids'])+at+row
                expected_positions.append(position)
                records.append(self._capture(stream, logits[begin+row], position,
                    role='target-verify', row=row, rows=end-begin, packed_row=begin+row, batch_rows=total))
        require(list(positions) == expected_positions, 'actual target sampling positions differ')
        self.delegating = True
        try:
            result = original(*args, **kwargs)
        finally:
            self.delegating = False
        require(type(result) is list and len(result) == total and all(type(token) is int for token in result),
                'original complete target sampler outcomes required')
        for row, token in zip(records, result):
            row['original_sampled'] = token
            require(len(frames.json.dumps(row, allow_nan=False).encode()) <= 4096,
                    'completed target frame metadata exceeds its declared envelope')
        self.sampled = records
        # Production _round forwards these kept rows to _draft_all. Force the
        # same gold inputs there as well; retaining original argmax rows here
        # would silently condition MTP on a different prefix in each arm.
        return [row['gold_token'] for row in records]

    def _accept(self, original, tokens, parents, sampled, room, eos=()):
        require(len(self.accepted) < len(self.staged), 'unplanned target acceptance call')
        stream, begin, end, lane = self.staged[len(self.accepted)]
        case = self.streams[stream.sid][0]
        at, keep = len(stream.out), lane['keep']
        require(list(tokens) == case['forced_ids'][at-1:at-1+lane['rows']] and list(parents) == list(range(-1, lane['rows']-1))
                and room == len(case['forced_ids'])-at and not eos and keep <= room,
                'production acceptance input/remaining room differs from frozen schedule')
        rows = self.sampled[begin:end]
        require(len(rows) == lane['rows'] and all(row['case_id'] == case['case_id'] for row in rows)
                and list(sampled) == [row['gold_token'] for row in rows] == case['forced_ids'][at:at+lane['rows']],
                'production MTP conditioning must consume the explicit gold target rows')
        original_path, original_terminal = original(tokens, parents, [row['original_sampled'] for row in rows], room, eos)
        terminal = case['forced_ids'][at+keep-1]
        path = list(range(keep))
        self.accepted.append({'case_id': case['case_id'], 'original_path': list(original_path),
            'original_terminal': original_terminal, 'forced_path': path, 'forced_terminal': terminal,
            'role': 'diagnostic-forced-keep-not-serving-acceptance',
            'original_scope': 'counterfactual-forced-proposals-and-prefix-v1'})
        return path, terminal

    def _fixed_policy(self):
        controller = self.decoder.controller
        require(controller is not None and controller.effective.get('prefill_tokens') == 256
                and controller.effective.get('row_budget') == 120
                and self.decoder._proposal_limit() == DEPTH,
                'forced measurement cannot change its original prefill/row/proposal envelope')

    def _gold_span(self, case, start, count):
        prompt, gold = case['prompt_ids'], case['forced_ids']
        require(type(start) is int and type(count) is int and start >= 0 and count > 0
                and start+count <= len(prompt)+len(gold), 'MTP input span has no explicit gold labels')
        result = prompt[start:min(start+count, len(prompt))] if start < len(prompt) else []
        if start+count > len(prompt):
            result = [*result, *gold[max(0, start-len(prompt)):start+count-len(prompt)]]
        return list(result)

    def _mtp_file(self, kind, record):
        refs = self.mtp_stage_refs if kind == 'stage' else self.mtp_pick_refs
        maximum = self.prepared['resources']['maximum_mtp_stages' if kind == 'stage' else 'maximum_mtp_picks']
        require(len(refs) < maximum, 'declared MTP callback/file extent exhausted before publication')
        path = self.directory/f'mtp-{kind}-{len(refs):06d}.json'
        frames._owned_teacher_json(path, record, MTP_RECORD_BYTES)
        ref = {'file': path.name, 'bytes': path.stat().st_size,
               'sha256': frames._owned_teacher_hash(path, path.stat().st_size, MTP_RECORD_BYTES)}
        require(len(frames.json.dumps(ref, indent=1).encode()) <= 256, 'bounded MTP index reference exceeded')
        refs.append(ref)
        return ref

    def _mtp_stage(self, original, buffers, windows, **kwargs):
        if buffers is not self.decoder.mbuf and buffers is not self.decoder.pbuf:
            return original(buffers, windows, **kwargs)
        self._fixed_policy()
        require(windows and len(windows) <= ACTIVE and self.decoder._proposal_limit() == DEPTH,
                'bounded actual fixed-depth MTP cohort required')
        lanes, flattened, offset = [], [], 0
        for state, tokens, _hidden in windows:
            require(id(state) in self.by_state and state.image_positions is None and type(state.rope_delta) is int,
                    'actual admitted text State/coordinates required for MTP staging')
            stream = self.by_state[id(state)]
            case = self.streams[stream.sid][0]
            count, start = len(tokens), state.mtp_len+1
            require(count > 0 and list(tokens) == self._gold_span(case, start, count),
                    'MTP compute would consume a non-gold prefix')
            lanes.append({'case_id': case['case_id'], 'sid': stream.sid, 'history_slot': state.history_slot,
                'slot_generation': state.slot_generation, 'cache_start': state.mtp_len,
                'coordinate_start': state.mtp_len+state.rope_delta, 'rope_delta': state.rope_delta,
                'token_position_start': start, 'token_ids': list(tokens), 'packed_first': offset, 'packed_last': offset+count})
            flattened.extend(tokens)
            offset += count
        require(offset <= buffers.rows and len({lane['case_id'] for lane in lanes}) == len(lanes),
                'MTP input cohort exceeds its actual unique-State buffer')
        result = original(buffers, windows, **kwargs)
        require(len(result) == len(windows) and all(actual is expected and begin == row['packed_first'] and end == row['packed_last']
                    for (actual, begin, end), (expected, _ids, _hidden), row in zip(result, windows, lanes)),
                'actual MTP staging layout differs from its input authority')
        buffers.staged.synchronize()
        require(buffers.ids_host[:offset].tolist() == flattened, 'actual MTP host-staged token IDs differ from gold')
        require(buffers.ids[:offset].tolist() == flattened, 'actual MTP device-staged token IDs differ from gold')
        from tensorfold.families.qwen4_exp.cuda.compact_indexer import validate_mtp_indexer_frame

        coordinates = validate_mtp_indexer_frame(buffers, result)
        require(coordinates is not None and coordinates.tolist() == [
            [coordinate]*3 for lane in lanes
            for coordinate in range(lane['coordinate_start'], lane['coordinate_start']+len(lane['token_ids']))],
            'actual MTP staged coordinates differ from the declared text prefix')
        record = {'schema': 'tensorfold.target-window-teacher-mtp-stage-v1', 'rank': self.engine.rank,
            'worker_PID': frames.os.getpid(), 'input_sha256': self.prepared['input_sha256'],
            'stage_index': len(self.mtp_stage_refs), 'window': self.round_index if self.current is not None else None,
            'participants': len(lanes), 'rows': offset, 'lanes': lanes,
            'kind': 'prefill' if buffers is self.decoder.pbuf else 'mtp'}
        self._mtp_file('stage', record)
        self.last_mtp_stage = record
        return result

    def _record_pick(self, branch, streams, positions, original, *, proposal_index=None):
        require(type(original) is list and len(original) == len(streams) == len(positions) and self.last_mtp_stage is not None
                and self.decoder._proposal_limit() == DEPTH, 'actual fixed-depth MTP picks/staging required')
        require(branch != 'serial-sample' or self.prepared['data']['mtp_confidence'] == 0,
                'probability-free serial picks require the actual zero-confidence branch')
        require(branch != 'serial-sample-draft' or self.prepared['data']['mtp_confidence'] > 0,
                'serial confidence picks require the actual configured confidence branch')
        supplied = {row['case_id']: row for row in self.last_mtp_stage['lanes']}
        rows, forced = [], []
        for stream, position, (chosen, probability) in zip(streams, positions, original):
            case = self.streams[stream.sid][0]
            require(case['case_id'] in supplied and type(position) is int,
                    'MTP pick lacks its actual supplying State stage')
            lane = supplied[case['case_id']]
            require(position == lane['token_position_start']+len(lane['token_ids'])
                    and (stream.st.history_slot, stream.st.slot_generation) == (lane['history_slot'], lane['slot_generation']),
                    'MTP pick input/position/generation linkage differs')
            if self.current is None:
                index = position-len(case['prompt_ids'])-1
                cap = self.decoder._proposal_count(stream, stream.count-1)
            else:
                index = proposal_index
                keep = next(row['keep'] for row in self.current['lanes'] if row['case_id'] == case['case_id'])
                cap = self.decoder._proposal_count(stream, stream.count-len(stream.out)-keep)
            offset = position-len(case['prompt_ids'])
            require(type(index) is int and 0 <= index < cap <= DEPTH and 0 <= offset < len(case['forced_ids'])
                    and type(chosen) is int and 0 <= chosen < VOCAB
                    and (probability is None if branch == 'serial-sample' else
                         type(probability) in (int, float) and math.isfinite(probability) and probability >= 0),
                    'actual MTP sample or cap-aware full-depth label differs')
            token = case['forced_ids'][offset]
            rows.append({'case_id': case['case_id'], 'sid': stream.sid, 'history_slot': stream.st.history_slot,
                'slot_generation': stream.st.slot_generation, 'absolute_position': position, 'gold_token': token,
                'original_token': chosen, 'original_probability': probability, 'forced_token': token,
                'forced_probability': 1.0, 'continuation_policy': PROPOSAL_POLICY,
                'proposal_cap': cap, 'proposal_index': index})
            forced.append(token)
        record = {'schema': 'tensorfold.target-window-teacher-mtp-pick-v1', 'rank': self.engine.rank,
            'worker_PID': frames.os.getpid(), 'input_sha256': self.prepared['input_sha256'],
            'pick_index': len(self.mtp_pick_refs), 'stage_index': self.last_mtp_stage['stage_index'],
            'window': self.round_index if self.current is not None else None, 'branch': branch,
            'source_kind': 'prefill' if self.current is None else 'mtp', 'rows': rows}
        self._mtp_file('pick', record)
        return forced

    def _serial_pick(self, original, slot, logits, position, sampling):
        result = original(slot, logits, position, sampling)
        if slot.w is not self.engine.w or id(slot.st) not in self.by_state:
            return result
        require(type(result) is tuple and len(result) == 2, 'original serial MTP token/probability pair required')
        token = self._record_pick('serial-sample-draft', [self.by_state[id(slot.st)]], [position], [result])[0]
        return token, 1.0  # Explicit diagnostic continuation, never an asserted model probability.

    def _mtp_picks(self, logits, positions, samplings, words):
        result = self.originals['_picks'](logits, positions, samplings, words)
        require(self.current is not None and len(self.staged) == ACTIVE, 'batched MTP picks require their actual target window')
        step = self.mtp_pick_step
        active = []
        for stream, _begin, _end, lane in self.staged:
            cap = self.decoder._proposal_count(stream, stream.count-len(stream.out)-lane['keep'])
            if cap > step:
                active.append(stream)
        require(len(active) == len(positions), 'actual MTP proposal participation differs from fixed cap schedule')
        forced = self._record_pick('multi-picks', active, list(positions), result, proposal_index=step)
        self.mtp_pick_step += 1
        return [(token, 1.0) for token in forced]

    def rounded(self):
        from contextlib import contextmanager

        from tensorfold.families.qwen4_exp.cuda import decode, mtp, multi
        require(self.thread is None and self.failed is None, 'nonreentrant healthy target-window round required')
        if len(self.streams) < ACTIVE:
            return []  # Explicit cold admission barrier; no prompt/cache work before all8 requests exist.
        ready = len(self.streams) == ACTIVE and not self.decoder.filling and all(
            stream.sid in self.decoder.streams and stream.out for _, stream in self.streams.values())
        self.window_enabled = ready and self.round_index < len(self.prepared['data']['rounds'])
        self.current = self.prepared['data']['rounds'][self.round_index] if self.window_enabled else None
        self.staged, self.sampled, self.accepted, self.graph_events = [], [], [], []
        self.mtp_pick_step = 0
        before = {case['case_id']: self._snapshot(stream) for case, stream in self.streams.values()}
        counts_before = graph_counts(self.decoder.graphs)
        if self.current is not None:
            for lane in self.current['lanes']:
                case, stream = next(pair for pair in self.streams.values() if pair[0]['case_id'] == lane['case_id'])
                at = len(stream.out)
                require(list(stream.out) == case['forced_ids'][:at], 'committed output prefix drifted before target stage')
                stream.drafts = list(case['forced_ids'][at:at+lane['rows']-1])
        aliases = []
        original_sample = decode.Engine.sample
        def first(slot, *args, **kwargs):
            return self._guard(self._first_sample, original_sample, slot, *args, **kwargs)
        aliases.append((decode.Engine, 'sample', original_sample, first))
        original_draft = decode.Engine.sample_draft
        def draft_pick(slot, *args, **kwargs):
            return self._guard(self._serial_pick, original_draft, slot, *args, **kwargs)
        aliases.append((decode.Engine, 'sample_draft', original_draft, draft_pick))
        original_mtp_stage = mtp._mtp_stage
        def mtp_staged(*args, **kwargs):
            return self._guard(self._mtp_stage, original_mtp_stage, *args, **kwargs)
        aliases.append((mtp, '_mtp_stage', original_mtp_stage, mtp_staged))
        for name, gathered in (('choose_gathered', True), ('tp_sample_rows', False)):
            original = getattr(multi, name)
            def selected(*args, _original=original, _gathered=gathered, **kwargs):
                return self._guard(self._sample, _original, args, kwargs, gathered=_gathered)
            aliases.append((multi, name, original, selected))
        original_stage, original_accept = multi.stage, multi.accept
        aliases.extend(((multi, 'stage', original_stage,
                         lambda *a, **kw: self._guard(self._stage, original_stage, *a, **kw)),
                        (multi, 'accept', original_accept,
                         lambda *a, **kw: self._guard(self._accept, original_accept, *a, **kw))))
        for name, role in (('verify_forward', 'verify'), ('mtp_step', 'mtp')):
            original = getattr(self.decoder.graphs, name)
            def observed(*args, _original=original, _role=role, **kwargs):
                return self._guard(self._graph_call, _original, _role, *args, **kwargs)
            aliases.append((self.decoder.graphs, name, original, observed))
        @contextmanager
        def installed():
            from types import MethodType
            primary, errors, applied = None, [], []
            try:
                for module, name, original, replacement in aliases:
                    current = getattr(module, name)
                    same = current is original or (type(current) is MethodType and type(original) is MethodType
                        and current.__self__ is original.__self__ and current.__func__ is original.__func__)
                    require(same, 'target-window alias changed before its transaction')
                    existed, old = name in vars(module), vars(module).get(name)
                    setattr(module, name, replacement)
                    applied.append((module, name, original, replacement, existed, old))
                yield
            except BaseException as error:  # noqa: BLE001 - retain the original failure while every alias is restored
                primary = error
            for module, name, original, replacement, existed, old in reversed(applied):
                try:
                    require(getattr(module, name) is replacement, 'target-window alias changed during its transaction')
                    if existed:
                        setattr(module, name, old)
                    else:
                        delattr(module, name)
                except BaseException as error:  # noqa: BLE001 - collect alias cleanup failures before propagating the primary
                    errors.append(error)
            self.raise_failures(primary, errors)
        self.thread = self.thread_id()
        try:
            with installed():
                result = self.originals['round']()
        finally:
            self.thread = None
            self.window_enabled = False
        if self.current is not None:
            require(len(self.staged) == ACTIVE and len(self.accepted) == ACTIVE
                    and len(self.sampled) == sum(lane['rows'] for lane in self.current['lanes']),
                    'production round did not exercise its complete declared target window')
            after = {case['case_id']: self._snapshot(stream) for case, stream in self.streams.values()}
            for lane in self.current['lanes']:
                case = lane['case_id']
                require(after[case]['state_pos'] == before[case]['state_pos']+lane['keep']
                        and after[case]['output_tokens'] == before[case]['output_tokens']+lane['keep']
                        and all(after[case][key] == before[case][key] for key in ('sid', 'history_slot', 'slot_generation', 'kv_identity')),
                        'production commit did not preserve the exact forced State transaction')
            self.torch.cuda.current_stream(self.engine.w.device).synchronize()
            counts_after = graph_counts(self.decoder.graphs)
            delta = {kind: {role: counts_after[kind][role]-counts_before[kind][role]
                            for role in ('verify', 'mtp')} for kind in counts_before}
            require(all(n >= 0 for counts in delta.values() for n in counts.values()), 'graph counters regressed')
            self._write_window({'schema': 'tensorfold.target-window-teacher-window-v1', 'rank': self.engine.rank,
                'worker_PID': frames.os.getpid(), 'window': self.round_index,
                'input_sha256': self.prepared['input_sha256'], 'schedule': self.current,
                'before': before, 'after': after, 'frames': self.sampled, 'acceptance': self.accepted,
                'graph_counts_before': counts_before, 'graph_counts_after': counts_after, 'graph_deltas': delta,
                'graph_calls': self.graph_events,
                'method': 'mtp', 'depth': DEPTH, 'active_states': ACTIVE,
                'role': 'forced-target-window', 'qualification': False, 'P3_valid': False})
            self.round_index += 1
        return result

    def _graph_call(self, original, role, segs, **kwargs):
        before = graph_counts(self.decoder.graphs)
        result = original(segs, **kwargs)
        after = graph_counts(self.decoder.graphs)
        hit = after['serving_hits'][role]-before['serving_hits'][role]
        miss = after['serving_misses'][role]-before['serving_misses'][role]
        require(hit in (0, 1) and miss in (0, 1) and hit+miss == 1
                and (result is not None) == (hit == 1), 'actual graph call/result/counter linkage differs')
        require(len(self.graph_events) < DEPTH+1, 'target/MTP graph-call extent exceeds one frozen window')
        self.graph_events.append({'role': role, 'participants': len(segs),
            'rows': segs[-1][2] if segs else 0, 'serving_hits_delta': hit, 'serving_misses_delta': miss,
            'history_slots': [state.history_slot for state, _begin, _end in segs],
            'slot_generations': [state.slot_generation for state, _begin, _end in segs]})
        return result

    def _release(self):
        # All source consumers retired, or setup failed before listening and
        # every callback was restored. Metadata survives; tensor owners do not.
        self.gathered = self.cpu = self.stream = None
        self.streams.clear()
        self.by_state.clear()
        self.originals.clear()
        self.staged = []
        self.restores = []
        self.old_close = None
        self.last_mtp_stage = None
        self.decoder = self.engine = None

    def _write_window(self, value):
        path = self.directory/f'window-{self.round_index:06d}.json'
        maximum = self.prepared['resources']['receipt_bytes_limit']
        frames._owned_teacher_json(path, value, maximum)
        self.window_refs.append({'file': path.name, 'bytes': path.stat().st_size,
                                 'sha256': frames._owned_teacher_hash(path, path.stat().st_size, maximum)})

    def finish(self, done):
        for stream in done:
            require(stream.sid in self.streams, 'unplanned teacher request completion')
            case = self.streams[stream.sid][0]
            require(stream.error is None and list(stream.out) == case['forced_ids']
                    and self.round_index == len(self.prepared['data']['rounds']),
                    'forced request ended before its complete planned State path')
            self.finished.add(case['case_id'])
        return self.originals['finish'](done)

    def close(self):
        require(not self.close_attempted, 'target-window Engine.close cannot retry uncertain retirement')
        self.close_attempted = True
        primary, errors, retired = self.failed, [], False
        final_counts = None
        try:
            final_counts = graph_counts(self.decoder.graphs)
        except BaseException as error:  # noqa: BLE001 - retire the engine even when evidence observation fails
            if primary is None:
                primary = error
            else:
                errors.append(error)
        try:
            self.old_close()
            with self.engine._lifecycle:
                require(self.engine._closed is True, 'target-window engine did not retire its owned lifecycle')
            retired = True
        except BaseException as error:  # noqa: BLE001 - preserve retirement failure alongside the original outcome
            if primary is None:
                primary = error
            elif error is not primary:
                errors.append(error)
        if retired:
            self.closed = True
            for owner, name, existed, old, replacement in reversed(self.restores):
                try:
                    require(getattr(owner, name) is replacement, 'target-window installed callback changed before retirement')
                    if existed:
                        setattr(owner, name, old)
                    else:
                        delattr(owner, name)
                except BaseException as error:  # noqa: BLE001 - finish every callback restoration before propagating cleanup failures
                    errors.append(error)
        try:
            require(retired and len(self.finished) == ACTIVE and self.round_index == len(self.prepared['data']['rounds'])
                    and self.frame_index == self.prepared['resources']['maximum_total_frames'],
                    'target-window teacher lacks complete requests/windows/frames/retirement')
            # Global counts may include prefill. Only window deltas above can
            # prove actual eight-participant verify/MTP graph execution.
            coverage = {'verify': False, 'mtp': False}
            for ref in self.window_refs:
                raw = frames._owned_teacher_read(self.directory/ref['file'], ref['bytes'])
                require(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'window evidence changed before retirement')
                window = frames._owned_teacher_object(raw)
                for role in coverage:
                    coverage[role] |= any(event['role'] == role and event['participants'] == ACTIVE
                        and event['serving_hits_delta'] > 0 for event in window['graph_calls'])
            require(all(coverage.values()), 'actual C8 target and MTP graph replay evidence required')
        except BaseException as error:  # noqa: BLE001 - publish incomplete coverage without replacing the primary failure
            if primary is None:
                primary = error
            else:
                errors.append(error)
        try:
            frames._owned_teacher_json(self.directory/'summary.json', {
                'schema': 'tensorfold.target-window-teacher-rank-result-v1', 'rank': self.engine.rank,
                'worker_PID': frames.os.getpid(), 'input_sha256': self.prepared['input_sha256'],
                'closed': retired, 'finished': sorted(self.finished), 'failed': primary is not None or bool(errors),
                'profile': self.profile, 'runtime_sources': self.runtime_sources, 'prefill': self.prefill_refs, 'windows': self.window_refs,
                'mtp_stages': self.mtp_stage_refs, 'mtp_picks': self.mtp_pick_refs,
                'frame_count': self.frame_index, 'initial_graph_counts': self.initial_graphs,
                'final_graph_counts': final_counts, 'resources': self.prepared['resources'],
                'quality_scope': self.prepared['data']['quality_scope'], 'metrics': METRICS,
                'qualification': False, 'P3_valid': False}, self.prepared['resources']['receipt_bytes_limit'])
        except BaseException as error:  # noqa: BLE001 - release retired owners even when report publication fails
            if primary is None:
                primary = error
            else:
                errors.append(error)
        if retired:
            self._release()
        self.raise_failures(primary, errors)

    def install(self):
        def bind(owner, name, replacement):
            existed, old = name in vars(owner), vars(owner).get(name)
            self.restores.append((owner, name, existed, old, replacement))
            setattr(owner, name, replacement)
        try:
            for name, operation in (('admit', self.admitted), ('round', self.rounded),
                                    ('finish', self.finish), ('_runnable', self.runnable), ('_picks', self._mtp_picks)):
                def guarded(*args, _operation=operation, **kwargs):
                    return self._guard(_operation, *args, **kwargs)
                bind(self.decoder, name, guarded)
            bind(self.engine, 'close', self.close)
        except BaseException as primary:  # noqa: BLE001 - roll back partial callback installation before propagating failure
            errors = []
            for owner, name, existed, old, replacement in reversed(self.restores):
                try:
                    if existed:
                        setattr(owner, name, old)
                    elif name in vars(owner):
                        delattr(owner, name)
                except BaseException as error:  # noqa: BLE001 - complete the callback rollback and preserve all cleanup errors
                    errors.append(error)
            if not errors:
                self._release()
            self.raise_failures(primary, errors)
        return {'schema': 'tensorfold.target-window-teacher-setup-v1', 'input_sha256': self.prepared['input_sha256'],
                'actual_profile': self.profile, 'runtime_sources': self.runtime_sources, 'resources': self.prepared['resources'],
                'writer': 'existing-owned-full-vocabulary-npy-writer', 'qualification': False, 'P3_valid': False}


def _legacy_validate_preparation(value, trial):
    numerical.validate_trial(trial)
    require(type(value) is dict, 'complete explicit window preparation required')
    extra = {'native_gdn_capture', 'native_hc_writeback_capture', 'exact_graph_policy'}
    require(extra <= set(value) and value['native_gdn_capture'] is True and value['native_hc_writeback_capture'] is True,
            'C8 teacher requires actual native exact target/MTP capture preparation')
    numerical.preparation_projection({k: v for k, v in value.items() if k not in extra}, trial)
    policy = value['exact_graph_policy']
    require(type(policy) is dict and set(policy) == {'policy', 'keys', 'evidence_sha256'}
            and policy['policy'] == 'paid-subset-v1' and legacy.sha(policy['evidence_sha256'])
            and type(policy['keys']) is list and 1 <= len(policy['keys']) <= 30,
            'explicit finite exact-graph catalogue authority required')
    legal = {(role, rows, count) for role in ('verify', 'mtp') for count in (1, 2, 4, 8, 16, 24)
             for rows in (range(1, 6) if count == 1 else (count, 5*count))}
    keys = []
    for row in policy['keys']:
        require(type(row) is list and len(row) == 3 and type(row[0]) is str
                and all(type(n) is int for n in row[1:]) and tuple(row) in legal and row not in keys,
                'exact graph key differs from the supported finite catalogue')
        keys.append(row)
    require(any(row[0] == 'verify' and row[2] == ACTIVE for row in keys)
            and any(row[0] == 'mtp' and row[2] == ACTIVE for row in keys),
            'actual eight-participant target and MTP graph keys required; PAID10 is insufficient')
    return value


PREPARATION_CONTRACT = 'source30-rn12-candidates-v1'
ATEN_PREPARATION_CONTRACT = 'source33-aten-q8-layouts-v1'
ATEN_SCOPE = legacy.ATEN_WINDOW_TEACHER_SCOPE
ATEN_FAMILY = 'aten-cutlass-q8-normalized-lse-owner-v1'
ATEN_PANEL_ROWS = 16
ATEN_LAYOUTS = ('native-gqa-v1', 'fixed12-head-as-query-lse-v1')
TRANSPORT_OPTIONS = {'prepared_router': 'owned-serial-router-v1',
    'row_sampling_scratch': 'canonical-row-scratch-v1',
    'coordinate_staging': 'owned-host-int64-coordinates-v1',
    'descriptor_upload': 'pinned-descriptor-ring2-v1', 'owned_q8_window_dispatch': 'grid-v1'}
NUMERICAL_OPTIONS = {'dense_projections', 'native_hc_writeback_target_eager', 'gdn_value8_capture'}
PREPARATION_OPTIONS = set(TRANSPORT_OPTIONS) | NUMERICAL_OPTIONS | set(numerical.AXES)


def validate_preparation(value, trial, contract=None):
    if contract is None:
        return _legacy_validate_preparation(value, trial)
    if contract == ATEN_PREPARATION_CONTRACT:
        return _validate_aten_preparation(value, trial)
    require(contract == PREPARATION_CONTRACT, 'unsupported versioned C8 preparation contract')
    numerical.validate_trial(trial)
    require(type(value) is dict and value.get('native_gdn_capture') is True
            and value.get('native_hc_writeback_capture') is True,
            'actual native target/MTP capture preparation required')
    require('row_buckets' in value and 'collective_family' in value,
            'versioned C8 preparation requires a canonical manifest with explicit owner defaults')
    extras = {'native_gdn_capture', 'native_hc_writeback_capture', 'exact_graph_policy', *TRANSPORT_OPTIONS, *NUMERICAL_OPTIONS}
    numerical.preparation_projection({k: v for k, v in value.items() if k not in extras}, trial)
    # The outer Host is deliberately Torch-absent. These finite DATA domains
    # are rechecked by the actual runtime owner parser before admission; no
    # runtime implementation, provider loading or numeric identity is duplicated.
    for key, expected in TRANSPORT_OPTIONS.items():
        require(key not in value or type(value[key]) is str and value[key] == expected,
                'unsupported explicit C8 transport policy: '+key)
    require('owned_q8_window_dispatch' not in value or trial['owned_q8_window_prepare'],
            'Q8 grid dispatch requires its explicit Q8 window axis')
    for key in NUMERICAL_OPTIONS - {'dense_projections'}:
        require(key not in value or value[key] is True, 'explicit C8 numerical selection must be true: '+key)
    dense = value.get('dense_projections')
    if dense is not None:
        require(type(dense) is dict and set(dense) == {'backend', 'geometry', 'roles', 'panel_rows'}
                and dense['backend'] in ('cublas', 'cublaslt', 'cutlass')
                and dense['geometry'] in ('fixed-m32', 'direct')
                and (dense['backend'] == 'cutlass' or dense['geometry'] == 'fixed-m32')
                and type(dense['roles']) is list and dense['roles'] and all(type(role) is str and role in ('decode', 'prefill') for role in dense['roles'])
                and len(set(dense['roles'])) == len(dense['roles'])
                and type(dense['panel_rows']) is int and 0 < dense['panel_rows'] <= (1 << 31)-1
                and (dense['geometry'] != 'fixed-m32' or dense['panel_rows'] == 32),
                'C8 vendor dense selection differs from its supported runtime contract')
        require(not trial['prepared_linears'] or dense['roles'] == ['prefill'],
                'prepared M32 and vendor dense projections must retain disjoint roles')
    require('dense_projections' not in value or dense is not None, 'absent vendor selection must be omitted')
    return _validate_window_graph_policy(value)



def _validate_aten_preparation(value, trial):
    numerical.validate_trial(trial)
    require(not any(trial.values()), 'attention-family contract leaves the three independent RN numerical axes off')
    required = {'kernel_family', 'graph_family', 'graph_budget_bytes', 'trial', 'collective_family',
                'row_buckets', 'native_gdn_capture', 'native_hc_writeback_capture', 'exact_graph_policy'}
    require(type(value) is dict and value.get('kernel_family') in (SERVICE['kernel_family'], ATEN_FAMILY),
            'explicit RN12 reference or ATen Q8 family required')
    aten = value['kernel_family'] == ATEN_FAMILY
    require(set(value) == required | ({'aten_tiles'} if aten else {'rn_heads_per_block'})
            and value['graph_family'] == 'uncaptured' and type(value['graph_budget_bytes']) is int
            and value['graph_budget_bytes'] == 0 and value['trial'] is True
            and value['collective_family'] == 'tp8-allgather-v1' and type(value['row_buckets']) is dict
            and value['row_buckets'] == {} and value['native_gdn_capture'] is True
            and value['native_hc_writeback_capture'] is True,
            'attention-family contract requires its canonical isolated C8 preparation')
    if aten:
        tiles = value['aten_tiles']
        require(type(tiles) is dict and {'panel_rows', 'tensor_budget_bytes'} <= set(tiles)
                and set(tiles) <= {'panel_rows', 'tensor_budget_bytes', 'query_layout'}
                and type(tiles['panel_rows']) is int and tiles['panel_rows'] == ATEN_PANEL_ROWS
                and type(tiles['tensor_budget_bytes']) is int and 0 < tiles['tensor_budget_bytes'] < 2**63
                and ('query_layout' not in tiles or type(tiles['query_layout']) is str
                     and tiles['query_layout'] in ATEN_LAYOUTS),
                'explicit paid panel16 ATen default/native-GQA/head-row layout required')
    else:
        require(type(value['rn_heads_per_block']) is int and value['rn_heads_per_block'] == 12,
                'attention-family reference must remain RN12')
    require(type(value.get('exact_graph_policy')) is dict
            and type(value['exact_graph_policy'].get('keys')) is list
            and len(value['exact_graph_policy']['keys']) <= 128,
            'attention-family graph catalogue exceeds its finite runtime bound')
    return _validate_window_graph_policy(value)


def _validate_window_graph_policy(value):
    policy = value.get('exact_graph_policy')
    require(type(policy) is dict and set(policy) == {'policy', 'keys', 'evidence_sha256'}
            and policy['policy'] in ('paid-subset-v1', 'observed-demand-v1') and legacy.sha(policy['evidence_sha256'])
            and type(policy['keys']) is list and policy['keys'], 'explicit bounded exact graph policy required')
    keys = []
    for row in policy['keys']:
        require(type(row) is list and len(row) == 3 and row[0] in ('verify', 'mtp')
                and all(type(n) is int for n in row[1:]) and 1 <= row[2] <= 24
                and row[2] <= row[1] <= min(5*row[2], 120),
                'exact graph demand exceeds actual C8 teacher State/row/depth envelope')
        if policy['policy'] == 'paid-subset-v1':
            require(row[2] in (1, 2, 4, 8, 16, 24)
                    and row[1] in (range(1, 6) if row[2] == 1 else (row[2], 5*row[2])),
                    'paid graph key differs from the unchanged original catalogue')
        keys.append(tuple(row))
    require(keys == sorted(set(keys)) and any(k[0] == 'verify' and k[2] == 8 for k in keys)
            and any(k[0] == 'mtp' and k[2] == 8 for k in keys),
            'canonical unique catalogue and actual eight-participant target/MTP keys required')
    return value


def _teacher_scope(value):
    return ATEN_SCOPE if _preparation_contract(value) == ATEN_PREPARATION_CONTRACT else SCOPE


def expected_service(value):
    if _preparation_contract(value) != ATEN_PREPARATION_CONTRACT:
        return dict(SERVICE)
    preparation = value.get('prepared_execution')
    require(type(preparation) is dict and preparation.get('kernel_family') in (SERVICE['kernel_family'], ATEN_FAMILY),
            'closed attention service family required')
    return {**SERVICE, 'kernel_family': preparation['kernel_family']}

def _preparation_contract(value):
    if 'preparation_contract' not in value:
        return None
    require(value['preparation_contract'] in (PREPARATION_CONTRACT, ATEN_PREPARATION_CONTRACT), 'unknown C8 preparation contract')
    return value['preparation_contract']


def _preparation_owner(operation):
    from tensorfold.families.qwen4_exp.cuda.owner_preparation import OwnerPreparation
    owner = OwnerPreparation(operation['prepared_execution'], role_rows={'decode': 120, 'mtp': 120, 'prefill': 256},
                             manifest_sha256=operation['execution_manifest_sha256'] if 'execution_manifest_sha256' in operation
                             else operation['execution_manifest']['sha256'])
    require(owner.specification == operation['prepared_execution'], 'C8 preparation DATA must equal actual canonical owner parsing')
    return owner


def _expected_preparation_numerics(operation):
    if _preparation_contract(operation) is None:
        return numerical.expected_numerics(operation['numerical_trial'], mtp=True)
    return _preparation_owner(operation).numerical_selection


def _validate_preparation_numerics(value, operation):
    expected = _expected_preparation_numerics(operation)
    require(type(value) is dict and ('owner_preparation_numerics' in value) == (expected is not None)
            and value.get('owner_preparation_numerics') == expected,
            'actual C8 preparation numerical registration differs from its runtime owner')



def _validate_attention_provider(profile, operation):
    """Bind builtin ATen metadata and the two actual protected Torch providers."""
    from tensorfold.cuda.sm70.attention_backends import POLICY
    from tensorfold.cuda.sm70.attention_backends import SCHEMA as operator_schema
    provider = profile.get('attention_provider')
    require(type(provider) is dict and set(provider) == {'operator', 'libraries', 'roles'},
            'actual attention SDK/operator/role registration required')
    operator = provider['operator']
    require(type(operator) is dict and operator.get('policy') == POLICY and operator.get('schema') == operator_schema
            and operator.get('torch') == '2.16.0.dev20261111+tfcu129sm70'
            and operator.get('torch_git') == '05adb434bd30d298eb408a72420b0496ee05f8e3'
            and operator.get('torch_cuda') == '12.9' and operator.get('cuda_registered') is True,
            'attention requires the genuine protected Source33 Torch/ATen stack')
    libraries = provider['libraries']
    require(type(libraries) is dict and set(libraries) == {'torch_C', 'torch_cuda'},
            'both actual builtin Torch providers required')
    for row in libraries.values():
        require(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'} and absolute(row['path'])
                and len(row['path']) <= MODEL_OBSERVER_LIMITS['path_characters']
                and type(row['bytes']) is int and 0 < row['bytes'] <= MODEL_OBSERVER_LIMITS['elf_bytes']
                and legacy.sha(row['sha256']), 'bounded actual builtin Torch library identity required')
    cpp, cuda = Path(libraries['torch_C']['path']), Path(libraries['torch_cuda']['path'])
    require(cpp.name == '_C.cpython-312-x86_64-linux-gnu.so' and cuda.name == 'libtorch_cuda.so'
            and cuda.parent.name == 'lib' and cpp.parent == cuda.parent.parent,
            'builtin Torch providers must share their actual ordinary installed package')
    contents = profile.get('teacher_execution_registration', {}).get('contents', {})
    require(all(contents.get('library/'+row['path']) == {'size':row['bytes'], 'sha256':row['sha256']}
                for row in libraries.values()), 'builtin Torch provider differs from issued complete library authority')
    selected = operation['prepared_execution']
    roles = provider['roles']
    aten = selected['kernel_family'] == ATEN_FAMILY
    require(type(roles) is dict and set(roles) == ({'decode', 'mtp', 'prefill'} if aten else set()),
            'actual attention family role catalogue differs')
    if aten:
        expected = _preparation_owner(operation)
        fields = {'family','requirements','operator','closed','numeric_contract',
                  'key_capacity_policy','prepared_panel_counts','qualification'}
        if 'query_layout' in selected['aten_tiles']:
            fields.add('query_layout')
        for role, row in roles.items():
            require(type(row) is dict and set(row) == fields
                    and row.get('numeric_contract') == 'Half normalized local values and natural-log LSE; ordered FP32 rank weights/merge'
                    and row.get('key_capacity_policy') == 'fixed2051 masked keys; no context-dependent shape or row dispatch'
                    and row.get('qualification') == 'unselected; operator/component/model/PPL/resource/P3 gates remain separate'
                    and row.get('family') == ATEN_FAMILY and row.get('closed') is False
                    and row.get('requirements') == expected.aten_requirements[role] and row.get('operator') == operator
                    and ('query_layout' in row) == ('query_layout' in selected['aten_tiles'])
                    and row.get('query_layout') == selected['aten_tiles'].get('query_layout')
                    and row.get('prepared_panel_counts') == list(range(1, ATEN_PANEL_ROWS+1)),
                    'actual paid ATen roles/layout differ from canonical runtime preparation')
    return provider


def _runtime_attention_provider(engine, operation, contents):
    import torch

    from tensorfold.cuda.sm70.attention_backends import OwnerTiles, operator_identity
    prepared = engine._owner_preparation
    paths = {'torch_C':Path(torch._C.__file__).resolve(strict=True),
             'torch_cuda':(Path(torch.__file__).resolve(strict=True).parent/'lib/libtorch_cuda.so').resolve(strict=True)}
    libraries = {}
    for name, path in paths.items():
        row = contents.get('library/'+str(path))
        require(type(row) is dict, 'actual protected Torch library was not issued into the model binding')
        libraries[name] = {'path':str(path), 'bytes':row['size'], 'sha256':row['sha256']}
    operator = operator_identity()
    operator['torch'] = str(operator['torch'])
    roles = {}
    aten = operation['prepared_execution']['kernel_family'] == ATEN_FAMILY
    require(set(prepared.workspaces) == {'decode', 'mtp', 'prefill'}, 'actual attention preparation roles required')
    for role, workspace in prepared.workspaces.items():
        require(workspace.native_kernel_family == operation['prepared_execution']['kernel_family'],
                'actual workspace attention family differs')
        tile = workspace.aten_tiles
        if aten:
            require(type(tile) is OwnerTiles and not tile.closed and tile.query_layout ==
                    operation['prepared_execution']['aten_tiles'].get('query_layout'),
                    'actual ATen layout capability differs from its requested policy')
            require(all(item.spec.query_layout == tile.query_layout for item in tile.panels[1:]),
                    'actual ATen panel descriptor layout differs')
            roles[role] = tile.record()
        else:
            require(tile is None and workspace.native_heads_per_block == 12,
                    'RN12 reference acquired an unrequested ATen capability')
    provider = {'operator':operator,'libraries':libraries,'roles':roles}
    _validate_attention_provider({'attention_provider':provider,
        'teacher_execution_registration':{'contents':contents}}, operation)
    # This tuple is consumed only by the existing positive-retirement observer;
    # no extra library is loaded and its original limits/reserve stay unchanged.
    engine._teacher_attention_provider_paths = tuple(row['path'] for row in libraries.values())
    return provider


def _validate_attention_vmas(profile, operation, native):
    provider = _validate_attention_provider(profile, operation)
    paths = [row['path'] for row in provider['libraries'].values()]
    require(native.get('required_provider_paths') == paths and native.get('missing_required_provider_paths') == [],
            'closed attention observer did not require the actual builtin Torch providers')
    for expected in provider['libraries'].values():
        require(any(row.get('path') == expected['path'] and row.get('bytes') == expected['bytes']
                    and row.get('sha256') == expected['sha256'] and type(row.get('vmas')) is list and row['vmas']
                    for row in native['mapped_elfs']),
                'builtin attention Torch provider lacks its actual closed VMA backing proof')

def validate_operation(value):
    required = {'teacher_plan', 'teacher_client', 'model_entrypoint', 'model_assets', 'mtp_artifact',
                'frontend_root', 'frontend_assets', 'execution_manifest', 'prepared_execution', 'numerical_trial',
                'selector', 'numa', 'teacher_scope', 'mtp_confidence', 'output_grants'}
    require(type(value) is dict and required <= set(value) <= required | {'preparation_contract', 'capacity_contract'} and value['teacher_scope'] == _teacher_scope(value),
            'closed C8 MTP4 forced-window operation required')
    _capacity_contract(value)
    for name in ('teacher_plan', 'teacher_client', 'execution_manifest'):
        identity(value[name])
        require(Path(value[name]['path']).parent == Path('/operator')
                and value[name]['bytes'] <= (MAX_PLAN_BYTES if name == 'teacher_plan' else 8 << 20),
                'bounded flat held window input required')
    require(len({value[k]['path'] for k in ('teacher_plan', 'teacher_client', 'execution_manifest')}) == 3,
            'distinct window DATA files required')
    legacy._validate_model_assets({**value, 'execution_manifest': None})
    identity(value['mtp_artifact'])
    artifact = value['mtp_artifact']
    require(artifact['path'].startswith('/models/hf/hub/')
            and Path(artifact['path']).name == 'mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf'
            and artifact['bytes'] == 2786568256
            and artifact['sha256'] == '5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6',
            'actual pinned compatible shared-Q8 MTP artifact required')
    require(type(value['mtp_confidence']) in (int, float) and math.isfinite(value['mtp_confidence'])
            and 0 <= value['mtp_confidence'] <= 1, 'explicit valid MTP confidence required')
    validate_preparation(value['prepared_execution'], value['numerical_trial'], _preparation_contract(value))
    grants = value['output_grants']
    require(type(grants) is dict and set(grants) == {'frame_bytes', 'metadata_bytes', 'disk_bytes', 'whole_timeout_s'}
            and all(type(grants[k]) is int and 0 < grants[k] < 2**63 for k in ('frame_bytes', 'metadata_bytes', 'disk_bytes'))
            and grants['disk_bytes'] >= grants['frame_bytes']+grants['metadata_bytes']
            and type(grants['whole_timeout_s']) in (int, float) and math.isfinite(grants['whole_timeout_s'])
            and 0 < grants['whole_timeout_s'] <= 7500, 'explicit bounded window frame/metadata/disk/time grants required')
    require(type(value['selector']) is dict and value['selector'] in selectors(), 'finite expert selector required')
    validate_numa(value['numa'])
    return value


def validate_sources(value):
    numerical.validate_sources(value)
    require({'tools/v100_target_window_teacher.py', 'tools/v100_target_window_evidence.py'} <= set(value),
            'actual forced-window control and sharded evidence source required')
    return value


def validate_teacher_client(value):
    extra = {'teacher_scope', 'prepared_execution', 'numerical_trial', 'selector', 'numa', 'execution_manifest_sha256', 'mtp_confidence'}
    if type(value) is dict:
        extra.update(key for key in ('preparation_contract', 'capacity_contract') if key in value)
    required = {'schema', 'teacher_plan_sha256', 'base', 'model', 'expected_service', 'readiness_timeout_s',
                'whole_timeout_s', 'socket_timeout_s', 'response_bytes', *extra}
    if type(value) is dict:
        _capacity_contract(value)
    service = expected_service(value) if type(value) is dict else SERVICE
    require(type(value) is dict and set(value) == required and value['schema'] == CLIENT_SCHEMA
            and value['teacher_scope'] == _teacher_scope(value) and value['expected_service'] == service
            and all(type(value['expected_service'][k]) is type(v) for k, v in service.items())
            and legacy.sha(value['execution_manifest_sha256']), 'closed versioned C8 window client required')
    validate_preparation(value['prepared_execution'], value['numerical_trial'], _preparation_contract(value))
    require(value['selector'] in selectors(), 'explicit window expert selection required')
    validate_numa(value['numa'])
    require(type(value['mtp_confidence']) in (int, float) and math.isfinite(value['mtp_confidence'])
            and 0 <= value['mtp_confidence'] <= 1, 'explicit window confidence required')
    projected = {k: v for k, v in value.items() if k not in extra}
    legacy.validate_teacher_client({**projected, 'schema': 'tensorfold.owned-teacher-client-v1',
                                    'expected_service': dict(legacy.SERVICE)})
    return value


def validate_teacher_operation_plan(operation, plan, client, *, profile=None, **unused):
    validate_operation(operation)
    validate_plan(plan)
    validate_teacher_client(client)
    require(type(profile) is dict and set(profile) == {'schema', 'version', 'artifacts', 'preparation'}
            and profile.get('schema') == 'tensorfold.execution-profile-manifest'
            and type(profile.get('version')) is int and profile['version'] == 1
            and profile.get('preparation') == operation['prepared_execution']
            and plan['identity']['profile_sha256'] == operation['execution_manifest']['sha256']
            and plan['mtp_confidence'] == operation['mtp_confidence'],
            'window plan and actual candidate profile differ')
    require(_capacity_contract(client) == _capacity_contract(operation), 'capacity client/operation contract differs')
    require(_preparation_contract(client) == _preparation_contract(operation), 'C8 client/operation preparation contract differs')
    require(client['teacher_plan_sha256'] == operation['teacher_plan']['sha256']
            and client['execution_manifest_sha256'] == operation['execution_manifest']['sha256']
            and all(client[key] == operation[key] for key in
                    ('teacher_scope', 'prepared_execution', 'numerical_trial', 'selector', 'numa', 'mtp_confidence')),
            'window client/operation/plan identity differs')
    from tensorfold.families.qwen4_exp.cuda.execution_identity import _METADATA

    # Match the runtime loader's inventory using already validated held assets.
    # ready.json belongs to frontend preparation, not runtime model metadata.
    expected = {'target/' + Path(item['path']).name: {'size': item['bytes'], 'sha256': item['sha256']}
                for item in operation['model_assets']}
    expected.update({'metadata/' + Path(item['path']).name: {'size': item['bytes'], 'sha256': item['sha256']}
                     for item in operation['frontend_assets'] if Path(item['path']).name in _METADATA})
    item = operation['mtp_artifact']
    expected['draft/' + Path(item['path']).name] = {'size': item['bytes'], 'sha256': item['sha256']}
    require(type(profile['artifacts']) is dict and profile['artifacts'] == expected
            and all(type(record) is dict and type(record.get('size')) is int
                    for record in profile['artifacts'].values()),
            'canonical manifest artifacts differ from actual held target/MTP/runtime frontend content')
    projected = resources(operation['teacher_plan']['bytes'], plan)
    require(operation['output_grants']['frame_bytes'] == projected['max_total_frame_file_bytes']
            and operation['output_grants']['metadata_bytes'] >= projected['metadata_output_bytes_bound']
            and client['whole_timeout_s'] == operation['output_grants']['whole_timeout_s'],
            'actual full raw frame/metadata/time extent lacks its explicit original grant')
    return {'mode': 'forced-window', 'sequences': ACTIVE, 'window_count': len(plan['rounds'])}


def operation_assets(value):
    validate_operation(value)
    return [*legacy.operation_assets(value), value['mtp_artifact'], value['execution_manifest'],
            *[value['numa'][key] for key in FILES]]


def qualifier_argv(operation, client, uuids, output):
    validate_operation(operation)
    validate_teacher_client(client)
    # Reuse the fixed common server/launcher shape, then replace only the
    # versioned method, artifact, graph, row and token settings below.
    args = legacy._qualifier_argv(operation, client, uuids, output,
        ['--teacher-plan', operation['teacher_plan']['path'], '--teacher-client', operation['teacher_client']['path']])
    args.remove('--no-graphs')
    args[args.index('--draft-method')+1] = 'mtp'
    args[args.index('--max-tokens')+1] = str(MAX_SEQUENCE_TOKENS)
    split = args.index('--')
    selector = operation['selector']
    flags = (['--grouped-gate-up-rows', '4'] if selector['gu'] == 'incumbent' else
        ['--onecat-gu-method', selector['gu'], '--onecat-down-method', selector['down'],
         '--onecat-iq2-source', selector['iq2_source'], '--onecat-prefill-method', selector['prefill_method'],
         '--onecat-device-budget-bytes', '117976064', '--onecat-global-host-bytes', '67108864'])
    return [*args[:split], '--numa-plan', operation['numa']['plan']['path'], '--numa-plan-sha256',
        operation['numa']['plan']['sha256'], '--execution-rows', '120', *flags,
        *args[split:], '--drafter', operation['mtp_artifact']['path'], '--mtp-drafts', str(DEPTH),
        '--mtp-confidence', str(operation['mtp_confidence']), '--execution-profile-manifest', operation['execution_manifest']['path']]


def validate_host_operation(host, inner, *, numa_bundle=None):
    validate_operation(inner['operation'])
    validate_sources(inner['source_files'])
    legacy._validate_model_host(host, inner, schema=SCHEMA, mode=MODE)
    operation = inner['operation']
    require(host['disk_bytes']['start'] >= host['disk_bytes']['running']+operation['output_grants']['disk_bytes'],
            'original disk admission does not fund the complete raw window output')
    stacks = [r for r in host['resources']['ulimits'] if r['Name'] == 'stack']
    require(stacks == [{'Name': 'stack', 'Soft': 64 << 20, 'Hard': 64 << 20}],
            'window HTTP thread stacks require their original bounded Host envelope')
    owners = [m for m in host['mounts'] if Path(operation['mtp_artifact']['path']).is_relative_to(Path(m['target']))]
    require(len(owners) == 1 and owners[0]['read_only'] is True, 'MTP artifact requires its unique held RO bind')
    require(numa_bundle is not None and 'verification_launcher' in host and 'numa' in host,
            'fresh actual CAP40/NUMA authority required for window execution')
    require(host['environment'].get('TENSORFOLD_PREFILL_ROWS') == '256'
            and host['environment'].get('TENSORFOLD_SM70_PROFILE', 'reference') == 'reference'
            and host['environment'].get('TENSORFOLD_PRIVATE_SM70_GROUPED_DOWN') == 'grouped-v1'
            and host['environment'].get('TENSORFOLD_NUMA_RECEIPT_DIR') == RECEIPT_DIR
            and host['environment'].get('NCCL_IGNORE_CPU_AFFINITY') == '0',
            'window numeric/placement environment differs')
    for key in ('TENSORFOLD_PRIVATE_SM70_MOE_PARTIAL', 'TENSORFOLD_PRIVATE_SM70_ROUTER_LOGITS', 'TENSORFOLD_PRIVATE_SM70_EMPTY_PREFIX'):
        require(host['environment'].get(key) in (None, 'inherit'), 'unrelated numerical environment selection forbidden')
    source = operation['numa']['placement_source']
    require(all(inner['expected_app'].get(PLACEMENT_MEMBER, {}).get(k) == source[k] for k in ('bytes', 'sha256')),
            'installed window placement implementation differs')
    operator = next(m['source'] for m in host['mounts'] if m['target'] == '/operator')
    validate_numa(host['numa'], operator_root=operator)
    require(host['numa']['placement_source'] == source and all(
        {k: host['numa'][key][k] for k in ('bytes', 'sha256')} == {k: operation['numa'][key][k] for k in ('bytes', 'sha256')}
        for key in FILES), 'Host/Driver window NUMA input identities differ')
    return inner


def prepare_plan(path, expected):
    plan = load_plan(path, expected_sha256=expected['sha256'], expected_bytes=expected['bytes'])
    return {'data': plan, 'input_sha256': expected['sha256'], 'reference_sha256': None,
            'resources': resources(expected['bytes'], plan)}


def runtime_profile(engine, operation):
    """Bind the real MTP/graph owner and fresh issued source/ELF identities."""
    if __package__:
        from . import v100_owned_qualification as owner
    else:
        import v100_owned_qualification as owner
    from tensorfold.families.qwen4_exp.cuda.execution_identity import digest, retention_identity_payload
    profile = owner.bound_receipt(engine)
    prepared = engine._owner_preparation
    require(prepared.specification == operation['prepared_execution']
            and engine.w.mtp is not None and engine.multi.mbuf is not None
            and engine.multi.speculation.method == 'mtp' and engine.multi.depth == DEPTH
            and engine.multi.confidence == operation['mtp_confidence'] and engine.multi.share == 0
            and engine.multi.buf.rows == 120 and engine.multi.prefill_rows == 256
            and engine.multi.full_resident and len(engine.multi.states) == 24,
            'actual full MTP4 prepared owner differs from window authority')
    graph_counts(engine.multi.graphs)
    numerical.validate_engine(engine, operation)
    if _preparation_contract(operation) is not None:
        require(prepared.prepared and not prepared.closed, 'actual admitted C8 preparation must be live')
        _validate_preparation_numerics(engine._execution_bind_local['numerics'], operation)
    pair = engine.kv_pair
    require((pair.key.bits, pair.key.codec, pair.value.bits, pair.value.codec) == (8, 0, 8, 0),
            'actual ordinary Q8 target history required')
    local = engine._execution_bind_local
    binding = engine.capacity_plan.get('execution_profile_binding')
    from tensorfold.families.qwen4_exp.cuda.execution_identity import execution_source_contents
    contents, content_projection = execution_source_contents(engine)
    require(type(binding) is dict and binding.get('content_passes') == 1
            and type(binding.get('hashes_reused_this_binding')) is int and binding['hashes_reused_this_binding'] == 0
            and binding.get('manifest_sha256') == local['manifest']
            and all(binding.get('identity', {}).get(key) == digest({label: {
                name: value for name, value in contents.items() if name.startswith(prefix)}})
                for key, label, prefix in (('source', 'files', 'source/'), ('kernel_build', 'libraries', 'library/'))),
            'window profile lacks fresh exact issued source/native content admission')
    retention = retention_identity_payload(engine)
    require(binding['identity']['retention'] == digest(retention), 'window raw retention differs from issued identity')
    states = engine.multi.states
    require(states and all(state.kv_identity == states[0].kv_identity for state in states),
            'window States do not share the actual immutable numerical policy')
    if _preparation_contract(operation) == ATEN_PREPARATION_CONTRACT:
        profile['attention_provider'] = _runtime_attention_provider(engine, operation, contents)
    profile.update(teacher_execution_registration={'binding': binding, 'numerics': local['numerics'], 'contents': contents,
                   'content_projection':content_projection},
                   numerical_trial_retention=retention,
                   window_teacher_semantics={'method': 'mtp', 'depth': DEPTH,
                       'mtp_confidence': engine.multi.confidence, 'proposal_policy': PROPOSAL_POLICY,
                       'resident_states': 24, 'active_states': ACTIVE,
                       'target_rows': 120, 'prefill_rows': 256, 'context': engine.context_window,
                       'state_kv_identity': states[0].kv_identity, 'prepared_execution': prepared.specification})
    capacity = _runtime_capacity_proof(engine, operation, retention)
    if capacity is not None:
        profile['full_window_capacity'] = capacity
        profile['teacher_execution_registration']['capacity_projection'] = {'schema':CAPACITY_PROOF_SCHEMA,
            'capacity_sha256':digest(capacity), 'retention_sha256':digest(retention)}
    if _preparation_contract(operation) == ATEN_PREPARATION_CONTRACT:
        _attention_retention({'teacher_profile':profile}, operation, {})
    # Use the existing strict JSON owner; no tensor, object repr or identity alias.
    return frames._owned_teacher_object(frames.json.dumps(profile, allow_nan=False))


def prepare_engine(engine, prepared, output):
    if __package__:
        from . import v100_owned_qualification as owner
    else:
        import v100_owned_qualification as owner
    operation = prepared['prepared_expert_operation']
    profile = runtime_profile(engine, operation)
    observer = object.__new__(WindowTeacher)
    engine._owned_target_window_teacher = observer  # Journal before allocation/setup can fail.
    observer.__init__(engine, prepared, output, fail=owner._observer_failure, profile=profile)
    receipt = observer.install()
    frames._owned_teacher_json(observer.directory/'setup.json', receipt, prepared['resources']['receipt_bytes_limit'])
    return receipt


def preflight(args, teacher, cli, *, selected, operation):
    validate_operation(operation)
    require(not selected.graph_use_receipts and args.draft_method == 'mtp' and args.mtp_drafts == DEPTH
            and args.mtp_confidence == operation['mtp_confidence']
            and str(args.drafter) == operation['mtp_artifact']['path']
            and args.context == 262144 and str(args.parallel).lower() != 'auto'
            and cli._parallel(args.parallel) == 24 and not args.no_graphs
            and selected.execution_rows == 120 and not selected.reference_tiles
            and selected.dense_implementation is None
            and str(args.execution_profile_manifest) == operation['execution_manifest']['path']
            and frames.os.environ.get('TENSORFOLD_PREFILL_ROWS') == '256',
            'window worker requires its exact full24/MTP4/E120/P256/graph CLI')
    expected = operation['selector']
    actual = {'gu': selected.onecat_gu_method or 'incumbent', 'down': selected.onecat_down_method,
              'iq2_source': selected.onecat_iq2_source, 'prefill_method': selected.onecat_prefill_method}
    require(actual == expected and selected.grouped_gate_up_rows == (4 if expected['gu'] == 'incumbent' else None)
            and selected.onecat_device_budget_bytes == (None if expected['gu'] == 'incumbent' else 117976064)
            and selected.onecat_global_host_bytes == (None if expected['gu'] == 'incumbent' else 67108864),
            'window worker expert selector/credit differs')
    require(frames.os.environ.get('TENSORFOLD_NUMA_PLAN') == operation['numa']['plan']['path']
            and frames.os.environ.get('TENSORFOLD_NUMA_PLAN_SHA256') == operation['numa']['plan']['sha256']
            and frames.os.environ.get('TENSORFOLD_NUMA_RANK') == str(args.rank)
            and frames.os.environ.get('TENSORFOLD_NUMA_RECEIPT_DIR') == RECEIPT_DIR
            and frames.os.environ.get('NCCL_IGNORE_CPU_AFFINITY') == '0',
            'window worker bootstrap placement authority differs')


def validate_service(stats, spec, instance, *, prepared_manifest):
    if __package__:
        from . import v100_adaptive_qualification as qualification
    else:
        import v100_adaptive_qualification as qualification
    validate_teacher_client(spec)
    effective = qualification.profile(stats)
    qualification.require_idle(stats)
    qualification.materialization(stats, effective)
    if _capacity_contract(spec) is not None:
        plan = stats['capacity_plan']
        require(all(k in plan for k in _CAPACITY_FIELDS), 'service lacks complete scoped capacity policy')
        _validate_capacity_plan({k:plan[k] for k in _CAPACITY_FIELDS}, slots=24, bits=8)
        _validate_capacity_catalogue(plan, spec)
    else:
        _legacy_capacity_plan(stats['capacity_plan'])
    expected = spec['expected_service']
    expert = stats['capacity_plan']['expert_parallel']
    runtime = stats['runtime']
    families = runtime['execution']['execution_families']
    binding = runtime['capacity_plan']['execution_profile_binding']
    require(effective['instance_id'] == instance and effective['model']['id'] == spec['model']
            and effective['context_tokens'] == 262144 and effective['resident_capacity'] == 24
            and effective['session_capacity']['configured_ceiling'] == 24 and effective['model']['kv_dtype'] == 'int8'
            and all(expert.get(k) == expected[k] for k in ('tp', 'pp', 'etp', 'ep'))
            and families['effective'] == {'kernel_family': expected['kernel_family'], 'graph_family': expected['graph_family']}
            and families.get('trial') == {k: expected[k] for k in ('kernel_family', 'graph_family')}
            and families.get('native_prepared') is True and families.get('graph_rows') == 120,
            'actual window HTTP service differs from full MTP/graph preparation')
    require(type(prepared_manifest) is dict and binding.get('manifest_sha256') == prepared_manifest['manifest_canonical_sha256']
            and all(type(binding.get(key)) is int for key in
                    ('verified_input_files', 'verified_input_bytes', 'content_passes', 'hashes_reused_this_binding'))
            and binding['verified_input_files'] == prepared_manifest['manifest_artifact_files']
            and binding['verified_input_bytes'] == prepared_manifest['manifest_artifact_bytes']
            and binding['content_passes'] == 1 and binding['hashes_reused_this_binding'] == 0
            and binding.get('shared_immutable_same_host') is True,
            'window HTTP profile lacks its fresh complete actual target/MTP/frontend admission')
    return effective


class WindowClient:
    """Eight bounded HTTP owners inside the existing owned-client supervisor."""
    def __init__(self, teacher, selected, output):
        import threading
        self.teacher, self.selected, self.output = teacher, selected, Path(output)
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
        # The actual CAP40 runtime lowers soft to8MiB while retaining hard64MiB.
        # Admission funds an upper bound, not an exact or minimum soft limit.
        require(0 < soft <= hard == 64 << 20 and threading.stack_size() == 0,
                'window HTTP runtime stack policy exceeds original admission')
        self.cancelled = threading.Event()
        self.lock = threading.Lock()
        self._error = None
        self.result, self.post_attempts = None, 0

    def failure(self):
        with self.lock:
            return self._error

    def run(self):
        if __package__:
            from . import openai_protocol as protocol
            from . import v100_owned_qualification as owner
            from .worker_lifetime import Task, drain
        else:
            import openai_protocol as protocol
            import v100_owned_qualification as owner
            from worker_lifetime import Task, drain
        spec, plan = self.selected['data'], self.teacher['data']
        client, control, before, instance = owner._owned_client_open(self, 'teacher')
        operation = self.selected['prepared_operation']
        readiness = collect(operation, self.output)
        frames._owned_teacher_json(self.output/'numa-readiness.json', readiness, 8 << 20, control)
        self.selected['numa_readiness'] = readiness
        results, tasks, primary = [None]*ACTIVE, [], None
        def one(index, case):
            try:
                body = {'model': spec['model'], 'prompt': case['prompt_ids'], 'max_tokens': len(case['forced_ids']),
                    'temperature': 0, 'seed': 0, 'draft': True, 'ignore_eos': True, 'stream': False, 'return_token_ids': True}
                with self.lock:
                    self.post_attempts += 1
                reply = client.call(f'window-completion-{index}', body, path='/v1/completions', read_control=control)
                protocol.chunk_fields(reply)
                count = protocol.completion_usage(reply['usage'], len(case['forced_ids']), prompt=True)
                choices, runtime = reply.get('choices'), reply.get('tensorfold', {})
                ids = runtime.get('token_ids')
                require(type(choices) is list and len(choices) == 1 and type(choices[0].get('text')) is str
                        and choices[0].get('index') == 0 and choices[0].get('finish_reason') == 'length'
                        and ids == case['forced_ids'] and count == len(ids)
                        and runtime.get('token_sha') == hashlib.sha256(','.join(map(str, ids)).encode()).hexdigest()[:12]
                        and reply['usage']['prompt_tokens'] == len(case['prompt_ids'])
                        and reply['usage'].get('total_tokens') == len(case['prompt_ids'])+len(ids)
                        and runtime.get('drafts') is True
                        and type(runtime.get('cached')) is int and runtime['cached'] == 0
                        and reply['usage'].get('prompt_tokens_details', {}).get('cached_tokens') == 0,
                        'complete cold forced MTP HTTP case differs from its actual token path')
                lanes = [row['lanes'][index] for row in plan['rounds']]
                require(type(runtime.get('drafted')) is int and runtime['drafted'] == sum(lane['rows']-1 for lane in lanes)
                        and type(runtime.get('accepted')) is int and runtime['accepted'] == sum(lane['keep']-1 for lane in lanes),
                        'forced target-window counters differ from the explicit schedule')
                results[index] = {'case_id': case['case_id'], 'token_ids': ids,
                    'prompt_tokens': len(case['prompt_ids']), 'completion_tokens': len(ids),
                    'forced_target_drafted': runtime['drafted'], 'forced_target_accepted': runtime['accepted'],
                    'runtime': runtime, 'text': choices[0]['text'], 'finish_reason': choices[0]['finish_reason']}
            except BaseException as error:
                with self.lock:
                    if self._error is None:
                        self._error = error
                self.cancelled.set()
                raise
        try:
            for index, case in enumerate(plan['sequences']):
                task = Task(lambda i=index, c=case: one(i, c), name=f'owned-window-http-{index}')
                tasks.append(task)
                task.start()
                # Admit in declared order without waiting for generation. The
                # production-round gate keeps all eight cases live together.
                admission = self.output/'window-teacher-rank0'/f'admission-{index}.json'
                while not admission.is_file():
                    error = self.failure()
                    if error is not None:
                        raise error
                    control.check()
                    self.cancelled.wait(.05)
                raw = frames._owned_teacher_read(admission, 1 << 20, control)
                record = frames._owned_teacher_object(raw)
                require(record.get('schema') == 'tensorfold.target-window-teacher-admission-v1'
                        and record.get('index') == index and record.get('case_id') == case['case_id']
                        and record.get('rank') == 0 and record.get('input_sha256') == self.teacher['input_sha256'],
                        'actual ordered admission differs from declared case')
            while not all(task.done.is_set() for task in tasks):
                error = self.failure()
                if error is not None:
                    raise error
                control.check()
                self.cancelled.wait(.05)
        except BaseException as error:  # noqa: BLE001 - cancel producers and drain every accepted HTTP callback
            primary = error
            self.cancelled.set()
        drain(tasks, primary=primary)  # No resource/result publication before every HTTP callback completes.
        require(len(tasks) == ACTIVE and all(row is not None for row in results), 'all eight forced HTTP outcomes required')
        final = client.call('window-final-stats', path='/stats', object_response=True, read_control=control)
        observed = validate_service(final, spec, instance, prepared_manifest=self.selected['prepared_manifest'])
        require(observed == before, 'window service identity changed during the cohort')
        self.result = {'schema': CLIENT_REPORT_SCHEMA, 'complete': True, 'post_attempts': self.post_attempts,
            'teacher_plan_sha256': self.teacher['input_sha256'], 'teacher_client_sha256': self.selected['sha256'],
            'mode': 'forced-window', 'server_instance_id': instance, 'cases': results,
            'client_protocol_passed': True, 'quality_acceptance': False, 'P3_valid': False,
            'scope': 'diagnostic forced tokens/keep paths; HTTP draft counters are not ordinary serving acceptance'}
        frames._owned_teacher_json(self.output/'teacher-client-report.json', self.result, 8 << 20, control)


def client_credit(teacher, client_bytes):
    if __package__:
        from . import v100_owned_qualification as owner
    else:
        import v100_owned_qualification as owner
    before = teacher['resources']['global_host_obligation_bytes']
    owner._teacher_client_credit(teacher, client_bytes)
    resource = teacher['resources']
    one = resource['global_host_obligation_bytes']-before
    # The fixed Host envelope supplies a64MiB pthread stack limit. Fund all
    # eight response owners, their individual parser/socket work, and worst
    # stack commitment before Engine admission. Virtual/RSS remain distinct.
    stacks = ACTIVE*(64 << 20)
    resource['global_host_obligation_bytes'] += (ACTIVE-1)*one + stacks
    resource['client_parent_payload_bytes'] = ACTIVE*one + stacks
    resource['client_maximum_threads'] = ACTIVE
    resource['client_thread_stack_virtual_bound'] = stacks
    resource['client_thread_stack_per_thread_bound'] = 64 << 20


def validate_startup(row, selected):
    _validate_startup_capacity(row.get('teacher_profile', {}), selected)
    validate_preparation(selected['prepared_execution'], selected['numerical_trial'], _preparation_contract(selected))
    profile, windows = row.get('teacher_profile', {}), row.get('teacher_windows', {})
    require(type(row.get('rank')) is int and 0 <= row['rank'] < WORLD and row.get('world') == WORLD
            and all(type(profile.get(k)) is int for k in ('rank', 'world', 'target_banks', 'mtp_banks'))
            and profile.get('rank') == row['rank'] and profile.get('world') == WORLD
            and profile.get('capability') == [7, 0] and all(type(n) is int for n in profile['capability'])
            and profile.get('target_banks') == 48
            and profile.get('mtp_banks') == 1 and type(row.get('mtp_banks')) is int and row['mtp_banks'] == 1
            and profile.get('plan_identity') == row.get('plan_identity') and row.get('state_count') == 24
            and windows.get('full_resident') is True and windows.get('slots') == 24
            and all(type(windows.get(k)) is int for k in
                    ('capacity_min', 'capacity_max', 'limit_min', 'limit_max', 'logical_min', 'logical_max'))
            and windows['capacity_min'] == windows['capacity_max'] >= 262144
            and windows['limit_min'] == windows['limit_max'] >= 262144
            and windows['logical_min'] == windows['logical_max'] == 262144,
            'actual all-rank MTP bank/full24 native-window materialization required')
    semantics = profile.get('window_teacher_semantics', {})
    require(semantics.get('method') == 'mtp' and semantics.get('depth') == DEPTH
            and type(semantics.get('mtp_confidence')) in (int, float)
            and semantics.get('mtp_confidence') == selected['mtp_confidence']
            and semantics.get('proposal_policy') == PROPOSAL_POLICY
            and semantics.get('resident_states') == 24 and semantics.get('active_states') == ACTIVE
            and semantics.get('target_rows') == 120 and semantics.get('prefill_rows') == 256
            and semantics.get('context') == 262144 and semantics.get('prepared_execution') == selected['prepared_execution'],
            'actual window teacher numerical/row/method region differs')
    plan = row['plan_identity']
    require(type(plan) is list and len(plan) >= 11
            and plan[:11] == ['owned-etp4-native-slots-v1', 8, 4, 512, 640, 2560, 11, 120, 256, True, 'grouped-v1'],
            'actual MTP expert Plan geometry differs')
    # Validate the unchanged expert selector encoding with the historical owner
    # only after proving every new method/row dimension. Raw identity stays intact.
    projection = list(plan)
    projection[7], projection[9] = 32, False
    selector = selected['selector']
    frames._candidate_plan_identity(projection, None if selector['gu'] == 'incumbent' else selector['gu'],
        selector['down'], selector['iq2_source'], selector['prefill_method'], teacher_scope=numa.SCOPE)
    if _preparation_contract(selected) == ATEN_PREPARATION_CONTRACT:
        _validate_attention_provider(profile, selected)
    registration = profile.get('teacher_execution_registration', {})
    _validate_preparation_numerics(registration.get('numerics'), selected)
    from tensorfold.families.qwen4_exp.cuda.execution_identity import digest
    raw = profile.get('numerical_trial_retention')
    require(type(raw) is dict and registration.get('binding', {}).get('identity', {}).get('retention') == digest(raw)
            and raw.get('owner_preparation', {}).get('specification') == {
                k: v for k, v in selected['prepared_execution'].items() if k != 'trial'},
            'actual window retention payload differs from its issued fingerprint')
    if _preparation_contract(selected) == ATEN_PREPARATION_CONTRACT:
        _attention_retention(row, selected, {})
    for axis in ('native_family', 'prepared_linears'):
        require((axis in row) == selected['numerical_trial'][axis], 'window numerical capability receipt differs')
        if selected['numerical_trial'][axis]:
            require(row[axis].get('requested') == numerical.AXES[axis], 'actual window numerical policy differs')
    if selected['numerical_trial']['prepared_linears']:
        require(set(row['prepared_linears'].get('roles', {})) == {'decode', 'mtp'}
                and all(r.get('logical_rows') == 120 and r.get('physical_rows') == 128
                        for r in row['prepared_linears']['roles'].values()), 'actual target/MTP M32 roles differ')
    q8 = row.get('owned_q8_preparation', {})
    require(bool(q8.get('window_roles')) == selected['numerical_trial']['owned_q8_window_prepare'],
            'actual window Q8 capability selection differs')
    if selected['numerical_trial']['owned_q8_window_prepare']:
        require(set(q8['window_roles']) == {'decode', 'mtp'}, 'both actual MTP and target Q8 window roles required')
    return row


def validate_model_result(value):
    fields = {'schema', 'teacher_plan_sha256', 'teacher_client_sha256', 'launch_exit', 'client_report',
              'teacher_authority', 'rank_receipts', 'startup_receipts', 'child_native_records', 'numa'}
    require(type(value) is dict and set(value) == fields and value['schema'] == RESULT_SCHEMA
            and legacy.sha(value['teacher_plan_sha256']) and legacy.sha(value['teacher_client_sha256'])
            and type(value['launch_exit']) is int and value['launch_exit'] == 0,
            'closed positive sharded window model result required')
    for key, expected in (('client_report', 'teacher/teacher-client-report.json'),
                          ('teacher_authority', 'teacher/teacher-authority.json')):
        identity(value[key], key='file')
        require(value[key]['file'] == expected and value[key]['bytes'] <= 8 << 20, 'fixed bounded window result required')
    for key in ('rank_receipts', 'startup_receipts', 'child_native_records'):
        require(type(value[key]) is list and len(value[key]) == WORLD, 'all eight closed window ranks required')
        for rank, row in enumerate(value[key]):
            identity(row, key='file', rank=True)
            expected = (f'teacher/window-teacher-rank{rank}/summary.json' if key == 'rank_receipts' else
                        f'teacher/rank-{rank}.json' if key == 'startup_receipts' else None)
            require(row['rank'] == rank and row['bytes'] <= (4 << 20 if key == 'child_native_records' else 8 << 20)
                    and (expected is None or row['file'] == expected), 'window result rank/path/extent differs')
    numa.validate_evidence_refs(value['numa'])
    return value


def validate_host_result(value, expected):
    result = legacy._validate_host_result(value, expected, schema=HOST_RESULT_SCHEMA, mode=MODE,
                                          result_validator=validate_model_result)
    require(result['teacher_plan_sha256'] == expected['operation']['teacher_plan']['sha256']
            and result['teacher_client_sha256'] == expected['operation']['teacher_client']['sha256'],
            'window result plan/client identity differs')
    return result


def validate_completed_teacher_host(value, *, authority_sha256, authority_bytes, image_id=None):
    require(type(value) is dict and value.get('schema') == 'tensorfold.native-contract-host-result-v1'
            and value.get('report_complete') is True and value.get('source_match') is True
            and value.get('health_match') is True and value.get('primary_error') is None
            and value.get('cleanup_errors') == [] and type(value.get('attach_exit')) is int and value['attach_exit'] == 0,
            'window Host has not completed positively')
    state = value.get('container', {}).get('State', {})
    require(state.get('Status') == 'exited' and state.get('Running') is False and state.get('OOMKilled') is False
            and type(state.get('Pid')) is int and state['Pid'] == 0
            and type(state.get('ExitCode')) is int and state['ExitCode'] == 0, 'window container retirement is not positive')
    inner = value.get('inner')
    require(type(inner) is dict and value['container'].get('Image') == inner.get('image_id_declared')
            and (image_id is None or inner['image_id_declared'] == image_id), 'window actual immutable image differs')
    validate_operation(inner['operation'])
    result = validate_host_result(inner, {'operation': inner['operation'], 'source_commit': inner['source_commit'],
                                        'image_id': inner['image_id_declared']})
    require(result['teacher_authority'] == {'file': 'teacher/teacher-authority.json', 'bytes': authority_bytes,
                                           'sha256': authority_sha256}, 'window owning result names a different authority')
    return inner


def prepare_input(operation, teacher, plan_path):
    validate_operation(operation)
    require(str(Path(plan_path).absolute()) == operation['teacher_plan']['path']
            and teacher['input_sha256'] == operation['teacher_plan']['sha256']
            and teacher['resources']['input_bytes'] == operation['teacher_plan']['bytes'],
            'actual held window plan differs from its closed operation')
    payloads, sizes = {}, []
    for label in ('execution_manifest', 'teacher_client'):
        expected = operation[label]
        raw = frames._owned_teacher_read(expected['path'], 8 << 20)
        require(len(raw) == expected['bytes'] and hashlib.sha256(raw).hexdigest() == expected['sha256'],
                'actual held window input differs: ' + label)
        payloads[label] = frames._owned_teacher_object(raw)
        sizes.append(len(raw))
    validate_teacher_operation_plan(operation, teacher['data'], payloads['teacher_client'], profile=payloads['execution_manifest'])
    control = frames.json.dumps(operation, indent=1, ensure_ascii=True, allow_nan=False).encode()
    require(len(control) <= 8 << 20, 'window control metadata exceeds bound')
    sizes.append(len(control))
    from tensorfold.families.qwen4_exp.cuda.execution_identity import digest
    artifacts = payloads['execution_manifest'].get('artifacts')
    require(type(artifacts) is dict and artifacts and all(type(row) is dict and type(row.get('size')) is int
            and row['size'] >= 0 and legacy.sha(row.get('sha256')) for row in artifacts.values()),
            'actual window manifest artifact census differs')
    bundle = load_inputs(operation['numa'])
    sizes.extend(len(raw) for raw in bundle['raw'].values())
    return {'operation': operation, 'control': control, 'sizes': sizes, 'numa_bundle': bundle,
        'manifest_canonical_sha256': digest(payloads['execution_manifest']),
        'manifest_artifact_files': len(artifacts), 'manifest_artifact_bytes': sum(row['size'] for row in artifacts.values())}


def _numa_refs(model):
    validate_model_result(model)
    item = model['numa']
    refs = [*item['inputs'].values(), item['readiness']]
    for row in item['ranks']:
        refs.extend(row['stages'].values())
    return refs


def reference_result_files(model):
    return {row['file']: {k: row[k] for k in ('bytes', 'sha256')} for row in _numa_refs(model)}


def validate_result_files(root, model, operation, *, plan=None):
    validate_operation(operation)
    validate_model_result(model)
    root = Path(root)
    if plan is None:
        plan = load_plan(operation['teacher_plan']['path'], expected_sha256=operation['teacher_plan']['sha256'],
                         expected_bytes=operation['teacher_plan']['bytes'])
    if __package__:
        from . import v100_target_window_evidence as shards
    else:
        import v100_target_window_evidence as shards
    authority = shards.validate_artifacts(root, model, operation, plan=plan)
    actual = collect(operation, root/'teacher', native_records=model['child_native_records'])
    evidence_refs = model['numa']
    require({k: v for k, v in evidence_refs.items() if k not in ('readiness', 'schema')} ==
            {k: v for k, v in actual.items() if k != 'schema'}, 'closed window NUMA evidence differs from actual receipts')
    for ref in _numa_refs(model):
        numa._ref(root, root/ref['file'], {k: ref[k] for k in ('bytes', 'sha256')}, maximum=8 << 20)
    _, raw = numa._ref(root, root/evidence_refs['readiness']['file'], evidence_refs['readiness'], maximum=8 << 20)
    require(frames._owned_teacher_object(raw) == actual, 'window placement/readiness changed after first POST')
    return authority


def finish_model(output, teacher, selected, observer, launch_exit):
    if __package__:
        from . import v100_owned_qualification as owner
        from . import v100_target_window_evidence as shards
    else:
        import v100_owned_qualification as owner
        import v100_target_window_evidence as shards
    output, operation = Path(output), selected['prepared_operation']
    authority = shards.seal(output, teacher, launch_exit)
    root, native = output.parent, []
    for rank in range(WORLD):
        paths = sorted(Path(observer['directory']).glob(f'teacher-closed-rank-{rank}-pid-*.json'))
        require(len(paths) == 1, 'actual closed native VMA record required for every window worker')
        native.append(owner._teacher_file(paths[0], root, rank=rank))
    result = {'schema': RESULT_SCHEMA, 'teacher_plan_sha256': teacher['input_sha256'],
        'teacher_client_sha256': selected['sha256'], 'launch_exit': launch_exit,
        'client_report': owner._teacher_file(output/'teacher-client-report.json', root),
        'teacher_authority': owner._teacher_file(output/'teacher-authority.json', root),
        'rank_receipts': [owner._teacher_file(output/f'window-teacher-rank{r}/summary.json', root, rank=r) for r in range(WORLD)],
        'startup_receipts': [owner._teacher_file(output/f'rank-{r}.json', root, rank=r) for r in range(WORLD)],
        'child_native_records': native,
        'numa': evidence(operation, output, selected['numa_readiness'], native)}
    validate_model_result(result)
    validate_result_files(root, result, operation, plan=teacher['data'])
    frames._owned_teacher_json(output/'teacher-model-result.json', result, 8 << 20)
    return authority


def _json_ref(root, ref, *, maximum=8 << 20):
    control = frames._teacher_read_control(60)
    return frames._owned_comparison_file(Path(root), ref, control, maximum=maximum)


def _validate_content_projection(registration, operation, rank):
    fields = {'binding','numerics','contents','content_projection'}
    if _capacity_contract(operation) is not None:
        fields.add('capacity_projection')
    require(type(registration) is dict and set(registration) == fields,
            'window actual source/numerical/projection registration missing')
    binding, projection = registration['binding'], registration['content_projection']
    require(type(binding) is dict and type(projection) is dict
            and set(projection) == {'schema','issuer_rank','local_authority_owned','cohort_content_passes',
                'local_payload_bytes_read_this_binding','cohort_hashes_reused_this_binding',
                'manifest_sha256','retained_host_bytes'}
            and projection['schema'] == 'tensorfold.shared-execution-content-projection-v1'
            and type(projection['issuer_rank']) is int and projection['issuer_rank'] == 0
            and projection['local_authority_owned'] is (rank == 0)
            and type(projection['cohort_content_passes']) is int and projection['cohort_content_passes'] == 1
            and type(projection['cohort_hashes_reused_this_binding']) is int
            and projection['cohort_hashes_reused_this_binding'] == 0
            and type(projection['local_payload_bytes_read_this_binding']) is int
            and type(binding.get('bytes_read_this_binding')) is int and binding['bytes_read_this_binding'] > 0
            and projection['local_payload_bytes_read_this_binding'] == (binding['bytes_read_this_binding'] if rank == 0 else 0)
            and type(projection['retained_host_bytes']) is int and projection['retained_host_bytes'] > 0
            and legacy.sha(projection['manifest_sha256'])
            and projection['manifest_sha256'] == binding.get('manifest_sha256'),
            'window source content provenance is not the actual shared rank-zero pass')


def validate_runtime_records(root, model, operation, *, observer_source):
    """Bind all raw numerical registrations to closed genuine provider VMAs."""
    from tensorfold.families.qwen4_exp.cuda.execution_identity import digest
    registrations, startups, native_rows, pids = [], [], [], set()
    for rank in range(WORLD):
        startup = _json_ref(root, model['startup_receipts'][rank])
        validate_startup(startup, operation)
        closed = _json_ref(root, model['rank_receipts'][rank])
        require(closed.get('profile') == startup['teacher_profile'], 'window profile changed before positive retirement')
        native = _json_ref(root, model['child_native_records'][rank], maximum=4 << 20)
        pid = native.get('PID')
        require(native.get('schema') == 'tensorfold.native-elf-observation-v3'
                and native.get('identity_method') == 'kernel-vma-backed-fd-v1'
                and type(pid) is int and pid > 1 and native.get('rank') == rank and native.get('tag') == 'teacher-closed'
                and startup.get('teacher_worker_PID') == pid and closed.get('worker_PID') == pid
                and model['child_native_records'][rank]['file'] == f'child-native/teacher-closed-rank-{rank}-pid-{pid}.json'
                and all(native.get('observer', {}).get(k) == observer_source[k] for k in ('bytes', 'sha256'))
                and all(startup.get('teacher_observer', {}).get(k) == observer_source[k] for k in ('bytes', 'sha256'))
                and str(startup.get('teacher_observer', {}).get('path', '')).endswith('/tools/v100_native_contract_driver.py')
                and native.get('operation_limits') == MODEL_OBSERVER_LIMITS and native.get('missing_required_prefixes') == []
                and native.get('access_precondition') == {'uids': [1000]*4, 'gids': [1000]*4, 'NoNewPrivs': '0',
                    'caps': {key: '0000010000000000' for key in ('CapInh', 'CapPrm', 'CapEff', 'CapBnd', 'CapAmb')}},
                'window rank lacks actual PID/access/provider VMA linkage')
        pids.add(pid)
        mapped = native.get('mapped_elfs')
        require(type(mapped) is list and 0 < len(mapped) <= MODEL_OBSERVER_LIMITS['elfs']
                and len({item['name'] for item in mapped}) == len(mapped)
                and any(item['name'].startswith('tf_gguf_') for item in mapped),
                'complete unique window native provider set required')
        registration = startup['teacher_profile'].get('teacher_execution_registration')
        _validate_content_projection(registration, operation, rank)
        binding, contents = registration['binding'], registration['contents']
        require(type(binding) is dict and type(binding.get('identity')) is dict
                and type(contents) is dict and contents and type(registration['numerics']) is dict
                and all(type(name) is str and name.startswith(('source/', 'library/'))
                    and type(row) is dict and set(row) == {'size', 'sha256'} and type(row['size']) is int
                    and row['size'] >= 0 and legacy.sha(row['sha256']) for name, row in contents.items())
                and type(binding.get('content_passes')) is int and binding['content_passes'] == 1
                and type(binding.get('hashes_reused_this_binding')) is int and binding['hashes_reused_this_binding'] == 0
                and binding.get('shared_immutable_same_host') is True
                and all(type(binding.get(key)) is int and binding[key] > 0 for key in ('verified_input_files', 'verified_input_bytes')),
                'window lacks fresh complete source/SDK/numerical content proof')
        for name, item in contents.items():
            path = name.removeprefix('library/')
            if name.startswith('library/') and Path(path).name.startswith('tf_'):
                require(any(row.get('path') == path and row.get('bytes') == item['size'] and row.get('sha256') == item['sha256']
                            and type(row.get('vmas')) is list and row['vmas'] for row in mapped),
                        'issued window provider is not backed by its actual closed executable VMA')
        if operation['numerical_trial']['native_family']:
            name = startup['native_family']['actual'].get('module')
            require(type(name) is str and name.startswith('tf_sm70_family_')
                    and any(Path(row['path']).name == name+'.so' for row in mapped), 'actual native-family VMA differs')
        if operation['numerical_trial']['owned_q8_window_prepare']:
            path = startup['owned_q8_preparation'].get('native_module', {}).get('path')
            require(any(row['path'] == path for row in mapped), 'actual Q8 window provider VMA differs')
        if _preparation_contract(operation) == ATEN_PREPARATION_CONTRACT:
            _validate_attention_vmas(startup['teacher_profile'], operation, native)
        frames._prepared_expert_binding(startup, operation['selector'], mapped, rank)
        require(registration['numerics'].get('owned_expert_native') == startup['plan_identity'],
                'actual window expert numerical Plan differs')
        registrations.append(registration)
        startups.append(startup)
        native_rows.append(native)
    require(len(pids) == WORLD, 'eight actual unique window workers required')
    for registration in registrations:
        identity_value, contents = registration['binding']['identity'], registration['contents']
        require(identity_value == registrations[0]['binding']['identity'] and contents == registrations[0]['contents']
                and identity_value.get('numerics') == digest({'ranks': [r['numerics'] for r in registrations]})
                and all(identity_value.get(key) == digest({label: {
                    name: value for name, value in contents.items() if name.startswith(prefix)}})
                    for key, label, prefix in (('source', 'files', 'source/'), ('kernel_build', 'libraries', 'library/')))
                and identity_value.get('context_tokens') == 262144 and identity_value.get('resident_sessions') == 24
                and identity_value.get('method') == 'mtp' and identity_value.get('history_layout') == 'unique'
                and identity_value.get('draft') == digest({'files': {
                    'draft/'+Path(operation['mtp_artifact']['path']).name: {
                        'size': operation['mtp_artifact']['bytes'], 'sha256': operation['mtp_artifact']['sha256']}}}),
                'all-rank actual MTP execution fingerprints differ from their raw bindings')
    return {'registrations': registrations, 'startups': startups, 'native': native_rows}


def accept_teacher_result(root, spec, source, *, teacher_binding, native_acceptor):
    require(type(teacher_binding) is dict and teacher_binding.get('mode') == 'forced-window',
            'explicit forced-window Driver binding required')
    root = Path(root)
    raw = frames._owned_teacher_read(root/'teacher/teacher-model-result.json', 8 << 20)
    model = validate_model_result(frames._owned_teacher_object(raw))
    operation = spec['operation']
    authority = validate_result_files(root, model, operation)
    rank0 = native_acceptor(root, model, spec, source, role='teacher')
    observer = spec['source_files']['tools/v100_native_contract_driver.py']
    proofs = validate_runtime_records(root, model, operation, observer_source=observer)
    for rank, native in enumerate(proofs['native']):
        require(all(native.get('observer', {}).get(k) == observer[k] for k in ('bytes', 'sha256')),
                'actual frozen window observer source differs')
    report = _json_ref(root, model['client_report'])
    plan = authority['plan']
    require(report.get('schema') == CLIENT_REPORT_SCHEMA and report.get('complete') is True
            and report.get('client_protocol_passed') is True and report.get('quality_acceptance') is False
            and report.get('P3_valid') is False and report.get('mode') == 'forced-window'
            and type(report.get('post_attempts')) is int and report['post_attempts'] == ACTIVE
            and report.get('teacher_plan_sha256') == model['teacher_plan_sha256']
            and report.get('teacher_client_sha256') == model['teacher_client_sha256']
            and type(report.get('server_instance_id')) is str and report['server_instance_id']
            and report['server_instance_id'] == rank0.get('teacher_server_instance_id')
            and type(report.get('cases')) is list and len(report['cases']) == ACTIVE,
            'actual eight-request window HTTP closure differs from original worker/service')
    for actual, case in zip(report['cases'], plan['sequences']):
        require(actual.get('case_id') == case['case_id'] and actual.get('token_ids') == case['forced_ids']
                and actual.get('prompt_tokens') == len(case['prompt_ids'])
                and actual.get('completion_tokens') == len(case['forced_ids']),
                'actual HTTP output differs from the complete forced corpus')
    return model


def _held_input(side, item, control):
    inputs = side['host']['source_after']['inputs']
    matches = [row for row in inputs if Path(row['path']).name == Path(item['path']).name
               and all(row[k] == item[k] for k in ('bytes', 'sha256'))]
    require(len(matches) == 1, 'window comparison lacks a unique exact held Host input')
    raw = frames._owned_teacher_read(matches[0]['path'], item['bytes'], control)
    require(len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256'],
            'actual held window comparison input changed')
    return frames._owned_teacher_object(raw)


def _comparison_side(selected, control):
    require(type(selected) is dict and set(selected) == {'root', 'host_result', 'completion'}
            and absolute(selected['root']), 'closed window comparison side required')
    root = Path(selected['root'])
    host = frames._owned_comparison_file(root, selected['host_result'], control)
    inner = host.get('inner', {})
    model = validate_model_result(inner.get('model_result'))
    authority = frames._owned_comparison_file(root, model['teacher_authority'], control)
    validate_completed_teacher_host(host, authority_sha256=model['teacher_authority']['sha256'],
                                    authority_bytes=model['teacher_authority']['bytes'])
    completion_ref = selected['completion']
    identity(completion_ref)
    require(completion_ref['bytes'] <= 8 << 20, 'bounded exact external CLI completion required')
    raw = frames._owned_teacher_read(completion_ref['path'], completion_ref['bytes'], control)
    require(len(raw) == completion_ref['bytes'] and hashlib.sha256(raw).hexdigest() == completion_ref['sha256'],
            'actual external window CLI completion changed')
    validate_teacher_completion(frames._owned_teacher_object(raw), host_result_sha256=selected['host_result']['sha256'],
                                authority_sha256=model['teacher_authority']['sha256'])
    source = host.get('source_after', {}).get('source', {})
    require(type(source) is dict and source, 'complete canonical window source closure required')
    for module in (Path(__file__), Path(frames.__file__), Path(numerical.__file__)):
        expected = source.get('tools/'+module.name)
        require(type(expected) is dict and module.stat().st_size == expected['bytes']
                and frames._owned_teacher_hash(module, expected['bytes'], 8 << 20, control) == expected['sha256'],
                'window comparison must execute the exact closed source version')
    side = {'root': root, 'host': host, 'inner': inner, 'model': model, 'authority': authority}
    plan = _held_input(side, inner['operation']['teacher_plan'], control)
    actual = validate_result_files(root, model, inner['operation'], plan=plan)
    require(actual == authority, 'window authority changed across its complete closure')
    content = frames._owned_comparison_file(root, inner['content_binding'], control)
    require(content.get('schema') == 'tensorfold.owned-model-content-binding-v1' and content.get('completed') is True
            and [row.get('identity') for row in content.get('assets', [])] == operation_assets(inner['operation']),
            'window lacks actual held target/MTP/frontend/control/NUMA content lifetime proof')
    observer = source.get('tools/v100_native_contract_driver.py')
    require(type(observer) is dict and all(k in observer for k in ('bytes', 'sha256')), 'actual closed native observer source missing')
    side['proofs'] = validate_runtime_records(root, model, inner['operation'], observer_source=observer)
    return side



def _attention_preparation_projection(value, changes):
    from copy import deepcopy
    projected = deepcopy(value)
    if 'kernel_family' in changes:
        for key in ('kernel_family', 'rn_heads_per_block', 'aten_tiles'):
            projected.pop(key, None)
    elif 'aten_tiles' in changes:
        projected['aten_tiles'].pop('query_layout', None)
    return projected


def _attention_preparation_changes(reference, candidate):
    old, new = reference['prepared_execution'], candidate['prepared_execution']
    require(new['kernel_family'] == ATEN_FAMILY,
            'attention comparison candidate must use the explicitly admitted ATen family')
    changes = {}
    if old['kernel_family'] != new['kernel_family']:
        require(old['kernel_family'] == SERVICE['kernel_family'], 'attention family reference must be RN12')
        for key in ('kernel_family', 'rn_heads_per_block', 'aten_tiles'):
            changes[key] = {'classification':'numerical-family', 'reference':old.get(key), 'candidate':new.get(key)}
    else:
        require(old['aten_tiles'].get('query_layout') in (None, 'native-gqa-v1'),
                'ATen layout reference must retain default or explicit native GQA')
        if old['aten_tiles'] != new['aten_tiles']:
            changes['aten_tiles'] = {'classification':'numerical-layout',
                'reference':old['aten_tiles'], 'candidate':new['aten_tiles']}
    require(_attention_preparation_projection(old, changes) == _attention_preparation_projection(new, changes),
            'attention comparison changed unrelated canonical preparation/credit/graph authority')
    return changes


def _attention_retention(row, operation, changes):
    from copy import deepcopy

    from tensorfold.cuda.sm70.paged_attention import rn_preparation_identity
    from tensorfold.families.qwen4_exp.cuda.execution_identity import digest
    profile = row['teacher_profile']
    provider = _validate_attention_provider(profile, operation)
    raw = profile['numerical_trial_retention']
    require(profile['teacher_execution_registration']['binding']['identity']['retention'] == digest(raw),
            'raw attention retention differs from its actual issued fingerprint')
    selected = operation['prepared_execution']
    owner = raw.get('owner_preparation')
    require(type(owner) is dict and owner.get('specification') == {k:v for k,v in selected.items() if k != 'trial'}
            and owner.get('roles') == {'decode':120,'mtp':120,'prefill':256},
            'actual attention retention preparation/roles differ')
    aten = selected['kernel_family'] == ATEN_FAMILY
    expected = _preparation_owner(operation)
    if aten:
        require('rn_numerical_identity' not in owner and owner.get('aten_operator_identity') == provider['operator']
                and owner.get('aten_tensor_requirements') == expected.aten_requirements,
                'ATen retained operator/resource records differ from actual runtime requirements')
    else:
        require(owner.get('rn_numerical_identity') == rn_preparation_identity()
                and 'aten_operator_identity' not in owner and 'aten_tensor_requirements' not in owner,
                'RN12 reference acquired unrequested attention arithmetic or resources')
    normalized = deepcopy(raw)
    retained = normalized['owner_preparation']
    retained['specification'] = _attention_preparation_projection(retained['specification'], changes)
    allocations = retained.get('allocations')
    require(type(allocations) is dict and set(allocations) == {'decode','mtp','prefill'},
            'actual attention allocation role lists required')
    for role, records in allocations.items():
        require(type(records) is list and len(records) == WORLD
                and all(type(record) is dict and type(record.get('rank')) is int and record['rank'] == rank
                        and record.get('rows') == retained['roles'][role] for rank,record in enumerate(records)),
                'actual eight-rank attention allocation records required')
        if aten:
            payload = expected.aten_requirements[role]['persistent_bytes']
            for record in records:
                for key in ('device_tensor_bytes','native_tensor_bytes'):
                    require(type(record.get(key)) is int and record[key] >= payload,
                            'paid ATen attention payload is missing from its actual allocation charge')
                    if 'kernel_family' in changes:
                        record[key] -= payload
    if 'kernel_family' in changes:
        for key in ('rn_numerical_identity','aten_operator_identity','aten_tensor_requirements'):
            retained.pop(key,None)
    return normalized


def _attention_provider_comparison(profile, operation, changes):
    from copy import deepcopy
    value = deepcopy(_validate_attention_provider(profile, operation))
    if 'kernel_family' in changes:
        value['roles'] = {}  # Only the verified, paid family-specific tile metadata varies.
    elif 'aten_tiles' in changes:
        for role in value['roles'].values():
            role.pop('query_layout',None)
    return value



def preparation_changes(reference, candidate):
    require(_preparation_contract(reference) == _preparation_contract(candidate), 'paired C8 preparation contracts differ')
    if _preparation_contract(reference) is None:
        return None
    for operation in (reference, candidate):
        validate_preparation(operation['prepared_execution'], operation['numerical_trial'], _preparation_contract(operation))
    if _preparation_contract(reference) == ATEN_PREPARATION_CONTRACT:
        return _attention_preparation_changes(reference, candidate)
    old, new = reference['prepared_execution'], candidate['prepared_execution']
    require(not any(key in old for key in PREPARATION_OPTIONS), 'original C8 preparation reference must leave all candidate options off')
    changes = {key: {'classification': 'transport-contract' if key in TRANSPORT_OPTIONS else 'numerical',
                     'reference': old.get(key), 'candidate': new.get(key)}
               for key in PREPARATION_OPTIONS if old.get(key) != new.get(key)}
    require({k: v for k, v in old.items() if k not in changes} == {k: v for k, v in new.items() if k not in changes},
            'candidate changed an unrelated C8 preparation policy')
    return changes


def _comparison_retention(row, operation, changes):
    result = _preparation_retention(row, operation, changes)
    _validate_startup_capacity(row['teacher_profile'], operation)
    if _capacity_contract(operation) is not None:
        from copy import deepcopy
        result = deepcopy(result)
        proof = row['teacher_profile']['full_window_capacity']
        validated = _validate_capacity_plan(proof['capacity'], slots=24, bits=8, local_rank=proof['rank'])
        result['window_capacity_contract'] = {'schema':CAPACITY_PROOF_SCHEMA,
            'contract':CAPACITY_CONTRACT, 'slots':24, 'bits':8, 'logical_context':262144,
            'kv_pair':proof['kv_pair'],
            'physical_capacity':validated['physical'], 'future_service':validated['future'],
            'absolute_grant_caps':validated['absolute_caps']}
    return result


def _preparation_retention(row, operation, changes):
    if _preparation_contract(operation) == ATEN_PREPARATION_CONTRACT:
        return _attention_retention(row, operation, changes)
    if changes is None:
        return numerical._normalized_retention(row, operation)
    from copy import deepcopy

    from tensorfold.families.qwen4_exp.cuda.execution_identity import digest
    profile = row['teacher_profile']
    raw = profile['numerical_trial_retention']
    require(profile['teacher_execution_registration']['binding']['identity']['retention'] == digest(raw),
            'raw C8 preparation retention must retain its actual issued identity')
    owner = raw.get('owner_preparation')
    require(type(owner) is dict and owner.get('specification') == {
            k: v for k, v in operation['prepared_execution'].items() if k != 'trial'},
            'C8 preparation retention differs from the explicit profile')
    normalized = deepcopy(raw)
    selected = operation['prepared_execution']
    owner = normalized['owner_preparation']
    owner['specification'] = {k: v for k, v in owner['specification'].items() if k not in changes}
    for key in ('native_family', 'prepared_linears', 'prepared_router', 'coordinate_staging', 'row_sampling_scratch', 'dense_projections'):
        require((key in owner) == (key in selected), 'unrequested or missing C8 retained capability: '+key)
        if key in changes:
            owner.pop(key, None)
    allocations = owner.get('allocations')
    require(type(allocations) is dict and set(allocations) == {'decode', 'mtp', 'prefill'},
            'actual C8 preparation allocation roles required')
    for records in allocations.values():
        require(type(records) is list and len(records) == WORLD
                and all(type(record) is dict and type(record.get('rank')) is int and record['rank'] == rank
                        for rank, record in enumerate(records)),
                'actual eight-rank C8 preparation allocation records required')
        for allocation in records:
            if 'row_sampling_scratch' in changes:
                allocation.pop('row_sampling_scratch', None)
            if 'descriptor_upload' in changes:
                # Only the three descriptor host extents vary with the selected ring.
                # Raw paid allocations remain in the owning startup/retention record.
                allocation.pop('host_staging_bytes', None)
                allocation.pop('pinned_host_staging_bytes', None)
                allocation.pop('host_metadata_bytes', None)
    return normalized


def compare(reference, candidate, *, control, output):
    if __package__:
        from . import v100_target_window_evidence as shards
    else:
        import v100_target_window_evidence as shards
    left, right = _comparison_side(reference, control), _comparison_side(candidate, control)
    a, b = left['inner']['operation'], right['inner']['operation']
    require(_capacity_contract(a) == _capacity_contract(b), 'comparison capacity contract differs')
    changes = preparation_changes(a, b)
    require(not any(a['numerical_trial'].values()) and a['selector'] == selectors()[0],
            'window comparison requires its all-off original expert control')
    require(left['inner']['image_id_declared'] == right['inner']['image_id_declared'], 'same immutable window image required')
    for key in ('model_assets', 'model_entrypoint', 'mtp_artifact', 'frontend_assets', 'frontend_root',
                'teacher_scope', 'mtp_confidence', 'numa', 'output_grants'):
        require(a[key] == b[key], 'window comparison changed unrelated held inputs: '+key)
    for key in ('source_manifest', 'source_fixture', 'source', 'image_id'):
        old = left['host'].get('source_after', {}).get(key)
        require(old is not None and old == right['host'].get('source_after', {}).get(key),
                'window canonical source/image/fixture differs')
    profiles = [_held_input(side, side['inner']['operation']['execution_manifest'], control) for side in (left, right)]
    normalized = []
    for side, profile in zip((left, right), profiles):
        operation = side['inner']['operation']
        require(profile.get('preparation') == operation['prepared_execution']
                and side['authority']['plan']['identity']['profile_sha256'] == operation['execution_manifest']['sha256'],
                'window profile/forced plan must name its own exact authority')
        preparation = (_attention_preparation_projection(profile['preparation'], changes)
            if _preparation_contract(operation) == ATEN_PREPARATION_CONTRACT else
            {k:v for k,v in profile['preparation'].items() if k not in (numerical.AXES if changes is None else changes)})
        normalized.append({**profile, 'preparation':preparation})
    require(normalized[0] == normalized[1], 'window comparison changed unrelated profile artifacts or preparation')
    for rank in range(WORLD):
        old, new = left['proofs'], right['proofs']
        require(_comparison_retention(old['startups'][rank], a, changes) ==
                _comparison_retention(new['startups'][rank], b, changes), 'unrelated window retention policy changed')
        if _preparation_contract(a) == ATEN_PREPARATION_CONTRACT:
            require(_attention_provider_comparison(old['startups'][rank]['teacher_profile'], a, changes) ==
                    _attention_provider_comparison(new['startups'][rank]['teacher_profile'], b, changes),
                    'attention comparison changed protected SDK or unrelated role metadata')
        registrations = [side['registrations'][rank] for side in (old, new)]
        numbers = []
        for registration, operation in zip(registrations, (a, b)):
            actual = dict(registration['numerics'])
            actual.pop('owner_preparation_numerics', None)  # Already validated against the actual runtime-owned selection.
            plan = actual['owned_expert_native']
            actual['owned_expert_native'] = plan[:11]+['<explicit-expert-selector>']
            numbers.append(actual)
        require(numbers[0] == numbers[1], 'window comparison changed unrelated numerical semantics')
        for key, value in registrations[0]['binding']['identity'].items():
            if key not in ('numerics', 'kernel_build', 'retention'):
                require(registrations[1]['binding']['identity'].get(key) == value, 'unrelated window execution identity changed: '+key)
        require({k: v for k, v in registrations[0]['contents'].items() if k.startswith('source/')} ==
                {k: v for k, v in registrations[1]['contents'].items() if k.startswith('source/')},
                'complete installed source differs between window arms')
        old_native = {row['name']: row for row in old['native'][rank]['mapped_elfs']}
        new_native = {row['name']: row for row in new['native'][rank]['mapped_elfs']}
        require(all(name in new_native and new_native[name]['sha256'] == row['sha256'] for name, row in old_native.items()),
                'window trial changed a baseline native provider')
        allowed = []
        if changes is not None:
            if b['prepared_execution'].get('prepared_router') is not None:
                allowed.append('tf_sm70_routing_')
            if b['prepared_execution'].get('dense_projections') is not None:
                allowed.extend(('tf_sm70_dense_', 'tf_sm70_prefill_dense_'))
            if any(b['prepared_execution'].get(key) for key in ('native_hc_writeback_target_eager', 'gdn_value8_capture')):
                allowed.append('tf_sm70_family_')
        if b['numerical_trial']['native_family'] or b['numerical_trial']['owned_q8_window_prepare']:
            allowed.append('tf_sm70_family_')
        if b['selector']['gu'] not in ('incumbent', 'tensorfold'):
            allowed.extend(('tf_onecat_gu_', 'tf_onecat_pack_'))
        if b['selector']['down'] != 'tensorfold':
            allowed.append('tf_onecat_down_')
        require(all(name.startswith(tuple(allowed)) for name in new_native.keys()-old_native.keys()),
                'window trial introduced an unrelated native library')
        if _preparation_contract(a) == ATEN_PREPARATION_CONTRACT:
            old_libraries = {name:row for name,row in registrations[0]['contents'].items() if name.startswith('library/')}
            new_libraries = {name:row for name,row in registrations[1]['contents'].items() if name.startswith('library/')}
            require(all(new_libraries.get(name) == row for name,row in old_libraries.items()),
                    'attention trial changed a baseline SDK/library authority')
            require(all(Path(name.removeprefix('library/')).name.startswith(tuple(allowed))
                        for name in new_libraries.keys()-old_libraries.keys()),
                    'attention trial introduced an unrelated external library bundle')
    report = shards.compare_rows(left['root']/'teacher', right['root']/'teacher', left['authority'], right['authority'],
                                  control=control, output=output)
    row_path = Path(output)/'comparison.json'
    row_ref = {'file': row_path.name, 'bytes': row_path.stat().st_size,
               'sha256': frames._owned_teacher_hash(row_path, row_path.stat().st_size, 8 << 20, control)}
    report.update(schema='tensorfold.target-window-teacher-owned-comparison-v4', row_report=row_ref,
                  reference=reference, candidate=candidate, numerical_trial=b['numerical_trial'], selector=b['selector'],
                  image_id=left['inner']['image_id_declared'],
                  actual_profiles=[side['proofs']['startups'][0]['teacher_profile'] for side in (left, right)],
                  numa_evidence=[left['model']['numa'], right['model']['numa']],
                  scope='forced production C8 target windows with actual MTP/graph records; target-logit metrics only; no held-out accuracy, MTP-head KL or free-generation acceptance')
    if changes is not None:
        report['preparation_contract'] = _preparation_contract(b)
        report['preparation_changes'] = changes
        report['preparation_comparison_scope'] = 'Declared finite experiment variables only; raw profiles, paid retention and issued identities remain in both positive owners. Transport contracts are not measured equivalence or quality acceptance.'
    frames._owned_teacher_json(Path(output)/'owned-comparison.json', report, 8 << 20, control)
    (Path(output)/'owned-comparison.json').chmod(0o444)
    return report


def main(argv=None):
    import argparse
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run', help='own one exact reviewed forced-window Host operation')
    for name in ('spec', 'out', 'operation-gate', 'review'):
        run.add_argument('--'+name, type=Path, required=True)
    run.add_argument('--spec-sha256', required=True)
    run.add_argument('--spec-bytes', type=int, required=True)
    read = commands.add_parser('compare', help='validate both positive owners and stream conditional diagnostics')
    read.add_argument('--input', type=Path, required=True)
    read.add_argument('--input-sha256', required=True)
    read.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'run':
        require(legacy.sha(args.spec_sha256) and type(args.spec_bytes) is int and 0 < args.spec_bytes <= 8 << 20,
                'bounded exact Host identity required')
        numerical.base.complete_cli(args.spec, {'bytes': args.spec_bytes, 'sha256': args.spec_sha256}, args.out.absolute(),
            args.operation_gate, args.review, contract=sys.modules[__name__])
        return 0
    raw = frames._owned_teacher_read(args.input, 8 << 20)
    require(legacy.sha(args.input_sha256) and hashlib.sha256(raw).hexdigest() == args.input_sha256,
            'exact closed window comparison input required')
    data = frames._owned_teacher_object(raw)
    require(type(data) is dict and set(data) == {'schema', 'reference', 'candidate', 'read_timeout_s'}
            and data['schema'] == 'tensorfold.target-window-teacher-comparison-input-v4',
            'closed versioned window comparison DATA required')
    require(type(data['read_timeout_s']) in (int, float) and math.isfinite(data['read_timeout_s'])
            and 0 < data['read_timeout_s'] <= 7500, 'explicit bounded comparison read deadline required')
    compare(data['reference'], data['candidate'], control=frames._teacher_read_control(data['read_timeout_s']),
            output=args.output.absolute())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
