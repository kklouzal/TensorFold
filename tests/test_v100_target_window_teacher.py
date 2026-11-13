"""Finite forced-window contract/resource controls; no fixture is GPU evidence."""
import json

import pytest

from tools import v100_target_window_teacher as window


def plan(tokens=128):
    cases = [{'case_id': f'case_{i}', 'prompt_ids': [1, 2, 3], 'forced_ids': [7]*tokens} for i in range(8)]
    rounds = [{'lanes': [{'case_id': row['case_id'], 'rows': min(5, tokens-1-step), 'keep': 1}
                         for row in cases]} for step in range(tokens-1)]
    return {'schema': window.PLAN_SCHEMA, 'identity': {'run_id': 'fixture', 'corpus_id': 'explicit_fixture',
        'model_sha256': '1'*64, 'frontend_sha256': '2'*64, 'profile_sha256': '3'*64},
        'vocab': 248320, 'context': 262144, 'resident_states': 24, 'target_rows': 120, 'prefill_rows': 256,
        'method': 'mtp', 'depth': 4, 'mtp_confidence': 0.7, 'proposal_policy': window.PROPOSAL_POLICY, 'sequences': cases, 'rounds': rounds, 'metrics': dict(window.METRICS),
        'read_timeout_s': 60, 'quality_scope': 'diagnostic-forced-prefix-not-heldout'}


def test_overlap_rows_are_retained_without_becoming_independent_corpus_samples():
    value = plan()
    window.validate_plan(value)
    budget = window.resources(len(json.dumps(value)), value)
    assert budget['maximum_total_frames'] == 5008  # Eight prefills + 8*(123*5+4+3+2+1).
    assert budget['maximum_total_frames'] > 8*128
    assert budget['additional_project_device_bytes'] == 993280
    assert budget['device_scratch']['shape'] == [8, 31040]
    assert budget['max_total_frame_file_bytes'] == 5008*(248320*4+4096)
    assert value['metrics']['corpus'] == 'unique-case-position-first-observation-v1'
    assert value['metrics']['windows'] == 'all-window-rows-v1'


@pytest.mark.parametrize('fault', ['early_finish', 'missing_case', 'oversized_window', 'invalid_keep',
                                  'repeated_case', 'gold_missing', 'metrics', 'booleans'])
def test_schedule_refuses_incomplete_or_ambiguous_transactions(fault):
    value = plan(8)
    if fault == 'early_finish':
        value['rounds'][0]['lanes'][0]['keep'] = 5
    elif fault == 'missing_case':
        value['rounds'][0]['lanes'].pop()
    elif fault == 'oversized_window':
        value['rounds'][0]['lanes'][0]['rows'] = 6
    elif fault == 'invalid_keep':
        value['rounds'][0]['lanes'][0]['keep'] = 0
    elif fault == 'repeated_case':
        value['rounds'][0]['lanes'][1]['case_id'] = 'case_0'
    elif fault == 'gold_missing':
        value['rounds'][-1]['lanes'][0]['rows'] = 2
    elif fault == 'metrics':
        value['metrics']['corpus'] = 'unspecified-average'
    elif fault == 'booleans':
        value['rounds'][0]['lanes'][0]['rows'] = True
    with pytest.raises(ValueError):
        window.validate_plan(value)


def test_each_actual_case_may_share_prompt_but_not_identity():
    value = plan(4)
    window.validate_plan(value)
    value['sequences'][1]['case_id'] = value['sequences'][0]['case_id']
    with pytest.raises(ValueError, match='unique'):
        window.validate_plan(value)


def test_funded_device_obligation_is_explicit_and_input_bound():
    from tensorfold.cli import _OwnedExpertQualification
    def callback(engine):
        return None
    host = (1 << 20, 'a'*64)
    good = _OwnedExpertQualification(callback, host_obligation=host, device_obligation=(993280, host[1]))
    assert good.device_obligation == (993280, host[1])
    for device in ((993280, 'b'*64), (0, host[1]), (True, host[1]), (993280,), None):
        with pytest.raises(TypeError, match='device'):
            _OwnedExpertQualification(callback, host_obligation=host, device_obligation=device)
    with pytest.raises(TypeError, match='device'):
        _OwnedExpertQualification(callback, device_obligation=(993280, host[1]))


def test_round_failure_restores_every_original_alias_without_running_an_engine(monkeypatch):
    import threading
    from types import SimpleNamespace

    from tensorfold.cleanup import raise_failures
    from tensorfold.families.qwen4_exp.cuda import decode, mtp, multi
    observer = object.__new__(window.WindowTeacher)
    observer.thread, observer.failed = None, None
    observer.thread_id, observer.round_index = threading.get_ident, 0
    observer.prepared = {'data': plan(2)}
    observer.streams = {i: ({'case_id': f'case_{i}'}, SimpleNamespace(sid=i, out=[])) for i in range(8)}
    observer._snapshot = lambda stream: {}
    observer.decoder = SimpleNamespace(filling=[], streams={}, graphs=SimpleNamespace(
        verify_forward=lambda *a, **kw: None, mtp_step=lambda *a, **kw: None))
    marker = RuntimeError('original-round-marker')
    def failed_round():
        raise marker
    observer.originals, observer.raise_failures = {'round': failed_round}, raise_failures
    monkeypatch.setattr(window, 'graph_counts', lambda _: {'serving_hits': {'verify': 0, 'mtp': 0},
                                                        'serving_misses': {'verify': 0, 'mtp': 0}})
    originals = [(decode.Engine, 'sample', decode.Engine.sample), (decode.Engine, 'sample_draft', decode.Engine.sample_draft),
        (mtp, '_mtp_stage', mtp._mtp_stage)] + [
        (multi, name, getattr(multi, name)) for name in ('choose_gathered', 'tp_sample_rows', 'stage', 'accept')]
    with pytest.raises(RuntimeError) as caught:
        observer.rounded()
    assert caught.value is marker and observer.thread is None
    assert all(getattr(owner, name) is original for owner, name, original in originals)


def test_graph_evidence_failure_does_not_skip_original_engine_retirement(monkeypatch, tmp_path):
    import threading
    from types import SimpleNamespace

    from tensorfold.cleanup import raise_failures
    observer = object.__new__(window.WindowTeacher)
    engine = SimpleNamespace(rank=0, _closed=False, _lifecycle=threading.RLock())
    observer.engine, observer.decoder = engine, SimpleNamespace(graphs=object())
    observer.prepared = {'data': plan(2), 'input_sha256': 'a'*64,
        'resources': {'maximum_total_frames': 16, 'receipt_bytes_limit': 1 << 20}}
    observer.directory = tmp_path
    observer.failed, observer.close_attempted, observer.closed = None, False, False
    observer.finished = {f'case_{i}' for i in range(8)}
    observer.frame_index, observer.round_index = 16, 1
    observer.window_refs, observer.prefill_refs, observer.restores = [], [], []
    observer.mtp_stage_refs, observer.mtp_pick_refs = [], []
    observer.runtime_sources = {key: '1'*64 for key in ('engine', 'multi', 'decode', 'sampling', 'teacher', 'window_teacher')}
    observer.profile, observer.initial_graphs = {}, {}
    observer.streams, observer.by_state, observer.originals = {}, {}, {}
    observer.gathered, observer.cpu = object(), object()
    observer.raise_failures = raise_failures
    retired = []
    def old_close():
        retired.append(True)
        engine._closed = True
    observer.old_close = old_close
    marker = ValueError('graph-evidence-marker')
    def fail_graphs(_):
        raise marker
    monkeypatch.setattr(window, 'graph_counts', fail_graphs)
    with pytest.raises(ValueError) as caught:
        observer.close()
    assert caught.value is marker and retired == [True]
    assert observer.closed and observer.gathered is None and observer.cpu is None
    with pytest.raises(ValueError, match='cannot retry'):
        observer.close()


def test_forced_keep_records_original_acceptance_and_uses_actual_commit_path_shape():
    from types import SimpleNamespace

    from tensorfold.cuda.streams import accept
    observer = object.__new__(window.WindowTeacher)
    case = plan(8)['sequences'][0]
    stream = SimpleNamespace(sid=1, out=[7])
    lane = {'case_id': case['case_id'], 'rows': 5, 'keep': 2}
    observer.streams = {1: (case, stream)}
    observer.staged = [(stream, 0, 5, lane)]
    observer.accepted = []
    observer.sampled = [{'case_id': case['case_id'], 'gold_token': 7, 'original_sampled': 7} for _ in range(5)]
    path, terminal = observer._accept(accept, [7]*5, [-1, 0, 1, 2, 3], [7]*5, 7)
    assert path == [0, 1] and terminal == 7
    assert observer.accepted == [{'case_id': case['case_id'], 'original_path': [0, 1, 2, 3, 4],
        'original_terminal': 7, 'forced_path': [0, 1], 'forced_terminal': 7,
        'role': 'diagnostic-forced-keep-not-serving-acceptance',
        'original_scope': 'counterfactual-forced-proposals-and-prefix-v1'}]


def test_forced_sampler_rows_reach_mtp_refresh_while_original_predictions_remain_recorded():
    from types import SimpleNamespace

    import torch
    observer = object.__new__(window.WindowTeacher)
    plan_token = object()
    logits = torch.zeros((8, window.ROW_WIDTH), dtype=torch.float32)
    observer.decoder = SimpleNamespace(_sample_plan=plan_token, buf=SimpleNamespace(logits=logits))
    observer.current, observer.sampled, observer.delegating = {}, [], False
    observer.streams, observer.staged = {}, []
    for index, case in enumerate(plan(4)['sequences']):
        stream = SimpleNamespace(sid=index, out=[7])
        observer.streams[index] = (case, stream)
        observer.staged.append((stream, index, index+1, {'case_id': case['case_id'], 'rows': 1, 'keep': 1}))
    def capture(stream, source, position, **kwargs):
        return {'case_id': observer.streams[stream.sid][0]['case_id'], 'gold_token': 7}
    observer._capture = capture
    calls = []
    def original(*args, **kwargs):
        calls.append(True)
        return [3]*8
    got = observer._sample(original, (None, None, 8, [4]*8, None), {'plan': plan_token, 'logits': logits}, gathered=True)
    assert calls == [True] and got == [7]*8
    assert [row['original_sampled'] for row in observer.sampled] == [3]*8
    assert not observer.delegating and torch.equal(logits, torch.zeros_like(logits))


def test_forced_keep_does_not_relabel_original_argmax_acceptance():
    from types import SimpleNamespace

    from tensorfold.cuda.streams import accept
    observer = object.__new__(window.WindowTeacher)
    case = plan(8)['sequences'][0]
    stream = SimpleNamespace(sid=1, out=[7])
    observer.streams = {1: (case, stream)}
    observer.staged = [(stream, 0, 5, {'case_id': case['case_id'], 'rows': 5, 'keep': 2})]
    observer.accepted = []
    observer.sampled = [{'case_id': case['case_id'], 'gold_token': 7, 'original_sampled': 3} for _ in range(5)]
    assert observer._accept(accept, [7]*5, [-1, 0, 1, 2, 3], [7]*5, 7) == ([0, 1], 7)
    assert observer.accepted[0]['original_path'] == [0] and observer.accepted[0]['original_terminal'] == 3
    with pytest.raises(ValueError, match='conditioning'):
        observer.accepted = []
        observer._accept(accept, [7]*5, [-1, 0, 1, 2, 3], [3]*5, 7)


@pytest.mark.parametrize('confidence', [0.0, 0.7])
def test_actual_serial_draft_loop_consumes_gold_at_every_proposal_step(confidence):
    from types import MethodType, SimpleNamespace

    import torch

    from tensorfold.families.qwen4_exp.cuda.decode import draft
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder
    from tensorfold.families.qwen4_exp.cuda.speculation import MethodGeometry
    observer = object.__new__(window.WindowTeacher)
    data = plan(8)
    data['mtp_confidence'] = confidence
    case = data['sequences'][0]
    state = SimpleNamespace(pos=3, mtp_len=2, mtp_drafted=0, history_slot=2, slot_generation=1)
    state.set_mtp_len = lambda value: setattr(state, 'mtp_len', value)
    state.begin_mtp_proposals = lambda: None
    stream = SimpleNamespace(sid=5, st=state, count=8, out=[7])
    decoder = SimpleNamespace(capacity=262149, logical_context=262144, depth=4,
        speculation=MethodGeometry('mtp', 4), buf=SimpleNamespace(rows=120), _proposal_limit=lambda: 4)
    decoder._proposal_count = MethodType(MultiDecoder._proposal_count, decoder)
    weights = SimpleNamespace(dspark=None)
    slot = SimpleNamespace(w=weights, st=state, mbuf=SimpleNamespace(streams=torch.zeros((5, 1))))
    observer.engine, observer.decoder = SimpleNamespace(w=weights, rank=0), decoder
    observer.prepared, observer.current = {'data': data, 'input_sha256': 'a'*64}, None
    observer.streams, observer.by_state = {5: (case, stream)}, {id(state): stream}
    observer.mtp_pick_refs = []
    captured, consumed = [], []
    def write_pick(kind, record):
        assert kind == 'pick'
        captured.append(record)
        observer.mtp_pick_refs.append(record)
    observer._mtp_file = write_pick
    def forward(tokens, hidden):
        assert tokens == observer._gold_span(case, state.mtp_len+1, len(tokens))
        consumed.append(list(tokens))
        observer.last_mtp_stage = {'stage_index': len(consumed)-1,
            'lanes': [{'case_id': case['case_id'], 'token_position_start': state.mtp_len+1,
                      'token_ids': list(tokens), 'history_slot': state.history_slot, 'slot_generation': state.slot_generation}]}
        return torch.zeros((1, 1))
    slot.mtp_forward = forward
    def original_confidence(*args):
        return 3, .1  # Deliberately wrong and below0.7; ordinary code would stop.
    def sampled(logits, position, sampling):
        return observer._serial_pick(original_confidence, slot, logits, position, sampling)
    slot.sample_draft = sampled
    def original_sample(*args, **kwargs):
        return [3]
    def sample(logits, positions, sampling, **kwargs):
        return observer._first_sample(original_sample, slot, logits, positions, sampling, **kwargs)
    slot.sample = sample
    got = draft(slot, torch.zeros((1, 1)), [7], 4, 4, None, confidence)
    assert got == [7]*4 and consumed == [[7]]*4
    assert state.mtp_len == 6 and state.mtp_drafted == 3
    assert [row['rows'][0]['original_token'] for row in captured] == [3]*4
    assert [row['rows'][0]['original_probability'] for row in captured] == ([.1]*4 if confidence else [None]*4)
    assert all(row['rows'][0]['forced_probability'] == 1.0 for row in captured)
    assert [row['rows'][0]['proposal_index'] for row in captured] == list(range(4))


def test_actual_batched_drafter_chains_gold_after_recording_low_probability_original_picks(monkeypatch):
    from types import MethodType, SimpleNamespace

    import torch

    from tensorfold.families.qwen4_exp.cuda import multi
    from tensorfold.families.qwen4_exp.cuda.speculation import MethodGeometry
    observer = object.__new__(window.WindowTeacher)
    data = plan(8)
    decoder = SimpleNamespace(w=SimpleNamespace(dspark=None), graphs=None, depth=4, confidence=.7,
        capacity=262149, logical_context=262144, speculation=MethodGeometry('mtp', 4), controller=None,
        mbuf=SimpleNamespace(streams=torch.zeros((120, 1)), input_preparation_context=None),
        buf=SimpleNamespace(streams=torch.zeros((120, 1)), rows=120), kv_host=object(), _proposal_limit=lambda: 4)
    decoder._proposal_count = MethodType(multi.MultiDecoder._proposal_count, decoder)
    observer.decoder, observer.engine = decoder, SimpleNamespace(rank=0)
    observer.prepared = {'data': data, 'input_sha256': 'a'*64}
    observer.current, observer.mtp_pick_step, observer.round_index = data['rounds'][0], 0, 0
    observer.streams, observer.staged, observer.mtp_pick_refs = {}, [], []
    supplied, consumed, picks = [], [], []
    def publish(kind, record):
        assert kind == 'pick'
        observer.mtp_pick_refs.append(record)
        picks.append(record)
    observer._mtp_file = publish
    for i, case in enumerate(data['sequences']):
        state = SimpleNamespace(pos=4, mtp_len=6, mtp_drafted=3, history_slot=i, slot_generation=1)
        state.set_mtp_len = lambda value, st=state: setattr(st, 'mtp_len', value)
        state.begin_mtp_proposals = lambda: None
        stream = SimpleNamespace(sid=i, st=state, count=8, out=[7], drafts=[], copies=None, sampling=None)
        observer.streams[i] = (case, stream)
        observer.staged.append((stream, 5*i, 5*(i+1), {'case_id': case['case_id'], 'rows': 5, 'keep': 1}))
        supplied.append((stream, 5*i, [7]))
    def stage(buffers, windows):
        lanes, segments = [], []
        for index, (state, ids, hidden) in enumerate(windows):
            case, stream = observer.streams[state.history_slot]
            assert ids == observer._gold_span(case, state.mtp_len+1, len(ids))
            consumed.append((stream.sid, list(ids)))
            lanes.append({'case_id': case['case_id'], 'token_position_start': state.mtp_len+1,
                'token_ids': list(ids), 'history_slot': state.history_slot, 'slot_generation': state.slot_generation})
            segments.append((state, index, index+1))
        observer.last_mtp_stage = {'stage_index': len(consumed)//8-1, 'lanes': lanes}
        return segments
    monkeypatch.setattr(multi, 'mtp_stage', stage)
    class Words:
        def __init__(self, segments, host, completed):
            self.completed = completed
        def settle(self):
            if self.completed is not None:
                completed, self.completed = self.completed, None
                completed()
    monkeypatch.setattr(multi, 'KVWords', Words)
    def original_picks(logits, positions, samplings, words):
        words.settle()
        return [(3, .1)]*len(positions)
    observer.originals = {'_picks': original_picks}
    decoder._picks = observer._mtp_picks
    decoder._mtp = lambda segments, **kwargs: torch.zeros((len(segments), 1))
    multi.MultiDecoder._draft_all(decoder, supplied)
    assert len(consumed) == 32 and all(ids == [7] for sid, ids in consumed)
    assert all(stream.drafts == [7]*4 for stream, _, _ in supplied)
    assert all(stream.st.mtp_len == 7 and stream.st.mtp_drafted == 3 for stream, _, _ in supplied)
    assert len(picks) == 4 and all(len(row['rows']) == 8 for row in picks)
    assert all(row['original_token'] == 3 and row['original_probability'] == .1
               and row['forced_token'] == 7 and row['forced_probability'] == 1.0 for pick in picks for row in pick['rows'])


@pytest.mark.parametrize('fault', [None, 'device_ids', 'coordinates', 'input_ids'])
def test_actual_eight_state_mtp_staging_binds_gold_device_ids_and_coordinates(fault):
    from types import SimpleNamespace

    import torch
    from test_qwen4_exp_position_coordinates import policy

    from tensorfold.families.qwen4_exp.cuda.compact_indexer import Staging
    from tensorfold.families.qwen4_exp.cuda.mtp import _mtp_stage

    observer = object.__new__(window.WindowTeacher)
    data = plan(8)
    position = policy('mtp', limit=262144)
    staging = Staging(120, 13, 'cpu', mtp=True)
    event = SimpleNamespace(synchronize=lambda: None, record=lambda: None)
    buffers = SimpleNamespace(rows=120, staged=event, mtp_in=torch.zeros((120, 1)),
        ids_host=torch.zeros(120, dtype=torch.int64), ids=torch.zeros(120, dtype=torch.int64),
        last_host=torch.zeros(8, dtype=torch.int64), last=torch.zeros(8, dtype=torch.int64),
        input_preparation_context=None, compact_indexer_stage=staging,
        mtp_indexer_position=position, mtp_indexer_coordinate_storage=staging.mtp_coordinates,
        mtp_indexer_frame_id=0, mtp_indexer_coordinates=None, mtp_indexer_frame=None)
    buffers.ids_host_array, buffers.last_host_array = buffers.ids_host.numpy(), buffers.last_host.numpy()
    observer.decoder = SimpleNamespace(mbuf=buffers, pbuf=object(),
        controller=SimpleNamespace(effective={'prefill_tokens': 256, 'row_budget': 120}),
        _proposal_limit=lambda: 4)
    observer.engine, observer.current, observer.round_index = SimpleNamespace(rank=0), data['rounds'][0], 0
    observer.prepared = {'data': data, 'input_sha256': 'a'*64}
    observer.streams, observer.by_state, observer.mtp_stage_refs = {}, {}, []
    captured = []
    observer._mtp_file = lambda kind, record: captured.append(record)
    windows = []
    for index, case in enumerate(data['sequences']):
        state = SimpleNamespace(pos=4, mtp_len=3, capacity=262149, history_slot=index,
            slot_generation=1, history_generation=1, image_positions=None, rope_delta=index,
            position_profile_identity=position.identity, position_epoch=position.epoch,
            coordinate_profile_identity='b'*64, media_identity=None, media_input_span=None,
            media_profile_identity=None, mtp_boundary=None, pos_dev=torch.zeros(1, dtype=torch.int32))
        stream = SimpleNamespace(sid=index, st=state)
        observer.streams[index], observer.by_state[id(state)] = (case, stream), stream
        windows.append((state, [7, 7], torch.zeros((2, 1))))
    if fault == 'input_ids':
        windows[0][1][0] = 3
    def actual_stage(actual_buffers, actual_windows, **kwargs):
        if fault == 'coordinates':
            actual_windows[0][0].rope_delta += 1
        result = _mtp_stage(actual_buffers, actual_windows, **kwargs)
        if fault == 'coordinates':
            actual_windows[0][0].rope_delta -= 1
        if fault == 'device_ids':
            actual_buffers.ids[0] = 3
        return result
    if fault is not None:
        with pytest.raises(ValueError, match='MTP'):
            observer._mtp_stage(actual_stage, buffers, windows)
        assert captured == []
    else:
        segments = observer._mtp_stage(actual_stage, buffers, windows)
        assert len(segments) == 8 and buffers.ids[:16].tolist() == [7]*16
        assert len(captured) == 1 and captured[0]['rows'] == 16
        assert [lane['coordinate_start'] for lane in captured[0]['lanes']] == list(range(3, 11))
        assert buffers.mtp_indexer_coordinates.tolist() == [
            [coordinate]*3 for start in range(3, 11) for coordinate in (start, start+1)]
