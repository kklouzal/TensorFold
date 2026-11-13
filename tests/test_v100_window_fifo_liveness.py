"""Real installed FIFO/scheduler readiness through cold prefill into a forced window.

Only physical State allocation, GPU prefill compute/capture and final sampler
are CPU stand-ins. Every readiness, ordering, piece-selection, fill/round,
first-sample forcing and teacher transaction method is the production method.
"""
import threading
from types import MethodType, SimpleNamespace

import pytest
import torch

from tensorfold.cuda.scheduler import Scheduler
from tensorfold.cuda.streams import Stream
from tensorfold.families.qwen4_exp.cuda import decode, multi
from tensorfold.families.qwen4_exp.cuda.exact_graphs import ExactStateGraphs
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder
from tests.test_v100_target_window_teacher import plan
from tools import v100_target_window_teacher as window


class ForcedWindowPrepared(BaseException):
    """Stop before executing a GPU target verification window in this CPU test."""


def cohort(tmp_path, monkeypatch):
    monkeypatch.setattr(multi.pdl, 'ON', multi.pdl.ON)  # Restore the real round's process-local launch flag.
    prepared = plan(4)
    for case in prepared['sequences']:
        case['prompt_ids'] = [1]*256
    decoder = object.__new__(MultiDecoder)
    decoder.admission_policy = 'fifo'
    decoder.w = SimpleNamespace(meta={'execution_provider': 'sm70', 'sm70_profile': 'optimized'}, dflash2=None)
    decoder.streams, decoder.filling, decoder.fills, decoder.passed = {}, [], {}, {}
    decoder.controller = decoder._execution_plan = decoder.media_runtime = None
    decoder.prefill_rows, decoder.share, decoder.round_s, decoder.row_s = 256, .5, None, None
    decoder.pdl, decoder.converged = False, False
    decoder.arrived = lambda: False
    decoder._migrate_boundary = lambda migration: None  # No physical migration in this CPU fixture.
    decoder._make_room = list  # No issued GPU owners exist in this CPU fixture.
    decoder.maintenance_pending = lambda: False
    def unexpected_graph(*args, **kwargs):
        raise AssertionError('GPU graph must not execute before the prepared-window stop')
    decoder.graphs = SimpleNamespace(bound=object.__new__(ExactStateGraphs),
        record=lambda: {'serving_hits': {'verify': 0, 'mtp': 0}, 'serving_misses': {'verify': 0, 'mtp': 0}},
        verify_forward=unexpected_graph, mtp_step=unexpected_graph)
    teacher = object.__new__(window.WindowTeacher)
    teacher.decoder, teacher.engine = decoder, SimpleNamespace(w=decoder.w, rank=0)
    teacher.prepared = {'data': prepared, 'input_sha256': '1'*64}
    teacher.directory = tmp_path
    teacher.streams, teacher.by_state, teacher.prefill_refs = {}, {}, []
    teacher.thread, teacher.failed, teacher.round_index = None, None, 0
    teacher.thread_id = threading.get_ident
    from tools.worker_lifetime import raise_failures
    teacher.raise_failures = raise_failures
    sampled, passes, frozen = [], [], []
    def cold_allocation(stream):
        stream.st = SimpleNamespace(pos=0, history_slot=stream.sid, slot_generation=1,
            mtp_len=0, mtp_drafted=0, kv_identity='cpu-state', image_positions=None)
        stream.cached = 0
        decoder.fills[stream.sid] = [SimpleNamespace(stops=[]), True, 0, [], None]
        decoder.filling.append(stream)
    original_round = MethodType(MultiDecoder.round, decoder)
    def owned_round():
        if teacher.current is not None:
            assert not decoder.filling and len(decoder.streams) == 8
            assert teacher.window_enabled and all(decoder._runnable(s) for s in decoder.streams.values())
            assert teacher.current == prepared['rounds'][0]
            assert all(s.drafts == [7]*(prepared['rounds'][0]['lanes'][s.sid]['rows']-1)
                       for s in decoder.streams.values())
            raise ForcedWindowPrepared
        return original_round()
    teacher.originals = {'admit': cold_allocation, '_runnable': MethodType(MultiDecoder._runnable, decoder),
                         'round': owned_round}
    decoder._runnable = teacher.runnable
    decoder.round = teacher.rounded
    monkeypatch.setattr(decode.Engine, 'sample', lambda slot, logits, positions, sampling, **kwargs: [99])
    def capture(stream, source, position, *, role, row, rows, packed_row, batch_rows):
        assert role == 'prefill' and position == 256 and source.shape == (window.ROW_WIDTH,)
        sampled.append(stream.sid)
        return {'gold_token': 7, 'role': role}
    teacher._capture = capture  # GPU all-gather/raw IO is outside this CPU liveness contract.
    def gpu_prefill_completion():
        pieces = decoder._pieces()  # REAL FIFO order/eligibility/256-row shape.
        assert len(pieces) == 1
        stream, start, count = pieces[0]
        assert (start, count) == (0, 256)
        assert teacher.current is None and not teacher.window_enabled
        passes.append(stream.sid)
        stream.st.pos = 256
        slot = SimpleNamespace(w=decoder.w, st=stream.st)
        tokens = decode.Engine.sample(slot, torch.zeros((1, window.ROW_WIDTH)), [256], None)
        assert tokens == [7]  # Actual teacher hook overrides the deliberately wrong original pick.
        stream.take(tokens, ())
        decoder.filling.remove(stream)
        decoder.fills.pop(stream.sid)
        decoder.streams[stream.sid] = stream
        # Even the LAST prompt cannot decode in this prefill transaction.
        assert not decoder._runnable(stream)
        return []
    decoder._pass = gpu_prefill_completion
    def freeze_boundary():
        class Controller:
            def freeze(self, sessions, *, workload):
                frozen.append((sessions, workload))
                return {'order': sessions}
        decoder.controller = Controller()
        try:
            snapshot = decoder.freeze_execution()  # REAL unchanged all-session projection.
        finally:
            decoder.controller = None
        assert snapshot['order'] == list(range(8))
        assert frozen[-1][1]['active_requests'] == 8
    return teacher, decoder, prepared, sampled, passes, frozen, freeze_boundary


def test_actual_fifo_scheduler_cold_prefill_and_first_window_entry(tmp_path, monkeypatch):
    teacher, decoder, prepared, sampled, passes, frozen, freeze = cohort(tmp_path, monkeypatch)
    for index, case in enumerate(prepared['sequences']):
        stream = Stream(case['prompt_ids'], 4, sid=index, stop_eos=False)
        stream.admission_ticket = index
        teacher.admitted(stream)  # REAL request validation and atomic admission publication.
        if index < 7:
            assert not decoder.media_runnable() and decoder._order() == [] and decoder._pieces() == []
    assert decoder.media_runnable()
    assert [s.sid for s in decoder._order()] == list(range(8))
    assert [(s.sid, start, count) for s, start, count in decoder._pieces()] == [(0, 0, 256)]
    class NoNewArrival:
        def get(self):
            raise AssertionError('runnable cold cohort must not wait for another arrival')
        def get_control(self):
            raise AssertionError('runnable cold cohort must not wait for another control')
    scheduler = object.__new__(Scheduler)
    scheduler.decoder, scheduler.waiting = decoder, NoNewArrival()
    scheduler._retiring, scheduler.held = [], None
    scheduler._stopping = scheduler._media_held_wait = False
    scheduler._serving_profile_available = True
    scheduler._execution_boundary = freeze
    scheduler._yield = lambda: None
    scheduler._admit = lambda first: []
    scheduler._cancelled = list
    scheduler._finish_ready = lambda done, *, finish_empty=True: None
    scheduler.serving_profiles = None
    # REAL legacy scheduler media_wait decision and shared round-entry method;
    # REAL MultiDecoder.round/_round/_fill/_pieces and teacher.rounded hooks.
    with pytest.raises(ForcedWindowPrepared):
        scheduler._run_legacy()
    assert sampled == passes == list(range(8))
    assert len(teacher.prefill_refs) == 8 and len(frozen) == 9
    assert teacher.thread is None and not teacher.window_enabled
    assert all(s.out == [7] for s in decoder.streams.values())
    assert decoder.media_runnable()  # Next boundary can prepare its window again.
    assert all(row[1]['pending_prompt_tokens'] >= 0 for row in frozen)


@pytest.mark.parametrize('fault', ['waiting', 'blocked_input', 'budget', 'finished_schedule'])
def test_private_phase_gate_keeps_original_fifo_restrictions(tmp_path, monkeypatch, fault):
    teacher, decoder, prepared, _, _, _, _ = cohort(tmp_path, monkeypatch)
    for index, case in enumerate(prepared['sequences']):
        stream = Stream(case['prompt_ids'], 4, sid=index, stop_eos=False)
        stream.admission_ticket = index
        teacher.admitted(stream)
    target = decoder.filling[0]
    if fault == 'waiting':
        target.waiting = True
    elif fault == 'blocked_input':
        target.blocked_input = object()
    elif fault == 'budget':
        target.production_budget = 0
    else:
        for stream in list(decoder.filling):
            stream.out = [7]
            decoder.streams[stream.sid] = stream
        decoder.filling.clear()
        decoder.fills.clear()
        teacher.round_index = len(prepared['rounds'])
    assert not decoder._runnable(target)
    if fault != 'finished_schedule':
        assert target not in decoder._order()
