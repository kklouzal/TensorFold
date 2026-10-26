"""Real CPU State/archive snapshots through serial and concurrent prefill policy.

Heavy model kernels are controlled producers of exact raw history and scratch
rows. State validation, metadata, commit and restore remain production code.
These checks establish prefix ordering and extents, not GPU/NUMA qualification.
"""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from tensorfold.cuda.streams import Stream
from tensorfold.families.qwen4_exp.cuda import decode, forward, multi
from tensorfold.families.qwen4_exp.cuda.indexer_history import HistoryPlan, HistorySession
from tensorfold.families.qwen4_exp.cuda.state import State
from tests.test_indexer_history import Allocator, data
from tests.test_sm70_rotorquant_state import model


@contextmanager
def runtime():
    w = model(1)
    w.comm = None
    layers = sum(not layer.linear for layer in w.layers) + 1
    plan = HistoryPlan(layers, w.cfg.index_dim, w.cfg.index_ratio, 16, 4)
    session = HistorySession(plan, Allocator(), 'cpu', maximum_owners=4)
    w.meta.update(execution_provider='sm70', indexer_history='ram', indexer_history_session=session)
    try:
        yield w
    finally:
        session.close()


def produce(state, start, rows, *, mtp=False):
    """Real submitted/validated archive frames; no model or CUDA operations."""
    owner = state.indexer_history_owner
    layers = (len(state.kc),) if mtp else tuple(range(len(state.kc)))
    frame = owner.begin('prefill', start, rows, layers=layers)
    state.kv_begin()
    for layer in layers:
        offset = start - int(owner.origins_host[layer])
        frame.stage(layer)[offset:offset + rows].copy_(data(start, rows, layer, owner.plan.index_dim))
    frame.enqueue()
    frame.mark_submitted()
    state.kv_check()


def seed(state, rows, *, mtp):
    for start in range(0, rows, 16):
        count = min(16, rows - start)
        produce(state, start, count)
        state.set_pos(start + count)
        if mtp:
            produce(state, start, count, mtp=True)
            state.set_mtp_len(start + count)


def track_metadata(state, events, sid=0):
    actual = state.indexer_snapshot_metadata

    def metadata(pos, mtp_len):
        events.append(('metadata', sid, pos, mtp_len, state.pos))
        return actual(pos, mtp_len)

    state.indexer_snapshot_metadata = metadata


def assert_snapshot(state, snapshot, pos, mtp_len):
    assert (snapshot['pos'], snapshot['mtp_len']) == (pos, mtp_len)
    expected = (pos,) * len(state.kc) + (mtp_len,)
    assert snapshot['indexer_history']['lengths'] == expected
    state.kv_validate_snapshot(snapshot)
    state.restore(snapshot)
    assert (state.pos, state.mtp_len) == (pos, mtp_len)
    for layer, length in enumerate(expected):
        raw = state.indexer_history_owner.archive[layer, :length]
        assert torch.equal(raw, data(0, length, layer, state.index_dim))


@pytest.mark.parametrize('use_mtp', [False, True])
@pytest.mark.parametrize('start,rows,prompt_rows,point', [(0, 6, 8, 3), (5, 4, 12, 7)])
def test_serial_cut_waits_for_mtp_archive_before_metadata_and_main_commit(
        monkeypatch, use_mtp, start, rows, prompt_rows, point):
    with runtime() as w:
        state = State(w, 64, 4, 'fp16', limit=512)
        seed(state, start, mtp=use_mtp)
        events = []
        track_metadata(state, events)
        pb = SimpleNamespace(prefill=True, h=torch.empty(rows, 1, dtype=torch.float16),
                             streams=torch.arange(start, start + rows, dtype=torch.float16).reshape(rows, 1))
        original_streams = pb.streams.clone()
        engine = SimpleNamespace(w=w, st=state, pbuf=pb, mbuf=object() if use_mtp else None,
                                 prefill_rows=rows, kept=None)

        def target(weight, st, buffers, tokens, *, logits, cut):
            produce(st, start, len(tokens))
            cut.rec = torch.full_like(st.rec[0], 17)
            cut.conv = torch.full_like(st.conv, 19)
            events.append(('target_validated',))
            return torch.full((rows, 2), 7.)

        def capture(weight, st, buffers, cut, mtp_len):
            events.append(('cut',))
            return forward.cut_snapshot(weight, st, buffers, cut, mtp_len)

        def mtp(weight, st, buffers, tokens, streams, *, logits):
            assert ('cut',) in events
            assert torch.equal(streams, original_streams[:len(tokens)])
            produce(st, st.mtp_len, len(tokens), mtp=True)
            buffers.streams.fill_(-999)
            events.append(('mtp_validated',))

        def commit(*args, **kwargs):
            events.append(('commit',))
            return forward.commit(*args, **kwargs)

        monkeypatch.setattr(decode, 'forward', target)
        monkeypatch.setattr(decode, 'cut_snapshot', capture)
        monkeypatch.setattr(decode, 'mtp_forward', mtp)
        monkeypatch.setattr(decode, 'commit', commit)
        result = decode.prefill_chunk(engine, list(range(prompt_rows)), start, keep_at=point, mtp=use_mtp)
        assert result is None
        kept = engine.kept
        expected_mtp = start + point - start - 1 if use_mtp else 0
        wanted = ('metadata', 0, point, expected_mtp, start)
        assert wanted in events and events.index(wanted) < events.index(('commit',))
        assert events.index(('cut',)) < events.index(wanted)
        if use_mtp:
            assert events.index(('cut',)) < events.index(('mtp_validated',)) < events.index(wanted)
            assert torch.equal(kept['tail'], original_streams[point - start - 1:point - start])
        else:
            assert ('mtp_validated',) not in events and kept['tail'] is None
        assert torch.equal(engine.last_streams, original_streams[-1:])
        assert state.pos == start + rows
        assert_snapshot(state, kept['state'], point, expected_mtp)
        assert torch.equal(state.rec[0], torch.full_like(state.rec[0], 17))
        assert torch.equal(state.conv, torch.full_like(state.conv, 19))


@pytest.mark.parametrize('final', [False, True])
def test_serial_end_piece_keeps_one_fewer_mtp_row_with_coherent_metadata(monkeypatch, final):
    with runtime() as w:
        rows = 6
        state = State(w, 64, 4, 'fp16', limit=512)
        pb = SimpleNamespace(prefill=True, h=torch.empty(rows, 1, dtype=torch.float16),
                             streams=torch.arange(rows, dtype=torch.float16).reshape(rows, 1),
                             logits=torch.full((rows, 2), 7.))
        before = pb.streams.clone()
        engine = SimpleNamespace(w=w, st=state, pbuf=pb, mbuf=object(), prefill_rows=rows, kept=None)

        def target(weight, st, buffers, tokens, *, logits, cut):
            assert cut is None
            produce(st, 0, len(tokens))
            return buffers.logits

        def mtp(weight, st, buffers, tokens, streams, *, logits):
            produce(st, 0, len(tokens), mtp=True)
            buffers.streams.fill_(-999)
            buffers.logits.fill_(-999)

        monkeypatch.setattr(decode, 'forward', target)
        monkeypatch.setattr(decode, 'mtp_forward', mtp)
        prompt = list(range(rows if final else rows + 2))
        result = decode.prefill_chunk(engine, prompt, 0, keep_at=rows)
        assert state.mtp_len == rows - int(final)
        assert_snapshot(state, engine.kept['state'], rows, rows - 1)
        assert torch.equal(engine.kept['tail'], before[-1:])
        if final:
            assert torch.equal(result, torch.full((rows, 2), 7.))
        else:
            assert result is None


@pytest.mark.parametrize('use_mtp', [False, True])
def test_multi_absorb_captures_cuts_before_scratch_reuse_and_metadata_after_validation(monkeypatch, use_mtp):
    with runtime() as w:
        decoder = object.__new__(multi.MultiDecoder)
        decoder.w = w
        decoder.controller = None
        states = [State(w, 64, 4, 'fp16', limit=512) for _ in range(2)]
        streams = [Stream(list(range(length)), 4, sid=index, st=state)
                   for index, (state, length) in enumerate(zip(states, (12, 10)))]
        pieces = [(streams[0], 0, 6), (streams[1], 0, 4)]
        segs = [(states[0], 0, 6), (states[1], 6, 10)]
        decoder.fills = {index: [SimpleNamespace(stops=points), use_mtp, 0, [], None]
                         for index, points in enumerate(([2, 6], [3, 4]))}
        pb = decoder.pbuf = SimpleNamespace(prefill=True, h=torch.empty(10, 1, dtype=torch.float16),
                                             streams=torch.arange(10, dtype=torch.float16).reshape(10, 1))
        before, events = pb.streams.clone(), []
        for index, (state, count) in enumerate(zip(states, (6, 4))):
            produce(state, 0, count)
            track_metadata(state, events, index)
        cuts = decoder._cuts(pieces, segs)
        for cut in cuts:
            cut.rec = torch.full_like(states[0].rec[0], cut.row)
            cut.conv = torch.full_like(states[0].conv, cut.row + 10)

        def capture(weight, st, buffers, cut, mtp_len):
            events.append(('cut', states.index(st)))
            return forward.cut_snapshot(weight, st, buffers, cut, mtp_len)

        def mtp_stage(buffers, absorb):
            assert all(('cut', sid) in events for sid in (0, 1))
            return absorb

        def mtp_compute(weight, absorb, buffers, *, logits):
            for st, tokens, source in absorb:
                sid = states.index(st)
                offset = segs[sid][1]
                assert torch.equal(source, before[offset:offset + len(tokens)])
                produce(st, st.mtp_len, len(tokens), mtp=True)
                events.append(('mtp_validated', sid))
            buffers.streams.fill_(-999)

        def commit(weight, st, buffers, *args, **kwargs):
            events.append(('commit', states.index(st)))
            return forward.commit(weight, st, buffers, *args, **kwargs)

        monkeypatch.setattr(multi, 'cut_snapshot', capture)
        monkeypatch.setattr(multi, 'mtp_stage', mtp_stage)
        monkeypatch.setattr(multi, 'mtp_compute', mtp_compute)
        monkeypatch.setattr(multi, 'commit', commit)
        lasts = decoder._absorb(pieces, segs, cuts)
        assert torch.equal(lasts[0], before[5:6]) and torch.equal(lasts[1], before[9:10])
        first_commit = events.index(('commit', 0))
        for sid, (strict, end) in enumerate(((2, 6), (3, 4))):
            point = ('metadata', sid, strict, strict - 1 if use_mtp else 0, 0)
            assert events.index(('cut', sid)) < events.index(point) < first_commit
            if use_mtp:
                assert events.index(('mtp_validated', sid)) < events.index(point)
            snapshots = decoder.fills[sid][3]
            assert len(snapshots) == 2
            for (snapshot, tail), position in zip(snapshots, (strict, end)):
                assert_snapshot(states[sid], snapshot, position, position - 1 if use_mtp else 0)
                if use_mtp:
                    at = segs[sid][1] + position
                    assert torch.equal(tail, before[at - 1:at])
                else:
                    assert tail is None


@pytest.mark.parametrize('use_mtp', [False, True])
def test_cancelled_prefill_snapshot_has_the_exact_shorter_mtp_archive_extent(use_mtp):
    with runtime() as w:
        state = State(w, 512, 4, 'fp16', limit=512)
        seed(state, multi.MIN_GAP, mtp=use_mtp)
        stream = Stream([1] * (multi.MIN_GAP + 8), 4, st=state)
        decoder = object.__new__(multi.MultiDecoder)
        decoder.filling, decoder.passed = [stream], {stream.sid: 3}
        tail, remembered, dropped = torch.tensor([[31.]], dtype=torch.float16), [], []
        decoder.fills = {stream.sid: [None, use_mtp, multi.MIN_GAP, [], tail]}
        decoder._remember = lambda *args: remembered.append(args)
        decoder._drop_kept = lambda st: dropped.append(st)
        assert decoder.cancel([stream]) == [stream]
        assert stream.done and stream.error is None and not dropped
        assert not decoder.filling and not decoder.passed and not decoder.fills
        assert len(remembered) == 1
        prompt, kept_state, snapshot, kept_tail = remembered[0]
        assert len(prompt) == multi.MIN_GAP and kept_state is state and kept_tail is tail
        assert_snapshot(state, snapshot, multi.MIN_GAP, multi.MIN_GAP - 1 if use_mtp else 0)
