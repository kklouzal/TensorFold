"""Exercise the opt-in DeepSeek DSpark serving protocol without native GPU work."""

from threading import Event
from types import SimpleNamespace

import pytest
import torch

from tensorfold.cuda.streams import Stream
from tensorfold.families import deepseek_v4
from tensorfold.families.deepseek_v4.cuda import engine, round_sampling
from tensorfold.families.deepseek_v4.cuda.dspark_serving import DraftFlight, DraftOwner, decide


def test_cuda_family_draft_request_is_explicit(monkeypatch, tmp_path):
    from tensorfold import families, hub
    from tensorfold.cli import _drafter

    calls = []
    monkeypatch.setattr(families, "read_config", lambda _: {"model_type": "deepseek_v41"})
    monkeypatch.setattr(engine, "DS41Engine", lambda *a, **kw: calls.append(kw))
    monkeypatch.setattr(hub, "cached", lambda repo: pytest.fail(f"unexpected CUDA auto lookup {repo}"))
    family = SimpleNamespace(package=deepseek_v4, model_type="deepseek_v41")
    assert _drafter(family, "auto", "cuda", model_dir=tmp_path) == ""
    deepseek_v4.cuda_engine(tmp_path, drafter=str(tmp_path))
    assert calls[-1]["draft_root"] is calls[-1]["draft_depth"] is None
    deepseek_v4.cuda_engine(tmp_path, drafter=str(tmp_path), draft_method="dspark", mtp_drafts=3)
    assert calls[-1]["draft_root"] == tmp_path and calls[-1]["draft_depth"] == 3
    deepseek_v4.cuda_engine(tmp_path, draft_method="dspark", no_drafts=True)
    assert calls[-1]["draft_root"] is calls[-1]["draft_depth"] is None
    with pytest.raises(ValueError, match="requires --drafter"):
        deepseek_v4.cuda_engine(tmp_path, draft_method="dspark")


def test_seeded_prompt_frame_and_paused_window_publish_once():
    e = object.__new__(engine.DS41Engine)
    events = []
    e.announce = lambda *args: e._run(*args)
    e._step = lambda entries, ids, rows, flags, tap: events.append((entries, ids, rows, flags, tap))
    e.step([(2, 128, 4)], [101, 102, 103, 104], [3], 17, tap_from=1)
    assert events == [([(2, 128, 4)], [101, 102, 103, 104], [3], 17, 1)]
    e.units = 80
    e.drafter = SimpleNamespace(head=SimpleNamespace(insert_context=lambda *args: events.append(args)))
    e.target = SimpleNamespace(layers_with_draft_taps=lambda *args: events.append((args[1], args[2].tolist())))
    rows, taps = torch.arange(1, 4), torch.empty(3, 5)
    state = {"step": SimpleNamespace(positions=torch.arange(128, 132)), "draft_context": (2, rows, taps)}
    e._advance_layers(state, 77, None)
    e._advance_layers(state, 80, None)
    assert events[1:3] == [(77, [1, 2, 3]), (80, [1, 2, 3])]
    slots, values, positions = events[3]
    assert slots.tolist() == [2, 2, 2] and positions.tolist() == [129, 130, 131] and values is taps


def test_draft_owner_enforces_exactly_one_commit_and_discard():
    owner = object.__new__(DraftOwner)
    owner.depth, owner.pending = 3, None
    owner._slot = torch.empty(1, dtype=torch.int32)
    owner._count = torch.empty(1, dtype=torch.int64)
    actions = []
    owner.shape = SimpleNamespace(run=lambda entries: actions.append(entries[0]))
    owner.spec = SimpleNamespace(commit=lambda slot, count: actions.append((int(slot[0]), int(count[0]))))
    owner.target = SimpleNamespace(hash_state=SimpleNamespace(err=torch.zeros(1)), c2={})
    owner.verify(4, 127, 9)
    assert owner.pending == 4
    with pytest.raises(RuntimeError, match="unresolved"):
        owner.verify(5, 255, 8)
    with pytest.raises(RuntimeError, match="invalid owner"):
        owner.commit(4, 5)
    owner.commit(4, 2)
    owner.verify(5, 255, 8)
    owner.discard(4)
    owner.discard(5)
    owner.discard(5)
    assert actions[1::2] == [(4, 2), (5, 0)] and owner.pending is None


@pytest.mark.parametrize(("eos", "count"), (((12,), 5), ((), 3)))
def test_only_consumed_scores_choose_and_settle_before_stream_delivery(monkeypatch, eos, count):
    e = SimpleNamespace(drafter=SimpleNamespace(depth=3, pending=4))
    actions = []
    e.announce = lambda op, slot, count: actions.append((op, slot, count))
    d = object.__new__(engine.Decoder)
    d.e, d.eos = e, eos
    stream = Stream([99], count, sid=2, out=[10], cancel=Event())
    d.streams = {2: stream}
    monkeypatch.setattr(round_sampling, "greedy_tokens", lambda _, n: [
        (11, None), (12, None), (None, ValueError("unconsumed")), (None, ValueError("unconsumed")),
    ])
    flight = DraftFlight(stream, 4, 10, SimpleNamespace(compact=torch.empty(1), drafts=torch.tensor([[11, 12, 13]])),
                         SimpleNamespace(synchronize=lambda: actions.append("wait")))
    assert d._collect_draft(flight) == [(stream, [11, 12])]
    assert actions == ["wait", (engine.OP_DRAFT_COMMIT, 4, 2)]
    assert stream.out == [10] and stream.rounds == 1 and stream.drafted == 3
    stream.take([11, 12], eos)
    assert stream.done and stream.accepted == 2
    with pytest.raises(ValueError, match="consumed"):
        decide([(11, None), (None, ValueError("consumed"))], 10, [11], 2, ())


def test_cancelled_flight_does_not_observe_or_count_proposals():
    e = SimpleNamespace(drafter=SimpleNamespace(depth=3, pending=4))
    actions = []
    e.announce = lambda op, slot, count: actions.append((op, slot, count))
    d = object.__new__(engine.Decoder)
    d.e, d.eos = e, ()
    stream = Stream([99], 5, sid=2, out=[10], cancel=Event())
    d.streams = {2: stream}
    stream.cancel.set()
    flight = DraftFlight(stream, 4, 10, None, SimpleNamespace(synchronize=lambda: actions.append("wait")))
    assert d._collect_draft(flight) == [(stream, None)]
    assert actions == ["wait", (engine.OP_DRAFT_COMMIT, 4, 0)]
    assert stream.done and stream.rounds == stream.drafted == stream.accepted == 0


def test_http_stop_prefix_limiter_keeps_one_row_and_the_delivery_callback():
    from inspect import signature
    from tensorfold.cuda.reply_text import StopStrings, stop_prefix_limit

    class Characters:
        def decode(self, ids, **kwargs):
            return "".join(map(chr, ids))

    stops = StopStrings(("AB",), Characters())
    limiter = lambda new, prefix: stop_prefix_limit(stops, (), prefix, new)
    e = object.__new__(engine.DS41Engine)
    e.rank, e.vocab, e.context, e._closed = 0, 500, 256, False
    e.drafter, e.budget, e.slot_bytes = object(), 1000, lambda rows: rows
    forwarded = []
    e.scheduler = SimpleNamespace(submit=lambda *args, **kw: forwarded.append(kw))
    assert "token_limit" in signature(e.generate).parameters  # CUDA App's stop-policy boundary
    e.generate([90, 91], 5, None, lambda _: None, token_limit=limiter)
    assert forwarded[0]["token_limit"] is limiter
    d = object.__new__(engine.Decoder)
    d.e, d.filling, d.next_id, d._broken = e, [], 0, None
    d._cap = lambda rows: rows
    d.place = lambda *args: None
    emitted = []
    stream = Stream([90, 91], 5, token_limit=limiter, emit=emitted.append, out=[65])
    stream.head, stream.pos = None, 0
    d.admit(stream)
    assert not stream.draft_seed
    selected = d._limit_token(stream, 66)
    assert selected == [66] and stream.done and stream.out == [65]
    stream.take(selected)
    assert emitted == [[66]] and stream.out == [65, 66]


def test_ordinary_mixed_fallback_drops_draft_window_eligibility():
    streams = [Stream([90], 4, sid=i, out=[10]) for i in range(2)]
    for stream in streams:
        stream.slot, stream.draft_seed, stream.forced = stream.sid, True, None
    e = SimpleNamespace(drafter=object(), graphs=SimpleNamespace(sizes=(2,)))
    ops = []
    e.step = lambda *args: (ops.append(args) or object(), None)
    d = object.__new__(engine.Decoder)
    d.e, d.filling = e, []
    event = SimpleNamespace(record=lambda: None)
    d._events = lambda: (event, event)
    d._launch(streams, {})
    assert ops == [([(0, 1, 1), (1, 1, 1)], [10, 10], [0, 1], engine.F_COMPACT)]
    assert not any(stream.draft_seed for stream in streams)
    assert not isinstance(d.inflight, DraftFlight)


def test_c1_launch_and_retirement_settle_the_owned_slot_first():
    e = SimpleNamespace(drafter=SimpleNamespace(depth=5, pending=None), context=256,
                        target=SimpleNamespace(pool=SimpleNamespace(owned={4: [0, 1]})))
    events = []

    def announce(op, slot, *args):
        events.append((op, slot, *args))
        e.drafter.pending = slot if op == engine.OP_DRAFT_VERIFY else None

    e.announce = announce
    d = object.__new__(engine.Decoder)
    d.e, d.filling, d._broken = e, [], None
    fake_event = SimpleNamespace(record=lambda: None)
    d._events = lambda: (fake_event, fake_event)
    request = Stream([90] * 80, 32, sid=2, out=[42], st=object())
    request.slot, request.draft_seed = 4, True
    d.streams = {request.sid: request}
    d._launch([request], {})
    assert isinstance(d.inflight, DraftFlight) and e.drafter.pending == 4
    d.retire_slot = lambda s: events.append(("retire", s.slot, e.drafter.pending))
    d.finish([request])
    assert events == [(engine.OP_DRAFT_VERIFY, 4, 80, 42), (engine.OP_DRAFT_COMMIT, 4, 0),
                      ("retire", None, None)]


def test_commit_publication_waits_for_local_fence_and_all_reply_frames(monkeypatch):
    import os
    from tensorfold.cuda.control_pipes import ControlPipes, PeerDeparted

    events = []
    frame = engine._HEAD.pack(engine.OP_DRAFT_COMMIT, 3, 2, 0, 0)
    monkeypatch.setattr(torch.cuda, "current_stream",
                        lambda: SimpleNamespace(synchronize=lambda: events.append("device-fence")))
    e = object.__new__(engine.DS41Engine)
    e.rank, e.world = 0, 8
    e.drafter = SimpleNamespace(commit=lambda slot, count: events.append(("commit", slot, count)))

    def replies():
        events.append("join")
        return [frame] * 7

    e.pipes = SimpleNamespace(send=lambda data: events.append(("send", data)), acknowledgments=replies)
    e.announce(engine.OP_DRAFT_COMMIT, 3, 2)
    assert events == [("send", frame), ("commit", 3, 2), "device-fence", "join"]
    e.pipes.acknowledgments = lambda: [frame] * 6
    with pytest.raises(RuntimeError, match="invalid DeepSeek DSpark commit"):
        e.announce(engine.OP_DRAFT_COMMIT, 3, 2)

    read_fd, write_fd = os.pipe()
    os.set_blocking(read_fd, False)
    os.close(write_fd)
    pipes = object.__new__(ControlPipes)
    pipes.peers = frozenset({1})
    pipes.readers = {1: read_fd}
    pipes.broken, pipes.timeout_s, pipes.max_ack = False, 0.1, len(frame)
    pipes.send = lambda data: None     # the departed follower already received its command
    e.pipes = pipes
    try:
        with pytest.raises(PeerDeparted):
            e.announce(engine.OP_DRAFT_COMMIT, 3, 2)
        assert pipes.broken
    finally:
        os.close(read_fd)


@pytest.mark.parametrize("fail", (False, True))
def test_follower_only_acknowledges_after_its_device_settles(monkeypatch, fail):
    from tensorfold.cuda import cohort

    events = []
    frames = iter((engine._HEAD.pack(engine.OP_DRAFT_COMMIT, 3, 2, 0, 0), None))
    e = object.__new__(engine.DS41Engine)
    e.rank = 1
    e.target = SimpleNamespace(rotor_cache=object())
    e.drafter = SimpleNamespace(commit=lambda slot, count: events.append(("commit", slot, count)))
    e.pipes = SimpleNamespace(receive=lambda *_: next(frames), acknowledge=lambda data: events.append(("ack", data)))

    def fence():
        events.append("device-fence")
        if fail:
            raise RuntimeError("simulated follower device error")

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(synchronize=fence))
    monkeypatch.setattr(cohort, "notify_failure", lambda: events.append("cohort-failure"))
    if fail:
        with pytest.raises(RuntimeError, match="follower device error"):
            e.follow()
        assert events == [("commit", 3, 2), "device-fence", "cohort-failure"]
    else:
        e.follow()
        assert events == [("commit", 3, 2), "device-fence",
                          ("ack", engine._HEAD.pack(engine.OP_DRAFT_COMMIT, 3, 2, 0, 0))]


def test_follower_close_discards_pending_before_releasing_graphs(monkeypatch):
    events = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("fence"))
    e = object.__new__(engine.DS41Engine)
    e.rank, e._closed, e.graphs, e.engram = 1, False, object(), None
    e.drafter = SimpleNamespace(pending=4, discard=lambda slot: events.append(("discard", slot)))
    e.close()
    e.close()
    assert events == [("discard", 4), "fence"]
    assert e.graphs is e.drafter is None
