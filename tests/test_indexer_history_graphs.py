"""Archive graph submission boundaries use live owners, logical rows and external completion."""
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.cuda.sm70 import attention_qualification
from tensorfold.families.qwen4_exp.cuda import bound_graphs, graphs, history_runtime, multi_graphs


class Replay:
    def __init__(self, events, label="replay", error=None):
        self.events, self.label, self.error = events, label, error

    def replay(self):
        self.events.append(self.label)
        if self.error is not None:
            raise self.error


def history_spies(monkeypatch, events):
    def begin(segs, **kwargs):
        events.append(("begin", segs, kwargs))

    def submitted(segs, **kwargs):
        events.append(("submitted", segs, kwargs))

    monkeypatch.setattr(history_runtime, "begin", begin)
    monkeypatch.setattr(history_runtime, "submitted", submitted)


def event_names(events):
    return [event[0] if isinstance(event, tuple) else event for event in events]


@pytest.mark.parametrize("backend", [None, "gpu", "ram"])
@pytest.mark.parametrize("mtp", [False, True])
def test_serial_graph_replay_registers_and_completes_before_numeric_wait(monkeypatch, backend, mtp):
    events = []
    history_spies(monkeypatch, events)
    state = NS(pos=128, mtp_len=128, capacity=8192, cur=[], kv_begin=lambda: events.append("kv_begin"))
    b = NS(prefill=False, logits=torch.zeros((3, 2)))
    w = NS(comm=None, **({"meta": {"indexer_history": backend}} if backend is not None else {}))
    e = NS(w=w, st=state, buf=b, mbuf=b)
    owner = object.__new__(graphs.Graphs)
    owner.e, owner.max_rows = e, 8
    owner._check_owner = lambda: None
    owner._bucket = lambda end: 256
    graph = Replay(events)
    owner.main, owner.mtp = {(3, 0, 256): graph}, {(3, 256): graph}
    owner.mtp_out = {(3, 256): b.logits}
    segs = [(state, 0, 3)]
    monkeypatch.setattr(attention_qualification, "allows", lambda *args, **kwargs: True)
    monkeypatch.setattr(graphs, "stage", lambda *args, **kwargs: segs)
    monkeypatch.setattr(graphs, "mtp_stage", lambda *args, **kwargs: segs)
    monkeypatch.setattr(graphs, "_kv_check", lambda *args, **kwargs: events.append("check"))
    result = owner.mtp_forward([1, 2, 3], torch.zeros((3, 2))) if mtp else owner.forward([1, 2, 3])
    assert result.shape == (3, 2)
    assert event_names(events) == (["begin", "kv_begin", "replay", "submitted", "check"] if backend == "ram"
                                   else ["kv_begin", "replay", "check"])
    if backend == "ram":
        assert events[0][1] is segs and events[3][1] is segs
        assert events[0][2] == events[3][2] == {"prefill": False, **({"mtp": True} if mtp else {})}


@pytest.mark.parametrize("mtp", [False, True])
def test_serial_warm_and_capture_do_not_publish_a_fake_archive_frame(monkeypatch, mtp):
    events, capturing = [], [False]
    history_spies(monkeypatch, events)
    state = NS(pos=128, mtp_len=128, capacity=8192, cur=[], kv_begin=lambda: events.append("kv_begin"))
    b = NS(prefill=False, logits=torch.zeros((3, 2)))
    owner = object.__new__(graphs.Graphs)
    owner.e, owner.max_rows = NS(w=NS(meta={"indexer_history": "ram"}, comm=None), st=state, buf=b, mbuf=b), 8
    owner._check_owner, owner._bucket = lambda: None, lambda end: 256
    owner.main, owner.mtp, owner.mtp_out = {}, {}, {}
    segs = [(state, 0, 3)]
    monkeypatch.setattr(attention_qualification, "allows", lambda *args, **kwargs: True)
    monkeypatch.setattr(graphs, "stage", lambda *args, **kwargs: segs)
    monkeypatch.setattr(graphs, "mtp_stage", lambda *args, **kwargs: segs)
    monkeypatch.setattr(graphs, "_kv_check", lambda *args, **kwargs: events.append("check"))

    def compute(*args, **kwargs):
        # Root's ordinary eager helper contract: begin/submit outside capture,
        # while its copy node is recorded during capture, without publication.
        events.append("copy_recorded" if capturing[0] else "eager_begin_copy_submit")
        return b.logits

    def capture(fn):
        capturing[0] = True
        try:
            fn()
        finally:
            capturing[0] = False
        return Replay(events)

    owner._capture = capture
    monkeypatch.setattr(graphs, "compute", compute)
    monkeypatch.setattr(graphs, "mtp_compute", compute)
    if mtp:
        owner.mtp_forward([1, 2, 3], torch.zeros((3, 2)))
    else:
        owner.forward([1, 2, 3])
    assert event_names(events) == ["eager_begin_copy_submit", "copy_recorded", "begin", "kv_begin", "replay",
                                   "submitted", "check"]


@pytest.mark.parametrize("backend", [None, "gpu", "ram"])
@pytest.mark.parametrize("mtp", [False, True])
def test_round_replay_marks_completion_after_all_pieces_and_inserts(monkeypatch, backend, mtp):
    events = []
    history_spies(monkeypatch, events)
    owner = object.__new__(multi_graphs.RoundGraphs)
    owner.dec = NS(w=NS(**({"meta": {"indexer_history": backend}} if backend is not None else {})))
    b = NS(prefill=False, attn_step=object(), ple_late=None)
    segs = [(object(), 0, 1)]
    pieces = NS(graphs=[Replay(events, "piece0"), Replay(events, "piece1")], inserts=[("select", object())])
    monkeypatch.setattr(multi_graphs, "_kv_begin", lambda *args: events.append("kv_begin"))
    monkeypatch.setattr(multi_graphs.attn_multi, "select", lambda *args: events.append("select"))
    owner._replay(pieces, segs, b, mtp=mtp)
    expected = ["kv_begin", "piece0", "select", "piece1"]
    assert event_names(events) == (["begin", *expected, "submitted"] if backend == "ram" else expected)
    if backend == "ram":
        assert events[0][2] == events[-1][2] == {"prefill": False, "mtp": mtp}


def test_failed_graph_replay_does_not_record_successful_submission(monkeypatch):
    events = []
    history_spies(monkeypatch, events)
    owner = object.__new__(multi_graphs.RoundGraphs)
    owner.dec = NS(w=NS(meta={"indexer_history": "ram"}))
    pieces = NS(graphs=[Replay(events, error=RuntimeError("replay failed"))], inserts=[])
    monkeypatch.setattr(multi_graphs, "_kv_begin", lambda *args: events.append("kv_begin"))
    with pytest.raises(RuntimeError, match="replay failed"):
        owner._replay(pieces, [(object(), 0, 1)], NS(prefill=False))
    assert event_names(events) == ["begin", "kv_begin", "replay"]


def buffer():
    return NS(rotor_lookup=False, prefill=False, streams=torch.zeros((4, 2)), proj=torch.zeros((1, 4, 2)),
              ple_nrow=torch.zeros((4, 2)), logits=torch.zeros((4, 2)))


@pytest.mark.parametrize("kind", ["verify", "mtp"])
@pytest.mark.parametrize("backend", [None, "ram"])
def test_bound_replay_uses_shared_owner_view_logical_counts_and_provisional_mtp_position(monkeypatch, kind, backend):
    events = []
    history_spies(monkeypatch, events)
    source, bound_buffer = buffer(), buffer()
    state = NS(pos=17, mtp_len=19, capacity=128, image_positions=None)
    view = NS(pos=17, mtp_len=31, history_rows=1)
    segs, bound = [(state, 0, 1)], [(view, 0, 4)]
    lane = NS(publish=lambda: events.append("publish"))
    pieces = NS(graphs=[Replay(events)], out=torch.zeros((1, 2)))
    pool = NS(room_bytes=0, verify={"key": pieces}, mtp={"key": pieces})
    owner = object.__new__(bound_graphs.BoundStateGraphs)
    owner.closed, owner.owner, owner.hits = False, pool, 0
    owner.dec = NS(w=NS(comm=None, **({"meta": {"indexer_history": backend}} if backend is not None else {})),
                   buf=source, mbuf=source)
    owner._width = lambda *args: 4
    owner._prepare = lambda *args, **kwargs: ("key", [lane], bound_buffer, bound, None, (1,), torch.tensor([0]),
                                             torch.tensor([0]))
    monkeypatch.setattr(bound_graphs, "_kv_begin", lambda *args: events.append("kv_begin"))
    owner._forward(kind, segs, mtp_positions=[31] if kind == "mtp" else None)
    if backend == "ram":
        assert event_names(events)[:4] == ["begin", "kv_begin", "replay", "submitted"]
        assert events[0][1] is bound and events[3][1] is bound
        assert events[0][2] == {"prefill": False, "mtp": kind == "mtp", "counts": (1,)}
        assert events[3][2] == {"prefill": False, "mtp": kind == "mtp"}
        assert events[0][1][0][0].mtp_len == 31
        if kind == "verify":
            assert event_names(events)[4] == "publish"
    else:
        assert event_names(events) == ["kv_begin", "replay", *(["publish"] if kind == "verify" else [])]


def test_bound_lane_host_history_rows_track_the_same_device_mask_without_readback():
    state = NS(kv_identity="stored-format", version=0, cur=[0], rec=torch.tensor([[2.], [3.]]), mtp_len=19)
    lane = object.__new__(bound_graphs.BoundLane)
    lane.state, lane.identity = state, (state.kv_identity, state.version)
    lane.view, lane.recurrent = NS(), torch.zeros_like(state.rec)
    lane.mtp_position, lane.keep_rows, lane.history_rows, lane.scratch = None, torch.ones(1, dtype=torch.int32), 1, []
    for rows in (1, 3, 2):
        view = lane.refresh(rows)
        assert view.history_rows == rows and view.graph_rows.item() == rows
        assert lane.refresh().history_rows == rows
