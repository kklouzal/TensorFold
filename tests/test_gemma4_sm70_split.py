"""Gemma 4 SM70 Decoder: a prompt chunk runs in layer groups while streams decode (engine.Decoder._plan/_fill),
against a recording fake engine. No GPU."""

from types import SimpleNamespace

from tensorfold.families.gemma4.sm70 import engine as E
from tensorfold.families.gemma4.sm70.ops import OP_FORWARD, OP_LAYERS

L = 30


class FakeEngine:
    prompt_rows, layer_count, head = 1024, L, None

    def __init__(self):
        self.ops, self.layer_marks = [], None

    def announce(self, op, a=0, b=0, c=0, d=0, ids=()):
        self.ops.append((op, a, b, c, d))
        if op == OP_FORWARD:
            stop = d >> E.STOP_SHIFT
            return None if 0 < stop < L else "logits"
        if op == OP_LAYERS:
            return "logits" if a == L else None


def decoder(layer_ms, round_ms=20.0):
    d = object.__new__(E.Decoder)
    d.e, d.streams, d.filling, d.split, d.measuring = FakeEngine(), {}, [], None, None
    d.layer_ms, d.round_ms, d.eos = layer_ms, round_ms, ()
    d.keep_before = lambda s, start: None
    return d


def request(n):
    return SimpleNamespace(prompt=list(range(n)), pos=0, head=None, background=False, probabilities=None,
                           slot=3, prefill_s=0.0, keep_end=False, on_fill=None, forced=None)


def test_plan_without_decoding_streams_is_one_forward():
    d = decoder([0.01] * L)                          # 1,024 rows: 307 ms a chunk
    assert d._plan(1024) == [L]


def test_plan_groups_keep_each_group_under_the_gap_target():
    cost = [0.008] * L
    for i in (5, 11, 17, 23, 29):                    # full-attention layers cost more at long positions
        cost[i] = 0.03
    d = decoder(cost)
    d.streams = {0: object()}
    stops = d._plan(1024)
    assert stops[-1] == L and stops == sorted(set(stops))
    edges = [0] + stops
    groups = [sum(cost[a:b]) * 1024 for a, b in zip(edges, edges[1:])]
    assert len(groups) == 3 and max(groups) <= (E.Decoder.GAP_MS - d.round_ms) * 1.25


def test_split_chunk_continues_before_any_other_prompt_and_ends_with_the_whole_chunk():
    d = decoder([0.01] * L)
    d.streams = {0: object()}
    a, b = request(2048), request(500)
    d.filling = [a, b]
    assert d._fill() == []                            # the first group of a's first chunk
    op, slot, start, rows, flags = d.e.ops[-1]
    assert (op, slot, start, rows) == (OP_FORWARD, 3, 0, 1024) and 0 < flags >> E.STOP_SHIFT < L
    while d.split is not None:
        d._fill()
        assert d.e.ops[-1][0] == OP_LAYERS
    assert d.e.ops[-1][1] == L and a.pos == 1024 and d.filling == [b, a]


def test_chunk_finishes_at_once_when_no_stream_decodes_any_more():
    d = decoder([0.01] * L)
    d.streams = {0: object()}
    s = request(2048)
    d.filling = [s]
    d._fill()
    d.streams = {}
    d._fill()
    assert d.e.ops[-1][:2] == (OP_LAYERS, L) and d.split is None and s.pos == 1024


def test_cancel_drops_the_paused_chunk_on_every_rank():
    d = decoder([0.01] * L)
    d.streams = {0: object()}
    s = request(2048)
    d.filling = [s]
    d._fill()
    assert d.cancel([s]) == [s]
    assert d.e.ops[-1][:2] == (OP_LAYERS, -1) and d.split is None and s.pos == 0


def test_plan_adds_groups_rather_than_overshoot_the_target():
    cost = [0.002] * L
    for i in (5, 11, 17, 23, 29):                    # long context: full-attention layers dominate
        cost[i] = 0.05
    d = decoder(cost)
    d.streams = {0: object()}
    stops = d._plan(1024)
    edges = [0] + stops
    groups = [sum(cost[a:b]) * 1024 for a, b in zip(edges, edges[1:])]
    target = E.Decoder.GAP_MS - d.round_ms
    assert stops[-1] == L and stops == sorted(set(stops))
    assert max(groups) <= target or len(groups) == L


def test_draft_packet_and_proposals_land_before_the_round_done_event(monkeypatch):
    import pytest

    torch = pytest.importorskip("torch")
    from tensorfold.families.gemma4.sm70.graphs import MODE_ROUND, Shape

    copies = []
    packet = torch.zeros((8, 2, 3))
    packet[0] = torch.tensor([[2., 9., 0.], [3., 12., 0.]])
    packet[1, 0] = torch.tensor([2., 13., 0.])  # equal score chooses rank zero
    proposals = torch.tensor([9])

    def landing(source):
        data = torch.empty_like(source)

        def enqueue(value, *, non_blocking):
            assert non_blocking
            copies.append((data, value))

        return SimpleNamespace(copy_=enqueue, numpy=data.numpy, view=data.view)

    shape = object.__new__(Shape)
    shape.gather_all = False
    shape.host_compact, shape.host_proposals = landing(packet), landing(proposals)
    shape.stage = lambda *args: None
    shape.run = lambda: (packet, proposals)
    e = object.__new__(E.Gemma4SM70Engine)
    e.streams, e.block, e.pad = 1, 2, 1
    e.shapes = {(MODE_ROUND, 1): shape}
    e.announce = e._run

    def record():
        assert [source.data_ptr() for _, source in copies] == [packet.data_ptr(), proposals.data_ptr()]

    def synchronize():
        for target, source in copies:
            target.copy_(source)

    s = SimpleNamespace(sid=0, slot=0, prompt=[3, 4], out=[6], count=8, done=False, full=False,
                        drafts=True, forced=None, probabilities=None, constraint=None, history=None,
                        stop_eos=False, counted=lambda n: None)
    d = object.__new__(E.Decoder)
    d.e, d.eos, d.streams = e, (), {s.sid: s}
    d.costs = dict.fromkeys(e.shapes, 1.0)
    d.round_ms = d.tokens_per_round = 1.0
    d.started = SimpleNamespace(record=lambda: None, elapsed_time=lambda event: 1.0)
    d.done = SimpleNamespace(record=record, synchronize=synchronize)
    monkeypatch.setattr(E, "TRACE", False)
    d._launch([s], {})
    assert d._collect() == [(s, [9, 12])]

