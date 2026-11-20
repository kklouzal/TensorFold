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
