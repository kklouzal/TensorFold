"""Flash Next prompt passes in layer groups while replies decode (MultiDecoder host logic, no GPU): the shared group
plan, the rank-identical stops, a grouped pass's progress/completion, and cancellation/settling."""

from types import SimpleNamespace

import pytest
import torch

from tensorfold.cuda.geometry import DECODE_GAP_MS, layer_groups
from tensorfold.families.qwen4_exp.cuda import multi as M

L = 8


def test_layer_groups_keep_groups_under_target_and_end_with_every_layer():
    cost = [10.0] * L
    assert layer_groups(cost, 100.0) == [L]
    stops = layer_groups(cost, 30.0)
    edges = [0] + stops
    assert stops[-1] == L and stops == sorted(set(stops))
    assert max(sum(cost[a:b]) for a, b in zip(edges, edges[1:])) <= 30.0


class Event:
    clock = 0.0

    def __init__(self, enable_timing=False):
        self.t = None

    def record(self):
        Event.clock += 2.0                              # every layer: 2 ms
        self.t = Event.clock

    def query(self):
        return True

    def elapsed_time(self, other):
        return other.t - self.t


def decoder(monkeypatch, gap=100.0):
    calls = []

    def layers(w, segs, b, pending, l0, l1, *, cuts=(), marks=None):
        calls.append((l0, l1))
        for _ in range(l0, l1):
            event = Event()
            event.record()
            marks.append(event)
        return ("pending", l1)

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(M, "prompt_layers", layers)
    monkeypatch.setattr(M, "prompt_begin", lambda w, segs, b: calls.append("begin"))
    monkeypatch.setattr(M, "finish", lambda w, mixer, b, R, pending, logits, ends: calls.append(("finish", pending))
                        or torch.zeros(4, 3))
    monkeypatch.setattr(M, "_kv_check", lambda segs, comm=None: None)
    d = object.__new__(M.MultiDecoder)
    d.w = SimpleNamespace(layers=[object()] * L, mixer=None, comm=None)
    d.gap_ms, d.layer_ms, d.group_round_ms, d._gap_inputs = gap, None, None, None
    d._grouped = d._grouped_ready = None
    d.row_s = d.round_s = None
    d.controller, d.passed = None, {}
    d.pbuf = SimpleNamespace(rotor_lookup=True)
    s = SimpleNamespace(sid=1, prompt=list(range(4096)), cancel=None, carry=None)
    d.filling = [s]
    d._prepare_pass = lambda pieces: (None, [("state", 0, 1024)], [], [])
    d._absorb = lambda pieces, segs, cuts: ["last"]
    d._joined = lambda pieces, heads, lasts, spent: calls.append(("joined", lasts)) or []
    return d, s, calls


def test_stops_need_measured_costs_and_follow_the_leaders_estimates(monkeypatch):
    d, _, _ = decoder(monkeypatch, gap=DECODE_GAP_MS)
    assert d._layer_stops(1024) is None                 # nothing measured yet: a merged whole pass
    d.row_s, d.round_s = 0.32e-3, 0.065                 # 1,024 rows: 328 ms a pass; 65 ms a round
    stops = d._layer_stops(1024)
    assert stops == [2, 4, 6, L]                        # 110 ms groups: two 41 ms layers each
    assert d._layer_stops(256) is None                  # 82 ms: one group, so the window rides it (merged)
    d._gap_inputs = {"layer_ms": None, "row_ms": 0.32, "round_ms": 160.0}   # the leader's, sent with the round
    assert len(d._layer_stops(1024)) == 8               # target floor gap/4: every rank cuts the same layers


def test_grouped_pass_runs_one_group_a_round_then_completes_and_measures(monkeypatch):
    d, s, calls = decoder(monkeypatch)
    d._group_start([(s, 0, 1024)], [3, 6, L])
    assert d.pbuf.rotor_lookup is False and calls == ["begin"]
    marks, done = d._group()
    assert done is None and d._grouped.layer == 3 and d._grouped.pending == ("pending", 3)
    d._group_timed(marks, 0.050)                        # the group's 3 layers took 6 ms of a 50 ms round
    assert d.layer_ms[:3] == [2.0 / 1024] * 3 and d.layer_ms[3:] == [None] * 5
    assert d.group_round_ms == pytest.approx(44.0)
    marks, done = d._group()
    assert done is None
    d._group_timed(marks, 0.050)
    marks, (g, heads, lasts) = d._group()
    assert d._grouped is None and lasts == ["last"] and heads is None
    assert calls[1:] == [(0, 3), (3, 6), (6, L), ("finish", ("pending", L))]
    d._group_timed(marks, 0.050)
    assert None not in d.layer_ms


def test_cancel_waits_for_the_grouped_pass_and_the_next_round_settles_it(monkeypatch):
    d, s, calls = decoder(monkeypatch)
    d._group_start([(s, 0, 1024)], [3, 6, L])
    d._group()
    assert d.cancel([s]) == [] and d.filling == [s]     # its DeltaNet states hold 3 layers: not cancelled mid-pass
    s.cancel = SimpleNamespace(is_set=lambda: True)
    assert d._group_settle([object()]) == []            # a cancelled prompt in it: the remaining layers at once
    assert calls[-3:] == [(3, L), ("finish", ("pending", L)), ("joined", ["last"])]
    assert d._grouped is None and d._grouped_ready is None


def test_without_decoding_replies_the_pass_completes_before_any_other(monkeypatch):
    d, s, calls = decoder(monkeypatch)
    d._group_start([(s, 0, 1024)], [3, 6, L])
    d._group()
    d._group_settle([])
    assert calls[-3:] == [(3, L), ("finish", ("pending", L)), ("joined", ["last"])] and d._grouped is None
