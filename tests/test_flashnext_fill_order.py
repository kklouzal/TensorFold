"""Shortest-first CUDA prompt passes preserve foreground priority and bounded starvation."""

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.torch


def _decoder(prompts):
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    m = object.__new__(MultiDecoder)
    m.filling = [SimpleNamespace(sid=i, prompt=[0] * n, background=bg, cancel=None)
                 for i, (n, bg) in enumerate(prompts)]
    m.fills = {s.sid: [None, False, done, None] for s, (_, _, done) in zip(m.filling, [(*p, 0) for p in prompts])}
    m.passed = {}
    return m


def test_fewest_rows_left_first_then_arrival():
    m = _decoder([(120_000, False), (2_000, False), (2_000, False), (30_000, False)])
    assert [s.sid for s in m._order()] == [1, 2, 3, 0]


def test_background_prompts_after_foreground_ones():
    m = _decoder([(2_000, True), (50_000, False)])
    assert [s.sid for s in m._order()] == [1, 0]


def test_a_prompt_passed_over_fill_guard_passes_goes_first():
    from tensorfold.families.qwen4_exp.cuda.multi import FILL_GUARD

    m = _decoder([(120_000, False), (2_000, False)])
    short = m.filling[1]
    for _ in range(FILL_GUARD):
        m._note_passed([(short, 0, 2_000)])
    assert m.passed == {0: FILL_GUARD, 1: 0}
    assert [s.sid for s in m._order()][0] == 0                     # the long one is due
    m._note_passed([(m.filling[0], 0, 2_048)])
    assert m.passed == {0: 0, 1: 1}


def test_counts_drop_with_prompts_that_left():
    m = _decoder([(4_000, False), (2_000, False)])
    m._note_passed([(m.filling[1], 0, 2_000)])
    m.filling = m.filling[:1]
    m._note_passed([])
    assert m.passed == {0: 2}


def test_a_waiting_request_stops_a_lone_prompts_passes():
    """With no stream decoding, passes stop for a request that waits to be admitted (it then fills beside them)."""
    m = _decoder([(32_000, False)])
    m.streams, passes = {}, []

    def one_pass():
        passes.append(1)
        if len(passes) == 16:
            m.filling = []                     # the prompt ended
        return []

    m._pass = one_pass
    m.arrived = lambda: len(passes) >= 3
    m._fill()
    assert len(passes) == 3
    m.arrived = lambda: False
    m._fill()
    assert len(passes) == 16


def test_a_departed_client_stops_a_lone_prompts_passes():
    """With no stream decoding, passes also stop once a filling prompt's client has left: the scheduler's next step
    cancels it (``MultiDecoder.cancel``), rather than after the whole prompt."""
    import threading

    m = _decoder([(32_000, False), (64_000, False)])
    m.streams, m.arrived, passes = {}, lambda: False, []
    m.filling[1].cancel = threading.Event()

    def one_pass():
        passes.append(1)
        if len(passes) == 4:
            m.filling[1].cancel.set()          # its client left during this pass
        return []

    m._pass = one_pass
    m._fill()
    assert len(passes) == 4 and len(m.filling) == 2


def test_order_uses_remaining_rows_and_preserves_oldest_equal_prompt():
    m = _decoder([(8_000, False), (2_000, False), (1_000, False)])
    m.fills[0][2] = 7_000
    assert [s.sid for s in m._order()] == [0, 2, 1]


def test_pieces_cut_at_message_boundaries_within_live_row_limits():
    """A pass holds a prompt's message starts as cuts (its kept states), up to CUTS a piece; a further one ends it."""
    from tensorfold.families.qwen4_exp.cuda.prompt_plan import CUTS

    m = _decoder([(9_000, False), (3_000, False)])
    m.prefill_rows, m.share, m.round_s, m.row_s = 4_096, 0.0, None, None
    m.streams = {7: SimpleNamespace(done=False)}
    for s in m.filling:
        s.draft, s.st = True, SimpleNamespace(image_positions=None)
    for fill in m.fills.values():
        fill[0], fill[3] = SimpleNamespace(stops=[]), []
    m.fills[1][0].stops = [800]
    pieces = m._pieces()
    assert [(s.sid, a, n) for s, a, n in pieces] == [(1, 0, 2_048)]          # a live reply: one released pass
    assert [(cut.row, cut.at) for cut in m._cuts(pieces, [(None, 0, 2_048)])] == [(800, 0)]
    m.fills[1][0].stops = [100 * (i + 1) for i in range(CUTS + 2)]
    end = 100 * (CUTS + 1)                                                   # the point past CUTS ends the piece
    assert [(s.sid, a, n) for s, a, n in m._pieces()] == [(1, 0, end), (0, 0, 2_048 - end)]
    m.fills[1][0].stops = [800]
    m.streams.clear()
    assert sum(n for _, _, n in m._pieces()) == 4_096


def test_scheduler_wires_the_foreground_check_before_starting(monkeypatch):
    from tensorfold.cuda.scheduler import Scheduler
    from tensorfold.cuda.streams import Stream
    import tensorfold.cuda.scheduler as scheduler

    m = SimpleNamespace(arrived=lambda: False)
    started = []
    def start():
        started.append(1)
        sched._worker_entered.set()               # this wiring fixture substitutes the actor-entry handshake

    monkeypatch.setattr(scheduler.threading, "Thread", lambda **kw: SimpleNamespace(start=start))
    sched = Scheduler(m)
    sched.start()
    assert started == [1] and not m.arrived()
    sched.waiting.put((Stream([1], 1, background=True), None))
    assert not m.arrived()
    sched.waiting.put((Stream([2], 1), None))
    assert m.arrived()
