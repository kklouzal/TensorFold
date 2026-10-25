"""Admission identity changes precede cached writes; rollback keeps its identity."""

from types import SimpleNamespace as NS

import pytest

pytest.importorskip("torch")

from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill_begin
from tensorfold.families.qwen4_exp.cuda.state import State


def generation_state(value=0):
    state = object.__new__(State)
    state._resize_error = state._kv_error = state._history_error = None
    state.slot_generation, state.history_generations = value, None
    return state


@pytest.mark.parametrize("value", [-1, True, None, 1.5, (1 << 63) - 1])
def test_invalid_or_exhausted_generation_is_rejected_before_publication(value):
    state = generation_state(value)
    writes = []
    state.history_generations = [NS(fill_=lambda number: writes.append(number))]
    with pytest.raises(RuntimeError, match="invalid or exhausted"):
        state.begin_sequence_generation()
    assert writes == [] and state.slot_generation is value


def test_publication_uses_the_next_signed64_generation_before_host_commit():
    state = generation_state((1 << 63) - 2)
    writes = []
    state.history_generations = [NS(fill_=lambda number: writes.append((number, state.slot_generation)))]
    assert state.begin_sequence_generation() == (1 << 63) - 1
    assert writes == [((1 << 63) - 1, (1 << 63) - 2)]
    with pytest.raises(RuntimeError, match="exhausted"):
        state.begin_sequence_generation()
    assert len(writes) == 1


@pytest.mark.parametrize("poison", ["_resize_error", "_history_error"])
def test_terminal_state_never_starts_another_sequence(poison):
    state = generation_state(4)
    primary = RuntimeError("terminal state")
    setattr(state, poison, primary)
    with pytest.raises(RuntimeError) as caught:
        state.begin_sequence_generation()
    assert caught.value is primary and state.slot_generation == 4


def test_prefill_restore_begins_new_sequence_before_writing_cached_state():
    events = []
    state = NS(pos=2)
    state.begin_sequence_generation = lambda: events.append("begin")
    state.restore = lambda snapshot: events.append(("restore", snapshot))
    engine = NS(st=state, w=NS(mtp=None), mbuf=None)
    snapshot = {"pos": 2}
    assert prefill_begin(engine, [1, 2, 3], resume={"state": snapshot}) == 2
    assert events == ["begin", ("restore", snapshot)]
    events.clear()
    # Multi admission already published the generation before a possible fork
    # copy. Its later snapshot restore must not start a second sequence.
    assert prefill_begin(engine, [1, 2, 3], resume={"state": snapshot}, new_sequence=False) == 2
    assert events == [("restore", snapshot)]


def test_cold_multi_prefill_clears_payload_without_a_second_generation():
    calls = []
    engine = object.__new__(Engine)
    engine.st = NS(reset=lambda weights, *, new_sequence: calls.append((weights, new_sequence)))
    engine.w = object()
    assert prefill_begin(engine, [1], new_sequence=False) == 0
    assert calls == [(engine.w, False)]


def test_real_state_restore_and_in_request_reset_preserve_sequence_generation():
    from tests.test_sm70_rotorquant_state import model

    weights = model(mtp=True)
    state = State(weights, 37, 4, "fp16")
    state.begin_sequence_generation()
    snapshot = state.snapshot()
    generation = state.slot_generation
    state.restore(snapshot)
    assert state.slot_generation == generation
    state.reset(weights, new_sequence=False)
    assert state.slot_generation == generation
    state.reset(weights)
    assert state.slot_generation == generation + 1
