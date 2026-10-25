"""Distinct retained endpoints survive optional prefix-ladder eviction and load changes."""

from types import SimpleNamespace as NS

import pytest

from tensorfold.families.qwen4_exp.cuda import prefixes


def owner(count):
    pytest.importorskip("torch")
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    decoder = object.__new__(MultiDecoder)
    decoder.retain_sessions = decoder.full_resident = True
    decoder.retained = {}
    decoder.retention_clock = decoder.retention_retirements = 0
    decoder.last_retirement = None
    decoder.states = tuple(NS(history_slot=i, slot_generation=0, pos=0, mtp_len=0,
                              data=bytearray(64), recurrence=bytearray(16)) for i in range(count))
    decoder.free = list(decoder.states)
    decoder.streams, decoder.filling, decoder.held = {}, [], {}
    decoder.kept, decoder.keep, decoder.used, decoder.clock, decoder.depth = [], 8, {}, 0, 3
    decoder._grow = lambda *_args, **_kwargs: True
    decoder._shrink = lambda *_args, **_kwargs: pytest.fail("idle retained endpoint was reset or shrunk")
    return decoder


def complete(decoder, number, *, keep_prefix=True):
    state, resume, cached = decoder._slot_for([number] * 33, False)
    assert resume is None and cached == 0
    state.slot_generation += 1
    state.data[:] = bytes([number % 256]) * len(state.data)
    state.recurrence[:] = bytes([(number + 17) % 256]) * len(state.recurrence)
    state.pos = state.mtp_len = 32
    state.snapshot = lambda: pytest.fail("retaining a live endpoint made a redundant snapshot")
    stream = NS(sid=number, st=state, out=[number], error=None, done=True)
    decoder.streams[stream.sid] = stream
    if keep_prefix:
        prefixes.remember(decoder, [number] * 31, state, {"pos": 31, "mtp_len": 30}, None)
    decoder.finish([stream])
    return state


@pytest.mark.parametrize("count", [1, 18, 37])
def test_every_distinct_idle_context_survives_the_eight_entry_optional_prefix_limit(count):
    decoder = owner(count)
    states = [complete(decoder, number) for number in range(count)]
    assert len({id(state) for state in states}) == count
    assert len(decoder.retained) == count and len(decoder.kept) <= 8
    assert {id(state) for state in decoder.retained_states()} == {id(state) for state in states}
    assert decoder.retention_retirements == 0
    assert [bytes(state.data) for state in states] == [bytes([number]) * 64 for number in range(count)]
    assert [bytes(state.recurrence) for state in states] == [bytes([number + 17]) * 16 for number in range(count)]


@pytest.mark.parametrize("count", [1, 18, 37])
def test_one_new_admission_replaces_only_the_global_lru_endpoint(count):
    decoder = owner(count)
    states = [complete(decoder, number) for number in range(count)]
    before = {id(state): (bytes(state.data), bytes(state.recurrence), state.slot_generation) for state in states}
    selected, resume, cached = decoder._slot_for([99] * 33, False)
    assert selected is states[0] and resume is None and cached == 0
    assert decoder.retention_retirements == 1
    assert decoder.last_retirement["slot"] == selected.history_slot
    assert len(decoder.retained) == count - 1
    for state in states[1:]:
        assert (bytes(state.data), bytes(state.recurrence), state.slot_generation) == before[id(state)]
    # One active request beside N-1 idle histories still owns precisely N slots.
    decoder.streams[99] = NS(sid=99, st=selected, out=[99], done=False, error=None)
    assert len(decoder.retained_states()) + decoder.live() == count


def test_global_lru_includes_a_slot_with_kept_points_even_if_other_slots_are_free():
    decoder = owner(3)
    states = [complete(decoder, number, keep_prefix=number == 0) for number in range(3)]
    assert all(state is not states[0] for state in decoder.free)
    selected, _, _ = decoder._slot_for([99], False)
    assert selected is states[0]
    assert all(entry[1] is not selected for entry in decoder.kept)
    assert {id(state) for state in decoder.retained_states()} == {id(state) for state in states[1:]}


def test_unused_slot_precedes_any_retained_endpoint_and_ties_use_stable_slot_numbers():
    decoder = owner(4)
    first = complete(decoder, 1, keep_prefix=False)
    assert first.history_slot == 0
    selected = prefixes._take_free(decoder)
    assert selected.history_slot == 1
    assert decoder.retention_retirements == 0 and id(first) in decoder.retained


def test_active_and_protected_contexts_cannot_be_replaced():
    decoder = owner(4)
    states = [complete(decoder, number, keep_prefix=False) for number in range(4)]
    decoder.streams[100] = NS(st=states[0])
    selected = prefixes._take_free(decoder, protect=states[1])
    assert selected is states[2]
    assert id(states[0]) in decoder.retained and id(states[1]) in decoder.retained


def test_no_idle_slot_leaves_every_endpoint_owned_and_reports_admission_failure():
    decoder = owner(3)
    states = [complete(decoder, number, keep_prefix=False) for number in range(3)]
    decoder.streams = {number: NS(st=state) for number, state in enumerate(states)}
    with pytest.raises(RuntimeError, match="no idle retained session slot"):
        decoder._slot_for([99], False)
    assert decoder.retention_retirements == 0 and len(decoder.retained) == 3


def test_in_place_prefix_resume_retires_only_that_endpoint_and_keeps_valid_snapshot():
    decoder = owner(1)
    state = complete(decoder, 1)
    state.kv_validate_snapshot = lambda _snap: None
    snapshot = decoder.kept[0][2]
    selected, resume, cached = decoder._slot_for([1] * 32, True)
    assert selected is state and cached == 31
    assert resume["state"] is snapshot
    assert decoder.retention_retirements == 1 and decoder.retained == {}
    assert len(decoder.kept) == 1


def test_prefix_borrowing_disabled_by_policy_stays_cold_despite_an_exact_kept_match():
    decoder = owner(2)
    first, second = complete(decoder, 1), complete(decoder, 2)
    selected, resume, cached = decoder._slot_for([2] * 32, False)
    assert selected is first and selected is not second
    assert resume is None and cached == 0
    assert id(second) in decoder.retained


def test_fork_copy_retires_destination_before_writing_and_preserves_source_endpoint():
    decoder = owner(3)
    states = [complete(decoder, number) for number in range(3)]
    source = states[1]
    source.kv_validate_snapshot = lambda _snap: None
    prefixes.remember(decoder, [1] * 40, source, {"pos": 40, "mtp_len": 39}, None)
    writes = []

    def copy(other, pos, mtp_len):
        assert id(states[0]) not in decoder.retained
        assert id(source) in decoder.retained
        writes.append((other, pos, mtp_len))

    states[0].copy_prefix = copy
    selected, resume, cached = decoder._slot_for([1] * 31 + [99], True)
    assert selected is states[0] and cached == 31
    assert writes == [(source, 31, 30)] and resume["state"]["pos"] == 31
    assert decoder.retention_retirements == 1
    assert bytes(source.data) == bytes([1]) * 64
