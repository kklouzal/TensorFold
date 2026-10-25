"""Request retirement and bounded derived prefixes have distinct ownership."""

from types import SimpleNamespace as NS

import pytest

from tensorfold.families.qwen4_exp.cuda import prefixes


class Slot:
    def __init__(self, number):
        self.history_slot, self.slot_generation = number, 0
        self.pos = self.mtp_len = self.reset_calls = 0
        self.data, self.recurrence = bytearray(64), bytearray(16)
        self.events = []

    def begin_sequence_generation(self):
        self.slot_generation += 1
        self.events.append(("begin", self.slot_generation))

    def kv_validate_snapshot(self, _snapshot):
        pass

    def copy_prefix(self, source, pos, mtp_len):
        self.events.append(("copy", self.slot_generation, source, pos, mtp_len))
        self.data[:] = source.data

    def reset(self):
        self.reset_calls += 1
        self.pos = self.mtp_len = 0
        self.data[:] = bytes(len(self.data))
        self.recurrence[:] = bytes(len(self.recurrence))


def owner(count):
    pytest.importorskip("torch")
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    decoder = object.__new__(MultiDecoder)
    decoder.full_resident = True
    decoder.states = tuple(Slot(i) for i in range(count))
    decoder.free = list(decoder.states)
    decoder.streams, decoder.filling, decoder.held = {}, [], {}
    decoder.kept, decoder.keep, decoder.used, decoder.clock, decoder.depth = [], 8, {}, 0, 3
    decoder._grow = lambda *_args, **_kwargs: True
    decoder._shrink = lambda state, **_kwargs: state.reset()
    return decoder


def admit(decoder, number):
    state, resume, cached = decoder._slot_for([number] * 33, False)
    assert resume is None and cached == 0
    state.data[:] = bytes([number % 256]) * len(state.data)
    state.recurrence[:] = bytes([(number + 17) % 256]) * len(state.recurrence)
    state.pos = state.mtp_len = 32
    stream = NS(sid=number, st=state, out=[number], error=None, done=True)
    decoder.streams[stream.sid] = stream
    return stream


@pytest.mark.parametrize("count", [1, 18, 37])
def test_completed_requests_release_their_claim_and_clear_uncached_state(count):
    decoder = owner(count)
    streams = [admit(decoder, number) for number in range(count)]
    assert len({id(stream.st) for stream in streams}) == count
    before = [(bytes(stream.st.data), bytes(stream.st.recurrence)) for stream in streams]
    assert before == [(bytes([i]) * 64, bytes([i + 17]) * 16) for i in range(count)]
    decoder.finish(streams)
    assert decoder.live() == 0 and len(decoder.free) == count
    assert all(state.reset_calls == 1 and state.pos == state.mtp_len == 0 for state in decoder.states)
    assert all(not any(state.data) and not any(state.recurrence) for state in decoder.states)
    assert not hasattr(decoder, "retained")


@pytest.mark.parametrize("count", [1, 18, 37])
def test_derived_cache_is_bounded_and_does_not_keep_every_completed_endpoint(count):
    decoder = owner(count)
    for number in range(count):
        stream = admit(decoder, number)
        prefixes.remember(decoder, [number] * 31, stream.st, {"pos": 31, "mtp_len": 30}, None)
        decoder.finish([stream])
    assert decoder.live() == 0 and len(decoder.kept) <= decoder.keep
    cached = {id(entry[1]) for entry in decoder.kept}
    assert len(cached) == min(count, decoder.keep)
    assert len(cached | {id(state) for state in decoder.free}) == count
    assert not hasattr(decoder, "retained")


def test_in_place_prefix_hit_begins_new_generation_before_restore_can_write():
    decoder = owner(1)
    stream = admit(decoder, 1)
    prefixes.remember(decoder, [1] * 31, stream.st, {"pos": 31, "mtp_len": 30}, None)
    decoder.finish([stream])
    before = stream.st.slot_generation
    selected, resume, cached = decoder._slot_for([1] * 32, True)
    assert selected is stream.st and cached == 31
    assert selected.slot_generation == before + 1
    assert selected.events[-1] == ("begin", before + 1)
    assert resume["state"] is decoder.kept[0][2]


def test_fork_generation_precedes_copy_and_preserves_the_source():
    decoder = owner(2)
    stream = admit(decoder, 1)
    prefixes.remember(decoder, [1] * 31, stream.st, {"pos": 31, "mtp_len": 30}, None)
    prefixes.remember(decoder, [1] * 40, stream.st, {"pos": 40, "mtp_len": 39}, None)
    decoder.finish([stream])
    source_generation, source_bytes = stream.st.slot_generation, bytes(stream.st.data)
    selected, resume, cached = decoder._slot_for([1] * 31 + [99], True)
    assert selected is not stream.st and cached == 31
    assert selected.events == [("begin", 1), ("copy", 1, stream.st, 31, 30)]
    assert stream.st.slot_generation == source_generation and bytes(stream.st.data) == source_bytes
    assert resume["state"]["pos"] == 31


def test_no_free_or_idle_derived_slot_preserves_active_request_ownership():
    decoder = owner(3)
    streams = [admit(decoder, number) for number in range(3)]
    before = [(bytes(s.st.data), s.st.slot_generation) for s in streams]
    with pytest.raises(RuntimeError, match="no free stream slot"):
        decoder._slot_for([99], False)
    assert [(bytes(s.st.data), s.st.slot_generation) for s in streams] == before


def test_prefix_borrowing_disabled_by_policy_stays_cold_despite_a_matching_cache():
    decoder = owner(2)
    stream = admit(decoder, 2)
    prefixes.remember(decoder, [2] * 31, stream.st, {"pos": 31, "mtp_len": 30}, None)
    decoder.finish([stream])
    selected, resume, cached = decoder._slot_for([2] * 32, False)
    assert selected is not stream.st and resume is None and cached == 0
    assert len(decoder.kept) == 1
