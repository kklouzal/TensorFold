"""A shared kept prefix is copied into a spare slot while its longer chain retains its own state."""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

from tests.test_flashnext_prefix_copy import slot as state_slot

from tensorfold.cuda.memory_gate import NoRoom
from tensorfold.cuda.streams import Stream
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder


def slot():
    state = state_slot("bf16", 1024, 512)
    state.copied = None

    def copy_prefix(source, pos, mtp_len):
        state.copied = (source, pos, mtp_len)

    state.copy_prefix = copy_prefix
    return state


def decoder(*, busy=False, room=True):
    dec = object.__new__(MultiDecoder)
    dec.used, dec.clock = {}, 0
    dec.retain_sessions, dec.retained = False, {}
    dec.retention_retirements, dec.last_retirement = 0, None
    dec.w = SimpleNamespace(comm=None, meta={"execution_provider": "cuda"})
    dec.graphs = None
    source, spare = slot(), slot()
    prefix = list(range(300))
    snap = {"pos": 300, "mtp_len": 299, **source.kv_snapshot_metadata()}
    dec.kept = [(prefix, source, snap, "tail"), (prefix + [4, 5], source, {"pos": 302, "mtp_len": 301, **source.kv_snapshot_metadata()}, "later")]
    dec.free, dec.filling, dec.fills, dec.failed_slots = [spare], [], {}, []
    dec.streams = {1: SimpleNamespace(st=source, waiting=False)} if busy else {}
    dec.depth, dec.capacity = 3, 1024
    dec.kv_pair = source.kv_pair
    growth = []
    def grow(st, rows, **kwargs):
        growth.append((st, rows, kwargs))
        return room
    dec._grow = grow
    return dec, source, spare, prefix, snap, growth


@pytest.mark.parametrize("busy", [False, True])
def test_a_spare_slot_reuses_the_shared_point_without_consuming_the_source_chain(busy):
    dec, source, spare, prefix, snap, growth = decoder(busy=busy)
    kept = list(dec.kept)
    st, resume, cached = dec._slot_for(prefix + [6, 7, 8], True)
    assert st is spare and cached == len(prefix)
    assert resume == {"state": snap, "tail": "tail"}
    assert spare.copied == (source, len(prefix), 299)
    assert dec.kept == kept and dec.free == []
    assert growth[0][2].get("protect") is source


def test_copy_reservation_refusal_keeps_the_source_and_returns_the_unadmitted_slot():
    dec, source, spare, prefix, _, _ = decoder(busy=True, room=False)
    kept = list(dec.kept)
    with pytest.raises(NoRoom):
        dec.admit(Stream(prefix + [6, 7, 8], 8))
    assert dec.free == [spare] and dec.kept == kept and spare.copied is None


def test_source_protection_excludes_it_from_growth_eviction():
    dec, source, spare, _, _, _ = decoder()
    other = slot()
    dec.kept.append(([9], other, {}, None))
    shrunk = []
    dec._shrink = lambda st, **kwargs: shrunk.append(st)
    assert dec._evict_kept(spare, protect=source)
    assert shrunk == [other] and all(k[1] is source for k in dec.kept)


def test_copy_failure_returns_the_spare_and_discards_partial_allocations():
    dec, _, spare, prefix, _, _ = decoder()
    kept, shrunk = list(dec.kept), []
    def fail(*args):
        raise RuntimeError("copy failed")
    spare.copy_prefix = fail
    dec._shrink = lambda st, **kw: shrunk.append((st, kw))
    with pytest.raises(RuntimeError, match="copy failed"):
        dec._slot_for(prefix + [6, 7, 8], True)
    assert dec.free == [spare] and dec.kept == kept
    assert shrunk == [(spare, {"force": True})]


@pytest.mark.parametrize("phase", ["growth", "copy"])
@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_fork_failure_preserves_source_and_exclusive_spare_through_retirement(
        phase, failure_type, cleanup_failure):
    dec, source, spare, prefix, snap, _ = decoder()
    kept = list(dec.kept)
    primary, cleanup = failure_type("fork failed"), RuntimeError("fork retirement failed")
    shrunk = []

    def fail(*args, **kwargs):
        raise primary

    def shrink(st, **kwargs):
        shrunk.append((st, kwargs))
        if cleanup_failure:
            raise cleanup

    if phase == "growth":
        dec._grow = fail
    else:
        spare.copy_prefix = fail
    dec._shrink = shrink
    with pytest.raises(failure_type) as caught:
        dec._slot_for(prefix + [6, 7, 8], True)
    assert caught.value is primary and dec.kept == kept
    assert all(entry[1] is source for entry in dec.kept)
    if cleanup_failure:
        assert primary.__cause__ is cleanup
        assert dec.free == [] and dec.failed_slots == [spare]
        dec._shrink = lambda st, **kwargs: shrunk.append((st, kwargs))
        dec.recover_failed_slots()
    assert dec.free == [spare] and dec.failed_slots == []
    assert all(st is spare and kwargs == {"force": True} for st, kwargs in shrunk)
    dec._grow = lambda *args, **kwargs: True
    spare.copy_prefix = lambda *args: None
    following, resume, cached = dec._slot_for(prefix + [9], True)
    assert following is spare and cached == len(prefix) and resume == {"state": snap, "tail": "tail"}
    assert dec.free == [] and dec.failed_slots == [] and dec.kept == kept


def test_a_partial_resize_failure_is_counted_before_the_spare_is_shrunk():
    from tensorfold.cuda.memory_gate import MemoryGate

    dec, _, spare, prefix, _, _ = decoder()
    del dec._grow
    dec.memory_gate = MemoryGate(10_000, 0)
    dec.memory_gate.held = 700
    spare.capacity, spare.limit, spare.allocated = 256, 1024, 100
    spare.cache_bytes = lambda size=None: spare.allocated if size is None else 900
    spare.layer_bytes = lambda size: 100
    def partial(size):
        spare.allocated = 300
        raise RuntimeError("partial allocation")
    spare.resize = partial
    held = []
    def shrink(st, **kwargs):
        held.append(dec.memory_gate.held)
        dec.memory_gate.give(st.allocated - 100)
        st.allocated = 100
    dec._shrink = shrink
    with pytest.raises(RuntimeError, match="partial allocation"):
        dec._slot_for(prefix + [6, 7, 8], True)
    assert held == [900] and dec.memory_gate.held == 700
    assert dec.free == [spare] and spare.allocated == 100


def test_invalid_kept_snapshot_stops_before_spare_growth_or_copy():
    dec, source, spare, prefix, snap, growth = decoder()
    kept = list(dec.kept)
    snap["pos"] = True
    with pytest.raises(ValueError, match="integer rows"):
        dec._slot_for(prefix + [6], True)
    assert dec.free == [spare] and dec.kept == kept
    assert growth == [] and spare.copied is None


def _with_end(dec, prefix, source):
    end_state = {"pos": len(prefix) + 1, "mtp_len": len(prefix), **source.kv_snapshot_metadata()}
    end = {"token": 7, "state": end_state, "tail": "end-tail", "last": "end-last", "head": "end-head"}
    dec.kept[0][2]["prompt_end"] = end
    return end_state


@pytest.mark.parametrize("busy", [False, True])
def test_an_exact_repeat_resumes_from_the_kept_prompt_end(busy):
    dec, source, spare, prefix, snap, _ = decoder(busy=busy)
    end_state = _with_end(dec, prefix, source)
    st, resume, cached = dec._slot_for(prefix + [7], True)
    assert st is spare and cached == len(prefix) + 1
    assert resume == {"state": end_state, "tail": "end-tail", "head": "end-head", "last": "end-last",
                      "kept": (prefix, snap, "tail")}
    assert spare.copied == (source, len(prefix) + 1, len(prefix))      # the end's last row is copied too


def test_an_exact_repeat_resumes_its_end_in_place_without_a_spare_slot():
    dec, source, _, prefix, _, _ = decoder()
    dec.free = []
    dec.kept = dec.kept[:1]
    end_state = _with_end(dec, prefix, source)
    st, resume, cached = dec._slot_for(prefix + [7], True)
    assert st is source and cached == len(prefix) + 1 and resume["state"] is end_state


@pytest.mark.parametrize("tail", [[8], [7, 9]])
def test_another_prompt_through_the_kept_point_resumes_the_point_not_the_end(tail):
    dec, source, spare, prefix, snap, _ = decoder()
    _with_end(dec, prefix, source)
    st, resume, cached = dec._slot_for(prefix + tail, True)
    assert st is spare and cached == len(prefix) and resume == {"state": snap, "tail": "tail"}
    assert spare.copied == (source, len(prefix), 299)


@pytest.mark.parametrize("exact", [False, True])
def test_prefill_begin_returns_an_exact_end_without_absorbing_its_tail(monkeypatch, exact):
    from tensorfold.families.qwen4_exp.cuda import decode

    absorbed = []
    monkeypatch.setattr(decode, "mtp_forward", lambda *args, **kwargs: absorbed.append(args[3]))
    st = SimpleNamespace(pos=0, mtp_len=0)
    st.restore = lambda snap: setattr(st, "pos", snap["pos"])
    st.set_mtp_len = lambda n: setattr(st, "mtp_len", n)
    e = SimpleNamespace(st=st, w=SimpleNamespace(mtp=object()), mbuf=object(), pbuf=None)
    prompt = list(range(10))
    resume = {"state": {"pos": 10 if exact else 9}, "tail": "tail", **({"head": "head"} if exact else {})}
    assert decode.prefill_begin(e, prompt, resume=resume, new_sequence=False) == (10 if exact else 9)
    assert absorbed == ([] if exact else [[9]])
    with pytest.raises(ValueError, match="extend the cached tokens"):          # a point needs a row to compute
        decode.prefill_begin(e, prompt, resume={"state": {"pos": 10}, "tail": None}, new_sequence=False)
    with pytest.raises(ValueError, match="extend the cached tokens"):          # an end must be the prompt's own
        decode.prefill_begin(e, prompt + [10], resume={**resume, "state": {"pos": 10}, "head": "h"},
                             new_sequence=False)


def test_a_refused_exact_repeat_keeps_the_point_and_end_it_resumed():
    dec, source, _, prefix, snap, _ = decoder(room=False)
    dec.free, dec.keep = [], 8
    dec.kept = dec.kept[:1]
    _with_end(dec, prefix, source)
    with pytest.raises(NoRoom):
        dec.admit(Stream(prefix + [7], 8))
    assert [(k[0], k[1], k[2]) for k in dec.kept] == [(prefix, source, snap)] and "prompt_end" in snap


def test_an_in_place_resume_through_the_point_drops_the_end_it_rewrites():
    dec, source, _, prefix, snap, _ = decoder()
    dec.free = []
    dec.kept = dec.kept[:1]
    _with_end(dec, prefix, source)
    st, resume, cached = dec._slot_for(prefix + [8], True)      # computes row len(prefix) with another token
    assert st is source and cached == len(prefix) and resume == {"state": snap, "tail": "tail"}
    assert "prompt_end" not in snap
    st, resume, cached = dec._slot_for(prefix + [7], True)      # the old end's prompt now resumes the point
    assert cached == len(prefix) and "head" not in resume


def test_a_retention_policy_keeps_no_end_beyond_its_one_version_pool():
    dec, source, _, prefix, snap, _ = decoder()
    dec.retention_catalog = SimpleNamespace(policy=object())
    source.image_positions = None
    s = SimpleNamespace(prompt=prefix + [7], st=source, draft=True)
    import torch

    dec._keep_end(s, True, torch.zeros(1, 4), torch.zeros(1, 4))
    assert "prompt_end" not in snap
