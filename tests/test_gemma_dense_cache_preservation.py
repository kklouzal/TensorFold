"""Dense Gemma cache ownership survives the separate MoE archive transition.

CPU source controls use the actual shared request sampler with a tiny target;
no checkpoint load, CUDA execution or serving-performance claim.
"""
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace as NS
import threading
import weakref
import gc

import pytest
import torch

from tensorfold.families.gemma4.cuda import e2b_engine


def bare_engine():
    engine = e2b_engine.E2BEngine.__new__(e2b_engine.E2BEngine)
    engine._owner = None
    engine._closed_lock = threading.Lock()
    engine._closed = False
    engine._run_lock = threading.Lock()
    engine._prefix_lock = threading.Lock()
    engine._prefix = None
    engine._prefix_error = None
    engine._retirement_failure = None
    engine._disposed = False
    engine._stream = NS(synchronize=lambda: None)
    engine._draft = engine.head = None
    return engine


def test_dense_capability_and_local_checkpoint_owner_are_distinct():
    engine = bare_engine()
    assert engine.supports_canonical_prefix_archive is False
    assert e2b_engine._Prefix.__module__ == e2b_engine.__name__
    with pytest.raises(ValueError):
        engine.cache_publication_intent("frontend")
    with pytest.raises(ValueError, match="dense Gemma"):
        engine.prefix_cache_identity([1])
    assert "_prefix_archive" not in vars(engine)


@pytest.mark.parametrize("draft", [False, True])
def test_dense_checkpoint_owns_actual_snapshot_protocol_and_detached_logits(draft):
    engine = bare_engine()
    target_owner, draft_owner = object(), object()
    calls = []
    state = NS(position=17, snapshot=lambda: calls.append("target") or target_owner)
    if draft:
        engine._draft = NS(snapshot=lambda: calls.append("draft") or draft_owner)
    logits = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    result = engine._checkpoint(state, logits)
    assert result.position == 17 and result.state is target_owner
    assert result.companion is (draft_owner if draft else None)
    assert calls == (["target", "draft"] if draft else ["target"])
    assert result.logits.data_ptr() != logits.data_ptr()
    assert torch.equal(result.logits, logits)
    logits.zero_()
    assert result.logits.tolist() == [list(range(8))]
    assert engine._prefix_error is None


def test_checkpoint_failure_is_sticky_until_positive_close():
    engine = bare_engine()
    primary = RuntimeError("companion snapshot failed")
    def failed():
        raise primary
    engine._draft = NS(snapshot=failed)
    state = NS(position=5, snapshot=lambda: object())
    with pytest.raises(RuntimeError) as caught:
        engine._checkpoint(state)
    assert caught.value is primary and engine._prefix_error is primary
    with pytest.raises(RuntimeError) as later:
        engine._check_open()
    assert later.value.__cause__ is primary


def test_dense_stats_are_one_real_prompt_and_do_not_invent_byte_measurements():
    engine = bare_engine()
    empty = engine.prefix_cache_stats()
    assert empty["entries"] == empty["tokens"] == 0
    engine._prefix = e2b_engine._Prefix((1, 2, 3), (), ())
    reading = engine.prefix_cache_stats()
    assert reading["entries"] == reading["capacity"] == reading["max_entries"] == 1
    assert reading["tokens"] == 3 and reading["bytes"] is None and reading["byte_budget"] is None
    assert reading["store_schema"] == "gemma-dense-turn-checkpoints-v1"
    assert "chunks" not in reading and "indexed_tokens" not in reading


def test_dense_turn_boundary_matches_actual_geometry_and_media_atomicity():
    engine = bare_engine()
    engine._turn = 99
    assert engine._turn_boundary([99, 1, 99, 2, 99, 3], 1, ()) == 4
    assert engine._turn_boundary([99, 1, 99, 2, 99, 3], 1, ((3, 5),)) == 0
    assert engine._turn_boundary([99, 1, 2], 0, ()) == 0
    engine._turn = None
    assert engine._turn_boundary([99, 1], 0, ()) == 0


def test_lookup_failure_retains_its_primary_and_stops_future_requests():
    engine = bare_engine()
    engine._turn = None
    primary = MemoryError("prefix comparison allocation")
    class Tokens:
        def __getitem__(self, key):
            raise primary
    engine._prefix = e2b_engine._Prefix(Tokens(), (), (NS(position=2, logits=object()),))
    with pytest.raises(MemoryError) as caught:
        engine._find_prefix([1, 2], ())
    assert caught.value is primary and engine._prefix_error is primary
    with pytest.raises(RuntimeError) as later:
        engine._check_open()
    assert later.value.__cause__ is primary


def test_clear_retires_native_consumers_before_releasing_real_dense_prefix():
    engine = bare_engine()
    events = []
    class Snapshot:
        def __del__(self):
            events.append("release")
    payload = Snapshot()
    owned = weakref.ref(payload)
    engine._prefix = e2b_engine._Prefix((1,), (), (e2b_engine._Checkpoint(1, payload, None, None),))
    del payload
    engine._stream = NS(synchronize=lambda: events.append("retire"))
    engine.clear_prefix_cache()
    gc.collect()
    assert owned() is None and events == ["retire", "release"]
    assert engine._prefix is None and not engine._closed


def test_failed_retirement_retains_payload_and_never_retries_native_retirement():
    engine = bare_engine()
    primary = RuntimeError("uncertain native stream retirement")
    attempts = []
    def failed():
        attempts.append(True)
        raise primary
    engine._stream = NS(synchronize=failed)
    payload = NS()
    prefix = e2b_engine._Prefix((1,), (), (e2b_engine._Checkpoint(1, payload, None, None),))
    engine._prefix = prefix
    engine._state = object()
    engine.target = NS(close=lambda: pytest.fail("target disposed without positive retirement"))
    with pytest.raises(RuntimeError) as caught:
        engine.clear_prefix_cache()
    assert caught.value is primary and engine._prefix is prefix
    with pytest.raises(RuntimeError) as later:
        engine.close()
    assert later.value.__cause__ is primary and len(attempts) == 1
    assert engine._prefix is prefix and engine._state is not None and not engine._disposed


def test_dense_shared_generate_never_executes_moe_response_publication(monkeypatch):
    from tensorfold.cuda import sampling as cuda_sampling
    engine = bare_engine()
    engine._assistant_sampled_q = False
    engine.context_window, engine.eos = 64, ()
    engine.target = NS(config=NS(vocab=8))
    engine._state = NS(position=2)
    engine._prefill = lambda *args, **kwargs: (engine._state, torch.ones(1, 8), 2)
    @contextmanager
    def lane(background):
        assert not background
        yield None
    engine._lane = lane
    engine._publish_canonical_locked = lambda *args, **kwargs: pytest.fail("dense state reached MoE archive")
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(cuda_sampling, "sample_rows", lambda *args: [3])
    emitted = []
    stats = engine.generate([1, 2], 1, None, lambda values: emitted.extend(values))
    assert emitted == [3] and stats["cached"] == 2 and stats["drafts"] is False
    assert "cache_publication" not in stats and "_prefix_archive" not in vars(engine)


def test_stats_planning_failure_keeps_ownership_failure_terminal():
    engine = bare_engine()
    primary = MemoryError("kept-token count allocation failed")
    class Tokens:
        def __len__(self):
            raise primary
    engine._prefix = e2b_engine._Prefix(Tokens(), (), ())
    with pytest.raises(MemoryError) as caught:
        engine.prefix_cache_stats()
    assert caught.value is primary and engine._prefix_error is primary
    with pytest.raises(RuntimeError) as later:
        engine._check_open()
    assert later.value.__cause__ is primary


def test_clear_from_active_callback_refuses_without_changing_retirement_authority():
    engine = bare_engine()
    prefix = e2b_engine._Prefix((1,), (), ())
    engine._prefix = prefix
    engine._owner = threading.get_ident()
    engine._stream = NS(synchronize=lambda: pytest.fail("callback tried native retirement"))
    with pytest.raises(RuntimeError, match="active request callback"):
        engine.clear_prefix_cache()
    assert engine._prefix is prefix and engine._retirement_failure is None and not engine._closed
