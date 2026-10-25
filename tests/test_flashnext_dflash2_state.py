"""Whole-family CPU state/cut/clone integration; no TP8/V100 qualification."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorfold.families.qwen4_exp.cuda import forward
from tensorfold.families.qwen4_exp.cuda.dflash2_runtime import DFlash2Runtime
from tensorfold.families.qwen4_exp.cuda.dflash2_weights import DFlash2Plan
from tensorfold.families.qwen4_exp.cuda.service_memory import future_service
from tensorfold.families.qwen4_exp.cuda.state import Buffers, State

from test_dspark_taps_forward import cpu_weights
from test_flashnext_dflash2 import make_weights, small_config


@pytest.fixture
def family(monkeypatch, tmp_path):
    weights = cpu_weights(monkeypatch)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *_args: SimpleNamespace(multi_processor_count=80))
    raw = small_config()
    raw.update(hidden_size=weights.cfg.hidden, vocab_size=weights.cfg.vocab,
               num_target_layers=weights.cfg.layers, head_dim=32)
    raw["dflash_config"].update(target_layer_ids=[0, 1], mask_token_id=weights.cfg.vocab - 1)
    _config, _source = make_weights(tmp_path, raw=raw)
    runtime = DFlash2Runtime(weights, DFlash2Plan.read(tmp_path, tp=(0, 1)), context=64)
    yield weights, runtime
    for ref in runtime._states:
        window = ref()
        if window is not None and not window.closed:
            if window.pending is not None:
                window.pending.frame.readers -= 1
                window.pending = None
            window.closed = True
            window.kv = None
    runtime.close()


def prefill(weights, state, buffers, tokens, *, cut=None, commit=True):
    count = len(tokens)
    buffers.ids[:count].copy_(torch.tensor(tokens, dtype=torch.int32))
    buffers.ple_v.fill_(.125)
    state.ple_last = (state.ple_history.copy(), np.asarray(tokens, dtype=np.int64))
    forward.compute(weights, [(state, 0, count)], buffers, cuts=() if cut is None else (cut,))
    snapshot = None if cut is None else forward.cut_snapshot(weights, state, buffers, cut, 0)
    if commit:
        forward.commit(weights, state, buffers, count, count)
    return snapshot


def assert_window_equal(a, b):
    assert (a.end, a.available) == (b.end, b.available)
    for index in range(a.config.layers):
        left, positions = a.layer(index)
        right, other_positions = b.layer(index)
        assert torch.equal(positions, other_positions) and torch.equal(left, right)


def test_partial_real_multi_constructor_keeps_windows_until_fenced_retirement(family, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import multi

    weights, runtime = family
    original, acquired = multi.State, []
    failure = RuntimeError("second actual slot constructor interrupted after allocation")

    def acquire(*args, **kwargs):
        state = original(*args, **kwargs)
        acquired.append(state)
        if len(acquired) == 2:
            raise failure
        return state

    monkeypatch.setattr(multi, "State", acquire)
    # The regular CPU fixture runs the actual Buffer constructors. HC native
    # plan preparation has a separate supported-target gate.
    monkeypatch.setattr("tensorfold.families.qwen4_exp.cuda.hc_plans.enroll", lambda *a, **kw: None)
    with pytest.raises(RuntimeError) as caught:
        multi.MultiDecoder(weights, slots=2, capacity=32, depth=3, prefill_rows=8, kv_dtype="fp16")
    assert caught.value is failure and len(acquired) == 2
    assert all(not state.dflash2.closed and state.dflash2.kv is not None for state in acquired)
    with pytest.raises(ValueError, match="fence"):
        runtime.retire_after_fence(fenced=False)
    runtime.retire_after_fence(fenced=True)  # CPU has no asynchronous device readers.
    assert all(state.dflash2.closed and state.dflash2.kv is None for state in acquired)
    assert all(ref() is None or ref().readers == 0 for ref in runtime._frames)


def test_fenced_retirement_refuses_unknown_reader_and_retains_poisoned_ring(family):
    weights, runtime = family
    state = State(weights, 32, 8, "fp16", limit=64)
    buffers = Buffers(weights, 8, 32, prefill=True)
    buffers.dflash2_features.readers = 1
    with pytest.raises(RuntimeError, match="unowned feature reader"):
        runtime.retire_after_fence(fenced=True)
    assert state.dflash2.poisoned and not state.dflash2.closed and state.dflash2.kv is not None
    buffers.dflash2_features.readers = 0  # independent test borrower retires.
    runtime.retire_after_fence(fenced=True)


def test_serving_resource_plan_matches_actual_owned_ring_and_feature_extents(family):
    weights, runtime = family
    states = [State(weights, 32, 4, "fp16", limit=64, slot=index) for index in range(2)]
    main, prompt = Buffers(weights, 4, 32), Buffers(weights, 8, 32, prefill=True)
    resource = runtime.plan.serving_resources(slots=2, context=64, execution_rows=4,
        prefill_rows=8, keep=1, points=4, target_vocab=weights.cfg.vocab)
    assert resource["state_bytes"] == sum(state.dflash2.nbytes for state in states)
    assert resource["feature_frame_bytes"] == main.dflash2_features.nbytes + prompt.dflash2_features.nbytes
    assert resource["snapshot_versions"] == 9
    assert resource["target_shared_weight_bytes"] == 0
    assert resource["target_probability_device_bytes"] > 0 and resource["host_serving_bytes"] > 0
    for state in states:
        runtime.close_state(state)


def test_real_prefill_coordinator_batches_conditioning_before_snapshots(family, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    weights, runtime = family
    states = [State(weights, 32, 4, "fp16", limit=64, slot=index) for index in range(2)]
    buffers = Buffers(weights, 8, 32, prefill=True)
    buffers.ids[:2].copy_(torch.tensor([4, 7], dtype=torch.int32))
    buffers.ple_v.fill_(.125)
    for state, token in zip(states, [4, 7]):
        state.ple_last = (state.ple_history.copy(), np.array([token], dtype=np.int64))
    segments = [(state, index, index + 1) for index, state in enumerate(states)]
    forward.compute(weights, segments, buffers)
    decoder = object.__new__(MultiDecoder)
    decoder.w, decoder.pbuf, decoder.controller = weights, buffers, None
    decoder._points = lambda *_args: []
    pieces = [(SimpleNamespace(st=state, sid=index, prompt=[token]), 0, 1)
              for index, (state, token) in enumerate(zip(states, [4, 7]))]
    decoder.fills = {index: [None, False, 0, [], None] for index in range(2)}
    calls, original = [], runtime.provider.materialize

    def materialize(features, positions):
        calls.append((len(features), positions.tolist()))
        return original(features, positions)

    monkeypatch.setattr(runtime.provider, "materialize", materialize)
    decoder._absorb(pieces, segments)
    assert calls == [(2, [0, 0])]
    assert all(state.pos == state.dflash2.end == 1 for state in states)
    assert buffers.dflash2_features.readers == 0
    for state in states:
        runtime.close_state(state)


def test_quiet_serial_target_skips_all_draft_capture_and_marks_snapshot_inactive(family, monkeypatch):
    weights, runtime = family
    state = State(weights, 32, 4, "fp16", limit=64)
    runtime.set_enabled(state, False)
    buffers = Buffers(weights, 8, 32, prefill=True)
    before = buffers.dflash2_features
    before.features.fill_(.375)
    monkeypatch.setattr(before, "begin", lambda *_a: pytest.fail("serial request captured draft features"))
    monkeypatch.setattr(runtime.provider, "materialize", lambda *_a: pytest.fail("serial request conditioned draft KV"))
    prefill(weights, state, buffers, [4, 7])
    assert state.pos == 2 and state.dflash2.end == 0 and not state.dflash2.enabled
    assert torch.all(before.features == .375) and before.readers == 0
    snapshot = state.snapshot()
    assert snapshot["dflash2"]["schema"] == "dflash2-inactive-v1"
    assert "kv" not in snapshot["dflash2"] and runtime.record(state)["draft_valid_length"] is None
    clone = state.clone()
    assert not clone.dflash2.enabled and clone.pos == 2
    fresh = State(weights, 32, 4, "fp16", limit=64)
    with pytest.raises(ValueError, match="inactive"):
        fresh.restore(snapshot)
    assert fresh.pos == 0
    with pytest.raises(ValueError, match="frontier"):
        runtime.set_enabled(state, True)
    for owner in (state, clone, fresh):
        runtime.close_state(owner)


def test_mixed_capture_touches_only_enabled_rows_and_target_results_stay_equal(family):
    weights, runtime = family
    states = [State(weights, 32, 4, "fp16", limit=64, slot=index) for index in range(2)]
    runtime.set_enabled(states[0], False)
    buffers = Buffers(weights, 8, 32, prefill=True)
    buffers.dflash2_features.features.fill_(.375)
    buffers.dflash2_features.completed.fill_(.25)
    buffers.ids[:2].copy_(torch.tensor([4, 7], dtype=torch.int32))
    buffers.ple_v.fill_(.125)
    segments = [(state, index, index + 1) for index, state in enumerate(states)]
    for state, token in zip(states, [4, 7]):
        state.ple_last = (state.ple_history.copy(), np.array([token], dtype=np.int64))
    logits = forward.compute(weights, segments, buffers).clone()
    assert torch.all(buffers.dflash2_features.features[0] == .375)
    assert torch.all(buffers.dflash2_features.completed[0] == .25)
    assert not torch.all(buffers.dflash2_features.features[1] == .375)
    assert states[0].dflash2.pending is None and states[1].dflash2.pending is not None
    for state, begin, end in segments:
        forward.commit(weights, state, buffers, end - begin, end - begin, at=begin)
    assert states[0].pos == states[1].pos == states[1].dflash2.end == 1
    assert states[0].dflash2.end == 0 and buffers.dflash2_features.readers == 0
    # An independent all-serial target reference has no feature work; selected
    # target logits/recurrence/PLE remain exactly equal for these same rows.
    refs = [State(weights, 32, 4, "fp16", limit=64, slot=index + 2) for index in range(2)]
    ref_buffer = Buffers(weights, 8, 32, prefill=True)
    ref_buffer.ids[:2].copy_(torch.tensor([4, 7], dtype=torch.int32))
    ref_buffer.ple_v.fill_(.125)
    for state, token in zip(refs, [4, 7]):
        runtime.set_enabled(state, False)
        state.ple_last = (state.ple_history.copy(), np.array([token], dtype=np.int64))
    ref_segments = [(state, index, index + 1) for index, state in enumerate(refs)]
    assert torch.equal(logits, forward.compute(weights, ref_segments, ref_buffer))
    for (state, begin, end), original in zip(ref_segments, states):
        forward.commit(weights, state, ref_buffer, end - begin, end - begin, at=begin)
        assert torch.equal(state.rec[state.cur[0]], original.rec[original.cur[0]])
        assert torch.equal(state.conv, original.conv) and torch.equal(state.ple_tail, original.ple_tail)
    for state in (*states, *refs):
        runtime.close_state(state)


def test_reconstruction_materialization_failure_retains_actual_pending_lease_until_fence(family, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    weights, runtime = family
    state = State(weights, 32, 4, "fp16", limit=64)
    prefill(weights, state, Buffers(weights, 8, 32, prefill=True), [4, 7])
    buffers = Buffers(weights, 4, 32)
    buffers.ids[0] = 12  # already-published canonical pending anchor, never sampled here
    buffers.ple_v.fill_(.125)
    state.ple_last = (state.ple_history.copy(), np.array([12], dtype=np.int64))
    forward.compute(weights, [(state, 0, 1)], buffers)
    forward.commit(weights, state, buffers, 1, 1, commit_dflash=False)
    decoder = object.__new__(MultiDecoder)
    decoder.w, decoder.controller = weights, None
    primary = RuntimeError("teacher-forced feature materialization interrupted")

    def failed(*_args):
        raise primary

    monkeypatch.setattr(runtime.provider, "materialize", failed)
    with pytest.raises(RuntimeError) as caught:
        decoder._commit_dflash([(state, 1)])
    assert caught.value is primary
    assert state.pos == 3 and state.dflash2.end == 2 and state.dflash2.poisoned
    assert state.dflash2.pending is not None and buffers.dflash2_features.readers == 1
    runtime.retire_after_fence(fenced=True)  # CPU has no asynchronous readers; owner retained above.
    assert state.dflash2.closed and state.dflash2.kv is None and buffers.dflash2_features.readers == 0


def test_real_target_commit_interior_cut_restore_clone_and_fork_preserve_bounded_conditioning(family):
    w, runtime = family
    source = State(w, 32, 8, "fp16", limit=64)
    pbuf = Buffers(w, 8, 32, prefill=True)
    assert pbuf.dflash2_features is not None
    cut = forward.Cut(2)
    cut_state = prefill(w, source, pbuf, [4, 7, 19], cut=cut)
    assert cut_state["pos"] == cut_state["dflash2"]["end"] == 2
    assert source.pos == source.dflash2.end == 3 and source.dflash2.available == 3
    assert pbuf.dflash2_features.readers == 0
    clone = source.clone()
    assert clone.dflash2 is not source.dflash2 and clone.dflash2.kv.data_ptr() != source.dflash2.kv.data_ptr()
    assert_window_equal(source.dflash2, clone.dflash2)
    # Advance beyond the saved cut's whole draft window. Target caches still
    # hold the prefix; draft restoration must use the saved bounded payload,
    # never try to reconstruct retired features from current target KV.
    prefill(w, source, pbuf, [23, 29, 31, 37, 41, 43])
    prefill(w, source, pbuf, [47, 53, 59, 61, 67, 71])
    assert source.pos == 15 and source.dflash2.end - source.dflash2.available > cut_state["pos"]
    with pytest.raises(ValueError, match="no longer available"):
        runtime.snapshot(source, cut_state["pos"])
    fork = State(w, 32, 8, "fp16", limit=64, slot=1)
    fork.begin_sequence_generation()
    fork.copy_prefix(source, 2, 0)
    assert fork.pos == fork.dflash2.end == 0  # raw target copy does not guess old draft features
    fork.restore(cut_state)
    reference = State(w, 32, 8, "fp16", limit=64, slot=2)
    prefill(w, reference, Buffers(w, 8, 32, prefill=True), [4, 7])
    assert fork.pos == reference.pos == 2
    assert_window_equal(fork.dflash2, reference.dflash2)
    assert torch.equal(fork.rec[fork.cur[0]], reference.rec[reference.cur[0]])
    for name in ("conv", "ple_tail"):
        assert torch.equal(getattr(fork, name), getattr(reference, name))
    assert np.array_equal(fork.ple_history, reference.ple_history)
    assert fork.slot_generation == fork.dflash2.generation == 1
    for state in (source, clone, fork, reference):
        runtime.close_state(state)


def test_snapshot_profile_and_tensor_shape_validation_precede_any_target_writes(family, monkeypatch):
    w, runtime = family
    source = State(w, 32, 8, "fp16", limit=64)
    prefill(w, source, Buffers(w, 8, 32, prefill=True), [4, 7])
    snapshot = source.snapshot()
    destination = State(w, 32, 8, "fp16", limit=64, slot=1)
    wrong = {**snapshot, "dflash2": {**snapshot["dflash2"], "profile": "wrong-runtime"}}
    writes, original = [], torch.Tensor.copy_

    def copy_tensor(self, *args, **kwargs):
        writes.append(self)
        return original(self, *args, **kwargs)

    with monkeypatch.context() as scope:
        scope.setattr(torch.Tensor, "copy_", copy_tensor)
        with pytest.raises(ValueError, match="profile"):
            destination.restore(wrong)
    assert not writes and destination.pos == destination.dflash2.end == 0
    assert not destination.dflash2.poisoned
    runtime.close_state(source)
    runtime.close_state(destination)


def test_ring_restore_interruption_poisoning_stops_whole_state_and_reset_recovers(family, monkeypatch):
    w, runtime = family
    state = State(w, 32, 8, "fp16", limit=64)
    prefill(w, state, Buffers(w, 8, 32, prefill=True), [4, 7])
    snapshot = state.snapshot()
    failed = {**snapshot, "dflash2": {**snapshot["dflash2"], "kv": snapshot["dflash2"]["kv"] + 1}}
    original = torch.Tensor.index_copy_
    primary = RuntimeError("ring write interrupted after mutation")

    def interrupted(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if self is state.dflash2.kv:
            raise primary
        return result

    with monkeypatch.context() as scope:
        scope.setattr(torch.Tensor, "index_copy_", interrupted)
        with pytest.raises(RuntimeError) as caught:
            state.restore(failed)
    assert caught.value is primary and state.dflash2.poisoned
    with pytest.raises(ValueError, match="poisoned"):
        state.kv_check()
    with pytest.raises(ValueError, match="poisoned"):
        runtime.snapshot(state)
    generation = state.slot_generation
    state.reset(w)
    assert state.pos == state.dflash2.end == 0 and not state.dflash2.poisoned
    assert state.slot_generation == state.dflash2.generation == generation + 1
    runtime.close_state(state)


def test_discard_releases_pending_feature_lease_even_when_conditioning_is_poisoned(family):
    w, runtime = family
    state = State(w, 32, 8, "fp16", limit=64)
    pbuf = Buffers(w, 8, 32, prefill=True)
    prefill(w, state, pbuf, [4, 7], commit=False)
    assert pbuf.dflash2_features.readers == 1
    state.dflash2.poisoned = True
    with pytest.raises(ValueError, match="poisoned"):
        state.discard_target_tentative()
    assert pbuf.dflash2_features.readers == 0 and state.dflash2.pending is None
    state.reset(w)
    runtime.close_state(state)


def test_fixed_window_accounting_and_actual_snapshot_version_budget(family):
    w, runtime = family
    state = State(w, 32, 8, "fp16", limit=64)
    target_bytes = sum(cache.nbytes for cache in state.kc) + sum(value.nbytes for value in (*state.ikc, *state.pooled))
    assert state.cache_bytes() == state.owned_cache_bytes() == target_bytes + state.dflash2.nbytes
    before = state.dflash2.nbytes
    state.resize(48)
    assert state.dflash2.nbytes == before
    versions = 8 + 4
    future = future_service(w, [state], streams=1, retained=versions, growth_bytes=0)
    each = (state.rec[0].numel() * 4 + state.conv.numel() * 2 + state.ple_tail.numel() * 2 + before)
    assert future["snapshot_bytes_each"] == each and future["snapshot_bytes"] == versions * each
    runtime.close_state(state)


def test_real_zero_proposal_decode_commit_keeps_conditioning_ready_for_reactivation(family):
    w, runtime = family
    state = State(w, 32, 8, "fp16", limit=64)
    prefill(w, state, Buffers(w, 8, 32, prefill=True), [4, 7, 19])
    buf = Buffers(w, 4, 32)
    buf.ids[0] = 23
    buf.ple_v.fill_(.125)
    state.ple_last = (state.ple_history.copy(), np.array([23], dtype=np.int64))
    forward.compute(w, [(state, 0, 1)], buf)
    forward.commit(w, state, buf, 1, 1)
    assert state.pos == state.dflash2.end == 4 and buf.dflash2_features.readers == 0
    packet = runtime.propose(state, 29, 0, None, 5)
    assert packet.tokens == ()
    warm = runtime.propose(state, 29, 1, None, 5)
    assert len(warm.tokens) == 1 and state.dflash2.end == 4
    runtime.close_state(state)


def test_deferred_draft_commit_batches_one_shared_feature_frame(family, monkeypatch):
    w, runtime = family
    a, b = [State(w, 32, 8, "fp16", limit=64, slot=i) for i in range(2)]
    buf = Buffers(w, 8, 32, prefill=True)
    buf.ids[:2].copy_(torch.tensor([4, 7], dtype=torch.int32))
    buf.ple_v.fill_(.125)
    for state, token in ((a, 4), (b, 7)):
        state.ple_last = (state.ple_history.copy(), np.array([token], dtype=np.int64))
    forward.compute(w, [(a, 0, 1), (b, 1, 2)], buf)
    forward.commit(w, a, buf, 1, 1, at=0, commit_dflash=False)
    forward.commit(w, b, buf, 1, 1, at=1, commit_dflash=False)
    assert a.pos == b.pos == 1 and a.dflash2.end == b.dflash2.end == 0
    assert buf.dflash2_features.readers == 2
    # A full-point snapshot can reconstruct the committed target frontier
    # while the coordinator still owns the pending batch lease.
    snap = a.snapshot()
    assert snap["dflash2"]["end"] == 1
    calls, original = [], runtime.provider.materialize

    def materialize(features, positions):
        calls.append((len(features), positions.tolist()))
        return original(features, positions)

    monkeypatch.setattr(runtime.provider, "materialize", materialize)
    runtime.commit_many([(a, 1), (b, 1)])
    assert calls == [(2, [0, 0])]
    assert a.dflash2.end == b.dflash2.end == 1 and buf.dflash2_features.readers == 0
    runtime.close_state(a)
    runtime.close_state(b)


def test_reference_selects_raw_topk_before_saturating_unary_transform(family):
    _w, runtime = family
    provider = runtime.provider
    provider.config = replace(provider.config, top_k=2, logit_softcap=1.0)
    logits = torch.full((1, provider.config.vocab), -100.)
    logits[0, :3] = torch.tensor([100., 200., 300.])
    provider.head = lambda _hidden: logits
    ids, values = provider.candidates(torch.zeros(1, provider.config.hidden, dtype=torch.float16))
    # This is the pinned SGLang compute_candidates ordering. All transformed
    # values tie, but token0 never enters the raw-top2 support.
    assert ids.tolist() == [[2, 1]] and values.tolist() == [[1., 1.]]


def test_dflash_finish_omits_unused_candidate_collective_and_ordinary_finish_retains_it(monkeypatch):
    calls = []
    cfg = SimpleNamespace(streams=4, hidden=256, eps=1e-6, low=32)
    weights = SimpleNamespace(cfg=cfg, head=object(), comm=object(), meta={"vocab_offset": 0}, dflash2=object())
    buffers = SimpleNamespace(h=torch.zeros(2, 1024, dtype=torch.float16), streams=torch.empty(2, 1024, dtype=torch.float16),
                              prefill=False, mixed=torch.zeros(2, 256, dtype=torch.float16),
                              xs_mixed=torch.zeros(2, 8), logits=torch.zeros(2, 16))
    monkeypatch.setattr(forward, "_writeback", lambda *_args: None)
    monkeypatch.setattr(forward, "_readout", lambda *_args: None)
    monkeypatch.setattr(forward, "_mm", lambda _x, _q, _xs, out, _b: out)
    monkeypatch.setattr(forward, "candidates", lambda *_args, **_kwargs: calls.append("candidate-collective"))
    assert forward.finish(weights, None, buffers, 2, pending=None) is not None and not calls
    weights.dflash2 = None
    assert forward.finish(weights, None, buffers, 2, pending=None) is not None
    assert calls == ["candidate-collective"]
def test_new_sequence_retires_native_and_draft_readers_before_generation_publication(monkeypatch):
    import weakref
    from types import SimpleNamespace
    from tensorfold.families.qwen4_exp.cuda.state import State

    events = []
    target = object.__new__(State)
    target._resize_error = target._kv_error = target._history_error = None
    target.slot_generation = 4
    target.dflash2 = SimpleNamespace(generation=4)

    class Reader:
        def wait_native_reuse(self):
            events.append("reader")

    class Words:
        is_cuda = True
        device = "cuda:0"

        def __getitem__(self, index):
            assert index == 0
            return self

        def fill_(self, generation):
            assert target.slot_generation == 4 and target.dflash2.generation == generation == 5
            events.append("word")

    def discard(state):
        assert state is target and state.dflash2.generation == 4
        events.append("draft")

    reader = Reader()
    target.history_native_readers = weakref.WeakSet([reader])
    target.history_generations = Words()
    target.dflash2_runtime = SimpleNamespace(discard=discard)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: SimpleNamespace(
        synchronize=lambda: events.append("fence")))
    assert target.begin_sequence_generation() == 5
    assert events == ["reader", "fence", "draft", "word"]
