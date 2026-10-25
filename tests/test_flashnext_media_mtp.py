"""Actual ready-span leases and independent bounded HC-boundary snapshots on CPU."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import numpy as np
import sys
from types import ModuleType

from tensorfold.families.qwen4_exp.cuda import media_mtp
from tensorfold.families.qwen4_exp.cuda.media_inputs import ReadyInputSpan
from test_flashnext_mtp_inputs import family as actual_mtp_family  # noqa: F401 (pytest fixture registration)


PROFILE, MEDIA = "a" * 64, "b" * 64


def state():
    return SimpleNamespace(media_input_span=None, media_profile_identity=None, media_identity=None,
        media_hidden_size=4, mtp_hidden_width=16, execution_dtype=torch.float16,
        slot_generation=3, pos=0, pos_dev=torch.tensor(0, dtype=torch.int32), limit=1048576,
        mtp_len=0, mtp_drafted=0, mtp_boundary=None, rope_delta=0,
        media_mtp_enabled=True,
        _history_error=None, _kv_error=None, set_rope_delta=lambda _value: None)


def span():
    return ReadyInputSpan(start=0, token_ids=[1, 31, 31], values=torch.arange(12).reshape(3, 4).half(),
        positions=torch.tensor([[0, 0, 0], [1, 2, 3], [1, 4, 5]], dtype=torch.int32), rope_delta=-1,
        profile_identity=PROFILE, generation=3, media_identity=MEDIA, max_rows=3,
        hidden_size=4, vocab_size=32, roles=("target", "mtp"))


def test_actual_target_and_shifted_mtp_consumers_retire_only_after_staging():
    st, ready = state(), span()
    media_mtp.attach_span(st, ready, profile_identity=PROFILE, media_identity=MEDIA)
    target = torch.empty_like(ready.values)
    with media_mtp.borrow_span(st, "target") as lease:
        target.copy_(lease.rows(0, [1, 31, 31]))
    next_inputs = torch.empty((2, 4), dtype=torch.float16)
    with media_mtp.borrow_span(st, "mtp") as lease:
        next_inputs.copy_(lease.rows(1, [31, 31]))
        # Both slices belong to the same fixed-role lease. Coordinates describe
        # their input rows, not the MTP hidden/cache row one position earlier.
        assert torch.equal(lease.rows(2, [31]), ready.values[2:3])
    assert torch.equal(target[1:], next_inputs) and not torch.equal(next_inputs[0], next_inputs[1])
    media_mtp.retire_span(st, cancelled=False)
    assert ready.closed and st.media_input_span is None and ready.values is None


def test_staging_input_refusal_preserves_primary_and_completes_known_unused_role():
    st, ready = state(), span()
    media_mtp.attach_span(st, ready, profile_identity=PROFILE, media_identity=MEDIA)
    primary = ValueError("input refused before model execution")
    with pytest.raises(ValueError) as caught:
        with media_mtp.borrow_span(st, "target"):
            raise primary
    assert caught.value is primary
    media_mtp.retire_span(st, cancelled=True)  # MTP was never borrowed, not a fabricated completion.
    assert ready.closed


def test_staging_completion_failure_retains_poisoned_request_owner(monkeypatch):
    st, ready = state(), span()
    media_mtp.attach_span(st, ready, profile_identity=PROFILE, media_identity=MEDIA)
    primary = RuntimeError("staging completion failed")

    def failed():
        raise primary

    with pytest.raises(RuntimeError) as caught:
        with media_mtp.borrow_span(st, "target") as lease:
            monkeypatch.setattr(lease, "complete", failed)
    assert caught.value is primary and st._history_error is primary and st.media_input_span is ready
    assert ready.values is not None and not ready.closed


def test_one_boundary_keeps_complete_hc_separate_coordinates_and_independent_snapshot():
    st = state()
    media_mtp.bind_context(st, PROFILE, MEDIA)
    hidden = torch.arange(16).reshape(1, 16).half()
    coordinates = torch.tensor([[17, 4, 9]], dtype=torch.int32)
    st.pos, st.mtp_len = 3, 2
    st.mtp_boundary = media_mtp.capture_boundary(st, 3, hidden, coordinates)
    assert st.mtp_boundary.row == 2 and st.mtp_boundary.generation == 3
    assert st.mtp_boundary.nbytes == 16 * 2 + 3 * 8
    assert st.mtp_boundary.hidden.data_ptr() != hidden.data_ptr()
    payload = media_mtp.snapshot_state(st)["media_state"]
    assert payload["mtp_boundary"]["hidden"].data_ptr() != st.mtp_boundary.hidden.data_ptr()
    torch.testing.assert_close(payload["mtp_boundary"]["hidden"], hidden, atol=0, rtol=0)
    assert torch.equal(payload["mtp_boundary"]["coordinates"], coordinates.long())
    st.slot_generation = 4
    media_mtp.validate_snapshot(st, payload, 3, 2)
    media_mtp.restore_state(st, payload)
    assert st.mtp_boundary.generation == 4
    assert st.mtp_boundary.hidden.data_ptr() != payload["mtp_boundary"]["hidden"].data_ptr()
    budget = media_mtp.boundary_resources(hidden_width=16, slots=2, keep=8, points=4)
    assert budget["protected_versions"] == 16
    assert budget["live_hc_bytes"] + budget["live_coordinate_bytes"] == 2 * st.mtp_boundary.nbytes
    assert budget["protected_coordinate_bytes"] == 16 * payload["mtp_boundary"]["coordinates"].numel() * 8


@pytest.mark.parametrize("mutation", ["profile", "media", "frontier", "shape", "generation"])
def test_boundary_identity_and_frontier_mismatch_refuses_before_copy(mutation, monkeypatch):
    st = state()
    media_mtp.bind_context(st, PROFILE, MEDIA)
    st.pos, st.mtp_len = 3, 2
    st.mtp_boundary = media_mtp.capture_boundary(st, 3, torch.ones((1, 16)).half(), torch.ones((1, 3)).long())
    payload = media_mtp.snapshot_state(st)["media_state"]
    if mutation in ("profile", "media"):
        payload[mutation] = "c" * 64
    elif mutation == "shape":
        payload["mtp_boundary"]["hidden"] = payload["mtp_boundary"]["hidden"][:, :-1]
    elif mutation == "generation":
        st.mtp_boundary = replace(st.mtp_boundary, generation=2)
    monkeypatch.setattr(torch.Tensor, "clone", lambda *_a, **_kw: pytest.fail("validation copied tensor data"))
    with pytest.raises(ValueError):
        if mutation == "generation":
            media_mtp.snapshot_state(st)
        else:
            media_mtp.validate_snapshot(st, payload, 3, 1 if mutation == "frontier" else 2)


def test_zero_cut_and_resolved_frontier_have_no_deferred_hc():
    st = state()
    media_mtp.bind_context(st, PROFILE, None)
    assert media_mtp.capture_boundary(st, 0, None, None) is None
    assert media_mtp.snapshot_state(st)["media_state"]["mtp_boundary"] is None
    st.pos = st.mtp_len = 524288  # metadata uses admitted rows, not a hardcoded native-reference ceiling.
    assert media_mtp.snapshot_state(st)["media_state"]["mtp_boundary"] is None


def test_real_state_media_boundary_clone_restore_and_prewrite_identity_guard(request, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import forward, mtp
    from tensorfold.families.qwen4_exp.cuda.state import Buffers, State

    weights = request.getfixturevalue("actual_mtp_family")
    st = State(weights, 32, 4, "fp16", limit=32)
    assert "media_state" not in st.snapshot()  # legacy None path remains its original schema
    buf = Buffers(weights, 4, 32, prefill=True)
    buf.staged = SimpleNamespace(synchronize=lambda: None, record=lambda: None)
    tokens = [4, 17, 31]
    buf.ids[:3].copy_(torch.tensor(tokens, dtype=torch.int32))
    buf.ple_v.fill_(.125)
    st.ple_last = (st.ple_history.copy(), np.asarray(tokens, dtype=np.int64))
    forward.compute(weights, [(st, 0, 3)], buf)
    hc = buf.streams[:3].clone()
    forward.commit(weights, st, buf, 3, 3)
    actual = weights.embed[0].lookup(torch.tensor(tokens[1:])).clone()
    mtp.mtp_forward(weights, st, buf, tokens[1:], hc[:2], logits=False, next_inputs=actual)
    st.set_mtp_len(2)
    st.bind_media_context(PROFILE, None)  # kernel math above is the legacy-text reference; lifecycle below is explicit.
    st.mtp_boundary = media_mtp.capture_boundary(st, 3, hc[2:3], torch.tensor([[2, 2, 2]], dtype=torch.int64))
    snap = st.snapshot()
    clone = st.clone()
    assert clone.media_input_span is None
    assert clone.mtp_boundary.hidden.data_ptr() != st.mtp_boundary.hidden.data_ptr()
    assert torch.equal(clone.mtp_boundary.hidden, st.mtp_boundary.hidden)
    wrong = {**snap, "media_state": {**snap["media_state"], "profile": "c" * 64}}
    with monkeypatch.context() as scope:
        scope.setattr(torch.Tensor, "copy_", lambda *_a, **_kw: pytest.fail("invalid identity reached State writes"))
        with pytest.raises(ValueError, match="identity"):
            st.restore(wrong)
    st.reset(weights)
    assert st.mtp_boundary is None and st.media_profile_identity is None
    st.bind_media_context(PROFILE, None)
    st.restore(snap)
    assert st.pos == 3 and st.mtp_len == 2 and st.mtp_boundary.row == 2
    assert st.mtp_boundary.generation == st.slot_generation == 1


def test_real_state_reset_retires_actual_span_and_clears_request_context(request):
    from tensorfold.families.qwen4_exp.cuda.state import State

    weights = request.getfixturevalue("actual_mtp_family")
    st = State(weights, 32, 4, "fp16", limit=32)
    ready = ReadyInputSpan(start=0, token_ids=[4], values=weights.embed[0].lookup(torch.tensor([4])).clone(),
        positions=torch.zeros((1, 3), dtype=torch.int32), rope_delta=0, profile_identity=PROFILE,
        generation=st.slot_generation, media_identity=MEDIA, max_rows=2, hidden_size=weights.cfg.hidden,
        vocab_size=weights.cfg.vocab, roles=("target", "mtp"))
    st.attach_media_input_span(ready, profile_identity=PROFILE, media_identity=MEDIA)
    with media_mtp.borrow_span(st, "target") as lease:
        assert lease.rows(0, [4]).shape == (1, weights.cfg.hidden)
    st.reset(weights)
    assert ready.closed and st.media_input_span is None and st.media_profile_identity is None
    assert st.media_identity is None and st.mtp_boundary is None


def test_actual_ready_span_deferred_visual_lookahead_and_final_text_anchor(request, monkeypatch):
    """Real target/MTP CPU math; compact transport is an observation boundary.

    The tiny full coordinate table is a legacy oracle. This does not qualify
    the separate compact-owner implementation or actual SC117 media kernels.
    """
    from tensorfold.families.qwen4_exp.cuda import forward, mtp
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, absorb
    from tensorfold.families.qwen4_exp.cuda.state import Buffers, State

    weights = request.getfixturevalue("actual_mtp_family")
    st, buf = State(weights, 32, 4, "fp16", limit=32), Buffers(weights, 4, 32, prefill=True)
    buf.staged = SimpleNamespace(synchronize=lambda: None, record=lambda: None)
    buf.mtp_indexer_coordinate_storage = torch.empty((4, 3), dtype=torch.int64)
    buf.compact_indexer_stage = SimpleNamespace(wait=lambda **_kw: None)  # no asynchronous CPU transport
    seen = []
    compact = ModuleType("tensorfold.families.qwen4_exp.cuda.compact_indexer")

    def stage_coordinates(buffer, segments, coordinates):
        seen.append(coordinates.clone() if coordinates is not None else None)
        buffer.mtp_indexer_coordinates = coordinates

    compact.stage_mtp_indexer_frame = stage_coordinates
    compact.resolve_indexer_coordinates = lambda state, start, rows: torch.arange(start, start + rows)[:, None].repeat(1, 3)
    monkeypatch.setitem(sys.modules, compact.__name__, compact)
    prompt = [4, 31, 31, 31]
    values = torch.stack([weights.embed[0].lookup(torch.tensor([4]))[0],
                          torch.full((weights.cfg.hidden,), .125).half(),
                          torch.full((weights.cfg.hidden,), -.25).half(),
                          torch.linspace(-.5, .75, weights.cfg.hidden).half()])
    positions = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0]], dtype=torch.int32)
    st.image_positions = positions.clone()  # independent tiny legacy-coordinate oracle, not production storage
    original_embed, lookups = mtp._embed, []

    def text_only(weights, ids, copies, out):
        lookups.extend(ids.tolist())
        assert all(token != 31 for token in ids.tolist()), "visual placeholder reached token lookup"
        return original_embed(weights, ids, copies, out)

    monkeypatch.setattr(mtp, "_embed", text_only)
    cut_hidden = None
    for start, end in ((0, 2), (2, 4)):
        ready = ReadyInputSpan(start=start, token_ids=prompt[start:end], values=values[start:end].clone(),
            positions=positions[start:end].clone(), rope_delta=0, profile_identity=PROFILE,
            generation=st.slot_generation, media_identity=MEDIA, max_rows=3, hidden_size=weights.cfg.hidden,
            vocab_size=weights.cfg.vocab, roles=("target", "mtp"))
        st.attach_media_input_span(ready, profile_identity=PROFILE, media_identity=MEDIA)
        n = end - start
        buf.ids[:n].copy_(torch.tensor(prompt[start:end], dtype=torch.int32))
        buf.ple_v.fill_(.125)
        st.ple_last = (st.ple_history.copy(), np.asarray(prompt[start:end], dtype=np.int64))
        with media_mtp.borrow_span(st, "target") as lease:
            forward.compute(weights, [(st, 0, n)], buf, input_embeddings=lease.rows(start, prompt[start:end]))
        hidden = buf.streams[:n]
        if start:
            cut_hidden = hidden[:1].clone()
        forward.commit(weights, st, buf, n, n)
        result = media_mtp.condition_chunks(weights, buf, [media_mtp.CommittedMTPChunk(
            st, prompt, start, end, hidden, (3,) if start else ())])[0]
        assert st.mtp_len == end - 1 and result[0] is st.mtp_boundary and st.mtp_boundary.row == end - 1
        if start:
            assert torch.equal(result[1][3].hidden, cut_hidden)
            assert torch.equal(result[1][3].coordinates, positions[2:3].long())
        st.retire_media_input_span(cancelled=False)
    assert not lookups  # both visual chunks consumed actual next-position embeddings
    assert torch.equal(seen[0], positions[:1].long())
    assert torch.equal(seen[1], positions[1:3].long())  # h1,h2 coords, not embedding rows2,3
    engine = object.__new__(Engine)
    engine.w, engine.st, engine.mbuf, engine.graphs = weights, st, buf, None
    absorb(engine, st.mtp_boundary.hidden, [5])  # permitted sampled text anchor after complete prompt
    assert lookups == [5] and st.mtp_len == 4 and st.mtp_boundary is None
    assert torch.equal(seen[-1], positions[3:4].long())
