"""Actual ready-span leases and independent bounded HC-boundary snapshots on CPU."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import numpy as np

from tensorfold.families.qwen4_exp.cuda import media_mtp
from tensorfold.families.qwen4_exp.cuda.media_inputs import ReadyInputSpan
from tensorfold.families.qwen4_exp.cuda.position_coordinates import issue_coordinates
from test_qwen4_exp_position_coordinates import policy
from test_flashnext_mtp_inputs import family as actual_mtp_family  # noqa: F401 (pytest fixture registration)


PROFILE, MEDIA = "a" * 64, "b" * 64


def state():
    return SimpleNamespace(media_input_span=None, media_profile_identity=None, media_identity=None,
        coordinate_profile_identity=PROFILE,
        media_hidden_size=4, mtp_hidden_width=16, execution_dtype=torch.float16,
        slot_generation=3, pos=0, pos_dev=torch.tensor(0, dtype=torch.int32), limit=1048576,
        mtp_len=0, mtp_drafted=0, mtp_boundary=None, rope_delta=0,
        media_mtp_enabled=True, media_inputs_enabled=True,
        media_target_position=policy(limit=1048576), media_mtp_position=policy("mtp", limit=1048576),
        _history_error=None, _kv_error=None, set_rope_delta=lambda _value: None)


def issued(rows, *, start=0, generation=3, media=MEDIA, position=None):
    position = policy(limit=1048576) if position is None else position
    return issue_coordinates(torch.as_tensor(rows, dtype=torch.int64), position=position,
        profile_identity=PROFILE, generation=generation, media_identity=media, row_start=start,
        max_rows=max(1, len(rows)), device="cpu")


def bind_positions(weights):
    target = replace(policy(limit=1048576), inv_freq=weights.inv_freq)
    mtp = replace(policy("mtp", limit=1048576), inv_freq=weights.inv_freq)
    weights.positions = SimpleNamespace(target=target, mtp=mtp,
        profile=SimpleNamespace(identity=target.identity, epoch=target.epoch))
    weights.meta["coordinate_profile_identity"] = PROFILE



def bind_reference_inputs(weights, state, buffers=None):
    """Attach component inputs to the independent standalone CPU math oracle.

    Production bound State/Buffers require actual unique-history geometry. This
    tiny legacy oracle tests the consumer and proof handoff without pretending
    its replicated cache is that supported serving configuration.
    """
    from tensorfold.families.qwen4_exp.cuda.input_preparation import install_input_preparation

    bind_positions(weights)
    state.coordinate_profile_identity = PROFILE
    state.media_target_position = weights.positions.target
    state.media_mtp_position = weights.positions.mtp
    state.position_profile_identity = weights.positions.profile.identity
    state.position_epoch = weights.positions.profile.epoch
    context = install_input_preparation(weights)
    state.input_preparation_context = context
    if buffers is not None:
        buffers.input_preparation_context = context
    return context

def span():
    return ReadyInputSpan(start=0, token_ids=[1, 31, 31], values=torch.arange(12).reshape(3, 4).half(),
        positions=issued([[0, 0, 0], [1, 2, 3], [1, 4, 5]]), position=policy(limit=1048576), rope_delta=-1,
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


def test_media_cannot_activate_a_different_persistent_profile():
    st = state()
    st.coordinate_profile_identity = "c" * 64
    with pytest.raises(ValueError, match="persistent numerical"):
        media_mtp.bind_context(st, PROFILE, MEDIA)
    assert st.media_profile_identity is None
    with pytest.raises(ValueError, match="persistent numerical"):
        media_mtp.attach_span(st, span(), profile_identity=PROFILE, media_identity=MEDIA)
    assert st.media_input_span is None


@pytest.mark.parametrize("entry", ["bind", "attach", "boundary", "snapshot"])
def test_media_capability_is_required_before_new_ownership_or_boundary_copies(entry, monkeypatch):
    st, ready = state(), span()
    st.media_inputs_enabled = False
    monkeypatch.setattr(torch.Tensor, "clone", lambda *_a, **_kw: pytest.fail("disabled media capability copied payload"))
    with pytest.raises(ValueError, match="admitted media capability"):
        if entry == "bind":
            media_mtp.bind_context(st, PROFILE, MEDIA)
        elif entry == "attach":
            media_mtp.attach_span(st, ready, profile_identity=PROFILE, media_identity=MEDIA)
        elif entry == "boundary":
            media_mtp.capture_boundary(st, 3, None, None)
        else:
            st.media_profile_identity, st.media_identity = PROFILE, MEDIA  # stale state cannot authorize capability
            media_mtp.snapshot_state(st)
    assert st.media_input_span is None and st.mtp_boundary is None
    assert not ready.closed and all(lease is None for lease in ready._leases.values())


def test_standalone_state_defaults_to_no_prepared_media_capability(request):
    from tensorfold.families.qwen4_exp.cuda.state import State

    weights = request.getfixturevalue("actual_mtp_family")
    st = State(weights, 32, 4, "fp16", limit=32)
    assert not st.media_inputs_enabled
    with pytest.raises(ValueError, match="admitted media capability"):
        st.bind_media_context(PROFILE, MEDIA)


def test_bound_state_requires_engine_binding_before_allocation(request, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda.state import State

    weights = request.getfixturevalue("actual_mtp_family")
    bind_positions(weights)
    weights.meta.pop("coordinate_profile_identity")
    monkeypatch.setattr(torch, "empty", lambda *_a, **_kw: pytest.fail("unbound State allocated device storage"))
    with pytest.raises(ValueError, match="Engine's current"):
        State(weights, 32, 4, "fp16", limit=32)


def test_boundary_coordinate_copy_does_not_pin_ready_span_parent():
    import gc
    import weakref

    st, ready = state(), span()
    media_mtp.attach_span(st, ready, profile_identity=PROFILE, media_identity=MEDIA)
    tensor = weakref.ref(ready.positions)
    boundary = media_mtp.capture_boundary(st, 3, torch.ones((1, 16)).half(), ready.coordinates,
                                          source_position=ready.position)
    media_mtp.retire_span(st, cancelled=True)
    del ready
    gc.collect()
    assert tensor() is None
    boundary.coordinate_record.proof.validate(boundary.coordinates, st.media_mtp_position,
        profile_identity=PROFILE, generation=st.slot_generation, media_identity=MEDIA,
        row_start=2, row_count=1)


@pytest.mark.parametrize("change", ["epoch", "generation", "tensor", "storage"])
def test_boundary_snapshot_validates_actual_proof_before_state_writes(change, monkeypatch):
    st = state()
    media_mtp.bind_context(st, PROFILE, MEDIA)
    st.pos, st.mtp_len = 3, 2
    coords = issued([[17, 4, 9]], start=2)
    st.mtp_boundary = media_mtp.capture_boundary(st, 3, torch.ones((1, 16)).half(), coords,
                                               source_position=st.media_target_position)
    payload = media_mtp.snapshot_state(st)["media_state"]
    record = payload["mtp_boundary"]["coordinates"]
    if change == "epoch":
        st.media_mtp_position = replace(st.media_mtp_position, epoch=1)
    elif change == "generation":
        payload["generation"] += 1
    elif change == "tensor":
        payload["mtp_boundary"]["coordinates"] = replace(record, tensor=record.tensor.clone())
    else:
        record.tensor.add_(1)
    monkeypatch.setattr(torch.Tensor, "copy_", lambda *_a, **_kw: pytest.fail("stale snapshot reached State writes"))
    with pytest.raises(ValueError):
        media_mtp.validate_snapshot(st, payload, 3, 2)


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


@pytest.mark.parametrize("notification_failure", [False, True])
def test_staging_completion_failure_retains_poisoned_request_owner(monkeypatch, notification_failure):
    st, ready = state(), span()
    media_mtp.attach_span(st, ready, profile_identity=PROFILE, media_identity=MEDIA)
    primary = RuntimeError("staging completion failed")
    secondary = RuntimeError("cohort failure notification failed")
    notified = []

    def notify(error):
        notified.append(error)
        if notification_failure:
            raise secondary

    st.input_preparation_context = SimpleNamespace(fail=notify)

    def failed():
        raise primary

    with pytest.raises(RuntimeError) as caught:
        with media_mtp.borrow_span(st, "target") as lease:
            monkeypatch.setattr(lease, "complete", failed)
    assert caught.value is primary and st._history_error is primary and st.media_input_span is ready
    assert ready.values is not None and not ready.closed
    assert notified == [primary]
    if notification_failure:
        assert primary.__cause__ is secondary


def test_one_boundary_keeps_complete_hc_separate_coordinates_and_independent_snapshot():
    st = state()
    media_mtp.bind_context(st, PROFILE, MEDIA)
    hidden = torch.arange(16).reshape(1, 16).half()
    coordinates = issued([[17, 4, 9]], start=2, position=st.media_target_position)
    st.pos, st.mtp_len = 3, 2
    st.mtp_boundary = media_mtp.capture_boundary(st, 3, hidden, coordinates, source_position=st.media_target_position)
    assert st.mtp_boundary.row == 2 and st.mtp_boundary.generation == 3
    assert st.mtp_boundary.nbytes == 16 * 2 + 3 * 8
    assert st.mtp_boundary.hidden.data_ptr() != hidden.data_ptr()
    payload = media_mtp.snapshot_state(st)["media_state"]
    assert payload["mtp_boundary"]["hidden"].data_ptr() != st.mtp_boundary.hidden.data_ptr()
    torch.testing.assert_close(payload["mtp_boundary"]["hidden"], hidden, atol=0, rtol=0)
    assert torch.equal(payload["mtp_boundary"]["coordinates"].tensor, coordinates.tensor.long())
    st.slot_generation = 4
    media_mtp.validate_snapshot(st, payload, 3, 2)
    media_mtp.restore_state(st, payload)
    assert st.mtp_boundary.generation == 4
    assert st.mtp_boundary.hidden.data_ptr() != payload["mtp_boundary"]["hidden"].data_ptr()
    budget = media_mtp.boundary_resources(hidden_width=16, slots=2, keep=8, points=4)
    assert budget["protected_versions"] == 16
    assert budget["live_hc_bytes"] + budget["live_coordinate_bytes"] == 2 * st.mtp_boundary.nbytes
    assert budget["protected_coordinate_bytes"] == 16 * payload["mtp_boundary"]["coordinates"].tensor.numel() * 8


@pytest.mark.parametrize("mutation", ["profile", "media", "frontier", "shape", "generation"])
def test_boundary_identity_and_frontier_mismatch_refuses_before_copy(mutation, monkeypatch):
    st = state()
    media_mtp.bind_context(st, PROFILE, MEDIA)
    st.pos, st.mtp_len = 3, 2
    coords = issued([[1, 1, 1]], start=2, position=st.media_target_position)
    st.mtp_boundary = media_mtp.capture_boundary(st, 3, torch.ones((1, 16)).half(), coords,
                                               source_position=st.media_target_position)
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
    weights.meta["media_inputs_enabled"] = True
    st = State(weights, 32, 4, "fp16", limit=32)
    assert "media_state" not in st.snapshot()  # legacy None path remains its original schema
    buf = Buffers(weights, 4, 32, prefill=True)
    bind_reference_inputs(weights, st, buf)
    weights.positions = None
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
    coordinates = issued([[2, 2, 2]], start=2, generation=st.slot_generation, media=None,
                          position=st.media_target_position)
    st.mtp_boundary = media_mtp.capture_boundary(st, 3, hc[2:3], coordinates, source_position=st.media_target_position)
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
    primary = RuntimeError("controlled boundary restore copy failed after target state writes")

    def failed_copy(*_a, **_kw):
        raise primary

    monkeypatch.setattr(media_mtp, "restore_coordinates", failed_copy)
    with pytest.raises(RuntimeError) as caught:
        st.restore(snap)
    assert caught.value is primary and st._history_error is primary and st._kv_error is primary
    # The protected old-version proof/payload remains available for diagnosis;
    # State containment never mutates it to manufacture current-generation trust.
    assert snap["media_state"]["mtp_boundary"]["coordinates"].proof.generation == 0


@pytest.mark.parametrize("notification_failure", [False, True])
def test_ordinary_bound_text_restore_write_failure_poison_preserves_primary(request, monkeypatch, notification_failure):
    """Real State recovery with the independent CPU cache/math fixture.

    The shared context is actual; this covers write-phase containment, not
    unique TP8 layout/model execution, which has separate target gates.
    """
    from tensorfold.families.qwen4_exp.cuda.state import State

    weights = request.getfixturevalue("actual_mtp_family")
    st = State(weights, 32, 4, "fp16", limit=32)
    context = bind_reference_inputs(weights, st)
    assert st.media_profile_identity is None and not st.media_inputs_enabled
    snap = st.snapshot()
    st.rec.zero_()
    snap["rec"].fill_(.75)
    original = torch.Tensor.copy_
    primary = RuntimeError("ordinary text restore failed after recurrent mutation")
    secondary = RuntimeError("restore cohort notification failed")

    def copy(destination, source, *args, **kwargs):
        result = original(destination, source, *args, **kwargs)
        if destination is st.conv:
            raise primary
        return result

    if notification_failure:
        fail = context.fail

        def notify(error):
            fail(error)
            raise secondary

        monkeypatch.setattr(context, "fail", notify)
    monkeypatch.setattr(torch.Tensor, "copy_", copy)
    with pytest.raises(RuntimeError) as caught:
        st.restore(snap)
    assert caught.value is primary
    assert torch.equal(st.rec[0], snap["rec"])
    assert st._history_error is primary and st._kv_error is primary
    assert context.cohort._execution_manifest_uncertain
    assert not notification_failure or primary.__cause__ is secondary


def test_ordinary_bound_text_restore_prevalidation_refusal_keeps_owner_healthy(request, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda.state import State

    weights = request.getfixturevalue("actual_mtp_family")
    st = State(weights, 32, 4, "fp16", limit=32)
    context = bind_reference_inputs(weights, st)
    snap = st.snapshot()
    bad = {**snap, "pos": st.limit + 1}
    with monkeypatch.context() as guard:
        guard.setattr(torch.Tensor, "copy_", lambda *_a, **_kw: pytest.fail("invalid snapshot reached restore writes"))
        with pytest.raises(ValueError):
            st.restore(bad)
    assert st._history_error is None and st._kv_error is None
    assert not context.cohort._execution_manifest_uncertain
    st.restore(snap)


def test_real_state_reset_retires_actual_span_and_clears_request_context(request):
    from tensorfold.families.qwen4_exp.cuda.state import State

    weights = request.getfixturevalue("actual_mtp_family")
    weights.meta["media_inputs_enabled"] = True
    st = State(weights, 32, 4, "fp16", limit=32)
    bind_reference_inputs(weights, st)
    ready = ReadyInputSpan(start=0, token_ids=[4], values=weights.embed[0].lookup(torch.tensor([4])).clone(),
        positions=issued([[0, 0, 0]], generation=st.slot_generation, position=st.media_target_position),
        position=st.media_target_position, rope_delta=0, profile_identity=PROFILE,
        generation=st.slot_generation, media_identity=MEDIA, max_rows=2, hidden_size=weights.cfg.hidden,
        vocab_size=weights.cfg.vocab, roles=("target", "mtp"))
    st.attach_media_input_span(ready, profile_identity=PROFILE, media_identity=MEDIA)
    with media_mtp.borrow_span(st, "target") as lease:
        assert lease.rows(0, [4]).shape == (1, weights.cfg.hidden)
    st.reset(weights)
    assert ready.closed and st.media_input_span is None and st.media_profile_identity is None
    assert st.media_identity is None and st.mtp_boundary is None
    assert st.coordinate_profile_identity == PROFILE


def test_actual_ready_span_deferred_visual_lookahead_and_final_text_anchor(request, monkeypatch):
    """Actual input preparation/frame proofs with independent legacy CPU math.

    The tiny full coordinate table supplies a numerical oracle. Actual compact
    coordinate staging is used; unique raw/pool ownership and SC117/V100 model
    execution remain separate qualification gates.
    """
    from tensorfold.families.qwen4_exp.cuda import forward, mtp
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, absorb
    from tensorfold.families.qwen4_exp.cuda.state import Buffers, State

    weights = request.getfixturevalue("actual_mtp_family")
    weights.meta["media_inputs_enabled"] = True
    st, buf = State(weights, 32, 4, "fp16", limit=32), Buffers(weights, 4, 32, prefill=True)
    context = bind_reference_inputs(weights, st, buf)
    buf.staged = SimpleNamespace(synchronize=lambda: None, record=lambda: None)
    from tensorfold.families.qwen4_exp.cuda import compact_indexer

    buf.compact_indexer_stage = compact_indexer.Staging(4, 1, "cpu", mtp=True)
    buf.target_indexer_coordinate_storage = buf.compact_indexer_stage.coordinates
    buf.mtp_indexer_coordinate_storage = buf.compact_indexer_stage.mtp_coordinates
    buf.target_indexer_position, buf.mtp_indexer_position = weights.positions.target, weights.positions.mtp
    seen = []
    stage_coordinates = compact_indexer.stage_mtp_indexer_frame

    def observed(buffer, segments, windows):
        stage_coordinates(buffer, segments, windows)
        seen.append(buffer.mtp_indexer_coordinates.clone())

    monkeypatch.setattr(compact_indexer, "stage_mtp_indexer_frame", observed)
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
            positions=issued(positions[start:end], start=start, generation=st.slot_generation,
                             position=st.media_target_position), position=st.media_target_position,
            rope_delta=0, profile_identity=PROFILE,
            generation=st.slot_generation, media_identity=MEDIA, max_rows=3, hidden_size=weights.cfg.hidden,
            vocab_size=weights.cfg.vocab, roles=("target", "mtp"))
        st.attach_media_input_span(ready, profile_identity=PROFILE, media_identity=MEDIA)
        n = end - start
        buf.ids[:n].copy_(torch.tensor(prompt[start:end], dtype=torch.int32))
        buf.ple_v.fill_(.125)
        st.ple_last = (st.ple_history.copy(), np.asarray(prompt[start:end], dtype=np.int64))
        with media_mtp.target_inputs(weights, buf, [(st, prompt[start:end])], [(st, 0, n)]) as input_rows:
            forward.compute(weights, [(st, 0, n)], buf, input_embeddings=input_rows)
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
    assert context.ordinal == 5  # two target phases, two committed-MTP phases, final anchor phase


def test_actual_mixed_plain_and_visual_input_frames_keep_each_generation_and_hidden_row(request, monkeypatch):
    """Actual frame/preparation adapter on the standalone CPU numerical oracle."""
    from tensorfold.families.qwen4_exp.cuda import compact_indexer, forward
    from tensorfold.families.qwen4_exp.cuda.state import Buffers, State

    weights = request.getfixturevalue("actual_mtp_family")
    weights.meta["media_inputs_enabled"] = True
    media, plain = [State(weights, 32, 4, "fp16", limit=32) for _ in range(2)]
    plain.reset(weights)  # two live requests may have different slot generations
    buf = Buffers(weights, 4, 32, prefill=True)
    context = bind_reference_inputs(weights, media, buf)
    bind_reference_inputs(weights, plain)
    buf.staged = SimpleNamespace(synchronize=lambda: None, record=lambda: None)
    buf.compact_indexer_stage = compact_indexer.Staging(4, 1, "cpu", mtp=True)
    buf.target_indexer_coordinate_storage = buf.compact_indexer_stage.coordinates
    buf.mtp_indexer_coordinate_storage = buf.compact_indexer_stage.mtp_coordinates
    buf.target_indexer_position, buf.mtp_indexer_position = weights.positions.target, weights.positions.mtp
    prompt, text = [4, 31], [17, 18, 19]
    projected = torch.full((2, weights.cfg.hidden), .125).half()
    projected[0].copy_(weights.embed[0].lookup(torch.tensor([4]))[0])
    coordinates = torch.tensor([[0, 0, 0], [1, 0, 1]], dtype=torch.int32)
    ready = ReadyInputSpan(start=0, token_ids=prompt, values=projected,
        positions=issued(coordinates, generation=media.slot_generation, position=media.media_target_position),
        position=media.media_target_position, rope_delta=0, profile_identity=PROFILE,
        generation=media.slot_generation, media_identity=MEDIA, max_rows=2, hidden_size=weights.cfg.hidden,
        vocab_size=weights.cfg.vocab, roles=("target", "mtp"))
    media.attach_media_input_span(ready, profile_identity=PROFILE, media_identity=MEDIA)
    media.image_positions = coordinates  # bounded independent legacy numerical oracle
    segments = [(media, 0, 2), (plain, 2, 4)]
    buf.ids[:4].copy_(torch.tensor([*prompt, *text[:2]], dtype=torch.int32))
    buf.ple_v.fill_(.125)
    for st, ids in ((media, prompt), (plain, text[:2])):
        st.ple_last = (st.ple_history.copy(), np.asarray(ids, dtype=np.int64))
    original, lookups = forward._embed, []

    def text_only(w, ids, copies, out):
        lookups.extend(ids.tolist())
        assert 31 not in ids.tolist(), "visual placeholder reached token lookup"
        return original(w, ids, copies, out)

    monkeypatch.setattr(forward, "_embed", text_only)
    with media_mtp.target_inputs(weights, buf, [(media, prompt), (plain, text[:2])], segments) as inputs:
        forward.compute(weights, segments, buf, input_embeddings=inputs)
    hidden = buf.streams[:4].clone()
    for st, first, last in segments:
        forward.commit(weights, st, buf, last - first, last - first, at=first)
    media_mtp.condition_chunks(weights, buf, [
        media_mtp.CommittedMTPChunk(media, prompt, 0, 2, buf.streams[:2]),
        media_mtp.CommittedMTPChunk(plain, text, 0, 2, buf.streams[2:4])])
    assert lookups == [17, 18, 18, 19]
    assert torch.equal(buf.mtp_e[0], projected[1])
    assert torch.equal(buf.mtp_e[1:3], weights.embed[0].lookup(torch.tensor(text[1:])))
    assert torch.equal(buf.mtp_in[:3], hidden[[0, 2, 3]])  # full HC rows, distinct from next-input coordinates
    assert media.mtp_len == 1 and media.mtp_boundary.row == 1
    assert plain.mtp_len == 2 and plain.media_profile_identity is None and plain.mtp_boundary is None
    issued_frame = buf.mtp_indexer_frame.issued
    assert [row[-1].proof.generation for row in issued_frame] == [0, 1]
    assert [row[-1].proof.media_identity for row in issued_frame] == [MEDIA, None]
    assert context.ordinal == 2
    media.retire_media_input_span(cancelled=False)


@pytest.mark.parametrize("role", ["target", "mtp"])
def test_preparation_owner_runs_before_any_media_window_or_lease_work(role, monkeypatch):
    primary = RuntimeError("cohort preparation refused before input ownership")

    def refused(_label, _operation):
        raise primary

    buf = SimpleNamespace(input_preparation_context=SimpleNamespace(prepare=refused))
    weights = SimpleNamespace(meta={})
    monkeypatch.setattr(media_mtp, "_target_inputs", lambda *_a: pytest.fail("target builder ran outside owned phase"))
    monkeypatch.setattr(media_mtp, "_condition_inputs", lambda *_a: pytest.fail("MTP builder ran outside owned phase"))
    with pytest.raises(RuntimeError) as caught:
        if role == "target":
            with media_mtp.target_inputs(weights, buf, None, None):
                pytest.fail("refused preparation published inputs")
        else:
            media_mtp.condition_chunks(weights, buf, None)
    assert caught.value is primary
