"""Actual producer proofs, role domains and candidate publication before writes."""

from types import SimpleNamespace as NS
import weakref

import numpy as np
import pytest
import torch

from tensorfold.families.qwen4_exp.cuda.compact_indexer import (
    CoordinateWindow, Staging, issue_text_indexer_coordinates,
    stage_target_indexer_frame, stage_mtp_indexer_frame, stage_text_indexer_frame,
    validate_target_indexer_frame, validate_mtp_indexer_frame,
)
from tensorfold.families.qwen4_exp.cuda.position_coordinates import issue_coordinates
from tensorfold.families.qwen4_exp.cuda.input_preparation import InputPreparationContext
from test_qwen4_exp_position_coordinates import policy


def prepared(*, mtp=False, profile_identity="1" * 64, context_identity=None, host_coordinates=False):
    position = policy("mtp" if mtp else "target")
    stage = Staging(4, 13, "cpu", mtp=True, host_coordinates=host_coordinates)
    prefix = "mtp" if mtp else "target"
    storage = stage.mtp_coordinates if mtp else stage.coordinates
    buffers = NS(rows=4, split=None, compact_indexer_stage=stage, input_preparation_context=None, **{
        prefix + "_indexer_position": position, prefix + "_indexer_coordinate_storage": storage,
        prefix + "_indexer_frame_id": 0, prefix + "_indexer_coordinates": None,
        prefix + "_indexer_frame": None})
    if context_identity is not None:
        buffers.input_preparation_context = InputPreparationContext(NS(
            positions=object(), comm=None, meta={"coordinate_profile_identity": context_identity}))
    states, windows = [], []
    for first, start, generation, media in ((0, 7, 3, "a"), (2, 12, 9, "b")):
        state = NS(pos=start, mtp_len=start, slot_generation=generation, history_generation=1,
                   position_profile_identity=position.identity, position_epoch=position.epoch,
                   coordinate_profile_identity=profile_identity, media_profile_identity=None, media_identity=media * 64,
                   media_input_span=None,
                   image_positions=None, rope_delta=0, capacity=128, pos_dev=torch.zeros(1, dtype=torch.int32))
        issued = issue_coordinates(np.array([[start, 21, 31], [start + 1, 22, 32]]), position=position,
            profile_identity=state.coordinate_profile_identity, generation=generation, media_identity=state.media_identity,
            row_start=start, max_rows=4, device="cpu")
        states.append(state)
        windows.append(CoordinateWindow(first, first + 2, issued, position))
    return buffers, position, states, tuple(windows), [(st, i * 2, i * 2 + 2) for i, st in enumerate(states)]


@pytest.mark.parametrize("mtp", [False, True])
@pytest.mark.parametrize("route", ["issued", "text", "clear"])
def test_foreign_full_profile_is_refused_before_frame_or_text_writes(monkeypatch, mtp, route):
    from tensorfold.families.qwen4_exp.cuda import position_coordinates as issuer

    buffers, position, states, windows, segments = prepared(mtp=mtp, context_identity="1" * 64)
    stage = stage_mtp_indexer_frame if mtp else stage_target_indexer_frame
    validate = validate_mtp_indexer_frame if mtp else validate_target_indexer_frame
    prefix = "mtp" if mtp else "target"
    stage(buffers, segments, windows)
    old = getattr(buffers, prefix + "_indexer_frame")
    expected = validate(buffers, segments).clone()
    version = old.storage._version
    buffers.input_preparation_context = InputPreparationContext(NS(
        positions=object(), comm=None, meta={"coordinate_profile_identity": "2" * 64}))
    assert all(st.position_profile_identity == position.identity for st in states)
    assert all(window.source.proof.profile_identity == "1" * 64 for window in windows)
    monkeypatch.setattr(buffers.compact_indexer_stage, "begin",
                        lambda **kw: pytest.fail("staging began before full profile validation"))
    monkeypatch.setattr(issuer, "issue_coordinates",
                        lambda *a, **kw: pytest.fail("coordinate writes before full profile validation"))
    with pytest.raises(ValueError, match="different full coordinate profiles"):
        if route == "text":
            stage_text_indexer_frame(buffers, segments, mtp=mtp)
        else:
            stage(buffers, segments, windows if route == "issued" else None)
    assert old.active and getattr(buffers, prefix + "_indexer_frame") is old
    assert getattr(buffers, prefix + "_indexer_frame_id") == old.frame_id
    assert old.storage._version == version
    assert torch.equal(getattr(buffers, prefix + "_indexer_coordinates"), expected)


@pytest.mark.parametrize("route", ["target", "mtp", "compute", "mixed", "mtp_compute"])
def test_foreign_full_profile_is_rejected_before_proofs_or_persistent_compute(monkeypatch, route):
    from tensorfold.cuda import cohort
    from tensorfold.families.qwen4_exp.cuda import forward
    from tensorfold.families.qwen4_exp.cuda import mtp as method, position_coordinates as issuer

    is_mtp = route in ("mtp", "mtp_compute")
    buffers, _position, _states, windows, segments = prepared(mtp=is_mtp, context_identity="1" * 64)
    stage = stage_mtp_indexer_frame if is_mtp else stage_target_indexer_frame
    validate = validate_mtp_indexer_frame if is_mtp else validate_target_indexer_frame
    stage(buffers, segments, windows)
    assert validate(buffers, segments).tolist() == [[7, 21, 31], [8, 22, 32], [12, 21, 31], [13, 22, 32]]
    context = InputPreparationContext(NS(positions=object(), comm=None,
        meta={"coordinate_profile_identity": "2" * 64}))
    buffers.input_preparation_context = context
    notifications = []
    monkeypatch.setattr(cohort, "notify_failure", lambda: notifications.append(True))
    monkeypatch.setattr(issuer.CoordinateProof, "validate",
                        lambda *a, **kw: pytest.fail("proof validation before full profile validation"))
    def persistent(*args):
        pytest.fail("persistent work before full profile validation")
    monkeypatch.setattr(forward, "_kv_begin", persistent)
    monkeypatch.setattr(method, "_kv_begin", persistent)
    weights = NS(cfg=NS(), dflash2=NS(begin=persistent))
    with pytest.raises(ValueError, match="different full coordinate profiles"):
        if route == "compute":
            forward.compute(weights, segments, buffers)
        elif route == "mixed":
            forward.compute_mixed(weights, segments, buffers, segments, buffers)
        elif route == "mtp_compute":
            method.mtp_compute(weights, segments, buffers)
        else:
            validate(buffers, segments)
    assert notifications == [True] and context.cohort._execution_manifest_uncertain


@pytest.mark.parametrize("mtp", [False, True])
def test_actual_multiple_state_proofs_bind_final_shared_versions_under_inference_mode(mtp):
    with torch.inference_mode():
        buffers, position, states, windows, segments = prepared(mtp=mtp)
        stage = stage_mtp_indexer_frame if mtp else stage_target_indexer_frame
        validate = validate_mtp_indexer_frame if mtp else validate_target_indexer_frame
        stage(buffers, segments, windows)
    prefix = "mtp" if mtp else "target"
    values, frame = validate(buffers, segments), getattr(buffers, prefix + "_indexer_frame")
    assert values.tolist() == [[7, 21, 31], [8, 22, 32], [12, 21, 31], [13, 22, 32]]
    assert len(frame.issued) == 2 and frame.active
    for st, first, last, start, issued in frame.issued:
        issued.proof.validate(issued.tensor, position, profile_identity=st.coordinate_profile_identity,
            generation=st.slot_generation, media_identity=st.media_identity,
            row_start=start, row_count=last - first, owner=frame)
        assert issued.tensor._version == values._version
    states[-1].media_identity = "e" * 64
    with pytest.raises(ValueError, match="another row"):
        validate(buffers, segments)


def test_failed_second_copy_revokes_old_frame_and_publishes_no_candidate(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import position_coordinates as issuer

    buffers, _position, _states, windows, segments = prepared()
    stage_target_indexer_frame(buffers, segments, windows)
    old = buffers.target_indexer_frame
    original, calls, candidate = issuer._perform_copy, [], []
    def partial(plan):
        candidate.append(weakref.ref(plan.owner))
        calls.append(plan)
        if len(calls) == 2:
            raise MemoryError("second packed copy")
        return original(plan)
    monkeypatch.setattr(issuer, "_perform_copy", partial)
    with pytest.raises(MemoryError, match="second packed"):
        stage_target_indexer_frame(buffers, segments, windows)
    assert len(calls) == 2 and not old.active
    assert buffers.target_indexer_frame is buffers.target_indexer_coordinates is None
    assert buffers.target_indexer_frame_id == old.frame_id + 1
    with pytest.raises(ValueError, match="issued coordinate frame"):
        validate_target_indexer_frame(buffers, segments)
    assert not any(ref() is old for ref in candidate)
    monkeypatch.setattr(issuer, "_perform_copy", original)
    stage_target_indexer_frame(buffers, segments, windows)
    assert validate_target_indexer_frame(buffers, segments).shape == (4, 3)


@pytest.mark.parametrize("partial", [False, True])
def test_unretired_frame_failure_keeps_primary_and_uses_existing_cohort_marker(monkeypatch, partial):
    from tensorfold.cuda import cohort
    from tensorfold.families.qwen4_exp.cuda import position_coordinates as issuer
    from tensorfold.families.qwen4_exp.cuda.execution_identity import _unretired

    buffers, _position, states, windows, segments = prepared()
    primary, retirement, notified = MemoryError("partial coordinate copy"), ValueError("coordinate event"), []
    class Event:
        def record(self, stream):
            raise retirement
    class Stream:
        cuda_stream = 2
        def wait_event(self, event):
            pass
    stage = buffers.compact_indexer_stage
    stage._events[0], stage._streams[0] = Event(), 1
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: Stream())
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *a: None)
    monkeypatch.setattr(cohort, "notify_failure", lambda: notified.append(True))
    if partial:
        original, calls = issuer._perform_copy, []
        def fail(plan):
            calls.append(plan)
            if len(calls) == 2:
                raise primary
            return original(plan)
        monkeypatch.setattr(issuer, "_perform_copy", fail)
    with pytest.raises(BaseException) as caught:
        stage_target_indexer_frame(buffers, segments, windows)
    assert caught.value is (primary if partial else retirement)
    assert notified == [True] and _unretired(caught.value)
    assert stage._errors[0] is retirement and stage._borrowers[0]
    assert all(st._history_error is st._kv_error is caught.value for st in states)
    assert buffers.target_indexer_frame is buffers.target_indexer_coordinates is None
    with pytest.raises(ValueError) as refused:
        stage.begin(mtp=False)
    assert refused.value is retirement


@pytest.mark.parametrize("changed", ["values", "profile", "generation", "epoch"])
def test_invalid_frame_is_rejected_before_target_or_mixed_persistent_work(monkeypatch, changed):
    from tensorfold.families.qwen4_exp.cuda import forward

    buffers, _position, states, windows, segments = prepared()
    stage_target_indexer_frame(buffers, segments, windows)
    if changed == "values":
        buffers.target_indexer_coordinates[0].fill_(99)
    elif changed == "profile":
        states[0].position_profile_identity = "e" * 64
    elif changed == "generation":
        states[0].slot_generation += 1
    else:
        states[0].position_epoch += 1
    monkeypatch.setattr(forward, "_kv_begin", lambda *a: pytest.fail("persistent work before coordinate authority"))
    weights = NS(dflash2=NS(begin=lambda *a: pytest.fail("DF capture before coordinate authority")))
    with pytest.raises(ValueError):
        forward.compute(weights, segments, buffers)
    with pytest.raises(ValueError):
        forward.compute_mixed(weights, segments, buffers, segments, buffers)


@pytest.mark.parametrize("mtp", [False, True])
@pytest.mark.parametrize("changed", ["mutated", "stale"])
def test_actual_frame_error_stays_primary_when_cohort_notification_fails(mtp, changed):
    buffers, _position, states, windows, segments = prepared(mtp=mtp)
    stage = stage_mtp_indexer_frame if mtp else stage_target_indexer_frame
    validate = validate_mtp_indexer_frame if mtp else validate_target_indexer_frame
    stage(buffers, segments, windows)
    if changed == "mutated":
        values = buffers.mtp_indexer_coordinates if mtp else buffers.target_indexer_coordinates
        values[0].fill_(99)
    else:
        states[0].slot_generation += 1
    secondary, observed = RuntimeError("cohort notification failed"), []
    def fail(error):
        observed.append(error)
        raise secondary
    buffers.input_preparation_context = NS(identity="1" * 64, fail=fail)
    with pytest.raises(ValueError) as caught:
        validate(buffers, segments)
    assert observed == [caught.value] and caught.value is observed[0]
    assert ("mutated" if changed == "mutated" else "generation") in str(caught.value)
    assert caught.value.__cause__ is secondary


def test_native_writer_increments_the_actual_shared_frame_counter_before_submission(monkeypatch):
    from tensorfold.cuda.sm70 import compact_indexer as native

    buffers, _position, _states, windows, segments = prepared()
    stage_target_indexer_frame(buffers, segments, windows)
    before = buffers.target_indexer_coordinates._version
    def publish(source, destination, *args):
        assert destination._version == before + 1
        destination.numpy()[:] = source.numpy()  # Raw write does not dispatch a Torch mutation.
    monkeypatch.setattr(native, "extension", lambda: NS(publish=publish))
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda tensor: True))
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: NS())
    native.publish(torch.zeros(4, 3, dtype=torch.int64), buffers.target_indexer_coordinates, 0,
                   torch.zeros(2, dtype=torch.int64), (0, 0), torch.tensor([0, -2], dtype=torch.int32), 1)
    with pytest.raises(ValueError, match="mutated"):
        validate_target_indexer_frame(buffers, segments)


def test_plain_text_domain_uses_live_rows_and_known_host_delta_without_a_device_readback():
    buffers, position, states, _windows, segments = prepared()
    stage_text_indexer_frame(buffers, segments)
    assert validate_target_indexer_frame(buffers, segments).shape == (4, 3)
    states[0].rope_delta = 121
    with pytest.raises(ValueError, match="coordinate"):
        stage_text_indexer_frame(buffers, segments)
    states[0].rope_delta = 3
    states[0].media_profile_identity = "1" * 64
    issued = issue_text_indexer_coordinates(states[0], position, 7, 2, max_rows=4)
    assert issued.tensor.tolist() == [[10, 10, 10], [11, 11, 11]]
    assert issued.proof.row_start == 7
    states[0].image_positions = torch.zeros(4, 3, dtype=torch.int32)
    with pytest.raises(ValueError, match="host row/context"):
        issue_text_indexer_coordinates(states[0], position, 7, 2, max_rows=4)


def test_state_snapshot_identity_contains_the_actual_position_profile_epoch():
    from tensorfold.families.qwen4_exp.cuda.state import State
    from tests.test_native_owner_adapter import _weights

    def make(epoch):
        weights = _weights()
        target, mtp = policy(epoch=epoch), policy("mtp", epoch=epoch)
        weights.meta["coordinate_profile_identity"] = "1" * 64
        weights.positions = NS(profile=NS(identity=target.identity, epoch=epoch), target=target, mtp=mtp)
        return State(weights, 32, 4, "int8")
    first, second = make(0), make(1)
    assert first.kv_identity != second.kv_identity
    snapshot = first.snapshot()
    with pytest.raises(ValueError, match="matching ordered formats"):
        second.kv_validate_snapshot(snapshot)


@pytest.mark.parametrize("mtp", [False, True])
def test_owner_attention_and_pooling_use_the_selected_role_operands(monkeypatch, mtp):
    from tensorfold.cuda.sm70 import family_ops
    from tensorfold.families.qwen4_exp.cuda.history_owner import prepare
    from tensorfold.families.qwen4_exp.cuda.state import State
    from tests.test_native_owner_adapter import _weights

    weights = _weights(4 if mtp else 0)
    c = weights.cfg
    c.eps, c.mrope_section, c.rope_attention_factor = 1e-6, (1, 1, 1), 1.
    weights.inv_freq = torch.tensor([7.])
    target = NS(inv_freq=torch.tensor([0.25]), amplitude=1., sections=(1, 1, 1),
                domain=policy().domain, identity="c" * 64, epoch=0)
    method = NS(inv_freq=torch.tensor([0.125]), amplitude=1.5, sections=(1, 1, 1),
                domain=policy("mtp").domain, identity="c" * 64, epoch=0)
    operands = method if mtp else target
    weights.meta["coordinate_profile_identity"] = "1" * 64
    weights.positions = NS(profile=NS(identity=operands.identity, epoch=0), target=target, mtp=method)
    state = State(weights, 32, 4, "int8")
    stage = Staging(4, 13, "cpu", mtp=True)
    stage.begin(mtp=mtp)
    buffers = NS(prefill=True, q=torch.empty(4, c.heads, c.head_dim, dtype=torch.float16),
                 iq=torch.empty(4, c.index_heads, c.index_dim, dtype=torch.float16), compact_indexer_stage=stage,
                 target_indexer_coordinates=None, mtp_indexer_coordinates=None,
                 target_indexer_position=target, mtp_indexer_position=method)
    layer = NS(index=-1 if mtp else 0, attn=NS(q_scale=torch.ones(c.head_dim), k_scale=torch.ones(c.head_dim),
                                            iq_scale=torch.ones(c.index_dim), ik_scale=torch.ones(c.index_dim)))
    records = []
    def rotary(value, frequencies, *args, **kwargs):
        records.append((frequencies, kwargs["sections"], kwargs["scale"]))
        return value
    monkeypatch.setattr(family_ops, "rotary", rotary)
    width = c.heads * 2 * c.head_dim + 2 * c.head_dim + (c.index_heads + 1) * c.index_dim
    prepare(torch.ones(4, width, dtype=torch.float16), state, layer, weights, buffers, 0, 4, mtp=mtp)
    assert len(records) == 4
    assert all(freq is operands.inv_freq and sections == operands.sections and scale == operands.amplitude
               for freq, sections, scale in records)


@pytest.mark.parametrize('mtp', [False, True])
def test_paid_host_text_stage_reuses_storage_and_publishes_current_full_batch(monkeypatch, mtp):
    from tensorfold.families.qwen4_exp.cuda import position_coordinates as issuer
    b, p, states, _, segments = prepared(mtp=mtp, host_coordinates=True)
    stage = b.compact_indexer_stage
    host = stage._host[int(mtp)]
    address = host.data_ptr()
    monkeypatch.setattr(issuer, 'issue_coordinates', lambda *a, **k: pytest.fail('GPU int32 source issuance'))
    original, sources = issuer._issued, []
    def issued(tensor, *a, **kw):
        value = original(tensor, *a, **kw)
        if tensor.untyped_storage().data_ptr() == host.untyped_storage().data_ptr():
            sources.append(value)
        return value
    monkeypatch.setattr(issuer, '_issued', issued)
    validate = validate_mtp_indexer_frame if mtp else validate_target_indexer_frame
    for turn in range(2):
        stage_text_indexer_frame(b, segments, mtp=mtp)
        result = validate(b, segments)
        assert result.dtype == torch.int64 and result.tolist() == [[7+turn]*3,[8+turn]*3,[12+turn]*3,[13+turn]*3]
        assert host.data_ptr() == address and stage.pinned_host_bytes == 2 * 4 * 24
        frame = getattr(b, ('mtp' if mtp else 'target') + '_indexer_frame')
        assert all(value.proof._owner() is frame for *_meta, value in frame.issued)
        if turn == 0:
            old = frame
            for st in states:
                st.pos += 1
                st.mtp_len += 1
    assert not old.active and frame is not old
    with pytest.raises(ValueError, match='changed or retired'):
        sources[0].proof.validate(sources[0].tensor,p,profile_identity=states[0].coordinate_profile_identity,
            generation=states[0].slot_generation,media_identity=states[0].media_identity,
            row_start=7,row_count=2,owner=old)
    stage.close_host()
    assert stage.pinned_host_bytes == 0


def test_host_text_mixed_with_immutable_media_boundary_keeps_actual_coordinates():
    b, p, states, windows, segments = prepared(host_coordinates=True)
    recipe = issue_text_indexer_coordinates(states[1], p, 12, 2, max_rows=b.rows, buffers=b)
    mixed = (windows[0], CoordinateWindow(2,4,recipe,p))
    stage_target_indexer_frame(b,segments,mixed)
    assert validate_target_indexer_frame(b,segments).tolist() == [[7,21,31],[8,22,32],[12]*3,[13]*3]
    assert windows[0].source.tensor.tolist() == [[7,21,31],[8,22,32]]


def test_host_dma_retirement_precedes_rewrite_and_record_failure_retains_sources(monkeypatch):
    b, _p, _states, _, segments = prepared(host_coordinates=True)
    stage = b.compact_indexer_stage
    host = stage._host[0]
    host.fill_(99)
    events = []
    class Event:
        def synchronize(self):
            assert torch.all(host == 99)
            events.append('wait')
        def record(self, stream):
            events.append('record')
            raise RuntimeError('uncertain host DMA endpoint')
    stage._host_events[0],stage._host_pending[0] = Event(),True
    monkeypatch.setattr(torch.cuda,'current_stream',lambda *a:object())
    with pytest.raises(RuntimeError,match='uncertain host DMA endpoint'):
        stage_text_indexer_frame(b,segments)
    assert events == ['wait','record']
    assert stage._host[0] is host and stage._borrowers[0]
    assert b.target_indexer_frame is None
    with pytest.raises(RuntimeError,match='uncertain host DMA endpoint'):
        stage.close_host()
    assert stage._host[0] is host


def test_invalid_host_recipe_stops_before_final_device_frame_write():
    b, _p, states, _, segments = prepared(host_coordinates=True)
    before = b.target_indexer_coordinate_storage._version
    states[1].rope_delta = -1000
    with pytest.raises(ValueError):
        stage_text_indexer_frame(b,segments)
    assert b.target_indexer_coordinate_storage._version == before
    assert b.target_indexer_frame is None


def test_copy_failure_remains_primary_when_host_completion_record_also_fails(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import position_coordinates as issuer
    b, _p, _states, _, segments = prepared(host_coordinates=True)
    primary,secondary=ValueError('copy failed'),RuntimeError('event failed')
    monkeypatch.setattr(issuer,'copy_coordinate_batch',lambda *a,**k:(_ for _ in ()).throw(primary))
    stage=b.compact_indexer_stage
    def fail(**kw):
        stage._errors[0]=secondary
        raise secondary
    monkeypatch.setattr(stage,'host_submitted',fail)
    with pytest.raises(ValueError) as caught:
        stage_text_indexer_frame(b,segments)
    assert caught.value is primary
    assert primary.__cause__ is secondary or isinstance(primary.__cause__,BaseExceptionGroup)
    assert stage._host[0] is not None and stage._borrowers[0]


@pytest.mark.parametrize('mtp,boundary', [(False,False),(True,False),(True,True)])
def test_actual_target_and_mtp_host_staging_preserve_ids_and_boundary_coordinates(mtp,boundary):
    from tensorfold.families.qwen4_exp.cuda import forward
    from tensorfold.families.qwen4_exp.cuda import mtp as method
    b,p,states,windows,segments=prepared(mtp=mtp,host_coordinates=True)
    b.staged=NS(synchronize=lambda:None,record=lambda:None)
    b.ids_host,b.ids=torch.zeros(4,dtype=torch.int32),torch.zeros(4,dtype=torch.int32)
    b.last_host,b.last=torch.zeros(2,dtype=torch.int32),torch.zeros(2,dtype=torch.int32)
    b.mtp_in=torch.zeros((4,1))
    for state in states:
        state.mtp_boundary=None
    if boundary:
        states[0].mtp_boundary=NS(row=7,generation=3,coordinate_record=windows[0].source,position=p)
    tokens=[[5,6],[9,10]]
    if mtp:
        actual=method._mtp_stage(b,[(st,ids,torch.ones((2,1)))for st,ids in zip(states,tokens)])
        result=validate_mtp_indexer_frame(b,actual)
        assert b.last[:2].tolist()==[1,3]
    else:
        actual=forward._stage(NS(layers=[]),b,list(zip(states,tokens)))
        result=validate_target_indexer_frame(b,actual)
    assert b.ids.tolist()==[5,6,9,10] and actual==segments
    expected=[[7,21,31]if boundary else [7]*3,[8]*3,[12]*3,[13]*3]
    assert result.tolist()==expected
