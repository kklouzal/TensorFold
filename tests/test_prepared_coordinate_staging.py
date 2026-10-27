"""Prepared text rows agree with the actual original issuer and keep DMA ownership."""

from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.qwen4_exp.cuda import compact_indexer as compact  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import position_coordinates as issuer  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.history_placement import prepared_coordinate_host_resources  # noqa: E402
from test_compact_position_frames import prepared  # noqa: E402
from test_qwen4_exp_position_coordinates import policy  # noqa: E402


def frame(*, mtp=False):
    buffers, position, states, windows, segments = prepared(mtp=mtp)
    stage = compact.Staging(4, 13, "cpu", mtp=True, text_coordinates=True)
    buffers.compact_indexer_stage = stage
    setattr(buffers, ("mtp" if mtp else "target") + "_indexer_coordinate_storage",
            stage.mtp_coordinates if mtp else stage.coordinates)
    return buffers, position, states, windows, segments


@pytest.mark.parametrize("mtp", [False, True])
@pytest.mark.parametrize("inference", [False, True])
def test_prepared_text_matches_original_issuer_without_tensor_constructor(monkeypatch, mtp, inference):
    old, position, old_states, _windows, old_segments = prepared(mtp=mtp)
    new, _position, states, _windows, segments = frame(mtp=mtp)
    for group in (old_states, states):
        group[0].rope_delta, group[1].rope_delta = 5, -2
    compact.stage_text_indexer_frame(old, old_segments, mtp=mtp)
    validate = compact.validate_mtp_indexer_frame if mtp else compact.validate_target_indexer_frame
    expected = validate(old, old_segments).clone()
    prefix = "mtp" if mtp else "target"
    storage = getattr(new, prefix + "_indexer_coordinate_storage")
    pointer = storage.data_ptr()
    monkeypatch.setattr(issuer, "issue_coordinates", lambda *a, **kw: pytest.fail("ordinary blocking issuer"))
    monkeypatch.setattr(torch, "tensor", lambda *a, **kw: pytest.fail("new temporary tensor construction"))
    with torch.inference_mode(inference):
        compact.stage_text_indexer_frame(new, segments, mtp=mtp)
    actual = validate(new, segments)
    assert torch.equal(actual, expected) and actual.dtype == torch.int64
    assert pointer == actual.data_ptr() and actual.stride() == storage.stride()
    current = getattr(new, prefix + "_indexer_frame")
    assert current.active and len(current.issued) == 2
    for state, first, last, start, issued in current.issued:
        assert issued.tensor.dtype == torch.int64 and issued.tensor._version == storage._version
        issued.proof.validate(issued.tensor, position, profile_identity=state.coordinate_profile_identity,
            generation=state.slot_generation, media_identity=state.media_identity,
            row_start=start, row_count=last - first, owner=current)
        assert issued.proof.minimum == tuple(expected[first:last].min(dim=0).values.tolist())
        assert issued.proof.maximum == tuple(expected[first:last].max(dim=0).values.tolist())


@pytest.mark.parametrize("change", ["storage", "slot", "history", "position", "media"])
def test_final_prepared_proofs_refuse_changed_storage_or_semantic_generation(change):
    buffers, _position, states, _windows, segments = frame()
    compact.stage_text_indexer_frame(buffers, segments)
    if change == "storage":
        buffers.target_indexer_coordinates.add_(1)
    elif change == "slot":
        states[0].slot_generation += 1
    elif change == "history":
        states[0].history_generation += 1
    elif change == "position":
        states[0].pos += 1
    else:
        states[0].media_identity = "f" * 64
    with pytest.raises(ValueError):
        compact.validate_target_indexer_frame(buffers, segments)


@pytest.mark.parametrize("change", ["range", "negative", "profile"])
def test_preflight_refuses_before_prepared_host_or_old_frame_changes(monkeypatch, change):
    buffers, _position, states, _windows, segments = frame()
    compact.stage_text_indexer_frame(buffers, segments)
    old = buffers.target_indexer_frame
    expected = buffers.target_indexer_coordinates.clone()
    host = buffers.compact_indexer_stage.text_coordinates.hosts[0]
    host_before = host.clone()
    if change == "range":
        states[0].pos = 127
    elif change == "negative":
        states[0].rope_delta = -128
    else:
        from tensorfold.families.qwen4_exp.cuda.input_preparation import InputPreparationContext
        buffers.input_preparation_context = InputPreparationContext(NS(positions=object(), comm=None,
            meta={"coordinate_profile_identity": "9" * 64}))
    monkeypatch.setattr(type(buffers.compact_indexer_stage.text_coordinates), "host",
                        lambda self, **_kw: pytest.fail("host reuse before input proof validation"))
    with pytest.raises(ValueError):
        compact.stage_text_indexer_frame(buffers, segments)
    assert old.active and buffers.target_indexer_frame is old
    assert torch.equal(host, host_before) and torch.equal(buffers.target_indexer_coordinates, expected)


@pytest.mark.parametrize("bad", ["foreign-array", "out-of-domain", "gap", "alias"])
def test_prepared_issuer_checks_actual_host_rows_before_copy_or_proof(bad):
    position = policy()
    host, destination = torch.zeros((2, 3), dtype=torch.int64), torch.full((2, 3), 77, dtype=torch.int64)
    raw = host.numpy()
    raw[:] = [[7, 7, 7], [8, 8, 8]]
    request = issuer.CoordinateIssue(0, 2, position, "a" * 64, 3, None, 7)
    if bad == "foreign-array":
        raw = raw.copy()
    elif bad == "out-of-domain":
        raw[1, 2] = 128
    elif bad == "gap":
        request = issuer.CoordinateIssue(1, 2, position, "a" * 64, 3, None, 7)
    else:
        destination = host
    class Owner:
        pass
    owner, published, before = Owner(), [], destination.clone()
    with pytest.raises(ValueError):
        issuer.issue_prepared_coordinate_batch(host, raw, destination, (request,), owner=owner,
            max_rows=2, submitted=lambda: published.append(True))
    assert torch.equal(destination, before) and not published


def test_host_reuse_waits_for_previous_dma_before_any_source_mutation(monkeypatch):
    buffers, _position, states, _windows, segments = frame()
    compact.stage_text_indexer_frame(buffers, segments)
    owner = buffers.compact_indexer_stage.text_coordinates
    old = owner.hosts[0].clone()
    events = []
    class Event:
        def synchronize(self):
            assert torch.equal(owner.hosts[0], old)
            events.append("old-dma-completed")
        def record(self, _stream):
            assert not torch.equal(owner.hosts[0], old)
            events.append("new-dma-recorded")
    owner.events[0], owner.pending[0] = Event(), True
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_args: NS(cuda_stream=1))
    states[0].pos, states[1].pos = 20, 25
    compact.stage_text_indexer_frame(buffers, segments)
    assert events == ["old-dma-completed", "new-dma-recorded"] and owner.pending[0]
    assert compact.validate_target_indexer_frame(buffers, segments).tolist() == [[20] * 3, [21] * 3, [25] * 3, [26] * 3]


def test_unknown_upload_retirement_keeps_exact_pins_and_poison_until_real_fence(monkeypatch):
    from tensorfold.cuda import cohort
    from tensorfold.families.qwen4_exp.cuda.execution_identity import _unretired

    buffers, _position, states, _windows, segments = frame()
    stage, failure, notifications = buffers.compact_indexer_stage, OSError("upload event record"), []
    owner = stage.text_coordinates
    host = owner.hosts[0]
    class Event:
        def record(self, _stream):
            raise failure
    owner.events[0] = Event()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_args: NS(cuda_stream=1))
    monkeypatch.setattr(cohort, "notify_failure", lambda: notifications.append(True))
    with pytest.raises(OSError) as caught:
        compact.stage_text_indexer_frame(buffers, segments)
    assert caught.value is failure and stage._errors[0] is failure and _unretired(failure)
    assert all(state._history_error is failure for state in states)
    assert owner.pending[0] and owner.hosts[0] is host and stage._borrowers[0] == (host,)
    assert notifications == [True] and buffers.target_indexer_frame is None
    before = host.clone()
    with pytest.raises(OSError) as again:
        compact.stage_text_indexer_frame(buffers, segments)
    assert again.value is failure and torch.equal(host, before)
    with pytest.raises(ValueError, match="completed model device fence"):
        owner.retire_after_fence(fenced=False)
    assert owner.hosts[0] is host
    owner.retire_after_fence(fenced=True)  # CPU lifecycle control; Engine's real fence is tested separately.
    assert owner.closed and owner.hosts == [None, None] and owner.arrays == [None, None]


def test_media_profiles_keep_the_original_issued_coordinate_route(monkeypatch):
    buffers, _position, states, _windows, segments = frame()
    for state in states:
        state.media_profile_identity = "b" * 64
    original, calls = issuer.issue_coordinates, []
    def ordinary(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(issuer, "issue_coordinates", ordinary)
    compact.stage_text_indexer_frame(buffers, segments)
    assert calls == [True, True]
    assert compact.validate_target_indexer_frame(buffers, segments).tolist() == [[7] * 3, [8] * 3, [12] * 3, [13] * 3]


@pytest.mark.parametrize("rows,prefill,mtp", [(1, 4, False), (32, 256, True), (64, 2048, True)])
def test_pin_payload_and_known_headers_are_added_separately(rows, prefill, mtp):
    import sys

    receipt = prepared_coordinate_host_resources(rows, prefill, mtp=mtp)
    stages = [compact.Staging(rows, 13, "cpu", mtp=False, text_coordinates=True),
              compact.Staging(prefill, 13, "cpu", mtp=mtp, text_coordinates=True)]
    if mtp:
        stages.append(compact.Staging(rows, 13, "cpu", mtp=True, mtp_only=True, text_coordinates=True))
    actual = sum(t.nbytes for stage in stages for t in stage.text_coordinates.hosts if t is not None)
    assert actual == receipt["pinned_payload_bytes"] == 24 * (rows + prefill) * (1 + int(mtp))
    assert receipt["total_host_bytes"] == actual + receipt["known_header_bytes"]
    assert receipt["known_header_bytes"] > receipt["roles"] * (72 + 176 + 96 + 24 + 112)
    for stage in stages:
        for array in stage.text_coordinates.arrays:
            if array is None:
                continue
            interface = array.__array_interface__
            assert set(interface) == {"data", "strides", "descr", "typestr", "shape", "version"}
            pending, seen, observed = [interface], set(), 0
            while pending:
                value = pending.pop()
                if id(value) in seen:
                    continue
                seen.add(id(value))
                observed += sys.getsizeof(value)
                if type(value) is dict:
                    pending.extend(value.keys())
                    pending.extend(value.values())
                elif type(value) in (tuple, list):
                    pending.extend(value)
            assert observed <= receipt["array_interface_headers"]
    assert compact.Staging(rows, 13, "cpu", mtp=mtp).text_coordinates is None
