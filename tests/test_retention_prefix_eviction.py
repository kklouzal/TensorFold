# ruff: noqa: E402 - optional Torch must be checked before backend imports.
"""Actual catalog/TP CPU selection-to-reservation cache lifetime boundary."""

import threading
import time
from types import SimpleNamespace as NS

import pytest
pytest.importorskip("torch")
from torch.distributed import HashStore

from tensorfold.cuda.scheduler import AdmissionStatus
from tensorfold.cuda.streams import Stream
from tensorfold.families.qwen4_exp.cuda.input_preparation import install_input_preparation
from tensorfold.families.qwen4_exp.cuda.retained_catalog import LOCAL_PRINCIPAL, LogicalReference, RetentionCapacity, make_retention_catalog
from tensorfold.families.qwen4_exp.cuda.retained_host import RetentionIntent
from tensorfold.families.qwen4_exp.cuda.retention_policy import RetentionPolicy
from tensorfold.families.qwen4_exp.cuda.tp_scheduler import TensorParallelDecoder
from test_retained_multi_caller import decoder_for, owner

pytest_plugins = ("test_state_retention",)


@pytest.mark.parametrize("source_kind", ["unleased-optional", "leased-optional", "required-logical"])
def test_required_target_reservation_reclaims_only_unleased_optional_selected_prefix(materialized, monkeypatch, source_kind):
    probe, future, _wake = owner(materialized)
    states, weights = probe.states, probe.w
    probe.close(timeout=5)
    # Actual queried future claim, not a copied planner formula or larger quota.
    catalog = make_retention_catalog(RetentionPolicy("ram", 2, 10, future, 37, 74, 0),
        weights, states, prefix_limit=2, recovery_versions=2)
    assert catalog.future_claim() == future
    catalog.configure_admission(max_waiting_input_bytes=4096, max_pending_requests=2)
    decoder = decoder_for(catalog, monkeypatch)
    decoder._pass_rows, decoder.points = lambda: 4, None
    decoder.w.positions = object()
    decoder.w.meta.update(rank=0, world=1, coordinate_profile_identity="a" * 64)
    decoder.input_preparation_context = install_input_preparation(decoder.w)
    decoder.input_preparation_context.bind_decoder(decoder)
    peer = TensorParallelDecoder(decoder, NS(_abort_comm=False))
    # One CPU actor uses the real ordered TP dispatch. Eight-peer transport and
    # common ownership are separately exercised by the live cohort controls.
    peer.comm = NS(rank=0, world=1, store=HashStore())
    peer.configure_admission_policy("fifo")
    source_input = Stream([3, 7, 11, 9], 2, draft=True, st=materialized)
    parent_ref = LogicalReference(LOCAL_PRINCIPAL, "response:required", 1)
    if source_kind == "required-logical":
        source = catalog.reserve(parent_ref, future_bytes=future)
        catalog.capture_idle(source_input, handle=source)
        catalog.logical_reference_committed(parent_ref.principal, parent_ref.sequence, generation=1)
        catalog.progress()
    else:
        source = catalog.capture_prefix(source_input, principal=LOCAL_PRINCIPAL)
    held = catalog.borrow_prefix(source, principal=LOCAL_PRINCIPAL) if source_kind == "leased-optional" else None
    before, generation = materialized.rec.clone(), materialized.slot_generation
    target = LogicalReference(LOCAL_PRINCIPAL, "response:new", 1)
    stream = Stream([3, 7, 11, 9, 5], 1, draft=True, cancel=threading.Event())
    stream.principal, stream.admission_ticket = None, 7
    stream.admission_status = AdmissionStatus(7, time.monotonic(), 256, None)
    stream.retention_intent = RetentionIntent(target,
        parent_ref if source_kind == "required-logical" else None, (), "new")
    selected = []
    invoke = peer._invoke
    def recorded(body, source=None):
        if body["kind"] == "enqueue":
            selected.append((body["prefix_serial"], body["parent"]))
        return invoke(body, source)
    monkeypatch.setattr(peer, "_invoke", recorded)
    def wake():
        return None
    try:
        if source_kind == "unleased-optional":
            assert peer.stage_request(stream, wake=wake, max_waiting_input_bytes=4096, max_pending_requests=2) is None
            assert selected == [(source.serial, None)]  # Selected before its required-target reserve reclaimed it.
            assert not source.internal_owner and stream.retention_request is None
            assert stream.retention_handle.owners == {target: False}
            assert stream.retention_handle.phase == "reserved" and stream.st is None
            assert catalog.record()["future_backing_bytes"] == future
        else:
            with pytest.raises(RetentionCapacity):
                peer.stage_request(stream, wake=wake, max_waiting_input_bytes=4096, max_pending_requests=2)
            assert catalog.by_serial(source.serial) is source
            assert source.state is materialized and materialized.slot_generation == generation
            assert bool((materialized.rec == before).all())
            assert stream.retention_request is None and stream.retention_handle is None
            assert held is None or not held.closed
            assert catalog.record()["future_backing_bytes"] == future
        assert peer.failure is None and not peer.engine._abort_comm
    finally:
        if id(stream) in peer.wires:
            peer.retire_request(stream)
        if held is not None:
            held.close()
        if getattr(stream, "retention_handle", None) is not None:
            catalog.logical_reference_unavailable(target.principal, target.sequence, generation=1, reason="test complete")
            catalog.progress()
        if source.internal_owner or source.owners:
            catalog.release_owner(source)
        catalog.close(timeout=5)
