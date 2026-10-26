# ruff: noqa: E402 - optional Torch must be checked before backend imports.
"""Existing TP actor drives real catalog readers/expiry/transfer/ACK across eight CPU peers."""

from concurrent.futures import ThreadPoolExecutor
import threading
import json
import time
from types import SimpleNamespace as NS

import pytest
pytest.importorskip("torch")

from tensorfold.cuda.scheduler import AdmissionStatus
from tensorfold.cuda.streams import Stream
from tensorfold.families.qwen4_exp.cuda.input_preparation import install_input_preparation
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder
from tensorfold.families.qwen4_exp.cuda.retained_catalog import LogicalReference, make_retention_catalog
from tensorfold.families.qwen4_exp.cuda.retained_host import RetentionIntent
from tensorfold.families.qwen4_exp.cuda.retention_policy import RetentionPolicy
from tensorfold.families.qwen4_exp.cuda.tp_scheduler import TensorParallelDecoder
from test_retained_catalog_cohort import _cohort_states


def test_enqueue_readers_precede_common_expiry_and_actual_idle_ram_transfer_then_cancel():
    server, baseline, worlds, sources, destinations = _cohort_states()
    policy = RetentionPolicy("ram", 2, 30, 16 << 30, 128 << 10, 2 * 8 * (128 << 10), 0)
    with ThreadPoolExecutor(max_workers=8) as pool:
        catalogs = list(pool.map(lambda rank: make_retention_catalog(policy, worlds[rank],
            [sources[rank], destinations[rank]], prefix_limit=2, recovery_versions=0), range(8)))
    peers, handles, errors = [], [], []
    parent = LogicalReference("p", "response:parent", 1)
    target = LogicalReference("p", "response:child", 1)
    for rank, (catalog, weights, source, destination) in enumerate(zip(catalogs, worlds, sources, destinations)):
        handle = catalog.reserve(parent, future_bytes=catalog.future_claim())
        catalog.capture_idle(Stream([1] * 129, 4, draft=False, st=source), handle=handle)
        handles.append(handle)
        weights.positions = object()
        weights.cfg.vocab = 64
        weights.meta["coordinate_profile_identity"] = "a" * 64
        decoder = object.__new__(MultiDecoder)
        decoder.w, decoder.retention_catalog = weights, catalog
        decoder.states, decoder.free, decoder._kept = (source, destination), [destination], None
        decoder.streams, decoder.filling, decoder.held = {}, [], {}
        decoder.prefill_rows, decoder.points, decoder.arrived = 16, None, lambda: False
        decoder.capacity, decoder.depth, decoder.kv_pair = 257, 0, source.kv_pair
        decoder.memory_gate = NS(held=sum(state.cache_bytes() for state in decoder.states))
        decoder.graphs, decoder.full_resident = None, True
        decoder._pass_rows = lambda: 16
        decoder.input_preparation_context = install_input_preparation(weights)
        decoder.input_preparation_context.bind_decoder(decoder)
        peer = TensorParallelDecoder(decoder, NS(_abort_comm=False))
        peer.configure_admission_policy("fifo")
        peers.append(peer)
    root = peers[0]
    publications, received = [], [[] for _ in peers]
    invoke = root._invoke
    def traced_invoke(body, source=None):
        publications.append((root.sequence, body["kind"]))
        return invoke(body, source)
    root._invoke = traced_invoke
    wake = threading.Event()
    wake_callback = wake.set
    catalogs[0].set_wake(wake_callback)
    catalogs[0].logical_reference_committed(parent.principal, parent.sequence, generation=1)

    def follow(peer):
        try:
            while True:
                body = peer.receive()
                if body is None:
                    return
                received[peer.comm.rank].append((body["sequence"], body["kind"]))
                if body["kind"] == "enqueue":
                    assert peer.comm.store.check([peer._key(body["sequence"], peer.comm.rank)])
                peer.apply(body)
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=follow, args=(peer,), daemon=True) for peer in peers[1:]]
    for thread in threads:
        thread.start()
    stream = Stream([1] * 130, 2, draft=True, cancel=threading.Event())
    stream.principal, stream.admission_ticket = "p", 7
    stream.admission_status = AdmissionStatus(7, time.monotonic(), 4096, None, principal="p")
    stream.retention_intent = RetentionIntent(target, parent, (), "child")
    try:
        root.retention_boundary(idle=True)
        assert all(catalog.lookup(parent) is handle for catalog, handle in zip(catalogs, handles))
        # Slightly different follower clocks are deliberately irrelevant: only
        # this leader event is replicated, after queue reader publication.
        handles[0].expires = time.monotonic() - 1
        root.stage_request(stream, wake=wake_callback, max_waiting_input_bytes=1 << 20, max_pending_requests=4)
        assert all(len(handle.consumers) == 1 for handle in handles)
        assert all(peer.requests[0].retention_handle.serial == 2 for peer in peers)
        root.decoder.free.clear()  # Actual State pool occupied by a separate admitted window.
        root.retention_boundary(idle=True, needs_slot=True)
        assert all(handle.phase == "saving" for handle in handles)
        assert all(handle.transfer.retired.wait(10) for handle in handles)
        root.retention_boundary(idle=True)
        assert all(handle.phase == "ram" for handle in handles)
        assert all(catalog.lookup(parent) is None for catalog in catalogs)  # Owner expired identically.
        assert all(len(handle.consumers) == 1 for handle in handles)  # Already accepted readers remain pinned.
        stream.cancel.set()
        assert root.retire_request(stream)
        assert all(not handle.consumers for handle in handles)
        assert all(not peer.requests for peer in peers)
        catalogs[0].logical_reference_unavailable(target.principal, target.sequence, generation=1, reason="aborted")
        root.retention_boundary(idle=True)
        assert all(not catalog.has_retained for catalog in catalogs)
        assert root.retention_boundary(idle=True, shutdown=True)["drained"]
    finally:
        if root.failure is None:
            if getattr(stream, "_retention_staged", False) and root.wires.get(id(stream)) is not None:
                root.retire_request(stream)
            root.retention_boundary(idle=True, shutdown=True)
        publications.append((root.sequence, "stop"))
        root.shutdown()
        for thread in threads:
            thread.join(10)
            assert not thread.is_alive()
        for catalog in catalogs:
            catalog.close(timeout=10)
    assert errors == []
    if server.num_keys() != baseline:
        print(json.dumps({"keys": [(key, json.loads(server.get(key)).get("kind")) for key in server.list_keys()],
                          "published": publications, "received": received}))
    assert server.num_keys() == baseline, {
        "keys": [(key, json.loads(server.get(key)).get("kind")) for key in server.list_keys()],
        "published": publications, "received": received}
