# ruff: noqa: E402 - optional Torch must be checked before backend imports.
"""Actual FIFO/TP/catalog automatic live requeue over eight CPU Store peers.

The model step is controlled; real State/RAM/host/control ownership is exercised.
This is not CUDA, distributed-model output or throughput qualification.
"""

from concurrent.futures import ThreadPoolExecutor
import threading
import time
from types import SimpleNamespace as NS

import pytest
pytest.importorskip("torch")

from tensorfold.cuda.scheduler import Scheduler
from tensorfold.families.qwen4_exp.cuda.input_preparation import install_input_preparation
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder
from tensorfold.families.qwen4_exp.cuda.retained_catalog import make_retention_catalog
from tensorfold.families.qwen4_exp.cuda.retention_policy import RetentionPolicy
from tensorfold.families.qwen4_exp.cuda.retention_wake import RetentionWakeRelay
from tensorfold.families.qwen4_exp.cuda.tp_scheduler import TensorParallelDecoder
from test_flashnext_history_placement import _seed_zero_indexer_prefix
from test_retained_catalog_cohort import _cohort_states


def valid_prefix(decoder, state, length):
    _seed_zero_indexer_prefix(state, length)
    for ordinal, cache in enumerate(state.kc):
        for _consumed, offset, count in cache._spans(0, length):
            cache.k[offset:offset + count].fill_(decoder.w.comm.rank + ordinal + 1)
            cache.v[offset:offset + count].fill_(-decoder.w.comm.rank - ordinal - 1)


class Controlled(MultiDecoder):
    def admit(self, stream):
        if getattr(stream, "retained_host", None) is not None:
            self.restored.append((stream.admission_ticket, list(stream.out)))
            return super().admit(stream)
        state, _resume, _cached = self._slot_for(stream.prompt, False, principal=stream.principal)
        state.reset(self.w, new_sequence=False)
        valid_prefix(self, state, len(stream.prompt))
        state.cache_principal = stream.principal
        stream.sid, self.next_id = self.next_id, self.next_id + 1
        stream.st, stream.context = state, list(stream.prompt)
        stream.started = time.perf_counter()
        self.streams[stream.sid] = stream
        self.admitted.append((stream.prompt[0], stream.admission_ticket))

    def round(self):
        ended = []
        for stream in tuple(self.streams.values()):
            if not self._runnable(stream):
                continue
            if stream.out:
                stream.committed.append(stream.out[-1])
                valid_prefix(self, stream.st, stream.st.pos + 1)
            self._take(stream, [9 + len(stream.out)])
            if stream.prompt[0] == 1 and len(stream.out) == 1 and self.w.comm.rank == 0:
                if self.blockage == "input":
                    self.input_ready.clear()
                self.first.set()
            if stream.done:
                stream.finished = time.perf_counter()
                self._capture_retained(stream)
                ended.append(stream)
        return ended


@pytest.mark.parametrize("blockage", ["input", "output", "unbounded-output"])
def test_eligible_new_request_runs_while_old_same_request_is_in_ram_then_exact_ticket_resumes(blockage):
    server, baseline, worlds, sources, destinations = _cohort_states()
    for weights in worlds:
        weights.positions = object()
        weights.cfg.vocab = 64
        weights.meta["coordinate_profile_identity"] = "a" * 64
        install_input_preparation(weights)
    policy = RetentionPolicy("ram", 2, 30, 16 << 30, 128 << 10, 2 * 8 * (128 << 10), 0)
    with ThreadPoolExecutor(max_workers=8) as pool:
        catalogs = list(pool.map(lambda rank: make_retention_catalog(policy, worlds[rank],
            [sources[rank], destinations[rank]], prefix_limit=2, recovery_versions=4), range(8)))
    # Exercise the real CPU wake relay also used by CUDA catalogs. CPU-only
    # construction omits it because bare component callers can settle directly;
    # this automatic actor must receive every follower completion notification.
    for catalog in catalogs:
        RetentionWakeRelay.create(catalog.comm, "tf_retention/" + catalog.identity,
            wake=catalog._wake_local, fatal=catalog._wake_failed,
            register=lambda relay, owner=catalog: setattr(owner, "_wake_relay", relay))
        catalog._wake_relay.start()
    input_ready, first = threading.Event(), threading.Event()
    delivered, release_output = threading.Event(), threading.Event()
    input_ready.set()
    peers, errors = [], []
    for weights, catalog, states in zip(worlds, catalogs, zip(sources, destinations)):
        decoder = object.__new__(Controlled)
        decoder.w, decoder.states, decoder.free, decoder._kept = weights, states, list(states), []
        decoder.retention_catalog, decoder.keep = None, 2
        decoder.streams, decoder.filling, decoder.fills, decoder.held = {}, [], {}, {}
        decoder.controller, decoder.graphs, decoder.full_resident = None, None, True
        decoder.capacity, decoder.depth, decoder.prefill_rows = 257, 0, 16
        decoder.share, decoder.round_s, decoder.row_s, decoder.points = 0., None, None, None
        decoder.eos = ()
        decoder.buf = decoder.mbuf = decoder.pbuf = None
        decoder._draft_map, decoder.media_runtime, decoder.vision = None, None, None
        decoder.kv_pair, decoder.next_id, decoder.failed_slots = states[0].kv_pair, 0, []
        decoder.memory_gate = NS(held=sum(state.cache_bytes() for state in states))
        decoder.input_ready, decoder.first, decoder.blockage = input_ready, first, blockage
        decoder.admitted, decoder.restored = [], []
        decoder.input_preparation_context = weights.meta["input_preparation_context"]
        decoder.input_preparation_context.bind_decoder(decoder)
        decoder.bind_retention_catalog(catalog)
        peer = TensorParallelDecoder(decoder, NS(_abort_comm=False))
        peers.append(peer)
    root = peers[0]
    def follow(peer):
        try:
            while True:
                body = peer.receive()
                if body is None:
                    return
                peer.apply(body)
        except BaseException as error:
            errors.append(error)
    threads = [threading.Thread(target=follow, args=(peer,), daemon=True) for peer in peers[1:]]
    for thread in threads:
        thread.start()
    scheduler = Scheduler(root, max_streams=1, max_pending_requests=4,
        admission_policy="fifo", max_waiting_input_bytes=1 << 20,
        output_queue_tokens=1 if blockage == "output" else None).start()
    output, newer_output = [], []
    def emit(values):
        output.extend(values)
        if values == [9] and blockage != "input":
            delivered.set()
            assert release_output.wait(20)
    with ThreadPoolExecutor(max_workers=2) as clients:
        old = clients.submit(scheduler.submit, [1] * 129, 2, None, False,
            emit, eligibility=lambda: None if input_ready.is_set() else "client_input")
        if not first.wait(10):
            old.result(timeout=1)
            pytest.fail("controlled first model step did not run")
        if blockage != "input":
            assert delivered.wait(10)
        newer = clients.submit(scheduler.submit, [2] * 129, 1, None, False,
                               lambda values: newer_output.extend(values))
        try:
            newer.result(timeout=20)
            assert newer_output == [9] and not old.done()
            assert root.decoder.admitted[0][0] == 1 and root.decoder.admitted[1][0] == 2
            ticket = root.decoder.admitted[0][1]
            if blockage != "unbounded-output":
                stream = next(stream for stream in root.requests.values() if stream.prompt[0] == 1)
                assert stream.retention_handle.phase == "ram" and stream.st is None
                assert stream.admission_ticket == ticket and stream.out == [9]
            else:
                assert all(not peer.decoder.restored for peer in peers)  # Callback alone never preempts a runnable GPU sequence.
                assert not any(catalog.has_retained for catalog in catalogs)
            input_ready.set()
            release_output.set()
            scheduler.waiting.wake()
            result = old.result(timeout=20)
            assert output == [9, 10] and result["committed_output_tokens"] == 2
            if blockage != "unbounded-output":
                assert all(peer.decoder.restored and peer.decoder.restored[-1] == (ticket, [9]) for peer in peers)
            assert all(peer.decoder.live() == 0 and not catalog.has_retained for peer, catalog in zip(peers, catalogs))
        finally:
            input_ready.set()
            release_output.set()
            scheduler.waiting.wake()
            old.result(timeout=20)
            newer.result(timeout=20)
            scheduler.close()
    root.shutdown()
    for thread in threads:
        thread.join(10)
        assert not thread.is_alive()
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda catalog: catalog.close(timeout=10), catalogs))
    assert errors == [] and server.num_keys() == baseline
