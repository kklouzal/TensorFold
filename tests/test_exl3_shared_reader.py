"""EXL3 startup reader: exact payloads, bounded queues and failure-safe lifetimes."""

import threading
import weakref

import pytest
import torch

from tensorfold.cuda.exl3 import host_experts as host
from tensorfold.families.qwen4_exp.cuda.exl3_pack import Pack
from flashnext_exl3_fixture import write_checkpoint
from test_exl3_host_experts import Checkpoint, PREFIX, SHARED, PROJECTIONS


def test_real_multilayer_pack_reuses_one_worker_and_preserves_original_bytes(tmp_path):
    write_checkpoint(tmp_path)
    pk = Pack(tmp_path)
    original_read, workers = pk.read, set()

    def tracked(file, begin, end):
        if threading.current_thread() is not threading.main_thread():
            workers.add(threading.current_thread())
        return original_read(file, begin, end)

    pk.read = tracked
    with host.CompactReadSession(pk) as reads:
        first_pool = None
        for base in ("model.language_model.layers.0.mlp", "model.language_model.layers.1.mlp", "mtp.layers.0.mlp"):
            authority, tables = host.load_compact(pk, base + ".experts", 3, base + ".shared_expert", reads=reads)
            first_pool = reads.ahead.pool if first_pool is None else first_pool
            assert reads.ahead.pool is first_pool and not reads.ahead.ahead and not reads.queued
            for expert in range(4):
                name = base + (f".experts.{expert}" if expert < 3 else ".shared_expert")
                for index, projection in enumerate(PROJECTIONS):
                    assert torch.equal(authority.projection(expert, index), pk.get(name + "." + projection + ".trellis"))
            assert tables.count == 4
        assert len(workers) == 1 and all(thread.is_alive() for thread in workers)
    assert reads.closed and reads.pk is None and reads.ahead.reader.pk is None
    assert all(not thread.is_alive() for thread in workers)
    assert not torch.cuda.is_initialized()


def test_multiple_bounded_windows_reuse_without_cloning_scale_views(monkeypatch):
    pk = Checkpoint(widths=((4, 4, 4),) * 16 + ((8, 8, 8),), dims=2560, width=640)
    original_read, validate = pk.read, host.native.validate_scale_payload
    returned, workers = set(), set()

    def tracked(file, begin, end):
        workers.add(threading.current_thread())
        value = original_read(file, begin, end)
        returned.add(value.untyped_storage().data_ptr())
        return value

    def scale_view(value, context):
        assert value.untyped_storage().data_ptr() in returned
        return validate(value, context)

    pk.read = tracked
    monkeypatch.setattr(host.native, "validate_scale_payload", scale_view)
    with host.CompactReadSession(pk) as reads:
        first, first_tables = host.load_compact(pk, PREFIX, pk.count, SHARED, reads=reads)
        second, second_tables = host.load_compact(pk, PREFIX, pk.count, SHARED, reads=reads)
        assert torch.equal(first.data, second.data)
        assert torch.equal(first_tables.suh_g.view(torch.int16), second_tables.suh_g.view(torch.int16))
        assert len(workers) == 1 and not reads.ahead.ahead and not reads.queued
    assert max(end - begin for _, begin, end in pk.range_reads) <= 16 << 20
    assert sum(end - begin for _, begin, end in pk.range_reads) == 2 * sum(t.nbytes for t in pk.values.values())
    assert all(not thread.is_alive() for thread in workers)


def test_consumer_failure_drains_poison_and_retains_primary_through_shutdown_retry(monkeypatch):
    pk, workers = Checkpoint(), set()
    original_read = pk.read

    def tracked(file, begin, end):
        workers.add(threading.current_thread())
        return original_read(file, begin, end)

    pk.read = tracked
    primary = ValueError("primary finite-payload rejection")
    close_failure = RuntimeError("first shutdown interrupted")
    reads = host.CompactReadSession(pk)
    with pytest.raises(ValueError) as caught:
        with reads:
            host.load_compact(pk, PREFIX, pk.count, SHARED, reads=reads)
            shutdown = reads.ahead.pool.shutdown
            calls = []

            def interrupted(*args, **kwargs):
                calls.append(None)
                if len(calls) == 1:
                    raise close_failure
                return shutdown(*args, **kwargs)

            monkeypatch.setattr(reads.ahead.pool, "shutdown", interrupted)

            def reject(*args):
                raise primary

            monkeypatch.setattr(host.native, "validate_scale_payload", reject)
            host.load_compact(pk, PREFIX, pk.count, SHARED, reads=reads)
    assert caught.value is primary and reads.closed and reads.poisoned
    assert len(calls) == 2 and all(not thread.is_alive() for thread in workers)
    assert any("first shutdown interrupted" in note for note in primary.__notes__)
    assert not reads.ahead.ahead and not reads.queued


def test_late_queued_failure_is_observed_before_reader_scope_exits(monkeypatch):
    pk = Checkpoint()
    for expert, name in enumerate(pk.names):
        for key in pk.where:
            if key.startswith(name + "."):
                pk.where[key] = pk.headers[key][0] = f"frame{expert}.safetensors"
    failed, workers, active = threading.Event(), set(), []
    read, unpack = pk.read, host.struct.unpack

    def fault(file, begin, end):
        workers.add(threading.current_thread())
        active.append(file)
        try:
            if file == "frame1.safetensors":
                failed.set()
                raise OSError("late queued reader failure")
            value = read(file, begin, end)
            if file == "frame0.safetensors":
                offset = pk.headers[PREFIX + ".0.gate_proj.mul1"][1] - begin
                value[offset] ^= 1
            return value
        finally:
            active.remove(file)

    def marker(code, value):
        assert failed.wait(5)
        return unpack(code, value)

    pk.read = fault
    monkeypatch.setattr(host.struct, "unpack", marker)
    reads = host.CompactReadSession(pk)
    with reads:
        with pytest.raises(ValueError, match="marker payload") as caught:
            host.load_compact(pk, PREFIX, pk.count, SHARED, reads=reads)
        assert any("late queued reader failure" in note for note in caught.value.__notes__)
        assert reads.poisoned and not active and not reads.ahead.ahead and not reads.queued
        before = len(pk.reads)
        with pytest.raises(RuntimeError, match="unavailable"):
            host.load_compact(pk, PREFIX, pk.count, SHARED, reads=reads)
        assert len(pk.reads) == before
    assert all(not thread.is_alive() for thread in workers)


def test_ownership_overlap_and_closed_boundaries_are_rejected_before_reads():
    pk = Checkpoint()
    with host.CompactReadSession(pk) as reads:
        with pytest.raises(ValueError, match="checkpoint Pack"):
            host.load_compact(Checkpoint(), PREFIX, pk.count, SHARED, reads=reads)
        with host._joint_payloads(pk, [], reads):
            with pytest.raises(RuntimeError, match="unavailable"):
                host.load_compact(pk, PREFIX, pk.count, SHARED, reads=reads)
            with pytest.raises(RuntimeError, match="consuming"):
                reads.close()
        errors = []

        def foreign():
            try:
                host.load_compact(pk, PREFIX, pk.count, SHARED, reads=reads)
            except RuntimeError as error:
                errors.append(str(error))

        worker = threading.Thread(target=foreign)
        worker.start()
        worker.join(5)
        assert not worker.is_alive() and errors == ["EXL3 read session is owned by another thread"]
    with pytest.raises(RuntimeError, match="unavailable"):
        host.load_compact(pk, PREFIX, pk.count, SHARED, reads=reads)
    assert pk.reads == [] and pk.range_reads == []


def test_closed_session_releases_borrowed_pack_without_closing_external_owner():
    pk = Checkpoint()
    reference = weakref.ref(pk)
    reads = host.CompactReadSession(pk)
    with reads:
        host.load_compact(pk, PREFIX, pk.count, SHARED, reads=reads)
    assert reads.closed
    pk = None
    assert reference() is None


@pytest.mark.parametrize("kind", ["cancel", "drain"])
def test_cleanup_interruption_finalizes_layer_and_owner_joins_preserving_primary(monkeypatch, kind):
    pk, workers = Checkpoint(), set()
    original_read = pk.read

    def tracked(file, begin, end):
        workers.add(threading.current_thread())
        return original_read(file, begin, end)

    pk.read = tracked
    primary = ValueError("primary consumer rejection")
    cleanup = KeyboardInterrupt("cancel interrupted") if kind == "cancel" else RuntimeError("drain helper interrupted")
    observed, interrupted = [], []
    reads = host.CompactReadSession(pk)
    if kind == "cancel":
        original_queue = reads.ahead.queue

        def queue(*args, **kwargs):
            original_queue(*args, **kwargs)
            for future in dict.fromkeys(reads.ahead.ahead.values()):
                observed.append(future)
                original_cancel = future.cancel

                def cancel(original_cancel=original_cancel):
                    if not interrupted:
                        interrupted.append(True)
                        raise cleanup
                    return original_cancel()

                monkeypatch.setattr(future, "cancel", cancel)

        monkeypatch.setattr(reads.ahead, "queue", queue)
    else:
        original_finish = host._finish_reads

        def finish(*args, close=True):
            if not close and not interrupted:
                interrupted.append(True)
                raise cleanup
            return original_finish(*args, close=close)

        monkeypatch.setattr(host, "_finish_reads", finish)

    def reject(*args):
        raise primary

    monkeypatch.setattr(host.native, "validate_scale_payload", reject)
    with pytest.raises(ValueError) as caught:
        with reads:
            host.load_compact(pk, PREFIX, pk.count, SHARED, reads=reads)
    assert caught.value is primary and interrupted
    assert reads.closed and reads.poisoned and not reads.active
    assert not reads.ahead.ahead and not reads.queued and reads.ahead.reader.pk is None
    assert all(future.done() for future in observed)
    assert all(not thread.is_alive() for thread in workers)
    assert any(str(cleanup) in note for note in primary.__notes__)


def test_repeated_drain_failure_finally_observes_actual_failed_future(monkeypatch):
    pk, workers = Checkpoint(), set()
    started = threading.Event()
    original_read = pk.read
    reads = host.CompactReadSession(pk)
    primary = ValueError("primary load rejection")
    secondary = OSError("asynchronous reader failure after helper failures")
    drain = RuntimeError("repeated drain helper failure")

    def read(file, begin, end):
        workers.add(threading.current_thread())
        started.set()
        raise secondary

    def broken(*args, **kwargs):
        raise drain

    with pytest.raises(ValueError) as caught:
        with reads:
            # A real ReadAhead future, queued while its borrowed Pack is alive.
            pk.read = read
            _, begin, end, _, _ = pk.entry(PREFIX + ".0.gate_proj.trellis")
            reads.ahead.queue([("pending", "fixture.safetensors", begin, end, None)], cut=lambda raw, _: raw)
            future = next(iter(reads.ahead.ahead.values()))
            reads.queued[future] = None
            assert started.wait(5)
            monkeypatch.setattr(host, "_finish_reads", broken)
            raise primary
    pk.read = original_read
    assert caught.value is primary and reads.closed and reads.poisoned
    assert reads.ahead.pool is None and not reads.queued and not reads.ahead.ahead
    assert all(not worker.is_alive() for worker in workers) and future.done()
    assert any(str(secondary) in note for note in primary.__notes__)
    assert any(str(drain) in note for note in primary.__notes__)
    assert reads.pk is None and reads.ahead.reader.pk is None


def test_unobserved_status_retains_pack_and_cannot_report_closed(monkeypatch):
    pk = Checkpoint()
    reads = host.CompactReadSession(pk)
    _, begin, end, _, _ = pk.entry(PREFIX + ".0.gate_proj.trellis")
    reads.ahead.queue([("pending", "fixture.safetensors", begin, end, None)], cut=lambda raw, _: raw)
    future = next(iter(reads.ahead.ahead.values()))
    future.result(timeout=5)
    reads.queued[future] = None
    original_exception = future.exception

    def unobservable(*args, **kwargs):
        raise RuntimeError("future status temporarily unobservable")

    monkeypatch.setattr(future, "exception", unobservable)
    with pytest.raises(RuntimeError, match="unobservable"):
        reads.close()
    assert reads.ahead.pool is None and not reads.closed and reads.poisoned
    assert reads.pk is pk and reads.ahead.reader.pk is pk and future in reads.queued
    monkeypatch.setattr(future, "exception", original_exception)
    reads.close()  # resource-only recovery, never model work/reuse after failure
    assert reads.closed and not reads.queued and reads.pk is None and reads.ahead.reader.pk is None
