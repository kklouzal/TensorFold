"""CPU unit readiness and real replay/pixel lease ownership at controlled IO boundaries.

These fixtures use the actual store, lease and executor APIs. Pixel decoding is
controlled to test scheduling/lifetimes; installed fresh-worker parity has its
separate runtime control and is never inferred from these fixtures.
"""

import os
import tempfile
import threading
import time
from types import SimpleNamespace as NS
from dataclasses import replace

import numpy as np
import pytest

from tensorfold.thread_work import ThreadWork
from tensorfold.vision import media_worker
from tensorfold.vision.images_http import ImageInputError
from tensorfold.vision.media_frontend import MediaExecutor, PixelUnitLease, PreparedReplayMedia
from tensorfold.vision.media_work import MediaWork
from tensorfold.vision.replay import MediaBusy, MediaCancelled, MediaProbe, ReplayStore
from tests.test_media_replay import envelope, image_source


PROFILE, MEDIA = "a" * 64, "b" * 64
CONTEXT = dict(generation=3, profile_identity=PROFILE, media_identity=MEDIA)


def wait_ready(work, prepared, index=0):
    deadline = time.monotonic() + 3  # Test deadlock guard, not a production latency budget.
    while not work.ready(prepared, index, **CONTEXT):
        if time.monotonic() >= deadline:
            pytest.fail("controlled CPU unit did not publish readiness")
        threading.Event().wait(.001)


def make_prepared(tmp_path, monkeypatch, admitted=None):
    # Exercise real encoded staging without asking this lifetime fixture to
    # establish the separately pinned subprocess/native-codec installation.
    def stage(executor, record, output, cancelled, **_kwargs):
        assert record["operation"] == "stage"
        return media_worker.stage(record, executor.envelope, output.fileno())

    monkeypatch.setattr(MediaExecutor, "_run", stage)
    store = ReplayStore(tmp_path, envelope(max_replay_leases=2) if admitted is None else admitted)
    replay = store.stage([image_source(size=(32, 32))], allow_urls=False, cancelled=lambda: False)
    executor = MediaExecutor(store)
    probe = MediaProbe(replay.refs[0], store.envelope.identity, media_worker.PROCESSOR_ID,
        32, 32, 1, (), (), None, "encoded-orientation-pil-rgb-v1", (32, 32), (1, 2, 2), 1024, 4096)
    plan = NS(profile_identity=PROFILE, media_identity=MEDIA, token_ids=(1, 31),
              units=(NS(probe_index=0, run=NS(temporal_group=0)),))
    return PreparedReplayMedia(plan, (probe,), replay, executor)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    result = make_prepared(tmp_path, monkeypatch)
    yield result
    # A failed ownership test must remain visible rather than forcibly dropping claims.
    if result.executor.store._failure is None:
        result.close()


def pixel(executor, probe, index, value=1.):
    assert index == 0
    shape = (probe.grid_thw[1] * probe.grid_thw[2], 1536)
    values = np.full(shape, value, dtype="<f4")
    reservation = object()
    executor._reserve_staging(reservation, values.nbytes)
    held = executor.store.borrow_ref(probe.ref)
    handle = tempfile.TemporaryFile(dir=executor.store.root)
    handle.write(values.tobytes())
    handle.flush()
    return PixelUnitLease(executor, handle, values.nbytes, shape, held, reservation)


def test_ready_and_take_do_not_wait_for_cpu_decode_and_transfer_actual_lease(prepared, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def decode(probe, index, *, cancelled):
        entered.set()
        assert release.wait(3)
        assert not cancelled()
        return pixel(prepared.executor, probe, index, 7.)

    monkeypatch.setattr(prepared.executor, "pixel_unit", decode)
    work = MediaWork(prepared.executor)
    pixels = None
    try:
        work.check_health()
        assert work.request(prepared, 0, **CONTEXT)
        assert entered.wait(3)
        polled, result = threading.Event(), []

        def poll():
            result.extend((work.ready(prepared, 0, **CONTEXT), work.take_ready(prepared, 0, **CONTEXT)))
            polled.set()

        owned = ThreadWork(poll)
        thread = threading.Thread(target=owned.run)
        owned.launch(thread)
        try:
            assert polled.wait(1), "GPU readiness waited for controlled decoding"
            assert result == [False, None] and not release.is_set()
        finally:
            release.set()
            owned.drain(thread, timeout=3, cancel_unentered=False)
        wait_ready(work, prepared)
        assert prepared.replay._borrows == 2  # Queue borrow and completed pixel lease.
        pixels = work.take_ready(prepared, 0, **CONTEXT)
        assert type(pixels) is PixelUnitLease and prepared.replay._borrows == 1
        assert not work.ready(prepared, 0, **CONTEXT)
        with pixels.borrow() as borrowed:
            assert np.all(borrowed.values == 7.)
        with pytest.raises(RuntimeError, match="borrowers"):
            prepared.close()
    finally:
        release.set()
        work.close(timeout=3)
        if pixels is not None:
            pixels.close()
    assert prepared.replay._borrows == 0 and prepared.executor.store._staging == 0


def test_global_pending_and_one_ahead_bounds_include_ready_unconsumed_units(prepared, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def decode(probe, index, *, cancelled):
        entered.set()
        assert release.wait(3)
        if cancelled():
            raise MediaCancelled("cancelled")
        return pixel(prepared.executor, probe, index)

    monkeypatch.setattr(prepared.executor, "pixel_unit", decode)
    second = PreparedReplayMedia(prepared.plan, prepared.probes, prepared.replay, prepared.executor)
    third = PreparedReplayMedia(prepared.plan, prepared.probes, prepared.replay, prepared.executor)
    work = MediaWork(prepared.executor)
    pixels = None
    try:
        assert (work.pending_limit, work.worker_limit) == (1, 1)
        assert work.request(prepared, 0, **CONTEXT) and entered.wait(3)
        assert work.request(prepared, 0, **CONTEXT)  # Idempotence adds no borrow or queued job.
        assert not work.request(second, 0, **CONTEXT)
        assert not work.request(third, 0, **CONTEXT) and len(work._jobs) == 1
        assert prepared.replay._borrows == 1
        release.set()
        wait_ready(work, prepared)
        assert not work.request(second, 0, **CONTEXT)  # Completed but unconsumed is still one owned unit.
        pixels = work.take_ready(prepared, 0, **CONTEXT)
        assert work.request(third, 0, **CONTEXT)  # GPU caller may own current pixels while one next job runs.
        wait_ready(work, third)
        assert len(work._jobs) == 1 and not work.request(second, 0, **CONTEXT)
        assert work.cancel(third)
        assert not work._jobs
    finally:
        release.set()
        work.close(timeout=3)
        if pixels is not None:
            pixels.close()


def test_cancelled_unentered_callback_is_removed_before_encoded_release(prepared, monkeypatch):
    release = threading.Event()
    original = MediaWork._serve

    def delayed(owner):
        assert release.wait(3)
        original(owner)

    monkeypatch.setattr(MediaWork, "_serve", delayed)
    monkeypatch.setattr(prepared.executor, "pixel_unit", lambda *_a, **_kw: pytest.fail("cancelled queued decode ran"))
    work = MediaWork(prepared.executor)
    try:
        assert work.request(prepared, 0, **CONTEXT)
        assert prepared.replay._borrows == 1 and work.cancel(prepared)
        assert not work._jobs and not work._pending and prepared.replay._borrows == 0
    finally:
        release.set()
        work.close(timeout=3)


def test_cancellation_keeps_encoded_borrow_until_entered_callback_retires(prepared, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def decode(_probe, _index, *, cancelled):
        entered.set()
        assert release.wait(3)
        assert cancelled()
        raise MediaCancelled("cancelled")

    monkeypatch.setattr(prepared.executor, "pixel_unit", decode)
    work = MediaWork(prepared.executor)
    try:
        work.request(prepared, 0, **CONTEXT)
        assert entered.wait(3) and not work.cancel(prepared)
        assert prepared.replay._borrows == 1
        with pytest.raises(RuntimeError, match="borrowers"):
            prepared.close()
        release.set()
        wait_ready(work, prepared)
        assert work.cancel(prepared) and prepared.replay._borrows == 0
    finally:
        release.set()
        work.close(timeout=3)


@pytest.mark.parametrize("change", ["generation", "profile", "media", "plan", "probes"])
def test_stale_completion_never_transfers_a_late_pixel_lease(prepared, monkeypatch, change):
    monkeypatch.setattr(prepared.executor, "pixel_unit", lambda probe, index, **_kw: pixel(prepared.executor, probe, index))
    work = MediaWork(prepared.executor)
    try:
        work.request(prepared, 0, **CONTEXT)
        wait_ready(work, prepared)
        context = dict(CONTEXT)
        if change == "generation":
            context["generation"] += 1
        elif change in ("profile", "media"):
            context[change + "_identity"] = "c" * 64
        elif change == "plan":
            prepared.plan = NS(**vars(prepared.plan))
        else:
            prepared.probes = tuple(list(prepared.probes))
        with pytest.raises(ValueError, match="stale|current admitted"):
            work.take_ready(prepared, 0, **context)
        assert len(work._jobs) == 1 and prepared.replay._borrows == 2
        assert work.cancel(prepared)
    finally:
        work.close(timeout=3)


def test_failed_decode_is_reported_and_its_consumed_traceback_is_not_retained(prepared, monkeypatch):
    primary = ImageInputError("controlled decoder refused unit")

    def failed(*_args, **_kwargs):
        raise primary

    monkeypatch.setattr(prepared.executor, "pixel_unit", failed)
    work = MediaWork(prepared.executor)
    try:
        work.request(prepared, 0, **CONTEXT)
        wait_ready(work, prepared)
        controller = work._jobs[prepared].work
        with pytest.raises(ImageInputError) as caught:
            work.take_ready(prepared, 0, **CONTEXT)
        assert caught.value is primary and controller.failure is None
        assert prepared.replay._borrows == 0 and not work._jobs
        work.check_health()  # Ordinary bad input is retired, leaving other requests usable.
    finally:
        work.close(timeout=3)


def test_check_ready_retires_failed_cpu_unit_before_gpu_admission_and_cancel_is_inert(prepared, monkeypatch):
    primary = ImageInputError("bad CPU media before GPU publication")

    def failed(*_a, **_kw):
        raise primary

    monkeypatch.setattr(prepared.executor, "pixel_unit", failed)
    work = MediaWork(prepared.executor)
    try:
        work.request(prepared, 0, **CONTEXT)
        assert work._jobs[prepared].work.retired.wait(3)
        assert work.ready(prepared, 0, **CONTEXT)  # Existing completed-outcome contract is preserved.
        with pytest.raises(ImageInputError) as caught:
            work.check_ready(prepared, 0, **CONTEXT)
        assert caught.value is primary and not work._jobs and prepared.replay._borrows == 0
        assert not vars(primary).get("_tensorfold_retained_owners")
        assert work.cancel(prepared)  # Finish/cleanup cannot throw the already-consumed input error again.
        assert not work.check_ready(prepared, 0, **CONTEXT)
        work.check_health()
    finally:
        work.close(timeout=3)


def test_check_ready_keeps_successful_pixels_owned_until_actual_take(prepared, monkeypatch):
    monkeypatch.setattr(prepared.executor, "pixel_unit", lambda probe, index, **_kw: pixel(prepared.executor, probe, index))
    work, result = MediaWork(prepared.executor), None
    try:
        work.request(prepared, 0, **CONTEXT)
        assert work._jobs[prepared].work.retired.wait(3)
        lease = work._jobs[prepared].pixels
        assert work.check_ready(prepared, 0, **CONTEXT)
        assert work.check_ready(prepared, 0, **CONTEXT)
        assert work._jobs[prepared].pixels is lease and prepared.replay._borrows == 2
        result = work.take_ready(prepared, 0, **CONTEXT)
        assert result is lease and prepared.replay._borrows == 1
    finally:
        work.close(timeout=3)
        if result is not None:
            result.close()


def test_check_ready_failed_retirement_preserves_primary_and_all_owners(prepared, monkeypatch):
    primary, cleanup = ImageInputError("bad CPU media"), OSError("encoded borrow retirement failed")

    def failed(*_a, **_kw):
        raise primary

    monkeypatch.setattr(prepared.executor, "pixel_unit", failed)
    work = MediaWork(prepared.executor)
    work.request(prepared, 0, **CONTEXT)
    job = work._jobs[prepared]
    assert job.work.retired.wait(3)
    original = job.encoded.close

    def fail_close():
        raise cleanup

    monkeypatch.setattr(job.encoded, "close", fail_close)
    with pytest.raises(ImageInputError) as caught:
        work.check_ready(prepared, 0, **CONTEXT)
    assert caught.value is primary and primary.__cause__ is cleanup
    from tensorfold.families.qwen4_exp.cuda.execution_identity import _unretired

    assert vars(primary)["_tensorfold_retained_owners"][0] is job and _unretired(primary)
    assert work._jobs[prepared] is job and job.encoded is not None and prepared.replay._borrows == 1
    with pytest.raises(RuntimeError, match="closed or poisoned"):
        work.check_health()
    monkeypatch.setattr(job.encoded, "close", original)
    with pytest.raises(ImageInputError):
        work.close(timeout=3)
    assert work.closed and prepared.replay._borrows == 0


def test_http_health_inspection_is_read_only_while_coordinator_operation_is_held(prepared):
    work, observed, returned = MediaWork(prepared.executor), [], threading.Event()

    def inspect():
        try:
            work.check_health()
        except BaseException as error:
            observed.append(error)
        else:
            observed.append(None)
        returned.set()

    probe = ThreadWork(inspect)
    thread = threading.Thread(target=probe.run, name="http-media-health")
    try:
        with work._operation():
            probe.launch(thread)
            assert returned.wait(1), "HTTP health waited behind the held coordinator operation"
            assert observed == [None] and work._failure is None
        probe.drain(thread, timeout=3, cancel_unentered=False)
    finally:
        work.close(timeout=3)


def test_concurrent_http_health_keeps_authoritative_work_poison(prepared):
    work, observed, returned = MediaWork(prepared.executor), [], threading.Event()
    primary = OSError("coordinator retained unretired ownership")

    def inspect():
        try:
            work.check_health()
        except RuntimeError as error:
            observed.append(error.__cause__)
        returned.set()

    probe = ThreadWork(inspect)
    thread = threading.Thread(target=probe.run, name="http-media-health")
    with work._operation():
        work._set_failure(primary)
        prepared.executor.check_health()  # Executor-only inspection cannot see this owning coordinator failure.
        probe.launch(thread)
        assert returned.wait(1) and observed == [primary]
    probe.drain(thread, timeout=3, cancel_unentered=False)
    with pytest.raises(OSError) as caught:
        work.close(timeout=3)
    assert caught.value is primary and work.closed


def test_http_health_observes_closing_without_waiting_for_decoder_drain(prepared, monkeypatch):
    entered, release, inspected = threading.Event(), threading.Event(), threading.Event()
    observed = []

    def decode(probe, index, **_kw):
        entered.set()
        assert release.wait(3)
        return pixel(prepared.executor, probe, index)

    monkeypatch.setattr(prepared.executor, "pixel_unit", decode)
    work = MediaWork(prepared.executor)
    work.request(prepared, 0, **CONTEXT)
    assert entered.wait(3)
    closing = ThreadWork(lambda: work.close(timeout=3))
    close_thread = threading.Thread(target=closing.run, name="media-close")
    closing.launch(close_thread)

    def inspect():
        try:
            work.check_health()
        except RuntimeError as error:
            observed.append(str(error))
        inspected.set()

    probe = ThreadWork(inspect)
    probe_thread = threading.Thread(target=probe.run, name="http-media-health")
    try:
        with work._condition:
            assert work._condition.wait_for(lambda: work._closing, timeout=1)
        probe.launch(probe_thread)
        assert inspected.wait(1) and observed == ["CPU media work is closed or poisoned; retain its owners"]
        assert not release.is_set() and not closing.retired.is_set()
    finally:
        release.set()
        probe.drain(probe_thread, timeout=3)
        closing.drain(close_thread, timeout=3, cancel_unentered=False)
    assert work.closed


@pytest.mark.parametrize("poll", ["request", "ready", "take"])
def test_typed_temporary_pressure_requeues_only_on_poll_with_same_encoded_owner(prepared, monkeypatch, poll):
    calls = []

    def decode(probe, index, **_kw):
        calls.append((probe, index))
        if len(calls) == 1:
            raise MediaBusy("temporary admitted staging capacity is busy")
        return pixel(prepared.executor, probe, index)

    monkeypatch.setattr(prepared.executor, "pixel_unit", decode)
    work, pixels = MediaWork(prepared.executor), None
    try:
        work.request(prepared, 0, **CONTEXT)
        job = work._jobs[prepared]
        previous, encoded = job.work, job.encoded
        assert previous.retired.wait(3) and type(previous.failure) is MediaBusy
        assert len(calls) == 1  # No worker-side retry or sleep loop.
        if poll == "request":
            assert work.request(prepared, 0, **CONTEXT)
        elif poll == "ready":
            assert not work.ready(prepared, 0, **CONTEXT)
        else:
            assert work.take_ready(prepared, 0, **CONTEXT) is None
        assert previous.failure is None and job.encoded is encoded
        wait_ready(work, prepared)
        pixels = work.take_ready(prepared, 0, **CONTEXT)
        assert calls == [(prepared.probes[0], 0)] * 2
    finally:
        work.close(timeout=3)
        if pixels is not None:
            pixels.close()


def test_typed_busy_with_secondary_cleanup_failure_is_not_retried(prepared, monkeypatch):
    primary, secondary, calls = MediaBusy("capacity busy"), OSError("retirement status failed"), []

    def failed(*_args, **_kwargs):
        calls.append(True)
        raise primary from secondary

    monkeypatch.setattr(prepared.executor, "pixel_unit", failed)
    work = MediaWork(prepared.executor)
    try:
        work.request(prepared, 0, **CONTEXT)
        assert work._jobs[prepared].work.retired.wait(3)
        assert work.ready(prepared, 0, **CONTEXT)
        with pytest.raises(MediaBusy) as caught:
            work.take_ready(prepared, 0, **CONTEXT)
        assert caught.value is primary and primary.__cause__ is secondary and calls == [True]
        assert not work._jobs and prepared.replay._borrows == 0
    finally:
        work.close(timeout=3)


def test_typed_busy_cannot_hide_poisoned_process_ownership(prepared, monkeypatch):
    primary, calls = RuntimeError("worker retirement unresolved"), []

    def failed(*_args, **_kwargs):
        calls.append(True)
        prepared.executor.store._failure = primary
        raise MediaBusy("capacity busy")

    monkeypatch.setattr(prepared.executor, "pixel_unit", failed)
    work = MediaWork(prepared.executor)
    work.request(prepared, 0, **CONTEXT)
    job = work._jobs[prepared]
    assert job.work.retired.wait(3)
    with pytest.raises(RuntimeError, match="retirement unresolved"):
        work.ready(prepared, 0, **CONTEXT)
    assert calls == [True] and job.encoded is not None and prepared.replay._borrows == 1
    # Explicit fixture containment only; shipping code never repairs this latch.
    prepared.executor.store._failure = None
    work.close(timeout=3)


def test_pixel_retirement_failure_keeps_all_claims_and_never_forces_readers_closed(prepared, monkeypatch):
    monkeypatch.setattr(prepared.executor, "pixel_unit", lambda probe, index, **_kw: pixel(prepared.executor, probe, index))
    work = MediaWork(prepared.executor)
    work.request(prepared, 0, **CONTEXT)
    wait_ready(work, prepared)
    job = work._jobs[prepared]
    reader = job.pixels.borrow()
    with pytest.raises(RuntimeError, match="outstanding"):
        work.cancel(prepared)
    assert job.pixels is not None and job.encoded is not None and prepared.replay._borrows == 2
    assert not job.pixels._closed and reader._lease is job.pixels
    prepared.executor.check_health()  # A local retirement failure must still poison this coordinator.
    with pytest.raises(RuntimeError, match="closed or poisoned"):
        work.check_health()
    reader.close()
    with pytest.raises(RuntimeError, match="outstanding"):
        work.close(timeout=3)
    assert work.closed and not work._jobs and prepared.replay._borrows == 0


def test_partial_native_thread_launch_is_drained_before_constructor_failure(prepared, monkeypatch):
    primary, owners = RuntimeError("thread launch returned an uncertain status"), []
    original = ThreadWork.launch

    def failed(work, thread):
        owners.append(work._callback.__self__)
        original(work, thread)
        raise primary

    monkeypatch.setattr(ThreadWork, "launch", failed)
    with pytest.raises(RuntimeError) as caught:
        MediaWork(prepared.executor)
    assert caught.value is primary and len(owners) == 1
    assert owners[0].closed and all(work.retired.is_set() for work, _thread in owners[0]._workers)
    assert prepared.replay._borrows == 0


def test_uncertain_process_retirement_keeps_job_and_encoded_claim(prepared, monkeypatch):
    primary = RuntimeError("existing process session was not retired")

    def failed(*_args, **_kwargs):
        prepared.executor.store._failure = primary
        raise primary

    monkeypatch.setattr(prepared.executor, "pixel_unit", failed)
    work = MediaWork(prepared.executor)
    work.request(prepared, 0, **CONTEXT)
    job = work._jobs[prepared]
    assert job.work.retired.wait(3)
    with pytest.raises(RuntimeError, match="retirement unresolved"):
        work.take_ready(prepared, 0, **CONTEXT)
    assert job.encoded is not None and prepared.replay._borrows == 1
    with pytest.raises(RuntimeError):
        work.close(timeout=3)
    assert not work.closed and work._jobs[prepared] is job and job.encoded is not None
    # Test-only removal of the injected latch represents completed external
    # containment; shipping code never guesses or clears this authority.
    prepared.executor.store._failure = None
    try:
        work.close(timeout=3)
    except RuntimeError:
        assert work.closed
    assert prepared.replay._borrows == 0


def test_timeout_retains_owners_and_never_publishes_after_cancel(prepared, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def decode(probe, index, **_kw):
        entered.set()
        assert release.wait(3)
        return pixel(prepared.executor, probe, index)

    monkeypatch.setattr(prepared.executor, "pixel_unit", decode)
    work = MediaWork(prepared.executor)
    work.request(prepared, 0, **CONTEXT)
    assert entered.wait(3)
    with pytest.raises(TimeoutError):
        work.close(timeout=0)
    assert not work.closed and work._jobs[prepared].encoded is not None
    assert prepared.replay._borrows == 1
    release.set()
    assert work._jobs[prepared].work.retired.wait(3)
    with pytest.raises(RuntimeError, match="closed or poisoned"):
        work.ready(prepared, 0, **CONTEXT)
    with pytest.raises(TimeoutError):
        work.close(timeout=3)
    assert work.closed and prepared.replay._borrows == 0 and prepared.executor.store._staging == 0


def test_foreign_process_cannot_poll_or_retire_owner(prepared, monkeypatch):
    work = MediaWork(prepared.executor)
    try:
        monkeypatch.setattr(work, "_pid", os.getpid() + 1)
        with pytest.raises(RuntimeError, match="another process"):
            work.ready(prepared, 0, **CONTEXT)
        with pytest.raises(RuntimeError, match="another process"):
            work.close()
    finally:
        work._pid = os.getpid()
        work.close(timeout=3)


def test_bad_identity_or_unit_is_refused_before_borrow(prepared):
    work = MediaWork(prepared.executor)
    try:
        for context in ({**CONTEXT, "generation": True}, {**CONTEXT, "profile_identity": ""}):
            with pytest.raises(ValueError):
                work.request(prepared, 0, **context)
        for index in (True, -1, 1):
            with pytest.raises(ValueError):
                work.request(prepared, index, **CONTEXT)
        assert prepared.replay._borrows == 0 and not work._jobs
    finally:
        work.close(timeout=3)


def test_impossible_replay_borrow_envelope_is_refused_before_threads(tmp_path, monkeypatch):
    executor = MediaExecutor(ReplayStore(tmp_path, envelope(max_replay_borrows=1)))
    monkeypatch.setattr(ThreadWork, "launch", lambda *_a: pytest.fail("invalid envelope started CPU worker"))
    with pytest.raises(ValueError, match="encoded and pixel replay borrows"):
        MediaWork(executor)


def test_unrepresentable_retirement_wait_is_refused_before_threads(tmp_path, monkeypatch):
    executor = MediaExecutor(ReplayStore(tmp_path, envelope(max_worker_wall_seconds=1e308, retirement_grace_seconds=1e308)))
    monkeypatch.setattr(ThreadWork, "launch", lambda *_a: pytest.fail("overflowed wait started CPU worker"))
    with pytest.raises(ValueError, match="wait budget must be representable"):
        MediaWork(executor)


def test_finite_default_above_runtime_wait_limit_is_refused_before_threads(tmp_path, monkeypatch):
    executor = MediaExecutor(ReplayStore(tmp_path, envelope(max_worker_wall_seconds=threading.TIMEOUT_MAX,
                                                          retirement_grace_seconds=1.)))
    monkeypatch.setattr(ThreadWork, "launch", lambda *_a: pytest.fail("unsupported finite wait started CPU worker"))
    with pytest.raises(ValueError, match="threading.TIMEOUT_MAX"):
        MediaWork(executor)


def test_finite_explicit_close_above_runtime_wait_limit_preserves_active_work(prepared, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def decode(probe, index, **_kw):
        entered.set()
        assert release.wait(3)
        return pixel(prepared.executor, probe, index)

    monkeypatch.setattr(prepared.executor, "pixel_unit", decode)
    work = MediaWork(prepared.executor)
    try:
        work.request(prepared, 0, **CONTEXT)
        assert entered.wait(3)
        job = work._jobs[prepared]
        with pytest.raises(ValueError, match="threading.TIMEOUT_MAX"):
            work.close(timeout=threading.TIMEOUT_MAX * 2)
        assert not work._closing and not work.closed and not job.cancelled.is_set()
        assert job.encoded is not None and prepared.replay._borrows == 1
        work.check_health()
    finally:
        release.set()
        work.close(timeout=3)


def test_effective_worker_count_respects_all_admitted_limits(tmp_path):
    executor = MediaExecutor(ReplayStore(tmp_path, envelope(max_workers=8, max_replay_leases=4, max_replay_borrows=4)))
    work = MediaWork(executor)
    try:
        assert work.record() == {"workers": 2, "pending_limit": 2, "owned_units": 0,
                                 "one_ahead_per_request": True, "closing": False, "closed": False}
    finally:
        work.close(timeout=3)


def test_initial_encoded_borrow_pressure_backpressures_without_a_callback(prepared, monkeypatch):
    monkeypatch.setattr(prepared.executor, "pixel_unit", lambda *_a, **_kw: pytest.fail("borrow-busy request decoded"))
    work = MediaWork(prepared.executor)
    holders = [prepared.borrow_replay() for _ in range(prepared.executor.envelope.max_replay_borrows)]
    try:
        assert not work.request(prepared, 0, **CONTEXT)
        assert not work._jobs and not work._pending
        assert prepared.replay._borrows == len(holders)
    finally:
        for held in holders:
            held.close()
        work.close(timeout=3)


def test_factory_publishes_close_safe_owner_before_validation_or_launch(prepared, monkeypatch):
    published = []
    monkeypatch.setattr(ThreadWork, "launch", lambda *_a: pytest.fail("invalid construction launched a thread"))
    with pytest.raises(TypeError, match="actual MediaExecutor"):
        MediaWork.create(None, register=published.append)
    assert len(published) == 1 and published[0]._workers == []
    published[0].close()
    assert published[0].closed


def test_factory_registration_failure_has_no_background_reader(prepared, monkeypatch):
    primary, published = ValueError("owner journal publication failed"), []

    def register(owner):
        published.append(owner)
        raise primary

    monkeypatch.setattr(ThreadWork, "launch", lambda *_a: pytest.fail("failed publication launched a thread"))
    with pytest.raises(ValueError) as caught:
        MediaWork.create(prepared.executor, register=register)
    assert caught.value is primary
    published[0].close()
    assert published[0].closed and prepared.replay._borrows == 0


def test_shared_store_capacity_notification_is_outside_locks_and_only_after_release(prepared):
    executor, events = prepared.executor, []
    store = executor.store
    other = MediaExecutor(store)

    def wake():
        assert not store._lock._is_owned() and not executor._lock._is_owned()
        events.append(executor.unit_available(prepared.probes[0], replay_borrows=2))

    executor.set_capacity_wake(wake)
    reservation, token = object(), object()
    try:
        assert executor.unit_available(prepared.probes[0], replay_borrows=2)
        store._reserve_staging(reservation, store.envelope.max_staging_bytes)
        assert not executor.unit_available(prepared.probes[0], replay_borrows=2) and not events
        other._release_staging(reservation)
        assert events == [True]
        other._release_staging(reservation)
        assert events == [True]  # No invented availability event for an absent claim.
        with store._lock:
            store._active_workers[0] = token
        assert not executor.unit_available(prepared.probes[0], replay_borrows=2)
        other._retire_worker(0, object())
        assert events == [True]
        other._retire_worker(0, token)
        assert events == [True, True]
        borrowed = prepared.borrow_replay()
        borrowed.close()
        borrowed.close()
        assert events == [True, True, True]
    finally:
        executor.clear_capacity_wake(wake)


def test_irreducible_pixel_plus_metadata_peak_is_permanent_before_any_claim(tmp_path, monkeypatch):
    ordinary = envelope(max_replay_leases=2, max_upload_bytes=4096)
    numeric = np.empty((4, 1536), dtype="<f4").nbytes
    admitted = replace(ordinary, max_staging_bytes=numeric + ordinary.max_manifest_bytes + 2 * ordinary.max_result_bytes - 1)
    prepared = make_prepared(tmp_path, monkeypatch, admitted)
    try:
        assert numeric < admitted.max_staging_bytes
        with pytest.raises(ImageInputError, match="pixel and metadata peak") as caught:
            prepared.executor.unit_available(prepared.probes[0], replay_borrows=2)
        assert type(caught.value) is ImageInputError
        with pytest.raises(ImageInputError, match="pixel and metadata peak"):
            prepared.executor.pixel_unit(prepared.probes[0], 0, cancelled=lambda: False)
        assert prepared.replay._borrows == 0 and prepared.executor.store._staging == 0
    finally:
        prepared.close()


def test_busy_parks_without_spin_and_genuine_external_release_wakes_scheduler(prepared, monkeypatch):
    calls, released, woken = [], object(), threading.Event()
    executor = prepared.executor

    def decode(probe, index, **_kw):
        calls.append(True)
        if len(calls) == 1:
            executor.store._reserve_staging(released, executor.envelope.max_staging_bytes)
            raise MediaBusy("another operation acquired current capacity")
        return pixel(executor, probe, index)

    monkeypatch.setattr(executor, "pixel_unit", decode)
    work, result = MediaWork(executor), None
    callback = woken.set
    work.set_wake(callback)
    try:
        assert work.request(prepared, 0, **CONTEXT)
        job = work._jobs[prepared]
        assert job.work.retired.wait(3)
        assert woken.wait(3)
        woken.clear()
        for _ in range(20):
            assert work.request(prepared, 0, **CONTEXT)
            assert not work.ready(prepared, 0, **CONTEXT)
            assert work.take_ready(prepared, 0, **CONTEXT) is None
        assert calls == [True] and job.encoded is not None
        MediaExecutor(executor.store)._release_staging(released)  # Not a Work-owned operation or request arrival.
        assert woken.wait(3)
        wait_ready(work, prepared)
        result = work.take_ready(prepared, 0, **CONTEXT)
        assert calls == [True, True]
    finally:
        work.close(timeout=3)
        if result is not None:
            result.close()


def test_completion_wake_outside_locks_and_repeated_logical_subrun_has_no_tombstone(prepared, monkeypatch):
    monkeypatch.setattr(prepared.executor, "pixel_unit", lambda probe, index, **_kw: pixel(prepared.executor, probe, index))
    work, woken = MediaWork(prepared.executor), threading.Event()

    def wake():
        assert not work._condition._is_owned() and not work._control._is_owned()
        assert not prepared.executor.store._lock._is_owned()
        woken.set()

    work.set_wake(wake)
    result = None
    try:
        assert woken.wait(3)
        woken.clear()
        work.request(prepared, 0, **CONTEXT)
        assert woken.wait(3)
        wait_ready(work, prepared)
        assert work.cancel(prepared) and not prepared.replay._closed
        prepared.plan = NS(**{**vars(prepared.plan), "token_ids": (*prepared.plan.token_ids, 42)})
        context = {**CONTEXT, "generation": 4}
        assert work.request(prepared, 0, **context)
        assert work._jobs[prepared].work.retired.wait(3)
        result = work.take_ready(prepared, 0, **context)
        assert result is not None and not work._jobs
        with pytest.raises(RuntimeError, match="another scheduler"):
            work.set_wake(lambda: None)
    finally:
        work.close(timeout=3)
        if result is not None:
            result.close()


def test_wake_failure_poison_is_reported_without_leaking_completed_unit(prepared, monkeypatch):
    from tensorfold.cuda import cohort

    class WakeFailure(RuntimeError):
        def __bool__(self):
            pytest.fail("failure ownership inspected exception truthiness")

    primary, notified = WakeFailure("scheduler notification failed"), threading.Event()
    monkeypatch.setattr(cohort, "notify_failure", lambda: notified.set())
    work = MediaWork(prepared.executor)

    def failed():
        raise primary

    work.set_wake(failed)
    assert notified.wait(3)
    with pytest.raises(RuntimeError, match="closed or poisoned"):
        work.check_health()
    with pytest.raises(RuntimeError) as caught:
        work.close(timeout=3)
    assert caught.value is primary and work.closed and prepared.replay._borrows == 0


def test_capacity_notification_failure_preserves_body_primary_and_consumed_claim(prepared):
    primary, notification = ValueError("original media operation failed"), RuntimeError("capacity notification failed")

    def failed():
        raise notification

    prepared.executor.set_capacity_wake(failed)
    held = prepared.borrow_replay()
    try:
        with pytest.raises(ValueError) as caught:
            with held:
                raise primary
        assert caught.value is primary and primary.__cause__ is notification
        assert held._lease is None and prepared.replay._borrows == 0
        with pytest.raises(RuntimeError, match="retirement unresolved"):
            prepared.executor.check_health()
    finally:
        prepared.executor.clear_capacity_wake(failed)
        prepared.executor.store._failure = None  # Explicit fixture containment, never shipping repair.


def test_pixel_capacity_notification_follows_whole_close_outside_executor_lock(prepared):
    events = []
    lease = pixel(prepared.executor, prepared.probes[0], 0)

    def released():
        assert not prepared.executor.store._lock._is_owned() and not prepared.executor._lock._is_owned()
        assert lease._closed and lease._mapping is None and lease._replay is None
        events.append(True)

    prepared.executor.set_capacity_wake(released)
    try:
        lease.close()
        lease.close()
        assert events == [True]
    finally:
        prepared.executor.clear_capacity_wake(released)


def test_failed_second_factory_never_detaches_existing_capacity_owner(prepared):
    first, published = MediaWork(prepared.executor), []
    try:
        with pytest.raises(RuntimeError, match="another coordinator"):
            MediaWork.create(prepared.executor, register=published.append)
        assert published[0].closed and not published[0]._workers
        assert prepared.executor.store._capacity_wake is first._capacity_callback
        first.check_health()
    finally:
        first.close(timeout=3)


def test_same_set_wake_before_next_worker_iteration_cannot_starve_pending_decode(prepared, monkeypatch):
    release, wakes, seen_at_decode = threading.Event(), [], []
    original = MediaWork._serve

    def delayed(owner):
        assert release.wait(3)
        original(owner)

    def decode(probe, index, **_kw):
        seen_at_decode.append(len(wakes))
        return pixel(prepared.executor, probe, index)

    monkeypatch.setattr(MediaWork, "_serve", delayed)
    monkeypatch.setattr(prepared.executor, "pixel_unit", decode)
    work, result = MediaWork(prepared.executor), None

    def wake():
        wakes.append(True)
        if len(wakes) == 1:
            # Deterministic scheduler admission retry before the worker can
            # acquire its condition for the next iteration.
            work.set_wake(wake)

    try:
        work.set_wake(wake)
        work.request(prepared, 0, **CONTEXT)
        release.set()
        wait_ready(work, prepared)
        result = work.take_ready(prepared, 0, **CONTEXT)
        assert seen_at_decode == [1]  # No artificial second control ahead of actual CPU work.
    finally:
        release.set()
        work.close(timeout=3)
        if result is not None:
            result.close()


def test_same_set_wake_under_pressure_returns_worker_to_idle_without_polling(prepared, monkeypatch):
    idle, wakes, reservation = threading.Event(), [], object()
    work = MediaWork(prepared.executor)
    prepared.executor.store._reserve_staging(reservation, prepared.executor.envelope.max_staging_bytes)
    original_wait = work._condition.wait_for

    def observed_wait(predicate, timeout=None):
        if not predicate():
            idle.set()
        return original_wait(predicate, timeout)

    monkeypatch.setattr(work._condition, "wait_for", observed_wait)

    def wake():
        wakes.append(True)
        if len(wakes) == 1:
            work.set_wake(wake)
            assert not work.request(prepared, 0, **CONTEXT)

    try:
        # Publish observer while holding the real condition so the recorded
        # idle point must follow this initial wake, never a constructor wait.
        with work._condition:
            idle.clear()
            work.set_wake(wake)
        assert idle.wait(3)
        assert wakes == [True] and not work._jobs and not work._wake_due
    finally:
        prepared.executor.store._release_staging(reservation)
        work.close(timeout=3)
