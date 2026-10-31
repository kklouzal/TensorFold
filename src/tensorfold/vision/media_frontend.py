"""Bounded CPU media executor and opt-in source frontend for the SM70 encoder.

No GPU encoding, tokenizer substitution or service defaults live here. The
profile owner supplies the admitted envelope and canonical selected-tokenizer
planner. Only one normalized image or temporal pair is materialized per lease.
"""
from __future__ import annotations

from dataclasses import asdict
from contextlib import contextmanager
import json
import fcntl
import mmap
import os
import sys
import tempfile
import threading
import time

from tensorfold.cuda.expert_launch import supervise
from tensorfold.cleanup import finish, raise_failures, rollback
from .images_http import ImageInputError
from .replay import MediaBusy, MediaEnvelope, MediaProbe, ReplayStore, check_cancelled


@contextmanager
def _diagnostic_fd(limit, store, worker_index):
    """Journal the existing native owner before its sealable memfd acquisition."""
    from tensorfold._fd_owner import OwnedFD
    owner = OwnedFD()
    def close():
        if owner.closed:
            return
        # The native .closed state distinguishes a pre-entry interruption from
        # a consumed close. Never retry a numeric descriptor.
        try:
            owner.close()
        except BaseException as primary:
            try:
                if not owner.closed:
                    rollback(owner, primary, owner.close)
                raise
            finally:
                if not owner.closed:
                    # Retain before the first potentially fallible lock entry.
                    if store._retained_workers[worker_index] is None:
                        store._retained_workers[worker_index] = primary
                    with store._lock:
                        if store._failure is None:
                            store._failure = primary
        if not owner.closed:
            raise RuntimeError("media diagnostic descriptor did not retire")
    try:
        owner.memfd()
        os.ftruncate(owner.fileno(), limit)
        fcntl.fcntl(owner.fileno(), fcntl.F_ADD_SEALS, fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
        yield owner.fileno()
    except BaseException as primary:
        rollback(owner, primary, close)
    else:
        close()


class PixelUnitBorrow:
    def __init__(self, lease):
        self._lease = lease

    @property
    def values(self):
        """Read-only CPU FP32 NumPy rows, valid only while this borrow is held.

        Retain the borrow through asynchronous transfer completion. Consumers
        must not cast away read-only protection or mutate/reuse these pixels.
        """
        lease = self._lease
        if lease is None:
            raise RuntimeError("pixel unit borrow is retired")
        import numpy as np
        with lease._executor._lock:
            if self not in lease._borrowers:
                raise RuntimeError("pixel unit borrow is retired")
            # frombuffer retains a real exported Py_buffer. ndarray(buffer=mmap)
            # alone keeps a base reference but permits mmap.close under live arrays.
            return np.frombuffer(memoryview(lease._mapping), dtype="<f4").reshape(lease.shape)

    def close(self):
        lease = self._lease
        if lease is not None:
            with lease._executor._lock:
                if self._lease is None:
                    return
                lease._executor._check_owner()
                lease._borrowers.discard(self)
                self._lease = None

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        if error is not None:
            rollback(self, error, self.close)
        self.close()


class PixelUnitLease:
    """Immutable read-only mapping; encoder readers explicitly delay retirement."""
    def __init__(self, executor, handle, size, shape, replay, reservation):
        self._executor, self._handle, self.shape = executor, handle, tuple(shape)
        self._replay = replay
        self._borrowers, self._closed, self._retiring = set(), False, False
        self._reservation = reservation
        self._mapping = mmap.mmap(handle.fileno(), size, access=mmap.ACCESS_READ)

    @property
    def _borrows(self):
        return len(self._borrowers)

    def borrow(self):
        with self._executor._lock:
            self._executor._check_owner()
            if self._closed or self._retiring:
                raise RuntimeError("pixel unit lease is retired")
            if self._borrows >= self._executor.envelope.max_replay_borrows:
                raise ImageInputError("pixel unit borrower capacity is busy")
            result = PixelUnitBorrow(self)
            self._borrowers.add(result)
            return result

    def close(self):
        with self._executor._lock:
            self._executor._check_owner()
            if self._closed:
                return
            if self._borrows:
                raise RuntimeError("pixel unit still has outstanding encoder borrowers")
            self._retiring = True
            # A leaked exported buffer refuses close; retain mapping, file,
            # replay owner and capacity until the caller releases it and retries.
            if self._mapping is not None:
                self._mapping.close()
                self._mapping = None
            if self._handle is not None:
                if not self._handle.closed:
                    self._handle.close()
                self._handle = None
            if self._replay is not None:
                self._replay.close(_notify=False)
                self._replay = None
            self._executor._release_staging(self._reservation, notify=False)
            self._closed = True
        self._executor.store._notify_capacity()

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        if error is not None:
            rollback(self, error, self.close)
        self.close()


class MediaExecutor:
    """Explicit owner for worker and staging capacity, shared with ReplayStore."""
    def __init__(self, store: ReplayStore):
        if type(store) is not ReplayStore:
            raise TypeError("CPU media executor requires its leader-owned ReplayStore")
        self.store, self.envelope = store, store.envelope
        self._pid, self._lock = os.getpid(), threading.RLock()

    def _check_owner(self):
        self.store._check_owner()
        if self._pid != os.getpid():
            raise RuntimeError("CPU media executor belongs to another process")

    def check_health(self):
        """Refuse foreign ownership or any retained containment outcome.

        A completed operation plus this check permits ordinary error recovery;
        success alone does not retire other independently active operations.
        """
        self._check_owner()

    def set_capacity_wake(self, callback):
        self._check_owner()
        self.store.set_capacity_wake(callback)

    def clear_capacity_wake(self, callback):
        if self._pid != os.getpid():
            raise RuntimeError("CPU media executor belongs to another process")
        self.store.clear_capacity_wake(callback)

    def _metadata_bytes(self):
        return self.envelope.max_manifest_bytes + 2 * self.envelope.max_result_bytes

    def _unit_layout(self, probed):
        if type(probed) is not MediaProbe or probed.envelope_id != self.envelope.identity:
            raise ImageInputError("pixel unit does not match the admitted media probe")
        rows = probed.grid_thw[1] * probed.grid_thw[2]
        size = rows * 1536 * 4
        if size + self._metadata_bytes() > self.envelope.max_staging_bytes:
            raise ImageInputError("media unit pixel and metadata peak exceeds the entire staging byte capacity")
        return rows, size

    def unit_available(self, probed, *, replay_borrows=1):
        """Current opportunity only; the actual operation still owns admission.

        Two borrows cover a new request plus its pixel lease; one covers the
        pixel lease when the coordinator already retains its request borrow.
        """
        self._check_owner()
        if type(replay_borrows) is not int or not 1 <= replay_borrows <= self.envelope.max_replay_borrows:
            raise ImageInputError("media unit replay-borrow requirement exceeds its admitted capacity")
        _rows, size = self._unit_layout(probed)
        with self.store._lock:
            self._check_owner()
            lease = self.store._live.get(probed.ref.path)
            if lease is None or probed.ref not in lease.refs or lease._closed or lease._retiring:
                raise ImageInputError("media unit replay is not owned by this live store")
            return (any(slot is None for slot in self.store._active_workers)
                    and self.store._staging + size + self._metadata_bytes() <= self.envelope.max_staging_bytes
                    and lease._borrows + replay_borrows <= self.envelope.max_replay_borrows)

    def _reserve_staging(self, reservation, size):
        self.store._reserve_staging(reservation, size)

    def _release_staging(self, reservation, *, notify=True):
        return self.store._release_staging(reservation, notify=notify)

    def _retire_worker(self, index, token):
        retired = False
        with self.store._lock:
            if index is not None and self.store._active_workers[index] is token:
                self.store._active_workers[index] = None
                retired = True
        if retired:
            self.store._notify_capacity()

    def _run(self, record, pixels, cancelled, *, deadline=None):
        self._check_owner()
        check_cancelled(cancelled)
        e = self.envelope
        record = {**record, "envelope": asdict(e)}
        encoded = json.dumps(record, separators=(",", ":")).encode()
        if len(encoded) > e.max_manifest_bytes:
            raise ImageInputError("media worker manifest exceeds its explicit bound")
        metadata_size = self._metadata_bytes()
        metadata_reservation, worker_token = object(), object()
        worker_index, primary = None, None
        try:
            with self.store._lock:
                self._check_owner()
                try:
                    worker_index = self.store._active_workers.index(None)
                except ValueError:
                    raise MediaBusy("media CPU worker capacity is busy") from None
                self.store._active_workers[worker_index] = worker_token
                self._reserve_staging(metadata_reservation, metadata_size)
            with tempfile.TemporaryFile(dir=self.store.root) as request, \
                    tempfile.TemporaryFile(dir=self.store.root) as result, \
                    _diagnostic_fd(e.max_result_bytes, self.store, worker_index) as diagnostics:
                # Native stderr/stdout cannot grow beyond their separately
                # charged diagnostic bound, even when pixel outputs are larger.
                request.write(encoded)
                request.seek(0)
                descriptors = (request.fileno(), result.fileno(), pixels.fileno(), diagnostics)
                output_limit = e.max_result_bytes
                if record["operation"] == "stage":
                    output_limit = max(output_limit, record["remaining_bytes"])
                elif record["operation"] == "pixel_unit":
                    grid = record["probe"]["grid_thw"]
                    output_limit = max(output_limit, grid[1] * grid[2] * 1536 * 4)
                command = [sys.executable, "-m", "tensorfold.vision.media_worker",
                           *(str(fd) for fd in descriptors), str(e.max_manifest_bytes),
                           str(e.max_worker_cpu_seconds), str(e.max_worker_address_bytes), str(output_limit)]
                environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": str(e.max_worker_threads),
                               "OPENBLAS_NUM_THREADS": str(e.max_worker_threads), "MKL_NUM_THREADS": str(e.max_worker_threads),
                               "NUMEXPR_NUM_THREADS": str(e.max_worker_threads), "PYTHONDONTWRITEBYTECODE": "1"}
                deadline = min(deadline, time.monotonic() + e.max_worker_wall_seconds) if deadline is not None else \
                    time.monotonic() + e.max_worker_wall_seconds
                cancelled_seen = False

                def stop():
                    nonlocal cancelled_seen
                    cancelled_seen = bool(cancelled())
                    return cancelled_seen or time.monotonic() >= deadline or self.store._failure is not None

                try:
                    status = supervise([command], [environment], grace=e.retirement_grace_seconds,
                                       failure_probe=stop, pass_fds=descriptors)
                except BaseException as error:
                    # A normal return is the supervisor's retirement proof.
                    # Exceptions can own raw failed sessions when diagnostic
                    # construction itself failed; their type is not evidence.
                    self.store._retained_workers[worker_index] = error
                    with self.store._lock:
                        if self.store._failure is None:
                            self.store._failure = error
                    raise
                self._check_owner()
                if cancelled_seen:
                    check_cancelled(lambda: True)
                if time.monotonic() >= deadline:
                    raise ImageInputError("media worker exceeded its wall limit; session retired")
                if status:
                    raise ImageInputError("media CPU worker failed or exceeded a hard resource limit")
                if result.seek(0, 2) > e.max_result_bytes:
                    raise ImageInputError("media CPU worker result exceeded its bound")
                result.seek(0)
                try:
                    output = json.load(result)
                except (ValueError, UnicodeError) as error:
                    raise ImageInputError("media CPU worker returned invalid metadata") from error
                if "error" in output:
                    raise ImageInputError(f"media CPU worker refused input ({output['error']}): {output['message']}")
                return output
        except BaseException as error:
            primary = error
            raise
        finally:
            if self.store._failure is None and (
                    worker_index is None or self.store._retained_workers[worker_index] is None):
                try:
                    finish((lambda: self._release_staging(metadata_reservation),
                            lambda: self._retire_worker(worker_index, worker_token)))
                except BaseException as cleanup:
                    with self.store._lock:
                        held = metadata_reservation in self.store._staging_reservations or (
                            worker_index is not None and self.store._active_workers[worker_index] is worker_token)
                        if held:
                            retained = primary if primary is not None else cleanup
                            if self.store._failure is None:
                                self.store._failure = retained
                            if worker_index is not None and self.store._retained_workers[worker_index] is None:
                                self.store._retained_workers[worker_index] = retained
                    if primary is not None:
                        raise_failures(primary, [cleanup])
                    raise

    def probe_replay(self, ref, *, cancelled):
        with self.store.borrow_ref(ref), tempfile.TemporaryFile(dir=self.store.root) as unused:
            output = self._run({"operation": "probe", "ref": ref.record()}, unused, cancelled)
        probed = MediaProbe.from_record(output["probe"])
        if probed.ref != ref or probed.envelope_id != self.envelope.identity:
            raise ImageInputError("media probe changed replay or admitted envelope identity")
        return probed

    def pixel_unit(self, probed: MediaProbe, index: int, *, cancelled):
        self._check_owner()
        rows, size = self._unit_layout(probed)
        if type(index) is not int or not 0 <= index < probed.unit_count:
            raise ImageInputError("pixel unit does not match the admitted media probe")
        reservation = object()
        replay, handle = None, None
        try:
            self._reserve_staging(reservation, size)
            replay = self.store.borrow_ref(probed.ref)
            handle = tempfile.TemporaryFile(dir=self.store.root)
            output = self._run({"operation": "pixel_unit", "probe": probed.record(), "index": index}, handle, cancelled)
            if (output != {"shape": [rows, 1536], "dtype": "float32", "bytes": size}
                    or os.fstat(handle.fileno()).st_size != size):
                raise ImageInputError("media worker pixel layout differs from its admitted grid")
            return PixelUnitLease(self, handle, size, output["shape"], replay, reservation)
        except BaseException as primary:
            preparation_failure = primary
            def abandon():
                self._check_owner()
                try:
                    if handle is not None:
                        handle.close()
                    if replay is not None:
                        replay.close()
                    self._release_staging(reservation)
                except BaseException:
                    with self.store._lock:
                        if reservation in self.store._staging_reservations and self.store._failure is None:
                            self.store._failure = preparation_failure
                    raise
            rollback((self, handle, replay), primary, abandon)


def probe_replay(ref, envelope: MediaEnvelope, *, executor: MediaExecutor, cancelled):
    if executor.envelope != envelope:
        raise ImageInputError("probe executor does not own the admitted media envelope")
    return executor.probe_replay(ref, cancelled=cancelled)


class PreparedReplayMedia:
    """Canonical token plan plus encoded replay; no normalized history archive."""
    def __init__(self, plan, probes, replay, executor):
        self.plan, self.probes, self.replay, self.executor = plan, tuple(probes), replay, executor

    @property
    def token_ids(self):
        return self.plan.token_ids

    def borrow_replay(self):
        return self.replay.borrow()

    def close(self):
        self.replay.close()


class ReplayMediaFrontend:
    """Explicit capability installed only after the profile owner admits resources."""
    def __init__(self, store: ReplayStore, executor: MediaExecutor, planner, *, cancelled, allow_urls: bool):
        if executor.store is not store or not callable(planner) or not callable(cancelled) or type(allow_urls) is not bool:
            raise TypeError("media frontend requires one admitted store/executor, planner and cancellation owner")
        self.store, self.executor, self.planner = store, executor, planner
        self.media_envelope, self.cancelled, self.allow_urls = store.envelope, cancelled, allow_urls
        self.videos = bool(store.envelope.max_videos)

    def check_media_health(self):
        self.executor.check_health()

    def prepare_sources(self, rendered, sources, *, allow_urls, max_prompt_tokens, max_visual_tokens):
        if allow_urls != self.allow_urls:
            raise ImageInputError("media URL admission changed during preparation")
        lease = self.store.stage(sources, allow_urls=allow_urls, cancelled=self.cancelled)
        try:
            probes = tuple(self.executor.probe_replay(ref, cancelled=self.cancelled) for ref in lease.refs)
            e = self.media_envelope
            if (sum(p.visual_rows for p in probes) > e.max_expanded_rows
                    or sum(p.decoded_pixels for p in probes) > e.max_total_decoded_pixels
                    or sum(p.decoded_bytes for p in probes) > e.max_decoded_bytes):
                raise ImageInputError("combined media rows/decoded work exceed the admitted envelope")
            plan = self.planner(rendered, probes, max_prompt_tokens=max_prompt_tokens, max_visual_tokens=max_visual_tokens)
            check_cancelled(self.cancelled)
            return PreparedReplayMedia(plan, probes, lease, self.executor)
        except BaseException as primary:
            rollback(lease, primary, lease.close)
