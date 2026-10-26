"""Explicitly admitted encoded media and request/State/rank replay ownership.

The leader owns files on the shared filesystem. A serialized reference supplies
identity, never lifetime: every consumer must retain a borrow from its leader's
lease until its last asynchronous use retires.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from contextlib import contextmanager
import hashlib
import math
import os
from pathlib import Path
import shutil
import stat
import tempfile
import threading
import time
from typing import Callable

from tensorfold.cleanup import raise_failures, rollback
from .images import ImageSource, _check_source
from .images_http import ImageInputError


class MediaCancelled(ImageInputError):
    """Cancellation observed before publication; owned workers are retired."""


class MediaBusy(ImageInputError):
    """A legal work unit can fit after another bounded media owner retires."""


def check_cancelled(cancelled: Callable[[], bool]) -> None:
    if cancelled():
        raise MediaCancelled("media preparation cancelled")


@dataclass(frozen=True, slots=True)
class MediaEnvelope:
    """Administrator admission limits. Every value is mandatory; no service defaults.

    Bytes bound distinct encoded storage, transient staging and process virtual
    memory. Decoded pixel/byte bounds cover the complete scan, including dropped
    frames; staging covers only concurrent live output units. CPU seconds are a
    hard per fresh worker limit; wall time includes imports and decoding.
    """
    max_images: int
    max_videos: int
    max_upload_bytes: int
    max_total_upload_bytes: int
    max_dimension: int
    max_pixels_per_frame: int
    max_total_decoded_pixels: int
    max_decoded_bytes: int
    max_selected_frames: int
    max_video_seconds: float
    max_expanded_rows: int
    max_workers: int
    max_worker_threads: int
    max_worker_address_bytes: int
    max_worker_cpu_seconds: int
    max_worker_wall_seconds: float
    max_staging_wall_seconds: float
    max_single_allocation_bytes: int
    max_probe_bytes: int
    max_probe_seconds: float
    max_scan_frames: int
    max_scan_packets: int
    max_staging_bytes: int
    max_replay_bytes: int
    max_replay_items: int
    max_replay_leases: int
    max_replay_borrows: int
    max_manifest_bytes: int
    max_result_bytes: int
    max_redirects: int
    max_url_chars: int
    retirement_grace_seconds: float
    assumed_fps: tuple[int, int] | None

    def __post_init__(self):
        durations = {"max_video_seconds", "max_worker_wall_seconds", "max_staging_wall_seconds",
                     "max_probe_seconds", "retirement_grace_seconds"}
        zeros = {"max_images", "max_videos", "max_redirects", "retirement_grace_seconds"}
        for item in fields(self):
            name, value = item.name, getattr(self, item.name)
            if name == "assumed_fps":
                if value is not None and (type(value) is not tuple or len(value) != 2
                                         or any(type(v) is not int or v <= 0 for v in value)):
                    raise ValueError("assumed_fps must be an explicit positive rational or None")
                continue
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                    or value < 0 or (value == 0 and name not in zeros)
                    or (name not in durations and type(value) is not int)):
                raise ValueError(name + " must be a finite positive bound")
        if self.max_images + self.max_videos == 0:
            raise ValueError("media envelope must admit at least one media kind")
        if self.max_pixels_per_frame > 2**31 - 1:
            raise ValueError("FFmpeg max_pixels must fit its supported INT_MAX bound")
        if self.max_probe_bytes < 32:
            raise ValueError("FFmpeg probesize requires at least 32 bytes")
        import ctypes
        if self.max_single_allocation_bytes > (1 << (8 * ctypes.sizeof(ctypes.c_size_t))) - 1:
            raise ValueError("FFmpeg av_max_alloc exceeds the active size_t range")
        if self.max_probe_bytes > 2**63 - 1 or self.probe_microseconds > 2**63 - 1:
            raise ValueError("FFmpeg probe options exceed the signed int64 range")

    @property
    def probe_microseconds(self):
        from fractions import Fraction
        return math.ceil(Fraction(self.max_probe_seconds) * 1_000_000)

    @property
    def identity(self) -> str:
        import json
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @classmethod
    def from_record(cls, record):
        record = dict(record)
        if record.get("assumed_fps") is not None:
            record["assumed_fps"] = tuple(record["assumed_fps"])
        return cls(**record)


@dataclass(frozen=True, slots=True)
class ReplayRef:
    path: str
    sha256: str
    byte_length: int
    ordinal: int
    kind: str
    detail: str
    version: int = 1

    def __post_init__(self):
        if (self.version != 1 or type(self.version) is not int or self.kind not in {"image", "video"}
                or type(self.byte_length) is not int or self.byte_length <= 0
                or type(self.ordinal) is not int or self.ordinal < 0
                or self.detail not in {"auto", "low", "high"}
                or not isinstance(self.path, str) or not Path(self.path).is_absolute()
                or not isinstance(self.sha256, str) or len(self.sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.sha256)):
            raise ImageInputError("invalid immutable media replay reference")

    def record(self):
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MediaProbe:
    ref: ReplayRef
    envelope_id: str
    processor_id: str
    width: int
    height: int
    source_frames: int
    selected_indices: tuple[int, ...]
    selected_pts: tuple[tuple[int, int], ...]
    fps: tuple[int, int] | None
    timestamp_policy: str
    resized_hw: tuple[int, int]
    grid_thw: tuple[int, int, int]
    decoded_pixels: int
    decoded_bytes: int

    def __post_init__(self):
        if (type(self.ref) is not ReplayRef or not isinstance(self.envelope_id, str) or not self.envelope_id
                or not isinstance(self.processor_id, str) or not self.processor_id
                or any(type(v) is not int or v <= 0 for v in
                       (self.width, self.height, self.source_frames, self.decoded_pixels, self.decoded_bytes))
                or type(self.grid_thw) is not tuple or len(self.grid_thw) != 3
                or any(type(v) is not int or v <= 0 for v in self.grid_thw)
                or self.grid_thw[1] % 2 or self.grid_thw[2] % 2
                or self.resized_hw != (self.grid_thw[1] * 16, self.grid_thw[2] * 16)
                or type(self.selected_indices) is not tuple or type(self.selected_pts) is not tuple
                or len(self.selected_indices) != len(self.selected_pts)
                or any(type(v) is not int or not 0 <= v < self.source_frames for v in self.selected_indices)
                or any(a >= b for a, b in zip(self.selected_indices, self.selected_indices[1:]))
                or any(type(p) is not tuple or len(p) != 2 or any(type(v) is not int for v in p) or p[1] <= 0
                       for p in self.selected_pts)):
            raise ImageInputError("invalid media probe dimensions, grid or selected timestamps")
        if self.ref.kind == "image":
            if (self.source_frames != 1 or self.selected_indices or self.fps is not None or self.grid_thw[0] != 1
                    or self.timestamp_policy != "encoded-orientation-pil-rgb-v1"):
                raise ImageInputError("invalid still-image probe metadata")
        elif (len(self.selected_indices) < 2 or self.grid_thw[0] != (len(self.selected_indices) + 1) // 2
              or type(self.fps) is not tuple or len(self.fps) != 2
              or any(type(v) is not int or v <= 0 for v in self.fps)
              or self.timestamp_policy not in {"actual-pts-v1", "explicit-assumed-fps-v1"}):
            raise ImageInputError("invalid video probe temporal metadata")

    @property
    def identity(self):
        """Content/processor/selection identity, independent of replay location."""
        import json
        record = self.record()
        record["ref"] = {name: value for name, value in record["ref"].items() if name not in {"path", "ordinal"}}
        return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @property
    def visual_rows(self):
        t, h, w = self.grid_thw
        return t * h * w // 4

    @property
    def unit_count(self):
        return self.grid_thw[0]

    def temporal_timestamps(self):
        """Actual-PTS adapter: exact temporal means; pinned Qwen display format."""
        from fractions import Fraction
        times = [Fraction(*pts) for pts in self.selected_pts]
        if not times:
            return ()
        if len(times) % 2:
            times.append(times[-1])
        return tuple((times[i] + times[i + 1]) / 2 for i in range(0, len(times), 2))

    def timestamp_markers(self):
        return tuple(f"<{float(pts):.1f} seconds>" for pts in self.temporal_timestamps())

    def record(self):
        return asdict(self)

    @classmethod
    def from_record(cls, record):
        record = dict(record)
        record["ref"] = ReplayRef(**record["ref"])
        for name in ("selected_indices", "resized_hw", "grid_thw"):
            record[name] = tuple(record[name])
        record["selected_pts"] = tuple(tuple(pts) for pts in record["selected_pts"])
        if record["fps"] is not None:
            record["fps"] = tuple(record["fps"])
        return cls(**record)


class ReplayBorrow:
    """One named consumer's lifetime; close only after all of its readers retire."""
    def __init__(self, lease):
        self._lease = lease

    @property
    def refs(self):
        lease = self._lease
        if lease is None:
            raise RuntimeError("replay borrow is retired")
        with lease._store._lock:
            if self not in lease._borrowers:
                raise RuntimeError("replay borrow is retired")
            return lease.refs

    def close(self):
        lease = self._lease
        if lease is not None:
            with lease._store._lock:
                if self._lease is None:
                    return
                lease._store._check_owner()
                lease._borrowers.discard(self)
                self._lease = None

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        if error is not None:
            rollback(self, error, self.close)
        self.close()


class ReplayLease:
    def __init__(self, store, directory, refs, reservation):
        self._store, self._directory, self._refs = store, directory, tuple(refs)
        self._borrowers, self._closed, self._retiring = set(), False, False
        self._reservation = reservation
        self._directory_attempted = self._directory_removed = False
        self._directory_failure = None

    @property
    def _borrows(self):
        return len(self._borrowers)

    @property
    def refs(self):
        return self._refs

    def borrow(self):
        with self._store._lock:
            self._store._check_owner()
            if self._closed or self._retiring:
                raise RuntimeError("replay lease is retired")
            if self._borrows >= self._store.envelope.max_replay_borrows:
                raise MediaBusy("media replay borrower capacity is busy")
            borrowed = ReplayBorrow(self)
            self._borrowers.add(borrowed)
            return borrowed

    def close(self):
        with self._store._lock:
            self._store._check_owner()
            if self._closed:
                return
            if self._borrows:
                raise RuntimeError("replay lease still has outstanding borrowers")
            if self._directory_attempted and not self._directory_removed:
                raise RuntimeError("replay directory retirement uncertain; retain lease until process containment") from self._directory_failure
            # Uncertain removal retains capacity and owner for containment.
            self._retiring = True
            if not self._directory_removed:
                try:
                    # Spend pathname authority before the operation: an error
                    # may occur after removal and a later lease may reuse it.
                    self._directory_attempted = True
                    shutil.rmtree(self._directory)
                    self._directory_removed = True
                except BaseException as error:
                    self._directory_failure = error
                    if self._store._failure is None:
                        self._store._failure = error
                    raise
            for ref in self.refs:
                if self._store._live.get(ref.path) is self:
                    self._store._live.pop(ref.path, None)
            self._store._replay_reservations.pop(self._reservation, None)
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        if error is not None:
            rollback(self, error, self.close)
        self.close()


class ReplayStore:
    """Leader-owned bounded shared-FS storage; fail-fast admission, no hidden queue."""
    def __init__(self, root: str | Path, envelope: MediaEnvelope):
        if type(envelope) is not MediaEnvelope:
            raise TypeError("an admitted explicit MediaEnvelope is required")
        self.root = Path(root).absolute()
        if self.root.resolve() != self.root or not self.root.is_dir():
            raise ValueError("replay root must be an existing canonical shared directory")
        metadata = self.root.stat()
        if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o022:
            raise ValueError("replay root must be owned by the service and not writable by other users")
        self.envelope = envelope
        self._pid, self._lock = os.getpid(), threading.RLock()
        self._replay_reservations, self._staging_reservations = {}, {}
        self._live = {}
        self._active_workers = [None] * envelope.max_workers
        self._retained_workers = [None] * envelope.max_workers
        self._failure = None

    @property
    def _bytes(self):
        return sum(value[0] for value in self._replay_reservations.values())

    @property
    def _items(self):
        return sum(value[1] for value in self._replay_reservations.values())

    @property
    def _leases(self):
        return len(self._replay_reservations)

    @property
    def _staging(self):
        return sum(self._staging_reservations.values())

    def _reserve_staging(self, reservation, size):
        with self._lock:
            self._check_owner()
            if size > self.envelope.max_staging_bytes:
                raise ImageInputError("media work unit exceeds the entire staging byte capacity")
            if self._staging + size > self.envelope.max_staging_bytes:
                raise MediaBusy("media staging byte capacity is busy")
            self._staging_reservations[reservation] = size

    def _release_staging(self, reservation):
        with self._lock:
            self._staging_reservations.pop(reservation, None)

    def _check_owner(self):
        if self._pid != os.getpid():
            raise RuntimeError("replay store belongs to another process")
        with self._lock:
            failure = self._failure
            if failure is None:
                # Direct operation-owned fatal outcomes survive even a failed
                # fatal-latch operation; no exception graph is inspected.
                for retained in self._retained_workers:
                    if retained is not None:
                        failure = retained
                        break
        if failure is not None:
            raise RuntimeError("media worker retirement unresolved; retain store until process containment") from failure

    def stage(self, sources, *, allow_urls: bool, cancelled: Callable[[], bool]) -> ReplayLease:
        from .videos import VideoSource, video_source
        self._check_owner()
        e = self.envelope
        if (not isinstance(sources, (tuple, list)) or not sources
                or len(sources) > e.max_images + e.max_videos):
            raise ImageInputError("ordered media sources must be a nonempty bounded sequence")
        sources = tuple(sources)
        if (sum(type(s) is ImageSource for s in sources) > e.max_images
                or sum(type(s) is VideoSource for s in sources) > e.max_videos
                or any(type(s) not in (ImageSource, VideoSource) for s in sources)):
            raise ImageInputError("media source count or kind exceeds the envelope")
        check_cancelled(cancelled)
        directory, refs, reserved = None, [], 0
        reservation = object()
        try:
            with self._lock:
                if self._leases >= e.max_replay_leases or self._items + len(sources) > e.max_replay_items:
                    raise ImageInputError("media replay lease/item capacity is busy")
                self._replay_reservations[reservation] = (0, len(sources))
            directory = tempfile.mkdtemp(prefix="media-", dir=self.root)
            directory = Path(directory)
            deadline = time.monotonic() + e.max_staging_wall_seconds
            for ordinal, source in enumerate(sources):
                check_cancelled(cancelled)
                if time.monotonic() >= deadline:
                    raise ImageInputError("encoded media staging exceeded its wall limit")
                image = type(source) is ImageSource
                from types import SimpleNamespace
                source_limits = SimpleNamespace(max_encoded_bytes=e.max_upload_bytes, max_url_chars=e.max_url_chars)
                if image:
                    _check_source(source, source_limits, allow_urls)
                else:
                    video_source({"url": source.url}, source_limits, allow_urls)
                remaining = min(e.max_upload_bytes, e.max_total_upload_bytes - reserved)
                # Existing bounded HTTPS downloader holds bytearray + returned
                # bytes concurrently; reserve both, independent of replay quota.
                staging = 2 * remaining
                staged_reservation = object()
                item_primary = None
                try:
                    with self._lock:
                        if remaining <= 0 or self._bytes + remaining > e.max_replay_bytes:
                            raise ImageInputError("encoded media staging/replay byte capacity is busy")
                        prior = self._replay_reservations[reservation][0]
                        self._replay_reservations[reservation] = (prior + remaining, len(sources))
                        self._reserve_staging(staged_reservation, staging)
                    path = directory / str(ordinal)
                    with path.open("xb") as output:
                        # URL retrieval/base64 parsing has the same hard
                        # process CPU/address-space/wall containment as decode.
                        from .media_frontend import MediaExecutor
                        staged = MediaExecutor(self)._run({"operation": "stage", "url": source.url,
                                                          "kind": "image" if image else "video",
                                                          "remaining_bytes": remaining}, output, cancelled, deadline=deadline)
                        size = os.fstat(output.fileno()).st_size
                    if (set(staged) != {"sha256", "byte_length"} or type(staged["byte_length"]) is not int
                            or not 0 < size == staged["byte_length"] <= remaining):
                        raise ImageInputError("media staging worker returned invalid encoded metadata")
                    check_cancelled(cancelled)
                    path.chmod(0o400)
                    ref = ReplayRef(str(path), staged["sha256"], size, ordinal,
                                    "image" if image else "video", source.detail if image else "auto")
                    refs.append(ref)
                    reserved += size
                    with self._lock:
                        self._replay_reservations[reservation] = (reserved, len(sources))
                except BaseException as error:
                    item_primary = error
                    raise
                finally:
                    if self._failure is None:
                        try:
                            self._check_owner()
                            self._release_staging(staged_reservation)
                        except BaseException as cleanup:
                            with self._lock:
                                if staged_reservation in self._staging_reservations and self._failure is None:
                                    self._failure = item_primary if item_primary is not None else cleanup
                            if item_primary is not None:
                                raise_failures(item_primary, [cleanup])
                            raise
            check_cancelled(cancelled)
            if time.monotonic() >= deadline:
                raise ImageInputError("encoded media staging exceeded its wall limit")
            lease = ReplayLease(self, directory, refs, reservation)
            with self._lock:
                for ref in refs:
                    self._live[ref.path] = lease
            return lease
        except BaseException as primary:
            # Keep cleanup failures alongside the primary; do not release
            # capacity for files that could not be removed.
            def abandon():
                self._check_owner()
                if directory is not None:
                    shutil.rmtree(directory)
                with self._lock:
                    for ref in refs:
                        self._live.pop(ref.path, None)
                    self._replay_reservations.pop(reservation, None)
            rollback((self, directory, refs), primary, abandon)

    def borrow_ref(self, ref):
        with self._lock:
            self._check_owner()
            lease = self._live.get(ref.path)
            if lease is None or ref not in lease.refs:
                raise ImageInputError("replay reference is not owned by this live store")
            return lease.borrow()


@contextmanager
def open_replay(ref: ReplayRef, envelope: MediaEnvelope):
    """Validate a bounded extent through the existing exact native FD owner."""
    if ref.byte_length > envelope.max_upload_bytes:
        raise ImageInputError("replay exceeds the encoded media bound")
    from tensorfold.cuda.tensor_file import _regular_stream
    with _regular_stream(ref.path, no_follow=True) as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != ref.byte_length or before.st_mode & 0o222:
            raise ImageInputError("replay file is mutable or its length changed")
        digest = hashlib.sha256()
        remaining = ref.byte_length
        while remaining:
            block = handle.read(min(remaining, envelope.max_single_allocation_bytes, 1024 * 1024))
            if not block:
                raise ImageInputError("immutable encoded replay was truncated")
            digest.update(block)
            remaining -= len(block)
        if handle.read(1):
            raise ImageInputError("immutable encoded replay grew beyond its admitted extent")
        after = os.fstat(handle.fileno())
        def identity(value):
            return (value.st_dev, value.st_ino, value.st_mode, value.st_uid, value.st_gid,
                    value.st_nlink, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        if identity(before) != identity(after) or digest.hexdigest() != ref.sha256:
            raise ImageInputError("immutable encoded replay identity changed")
        handle.seek(0)
        yield handle
