"""Checkpoint-scoped CPU history for expert placement recommendations."""
import contextlib
import fcntl
import json
from math import fsum, isfinite, ulp
import os
from pathlib import Path
import stat
import tempfile
from threading import Lock

from .expert_balance import plan_expert_placement, rolling_expert_load
from .expert_placement import encode_placement


_PHASES = 2  # Prefill and decode, in the device snapshot's order.
_MAX_EXPOSURE = (1 << 64) - 1


def _positive(value):
    return (type(value) in (int, float) and value > 0
            and (type(value) is int or isfinite(value)))


def _table(values, lengths, valid, description):
    """Validate the expected bounded rectangular shape before taking ownership."""
    if not isinstance(values, (tuple, list)) or len(values) != lengths[0]:
        raise ValueError(f"invalid {description} shape")
    if len(lengths) == 1:
        if any(not valid(value) for value in values):
            raise ValueError(f"invalid {description} values")
        return tuple(values)
    return tuple(_table(row, lengths[1:], valid, description) for row in values)


def _atomic_replace(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


def _claim_history(path):
    """Keep the same companion inode across writers without editing it."""
    marker = path.with_name(path.name + ".lock")
    marker.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(marker, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise ValueError(f"invalid expert balance history lock: {marker}")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"expert balance history already has a writer: {path}") from exc
    except BaseException:
        os.close(fd)
        raise
    return fd


class ExpertBalanceState:
    """Own observed load history and publish restart-only placement proposals.

    The recorder accepts a detached CPU snapshot of nonnegative route counts.
    It returns whether a flush is due, based on observed routes since the last
    save. The caller may retry a failed flush without recording the batch again.
    Recommendations require a caller-selected minimum of fresh observations in
    both phases. The active placement is fixed for the running engine epoch.
    A persisted recommendation is the last emitted proposal, so it can remain
    after later observations yield no proposal. The stable companion lock is
    held until close, after the recorder and any final flush have finished.
    """

    def __init__(self, path, *, checkpoint, etp, layers, experts, half_life_routes, current):
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError("expert balance history requires a checkpoint identity")
        if type(etp) is not int or etp not in (1, 2, 4):
            raise ValueError("expert balance history requires TP8 ETP1, ETP2 or ETP4")
        if (type(layers) is not int or layers <= 0 or type(experts) is not int or experts <= 0
                or experts % (8 // etp)):
            raise ValueError("expert balance history requires balanced layer geometry")
        if not _positive(half_life_routes):
            raise ValueError("half_life_routes must be finite and positive")
        self.path = Path(path).expanduser().absolute()
        self.recommendation_path = self.path.with_name(self.path.name + ".placement.json")
        self._metadata = dict(version=1, checkpoint=checkpoint, etp=etp, layers=layers, experts=experts,
                              half_life_routes=half_life_routes)
        self._shape = (_PHASES, layers, experts)
        identity = tuple(range(experts))
        self._initial_placement = current is None
        if current is None:
            current = (identity,) * layers
        self._current = _table(current, (layers, experts), lambda value: type(value) is int, "expert placement")
        if any(set(row) != set(identity) for row in self._current):
            raise ValueError("each layer must contain every expert exactly once")
        self._empty_exposures = ((0,) * layers,) * _PHASES
        self._pending_recommendation = None
        self._lock = Lock()
        self._history_fd = _claim_history(self.path)
        try:
            saved = self._read()
            if saved is None:
                self._loads = (((0.0,) * experts,) * layers,) * _PHASES
                self._exposures = self._empty_exposures
                self._saved_exposures = None
            else:
                self._loads, self._exposures = saved
                self._saved_exposures = self._exposures
            self._initial_exposures = self._exposures
        except BaseException as error:
            try:
                self.close()
            except OSError as cleanup:
                raise error from cleanup
            raise

    @property
    def current(self):
        return self._current

    def _require_open(self):
        if self._history_fd is None:
            raise RuntimeError("expert balance history is closed")

    def close(self):
        """Release the writer after the caller stops recording and flushes."""
        with self._lock:
            if self._history_fd is not None:
                fd, self._history_fd = self._history_fd, None
                self._pending_recommendation = None
                os.close(fd)

    def _encode(self, loads, exposures):
        return json.dumps({**self._metadata, "loads": loads, "exposures": exposures},
                          separators=(",", ":"), allow_nan=False).encode()

    def _read(self):
        phases, layers, experts = self._shape
        # Finite binary64 proportions need at most 24 characters in JSON;
        # each saved exposure needs at most 20. Also budget the array delimiters.
        limit = len(self._encode((), ())) + phases * (4 + layers * (26 * experts + 24))

        def open_file(name, flags):
            return os.open(name, flags | os.O_NONBLOCK | os.O_NOFOLLOW)

        try:
            stream = open(self.path, "rb", opener=open_file)
        except FileNotFoundError:
            return None
        with stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
                raise ValueError(f"invalid or oversized expert balance history: {self.path}")
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise ValueError(f"oversized expert balance history: {self.path}")
        try:
            payload = json.loads(data)
        except (ValueError, RecursionError) as exc:
            raise ValueError(f"invalid expert balance history JSON: {self.path}") from exc
        if (type(payload) is not dict or payload.keys() != self._metadata.keys() | {"loads", "exposures"}
                or any(type(payload[key]) is not type(expected) or payload[key] != expected
                       for key, expected in self._metadata.items())):
            raise ValueError(f"expert balance history belongs to another checkpoint, geometry or decay policy: {self.path}")
        loads = _table(payload["loads"], self._shape, lambda v: type(v) in (int, float) and 0 <= v <= 1,
                       "expert load history")
        exposures = _table(payload["exposures"], (phases, layers),
                           lambda v: type(v) is int and 0 <= v <= _MAX_EXPOSURE, "expert route exposures")
        for phase, seen in zip(loads, exposures, strict=True):
            for row, total in zip(phase, seen, strict=True):
                mass = fsum(row)
                if (not total and mass) or (total and abs(mass - 1) > experts * ulp(1.0)):
                    raise ValueError("expert load history and route exposure disagree")
        return loads, exposures

    def snapshot(self):
        """Return immutable load and cumulative exposure tables in phase order."""
        with self._lock:
            return self._loads, self._exposures

    def record(self, raw_counts):
        counts = _table(raw_counts, self._shape, lambda v: type(v) is int and 0 <= v <= _MAX_EXPOSURE,
                        "expert route counts")
        increment = tuple(tuple(sum(row) for row in phase) for phase in counts)
        with self._lock:
            self._require_open()
            if any(total for phase in increment for total in phase):
                totals = tuple(tuple(value + added for value, added in zip(phase, more, strict=True))
                               for phase, more in zip(self._exposures, increment, strict=True))
                if any(total > _MAX_EXPOSURE for phase in totals for total in phase):
                    raise OverflowError("expert route exposure exceeded the history encoding")
                half_life = self._metadata["half_life_routes"]
                self._loads = tuple(rolling_expert_load(previous, phase, half_life_routes=half_life)
                                    for previous, phase in zip(self._loads, counts, strict=True))
                self._exposures = totals
                self._pending_recommendation = None
            saved = self._saved_exposures if self._saved_exposures is not None else self._empty_exposures
            return any(new - old >= self._metadata["half_life_routes"]
                       for phase, before in zip(self._exposures, saved, strict=True)
                       for new, old in zip(phase, before, strict=True))

    def flush(self):
        """Atomically preserve the latest observations without modifying placement."""
        with self._lock:
            self._require_open()
            if self._saved_exposures != self._exposures:
                _atomic_replace(self.path, self._encode(self._loads, self._exposures))
                self._saved_exposures = self._exposures

    def recommendation(self, *, minimum_routes=None):
        """Encode an improving proposal after sufficient fresh routes in both phases."""
        if minimum_routes is not None and not _positive(minimum_routes):
            raise ValueError("minimum_routes must be finite and positive")
        with self._lock:
            self._require_open()
            self._pending_recommendation = None
            if minimum_routes is None or any(
                total - first < minimum_routes
                for phase, start in zip(self._exposures, self._initial_exposures, strict=True)
                for total, first in zip(phase, start, strict=True)
            ):
                return None
            loads = self._loads
        current = None if self._initial_placement else self.current
        plan = plan_expert_placement(loads, etp=self._metadata["etp"], current=current)
        if plan == self.current:
            return None
        data = encode_placement(plan, checkpoint=self._metadata["checkpoint"], etp=self._metadata["etp"])
        with self._lock:
            self._require_open()
            if loads is not self._loads:
                return None
            self._pending_recommendation = data
            return data

    def publish_recommendation(self, data):
        """Publish only a proposal issued for the current observation snapshot."""
        with self._lock:
            self._require_open()
            if type(data) is not bytes or data != self._pending_recommendation:
                raise ValueError("expert placement recommendation is stale or was not issued by this state")
            _atomic_replace(self.recommendation_path, data)
            self._pending_recommendation = None
