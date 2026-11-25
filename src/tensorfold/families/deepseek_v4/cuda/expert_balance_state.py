"""Checkpoint-scoped route history and committed live expert placements."""
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
from .expert_transfer import ExpertTransferPlan, plan_expert_transfers


_PHASES = 2  # Prefill and decode, in the device snapshot's order.
_MAX_EXPOSURE = (1 << 64) - 1
_MAX_EPOCH = (1 << 64) - 1


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


def _placement(values, layers, experts):
    rows = _table(values, (layers, experts), lambda value: type(value) is int, "expert placement")
    expected = set(range(experts))
    if any(set(row) != expected for row in rows):
        raise ValueError("each layer must contain every expert exactly once")
    return rows


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
    """Own observed loads and, in live mode, the durable placement epoch.

    The recorder accepts a detached CPU snapshot of nonnegative route counts.
    It returns whether a flush is due, based on observed routes since the last
    save. The caller may retry a failed flush without recording the batch again.
    Recommendations require a caller-selected minimum of fresh observations in
    both phases. The scheduler commits a new live epoch only after all ranks
    have finished changing their expert banks and route tables.
    A persisted recommendation is the last emitted proposal, so it can remain
    after later observations yield no proposal. The stable companion lock is
    held until close, after the recorder and any final flush have finished.
    """

    def __init__(self, path, *, checkpoint, etp, layers, experts, half_life_routes, current=None, live=False):
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError("expert balance history requires a checkpoint identity")
        if type(etp) is not int or etp not in (1, 2, 4):
            raise ValueError("expert balance history requires TP8 ETP1, ETP2 or ETP4")
        if (type(layers) is not int or layers <= 0 or type(experts) is not int or experts <= 0
                or experts % (8 // etp)):
            raise ValueError("expert balance history requires balanced layer geometry")
        if not _positive(half_life_routes):
            raise ValueError("half_life_routes must be finite and positive")
        if type(live) is not bool:
            raise ValueError("live expert balance mode must be boolean")
        self.path = Path(path).expanduser().absolute()
        self.recommendation_path = self.path.with_name(self.path.name + ".placement.json")
        self._metadata = dict(checkpoint=checkpoint, etp=etp, layers=layers, experts=experts,
                              half_life_routes=half_life_routes)
        self._shape = (_PHASES, layers, experts)
        identity = (tuple(range(experts)),) * layers
        explicit = current is not None
        self._current = _placement(current, layers, experts) if explicit else identity
        self._initial_placement = not explicit
        self._epoch = 0 if live else None
        self._placement_poisoned = False
        self._empty_exposures = ((0,) * layers,) * _PHASES
        self._pending_recommendation = None
        self._lock = Lock()
        self._history_fd = _claim_history(self.path)
        try:
            saved = self._read()
            saved_epoch = None
            if saved is None:
                self._loads = (((0.0,) * experts,) * layers,) * _PHASES
                self._exposures = self._empty_exposures
                self._saved_exposures = None
            else:
                self._loads, self._exposures, saved_epoch, saved_placement = saved
                self._saved_exposures = self._exposures
                if saved_epoch is not None:
                    if explicit and self._current != saved_placement:
                        raise ValueError(f"startup expert placement conflicts with the committed epoch: {self.path}")
                    self._current = saved_placement
                    self._epoch = saved_epoch
                    self._initial_placement = not explicit and saved_epoch == 0 and saved_placement == identity
            self._initial_exposures = self._exposures
            if live and saved_epoch is None:
                self._save(epoch=self._epoch, placement=self._current)
        except BaseException as error:
            try:
                self.close()
            except OSError as cleanup:
                raise error from cleanup
            raise

    @property
    def current(self):
        with self._lock:
            return self._current

    def placement_snapshot(self):
        """Return the immutable active epoch and map together."""
        with self._lock:
            self._require_open()
            return self._epoch, self._current

    def _require_open(self):
        if self._history_fd is None:
            raise RuntimeError("expert balance history is closed")
        if self._placement_poisoned:
            raise RuntimeError("live expert placement publication failed; reopen history before reuse")

    def close(self):
        """Release the writer after the caller stops recording and flushes."""
        with self._lock:
            if self._history_fd is not None:
                fd, self._history_fd = self._history_fd, None
                self._pending_recommendation = None
                os.close(fd)

    def _encode(self, loads, exposures, *, epoch, placement):
        data = {"version": 1 if epoch is None else 2, **self._metadata, "loads": loads, "exposures": exposures}
        if epoch is not None:
            data.update(epoch=epoch, placement=placement)
        return json.dumps(data, separators=(",", ":"), allow_nan=False).encode()

    def _save(self, *, epoch, placement):
        _atomic_replace(self.path, self._encode(self._loads, self._exposures, epoch=epoch, placement=placement))
        self._saved_exposures = self._exposures

    def _read(self):
        phases, layers, experts = self._shape
        # Finite binary64 proportions need at most 24 characters in JSON;
        # each saved exposure needs at most 20. Also budget the array delimiters.
        limit = len(self._encode((), (), epoch=_MAX_EPOCH if self._epoch is not None else None, placement=()))
        limit += phases * (4 + layers * (26 * experts + 24))
        if self._epoch is not None:
            limit += layers * (3 + experts * (len(str(experts - 1)) + 1))

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
        version = payload.get("version") if type(payload) is dict else None
        if type(version) is not int or version not in (1, 2):
            raise ValueError(f"unsupported expert balance history version: {self.path}")
        fields = self._metadata.keys() | {"version", "loads", "exposures"}
        if version == 2:
            fields |= {"epoch", "placement"}
            if self._epoch is None:
                raise ValueError(f"committed expert placement requires live expert balance: {self.path}")
        if (payload.keys() != fields
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
        if version == 1:
            return loads, exposures, None, None
        epoch = payload["epoch"]
        if type(epoch) is not int or not 0 <= epoch <= _MAX_EPOCH:
            raise ValueError(f"invalid committed expert placement epoch: {self.path}")
        return loads, exposures, epoch, _placement(payload["placement"], layers, experts)

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
                self._save(epoch=self._epoch, placement=self._current)

    def recommendation(self, *, minimum_routes=None):
        """Encode an improving proposal after sufficient fresh routes in both phases."""
        return self._recommend(minimum_routes, live=False)

    def live_recommendation(self, *, minimum_routes):
        """Return the current epoch and its proposed in-place transfer plan."""
        return self._recommend(minimum_routes, live=True)

    def _recommend(self, minimum_routes, *, live):
        if minimum_routes is not None and not _positive(minimum_routes):
            raise ValueError("minimum_routes must be finite and positive")
        with self._lock:
            self._require_open()
            if live:
                self._require_live()
            else:
                self._pending_recommendation = None
            if minimum_routes is None or any(
                total - first < minimum_routes
                for phase, start in zip(self._exposures, self._initial_exposures, strict=True)
                for total, first in zip(phase, start, strict=True)
            ):
                return None
            loads, active, epoch = self._loads, self._current, self._epoch
            current = None if self._initial_placement else active
        etp = self._metadata["etp"]
        proposed = plan_expert_placement(loads, etp=etp, current=current)
        if proposed == active:
            return None
        if live:
            result = (epoch, plan_expert_transfers(active, proposed, etp=etp))
        else:
            result = encode_placement(proposed, checkpoint=self._metadata["checkpoint"], etp=etp)
        with self._lock:
            self._require_open()
            if loads is not self._loads or epoch != self._epoch:
                return None
            if not live:
                self._pending_recommendation = result
            return result

    def _require_live(self):
        if self._epoch is None:
            raise RuntimeError("expert balance history was opened without live placement")

    def _validate_live_placement(self, plan, expected_epoch):
        self._require_open()
        self._require_live()
        _, layers, experts = self._shape
        etp = self._metadata["etp"]
        if type(plan) is not ExpertTransferPlan or plan.etp != etp or plan.group_slots != experts // (8 // etp):
            raise ValueError("expert transfer plan belongs to another geometry")
        if type(expected_epoch) is not int or expected_epoch != self._epoch or plan.current != self._current:
            raise ValueError("expert transfer plan does not match the active placement epoch")
        if type(plan.proposed) is not tuple or any(type(row) is not tuple for row in plan.proposed):
            raise ValueError("proposed expert placement must be immutable")
        proposed = _placement(plan.proposed, layers, experts)
        changed = proposed != self._current
        if changed != bool(plan.rounds):
            raise ValueError("expert transfer plan does not match the proposed placement")
        if changed and self._epoch == _MAX_EPOCH:
            raise OverflowError("expert placement epoch exceeded the history encoding")
        return changed

    def validate_live_placement(self, plan, *, expected_epoch):
        """Check a proposal before the scheduler changes any device state."""
        with self._lock:
            return self._validate_live_placement(plan, expected_epoch)

    def commit_live_placement(self, plan, *, expected_epoch):
        """Persist only after the scheduler confirms all-rank device completion."""
        with self._lock:
            if not self._validate_live_placement(plan, expected_epoch):
                return self._epoch
            next_epoch = self._epoch + 1
            try:
                self._save(epoch=next_epoch, placement=plan.proposed)
                self._epoch = next_epoch
                self._current = plan.proposed
                self._initial_placement = False
                self._initial_exposures = self._exposures
                self._pending_recommendation = None
            except BaseException:
                # A failed directory fsync can leave either version on disk.
                # Further local writes could overwrite an already published epoch.
                self._placement_poisoned = True
                raise
            return next_epoch

    def publish_recommendation(self, data):
        """Publish only a proposal issued for the current observation snapshot."""
        with self._lock:
            self._require_open()
            if type(data) is not bytes or data != self._pending_recommendation:
                raise ValueError("expert placement recommendation is stale or was not issued by this state")
            _atomic_replace(self.recommendation_path, data)
            self._pending_recommendation = None
