"""Request epochs and nonblocking readiness for the admitted media producer.

Only the leader owns prepared replay and CPU work. A follower's request carries
immutable planning metadata, never an executor or an encoded-file consumer.
The request epoch survives admission retries; device spans use slot generations.
"""

from dataclasses import dataclass
from threading import RLock

import torch

from tensorfold.cleanup import raise_failures, rollback


def runtime(vision):
    return vision if getattr(vision, "is_media_runtime", False) is True else None


class MediaRequests:
    """Exclusive close authority shared by HTTP cleanup and the coordinator.

    Registry lock precedes ReplayStore's lock; the store never calls this owner.
    Entries key physical replay leases, so continuation views share authority.
    Backend subruns release claims; the logical caller requests the final close.
    A closed lease itself rejects a late claim, without tombstones.
    """
    def __init__(self, owner):
        self.owner, self._lock, self.active = owner, RLock(), {}

    def claim(self, request):
        from tensorfold.vision.media_frontend import PreparedReplayMedia
        from tensorfold.vision.images_http import ImageInputError
        from tensorfold.vision.replay import ReplayLease

        prepared = request.prepared
        with self._lock:
            if (type(prepared) is not PreparedReplayMedia or type(prepared.replay) is not ReplayLease
                    or prepared.executor is not self.owner.executor):
                raise ValueError("media claim requires this runtime's actual prepared replay owner")
            key = id(prepared.replay)
            previous = self.active.get(key)
            if previous is not None:
                if previous is not request:
                    raise ValueError("prepared media is already claimed by another request")
                return
            with prepared.replay._store._lock:
                prepared.replay._store._check_owner()
                if prepared.replay._closed or prepared.replay._retiring:
                    raise ImageInputError("prepared media replay retired before request admission")
                if len(self.active) >= self.owner.profile.cpu.max_replay_leases:
                    raise RuntimeError("active media owners exceed the admitted replay lease capacity")
                self.active[key] = request

    def close_if_unclaimed(self, prepared):
        """Final logical-caller close; False defers its intent until active retirement."""
        with self._lock:
            request = self.active.get(id(prepared.replay))
            if request is not None:
                # The caller's finalizer may precede CPU/rank retirement.
                # Keep final-close intent in this bounded active owner.
                request.close_requested = True
                return False
            prepared.close()
            return True

    def rollback_claim(self, request):
        """Undo only this unpublished claim; bind has not submitted any CPU job."""
        from tensorfold.vision.media_frontend import PreparedReplayMedia

        with self._lock:
            if type(request.prepared) is not PreparedReplayMedia:
                return
            if self.active.get(id(request.prepared.replay)) is request:
                self.release(request)

    def release(self, request):
        with self._lock:
            key = id(request.prepared.replay)
            if self.active.get(key) is not request:
                if request.released:
                    return  # A retried old release cannot mutate a later claim of this lease.
                raise RuntimeError("media request lost its exclusive prepared owner")
            if request.close_requested:
                request.prepared.close()
            # Publish the retry proof before dropping the registry entry. An
            # interruption on either side of dict.pop retains an idempotent
            # owner; a later subrun may safely claim the same physical lease.
            request.released = True
            self.active.pop(key)

    def check_drained(self):
        with self._lock:
            if self.active:
                raise RuntimeError("active media requests retain replay ownership during runtime retirement")


@dataclass
class MediaRequest:
    prepared: object
    epoch: int
    owner: object
    retired: bool = False
    released: bool = False
    failure: BaseException | None = None
    close_requested: bool = False

    @property
    def plan(self):
        return self.prepared.plan

    def _arguments(self):
        return {"generation": self.epoch, "profile_identity": self.owner.identity,
                "media_identity": self.plan.media_identity}

    def prime(self, start):
        """At most one current/ahead unit per request, bounded globally by MediaWork."""
        if self.failure is not None:
            raise self.failure
        if self.owner.work is None or self.retired:
            return
        for index, unit in enumerate(self.plan.units):
            if unit.run.end > start and not self.owner.producer.cached(self.plan, unit):
                self.owner.work.request(self.prepared, index, **self._arguments())
                return

    def ready(self, index):
        if self.retired:
            raise RuntimeError("media request has retired")
        return self.owner.work.check_ready(self.prepared, index, **self._arguments())

    def limit(self, start, count):
        """Keep one whole-unit encoder dependency and no unavailable visual row."""
        end = start + count
        for index, unit in enumerate(self.plan.units):
            if unit.run.end <= start:
                continue
            if unit.run.start >= end:
                break
            if not self.owner.producer.cached(self.plan, unit) and not self.ready(index):
                return max(0, unit.run.start - start)
            return min(end, unit.run.end) - start
        return count

    def pixels(self, unit):
        """Copy completed read-only pixels, retiring mappings before GPU work."""
        index = self.plan.units.index(unit)
        lease = self.owner.work.take_ready(self.prepared, index, **self._arguments())
        if lease is None:
            raise RuntimeError("a scheduled media unit lost its completed CPU owner")
        primary = None
        try:
            with lease.borrow() as borrowed:
                # NumPy owns this bounded copy. No exported mmap/Py_buffer is
                # retained by the tensor after the pixel lease closes.
                values = torch.from_numpy(borrowed.values.copy())
        except BaseException as error:
            primary = error
        try:
            lease.close()
        except BaseException as cleanup:
            raise_failures(primary, [cleanup])
        if primary is not None:
            raise primary
        return values

    def cancel(self):
        """False retains ownership until the worker's terminal callback wakes us."""
        if self.retired:
            return True
        if self.owner.work is not None and not self.owner.work.cancel(self.prepared):
            return False
        self.retired = True
        return True


def bind(stream, owner, epoch):
    request = getattr(stream, "_media_request", None)
    if request is None:
        if (type(epoch) is not int or not 0 <= epoch < 1 << 63
                or tuple(stream.prompt) != stream.vision.token_ids
                or stream.vision.plan.profile_identity != owner.identity):
            raise ValueError("media request differs from its canonical prompt/profile/CPU epoch")
        request = MediaRequest(stream.vision, epoch, owner)
        registry = getattr(owner, "requests", None)
        try:
            if registry is not None:
                registry.claim(request)
            stream._media_request = request
        except BaseException as primary:
            if registry is not None:
                rollback(request, primary, lambda: registry.rollback_claim(request))
            raise
    elif request.epoch != epoch or request.owner is not owner or request.retired or request.released:
        raise ValueError("media request CPU epoch/runtime/claim is no longer active")
    return request


def prepare(stream, owner, epoch, wake):
    if owner is None or stream.vision is None:
        return True
    owner.check_media_health()
    request = bind(stream, owner, epoch)
    owner.work.set_wake(wake)
    request.prime(0)
    return owner.producer.cached(request.plan, request.plan.units[0]) or request.ready(0)


def retire(stream):
    request = getattr(stream, "_media_request", None)
    if request is None:
        return True
    if not request.cancel():
        return False
    if stream.st is None and not getattr(stream, "_media_rank_owned", False):
        release(stream)
    return True


def release(stream):
    """Release a subrun claim after CPU and all rank/State consumers retire.

    The caller must retain this bridge on uncertain retirement/ACK failure. The
    logical caller retains replay through subsequent subruns. Its final close
    either executes unclaimed or defers to this release after retirement.
    """
    request = getattr(stream, "_media_request", None)
    if request is None:
        return
    registry = getattr(request.owner, "requests", None)
    if request.released:
        if registry is not None:
            registry.release(request)
        stream.vision = None
        return
    if not request.cancel():
        raise RuntimeError("media CPU owner has not completed retirement")
    if registry is not None:
        registry.release(request)
    request.released = True
    stream.vision = None


def publish(owner, pieces):
    """Collectives execute before the caller enters nested input preparation."""
    from .media_mtp import attach_span
    from .media_producer import MediaInputSlice

    media = [(stream, start, count) for stream, start, count in pieces
             if getattr(stream, "_media_request", None) is not None]
    if not media:
        return
    slices = tuple(MediaInputSlice(stream._media_request.plan, stream.st.slot_generation, start, count)
                   for stream, start, count in media)
    leader = owner.work is not None
    spans = owner.producer.publish(slices, pixel_loader=(
        lambda index, unit: media[index][0]._media_request.pixels(unit)) if leader else None)
    for (stream, start, count), span in zip(media, spans, strict=True):
        attach_span(stream.st, span, profile_identity=owner.identity,
                    media_identity=stream._media_request.plan.media_identity)
        if leader:
            from tensorfold.vision.images_http import ImageInputError
            try:
                stream._media_request.prime(start + count)
            except ImageInputError as error:
                from .execution_identity import _unretired

                if _unretired(error):
                    raise
                owner.check_media_health()
                # This next-unit refusal must not invalidate the already
                # published current chunk or diverge the peer collective path.
                stream._media_request.failure = error


def retire_batch(owner, pieces, *, cancelled):
    if owner is None:
        return
    from .media_mtp import retire_span

    failures = []
    for stream, _start, _count in pieces:
        if getattr(stream, "_media_request", None) is not None:
            try:
                retire_span(stream.st, cancelled=cancelled)
            except BaseException as error:
                failures.append(error)
    if not failures:
        try:
            owner.producer.retire(cancelled=cancelled)
        except BaseException as error:
            failures.append(error)
    raise_failures(None, failures)
