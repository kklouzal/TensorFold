"""One bounded immutable merged-input span and its target/MTP consumers.

The producer owns values and coordinates until all declared consumers finish
their staging on their borrowed streams. Consumers may slice their one lease
several times, but must join any additional streams before completing it. This
is a chunk plus optional one-row lookahead, never a retained prompt archive.
Values and coordinates share a device. A CUDA producer joins all output work
onto the current stream before publication; consumers use its readiness event.
One coordinator thread owns span/lease mutations; device streams carry readers.
"""

import torch

from .position_coordinates import CoordinateProof, IssuedCoordinates, _digest, _signature


class ReadyInputSpan:
    def __init__(self, *, start, token_ids, values, positions, position, rope_delta, profile_identity,
                 generation, media_identity, max_rows, hidden_size, vocab_size, roles):
        if (type(start) is not int or start < 0 or type(max_rows) is not int or max_rows <= 0
                or type(hidden_size) is not int or hidden_size <= 0
                or type(vocab_size) is not int or vocab_size <= 0
                or type(token_ids) not in (tuple, list) or not 0 < len(token_ids) <= max_rows
                or start > (1 << 63) - 1 - len(token_ids)
                or any(type(t) is not int or not 0 <= t < vocab_size for t in token_ids)):
            raise ValueError("merged input rows/IDs exceed their admitted span")
        if (not _digest(profile_identity) or media_identity is not None and not _digest(media_identity)
                or type(generation) is not int or not 0 <= generation < 1 << 63
                or type(rope_delta) is not int or not -(1 << 63) <= rope_delta < 1 << 63):
            raise ValueError("merged input identity/generation/continuation delta is invalid")
        if type(roles) not in (tuple, list) or len(roles) not in (1, 2):
            raise ValueError("merged inputs declare target and optional MTP consumers")
        roles = tuple(roles)
        if roles not in (("target",), ("target", "mtp")):
            raise ValueError("merged inputs declare target and optional MTP consumers")
        if type(positions) is not IssuedCoordinates or type(positions.proof) is not CoordinateProof:
            raise ValueError("merged inputs require coordinates issued from actual host rows")
        positions.proof.validate(positions.tensor, position, profile_identity=profile_identity,
                                 generation=generation, media_identity=media_identity,
                                 row_start=start, row_count=len(token_ids))
        issued = positions
        positions = issued.tensor
        if (values.dtype != torch.float16 or values.shape != (len(token_ids), hidden_size)
                or not values.is_contiguous() or positions.dtype != torch.int32
                or positions.shape != (len(token_ids), 3) or not positions.is_contiguous()):
            raise ValueError("merged inputs require contiguous FP16 rows and int32 three-axis coordinates")
        if positions.device != values.device:
            raise ValueError("merged input coordinates must share the value device")
        self.start, self.token_ids = start, tuple(token_ids)
        self.values, self.positions, self.rope_delta = values, positions, rope_delta
        self.coordinates, self.position, self.position_proof = issued, position, issued.proof
        self.profile_identity, self.generation, self.media_identity = profile_identity, generation, media_identity
        self._leases = dict.fromkeys(roles)
        self.closed = self.poisoned = False
        # These handles keep old storage alive even after an erroneous set_().
        self._storage = values.untyped_storage(), positions.untyped_storage()
        self._signature = self._current_signature()
        self._ready = None
        if values.is_cuda:
            with torch.cuda.device(values.device):
                self._ready = torch.cuda.Event()
                self._ready.record(torch.cuda.current_stream(values.device))

    def _current_signature(self):
        return _signature(self.values)

    def _check(self):
        if self.closed or self.poisoned:
            raise ValueError("merged input span is retired or poisoned")
        if self._current_signature() != self._signature:
            self.poisoned = True
            raise ValueError("merged input storage changed before consumer retirement")
        try:
            self.position_proof.validate(self.positions, self.position, profile_identity=self.profile_identity,
                                         generation=self.generation, media_identity=self.media_identity,
                                         row_start=self.start, row_count=len(self.token_ids))
        except BaseException:
            self.poisoned = True
            raise

    @property
    def end(self):
        return self.start + len(self.token_ids)

    def borrow(self, role, *, profile_identity, generation, media_identity, stream=None):
        """Borrow at the staging boundary, after validating current request identity.

        Coordinates belong to these input rows. MTP separately supplies its
        hidden/cache-row coordinates; shifting embeddings never shifts RoPE.
        """
        self._check()
        if (profile_identity != self.profile_identity or type(generation) is not int or generation != self.generation
                or media_identity != self.media_identity):
            raise ValueError("merged input span belongs to another request/profile/media generation")
        if role not in self._leases or self._leases[role] is not None:
            raise ValueError("merged input consumer is undeclared or already borrowed")
        if self.values.is_cuda:
            stream = torch.cuda.current_stream(self.values.device) if stream is None else stream
            if stream.device != self.values.device:
                raise ValueError("merged input consumer stream belongs to another device")
            stream.wait_event(self._ready)
        elif stream is not None:
            raise ValueError("CPU merged-input controls do not accept a CUDA stream")
        lease = InputLease(self, stream)
        self._leases[role] = lease
        return lease

    def retire(self, *, cancelled=False):
        """Release at bounded buffer reuse; failed fences retain every owner."""
        if self.closed:
            return
        self._check()
        if type(cancelled) is not bool:
            raise ValueError("merged input cancellation requires an explicit boolean")
        if (any(lease is not None and not lease.completed for lease in self._leases.values())
                or not cancelled and any(lease is None for lease in self._leases.values())):
            raise ValueError("merged input consumers have not completed staging")
        try:
            if self._ready is not None:
                self._ready.synchronize()
            for lease in self._leases.values():
                if lease is not None and lease.done is not None:
                    lease.done.synchronize()
        except BaseException:
            self.poisoned = True
            raise
        self.values = self.positions = self._ready = self._storage = None
        self.coordinates = self.position = self.position_proof = None
        self._leases.clear()
        self.closed = True


class InputLease:
    def __init__(self, owner, stream):
        self.owner, self.stream = owner, stream
        self.completed, self.done = False, None

    def rows(self, start, token_ids):
        """Get exact original-ID rows; visual IDs never imply token embeddings."""
        self.owner._check()
        if self.completed:
            raise ValueError("merged input consumer already completed staging")
        if type(token_ids) not in (tuple, list) or type(start) is not int:
            raise ValueError("merged input row request requires original IDs and an absolute start")
        lo, count = start - self.owner.start, len(token_ids)
        if (lo < 0 or count <= 0 or count > len(self.owner.token_ids) - lo
                or any(type(t) is not int for t in token_ids)
                or tuple(token_ids) != self.owner.token_ids[lo:lo + count]):
            raise ValueError("merged input rows do not match the original expanded sequence")
        return self.owner.values[lo:lo + count]

    def complete(self):
        """Record after the last staging read on the stream borrowed above."""
        if self.completed:
            return
        self.owner._check()
        if self.stream is not None:
            try:
                with torch.cuda.device(self.owner.values.device):
                    self.done = torch.cuda.Event()
                    self.done.record(self.stream)
            except BaseException:
                self.owner.poisoned = True
                raise
        self.completed = True
