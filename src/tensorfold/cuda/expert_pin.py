"""Model-owned CUDA registration of newly allocated immutable expert authority.

The plan charges the complete remaining allocation geometry before loading;
already allocated anonymous authority is never charged or copied a second time.
Resource observations are not reservations. Exact CPU bytes stay owned until
the cache has drained all copy and compute consumers and calls close().
"""

from __future__ import annotations

import threading
from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
from mmap import PAGESIZE
from pathlib import Path

import torch

from .capacity import host_cgroup_bytes
from .host_pin import _host


@dataclass(frozen=True)
class PinAdmission:
    mode: str
    planned_bytes: int
    backing_bytes: int
    future_bytes: int
    host_room_bytes: int | None
    cgroup_room_bytes: int | None
    admitted: bool
    reason: str
    snapshot_is_reservation: bool = False


class PinUnavailable(RuntimeError):
    """Required pinning was refused before an authority became usable."""


class PinCleanupError(RuntimeError):
    """Failed registration retirement retains its owner in this outcome."""

    def __init__(self, owner, code):
        super().__init__(f"expert host unregister failed with CUDA status {code}; owner retained")
        self.owner = owner


def backing_bytes(size: int) -> int:
    """Registered extent plus two pages for backing alignment/page rounding."""

    return ((size + PAGESIZE - 1) // PAGESIZE) * PAGESIZE + 2 * PAGESIZE if size else 0


class ExpertPinPolicy:
    """One loader's bounded authority allocation/pin owner; no concurrent close.

    Pin-capable modes require full allocation byte geometry. ``auto`` keeps the
    same valid pageable allocation on explicit resource/capability refusal;
    ``on`` fails. Unknown driver errors always fail. Abandoning a still-active
    native registration is a fatal ownership violation; callers must retain a
    failed close outcome and stop the owning model.
    """

    def __init__(self, planned_bytes: int, *, mode: str = "off", future_bytes: int = 0,
                 allocation_bytes: tuple[int, ...] | None = None):
        if mode not in ("auto", "on", "off"):
            raise ValueError("expert pin mode must be auto, on or off")
        if any(type(n) is not int or n < 0 for n in (planned_bytes, future_bytes)):
            raise ValueError("expert pin payload/future bytes must be nonnegative integers")
        if allocation_bytes is not None and (not isinstance(allocation_bytes, tuple)
                or any(type(n) is not int or n <= 0 for n in allocation_bytes)
                or sum(allocation_bytes) != planned_bytes):
            raise ValueError("expert pin geometry must exactly cover the planned positive allocations")
        self._remaining = None if allocation_bytes is None else Counter(allocation_bytes)
        self._pins = [None] * (0 if allocation_bytes is None else len(allocation_bytes))
        self._count = self.allocated_bytes = self.pinned_bytes = self.refusals = 0
        self._closed = False
        self._failure = None
        self._lock = threading.RLock()
        footprint = sum(map(backing_bytes, allocation_bytes or ()))
        host_room = cgroup_room = None
        admitted, reason = False, "explicit_off"
        if mode != "off":
            if allocation_bytes is None:
                reason = "allocation_geometry_unavailable"
            elif not planned_bytes:
                reason = "empty_authority"
            else:
                host = _host(Path("/proc/meminfo"))
                host_room = None if host is None else host["room_bytes"]
                allowance = host_cgroup_bytes()
                cgroup_room = None if allowance is None else max(0, allowance - (0 if host is None else host["reserve_bytes"]))
                if host_room is None:
                    reason = "host_memory_unavailable"
                elif host_room < footprint + future_bytes or (cgroup_room is not None and cgroup_room < footprint + future_bytes):
                    reason = "insufficient_host_or_cgroup_room"
                elif not torch.cuda.is_available():
                    reason = "cuda_unavailable"
                else:
                    admitted, reason = True, "admitted_new_authority"
        self.admission = PinAdmission(mode, planned_bytes, footprint, future_bytes, host_room, cgroup_room, admitted, reason)
        self._enabled = admitted
        if mode == "on" and not admitted:
            raise PinUnavailable(f"required expert host pinning refused: {reason}")

    def empty(self, shape, *, dtype, tag: str = "") -> torch.Tensor:
        if not isinstance(shape, (tuple, list)) or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError("expert authority shape must contain nonnegative integer dimensions")
        count = 1
        for value in shape:
            count *= value
        size = count * torch.empty((), dtype=dtype, device="meta").element_size()
        with self._lock, torch.inference_mode(False):
            if self._closed or self._failure is not None:
                raise RuntimeError("expert pin policy is closed or failed")
            if self.allocated_bytes + size > self.admission.planned_bytes:
                raise ValueError("expert authority allocation exceeds its admitted payload budget")
            if size and self._remaining is not None and not self._remaining[size]:
                raise ValueError("expert authority allocation differs from its admitted geometry")
            try:
                return self._allocate_and_commit(shape, dtype, size, tag)
            except BaseException:
                # Includes the interval after successful registration and
                # before every Python bookkeeping update has committed.
                self._failure = "authority_allocation_failed"
                raise

    def _allocate_and_commit(self, shape, dtype, size, tag):
        if self._enabled and size:
            extent = ((size + PAGESIZE - 1) // PAGESIZE) * PAGESIZE
            backing = torch.empty((extent + PAGESIZE - 1,), dtype=torch.uint8, device="cpu")
            offset = (-backing.data_ptr()) % PAGESIZE
            pages = backing.narrow(0, offset, extent)
            result = pages[:size].view(dtype).view(shape)
            native = _native()
            owner = native.Registration(pages)
            self._pins[self._count] = owner  # publish before the native registration can succeed
            code = owner.pin()
            if code:
                if self.admission.mode != "auto" or code not in (native.memory_allocation, native.not_supported):
                    self._failure = "host_registration_failed"
                    raise PinUnavailable(f"expert host registration failed ({tag[:160]}): {native.error_text(code)}")
                self.refusals += 1
                self._enabled = False
            else:
                self.pinned_bytes += size
        else:
            result = torch.empty(shape, dtype=dtype, device="cpu")
        self.allocated_bytes += size
        if size and self._remaining is not None:
            self._remaining[size] -= 1
            self._count += 1
        return result

    def owns(self, tensor: torch.Tensor) -> bool:
        """Registration capability for an immutable CPU subrange; resolved before HOT use."""

        with self._lock:
            if self._closed or self._failure is not None:
                raise RuntimeError("expert pin policy is closed or failed")
            return any(owner is not None and owner.contains(tensor) for owner in self._pins)

    def receipt(self) -> dict:
        with self._lock:
            return {**asdict(self.admission), "allocated_bytes": self.allocated_bytes,
                    "pinned_bytes": self.pinned_bytes, "registration_refusals": self.refusals,
                    "pin_enabled": self._enabled, "closed": self._closed}

    def close(self) -> None:
        """Called only after every copy/consumer has retired; failed owners remain retained."""

        with self._lock:
            if self._closed:
                return
            try:
                self._retire_all()
            except BaseException:
                self._failure = "host_unregister_failed"
                raise

    def _retire_all(self):
        errors = []
        primary = None
        for owner in reversed(self._pins):
            if owner is not None:
                try:
                    code = owner.close()
                except BaseException as error:
                    if primary is None:
                        primary = error
                    else:
                        errors.append(error)
                else:
                    if code:
                        errors.append(PinCleanupError(self, code))
        active = any(owner is not None and owner.active for owner in self._pins)
        if active:
            # Keep the model owner in the transported outcome even when
            # the primary was an opaque exception or KeyboardInterrupt.
            errors.append(PinCleanupError(self, "unretired registration"))
        if errors or primary is not None:
            self._failure = "host_unregister_failed"
            from tensorfold.cleanup import raise_failures

            if not active:
                self._pins.clear()
                self._closed = True
            raise_failures(primary, errors)
        self._pins.clear()
        self._closed = True


@lru_cache(maxsize=1)
def _native():
    from .build import load

    return load(name="tensorfold_expert_pin_v1", sources=[str(Path(__file__).with_suffix(".cpp"))],
                with_cuda=True, extra_cflags=["-O3"], verbose=False)
