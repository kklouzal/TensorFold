"""Preparation-only native CUDA allocator reservation bound.

The public snapshot exposes segments but omits effective rounding settings. The
allocator snapshot capability is therefore required as well as the documented
backend query. Its schema and rounding policy are checked here, without a Torch
hash allowlist. The verified source is PyTorch1a66f681 CUDACachingAllocator and
AllocatorConfig; changing runtime/policy requires the affected native gates.

The owner must prepare in a quiescent process: the bootstrap is the first CUDA
allocator tensor on its device, and configuration remains fixed through
planning/allocation. Validation detects changed configuration; it does not lock
foreign callers out of the process-global allocator. No snapshots, tensor or
addresses escape in the receipt. Reused cached storage may require fewer new
device bytes; this bounds an allocation's additional physical reservation.
It does not promise successful allocation under driver/virtual-address limits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

_MIB = 1 << 20
_MIN_BLOCK = 512
_MAX_BYTES = (1 << 63) - 1
_DIVISION_KEYS = tuple(str(1 << index) for index in range(16))


class UnsupportedAllocator(ValueError):
    """The active allocator cannot supply a proved automatic pool bound."""


def _integer(value: Any, name: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < int(positive) or value > _MAX_BYTES:
        raise UnsupportedAllocator(f"{name} must be a {'positive' if positive else 'nonnegative'} signed64 integer")
    return value


def _settings(snapshot: Any) -> tuple[str, bool, tuple[int, ...]]:
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("allocator_settings"), dict):
        raise UnsupportedAllocator("automatic expert pools require effective allocator snapshot settings")
    settings = snapshot["allocator_settings"]
    if type(settings.get("expandable_segments")) is not bool:
        raise UnsupportedAllocator("allocator snapshot must expose effective expandable_segments")
    if not isinstance(settings.get("PYTORCH_CUDA_ALLOC_CONF"), str):
        raise UnsupportedAllocator("allocator snapshot must expose its effective configuration string")
    divisions = settings.get("roundup_power2_divisions")
    if not isinstance(divisions, dict) or set(divisions) != set(_DIVISION_KEYS):
        raise UnsupportedAllocator("allocator snapshot must expose all16 native power-of-two rounding intervals")
    values = tuple(_integer(divisions[key], "roundup_power2_divisions") for key in _DIVISION_KEYS)
    if any((value and value & (value - 1)) or value > ((1 << 64) - 1) // _MIN_BLOCK for value in values):
        raise UnsupportedAllocator("allocator rounding divisions must be supported powers of two")
    try:
        identity = json.dumps(settings, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise UnsupportedAllocator("allocator settings schema is unsupported") from error
    return identity, settings["expandable_segments"], values


def _capture(torch) -> tuple[dict, tuple[str, bool, tuple[int, ...]]]:
    try:
        memory = torch.cuda.memory
        backend = memory.get_allocator_backend()
        if backend != "native":
            raise UnsupportedAllocator("automatic expert pools require the native CUDA caching allocator")
        snapshot = memory._snapshot()
        if memory.get_allocator_backend() != backend:
            raise UnsupportedAllocator("CUDA allocator backend changed during preparation")
    except (AttributeError, TypeError) as error:
        raise UnsupportedAllocator("CUDA allocator inspection capability is unavailable") from error
    return snapshot, _settings(snapshot)


def _segments(snapshot: Any, device: int) -> list[dict]:
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("segments"), list):
        raise UnsupportedAllocator("allocator snapshot segment schema is unsupported")
    segments = []
    for segment in snapshot["segments"]:
        if not isinstance(segment, dict) or type(segment.get("device")) is not int:
            raise UnsupportedAllocator("allocator segment device schema is unsupported")
        if segment["device"] == device:
            segments.append(segment)
    return segments


def _rounded(size: int, divisions: tuple[int, ...]) -> int:
    if size < _MIN_BLOCK:
        return _MIN_BLOCK
    index = min(max(size.bit_length() - 1 - 20, 0), len(divisions) - 1)
    divisor = divisions[index]
    if divisor > 1 and size > _MIN_BLOCK * divisor:
        if not size & (size - 1):
            return size
        floor = 1 << (size.bit_length() - 1)
        unit = floor >> (divisor.bit_length() - 1)
        return ((size + unit - 1) // unit) * unit if unit else 2 * floor
    return ((size + _MIN_BLOCK - 1) // _MIN_BLOCK) * _MIN_BLOCK


@dataclass(frozen=True)
class NativeAllocationBound:
    device: int
    large_quantum: int
    expandable: bool
    divisions: tuple[int, ...]
    bootstrap_bytes: int
    bootstrap_rounded_bytes: int
    torch_version: str
    torch_git: str | None
    _settings_identity: str = field(repr=False)
    _torch: Any = field(repr=False, compare=False)

    def __call__(self, requested: int) -> int:
        requested = _integer(requested, "allocation bytes")
        if not requested:
            return 0
        rounded = _rounded(requested, self.divisions)
        if rounded <= _MIB:
            bound = 2 * _MIB
        elif self.expandable:
            bound = -(-rounded // self.large_quantum) * self.large_quantum
        else:
            # Native's <10MiB case may reserve more than >=10MiB. Its
            # nondecreasing upper envelope is required by pool binary search.
            bound = max(self.large_quantum, -(-rounded // (2 * _MIB)) * (2 * _MIB))
        return _integer(bound, "allocator reservation bound", positive=True)

    @property
    def receipt(self) -> dict:
        return {"policy": "native-round512-divisions-observed-segment-v1", "backend": "native",
                "device": self.device, "large_segment_bytes": self.large_quantum,
                "expandable_segments": self.expandable,
                "roundup_power2_divisions": dict(zip(_DIVISION_KEYS, self.divisions)),
                "bootstrap_requested_bytes": self.bootstrap_bytes,
                "bootstrap_rounded_bytes": self.bootstrap_rounded_bytes,
                "torch_version": self.torch_version, "torch_git": self.torch_git,
                "allocator_settings": json.loads(self._settings_identity),
                "proof": "first-device-allocator-tensor; fixed allocator configuration",
                "bound": "additional physical reservation; no cached-storage credit"}

    def validate_current(self) -> None:
        _, settings = _capture(self._torch)
        if settings != (self._settings_identity, self.expandable, self.divisions):
            raise UnsupportedAllocator("CUDA allocator configuration changed after the bootstrap proof")


def pool_allocation_bound(torch, bootstrap, *, before_snapshot) -> NativeAllocationBound:
    """Prove rounding from a first, whole-storage CUDA allocation in (1,10)MiB.

    Capture ``before_snapshot`` with the same allocator snapshot API immediately
    before creating the bootstrap pool. It must contain no CUDA segments on the
    selected device. Host allocations and unrelated devices are permitted.
    The caller retains the bootstrap through transactional replacement and owns
    process-wide preparation exclusion/configuration stability.
    """
    snapshot, settings = _capture(torch)
    if _settings(before_snapshot) != settings:
        raise UnsupportedAllocator("CUDA allocator configuration changed around the bootstrap allocation")
    try:
        if bootstrap.device.type != "cuda" or not bootstrap.is_contiguous() or bootstrap.storage_offset() != 0:
            raise UnsupportedAllocator("bootstrap must be a complete contiguous CUDA storage")
        device = _integer(bootstrap.device.index, "bootstrap CUDA device")
        storage = bootstrap.untyped_storage()
        pointer = storage.data_ptr()
        if type(pointer) is not int or pointer <= 0:
            raise UnsupportedAllocator("bootstrap storage pointer identity is unavailable")
        requested = _integer(storage.nbytes(), "bootstrap storage bytes", positive=True)
        if bootstrap.data_ptr() != pointer or bootstrap.numel() * bootstrap.element_size() != requested:
            raise UnsupportedAllocator("bootstrap must cover exactly its original CUDA allocation")
    except AttributeError as error:
        raise UnsupportedAllocator("bootstrap CUDA storage inspection capability is unavailable") from error
    if _segments(before_snapshot, device):
        raise UnsupportedAllocator("automatic pool proof requires the bootstrap to be the first CUDA allocator tensor")
    segments = _segments(snapshot, device)
    if len(segments) != 1:
        raise UnsupportedAllocator("bootstrap proof requires one isolated CUDA segment")
    identity, expandable, divisions = settings
    rounded = _rounded(requested, divisions)
    if not _MIB < rounded < 10 * _MIB:
        raise UnsupportedAllocator("bootstrap rounded allocation must lie strictly between1and10MiB")
    segment = segments[0]
    if (segment.get("address") != pointer or segment.get("segment_type") != "large"
            or segment.get("segment_pool_id") not in ((0, 0), [0, 0])
            or type(segment.get("is_expandable")) is not bool or segment["is_expandable"] != expandable):
        raise UnsupportedAllocator("bootstrap is not the native global allocator's first large segment")
    quantum = _integer(segment.get("total_size"), "observed large segment bytes", positive=True)
    if quantum <= 10 * _MIB or quantum % _MIB:
        raise UnsupportedAllocator("observed large segment quantum is outside the supported native policy")
    blocks = segment.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise UnsupportedAllocator("bootstrap allocator block schema is unsupported")
    cursor, active = pointer, []
    for block in blocks:
        if not isinstance(block, dict) or block.get("address") != cursor:
            raise UnsupportedAllocator("bootstrap allocator block layout is unsupported")
        size = _integer(block.get("size"), "allocator block bytes", positive=True)
        cursor += size
        if block.get("state") == "active_allocated":
            active.append(block)
        elif block.get("state") != "inactive":
            raise UnsupportedAllocator("bootstrap allocator contains another unfinished operation")
    if (cursor != pointer + quantum or len(active) != 1 or active[0].get("address") != pointer
            or active[0].get("requested_size") != requested or active[0]["size"] != rounded
            or segment.get("allocated_size") != rounded or segment.get("active_size") != rounded):
        raise UnsupportedAllocator("bootstrap allocation identity/rounding does not match the native policy")
    return NativeAllocationBound(device, quantum, expandable, divisions, requested, rounded,
                                 str(torch.__version__), getattr(torch.version, "git_version", None),
                                 identity, torch)
