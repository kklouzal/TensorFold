"""Explicit preparation of the native allocator in a caller-owned fresh process.

The opt-in owner proves a new exec/import path, no previous allocator setters or
consumers, and exclusive initialization. Empty snapshots/default last text only
corroborate that construction: they do not expose every persistent native option.
The policy is process-global and remains fixed through every consumer's lifetime.
Failure is terminal startup failure; parsing can partially mutate native state,
so this module never retries, restores settings or allocates a proof Tensor.
"""
from __future__ import annotations

import os
import warnings

from tensorfold.cuda.allocation_bound import UnsupportedAllocator, _capture, _segments

SETTING = "expandable_segments:True"
ENVIRONMENT = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF",
               "PYTORCH_NO_CUDA_MEMORY_CACHING", "TORCH_CUDA_EXPANDABLE_SEGMENTS_IPC")


def check_request(policy: str, *, tp: int, rank: int, streams: int) -> None:
    """Cheap opt-in argument admission, before template/device/payload work."""
    if type(policy) is not str or policy != "expandable":
        raise ValueError("native_allocator must be None or 'expandable'")
    if any(type(value) is not int for value in (tp, rank, streams)) or (tp, rank, streams) != (1, 0, 1):
        raise ValueError("native_allocator='expandable' requires one rank and one stream (tp=1, rank=0, streams=1)")


def check_environment() -> None:
    """Refuse explicit overrides before downloads and renew that proof at mutation."""
    if any(name in os.environ for name in ENVIRONMENT):
        raise UnsupportedAllocator("explicit native expansion conflicts with allocator environment overrides")


def _fresh_settings() -> dict:
    # Actual pinned ef46 native settings schema. Unexposed large/non-split
    # fields require the fresh-process construction precondition above.
    return {"PYTORCH_CUDA_ALLOC_CONF": "", "max_split_size": -1,
            "garbage_collection_threshold": 0.0, "expandable_segments": False,
            "pinned_num_register_threads": 1, "release_lock_on_cudamalloc": False,
            "pinned_use_cuda_host_register": False, "graph_capture_record_stream_reuse": False,
            "max_round_threshold": -1, "max_cached_size": -1,
            "roundup_power2_divisions": {str(1 << index): 0 for index in range(16)}}


def _check_settings(settings, expected) -> None:
    if (type(settings) is not dict or set(settings) != set(expected)
            or any(type(settings[key]) is not type(value) for key, value in expected.items())
            or settings != expected):
        raise UnsupportedAllocator("explicit native expansion requires the known fresh native allocator settings")


def enable_fresh_expansion(torch) -> dict:
    """Configure exactly once after device admission and before allocator tensors.

    The caller owns freshness and exclusion; this function cannot discover old
    hidden settings or lock arbitrary foreign CUDA consumers out of a process.
    Omit the policy for preconfigured, embedded or shared-engine processes.
    """
    check_environment()
    if tuple(torch.cuda.get_device_capability(0)) != (12, 1):
        raise UnsupportedAllocator("explicit native expansion is qualified for SM12.1 only")
    before, _ = _capture(torch)
    _segments(before, 0)
    expected = _fresh_settings()
    _check_settings(before["allocator_settings"], expected)
    if before["segments"]:
        raise UnsupportedAllocator("explicit native expansion requires no previous CUDA allocator consumers")
    setter = getattr(torch._C, "_accelerator_setAllocatorSettings", None)
    if not callable(setter):
        raise UnsupportedAllocator("native allocator configuration capability is unavailable")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        setter(SETTING)
        after, _ = _capture(torch)
    _segments(after, 0)
    expected.update(expandable_segments=True, PYTORCH_CUDA_ALLOC_CONF=SETTING)
    _check_settings(after["allocator_settings"], expected)
    if after["segments"]:
        raise UnsupportedAllocator("allocator configuration unexpectedly created CUDA tensor storage")
    return {"policy": "expandable", "backend": "native", "settings": expected,
            "ownership": "caller-owned fresh process; no prior setters/consumers; fixed until process retirement"}
