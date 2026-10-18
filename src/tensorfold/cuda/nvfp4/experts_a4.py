"""Native A4 expert banks; GPU grouped execution consumes all calibrations.

Resident banks own six device fields. Cached banks hold only cache identity and
geometry; immutable padded CPU authority belongs solely to the model's cache.
No CPU calibration arrays or individual-expert math remain in serving.
"""

from contextlib import contextmanager

import torch

from .grouped_a4 import run
from .host_layout import authority_fields_a4


def pack_projection_a4(weight, scale):
    """Exact CPU/CUDA bit permutation into words followed by E4M3 block scales.

    No floating conversion is performed. Both output dimensions are multiples
    of64, so native tiles contain no absent/padded output rows.
    """
    if (weight.device.type not in ("cpu", "cuda") or scale.device != weight.device or weight.dtype != torch.uint8
            or weight.ndim != 2 or scale.dtype != torch.float8_e4m3fn):
        raise ValueError("native A4 packing requires CPU/CUDA U8 codes and E4M3 scales on one device")
    n, k = weight.shape[0], weight.shape[1] * 2
    if n <= 0 or k <= 0 or n % 64 or k % 64 or tuple(scale.shape) != (n, k // 16):
        raise ValueError("native A4 projection dimensions/scales must form complete64-wide tiles")
    if weight.device.type == "cuda":
        from .checkpoint import pack4

        words = pack4(weight, n)
    else:
        words = (weight.contiguous().view(torch.int32).view(n // 64, 8, 8, k // 64, 2, 4)
                 .permute(0, 3, 1, 2, 5, 4).contiguous())
    bs = scale.contiguous().view(torch.uint8).view(n // 64, 64, k // 64, 4).permute(0, 2, 1, 3).contiguous()
    return torch.cat((words.view(torch.uint8).reshape(-1), bs.reshape(-1)))


def _span(t):
    if not t.numel():
        return 0, 0
    return t.data_ptr(), t.data_ptr() + (1 + sum((n - 1) * s for n, s in zip(t.shape, t.stride()))) * t.element_size()


def _overlap(a, b):
    left, right = _span(a), _span(b)
    return a.device == b.device and left[0] < right[1] and right[0] < left[1]


class ExpertsA4:
    """Prepared resident bank, with a fixed grouped execution/lease contract.

    Inputs and outputs must be disjoint from each other and expert storage.
    Out-of-local-range picks are inactive zero slots. Per-projection raw input
    multipliers, weight reciprocals and native mode-specific K sums stay exact.
    """

    capturable = False
    run = run

    def __init__(self, fields, width, dims, *, swiglu):
        if len(fields) != 6 or fields[0].ndim != 3:
            raise ValueError("native A4 expert bank requires six fields")
        self.width, self.dims, self.count = width, dims, fields[0].shape[0]
        geometry = authority_fields_a4(self.count, width, dims)
        if (not callable(swiglu) or any(tuple(t.shape) != shape
                or t.dtype != (torch.uint8 if dtype == "U8" else torch.float32)
                or not t.is_contiguous() for t, (shape, dtype) in zip(fields, geometry))):
            raise ValueError("native A4 resident fields disagree with admitted geometry")
        self.device = fields[0].device
        if self.device.type != "cuda" or any(t.device != self.device for t in fields):
            raise ValueError("native A4 resident fields require one CUDA device")
        self.fields, self.swiglu = tuple(fields), swiglu
        self.native_down = self.fields[1].unsqueeze(1)

    @property
    def capacity(self):
        return self.count

    def device_tensors(self):
        return self.fields

    @contextmanager
    def lease(self, ids):
        yield self.fields, {expert: expert for expert in ids}

    def _check(self, x, picks, out, *, prefill):
        if (type(prefill) is not bool or x.device != self.device or x.dtype != torch.bfloat16 or x.ndim != 2
                or x.shape[1] != self.dims or x.stride(-1) != 1 or x.stride(0) < self.dims
                or picks.device != self.device or picks.dtype != torch.int32 or picks.ndim != 2
                or picks.shape[0] != x.shape[0] or picks.stride(-1) != 1
                or picks.stride(0) < picks.shape[1] or not 0 < picks.shape[1] <= 1024
                or out.device != self.device or tuple(out.shape) != (*picks.shape, self.dims)
                or out.dtype != (torch.bfloat16 if prefill else torch.float32)
                or out.stride(-1) != 1 or out.stride(1) < self.dims
                or out.stride(0) < picks.shape[1] * out.stride(1)):
            raise ValueError("native A4 requires BF16 rows, INT32 routed picks and matching non-overlapping output slots")
        storage = self.device_tensors()
        if _overlap(x, out) or _overlap(picks, out) or any(_overlap(t, out) or _overlap(t, x) or _overlap(t, picks)
                                                        for t in storage):
            raise ValueError("native A4 rows, picks and outputs must not overlap expert storage or each other")


class CachedExpertsA4(ExpertsA4):
    """Geometry and cache identity only; no retained CPU authority aliases."""

    def __init__(self, count, width, dims, *, cache, layer_id, swiglu):
        authority_fields_a4(count, width, dims)
        if not callable(swiglu):
            raise TypeError("native A4 requires the prepared BF16 SiLU/product callable")
        self.width, self.dims, self.count = width, dims, count
        self.cache, self.layer_id, self.device, self.swiglu = cache, layer_id, cache.device, swiglu

    @property
    def capacity(self):
        return self.cache.capacity

    @property
    def native_down(self):
        return self.cache._native_down

    def device_tensors(self):
        # Keep one strong snapshot for alias inspection. Concurrent close may
        # retire the cache, but the subsequent lease owns admission/fencing.
        pool = self.cache._pool
        if pool is None:
            raise RuntimeError("cached native A4 bank is closed")
        return (pool,)

    @contextmanager
    def lease(self, ids):
        with self.cache.lease(self.layer_id, ids) as result:
            yield result
