"""Native activation-FP4 routed experts, separate from Flash Next's W4A16 banks.

The authority contains original checkpoint code/scale bits in checkpoint MMA
order. CPU-owned scalar metadata supplies per-expert calibrations without a
GPU scalar readback. A run borrows one expert at a time, retires each cache
lease after its submitted consumers, and retains no per-bank device scratch.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

from . import checkpoint
from .host_layout import authority_fields_a4
from .linear import Fp4Linear

PAIRS = 128


def pack_projection_a4(weight, scale):
    """CPU-only bit permutation into code words followed by E4M3 block scales.

    No floating conversion is performed. Both output dimensions are multiples
    of64, so native tiles contain no absent/padded output rows.
    """
    if (weight.device.type != "cpu" or scale.device.type != "cpu" or weight.dtype != torch.uint8
            or weight.ndim != 2 or scale.dtype != torch.float8_e4m3fn):
        raise ValueError("native A4 packing requires CPU U8 codes and E4M3 scales")
    n, k = weight.shape[0], weight.shape[1] * 2
    if n <= 0 or k <= 0 or n % 64 or k % 64 or tuple(scale.shape) != (n, k // 16):
        raise ValueError("native A4 projection dimensions/scales must form complete64-wide tiles")
    words = (weight.contiguous().view(torch.int32).view(n // 64, 8, 8, k // 64, 2, 4)
             .permute(0, 3, 1, 2, 5, 4).contiguous())
    bs = scale.contiguous().view(torch.uint8).view(n // 64, 64, k // 64, 4).permute(0, 2, 1, 3).contiguous()
    return torch.cat((words.view(torch.uint8).reshape(-1), bs.reshape(-1)))


def projection_a4(blob, n, k, weight_scale, input_scale):
    """Borrow a packed projection without repacking or reading device scalars."""
    cut = n * k // 2
    words = blob[:cut].view(torch.int32).view(n // 64, k // 64, 8, 32, 2)
    scales = blob[cut:].view(n // 64, k // 64, 64, 4)
    return Fp4Linear(words, scales, weight_scale, n, k, act=input_scale)


def _span(t):
    if not t.numel():
        return 0, 0
    return t.data_ptr(), t.data_ptr() + (1 + sum((n - 1) * s for n, s in zip(t.shape, t.stride()))) * t.element_size()


def _overlap(a, b):
    left, right = _span(a), _span(b)
    return a.device == b.device and left[0] < right[1] and right[0] < left[1]


class ExpertsA4:
    """Resident six-field bank; the public picks are local expert IDs.

    ``swiglu`` is a prepared BF16 SiLU/product callable with a BF16 intermediate
    SiLU result. Inputs/outputs must not alias each other or any expert storage.
    Invalid picks (negative or >=count) write zero without reading authority.
    Native decode and prompt retain their existing independent K-chain rules.
    """

    capturable = False

    def __init__(self, fields, scalar_fields, width, dims, *, swiglu):
        self.width, self.dims = width, dims
        if len(fields) != 6 or fields[0].ndim != 3:
            raise ValueError("native A4 expert bank requires six fields")
        self.count = fields[0].shape[0]
        geometry = authority_fields_a4(self.count, width, dims)
        if (len(scalar_fields) != 4 or not callable(swiglu)
                or any(tuple(t.shape) != shape or t.dtype != (torch.uint8 if dtype == "U8" else torch.float32)
                       or not t.is_contiguous() for t, (shape, dtype) in zip(fields, geometry))):
            raise ValueError("native A4 expert fields disagree with authority geometry")
        self.fields, self.scalars, self.swiglu = tuple(fields), tuple(scalar_fields), swiglu
        self.device = fields[0].device
        if self.device.type != "cuda" or any(t.device != self.device for t in fields):
            raise ValueError("native A4 resident fields require one CUDA device")
        self._validate_scalars(geometry)

    def _validate_scalars(self, geometry):
        for t, (shape, _) in zip(self.scalars, geometry[2:]):
            if t.device.type != "cpu" or t.dtype != torch.float32 or tuple(t.shape) != shape or not t.is_contiguous():
                raise ValueError("native A4 calibrations must be immutable CPU FP32 expert fields")

    def device_tensors(self):
        return self.fields

    @contextmanager
    def _expert(self, expert):
        yield self.fields, expert

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

    def run(self, x, picks, out, *, prefill):
        self._check(x, picks, out, prefill=prefill)
        out.zero_()
        run = checkpoint.prompt if prefill else checkpoint.matmul
        for lo in range(0, x.shape[0], PAIRS):
            rows = picks[lo:lo + PAIRS].cpu().tolist()
            groups = {}
            for row, values in enumerate(rows):
                for slot, expert in enumerate(values):
                    if 0 <= expert < self.count:
                        groups.setdefault(expert, []).append((lo + row, slot))
            for expert in sorted(groups):
                pairs = groups[expert]
                with self._expert(expert) as (fields, cell):
                    gu_scale, dn_scale, gu_input, dn_input = self.scalars
                    gate = projection_a4(fields[0][cell, 0], self.width, self.dims,
                                         float(gu_scale[expert, 0]), float(gu_input[expert, 0]))
                    up = projection_a4(fields[0][cell, 1], self.width, self.dims,
                                       float(gu_scale[expert, 1]), float(gu_input[expert, 1]))
                    down = projection_a4(fields[1][cell], self.dims, self.width,
                                         float(dn_scale[expert, 0]), float(dn_input[expert, 0]))
                    for at in range(0, len(pairs), PAIRS):
                        block = pairs[at:at + PAIRS]
                        indices = torch.tensor(block, dtype=torch.int64, device=self.device)
                        selected = x.index_select(0, indices[:, 0])
                        g = run(checkpoint.A4, selected, gate)
                        u = run(checkpoint.A4, selected, up)
                        hidden = self.swiglu(g, u)
                        value = run(checkpoint.A4, hidden, down, f32=not prefill)
                        out[indices[:, 0], indices[:, 1]] = value


class CachedExpertsA4(ExpertsA4):
    """Same arithmetic with a model-owned cache's bounded six-field leases."""

    def __init__(self, authority, width, dims, *, cache, layer_id, swiglu):
        self.width, self.dims, self.count = width, dims, authority[0].shape[0]
        self.cache, self.layer_id, self.device = cache, layer_id, cache.device
        self.scalars, self.swiglu = tuple(authority[2:]), swiglu
        self._validate_scalars(authority_fields_a4(self.count, width, dims))
        if not callable(swiglu):
            raise TypeError("native A4 requires the prepared BF16 SiLU/product callable")
        cache.register(layer_id, tuple(authority))

    def device_tensors(self):
        return (self.cache._pool,)

    @contextmanager
    def _expert(self, expert):
        with self.cache.lease(self.layer_id, [expert]) as (fields, mapping):
            yield fields, mapping[expert]
