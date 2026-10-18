"""Native A4 expert banks; GPU grouped execution consumes all calibrations.

Resident banks own six device fields. Cached banks hold only cache identity and
geometry; immutable padded CPU authority belongs solely to the model's cache.
No CPU calibration arrays or individual-expert math remain in serving.
"""

from contextlib import contextmanager

import torch

from .grouped_a4 import _ext, run
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


def plan_tile(pairs, experts):
    """Device-plan items for ``pairs`` routed pairs over ``experts`` cells: 64 pairs once experts average 32 or more
    (each expert's weights then stream once per 64 pairs), else 16. A pair's bits never depend on it."""
    return 64 if pairs >= 32 * experts else 16


def uniform_input_bits(calibrations):
    """Prove exact prepared gate/up FP32 equality over the complete CPU bank."""
    if (calibrations.device.type != "cpu" or calibrations.dtype != torch.float32 or calibrations.ndim != 2
            or calibrations.shape[0] <= 0 or calibrations.shape[1] != 2 or not calibrations.is_contiguous()):
        raise ValueError("common input proof requires complete contiguous CPU FP32[experts,2] calibrations")
    bits = calibrations.view(torch.int32)
    return bool((bits == bits[0, 0]).all())


class ExpertsA4:
    """Prepared resident bank, with a fixed grouped execution/lease contract.

    Inputs and outputs must be disjoint from each other and expert storage.
    In ``run``, out-of-local-range picks are inactive zero slots; the device
    plan that ``run_plan`` consumes traps on IDs outside [0, plan experts), and
    the bank skips only the shared cell ``count``. Per-projection raw input
    multipliers, weight reciprocals and native mode-specific K sums stay exact.
    ``direct_input`` selects ModelOpt direct scales; CT banks keep reciprocal
    multipliers. Both execute the same device-planned kernels in ``run_plan``
    (graph replay never reads picks on the CPU); ``run`` remains the host-window
    path for leased (cached) banks and expert-parallel local picks with -1.
    """

    direct_input = False
    capturable = True
    run = run

    def __init__(self, fields, width, dims, *, swiglu, shared_input=False):
        # True is supplied only by the loader's complete CPU bit proof
        # (uniform_input_bits), before exact CPU->CUDA transfer; immutable bank
        # fields preserve that proof.
        self.shared_input = bool(shared_input)
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

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.fields)

    def run_plan(self, x, buf, *, prefill):
        """Fill routed slots of ``buf.y[:rows]`` from the device plan ``buf.plan``.

        ``buf`` holds ``rows`` (capacity), ``slots``, ``pick`` [rows, slots]
        INT32, a ``grouped.Plan`` routed over ``pick[:rows]`` with items of 16
        or 64 pairs (``plan_tile``), and ``y`` [rows, slots, dims] (BF16 when
        ``prefill``, else FP32). Plan items whose cell equals ``count`` (a
        resident shared expert) are skipped; their slots are the caller's.
        Three launches: activation codes (one per row for a proven shared input
        global), gate/up GEMMs + activation + down quantization (fused_up), and
        the down GEMM; each streams an expert's weights once per item. Shape-only
        K slices: SK1 for ``prefill``, else ``qmm.split_k``; a pair's bits never
        depend on the other pairs, the plan's items or the row count. The first
        call per buffer allocates and initializes retained scratch, so it must
        run eagerly before any CUDA graph capture over ``buf``.
        """
        from tensorfold.cuda.kernels import qmm

        rows, slots = x.shape[0], buf.slots
        plan = buf.plan
        if plan.tile not in (16, 64):
            raise ValueError("native A4 requires device-plan items of 16 or 64 pairs")
        activation = getattr(self.swiglu, "activation", None)
        if activation not in (0, 1):
            raise ValueError("native A4 device plans fuse a declared SiLU or tanh GELU activation")
        self._check(x, buf.pick[:rows], buf.y[:rows], prefill=prefill)
        # The buffer, not every expert bank, owns reusable shape-specific
        # storage. Allocate all retained tensors before publishing the tuple.
        key = (self.dims, self.width)
        held = getattr(buf, "a4_scratch", None)
        if held is None:
            held = buf.a4_scratch = {}
        scratch = held.get(key)
        if scratch is None:
            # Queried only on a first use; steady-state calls make no capture query.
            if torch.cuda.is_current_stream_capturing():
                # A captured allocation/initialization would leave eager users
                # of this buffer reading unwritten scratch until the first replay.
                raise RuntimeError("native A4 run_plan scratch must be retained by an eager call before capture")
            pairs = buf.rows * slots
            with torch.inference_mode(False):
                scratch = (torch.empty((2, pairs, self.dims // 2), dtype=torch.uint8, device=x.device),
                           torch.empty((2, pairs, self.dims // 16), dtype=torch.uint8, device=x.device),
                           torch.empty((1, pairs, self.width // 2), dtype=torch.uint8, device=x.device),
                           torch.empty((1, pairs, self.width // 16), dtype=torch.uint8, device=x.device))
            held[key] = scratch
        pairs = rows * slots
        dc, ds = (t[:, :pairs] for t in scratch[2:])
        ext = _ext()
        fields, direct = self.fields, self.direct_input
        # Plan backing has conservative spare rows, but every live item owns
        # at least one pair. Never expose more item rows than actual pairs.
        items = plan.items[:min(pairs, self.count + 1 + pairs // 16)]
        members = plan.members[:pairs]
        common = self.shared_input and rows <= 65535
        if common:
            row_plan = getattr(buf, "a4_row_plan", None)
            if row_plan is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("native A4 shared-input row plan must be built by an eager call before capture")
                with torch.inference_mode(False):
                    indices = torch.arange(buf.rows, dtype=torch.int32, device=x.device)
                    zero = torch.zeros_like(indices)
                    row_plan = torch.stack((indices, zero, zero), dim=1)
                buf.a4_row_plan = row_plan
                del indices, zero
            # Reuse the existing input backing; the shared representation is a
            # smaller prefix. Projection-major narrowing would leave non-contiguous
            # outer strides, so both forms are flat prefixes reshaped to the plan.
            codes = scratch[0].view(-1)[:rows * (self.dims // 2)].view(1, rows, self.dims // 2)
            scales = scratch[1].view(-1)[:rows * (self.dims // 16)].view(1, rows, self.dims // 16)
            ext.quantize(x, row_plan[:rows], fields[4][:1, :1], codes, scales, False, direct)
        else:
            codes = scratch[0].view(-1)[:2 * pairs * (self.dims // 2)].view(2, pairs, self.dims // 2)
            scales = scratch[1].view(-1)[:2 * pairs * (self.dims // 16)].view(2, pairs, self.dims // 16)
            ext.quantize_plan(x, items, plan.counts, members, fields[4], codes, scales, slots, direct, plan.tile)
        ext.fused_up(codes, scales, fields[0], fields[2], fields[4], fields[5], items, plan.counts, members, dc, ds,
                     self.width, self.dims, 1 if prefill else qmm.split_k(self.width, self.dims), slots, common, direct,
                     plan.tile, activation)
        ext.down_plan(dc, ds, self.native_down, fields[3], fields[5], items, plan.counts, members, buf.y[:rows],
                      self.dims, self.width, 1 if prefill else qmm.split_k(self.dims, self.width), slots, direct,
                      plan.tile)

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


class PlanBuffers:
    """Graph-static device routing plan and slot outputs for up to ``rows`` rows.

    Routing writes ``pick[:rows]``; ``route`` groups it; ``ExpertsA4.run_plan``
    fills ``y[:rows]``. All storage is allocated here or retained by the bank's
    first use, so CUDA graph replay reuses fixed addresses.
    """

    def __init__(self, rows, slots, experts, dims, device, *, prefill):
        from tensorfold.cuda.experts import Plan

        self.rows, self.slots = rows, slots
        self.pick = torch.empty((rows, slots), dtype=torch.int32, device=device)
        self.plan = Plan(rows, slots, experts, device)
        self.y = torch.empty((rows, slots, dims), dtype=torch.bfloat16 if prefill else torch.float32, device=device)

    def route(self, rows):
        """Group ``pick[:rows]`` into items of ``plan_tile`` pairs; a pair's bits never depend on the item size."""
        from tensorfold.cuda import pdl
        from tensorfold.cuda.experts import _ext

        plan = self.plan
        plan.tile = plan_tile(rows * self.slots, plan.experts)
        _ext().plan(self.pick[:rows], rows * self.slots, plan.experts, plan.tile, plan.members, plan.items,
                    plan.counts, plan.rank, plan.hist, pdl.ON)


class CachedExpertsA4(ExpertsA4):
    """Geometry and cache identity only; no retained CPU authority aliases."""

    capturable = False      # leases plan host windows from CPU picks

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
