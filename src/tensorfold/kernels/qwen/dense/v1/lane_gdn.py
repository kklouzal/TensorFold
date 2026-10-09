"""Share prefix state S0 across lanes using S_t = G_t S0 + sum_{i <= t} (G_t / G_i) d_i k_i^T; reordered arithmetic may change near-ties, while padding with decay 1 and beta 0 leaves state unchanged."""

from __future__ import annotations

from copy import copy
from itertools import islice
from operator import index
from typing import Any

import mlx.core as mx
import mlx.nn as nn

F32 = mx.float32
HIST = mx.bfloat16  # stored keys and deltas; the recurrence itself runs in float32
_INT32_MAX = (1 << 31) - 1
_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1


def _count(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer count")
    try:
        result = index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer count") from error
    if not minimum <= result <= _INT32_MAX:
        raise ValueError(f"{name} must be in [{minimum}, {_INT32_MAX}]")
    return result


def _storage(shape: tuple[int, ...], itemsize: int, name: str) -> None:
    """Prove MLX's size_t element/byte products before allocation or access.

    TARGET uses 64-bit size_t; MLX's ArrayDesc multiplication is unchecked.
    Zero-element tensors require no storage and retain their legal metadata.
    This is a representation bound, not an invented model/resource limit.
    """
    dimensions = tuple(_count(dimension, name + " dimension") for dimension in shape)
    size = index(itemsize)
    if size <= 0 or size > _UINT64_MAX:
        raise ValueError(f"{name} has an invalid element width")
    if 0 in dimensions:
        return
    elements = 1
    for dimension in dimensions:
        if elements > _UINT64_MAX // dimension:
            raise ValueError(f"{name} element product exceeds size_t")
        elements *= dimension
    if elements > _UINT64_MAX // size:
        raise ValueError(f"{name} byte product exceeds size_t")


def _array(value: Any, name: str, shape: tuple[int, ...] | None = None, dtype: Any = None) -> None:
    if not isinstance(value, mx.array):
        raise ValueError(f"{name} must be an MLX array")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if dtype is not None:
        valid = value.dtype == dtype
    else:
        valid = mx.issubdtype(value.dtype, mx.floating)
    if not valid:
        raise ValueError(f"{name} has an unsupported dtype {value.dtype}")
    _storage(tuple(value.shape), value.itemsize, name)


def _projection_storage(module: Any, input_shape: tuple[int, ...], input_dtype: Any,
                        output: int, name: str) -> Any:
    """Admit declared MLX projection allocations using current dtype promotion.

    Every required real floating result needs at least two bytes in MLX0.32.2/.3.
    That necessary bound also covers custom producers; their own allocations
    remain their responsibility. Known Linear/QuantizedLinear operations admit
    their exact input/parameter casts, intermediate result and bias addition.
    """
    shape = (*input_shape[:-1], output)
    _storage(shape, 2, name + " required floating result")
    if type(module) is getattr(nn, "QuantizedLinear", None):
        bits = _count(module.bits, name + " bits", minimum=1)
        group = _count(module.group_size, name + " group", minimum=1)
        k = input_shape[-1]
        if k * bits % 32 or k % group:
            raise ValueError(f"{name} packed input/group geometry is inconsistent")
        _array(module.weight, name + " weight", (output, k * bits // 32), mx.uint32)
        mode = getattr(module, "mode", "affine")
        if mode == "affine":
            for field in ("scales", "biases"):
                _array(getattr(module, field), name + " " + field, (output, k // group))
            dtype = mx.result_type(input_dtype, module.scales.dtype, module.biases.dtype)
            for value in (module.scales, module.biases):
                _storage(tuple(value.shape), dtype.size, name + " affine parameter cast")
        elif mode in ("mxfp4", "mxfp8", "nvfp4"):
            _array(module.scales, name + " scales", (output, k // group), mx.uint8)
            dtype = input_dtype
        else:
            raise ValueError(f"{name} has an unsupported quantization mode")
        _storage(input_shape, dtype.size, name + " input cast")
        _storage(shape, dtype.size, name + " quantized result")
        if "bias" in module:
            _array(module.bias, name + " bias", (output,))
            dtype = mx.result_type(dtype, module.bias.dtype)
    elif type(module) is getattr(nn, "Linear", None):
        _array(module.weight, name + " weight", (output, input_shape[-1]))
        dtype = mx.result_type(input_dtype, module.weight.dtype)
        if "bias" in module:
            _array(module.bias, name + " bias", (output,))
            dtype = mx.result_type(dtype, module.bias.dtype)
        _storage(input_shape, dtype.size, name + " input cast")
        _storage(tuple(module.weight.shape), dtype.size, name + " weight cast")
    else:
        return None
    _storage(shape, dtype.size, name + " output")
    return dtype


def _convolution_storage(module: Any, input_shape: tuple[int, ...], input_dtype: Any,
                         output_shape: tuple[int, ...]) -> None:
    """Admit stock convolution casts/results; custom producer owns its internals."""
    _storage(output_shape, 2, "required floating convolution result")
    if type(module) is not getattr(nn, "Conv1d", None):
        return
    dtype = mx.result_type(input_dtype, module.weight.dtype)
    _storage(input_shape, dtype.size, "convolution input cast")
    _storage(tuple(module.weight.shape), dtype.size, "convolution weight cast")
    _storage(output_shape, dtype.size, "convolution result")
    if "bias" in module:
        _array(module.bias, "convolution bias", (output_shape[-1],))
        _storage(output_shape, mx.result_type(dtype, module.bias.dtype).size, "convolution bias addition")

_STEP_SOURCE = """
    // One threadgroup per (lane, value head); thread dv owns output row dv.
    const uint dv = thread_position_in_threadgroup.x;
    const uint group = threadgroup_position_in_grid.x;
    const uint n = group / HV;
    const uint h = group % HV;
    const uint hk = h / (HV / HK);
    const uint sg = dv / 32;
    const uint sl = dv % 32;
    const int t = tlen[0];

    threadgroup float ks[DK];
    threadgroup float qs[DK];
    threadgroup float aw[CAP];
    threadgroup float cw[CAP];
    threadgroup float red[DV / 32];

    ks[dv] = static_cast<float>(k[(n * HK + hk) * DK + dv]);
    qs[dv] = static_cast<float>(q[(n * HK + hk) * DK + dv]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // S0 k and S0 q, from one batched matmul over all lanes outside the kernel.
    const float s0k = s0kq[((n * HV + h) * 2 + 0) * DV + dv];
    const float s0q = s0kq[((n * HV + h) * 2 + 1) * DV + dv];

    const float lg = log_prev[n * HV + h] + log_g[n * HV + h];
    const float decay = metal::exp(lg);

    // Past keys against this key and this query; one simdgroup per stride of i.
    for (int i = int(sg); i < t; i += int(DV / 32)) {
        const device HistT* krow = k_hist + ((n * HK + hk) * CAP + i) * DK;
        float a = 0.0f;
        float c = 0.0f;
        for (int j = int(sl); j < DK; j += 32) {
            float kv = static_cast<float>(krow[j]);
            a += kv * ks[j];
            c += kv * qs[j];
        }
        a = simd_sum(a);
        c = simd_sum(c);
        if (sl == 0) {
            float w = metal::exp(lg - lg_hist[(n * HV + h) * CAP + i]);
            aw[i] = w * a;
            cw[i] = w * c;
        }
    }
    float kq = simd_sum(ks[dv] * qs[dv]);
    if (sl == 0) {
        red[sg] = kq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    kq = 0.0f;
    for (int r = 0; r < int(DV / 32); ++r) {
        kq += red[r];
    }

    const device HistT* dcol = d_hist + (n * HV + h) * CAP * DV + dv;
    float memory = decay * s0k;
    float yv = decay * s0q;
    for (int i = 0; i < t; ++i) {
        float dval = static_cast<float>(dcol[i * DV]);
        memory += aw[i] * dval;
        yv += cw[i] * dval;
    }
    const float delta = (static_cast<float>(v[(n * HV + h) * DV + dv]) - memory) * beta[n * HV + h];
    yv += kq * delta;
    y[(n * HV + h) * DV + dv] = static_cast<OutT>(yv);
    delta_out[(n * HV + h) * DV + dv] = static_cast<HistT>(delta);
    if (dv == 0) {
        log_out[n * HV + h] = lg;
    }
"""

# Value heads sharing a key head reuse history dot products and read S0 k and S0 q in the batched matmul's [Hv, N, 2, Dv] order.
_STEP_SOURCE_KH = """
    const uint tid = thread_position_in_threadgroup.x;
    const uint r = tid / DV;
    const uint dv = tid % DV;
    const uint group = threadgroup_position_in_grid.x;
    const uint n = group / HK;
    const uint hk = group % HK;
    const uint h = hk * R + r;
    const uint sg = tid / 32;
    const uint sl = tid % 32;
    const int t = tlen[0];
    const uint lanes = uint(tlen[1]);
    constexpr int NSG = (R * DV) / 32;

    threadgroup float ks[DK];
    threadgroup float qs[DK];
    threadgroup float dk[CAP];
    threadgroup float dq[CAP];
    threadgroup float aw[R * CAP];
    threadgroup float cw[R * CAP];
    threadgroup float red[NSG];

    if (tid < DK) {
        ks[tid] = static_cast<float>(k[(n * HK + hk) * DK + tid]);
        qs[tid] = static_cast<float>(q[(n * HK + hk) * DK + tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float kqp = (tid < DK) ? ks[tid] * qs[tid] : 0.0f;
    kqp = simd_sum(kqp);
    if (sl == 0) {
        red[sg] = kqp;
    }
    // Past keys against this key and this query, once per key head.
    for (int i = int(sg); i < t; i += NSG) {
        const device HistT* krow = k_hist + ((n * HK + hk) * CAP + i) * DK;
        float a = 0.0f;
        float c = 0.0f;
        for (int j = int(sl); j < DK; j += 32) {
            float kv = static_cast<float>(krow[j]);
            a += kv * ks[j];
            c += kv * qs[j];
        }
        a = simd_sum(a);
        c = simd_sum(c);
        if (sl == 0) {
            dk[i] = a;
            dq[i] = c;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Decay weights per value head.
    for (int e = int(tid); e < R * t; e += R * DV) {
        const int rr = e / t;
        const int i = e % t;
        const uint hh = hk * R + uint(rr);
        const float lgr = log_prev[n * HV + hh] + log_g[n * HV + hh];
        const float w = metal::exp(lgr - lg_hist[(n * HV + hh) * CAP + i]);
        aw[rr * CAP + i] = w * dk[i];
        cw[rr * CAP + i] = w * dq[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float kq = 0.0f;
    for (int s2 = 0; s2 < DK / 32; ++s2) {
        kq += red[s2];
    }

    const float lg = log_prev[n * HV + h] + log_g[n * HV + h];
    const float decay = metal::exp(lg);
    const float s0k = s0kq[((h * lanes + n) * 2 + 0) * DV + dv];
    const float s0q = s0kq[((h * lanes + n) * 2 + 1) * DV + dv];
    const device HistT* dcol = d_hist + (n * HV + h) * CAP * DV + dv;
    float memory = decay * s0k;
    float yv = decay * s0q;
    for (int i = 0; i < t; ++i) {
        float dval = static_cast<float>(dcol[i * DV]);
        memory += aw[r * CAP + i] * dval;
        yv += cw[r * CAP + i] * dval;
    }
    const float delta = (static_cast<float>(v[(n * HV + h) * DV + dv]) - memory) * beta[n * HV + h];
    yv += kq * delta;
    y[(n * HV + h) * DV + dv] = static_cast<OutT>(yv);
    delta_out[(n * HV + h) * DV + dv] = static_cast<HistT>(delta);
    if (dv == 0) {
        log_out[n * HV + h] = lg;
    }
"""

_STEP_KERNEL = None
_STEP_KERNEL_KH = None


def _tn_array(t: int, n: int) -> mx.array:
    """History length and lane count owned by this native operation."""

    return mx.array([_count(t, "history position"), _count(n, "lane count")], dtype=mx.int32)


def _step_kernel_kh():
    global _STEP_KERNEL_KH
    if _STEP_KERNEL_KH is None and mx.metal.is_available():
        _STEP_KERNEL_KH = mx.fast.metal_kernel(
            name="tensorfold_lane_gdn_step_kh",
            input_names=["q", "k", "v", "log_g", "beta", "s0kq", "log_prev", "k_hist", "d_hist",
                         "lg_hist", "tlen"],
            output_names=["y", "delta_out", "log_out"],
            source=_STEP_SOURCE_KH,
        )
    return _STEP_KERNEL_KH


def _t_array(t: int) -> mx.array:
    """The history length owned by this native operation."""

    return mx.array([_count(t, "history position")], dtype=mx.int32)


@mx.compile
def _gates(a_log: mx.array, dt_bias: mx.array, a: mx.array, b: mx.array) -> tuple[mx.array, mx.array]:
    """log of the step decay and the write strength, both float32."""

    log_g = -mx.exp(a_log.astype(F32)) * nn.softplus((a + dt_bias).astype(F32))
    return log_g, mx.sigmoid(b.astype(F32))


def _step_kernel():
    global _STEP_KERNEL
    if _STEP_KERNEL is None and mx.metal.is_available():
        _STEP_KERNEL = mx.fast.metal_kernel(
            name="tensorfold_lane_gdn_step",
            input_names=["q", "k", "v", "log_g", "beta", "s0kq", "log_prev", "k_hist", "d_hist",
                         "lg_hist", "tlen"],
            output_names=["y", "delta_out", "log_out"],
            source=_STEP_SOURCE,
        )
    return _STEP_KERNEL


class LaneGDNCache:
    """One owner's gated-delta state; calls and state replacement are serialized.

    Head geometry is fixed at construction. ``filter`` changes lane count and
    history grows as steps complete. Arrays use MLX unified memory. Native steps
    use the Metal kernel's GPU stream, including with CPU-default MLX producers.
    MLX orders producer dependencies across streams. Evaluating
    returned graphs belongs to the caller; discard this cache if evaluation
    fails. Mutating a borrowed buffer while a
    graph uses it violates MLX's ownership contract.
    """

    growth = 32

    def __init__(self, conv: mx.array, s0: mx.array, *, key_heads: int, capacity: int = 64) -> None:
        # conv: [N, kernel - 1, C] per lane; s0: [Hv, Dv, Dk] float32, shared
        _array(s0, "s0")
        _array(conv, "conv")
        if s0.ndim != 3 or any(int(d) <= 0 for d in s0.shape):
            raise ValueError("s0 must be [Hv, Dv, Dk]")
        if conv.ndim != 3:
            raise ValueError("conv must be [N, kernel - 1, C]")
        heads = _count(key_heads, "key_heads", minimum=1)
        cap = max(1, _count(capacity, "capacity"))
        hv, dv, dk = map(int, s0.shape)
        if hv % heads:
            raise ValueError("key_heads must divide value heads")
        lanes = int(conv.shape[0])
        if max(lanes, hv, dv, dk) > _INT32_MAX:
            raise ValueError("LaneGDNCache dimensions exceed MLX integer ranges")
        for shape, itemsize, name in (
                ((hv, dv, dk), 4, "FP32 prefix state"),
                ((lanes, hv), 4, "decay state"),
                ((lanes, heads, cap, dk), 2, "key history"),
                ((lanes, hv, cap, dv), 2, "delta history"),
                ((lanes, hv, cap), 4, "log history")):
            _storage(shape, itemsize, name)
        self.conv = copy(conv)
        self.s0 = copy(mx.contiguous(s0.astype(F32)))                     # [Hv, Dv, Dk]
        self.value_heads, self.key_dim, self.value_dim = hv, dk, dv
        self.key_heads = heads
        self.repeat = self.value_heads // self.key_heads
        self.log_g = mx.zeros((lanes, self.value_heads), dtype=F32)
        self.t = 0
        self.k_hist = mx.zeros((lanes, self.key_heads, cap, self.key_dim), dtype=HIST)
        self.d_hist = mx.zeros((lanes, self.value_heads, cap, self.value_dim), dtype=HIST)
        self.lg_hist = mx.zeros((lanes, self.value_heads, cap), dtype=F32)
        self.lengths: mx.array | None = None
        self.left_padding = None
        self._pending: tuple[mx.array, mx.array, mx.array] | None = None
        self._failed = False

    @property
    def s0_t(self) -> mx.array:
        """Derive the transpose from the current authoritative prefix state."""
        return mx.contiguous(self.s0.transpose(0, 2, 1))

    def _admit_state(self) -> tuple[int, int]:
        if getattr(self, "_failed", False):
            raise RuntimeError("LaneGDNCache is unusable after a failed layer call; rebuild it")
        _array(self.conv, "conv")
        if self.conv.ndim != 3:
            raise ValueError("conv must be [N, kernel - 1, C]")
        n = _count(int(self.conv.shape[0]), "lane count")
        hk, hv, dk, dv = self.key_heads, self.value_heads, self.key_dim, self.value_dim
        for value, name in ((hk, "key_heads"), (hv, "value_heads"), (dk, "key_dim"), (dv, "value_dim")):
            _count(value, name, minimum=1)
        if hv % hk or self.repeat != hv // hk:
            raise ValueError("LaneGDNCache head geometry is inconsistent")
        _array(self.k_hist, "k_hist", dtype=HIST)
        if self.k_hist.ndim != 4:
            raise ValueError("k_hist must be [N, Hk, capacity, Dk]")
        cap = _count(int(self.k_hist.shape[2]), "history capacity", minimum=1)
        _count(self.t, "history position")
        if self.t > cap:
            raise ValueError("history position exceeds its capacity")
        for value, name, shape, dtype in (
                (self.s0, "s0", (hv, dv, dk), F32),
                (self.log_g, "log_g state", (n, hv), None),
                (self.k_hist, "k_hist", (n, hk, cap, dk), HIST),
                (self.d_hist, "d_hist", (n, hv, cap, dv), HIST),
                (self.lg_hist, "lg_hist", (n, hv, cap), F32)):
            _array(value, name, shape, dtype)
        if self._pending is not None:
            if not isinstance(self._pending, tuple) or len(self._pending) != 3:
                raise ValueError("pending history must hold key, delta and log state")
            for value, name, shape in zip(self._pending, ("pending key", "pending delta", "pending log"),
                                          ((n, hk, dk), (n, hv, dv), (n, hv))):
                _array(value, name, shape)
        if self.lengths is not None:
            if not isinstance(self.lengths, mx.array) or self.lengths.shape != (n,) \
                    or not mx.issubdtype(self.lengths.dtype, mx.integer):
                raise ValueError("lengths must be one integer count per lane")
        return n, cap

    def _admit_step(self, q: mx.array, k: mx.array, v: mx.array, log_g: mx.array, beta: mx.array) -> None:
        n, _ = self._admit_state()
        if n == 0:
            raise ValueError("step requires at least one active lane")
        for value, name, shape in (
                (q, "q", (n, self.key_heads, self.key_dim)),
                (k, "k", (n, self.key_heads, self.key_dim)),
                (v, "v", (n, self.value_heads, self.value_dim)),
                (log_g, "log_g", (n, self.value_heads)),
                (beta, "beta", (n, self.value_heads))):
            _array(value, name, shape)
        # The fallback explicitly widens stored histories and input vectors to
        # FP32. Head repetition and weighting allocate different logical shapes.
        upto = self.t + (1 if self._pending is not None else 0)
        history_bytes = max(4, self.log_g.itemsize, log_g.itemsize)
        work_bytes = max(history_bytes, beta.itemsize)
        for shape, itemsize, name in (
                (tuple(q.shape), 4, "FP32 query"),
                (tuple(k.shape), 4, "FP32 key"),
                (tuple(v.shape), 4, "FP32 value"),
                ((n, self.value_heads, self.key_dim), 4, "expanded FP32 query/key"),
                ((n, self.value_heads, self.value_dim), work_bytes, "recurrence output/delta"),
                ((n, self.key_heads, upto, self.key_dim), 4, "FP32 key history"),
                ((n, self.value_heads, upto, self.value_dim), 4, "FP32 delta history"),
                ((n, self.value_heads, upto), history_bytes, "history weights")):
            _storage(shape, itemsize, name)

    # -- the cache protocol the model reads ---------------------------------------
    @property
    def lanes(self) -> int:
        return int(self.conv.shape[0])

    @property
    def state(self) -> list[mx.array]:
        return [self.conv, self.log_g, self.k_hist, self.d_hist, self.lg_hist]

    def make_mask(self, N: int) -> Any:
        N = _count(N, "mask rows")
        self._admit_state()
        if self.lengths is not None:
            return mx.arange(N) < self.lengths[:, None]
        return None

    def prepare(self, lengths: Any = None, **kwargs: Any) -> None:
        self._admit_state()
        if lengths is not None:
            if isinstance(lengths, mx.array):
                if (lengths.ndim != 1 or tuple(lengths.shape) != (self.lanes,)
                        or not mx.issubdtype(lengths.dtype, mx.integer)):
                    raise ValueError("lengths must contain one integer count per lane")
                _storage(tuple(lengths.shape), lengths.itemsize, "prepared lengths")
                values = lengths.tolist()
            else:
                values = list(islice(iter(lengths), self.lanes + 1))
            if len(values) != self.lanes:
                raise ValueError("lengths must contain one count per lane")
            _storage((self.lanes,), 4, "int32 prepared lengths")
            self.lengths = mx.array([_count(value, "lane length") for value in values], dtype=mx.int32)

    def finalize(self) -> None:
        self.lengths = None

    def advance(self, n: int) -> None:
        n = _count(n, "advanced rows")
        self._admit_state()
        if self.lengths is not None:
            if any(not -(1 << 31) <= value - n <= _INT32_MAX for value in self.lengths.tolist()):
                raise ValueError("advanced lane lengths exceed their int32 representation")
            self.lengths = self.lengths - n

    def filter(self, keep: mx.array) -> None:
        lanes, _ = self._admit_state()
        if not isinstance(keep, mx.array) or keep.ndim != 1 or not mx.issubdtype(keep.dtype, mx.integer):
            raise ValueError("keep must be one-dimensional integer lane indices")
        selected_lanes = _count(int(keep.shape[0]), "selected lane count")
        _storage((selected_lanes,), 4, "selected indices")
        for value in self.state + list(self._pending or ()) + ([self.lengths] if self.lengths is not None else []):
            _storage((selected_lanes, *value.shape[1:]), value.itemsize, "selected cache state")
        indices = keep.tolist()
        if any(not -lanes <= value < lanes for value in indices):
            raise ValueError("keep contains a lane index outside the cache")
        selected = mx.array(indices, dtype=mx.int32)
        pending = tuple(a[selected] for a in self._pending) if self._pending is not None else None
        arrays = [a[selected] for a in self.state]
        lengths = self.lengths[selected] if self.lengths is not None else None
        self.conv, self.log_g, self.k_hist, self.d_hist, self.lg_hist = arrays
        self._pending, self.lengths = pending, lengths

    @property
    def nbytes(self) -> int:
        self._admit_state()
        return sum(int(a.nbytes) for a in self.state)

    # -- the recurrence ----------------------------------------------------------
    def _grow(self) -> None:
        n, cap = self._admit_state()
        extra = _count(self.growth, "history growth", minimum=1)
        if cap + extra > _INT32_MAX:
            raise ValueError("history growth exceeds MLX integer ranges")
        for value in (self.k_hist, self.d_hist, self.lg_hist):
            _storage((n, value.shape[1], cap + extra, *value.shape[3:]), value.itemsize, "grown cache history")
        keys = mx.concatenate(
            [self.k_hist, mx.zeros((n, self.key_heads, extra, self.key_dim), dtype=HIST)], axis=2)
        deltas = mx.concatenate(
            [self.d_hist, mx.zeros((n, self.value_heads, extra, self.value_dim), dtype=HIST)], axis=2)
        logs = mx.concatenate(
            [self.lg_hist, mx.zeros((n, self.value_heads, extra), dtype=F32)], axis=2)
        self.k_hist, self.d_hist, self.lg_hist = keys, deltas, logs

    def _shared(self, x: mx.array) -> mx.array:
        # x: [N, Hv, Dk] -> S0 x: [N, Hv, Dv]
        _storage((self.lanes, self.value_heads, self.value_dim), max(4, x.itemsize), "shared-state product")
        return (x.transpose(1, 0, 2) @ self.s0_t).transpose(1, 0, 2)

    def _history(self, x: mx.array, upto: int, current_log: mx.array | None = None) -> mx.array:
        # sum_{i < upto} (G_t / G_i) (k_i . x) d_i for x per key head [N, Hk, Dk]
        upto = _count(upto, "history prefix")
        if upto > int(self.k_hist.shape[2]):
            raise ValueError("history prefix exceeds its capacity")
        current_log = self.log_g if current_log is None else current_log
        work_bytes = max(4, x.itemsize, current_log.itemsize)
        for shape, itemsize, name in (
                ((self.lanes, self.key_heads, upto, self.key_dim), 4, "FP32 history keys"),
                ((self.lanes, self.value_heads, upto, self.value_dim), 4, "FP32 history deltas"),
                ((self.lanes, self.value_heads, upto), work_bytes, "weighted history"),
                ((self.lanes, self.value_heads, self.value_dim), work_bytes, "history product")):
            _storage(shape, itemsize, name)
        keys = self.k_hist[:, :, :upto].astype(F32)
        deltas = self.d_hist[:, :, :upto].astype(F32)
        logs = self.lg_hist[:, :, :upto]
        dots = (keys @ x[..., None])[..., 0]                     # [N, Hk, t]
        if self.repeat > 1:
            dots = mx.repeat(dots, self.repeat, axis=1)         # [N, Hv, t]
        weights = mx.exp(current_log[..., None] - logs) * dots  # [N, Hv, t]
        return (weights[:, :, None, :] @ deltas)[:, :, 0, :]     # [N, Hv, Dv]

    use_kernel = True
    kernel_version = 2  # 2: one threadgroup per key head; 1: one per value head

    def _flush(self) -> None:
        """Defer each history write until the next step so its buffer has one user and MLX can update it in place."""

        if self._pending is None:
            return
        self._admit_state()
        if self.t >= int(self.k_hist.shape[2]):
            self._grow()
        k, delta, log_new = self._pending
        t = self.t
        keys, deltas, logs = copy(self.k_hist), copy(self.d_hist), copy(self.lg_hist)
        keys[:, :, t, :] = k if k.dtype == HIST else k.astype(HIST)
        deltas[:, :, t, :] = delta if delta.dtype == HIST else delta.astype(HIST)
        logs[:, :, t] = log_new
        self.k_hist, self.d_hist, self.lg_hist = keys, deltas, logs
        self._pending = None
        self.t = t + 1

    def step(self, q: mx.array, k: mx.array, v: mx.array, log_g: mx.array, beta: mx.array) -> mx.array:
        """One position for every lane. q, k: [N, Hk, Dk]; v: [N, Hv, Dv]; log_g, beta: [N, Hv]."""

        self._admit_step(q, k, v, log_g, beta)
        if not isinstance(self.use_kernel, bool) or isinstance(self.kernel_version, bool) or self.kernel_version not in (1, 2):
            raise ValueError("LaneGDNCache requires a boolean kernel policy and kernel version 1 or 2")
        usable = (self.use_kernel and self.key_dim == 128 and self.value_dim == 128
                  and mx.metal.is_available()
                  and self.log_g.dtype == F32
                  and all(a.dtype in (mx.float16, mx.bfloat16, F32) for a in (q, k, v, log_g, beta)))
        if usable:
            n, cap = self.lanes, int(self.k_hist.shape[2])
            if self._pending is not None and self.t >= cap:
                cap = _count(cap + _count(self.growth, "history growth", minimum=1), "grown history capacity", minimum=1)
            if n * self.value_heads * self.value_dim > _INT32_MAX \
                    or max(n * self.key_heads * cap * self.key_dim,
                           n * self.value_heads * cap * self.value_dim) > _UINT32_MAX:
                raise ValueError("LaneGDNCache geometry exceeds native grid or offset ranges")
        self._flush()
        if usable and self.kernel_version == 2 and self.repeat * self.value_dim <= 1024:
            return self._step_kh(q, k, v, log_g, beta)
        kernel = _step_kernel() if usable else None
        if kernel is not None:
            n = self.lanes
            cap = int(self.k_hist.shape[2])
            # [Hv, 2N, Dk] @ [Hv, Dk, Dv]: S0 read once for every lane's key and query.
            kq = mx.concatenate([k, q], axis=1)                              # [N, 2Hk, Dk]
            kq = mx.repeat(kq.reshape(n, 2, self.key_heads, 1, self.key_dim), self.repeat, axis=3)
            kq = kq.reshape(n, 2, self.value_heads, self.key_dim).transpose(2, 0, 1, 3)
            s0kq = (kq.reshape(self.value_heads, 2 * n, self.key_dim).astype(F32) @ self.s0_t)
            s0kq = s0kq.reshape(self.value_heads, n, 2, self.value_dim).transpose(1, 0, 2, 3)
            y, delta, log_new = kernel(
                inputs=[q, k, v, log_g, beta, s0kq, self.log_g,
                        self.k_hist, self.d_hist, self.lg_hist, _t_array(self.t)],
                template=[("OutT", q.dtype), ("HistT", HIST), ("HK", self.key_heads),
                          ("HV", self.value_heads), ("DK", self.key_dim), ("DV", self.value_dim),
                          ("CAP", cap)],
                grid=(n * self.value_heads * self.value_dim, 1, 1),
                threadgroup=(self.value_dim, 1, 1),
                output_shapes=[(n, self.value_heads, self.value_dim),
                               (n, self.value_heads, self.value_dim), (n, self.value_heads)],
                output_dtypes=[q.dtype, HIST, F32],
            )
            pending = (copy(k), delta, copy(log_new))
            self.log_g, self._pending = log_new, pending
            return y
        qf, kf, vf = q.astype(F32), k.astype(F32), v.astype(F32)
        log_new = self.log_g + log_g
        decay = mx.exp(log_new)[..., None]                      # [N, Hv, 1]
        k_all = mx.repeat(kf, self.repeat, axis=1) if self.repeat > 1 else kf
        q_all = mx.repeat(qf, self.repeat, axis=1) if self.repeat > 1 else qf
        memory = decay * self._shared(k_all)
        output = decay * self._shared(q_all)
        if self.t:
            memory = memory + self._history(kf, self.t, log_new)
            output = output + self._history(qf, self.t, log_new)
        delta = (vf - memory) * beta[..., None]
        kq = (k_all * q_all).sum(axis=-1, keepdims=True)         # [N, Hv, 1]
        pending = (copy(kf), delta, copy(log_new))
        result = output + kq * delta
        self.log_g, self._pending = log_new, pending
        return result


def _step_kh(self: LaneGDNCache, q: mx.array, k: mx.array, v: mx.array, log_g: mx.array,
             beta: mx.array) -> mx.array:
    """``step`` through the key-head kernel (``_flush`` already done)."""

    n = self.lanes
    cap = int(self.k_hist.shape[2])
    # [Hv, 2N, Dk] @ [Hv, Dk, Dv] -> [Hv, N, 2, Dv], read in that order by the kernel.
    kq = mx.concatenate([k, q], axis=1)                              # [N, 2Hk, Dk]
    kq = mx.repeat(kq.reshape(n, 2, self.key_heads, 1, self.key_dim), self.repeat, axis=3)
    kq = kq.reshape(n, 2, self.value_heads, self.key_dim).transpose(2, 0, 1, 3)
    s0kq = kq.reshape(self.value_heads, 2 * n, self.key_dim).astype(F32) @ self.s0_t
    y, delta, log_new = _step_kernel_kh()(
        inputs=[q, k, v, log_g, beta, s0kq, self.log_g,
                self.k_hist, self.d_hist, self.lg_hist, _tn_array(self.t, n)],
        template=[("OutT", q.dtype), ("HistT", HIST), ("HK", self.key_heads), ("R", self.repeat),
                  ("HV", self.value_heads), ("DK", self.key_dim), ("DV", self.value_dim),
                  ("CAP", cap)],
        grid=(n * self.key_heads * self.repeat * self.value_dim, 1, 1),
        threadgroup=(self.repeat * self.value_dim, 1, 1),
        output_shapes=[(n, self.value_heads, self.value_dim),
                       (n, self.value_heads, self.value_dim), (n, self.value_heads)],
        output_dtypes=[q.dtype, HIST, F32],
    )
    pending = (copy(k), delta, copy(log_new))
    self.log_g, self._pending = log_new, pending
    return y


LaneGDNCache._step_kh = _step_kh  # type: ignore[attr-defined]


def lane_gdn_call(self: Any, inputs: mx.array, mask: Any, cache: LaneGDNCache) -> mx.array:
    """``GatedDeltaNet.__call__`` for a ``LaneGDNCache``: same projections, lane recurrence."""

    _array(inputs, "inputs")
    if inputs.ndim != 3:
        raise ValueError("inputs must be [N, rows, hidden dimensions]")
    B, S, _ = inputs.shape
    if B <= 0 or S <= 0:
        raise ValueError("lane_gdn_call requires active lanes and at least one row")
    lanes, _ = cache._admit_state()
    if lanes != B or (self.num_k_heads, self.num_v_heads, self.head_k_dim, self.head_v_dim) != (
            cache.key_heads, cache.value_heads, cache.key_dim, cache.value_dim):
        raise ValueError("layer and cache lane/head geometry must agree")
    n_keep = _count(self.conv_kernel_size, "convolution kernel size", minimum=1) - 1
    width = 2 * cache.key_heads * cache.key_dim + cache.value_heads * cache.value_dim
    if self.key_dim != cache.key_heads * cache.key_dim or self.conv_dim != width:
        raise ValueError("layer convolution/projected head geometry is inconsistent")
    _array(cache.conv, "convolution cache", (B, n_keep, width))
    conv_parameters = self.conv1d
    _array(conv_parameters.weight, "convolution weight", (width, n_keep + 1, 1))
    _array(self.A_log, "A_log", (cache.value_heads,))
    dt_parameters = self.dt_bias
    _array(dt_parameters, "dt_bias", (cache.value_heads,))
    if mask is not None:
        if not isinstance(mask, mx.array) or mx.broadcast_shapes(mask.shape, (B, S)) != (B, S):
            raise ValueError("mask must broadcast to [N, rows]")
    # Necessary result bounds do not evaluate arbitrary producer getters early.
    # At each original call point, capture once, admit and call that same value.
    for output, name in ((width, "qkv"), (cache.value_heads * cache.value_dim, "z"),
                         (cache.value_heads, "a/b"), (int(inputs.shape[-1]), "output projection")):
        _storage((B, S, output), 2, name + " required floating result")
    _storage((B, S, cache.value_heads), 4, "FP32 gate casts/results")
    _storage((B, S, cache.value_heads), max(2, dt_parameters.itemsize), "gate addition minimum")
    _storage((B, S, cache.value_heads, cache.value_dim), max(4, cache.log_g.itemsize), "recurrence output stack")
    _storage((B, S, cache.value_heads, cache.value_dim), inputs.itemsize, "recurrence output cast")
    _storage((B, n_keep + S, width), 2, "required floating convolution input")
    _storage((B, S, width), 2, "required floating convolution result")
    try:
        project_qkv = self.in_proj_qkv
        _projection_storage(project_qkv, tuple(inputs.shape), inputs.dtype, width, "qkv")
        qkv = project_qkv(inputs)
        project_z = self.in_proj_z
        _projection_storage(project_z, tuple(inputs.shape), inputs.dtype, cache.value_heads * cache.value_dim, "z")
        z = project_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
        project_b = self.in_proj_b
        _projection_storage(project_b, tuple(inputs.shape), inputs.dtype, cache.value_heads, "b")
        b = project_b(inputs)
        project_a = self.in_proj_a
        _projection_storage(project_a, tuple(inputs.shape), inputs.dtype, cache.value_heads, "a")
        a = project_a(inputs)
        _array(qkv, "qkv projection", (B, S, width))
        _array(z, "z projection", (B, S, cache.value_heads, cache.value_dim))
        _array(a, "a projection", (B, S, cache.value_heads))
        _array(b, "b projection", (B, S, cache.value_heads))

        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_dtype = mx.result_type(cache.conv.dtype, qkv.dtype)
        _storage((B, n_keep + S, width), conv_dtype.size, "convolution input")
        conv_input = mx.concatenate([cache.conv, qkv], axis=1)
        if cache.lengths is not None:
            ends = mx.clip(cache.lengths, 0, S)
            positions = (ends[:, None] + mx.arange(n_keep))[..., None]
            cache.conv = mx.take_along_axis(conv_input, positions, axis=1)
        else:
            cache.conv = mx.contiguous(conv_input[:, -n_keep:, :] if n_keep else conv_input[:, :0, :])
        convolve = self.conv1d
        _convolution_storage(convolve, (B, n_keep + S, width), conv_dtype, (B, S, width))
        conv_out = nn.silu(convolve(conv_input))
        _array(conv_out, "convolution output", (B, S, width))

        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        a_log, dt_bias = self.A_log, self.dt_bias
        _array(a_log, "A_log", (cache.value_heads,))
        _array(dt_bias, "dt_bias", (cache.value_heads,))
        _storage((B, S, cache.value_heads), mx.result_type(a.dtype, dt_bias.dtype).size, "gate addition")
        log_g, beta = _gates(a_log, dt_bias, a, b)
        if mask is not None:
            log_g = mx.where(mask[..., None], log_g, 0.0)
            beta = mx.where(mask[..., None], beta, 0.0)

        outs = [cache.step(q[:, t], k[:, t], v[:, t], log_g[:, t], beta[:, t]) for t in range(S)]
        out = outs[0][:, None] if S == 1 else mx.stack(outs, axis=1)
        if out.dtype != inputs.dtype:
            out = out.astype(inputs.dtype)
        cache.advance(S)
        out = self.norm(out, z)
        project_output = self.out_proj
        out = out.reshape(B, S, -1)
        _projection_storage(project_output, tuple(out.shape), out.dtype, int(inputs.shape[-1]), "output projection")
        return project_output(out)

    except BaseException:
        object.__setattr__(cache, "_failed", True)
        raise


def install_lane_gdn(model: Any) -> int:
    """Route gated-delta layers through ``lane_gdn_call`` when the cache is a ``LaneGDNCache``."""

    language_model = getattr(model, "language_model", model)
    core = getattr(language_model, "model", language_model)
    layers = [layer.linear_attn for layer in core.layers if getattr(layer, "is_linear", False)]
    for cls in {type(layer) for layer in layers}:
        if getattr(cls, "_tensorfold_lane_gdn", False):
            continue
        stock = cls.__call__

        def patched(self: Any, inputs: mx.array, mask: Any = None, cache: Any = None,
                    _stock: Any = stock) -> mx.array:
            if isinstance(cache, LaneGDNCache):
                return lane_gdn_call(self, inputs, mask, cache)
            return _stock(self, inputs, mask, cache)

        cls.__call__ = patched
        cls._tensorfold_lane_gdn = True
    return len(layers)


__all__ = ["LaneGDNCache", "install_lane_gdn", "lane_gdn_call"]
