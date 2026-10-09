"""Row-exact 2- to 8-bit lane matmul on the M5 tensor units: a group's fma order follows the weight shape, not M."""

from __future__ import annotations

from operator import index
from typing import Any

import mlx.core as mx

from tensorfold.kernels.inputs import ints
from tensorfold.kernels.qwen.dense.v1.lane_stage import SCALAR_PAIRS, staging_plan, tensor_source

MAX_ROWS = 128         # rows the lane kernel accepts in one call
ROW_BLOCK = 32         # rows per threadgroup above 32 rows (one 32-row op per weight group)
NT = 32                # output columns per simdgroup tile
BITS = (2, 3, 4, 5, 6, 8)   # weight widths the lane matmul takes (MLX affine; all but 4-bit in groups of 64)

_HEADER_TEMPLATE = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
"""
_HEADER = _HEADER_TEMPLATE + SCALAR_PAIRS

_XSUM = r"""
  const int M = mdims[0], MP = mdims[1];
  const uint m = thread_position_in_grid.y;
  const uint g = thread_position_in_grid.x;
  if (g >= K / GS || int(m) >= MP) return;
  float acc = 0.0f;
  if (int(m) < M) for (int i = 0; i < GS; i++) acc += float(X[m * K + g * GS + i]);
  XS[g * MP + m] = acc;
"""

_MAIN_TEMPLATE = r"""
  const ushort lane = thread_index_in_simdgroup;
  const ushort sg = simdgroup_index_in_threadgroup;     // K slice
  const short qid = lane >> 2;
  const short fm = (qid & 4) | ((lane >> 1) & 3);       // fragment row of this lane (and fm + 8)
  const short fn = ((qid & 2) | (lane & 1)) * 4;        // first of its four fragment columns
  const int M = mdims[0], MP = mdims[1];
  constexpr int KG = K / GS;
  constexpr int NF = NT / 16;
  const int n0 = threadgroup_position_in_grid.x * NT;
  const int rb = threadgroup_position_in_grid.y * 16 * TMR;   // first row of this threadgroup's row block
  const int g_begin = (sg * KG) / SK;
  const int g_end = ((sg + 1) * KG) / SK;

  // one op for all TMR 16-row blocks: each row gets the 16-row op's bits
  constexpr auto desc = matmul2d_descriptor(16 * TMR, NT, GS, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA((device bfloat*)X + (int64_t)rb * K, dextents<int32_t, 2>(K, M - rb));
  tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> tB((device uchar*)Wq, dextents<int32_t, 2>(K, N));

  float C[TMR][NF * 8];
  for (int t = 0; t < TMR; t++) for (int i = 0; i < NF * 8; i++) C[t][i] = 0.0f;
  const device uint4* sbv = (const device uint4*)SBt;   // (s, b) bf16 pairs, [g][n]
  bool colok[NF];
  for (int f = 0; f < NF; f++) colok[f] = n0 + f * 16 + fn < N;
  for (int g = g_begin; g < g_end; g++) {
    float s[NF][4], bb[NF][4];
    for (int f = 0; f < NF; f++) {
      const uint4 q = colok[f] ? sbv[(g * N + n0 + f * 16 + fn) / 4] : uint4(0);
      const vec<bfloat, 8> v = as_type<vec<bfloat, 8>>(q);
      for (int j = 0; j < 4; j++) { s[f][j] = float(v[2 * j]); bb[f][j] = float(v[2 * j + 1]); }
    }
    auto a = tA.slice(g * GS, 0);
    auto b = tB.slice(g * GS, n0);
    auto P = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
    op.run(a, b, P);
    for (int t = 0; t < TMR; t++) {
      const bool live = !EDGE || rb + t * 16 < MP;     // EDGE: the last 32-row block passes MP, where XS ends
      const float xs0 = live ? XS[g * MP + rb + t * 16 + fm] : 0.0f;
      const float xs1 = live ? XS[g * MP + rb + t * 16 + fm + 8] : 0.0f;
      for (int f = 0; f < NF; f++)
        for (int r = 0; r < 2; r++)
          for (int j = 0; j < 4; j++) {
            const int i = f * 8 + r * 4 + j;
            C[t][i] = fma(s[f][j], P[t * NF * 8 + i], fma(bb[f][j], r ? xs1 : xs0, C[t][i]));
          }
    }
  }
  // K slices are added in slice order, one 16-row block at a time
  threadgroup float part[(SK > 1 ? SK - 1 : 1) * NF * 8 * 32];
  for (int t = 0; t < TMR; t++) {
    if (SK > 1) {
      if (sg > 0) for (int i = 0; i < NF * 8; i++) part[((sg - 1) * NF * 8 + i) * 32 + lane] = C[t][i];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg == 0)
        for (int s2 = 1; s2 < SK; s2++) for (int i = 0; i < NF * 8; i++) C[t][i] += part[((s2 - 1) * NF * 8 + i) * 32 + lane];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sg == 0)
      for (int f = 0; f < NF; f++)
        for (int r = 0; r < 2; r++) {
          const int m = rb + t * 16 + fm + 8 * r;
          const int n = n0 + f * 16 + fn;
          if (m < M && n < N)
            for (int j = 0; j < 4; j++) Y[m * N + n + j] = static_cast<bfloat>(C[t][f * 8 + r * 4 + j]);
        }
  }
"""

# These literal templates retain the selected arithmetic. Only the legal
# compiler-owned stage variants below are compiled or exposed as sources.

# 64-wide tiles: two simdgroups run each group's 16 TMR x 64 op together; every output keeps _MAIN_TILED's arithmetic.
_COOP_TEMPLATE = r"""
  const ushort sg = simdgroup_index_in_threadgroup;
  const ushort slice = sg >> 1;                                  // K slice: a pair of simdgroups each
  const ushort tip = ushort(thread_position_in_threadgroup.x) - slice * 64;   // thread within its pair
  const int M = mdims[0], MP = mdims[1];
  constexpr int KG = K / GS;
  const int n0 = threadgroup_position_in_grid.x * 64;
  const int rb = threadgroup_position_in_grid.y * 16 * TMR;       // first row of this threadgroup's row block
  const int g_begin = (slice * KG) / SK;
  const int g_end = ((slice + 1) * KG) / SK;
  constexpr auto desc = matmul2d_descriptor(16 * TMR, 64, GS, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<2>> op;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA((device bfloat*)X + (int64_t)rb * K, dextents<int32_t, 2>(K, M - rb));
  auto a0 = tA.slice(0, 0);
  tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b0((device uchar*)Wq, dextents<int32_t, 2>(GS, 64));
  auto P = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
  constexpr int CAP = 16 * TMR;                                  // 16 TMR x 64 outputs over 64 threads
  short ecol[CAP], erow[CAP];
  for (int i = 0; i < CAP; i++) { auto ids = P.get_multidimensional_index(i); ecol[i] = ids[0]; erow[i] = ids[1]; }
  float C[CAP];
  for (int i = 0; i < CAP; i++) C[i] = 0.0f;
  const device uint* sbw = (const device uint*)SBt;              // (s, b) bf16 pairs, [g][n]
  for (int g = g_begin; g < g_end; g++) {
    auto a = tA.slice(g * GS, 0);
    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b(
        (device uchar*)Wq + (int64_t)(threadgroup_position_in_grid.x * KG + g) * (64 * GS / 2), dextents<int32_t, 2>(GS, 64));
    op.run(a, b, P);
    for (int i = 0; i < CAP; i++) {
      const vec<bfloat, 2> sb = as_type<vec<bfloat, 2>>(sbw[g * N + n0 + ecol[i]]);
      const float xs = !EDGE || rb + erow[i] < MP ? XS[g * MP + rb + erow[i]] : 0.0f;     // EDGE: see _MAIN
      C[i] = fma(float(sb[0]), P[i], fma(float(sb[1]), xs, C[i]));
    }
  }
  // K slices added in slice order, 16 outputs a thread at a time (the buffer stays within 28 KB at 8 slices)
  threadgroup float part[(SK > 1 ? SK - 1 : 1) * 16 * 64];
  if (SK > 1)
    for (int c0 = 0; c0 < CAP; c0 += 16) {
      if (slice > 0) for (int i = 0; i < 16; i++) part[((slice - 1) * 16 + i) * 64 + tip] = C[c0 + i];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (slice == 0)
        for (int s2 = 1; s2 < SK; s2++) for (int i = 0; i < 16; i++) C[c0 + i] += part[((s2 - 1) * 16 + i) * 64 + tip];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  if (slice == 0)
    for (int i = 0; i < CAP; i++) {
      const int m = rb + erow[i], n = n0 + ecol[i];
      if (m < M) Y[m * N + n] = static_cast<bfloat>(C[i]);
    }
"""
_MAIN = tensor_source(_MAIN_TEMPLATE)
_MAIN_TILED = tensor_source(_MAIN_TEMPLATE, tiled=True)
_COOP = tensor_source(_COOP_TEMPLATE, coop=True)
AB_FLAG = [False]                                    # The engine can switch kernel variants between rounds.

_kernels: dict[str, Any] = {}


def _named(base: str, source: str) -> str:
    """Kernel names carry a hash of their source: MLX caches compiled kernels by name."""

    import hashlib

    return f"{base}_{hashlib.sha256((_HEADER + source).encode()).hexdigest()[:16]}"


class _Baked:
    """Bake template integers into source to avoid MLX's per-call regex, caching each constant set under its source hash."""

    def __init__(self, base: str, body: str, inputs: list[str], outputs: list[str]) -> None:
        self.base, self.body, self.inputs, self.outputs = base, body, inputs, outputs
        self.compiled: dict[tuple, Any] = {}

    def __call__(self, *, template: Any = (), **kwargs: Any) -> Any:
        key = tuple(template)
        run = self.compiled.get(key)
        if run is None:
            source = "".join(f"  constexpr int {k} = {int(v)};\n" for k, v in key) + self.body
            run = self.compiled[key] = mx.fast.metal_kernel(name=_named(self.base, source), input_names=self.inputs,
                                                            output_names=self.outputs, source=source, header=_HEADER)
        return run(**kwargs)


def _kernel(name: str) -> Any:
    if name not in _kernels:
        if name == "xsum":
            _kernels[name] = _Baked("lane_qmm_xsum", _XSUM, ["X", "mdims"], ["XS"])
        elif name == "ordered_reduce":
            from tensorfold.kernels.qwen.dense.v1 import lane_stage

            _kernels[name] = _Baked("lane_qmm_ordered_reduce", lane_stage.ORDERED_REDUCE, ["PART"], ["Y"])
        else:
            from tensorfold.kernels.qwen.dense.v1 import lane_stage, lane_widen

            partials = name.endswith("_partials")
            base = name.removesuffix("_partials")
            source = {"coop": _COOP_TEMPLATE, "main_tiled": _MAIN_TEMPLATE, "main": _MAIN_TEMPLATE,
                      "lowbit": lane_widen._NIBBLES_TEMPLATE,
                      "bytes": lane_widen._BYTES_TEMPLATE, "lowbit_grouped": lane_widen._NIBBLES_GROUPED_TEMPLATE,
                      "bytes_grouped": lane_widen._BYTES_GROUPED_TEMPLATE}[base]
            if base in ("coop", "main_tiled", "main", "lowbit", "lowbit_grouped"):
                source = lane_stage.tensor_source(source, coop=base == "coop", tiled=base == "main_tiled",
                                                  nibbles=base.startswith("lowbit"), grouped=base.endswith("grouped"),
                                                  partials=partials)
            else:
                source = lane_stage.scalar_scale_source(source)
            _kernels[name] = _Baked("lane_qmm_" + name, source, ["X", "XS", "Wq", "SBt", "mdims"],
                                    ["PART"] if partials else ["Y"])
    return _kernels[name]


_mdims_cache: dict[tuple[int, int], mx.array] = {}


def _mdims(m: int, mp: int) -> mx.array:
    key = (m, mp)
    if key not in _mdims_cache:
        _mdims_cache[key] = ints((m, mp))
    return _mdims_cache[key]


def split_k(n: int, k: int) -> int:
    """K slices for an (n, k) weight: fixed by the shape, never by the row count."""

    tiles = -(-n // NT)
    sk = 1
    while sk < 8 and tiles * sk < 1024 and (k // 64) // (sk * 2) >= 8:
        sk *= 2
    return sk


def pack_scales(scales: mx.array, biases: mx.array) -> mx.array:
    """(N, K/GS) scales and biases -> (K/GS, N, 2) bf16 pairs, group-major."""

    return mx.stack([scales.T, biases.T], axis=-1).astype(mx.bfloat16)


def tile_weight(weight: mx.array, nt: int = NT, group: int = 64, *, bits: int) -> mx.array:
    """MLX's packed (N, K*bits/32) weight regrouped [N/nt][K/group][nt columns x a group's words], same bytes."""
    # bits must be the weight's: a 3-bit weight's shape can be a 4-bit one's of another K

    n, kw, w = int(weight.shape[0]), int(weight.shape[1]), group * bits // 32
    return mx.contiguous(weight.reshape(n // nt, nt, kw // w, w).transpose(0, 2, 1, 3).reshape(n, kw))


def untile_weight(weight: mx.array, nt: int = NT, group: int = 64, *, bits: int) -> mx.array:
    """``tile_weight`` undone: MLX's packed layout again."""

    n, kw, w = int(weight.shape[0]), int(weight.shape[1]), group * bits // 32
    return mx.contiguous(weight.reshape(n // nt, kw // w, nt, w).transpose(0, 2, 1, 3).reshape(n, kw))


def weight_bits(weight: mx.array, k: int) -> int:
    """The bit width of a packed ``weight`` for K inputs (words a row = K * bits / 32)."""

    words = int(weight.shape[1])
    if k <= 0 or (words * 32) % k:
        raise ValueError(f"a packed weight of {words} words a row does not fit K = {k}")
    return words * 32 // k


def readable(bits: int, group_size: int, mode: str = "affine") -> bool:
    """Whether the lane matmul reads MLX weights of this width, group size and mode."""

    return mode == "affine" and bits in BITS and group_size in ((32, 64) if bits == 4 else (64,))


def reads(bits: int, group_size: int) -> bool:
    """Whether lane_matmul's kernels take this width and group (every width in groups of 32 or 64): a family opts in."""

    return bits in BITS and group_size in (32, 64)


def supports(weight: mx.array, scales: mx.array, x: mx.array, bits: int, group_size: int, mode: str) -> bool:
    if not readable(bits, group_size, mode):
        return False
    if x.ndim < 1 or x.dtype != mx.bfloat16 or scales.dtype != mx.bfloat16 or weight.dtype != mx.uint32 or weight.ndim != 2:
        return False
    k = int(x.shape[-1])
    n = int(weight.shape[0])
    return k > 0 and n > 0 and k % 64 == 0 and int(weight.shape[1]) * 32 == k * bits and n % 4 == 0


def _admit_launch(mp: int, n: int, k: int, group: int, nt: int, sk: int, bits: int) -> None:
    """Prove shader integer arithmetic and target upper limits before native work.

    Apple Metal4 has at most 1024 threads and 32 KiB threadgroup storage.
    These are ceilings, not pipeline admission: MLX checks each compiled
    pipeline's actual maxTotalThreadsPerThreadgroup before dispatch, and Metal
    compilation admits its actual static threadgroup storage. No slice count
    is reduced to make a launch fit because that would change its arithmetic.
    """
    i32, u32 = (1 << 31) - 1, (1 << 32) - 1
    kg = k // group
    threads = (64 if bits == 4 and nt == 64 else 32) * sk
    tiles = -(-n // nt)
    if (max(k, n, nt, tiles * threads) > i32 or mp * n > i32
            or kg * n > i32 or sk * kg > i32 or mp * k > u32
            or nt * group > i32 or (bits != 4 and k * bits > i32)):
        raise ValueError("lane_matmul geometry exceeds native signed arithmetic or grid representation")
    if threads > 1024:
        raise ValueError("lane_matmul threadgroup exceeds the Metal4 thread ceiling")
    # Packed format tensors use compiler-owned aligned cohorts. If one cohort
    # cannot coexist with the original partials, float32 partials live in an
    # operation-owned device output; selected slices and addition order remain.
    if bits <= 4:
        staging_plan(nt, sk, coop=bits == 4 and nt == 64)
        return
    # Ordinary byte tensors do not have the packed format alignment contract.
    partial = ((sk - 1) * 16 * 64 * 4 if bits == 4 and nt == 64
               else (sk - 1) * (nt // 16) * 8 * 32 * 4) if sk > 1 else 0
    stage = sk * nt * group
    if partial + stage > 32768:
        raise ValueError("lane_matmul requires more than the Metal4 threadgroup storage ceiling")


def lane_matmul(x: mx.array, weight: mx.array, sbt: mx.array, *, tiled: bool = False,
                sk: int | None = None, nt: int = NT, group: int = 64) -> mx.array:
    """x (..., K) bf16 times the packed ``weight`` (N, K*bits/32) transposed, rows <= MAX_ROWS, tiled or not."""

    if not isinstance(tiled, bool):
        raise ValueError("lane_matmul tiled policy must be boolean")
    if (not isinstance(x, mx.array) or x.ndim < 1 or x.dtype != mx.bfloat16
            or not isinstance(weight, mx.array) or weight.ndim != 2 or weight.dtype != mx.uint32
            or not isinstance(sbt, mx.array) or sbt.dtype != mx.bfloat16):
        raise ValueError("lane_matmul requires bf16 inputs/scales and a rank-two uint32 packed weight")
    try:
        if isinstance(group, bool) or isinstance(nt, bool) or isinstance(sk, bool):
            raise TypeError("boolean layout parameter")
        group, nt = index(group), index(nt)
        sk = index(sk) if sk is not None else None
    except TypeError as error:
        raise ValueError("lane_matmul layout parameters must be integer counts") from error
    if group not in (32, 64) or nt < 16 or nt % 16 or (sk is not None and sk < 0):
        raise ValueError("lane_matmul requires groups32/64, positive whole SIMD tiles and nonnegative slices")
    K = int(x.shape[-1])
    N = int(weight.shape[0])
    if K <= 0 or K % group or N <= 0 or N % 4:
        raise ValueError("lane_matmul requires positive grouped inputs and whole four-column output vectors")
    if tuple(sbt.shape) != (K // group, N, 2):
        raise ValueError("lane_matmul scale/bias pairs do not match its current weight and group geometry")
    lead = x.shape[:-1]
    M = 1
    for dimension in lead:
        M *= int(dimension)
    if not 1 <= M <= MAX_ROWS:
        raise ValueError(f"lane_matmul takes at most {MAX_ROWS} rows, got {M}")
    bits = weight_bits(weight, K)
    if not reads(bits, group):
        raise ValueError(f"lane_matmul takes {'/'.join(map(str, BITS))}-bit weights in groups of 32 or 64, "
                         f"got {bits}-bit in groups of {group}")
    if bits != 4:
        if K % 64 or N % 4:
            raise ValueError(f"{bits}-bit weights need K a multiple of 64 and N a multiple of 4, got K={K}, N={N}")
        if tiled and int(nt) != NT:
            raise ValueError(f"{bits}-bit weights tile {NT} columns wide, got {nt}")
        if sk and int(sk) > 8:    # threadgroup memory: the stage and partial sums of each slice
            raise ValueError(f"{bits}-bit weights take at most 8 K slices (split_k's largest), got {sk}")
    nt = int(nt) if tiled else NT
    if tiled and N % nt:
        raise ValueError(f"tiled weights need N to be a multiple of {nt}, got {N}")
    sk = int(sk) if sk else split_k(N, K)       # a column's bits follow K and sk (lane_fuse's stacks)
    MP = 16 * ((M + 15) // 16)
    KG = K // group
    _admit_launch(MP, N, K, group, nt, sk, bits)
    if (MP * N > (1 << 31) - 1 or MP * K > (1 << 32) - 1
            or KG * N > (1 << 32) - 1 or N * int(weight.shape[1]) > (1 << 32) - 1):
        raise ValueError("lane_matmul geometry exceeds its native index representation")
    stages, partials = 0, False
    if bits <= 4:
        stages, partials, _ = staging_plan(nt, sk, coop=bits == 4 and nt == 64)
        if partials and sk * M * N > ((1 << 64) - 1) // 4:
            raise ValueError("lane_matmul float32 partial allocation exceeds native size_t")
    x2 = x.reshape(M, K)
    mdims = _mdims(M, MP)
    from tensorfold.kernels.qwen.dense.v1 import projection_operation

    # Only a private forward operation may reuse sums of its immutable input.
    # Generic array calls always derive sums from the current descriptor.
    xs = projection_operation.sums_of(x, group)
    if xs is None:
        xs = _kernel("xsum")(inputs=[x2, mdims], template=[("K", K), ("GS", group)], grid=(KG, MP, 1),
                             threadgroup=(min(KG, 256), 1, 1), output_shapes=[(KG, MP)],
                             output_dtypes=[mx.float32])[0]
        projection_operation.remember(x, xs, group)
    block = MP if MP <= ROW_BLOCK else ROW_BLOCK
    edge = int(MP % block != 0)     # a bound check only where the last block passes MP (33-48, 65-80, 97-112 rows)
    output_shapes = [(sk, M, N)] if partials else [(M, N)]
    output_dtypes = [mx.float32] if partials else [mx.bfloat16]
    suffix = "_partials" if partials else ""
    stage_template = [("STAGES", stages)] if bits <= 4 else []
    if bits != 4:
        grouped = [("GS", group)] if group != 64 else []    # groups of 64 keep the original kernels' source
        y = _kernel(("lowbit" if bits < 4 else "bytes") + ("_grouped" if grouped else "") + suffix)(
                              inputs=[x2, xs, weight, sbt, mdims],
                              template=[("TMR", block // 16), ("N", N), ("K", K), ("NT", NT), ("SK", sk),
                                        ("BITS", bits), ("TILED", int(bool(tiled))), *grouped, *stage_template],
                              grid=(-(-N // NT) * 32 * sk, -(-MP // block), 1), threadgroup=(32 * sk, 1, 1),
                              output_shapes=output_shapes, output_dtypes=output_dtypes)[0]
    elif nt == 64:
        y = _kernel("coop" + suffix)(inputs=[x2, xs, weight, sbt, mdims],
                            template=[("TMR", block // 16), ("N", N), ("K", K), ("SK", sk), ("GS", group),
                                      ("EDGE", edge), *stage_template],
                            grid=((N // 64) * 64 * sk, -(-MP // block), 1), threadgroup=(64 * sk, 1, 1),
                            output_shapes=output_shapes, output_dtypes=output_dtypes)[0]
    else:
        y = _kernel(("main_tiled" if tiled else "main") + suffix)(inputs=[x2, xs, weight, sbt, mdims],
                        template=[("TMR", block // 16), ("N", N), ("K", K), ("NT", nt), ("SK", sk), ("GS", group),
                                  ("EDGE", edge), *stage_template],
                        grid=(-(-N // nt) * 32 * sk, -(-MP // block), 1), threadgroup=(32 * sk, 1, 1),
                        output_shapes=output_shapes, output_dtypes=output_dtypes)[0]
    if partials:
        y = _kernel("ordered_reduce")(inputs=[y], template=[("M", M), ("N", N), ("SK", sk)],
                                       grid=(M * N, 1, 1), threadgroup=(256, 1, 1),
                                       output_shapes=[(M, N)], output_dtypes=[mx.bfloat16])[0]
    return y.reshape(*lead, N)


# -- routing the model's projections --------------------------------------------------------
_ORIG: Any = None
enabled = False
max_rows = MAX_ROWS


def _layout(module: Any) -> tuple[mx.array, bool, int]:
    """Derive a call's layout from current standard MLX parameters and install policy.

    Public module arrays remain authoritative and may be replaced or overwrite
    their MLX descriptor between calls. No derived weight or scales are retained.
    The synchronous caller must not mutate parameters while this call borrows them.
    """

    weight = module["weight"]
    nt = int(getattr(module, "_lane_nt", NT))
    if module.bits != 4 or int(weight.shape[0]) % nt:
        nt = NT
    words = int(weight.shape[-1])
    tiled = (bool(getattr(module, "_lane_tile", False)) and weight.dtype == mx.uint32
             and weight.ndim == 2 and int(weight.shape[0]) % nt == 0
             and words % (module.group_size * module.bits // 32) == 0)
    if tiled:
        weight = tile_weight(weight, nt, module.group_size, bits=module.bits)
    return weight, tiled, nt


def _call(self: Any, x: mx.array) -> mx.array:
    rows = 1
    for d in x.shape[:-1]:
        rows *= int(d)
    if not (enabled and 1 <= rows <= max_rows and supports(self["weight"], self["scales"], x, self.bits,
                                                     self.group_size, getattr(self, "mode", "affine"))):
        return _ORIG(self, x)
    shape = (int(self["weight"].shape[0]), int(x.shape[-1]) // self.group_size)
    if (tuple(self["scales"].shape) != shape or tuple(self["biases"].shape) != shape
            or self["biases"].dtype != mx.bfloat16):
        return _ORIG(self, x)
    # The generic SDK path remains valid outside the raw native index region.
    # Installed NT32/64 policies use split_k<=8; both satisfy the same ceilings.
    n, k = shape[0], int(x.shape[-1])
    if n * int(self["weight"].shape[1]) > (1 << 32) - 1:
        return _ORIG(self, x)
    try:
        _admit_launch(16 * ((rows + 15) // 16), n, k, self.group_size, NT, split_k(n, k), self.bits)
    except ValueError:
        return _ORIG(self, x)
    weight, tiled, nt = _layout(self)
    sbt = pack_scales(self["scales"], self["biases"])
    y = lane_matmul(x, weight, sbt, tiled=tiled, nt=nt, group=self.group_size)
    if "bias" in self:
        y = y + self["bias"]
    return y


# Kept 32 columns wide under ``wide``: in_proj_z stacks with in_proj_b/a (48 rows each), which only 32-column tiles divide
NARROW = ("in_proj_z",)


def takes(module: Any) -> bool:
    """Whether the lane matmul takes a QuantizedLinear: a width and group size it reads, bf16 scales."""

    return readable(module.bits, module.group_size, getattr(module, "mode", "affine")) \
        and module["scales"].dtype == mx.bfloat16


def install(model: Any = None, *, rows: int = MAX_ROWS, tile: bool = True, wide: bool = False) -> None:
    """Route supported calls; preselect tiling policy without changing public weights.

    Parameter replacement and same-object descriptor updates remain legal between
    calls. Layout/scales materialization is operation-local; ``warm`` compiles the
    selected variants. No model parameter is stored in a second retained layout.
    """

    global _ORIG, enabled, max_rows
    import mlx.nn as nn

    if _ORIG is None:
        _ORIG = nn.QuantizedLinear.__call__
    nn.QuantizedLinear.__call__ = _call
    enabled = True
    max_rows = min(int(rows), MAX_ROWS)
    if model is not None:
        for name, module in model.named_modules():
            if not (isinstance(module, nn.QuantizedLinear) and takes(module)):
                continue
            n = int(module["weight"].shape[0])
            nt = 64 if (wide and module.bits == 4 and n % 64 == 0 and not name.endswith(NARROW)) else NT
            object.__setattr__(module, "_lane_tile", bool(tile))
            object.__setattr__(module, "_lane_nt", nt)


def uncovered(model: Any) -> dict[str, int]:
    """The model's linear layers and head the lane matmul does not take, by kind ({"6-bit g32": 17, ...})."""

    import mlx.nn as nn

    counts: dict[str, int] = {}
    for _, module in model.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            if takes(module):
                continue
            mode = getattr(module, "mode", "affine")
            kind = f"{module.bits}-bit g{module.group_size}" + ("" if mode == "affine" else f" {mode}")
            kind += "" if module["scales"].dtype == mx.bfloat16 else f" {module['scales'].dtype} scales"
        elif isinstance(module, nn.Linear):
            kind = "unquantized"
        else:
            continue
        counts[kind] = counts.get(kind, 0) + 1
    language_model = getattr(model, "language_model", model)
    if getattr(getattr(language_model, "args", None), "tie_word_embeddings", False):
        counts["tied embedding head"] = 1        # embed_tokens.as_linear: MLX's kernel, not a QuantizedLinear
    return counts


def warm(model: Any, *, rows: tuple[int, ...] = (1, 17, 33)) -> int:
    """Compile every kernel variant the model's projections will use (one per shape and row tile)."""

    import mlx.nn as nn

    seen: set[tuple[int, ...]] = set()
    outs = []
    for _, module in model.named_modules():
        if not isinstance(module, nn.QuantizedLinear) or not takes(module):
            continue
        n, k = int(module["weight"].shape[0]), int(module["weight"].shape[1]) * 32 // module.bits
        weight, tiled, nt = _layout(module)
        sbt = pack_scales(module["scales"], module["biases"])
        key = (n, k, module.bits, module.group_size, tiled, nt)
        if key in seen:
            continue
        seen.add(key)
        for m in rows:
            outs.append(lane_matmul(mx.zeros((m, k), dtype=mx.bfloat16), weight, sbt,
                                    tiled=tiled, nt=nt,
                                    group=module.group_size))
    mx.eval(outs)
    return len(seen)


def uninstall() -> None:
    """Restore MLX's calls; public parameters already use its layout."""

    global enabled
    import mlx.nn as nn

    enabled = False
    if _ORIG is not None:
        nn.QuantizedLinear.__call__ = _ORIG


__all__ = ["BITS", "MAX_ROWS", "install", "lane_matmul", "pack_scales", "readable", "reads", "split_k", "supports", "takes",
           "tile_weight", "uncovered", "uninstall", "untile_weight", "warm", "weight_bits"]
