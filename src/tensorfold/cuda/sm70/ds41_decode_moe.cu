// SPDX-License-Identifier: Apache-2.0
// DeepSeek-V4.1 routed-expert decode MoE for SM70, 1..64 rows: expert-major integer-dot kernels.
//
// Provenance: the friend's 1Cat-vLLM overlay patch 240 (ds41_mu_em3, EM3, which extends patch 110 EM2),
// payload patches/dmoe_files/ds41mu_em3.cu of snapshot lp-snap-c1es8-...-pfxfix4 (catalogue
// friend-1cat-20261120/DS41_CHANGES.md s.2.1: 110 measured c1 +6 %, c8 +13 %, c16 +23 %, c32 +35 % on his
// PP2 x TP4 V100 container). Kernels, variants and tables are his, unchanged in arithmetic.
//
// Contract (his, kept): every kept route's arithmetic is EXACTLY the 6-row path's (onecat_ds41.cu = 1Cat
// gguf_dp4a_sm70.cu at 4d57473d: same integer words, same FP32 order, same FP16 / Q8_1 boundaries), so each
// output row equals onecat_ds41.v41_experts' bit for bit at every M (rows are independent: batch invariant).
// Only where the V100 cycles go changes: one plan groups the M x 6 routes by expert into units of <= chunk
// routes; each unit decodes its expert's rows once for all its routes (codebooks in shared memory, replicated
// per lane where it fits: R; window activation layout: L; 0.125 folded into the activation scale: S, exact in
// FP32's normal range; IQ1_M folded 4096-entry table: F; persistent CTAs: P). See his notes kept below.
//
// TensorFold changes (ETP4 x EP2 carriers: 576 gate/up rows, 768-column down with left 0/64/128/192, any
// E <= 1024 local experts; Target.moe passes non-local routes as id 0 with probability 0):
//  * routes with probability exactly 0 are dropped by the plan (no unit), skipped by the non-fused SwiGLU
//    and by combine. In the 6-row path such a route contributes exactly +-0 to a sum that starts at +0.f
//    (clamped SwiGLU x 0 -> +-0, a zero Q8_1 block, a zero dot), and +0.f + (+-0) and x + (+-0) (x != 0)
//    round to the same value, so dropping is bit-exact; it halves the routed work under EP2. A dropped
//    route's id is not checked (a kept route's id outside [0, E) traps the device, as before);
//  * the packaged-quantizer option is removed (prep's quantizer is quantize_q8_1_warp's arithmetic);
//  * one host op runs a whole layer on a caller-owned fixed workspace (workspace_bytes for 64 rows): no
//    allocation, no host synchronization, launch grids fixed by M (units are counted on the device) ->
//    capturable in CUDA graphs; the persistent grids query the occupancy once per process (one device type);
//  * Q8_1 / quantize_q8_1_warp / sign_mask / the codebooks come from onecat_ds41_dp4a.cuh (the same
//    upstream text); the non-fused gate/up variants use v41_swiglu_q8's arithmetic over all M x 6 routes.
//
// His notes (patch 240):
// EM2 runs at ~1.9x the DRAM byte floor on the V100 without reading extra
// bytes. The time goes to the shared L1/shared-memory pipe and the integer
// pipe:
//   * IQ2_XXS codebook + sign lookups (random 8-byte LDS from one copy: ~6
//     wavefronts instead of 2): gate/up -34%, down -28% when conflict-free;
//   * activation loads: gate/up -17% (IQ2_XXS) / -21% (IQ1_M);
//   * IQ1_M: lookups -8%; the rest is integer work (the per-octet bias select,
//     two adds and two xors around each lookup).
//   R  replicated IQ2 tables: each lane reads its own copy (lane & (copies-1);
//      entry e of copy c at e * copies + c): IQ2_XXS grid 16 copies (no bank
//      conflicts), IQ2_XS grid 8, sign masks 8 (down: as many as fit 46 KB);
//   L  window activation layout: per 16 groups, the low 16 bytes of every group
//      (256 B) then the high 16 bytes, so a warp's LDG.128 of 16 consecutive
//      groups reads 2 full 128-byte lines instead of 4 half-used ones;
//   S  0.125 folded into the activation scale: xs = xd / 8 (exact) and
//      (d * xs) * X == ((d * xd) * X) * 0.125 bit for bit, because every
//      product stays in FP32's normal range (|d|, |xd| >= 2^-24 or 0; |X| <=
//      2^20): scaling by 2^-3 commutes with round-to-nearest there;
//   F  IQ1_M folded table: 4096 entries indexed by (qs byte | qh nibble << 8)
//      holding EM2's (book + bias) ^ 0x80808080 words (32 KB shared memory):
//      one PRMT pair builds the four indices; no select / add / xor per octet;
//   P  persistent CTAs (SMs x resident CTAs, a fixed grid for CUDA graphs)
//      looping over units x row/column blocks: the tables are copied into
//      shared memory once per CTA instead of once per (unit, block).
// Variants: gate/up 0-9 = EM2 shapes + R L S; 10-19 = EM2 shapes + L S (one
// table copy); 20-23 = shapes {4, 8, 9, 0} + R L S F; 24-27 = same + P.
// Down 0-11 = EM2 shapes + R S; 12-23 = + S; 24-29 = shapes {0, 1, 2, 3, 4,
// 10} + R S F; 30-35 = shapes {0, 4, 5, 10, 11, 7} + R S F P (F where it fits
// 46 KB). The schedule (ds41_decode_moe.py) picks per type and M.
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>
#include "onecat_ds41_dp4a.cuh"

namespace tensorfold::ds41_decode_moe {
using tensorfold::onecat_ds41::Q8_1;
using tensorfold::onecat_ds41::quantize_q8_1_warp;
using tensorfold::onecat_ds41::sign_mask;

constexpr int kTopK = 6;
constexpr int kMaxRoutes = 64 * kTopK;  // 384
constexpr int kMaxExperts = 1024;

template <int Type>
constexpr int block_bytes() {
  return Type == 16 ? 66 : Type == 17 ? 74 : 56;
}
// uint2 book entries (two signed int8x4 words per codebook index).
template <int Type>
constexpr int book_entries() {
  return turbomind::gemm::LatticeCodebook<Type>::kBytes / 8;
}

// ---- tables ---------------------------------------------------------------
// IQ2: entry i = {word(2i), word(2i+1)} ^ 0x80808080 (V41LatticeDot's book).
// IQ1_M: the book after its x8 (three __vadd4(w, w)), stored offset-binary
// (xor 0x80808080) so the per-octet +-1 delta is one 32-bit add.
template <int Type>
__global__ void build_book(uint2* out) {
  using Book = turbomind::gemm::LatticeCodebook<Type>;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= book_entries<Type>()) return;
  uint32_t w[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    uint32_t word = Book::word(2 * i + h) ^ 0x80808080U;
    if constexpr (Type == 29) {
      word = __vadd4(word, word);
      word = __vadd4(word, word);
      word = __vadd4(word, word);
      word ^= 0x80808080U;
    }
    w[h] = word;
  }
  out[i] = make_uint2(w[0], w[1]);
}

// signs[s7] = {mask(low nibble), mask(high nibble)} of s8 = s7 | parity << 7,
// exactly the masks[] lookups of V41IntegerGroup.
__global__ void build_signs(uint2* out) {
  const int i = threadIdx.x;
  if (i >= 128) return;
  const int s = i | ((__popc(i) & 1) << 7);
  out[i] = make_uint2(sign_mask(s & 15), sign_mask(s >> 4));
}

// ---- one 32-weight group ----------------------------------------------------
__device__ __forceinline__ void ld8(const uint8_t* p, uint32_t& w0, uint32_t& w1) {
  // 8 bytes at a 2-byte-aligned address: three aligned words + funnel shifts.
  const uintptr_t a = reinterpret_cast<uintptr_t>(p);
  const uint32_t* q = reinterpret_cast<const uint32_t*>(a & ~uintptr_t(3));
  const uint32_t sh = uint32_t(a & 3) * 8;
  const uint32_t x0 = __ldg(q), x1 = __ldg(q + 1), x2 = __ldg(q + 2);
  w0 = __funnelshift_r(x0, x1, sh);
  w1 = __funnelshift_r(x1, x2, sh);
}
__device__ __forceinline__ uint32_t ld16(const uint8_t* p) {
  return __ldg(reinterpret_cast<const unsigned short*>(p));
}
__device__ __forceinline__ uint32_t apply_sign(uint32_t magnitude, uint32_t mask) {
  // == __vsub4(magnitude ^ mask, mask) for magnitude bytes in 1..127.
  return (magnitude ^ mask) + (mask & 0x01010101U);
}

// ---- one 32-weight group -----------------------------------------------------
// book / signs: this lane's copy (base + (lane & (copies - 1))); entry e at [e * copies].
// F (IQ1_M): book is the 4096-entry folded table (entry = final words).
template <int Type, int RG = 1, int SR = 1, int F = 0>
struct Group {
  uint32_t w[8];
  float d;
  int m0, m1;

  __device__ __forceinline__ Group(const uint8_t* row, int group, const uint2* book,
                                   const uint2* signs) {
    const uint8_t* b = row + (group >> 3) * block_bytes<Type>();
    const int sub = group & 7;
    if constexpr (Type == 29) {
      const uint32_t qs = __ldg(reinterpret_cast<const uint32_t*>(b + 4 * sub));
      const uint32_t qh = ld16(b + 32 + 2 * sub);
      const uint2 sc = __ldg(reinterpret_cast<const uint2*>(b + 48));
      if constexpr (F) {
        // index(octet) = qs byte | qh nibble << 8: bytes (qs0, n0, qs2, n2), (qs1, n1, qs3, n3).
        const uint32_t a = qh & 0x0F0FU, c = (qh >> 4) & 0x0F0FU;
        const uint32_t p02 = __byte_perm(qs, a, 0x5240), p13 = __byte_perm(qs, c, 0x5341);
        const uint2 e0 = book[p02 & 0xffff], e1 = book[p13 & 0xffff];
        const uint2 e2 = book[p02 >> 16], e3 = book[p13 >> 16];
        w[0] = e0.x, w[1] = e0.y, w[2] = e1.x, w[3] = e1.y;
        w[4] = e2.x, w[5] = e2.y, w[6] = e3.x, w[7] = e3.y;
      } else {
#pragma unroll
        for (int octet = 0; octet < 4; ++octet) {
          const int high = (qh >> (4 * octet)) & 15;
          const int index = ((qs >> (8 * octet)) & 255) | ((high & 7) << 8);
          const uint32_t bias = (high >> 3) ? 0xFEFEFEFFU : 0x01010101U;  // -1 / +1 per byte
          const uint2 e = book[index];
          w[2 * octet] = (e.x + bias) ^ 0x80808080U;
          w[2 * octet + 1] = (e.y + bias) ^ 0x80808080U;
        }
      }
      const uint32_t s0 = sc.x & 0xffff, s1 = sc.x >> 16, s2 = sc.y & 0xffff, s3 = sc.y >> 16;
      const uint16_t bits = uint16_t((s0 >> 12) | ((s1 >> 8) & 0xf0) | ((s2 >> 4) & 0xf00) |
                                     (s3 & 0xf000));
      d = __half2float(__ushort_as_half(bits));
      const uint32_t s = (sub >> 1) == 0 ? s0 : (sub >> 1) == 1 ? s1 : (sub >> 1) == 2 ? s2 : s3;
      const int shift = 6 * (sub & 1);
      m0 = 2 * int((s >> shift) & 7) + 1;
      m1 = 2 * int((s >> (shift + 3)) & 7) + 1;
    } else {
      uint32_t w0, w1;
      ld8(b + 2 + 8 * sub, w0, w1);
      d = __half2float(__ushort_as_half(uint16_t(ld16(b))));
      if constexpr (Type == 16) {
#pragma unroll
        for (int octet = 0; octet < 4; ++octet) {
          const uint2 e = book[((w0 >> (8 * octet)) & 255) * RG];
          const uint2 s = signs[((w1 >> (7 * octet)) & 127) * SR];
          w[2 * octet] = apply_sign(e.x, s.x);
          w[2 * octet + 1] = apply_sign(e.y, s.y);
        }
        m0 = m1 = 1 + 2 * int(w1 >> 28);
      } else {
#pragma unroll
        for (int octet = 0; octet < 4; ++octet) {
          const uint32_t code = ((octet < 2 ? w0 : w1) >> (16 * (octet & 1))) & 0xffff;
          const uint2 e = book[(code & 511) * RG];
          const uint2 s = signs[(code >> 9) * SR];
          w[2 * octet] = apply_sign(e.x, s.x);
          w[2 * octet + 1] = apply_sign(e.y, s.y);
        }
        const int sc = __ldg(b + 66 + sub);
        m0 = 1 + 2 * (sc & 15);
        m1 = 1 + 2 * (sc >> 4);
      }
    }
  }

  // EM2's finish (packaged Q8_1 down input: scale xd, 0.125 at the end).
  __device__ __forceinline__ float finish(int s0, int s1, float xd) const {
    const float scale = __fmul_rn(d, xd);
    return __fmul_rn(__fmul_rn(scale, float(s0 * m0 + s1 * m1)), 0.125f);
  }
  // Aligned activation with the folded scale xs = xd / 8 (S in the header).
  __device__ __forceinline__ float apply(const int4& lo, const int4& hi, float xs) const {
    int s0 = 0, s1 = 0;
    s0 = __dp4a(int(w[0]), lo.x, s0);
    s0 = __dp4a(int(w[1]), lo.y, s0);
    s0 = __dp4a(int(w[2]), lo.z, s0);
    s0 = __dp4a(int(w[3]), lo.w, s0);
    s1 = __dp4a(int(w[4]), hi.x, s1);
    s1 = __dp4a(int(w[5]), hi.y, s1);
    s1 = __dp4a(int(w[6]), hi.z, s1);
    s1 = __dp4a(int(w[7]), hi.w, s1);
    return __fmul_rn(__fmul_rn(d, xs), float(s0 * m0 + s1 * m1));
  }
  // Packaged Q8_1 (36 B).
  __device__ __forceinline__ float apply(const Q8_1& x) const {
    const int* a = reinterpret_cast<const int*>(x.qs);
    int s0 = 0, s1 = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) s0 = __dp4a(int(w[i]), a[i], s0);
#pragma unroll
    for (int i = 4; i < 8; ++i) s1 = __dp4a(int(w[i]), a[i], s1);
    return finish(s0, s1, __low2float(x.ds));
  }
};

// v41_swiglu, verbatim (gguf_dp4a_sm70.cu).
__device__ __forceinline__ float v41_swiglu(float gate, float up, float probability) {
  gate = gate > 10.f ? 10.f : gate;
  up = up > 10.f ? 10.f : up;
  up = up < -10.f ? -10.f : up;
  const float activated = __fdiv_rn(gate, __fadd_rn(1.f, expf(-gate)));
  return __fmul_rn(__fmul_rn(activated, up), probability);
}

// ---- prep: plan (block 0) + Q8_1 of x (blocks 1..) ---------------------------
// units[0][u] = expert, units[1][u] = first index into order, units[2][u] =
// routes (<= chunk); count[0] = number of units. Experts ascend; the order of
// routes inside an expert is irrelevant to the results (per-route arithmetic).
// Routes with probability 0 get no unit (TensorFold: see the header).
__device__ void plan_block(const int32_t* ids, const float* probabilities, int routes, int experts,
                           int chunk, int32_t* units, int32_t* count, int32_t* order) {
  __shared__ int cnt[kMaxExperts], estart[kMaxExperts];
  __shared__ int warp_a[16], warp_b[16];
  const int t = threadIdx.x, lane = t % 32, warp = t / 32;
  for (int e = t; e < kMaxExperts; e += blockDim.x) cnt[e] = 0;
  __syncthreads();
  int e = -1, slot = 0;
  if (t < routes && probabilities[t] != 0.f) {
    e = ids[t];
    if (e < 0 || e >= experts) asm volatile("trap;");
    slot = atomicAdd(&cnt[e], 1);
  }
  __syncthreads();
  const int e0 = 2 * t, e1 = 2 * t + 1;
  const int n0 = cnt[e0], n1 = cnt[e1];
  const int u0 = (n0 + chunk - 1) / chunk, u1 = (n1 + chunk - 1) / chunk;
  int a = n0 + n1, b = u0 + u1;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const int ya = __shfl_up_sync(0xffffffffU, a, offset);
    const int yb = __shfl_up_sync(0xffffffffU, b, offset);
    if (lane >= offset) a += ya, b += yb;
  }
  if (lane == 31) warp_a[warp] = a, warp_b[warp] = b;
  __syncthreads();
  int pa = 0, pb = 0;
  for (int w = 0; w < warp; ++w) pa += warp_a[w], pb += warp_b[w];
  const int ea = pa + a - (n0 + n1), eb = pb + b - (u0 + u1);  // exclusive
  estart[e0] = ea;
  estart[e1] = ea + n0;
  for (int k = 0; k < u0; ++k) {
    units[eb + k] = e0;
    units[kMaxRoutes + eb + k] = ea + k * chunk;
    units[2 * kMaxRoutes + eb + k] = min(chunk, n0 - k * chunk);
  }
  for (int k = 0; k < u1; ++k) {
    units[eb + u0 + k] = e1;
    units[kMaxRoutes + eb + u0 + k] = ea + n0 + k * chunk;
    units[2 * kMaxRoutes + eb + u0 + k] = min(chunk, n1 - k * chunk);
  }
  if (t == blockDim.x - 1) count[0] = pb + b;
  __syncthreads();
  if (e >= 0) order[estart[e] + slot] = t;
}

__device__ __forceinline__ int act_offset(int group, int byte) {
  return (group >> 4) * 512 + (byte >> 4) * 256 + (group & 15) * 16 + (byte & 15);
}

__global__ __launch_bounds__(512) void prep_kernel(
    int4* __restrict__ xq, float* __restrict__ xd, const half* __restrict__ x,
    const int32_t* __restrict__ ids, const float* __restrict__ probabilities, int routes,
    int experts, int chunk, int32_t* units, int32_t* count, int32_t* order) {
  if (blockIdx.x == 0) {
    plan_block(ids, probabilities, routes, experts, chunk, units, count, order);
    return;
  }
  const int b = blockIdx.x - 1, row = b / 10, lane = threadIdx.x % 32;
  const int col = (b % 10) * 512 + threadIdx.x, group = col / 32;
  // quantize_q8 + quantize_q8_1_warp (onecat_ds41_dp4a.cuh), the bytes relaid out.
  const float value = __half2float(x[int64_t(row) * 5120 + col]);
  float maximum = fabsf(value), sum = value;
#pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
    sum += __shfl_xor_sync(0xffffffff, sum, offset);
  }
  const float d = maximum / 127.f;
  const int8_t q = maximum == 0.f ? 0 : int8_t(roundf(value / d));
  const float scale = __low2float(__floats2half2_rn(d, sum));
  // Window layout: per 16 groups, the low 16 bytes of each group (256 B) then the
  // high 16 bytes: a warp's LDG.128 of 16 consecutive groups reads 2 full lines.
  reinterpret_cast<int8_t*>(xq)[int64_t(row) * 5120 + act_offset(group, lane)] = q;
  if (!lane) xd[row * 160 + group] = __fmul_rn(scale, 0.125f);
}

// Compact global tables -> this CTA's shared copies. IQ2: entry e of copy c at
// e * RG + c (signs e * SR + c). IQ1_M with F: entry i (i = qs byte | qh nibble
// << 8) = EM2's ((book[i & 2047] + bias(i >> 11)) ^ 0x80808080) words.
template <int Type, int RG, int SR, int F>
__device__ __forceinline__ void stage_tables(uint2* book, uint2* signs, const uint2* gbook,
                                             const uint2* gsigns) {
  constexpr int BOOK = book_entries<Type>();
  if constexpr (Type == 29 && F) {
    for (int i = threadIdx.x; i < 4096; i += blockDim.x) {
      const uint2 e = __ldg(gbook + (i & 2047));
      const uint32_t bias = (i >> 11) ? 0xFEFEFEFFU : 0x01010101U;
      book[i] = make_uint2((e.x + bias) ^ 0x80808080U, (e.y + bias) ^ 0x80808080U);
    }
  } else if constexpr (RG == 1) {
    for (int i = threadIdx.x; i < BOOK / 2; i += blockDim.x)
      reinterpret_cast<uint4*>(book)[i] = __ldg(reinterpret_cast<const uint4*>(gbook) + i);
  } else {
    for (int i = threadIdx.x; i < BOOK * RG; i += blockDim.x) book[i] = __ldg(gbook + i / RG);
  }
  if constexpr (Type != 29)
    for (int i = threadIdx.x; i < 128 * SR; i += blockDim.x) signs[i] = __ldg(gsigns + i / SR);
}
template <int Type, int RG, int F>
constexpr int table_entries() {
  return (Type == 29 && F) ? 4096 : book_entries<Type>() * RG;
}
template <int Type, int SR>
constexpr int sign_entries() {
  return Type == 29 ? 2 : 128 * SR;
}
// This lane's view of the shared copies.
template <int Type, int RG>
__device__ __forceinline__ const uint2* lane_view(const uint2* table) {
  return table + (Type == 29 ? 0 : (threadIdx.x & (RG - 1)));
}

// ---- gate/up ------------------------------------------------------------------
// CTA = WARPS warps; a row uses L lanes (16 / L partials per lane); ROWS rows per
// CTA block bx; one unit (<= C routes of one expert) u. EM2's arithmetic and
// schedule; activations in the window layout with folded scales.
template <int Type, int L, int WARPS, int C, int RG, int SR, int F>
__device__ __forceinline__ void gate_up_body(
    int u, int bx, const uint2* lbook, const uint2* lsigns, half* sg, half* su, int* s_route,
    Q8_1* __restrict__ down_in, half* __restrict__ projection, const int4* __restrict__ xq,
    const float* __restrict__ xd, const float* __restrict__ probabilities,
    const int32_t* __restrict__ units, const int32_t* __restrict__ order,
    const uint8_t* __restrict__ gate, const uint8_t* __restrict__ up, int stride, int fused) {
  constexpr int RPW = 32 / L, ROWS = WARPS * RPW, P = 16 / L;
  const int expert = __ldg(units + u), first = __ldg(units + kMaxRoutes + u);
  const int n = __ldg(units + 2 * kMaxRoutes + u);
  int xbase[C];
#pragma unroll
  for (int j = 0; j < C; ++j) {
    const int route = __ldg(order + first + min(j, n - 1));
    xbase[j] = (route / kTopK) * 160;
    if (threadIdx.x == 0) s_route[j] = route;
  }
  __syncthreads();
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, jl = lane % L;
  const int local = warp * RPW + lane / L, row = bx * ROWS + local;
  const uint8_t* g = gate + (int64_t(expert) * 576 + row) * stride;
  const uint8_t* v = up + (int64_t(expert) * 576 + row) * stride;
  float ga[C][P], ua[C][P];
#pragma unroll
  for (int j = 0; j < C; ++j)
#pragma unroll
    for (int k = 0; k < P; ++k) ga[j][k] = ua[j][k] = 0.f;
#pragma unroll 2
  for (int i = 0; i < 10; ++i) {
#pragma unroll
    for (int k = 0; k < P; ++k) {
      const int group = jl + L * k + 16 * i;
      const Group<Type, RG, SR, F> dg(g, group, lbook, lsigns), du(v, group, lbook, lsigns);
#pragma unroll
      for (int j = 0; j < C; ++j)
        if (j < n) {
          const int4* xp = xq + (int64_t(xbase[j]) * 2 + (group >> 4) * 32 + (group & 15));
          const int4 lo = __ldg(xp), hi = __ldg(xp + 16);
          const float s = __ldg(xd + xbase[j] + group);
          ga[j][k] = __fadd_rn(ga[j][k], dg.apply(lo, hi, s));
          ua[j][k] = __fadd_rn(ua[j][k], du.apply(lo, hi, s));
        }
    }
  }
#pragma unroll
  for (int j = 0; j < C; ++j) {
    if (j >= n) break;  // uniform over the CTA
    // Upper tree levels inside the lane: partial k is p = jl + L * k.
#pragma unroll
    for (int offset = 8; offset >= L; offset >>= 1) {
      const int step = offset / L;
#pragma unroll
      for (int k = 0; k < P / 2; ++k)
        if (k < step) {
          ga[j][k] = ga[j][k] + ga[j][k + step];
          ua[j][k] = ua[j][k] + ua[j][k + step];
        }
    }
    float gs = ga[j][0], us = ua[j][0];
#pragma unroll
    for (int offset = L / 2; offset; offset >>= 1) {
      gs += __shfl_down_sync(0xffffffffU, gs, offset, L);
      us += __shfl_down_sync(0xffffffffU, us, offset, L);
    }
    if (jl == 0) {
      if (fused) {
        sg[j * ROWS + local] = __float2half_rn(gs);
        su[j * ROWS + local] = __float2half_rn(us);
      } else {
        const int route = s_route[j];
        projection[int64_t(route) * 1152 + row] = __float2half_rn(gs);
        projection[int64_t(route) * 1152 + 576 + row] = __float2half_rn(us);
      }
    }
  }
  if (!fused) return;
  if constexpr (ROWS % 32 == 0) {
    __syncthreads();
    constexpr int BLOCKS = ROWS / 32;
    for (int item = warp; item < n * BLOCKS; item += WARPS) {
      const int j = item / BLOCKS, b = item % BLOCKS, r = b * 32 + lane;
      const int route = s_route[j];
      const float value = __half2float(__float2half_rn(v41_swiglu(
          __half2float(sg[j * ROWS + r]), __half2float(su[j * ROWS + r]), __ldg(probabilities + route))));
      quantize_q8_1_warp(down_in + int64_t(route) * 18 + (bx * ROWS + b * 32) / 32, value);
    }
  }
}

template <int Type, int L, int WARPS, int C, int RG, int SR, int F, int PERSIST>
__global__ __launch_bounds__(WARPS * 32) void gate_up_kernel(
    Q8_1* __restrict__ down_in, half* __restrict__ projection, const int4* __restrict__ xq,
    const float* __restrict__ xd, const float* __restrict__ probabilities,
    const int32_t* __restrict__ units, const int32_t* __restrict__ count,
    const int32_t* __restrict__ order, const uint8_t* __restrict__ gate,
    const uint8_t* __restrict__ up, int stride, const uint2* __restrict__ gbook,
    const uint2* __restrict__ gsigns, int fused) {
  static_assert(L == 16 || L == 8 || L == 4, "lanes per row");
  constexpr int ROWS = WARPS * 32 / L, NB = 576 / ROWS;
  __shared__ __align__(16) uint2 book[table_entries<Type, RG, F>()];
  __shared__ __align__(16) uint2 signs[sign_entries<Type, SR>()];
  __shared__ half sg[C * ROWS], su[C * ROWS];
  __shared__ int s_route[C];
  if constexpr (!PERSIST)
    if (blockIdx.y >= __ldg(count)) return;
  stage_tables<Type, RG, SR, F>(book, signs, gbook, gsigns);
  const uint2* lbook = lane_view<Type, RG>(book);
  const uint2* lsigns = lane_view<Type, SR>(signs);
  if constexpr (PERSIST) {
    const int items = __ldg(count) * NB;
#pragma unroll 1
    for (int item = blockIdx.x; item < items; item += gridDim.x) {
      __syncthreads();  // tables staged; the previous item's epilogue is done with sg/su/s_route
      gate_up_body<Type, L, WARPS, C, RG, SR, F>(item / NB, item % NB, lbook, lsigns, sg, su,
                                                 s_route, down_in, projection, xq, xd,
                                                 probabilities, units, order, gate, up, stride, fused);
    }
  } else {
    gate_up_body<Type, L, WARPS, C, RG, SR, F>(blockIdx.y, blockIdx.x, lbook, lsigns, sg, su,
                                               s_route, down_in, projection, xq, xd,
                                               probabilities, units, order, gate, up, stride, fused);
  }
}

// ---- down -------------------------------------------------------------------
// CTA = WARPS warps x 8 columns (4 lanes per column) x TILES column tiles (EM2).
template <int Type, int WARPS, int TILES, int C, int RG, int SR, int F>
__device__ __forceinline__ void down_body(int u, int bx, const uint2* lbook, const uint2* lsigns,
                                          half* __restrict__ partial, const Q8_1* __restrict__ input,
                                          const int32_t* __restrict__ units,
                                          const int32_t* __restrict__ order,
                                          const uint8_t* __restrict__ weights, int row_stride,
                                          int left_groups) {
  constexpr int kGroups = 18, COLS = WARPS * 8;
  const int expert = __ldg(units + u), first = __ldg(units + kMaxRoutes + u);
  const int n = __ldg(units + 2 * kMaxRoutes + u);
  int route[C];
#pragma unroll
  for (int j = 0; j < C; ++j) route[j] = __ldg(order + first + min(j, n - 1));
  __syncthreads();
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, q = lane & 3;
#pragma unroll 1
  for (int tile = 0; tile < TILES; ++tile) {
    const int col = (bx * TILES + tile) * COLS + warp * 8 + lane / 4;
    const uint8_t* row = weights + (int64_t(expert) * 5120 + col) * row_stride;
    float dot[C];
#pragma unroll
    for (int j = 0; j < C; ++j) dot[j] = 0.f;
#pragma unroll
    for (int start = 0; start < kGroups; start += 4) {
      const int group = start + q;
      // Lanes past the last group decode a valid group and contribute 0.f,
      // like raw_down_unroute's `group < groups ? dot : 0.f`.
      const Group<Type, RG, SR, F> dg(row, min(group, kGroups - 1) + left_groups, lbook, lsigns);
#pragma unroll
      for (int j = 0; j < C; ++j) {
        if (j >= n) break;  // uniform
        const float value = group < kGroups ? dg.apply(input[route[j] * kGroups + group]) : 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float ordered = __shfl_sync(0xffffffffU, value, (lane & ~3) + i);
          if (start + i < kGroups) dot[j] = __fadd_rn(dot[j], ordered);
        }
      }
    }
    if (q == 0)
#pragma unroll
      for (int j = 0; j < C; ++j)
        if (j < n) partial[int64_t(route[j]) * 5120 + col] = __float2half_rn(dot[j]);
  }
}

// Down, staged (EM2 pass 2): the unit's Q8_1 inputs staged in the aligned layout
// (scale folded), per-lane group values in shared memory, one lane per (route,
// column) adds the 18 values in ascending order from 0.f.
template <int Type, int WARPS, int TILES, int C, int RG, int SR, int F>
__device__ __forceinline__ void down_staged_body(
    int u, int bx, const uint2* lbook, const uint2* lsigns, int4 (*xq)[18][2], float (*xd)[18],
    float (*vals)[C][8][19], int* s_route, half* __restrict__ partial,
    const Q8_1* __restrict__ input, const int32_t* __restrict__ units,
    const int32_t* __restrict__ order, const uint8_t* __restrict__ weights, int row_stride,
    int left_groups) {
  constexpr int kGroups = 18, COLS = WARPS * 8;
  const int expert = __ldg(units + u), first = __ldg(units + kMaxRoutes + u);
  const int n = __ldg(units + 2 * kMaxRoutes + u);
  if (threadIdx.x < C) s_route[threadIdx.x] = __ldg(order + first + min(int(threadIdx.x), n - 1));
  __syncthreads();
  for (int i = threadIdx.x; i < n * kGroups; i += blockDim.x) {
    const int j = i / kGroups, g = i % kGroups;
    const Q8_1& q = input[s_route[j] * kGroups + g];
    const int* a = reinterpret_cast<const int*>(q.qs);
    xq[j][g][0] = make_int4(a[0], a[1], a[2], a[3]);
    xq[j][g][1] = make_int4(a[4], a[5], a[6], a[7]);
    xd[j][g] = __fmul_rn(__low2float(q.ds), 0.125f);  // folded (S)
  }
  __syncthreads();
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, q = lane & 3, c = lane / 4;
#pragma unroll 1
  for (int tile = 0; tile < TILES; ++tile) {
    const int col0 = (bx * TILES + tile) * COLS + warp * 8;
    const uint8_t* row = weights + (int64_t(expert) * 5120 + col0 + c) * row_stride;
#pragma unroll
    for (int start = 0; start < kGroups; start += 4) {
      const int group = start + q;
      if (group < kGroups) {
        const Group<Type, RG, SR, F> dg(row, group + left_groups, lbook, lsigns);
#pragma unroll
        for (int j = 0; j < C; ++j)
          if (j < n) vals[warp][j][c][group] = dg.apply(xq[j][group][0], xq[j][group][1], xd[j][group]);
      }
    }
    __syncwarp();
    for (int p = lane; p < C * 8; p += 32) {
      const int j = p / 8, cc = p % 8;
      if (j < n) {
        float dot = 0.f;
#pragma unroll
        for (int g = 0; g < kGroups; ++g) dot = __fadd_rn(dot, vals[warp][j][cc][g]);
        partial[int64_t(s_route[j]) * 5120 + col0 + cc] = __float2half_rn(dot);
      }
    }
    __syncwarp();
  }
}

template <int Type, int WARPS, int TILES, int C, int RG, int SR, int F, int STAGED, int PERSIST>
__global__ __launch_bounds__(WARPS * 32) void down_kernel(
    half* __restrict__ partial, const Q8_1* __restrict__ input,
    const int32_t* __restrict__ units, const int32_t* __restrict__ count,
    const int32_t* __restrict__ order, const uint8_t* __restrict__ weights,
    int row_stride, int left_groups, const uint2* __restrict__ gbook,
    const uint2* __restrict__ gsigns) {
  constexpr int NB = 5120 / (WARPS * 8 * TILES);
  __shared__ __align__(16) uint2 book[table_entries<Type, RG, F>()];
  __shared__ __align__(16) uint2 signs[sign_entries<Type, SR>()];
  __shared__ __align__(16) int4 xq[STAGED ? C : 1][18][2];
  __shared__ float xd[STAGED ? C : 1][18];
  __shared__ float vals[STAGED ? WARPS : 1][C][8][19];
  __shared__ int s_route[C];
  if constexpr (!PERSIST)
    if (blockIdx.y >= __ldg(count)) return;
  stage_tables<Type, RG, SR, F>(book, signs, gbook, gsigns);
  const uint2* lbook = lane_view<Type, RG>(book);
  const uint2* lsigns = lane_view<Type, SR>(signs);
  auto body = [&](int u, int bx) {
    if constexpr (STAGED)
      down_staged_body<Type, WARPS, TILES, C, RG, SR, F>(u, bx, lbook, lsigns, xq, xd, vals,
                                                         s_route, partial, input, units, order,
                                                         weights, row_stride, left_groups);
    else
      down_body<Type, WARPS, TILES, C, RG, SR, F>(u, bx, lbook, lsigns, partial, input, units,
                                                  order, weights, row_stride, left_groups);
  };
  if constexpr (PERSIST) {
    const int items = __ldg(count) * NB;
#pragma unroll 1
    for (int item = blockIdx.x; item < items; item += gridDim.x) {
      __syncthreads();  // tables staged; the previous item is done with xq/xd/s_route
      body(item / NB, item % NB);
    }
  } else {
    body(blockIdx.y, blockIdx.x);
  }
}

// out[t, c] = half(sum over warps w = 0..3 of (sum of routes w, w + 4)), the
// route-per-warp split and FP32 order of raw_down_unroute (top-6, 4 warps).
// Dropped (probability 0) routes have no partial and add nothing (+-0 there).
__global__ void combine_kernel(half* out, const half* partial, const float* probabilities, int tokens) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int t = blockIdx.y;
  if (c >= 5120 || t >= tokens) return;
  float sum = 0.f;
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    float total = 0.f;
    for (int route = w; route < kTopK; route += 4)
      if (probabilities[t * kTopK + route] != 0.f)
        total = __fadd_rn(total, __half2float(partial[(int64_t(t) * kTopK + route) * 5120 + c]));
    sum = __fadd_rn(sum, total);
  }
  out[int64_t(t) * 5120 + c] = __float2half_rn(sum);
}

// Non-fused gate/up variants: onecat_ds41.cu v41_swiglu_q8 over every kept route (grid 1 x M * 6, 576
// threads: 18 full Q8_1 warps). Dropped routes are skipped (down never reads them).
__global__ void swiglu_q8_kernel(Q8_1* out, const half* projection, const float* probabilities) {
  const int col = threadIdx.x, route = blockIdx.y;
  const float probability = probabilities[route];
  if (probability == 0.f) return;  // uniform over the CTA
  const float gate = __half2float(projection[int64_t(route) * 1152 + col]);
  const float up = __half2float(projection[int64_t(route) * 1152 + 576 + col]);
  const float value = __half2float(__float2half_rn(v41_swiglu(gate, up, probability)));
  quantize_q8_1_warp(out + route * 18 + col / 32, value);
}

// ---- variants and launch helpers (shared by the torch ops and the standalone harness) ----
struct GateUpShape {
  int lanes, warps, chunk;
};
// EM2's gate/up shapes (0 is patch 88's geometry).
__host__ __device__ constexpr GateUpShape gate_up_shape(int shape) {
  return shape == 0   ? GateUpShape{16, 4, 8}
         : shape == 1 ? GateUpShape{8, 8, 4}
         : shape == 2 ? GateUpShape{8, 8, 8}
         : shape == 3 ? GateUpShape{4, 8, 4}
         : shape == 4 ? GateUpShape{16, 16, 4}
         : shape == 5 ? GateUpShape{4, 4, 4}
         : shape == 6 ? GateUpShape{16, 4, 4}
         : shape == 7 ? GateUpShape{8, 4, 4}
         : shape == 8 ? GateUpShape{16, 8, 8}
         : shape == 9 ? GateUpShape{16, 16, 16}
                      : GateUpShape{0, 0, 0};
}
struct DownShape {
  int warps, tiles, chunk, staged = 0;
};
// EM2's down shapes (7-11: staged).
__host__ __device__ constexpr DownShape down_shape(int shape) {
  return shape == 0    ? DownShape{4, 1, 8}
         : shape == 1  ? DownShape{8, 2, 4}
         : shape == 2  ? DownShape{8, 4, 4}
         : shape == 3  ? DownShape{4, 4, 4}
         : shape == 4  ? DownShape{8, 1, 4}
         : shape == 5  ? DownShape{8, 1, 8}
         : shape == 6  ? DownShape{8, 2, 16}
         : shape == 7  ? DownShape{8, 1, 4, 1}
         : shape == 8  ? DownShape{8, 2, 4, 1}
         : shape == 9  ? DownShape{8, 4, 4, 1}
         : shape == 10 ? DownShape{4, 2, 4, 1}
         : shape == 11 ? DownShape{4, 2, 8, 1}
                       : DownShape{0, 0, 0};
}
// Variant -> (shape, mode). Modes: 0 = R L S; 1 = L S (one table copy); 2 = R L S F;
// 3 = R L S F P (see the header). Down variants carry no L (packaged Q8_1 input).
struct Variant {
  int shape, mode;
};
constexpr int kGateUpVariants = 28, kDownVariants = 36;
__host__ __device__ constexpr Variant gate_up_variant(int v) {
  constexpr int extra[4] = {4, 8, 9, 0};
  return v < 10 ? Variant{v, 0} : v < 20 ? Variant{v - 10, 1} : v < 24 ? Variant{extra[v - 20], 2}
                                                                       : Variant{extra[v - 24], 3};
}
__host__ __device__ constexpr Variant down_variant(int v) {
  constexpr int extra[6] = {0, 1, 2, 3, 4, 10}, persistent[6] = {0, 4, 5, 10, 11, 7};
  return v < 12 ? Variant{v, 0} : v < 24 ? Variant{v - 12, 1} : v < 30 ? Variant{extra[v - 24], 2}
                                                                       : Variant{persistent[v - 30], 3};
}
inline int gate_up_rows(int variant) {
  const auto s = gate_up_shape(gate_up_variant(variant).shape);
  return s.warps * 32 / s.lanes;
}
inline int gate_up_chunk(int variant) { return gate_up_shape(gate_up_variant(variant).shape).chunk; }
inline int down_chunk(int variant) { return down_shape(down_variant(variant).shape).chunk; }
inline int chunk_of(int gate_up_variant_, int down_variant_) {
  // One plan serves both kernels: the smaller chunk.
  const int a = gate_up_chunk(gate_up_variant_), b = down_chunk(down_variant_);
  return a < b ? a : b;
}

// Table copies per (kernel, type, mode); down: as many as fit 46 KB of static shared memory.
template <int Type, int Mode>
constexpr int gu_rg() { return (Mode == 1 || Type == 29) ? 1 : Type == 16 ? 16 : 8; }
template <int Type, int Mode>
constexpr int gu_sr() { return (Mode == 1 || Type == 29) ? 1 : 8; }
template <int Type, int Mode>
constexpr int gu_f() { return Type == 29 && Mode >= 2 ? 1 : 0; }
template <int W, int C, int Staged>
constexpr int dn_extra() {
  return Staged ? C * 18 * 32 + C * 18 * 4 + W * C * 8 * 19 * 4 + C * 4 : C * 4 + 8 * 19 * 4 * C + 32 * 3;
}
constexpr int kStaticSmem = 47104;
template <int Type, int W, int C, int Staged, int Mode>
constexpr int dn_f() {
  return (Type == 29 && Mode >= 2 && 4096 * 8 + 16 + dn_extra<W, C, Staged>() <= kStaticSmem) ? 1 : 0;
}
template <int Type, int W, int C, int Staged, int Mode>
constexpr int dn_rg() {
  if (Mode == 1 || Type == 29) return 1;
  constexpr int book = book_entries<Type>() * 8, extra = dn_extra<W, C, Staged>();
  return book * 8 + 8192 + extra <= kStaticSmem   ? 8
         : book * 4 + 8192 + extra <= kStaticSmem ? 4
         : book * 2 + 4096 + extra <= kStaticSmem ? 2
                                                  : 1;
}
template <int Type, int W, int C, int Staged, int Mode>
constexpr int dn_sr() {
  if (Mode == 1 || Type == 29) return 1;
  constexpr int rg = dn_rg<Type, W, C, Staged, Mode>();
  return rg >= 4 ? 8 : rg == 2 ? 4 : 1;
}

// Persistent grid: resident CTAs on every SM (queried once per kernel), capped by the
// largest possible item count (units <= routes).
template <class K>
inline int persistent_grid(K kernel, int threads, int max_items) {
  int device = 0, sms = 80, per = 1;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, kernel, threads, 0);
  if (per < 1) per = 1;
  const int grid = sms * per;
  return grid < max_items ? grid : max_items;
}

template <int Type, int V>
void launch_gate_up_v(cudaStream_t stream, int routes, Q8_1* down_in, half* projection,
                      const int4* xq, const float* xd, const float* prob, const int32_t* units,
                      const int32_t* count, const int32_t* order, const uint8_t* gate,
                      const uint8_t* up, int stride, const uint2* book, const uint2* signs,
                      int fused) {
  constexpr auto var = gate_up_variant(V);
  constexpr auto s = gate_up_shape(var.shape);
  constexpr int rows = s.warps * 32 / s.lanes, mode = var.mode, persist = mode == 3 ? 1 : 0;
  auto kernel = gate_up_kernel<Type, s.lanes, s.warps, s.chunk, gu_rg<Type, mode>(), gu_sr<Type, mode>(),
                               gu_f<Type, mode>(), persist>;
  if constexpr (persist) {
    static int grid = 0;  // per instantiation and process (one device per worker)
    if (!grid) grid = persistent_grid(kernel, s.warps * 32, 1 << 30);
    const int items = routes * (576 / rows);
    kernel<<<grid < items ? grid : items, s.warps * 32, 0, stream>>>(
        down_in, projection, xq, xd, prob, units, count, order, gate, up, stride, book, signs, fused);
  } else {
    kernel<<<dim3(576 / rows, routes), s.warps * 32, 0, stream>>>(
        down_in, projection, xq, xd, prob, units, count, order, gate, up, stride, book, signs, fused);
  }
}

template <int Type, int V>
void launch_down_v(cudaStream_t stream, int routes, half* partial, const Q8_1* input,
                   const int32_t* units, const int32_t* count, const int32_t* order,
                   const uint8_t* weights, int row_stride, int left_groups, const uint2* book,
                   const uint2* signs) {
  constexpr auto var = down_variant(V);
  constexpr auto s = down_shape(var.shape);
  constexpr int mode = var.mode, persist = mode == 3 ? 1 : 0;
  constexpr int rg = dn_rg<Type, s.warps, s.chunk, s.staged, mode>();
  constexpr int sr = dn_sr<Type, s.warps, s.chunk, s.staged, mode>();
  constexpr int f = dn_f<Type, s.warps, s.chunk, s.staged, mode>();
  constexpr int blocks = 5120 / (s.warps * 8 * s.tiles);
  auto kernel = down_kernel<Type, s.warps, s.tiles, s.chunk, rg, sr, f, s.staged, persist>;
  if constexpr (persist) {
    static int grid = 0;
    if (!grid) grid = persistent_grid(kernel, s.warps * 32, 1 << 30);
    const int items = routes * blocks;
    kernel<<<grid < items ? grid : items, s.warps * 32, 0, stream>>>(
        partial, input, units, count, order, weights, row_stride, left_groups, book, signs);
  } else {
    kernel<<<dim3(blocks, routes), s.warps * 32, 0, stream>>>(
        partial, input, units, count, order, weights, row_stride, left_groups, book, signs);
  }
}

#define DS41MU_EM3_CASES4(F, B) F(B + 0); F(B + 1); F(B + 2); F(B + 3)
template <int Type>
void launch_gate_up(int variant, cudaStream_t stream, int routes, Q8_1* down_in,
                    half* projection, const int4* xq, const float* xd, const float* prob,
                    const int32_t* units, const int32_t* count, const int32_t* order,
                    const uint8_t* gate, const uint8_t* up, int stride, const uint2* book,
                    const uint2* signs, int fused) {
#define DS41MU_EM3_GU(V)                                                                     \
  case V:                                                                                    \
    launch_gate_up_v<Type, V>(stream, routes, down_in, projection, xq, xd, prob, units, count, \
                              order, gate, up, stride, book, signs, fused);                  \
    break
  switch (variant) {
    DS41MU_EM3_CASES4(DS41MU_EM3_GU, 0);
    DS41MU_EM3_CASES4(DS41MU_EM3_GU, 4);
    DS41MU_EM3_CASES4(DS41MU_EM3_GU, 8);
    DS41MU_EM3_CASES4(DS41MU_EM3_GU, 12);
    DS41MU_EM3_CASES4(DS41MU_EM3_GU, 16);
    DS41MU_EM3_CASES4(DS41MU_EM3_GU, 20);
    DS41MU_EM3_CASES4(DS41MU_EM3_GU, 24);
    default:
      break;
  }
#undef DS41MU_EM3_GU
}

template <int Type>
void launch_down(int variant, cudaStream_t stream, int routes, half* partial, const Q8_1* input,
                 const int32_t* units, const int32_t* count, const int32_t* order,
                 const uint8_t* weights, int row_stride, int left_groups, const uint2* book,
                 const uint2* signs) {
#define DS41MU_EM3_DN(V)                                                                     \
  case V:                                                                                    \
    launch_down_v<Type, V>(stream, routes, partial, input, units, count, order, weights,     \
                           row_stride, left_groups, book, signs);                            \
    break
  switch (variant) {
    DS41MU_EM3_CASES4(DS41MU_EM3_DN, 0);
    DS41MU_EM3_CASES4(DS41MU_EM3_DN, 4);
    DS41MU_EM3_CASES4(DS41MU_EM3_DN, 8);
    DS41MU_EM3_CASES4(DS41MU_EM3_DN, 12);
    DS41MU_EM3_CASES4(DS41MU_EM3_DN, 16);
    DS41MU_EM3_CASES4(DS41MU_EM3_DN, 20);
    DS41MU_EM3_CASES4(DS41MU_EM3_DN, 24);
    DS41MU_EM3_CASES4(DS41MU_EM3_DN, 28);
    DS41MU_EM3_CASES4(DS41MU_EM3_DN, 32);
    default:
      break;
  }
#undef DS41MU_EM3_DN
}
#undef DS41MU_EM3_CASES4

inline int block_bytes_of(int type) { return type == 16 ? 66 : type == 17 ? 74 : 56; }
inline int book_entries_of(int type) {
  return type == 16 ? book_entries<16>() : type == 17 ? book_entries<17>() : book_entries<29>();
}

// ---- host: one fixed workspace per device ----------------------------------------
// Byte layout (each region 256-aligned), sized for 64 rows (384 routes).
constexpr int64_t align256(int64_t n) { return (n + 255) / 256 * 256; }
constexpr int64_t kXq = 0;
constexpr int64_t kXd = kXq + align256(64 * 5120);
constexpr int64_t kUnits = kXd + align256(64 * 160 * 4);
constexpr int64_t kCount = kUnits + align256(3 * kMaxRoutes * 4);
constexpr int64_t kOrder = kCount + 256;
constexpr int64_t kDownIn = kOrder + align256(kMaxRoutes * 4);
constexpr int64_t kProjection = kDownIn + align256(int64_t(kMaxRoutes) * 18 * 36);
constexpr int64_t kPartial = kProjection + align256(int64_t(kMaxRoutes) * 1152 * 2);
constexpr int64_t kBook16 = kPartial + align256(int64_t(kMaxRoutes) * 5120 * 2);
constexpr int64_t kBook17 = kBook16 + align256(book_entries<16>() * 8);
constexpr int64_t kBook29 = kBook17 + align256(book_entries<17>() * 8);
constexpr int64_t kSigns = kBook29 + align256(book_entries<29>() * 8);
constexpr int64_t kWorkspace = kSigns + align256(128 * 8);

struct Views {
  uint8_t* base;
  int4* xq() const { return reinterpret_cast<int4*>(base + kXq); }
  float* xd() const { return reinterpret_cast<float*>(base + kXd); }
  int32_t* units() const { return reinterpret_cast<int32_t*>(base + kUnits); }
  int32_t* count() const { return reinterpret_cast<int32_t*>(base + kCount); }
  int32_t* order() const { return reinterpret_cast<int32_t*>(base + kOrder); }
  Q8_1* down_in() const { return reinterpret_cast<Q8_1*>(base + kDownIn); }
  half* projection() const { return reinterpret_cast<half*>(base + kProjection); }
  half* partial() const { return reinterpret_cast<half*>(base + kPartial); }
  const uint2* book(int type) const {
    return reinterpret_cast<const uint2*>(base + (type == 16 ? kBook16 : type == 17 ? kBook17 : kBook29));
  }
  const uint2* signs() const { return reinterpret_cast<const uint2*>(base + kSigns); }
};

void require(bool ok, const char* what) { TORCH_CHECK(ok, "ds41_decode_moe: ", what); }
bool aligned(const torch::Tensor& t, int bytes) {
  return reinterpret_cast<uintptr_t>(t.data_ptr()) % bytes == 0;
}
bool is_type(int64_t type) { return type == 16 || type == 17 || type == 29; }

Views workspace_of(const torch::Tensor& ws, const torch::Tensor& like) {
  require(ws.is_cuda() && ws.scalar_type() == torch::kUInt8 && ws.is_contiguous() && ws.dim() == 1 &&
              ws.numel() == kWorkspace && aligned(ws, 256) && ws.device() == like.device(),
          "workspace must be a 256-byte aligned uint8 [workspace_bytes()] tensor on the operands' device");
  return Views{ws.data_ptr<uint8_t>()};
}

int rows_of(const torch::Tensor& probabilities) {
  require(probabilities.is_cuda() && probabilities.scalar_type() == torch::kFloat32 &&
              probabilities.is_contiguous() && probabilities.dim() == 2 && probabilities.size(1) == kTopK &&
              probabilities.size(0) >= 1 && probabilities.size(0) <= 64,
          "probabilities must be float32 [1 <= M <= 64, 6]");
  return int(probabilities.size(0));
}

void check_experts(const torch::Tensor& gate, const torch::Tensor& up, const torch::Tensor& down,
                   int64_t gate_type, int64_t down_type, int64_t left, const torch::Tensor& like) {
  require(is_type(gate_type) && is_type(down_type), "types must be IQ2_XXS(16)/IQ2_XS(17)/IQ1_M(29)");
  require(gate.device() == like.device() && gate.scalar_type() == torch::kUInt8 && gate.is_contiguous() &&
              gate.dim() == 3 && gate.size(0) >= 1 && gate.size(0) <= kMaxExperts && gate.size(1) == 576 &&
              gate.size(2) == 20 * block_bytes_of(gate_type) && gate.sizes() == up.sizes() &&
              up.device() == like.device() && up.scalar_type() == torch::kUInt8 && up.is_contiguous() &&
              aligned(gate, 8) && aligned(up, 8),
          "gate/up must be 8-byte aligned original rows uint8 [E <= 1024, 576, 20 * block]");
  require(left >= 0 && left % 64 == 0 && left < 256, "down left offset must be 0/64/128/192");
  require(down.device() == like.device() && down.scalar_type() == torch::kUInt8 && down.is_contiguous() &&
              down.dim() == 3 && down.size(0) == gate.size(0) && down.size(1) == 5120 &&
              down.size(2) == ((left + 576 + 255) / 256) * block_bytes_of(down_type) && aligned(down, 8),
          "down must be 8-byte aligned original rows uint8 [E, 5120, 3 * block]");
}

void launch_prep(const Views& v, const torch::Tensor& x, const torch::Tensor& ids,
                 const torch::Tensor& probabilities, int experts, int chunk, cudaStream_t stream) {
  const int m = rows_of(probabilities);
  require(x.is_cuda() && x.scalar_type() == torch::kFloat16 && x.is_contiguous() && x.dim() == 2 &&
              x.size(0) == m && x.size(1) == 5120 && aligned(x, 2),
          "x must be FP16 [M, 5120]");
  require(ids.device() == x.device() && ids.scalar_type() == torch::kInt32 && ids.is_contiguous() &&
              ids.sizes() == probabilities.sizes() && probabilities.device() == x.device(),
          "ids must be int32 [M, 6] like probabilities");
  require(experts >= 1 && experts <= kMaxExperts && chunk >= 1 && chunk <= 16, "experts/chunk");
  prep_kernel<<<1 + m * 10, 512, 0, stream>>>(v.xq(), v.xd(), reinterpret_cast<const half*>(x.data_ptr()),
                                              ids.data_ptr<int32_t>(), probabilities.data_ptr<float>(),
                                              m * kTopK, experts, chunk, v.units(), v.count(), v.order());
}

void launch_gate_up_any(const Views& v, const torch::Tensor& probabilities, const torch::Tensor& gate,
                        const torch::Tensor& up, int type, int variant, cudaStream_t stream) {
  const int routes = rows_of(probabilities) * kTopK, fused = gate_up_rows(variant) % 32 == 0;
#define DS41_GATE(T)                                                                                    \
  launch_gate_up<T>(variant, stream, routes, v.down_in(), v.projection(), v.xq(), v.xd(),             \
                    probabilities.data_ptr<float>(), v.units(), v.count(), v.order(),                  \
                    gate.data_ptr<uint8_t>(), up.data_ptr<uint8_t>(), int(gate.size(2)), v.book(T),    \
                    v.signs(), fused)
  if (type == 16)
    DS41_GATE(16);
  else if (type == 17)
    DS41_GATE(17);
  else
    DS41_GATE(29);
#undef DS41_GATE
  if (!fused)
    swiglu_q8_kernel<<<dim3(1, routes), 576, 0, stream>>>(v.down_in(), v.projection(),
                                                          probabilities.data_ptr<float>());
}

void launch_down_any(const Views& v, const torch::Tensor& probabilities, const torch::Tensor& down, int type,
                     int left, int variant, cudaStream_t stream) {
  const int routes = rows_of(probabilities) * kTopK;
#define DS41_DOWN(T)                                                                                   \
  launch_down<T>(variant, stream, routes, v.partial(), v.down_in(), v.units(), v.count(), v.order(),  \
                 down.data_ptr<uint8_t>(), int(down.size(2)), left / 32, v.book(T), v.signs())
  if (type == 16)
    DS41_DOWN(16);
  else if (type == 17)
    DS41_DOWN(17);
  else
    DS41_DOWN(29);
#undef DS41_DOWN
}

void launch_combine(const Views& v, const torch::Tensor& out, const torch::Tensor& probabilities,
                    cudaStream_t stream) {
  const int m = rows_of(probabilities);
  require(out.device() == probabilities.device() && out.scalar_type() == torch::kFloat16 &&
              out.is_contiguous() && out.dim() == 2 && out.size(0) == m && out.size(1) == 5120,
          "out must be FP16 [M, 5120]");
  combine_kernel<<<dim3(5120 / 128, m), 128, 0, stream>>>(reinterpret_cast<half*>(out.data_ptr()), v.partial(),
                                                          probabilities.data_ptr<float>(), m);
}

void check_variants(int64_t gate_up_variant_, int64_t down_variant_) {
  require(gate_up_variant_ >= 0 && gate_up_variant_ < kGateUpVariants, "gate/up variant");
  require(down_variant_ >= 0 && down_variant_ < kDownVariants, "down variant");
}
}  // namespace tensorfold::ds41_decode_moe

using namespace tensorfold::ds41_decode_moe;

int64_t ds41_decode_moe_workspace_bytes() { return kWorkspace; }

void ds41_decode_moe_init(torch::Tensor ws) {
  const Views v = workspace_of(ws, ws);
  const c10::cuda::CUDAGuard guard(ws.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  build_book<16><<<(book_entries<16>() + 255) / 256, 256, 0, stream>>>(const_cast<uint2*>(v.book(16)));
  build_book<17><<<(book_entries<17>() + 255) / 256, 256, 0, stream>>>(const_cast<uint2*>(v.book(17)));
  build_book<29><<<(book_entries<29>() + 255) / 256, 256, 0, stream>>>(const_cast<uint2*>(v.book(29)));
  build_signs<<<1, 128, 0, stream>>>(const_cast<uint2*>(v.signs()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<std::vector<int64_t>> ds41_decode_moe_variants() {
  // [gate/up: chunk, fused (32-row CTAs)] per variant, then [down: chunk] per variant.
  std::vector<std::vector<int64_t>> out;
  for (int v = 0; v < kGateUpVariants; ++v) out.push_back({0, gate_up_chunk(v), gate_up_rows(v) % 32 == 0});
  for (int v = 0; v < kDownVariants; ++v) out.push_back({1, down_chunk(v), 0});
  return out;
}

void ds41_decode_moe_experts(torch::Tensor ws, torch::Tensor out, const torch::Tensor& x,
                             const torch::Tensor& ids, const torch::Tensor& probabilities,
                             const torch::Tensor& gate, const torch::Tensor& up, const torch::Tensor& down,
                             int64_t gate_type, int64_t down_type, int64_t left, int64_t gate_up_variant_,
                             int64_t down_variant_) {
  const Views v = workspace_of(ws, x);
  check_experts(gate, up, down, gate_type, down_type, left, x);
  check_variants(gate_up_variant_, down_variant_);
  const c10::cuda::CUDAGuard guard(x.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  launch_prep(v, x, ids, probabilities, int(gate.size(0)), chunk_of(int(gate_up_variant_), int(down_variant_)),
              stream);
  launch_gate_up_any(v, probabilities, gate, up, int(gate_type), int(gate_up_variant_), stream);
  launch_down_any(v, probabilities, down, int(down_type), int(left), int(down_variant_), stream);
  launch_combine(v, out, probabilities, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Stage ops for the schedule sweep (tests/cuda/bench_ds41_decode_moe.py): same launches, one at a time.
void ds41_decode_moe_prep(torch::Tensor ws, const torch::Tensor& x, const torch::Tensor& ids,
                          const torch::Tensor& probabilities, int64_t experts, int64_t chunk) {
  const Views v = workspace_of(ws, x);
  const c10::cuda::CUDAGuard guard(x.device());
  launch_prep(v, x, ids, probabilities, int(experts), int(chunk), at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ds41_decode_moe_gate_up(torch::Tensor ws, const torch::Tensor& probabilities, const torch::Tensor& gate,
                             const torch::Tensor& up, int64_t type, int64_t variant) {
  const Views v = workspace_of(ws, gate);
  require(is_type(type) && gate.size(2) == 20 * block_bytes_of(type), "gate/up type");
  check_variants(variant, 0);
  const c10::cuda::CUDAGuard guard(gate.device());
  launch_gate_up_any(v, probabilities, gate, up, int(type), int(variant), at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ds41_decode_moe_down(torch::Tensor ws, const torch::Tensor& probabilities, const torch::Tensor& down,
                          int64_t type, int64_t left, int64_t variant) {
  const Views v = workspace_of(ws, down);
  require(is_type(type) && left >= 0 && left % 64 == 0 && left < 256 &&
              down.size(2) == ((left + 576 + 255) / 256) * block_bytes_of(type), "down type/left");
  check_variants(0, variant);
  const c10::cuda::CUDAGuard guard(down.device());
  launch_down_any(v, probabilities, down, int(type), int(left), int(variant), at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ds41_decode_moe_combine(torch::Tensor ws, torch::Tensor out, const torch::Tensor& probabilities) {
  const Views v = workspace_of(ws, out);
  const c10::cuda::CUDAGuard guard(out.device());
  launch_combine(v, out, probabilities, at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
