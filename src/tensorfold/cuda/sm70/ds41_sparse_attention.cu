// SPDX-License-Identifier: Apache-2.0
// DeepSeek-V4.1 fused sparse attention over window rings and packed compressed rows, SM70.
//
// Provenance: the friend's overlay patch 181 (ds41_pf_attn, SPDX Apache-2.0), snapshot
//   lp-snap-c1es8-numa-admit2-imgtext-share2-fix5-ram4-stepbud-ram3-memfix-ctx1m-ppovl-opt2-pfx-1-eng-pfxfix4,
//   patches/ds41mu_files/ds41mu_attn2.cu (648 lines; his TP4 = 16 query heads per rank).
// TensorFold changes (TP8, 2026-11-21):
//   - 8 query heads per rank (kHeads 16 -> 8): the m8n8k4 A operand is exactly the 8 heads, so the "high"
//     (heads 8..15) MMAs, accumulators and softmax slots are dropped; fragment layouts, key streaming, the
//     dequantization and the softmax arithmetic are his, unchanged.
//   - 80 registers and ~16 KB shared memory per CTA (ptxas, sm_70): 6 CTAs (24 warps) per SM instead of 3.
//   - pybind11 module (TensorFold JIT build) instead of TORCH_LIBRARY.
//   - His split mode (5 CTAs per token for <= 16-token steps) is not ported: inside the lane's decode CUDA graphs
//     patch 87's gather + bmm is faster at every decode bucket (gpuq attn-2: M=1 19 us vs 43 us split).
//
// Contract (patch 87 _attend: gather + bmm Q.K^T FP32 + _sink_softmax + bmm P.K with V = K):
//   out[t, h, :] = FP16( sum_k p[t, h, k] * key[t, k, :] ),  p = FP16(softmax over the 128 window keys and the
//   512 selected compressed keys (unused keys -inf) with the per-head sink in the denominator), scores
//   FP32(q . key) * scale. Window keys come from this forward's FP16 latents (positions >= the request's first
//   position in this step) or the request's slot ring; compressed keys are the packed rows decoded exactly as
//   kv_pack.unpack_c_row (E2M1 x E4M3 products are exact in FP16). The FP16 key values and the FP16
//   probability boundary are the reference's; FP32 summation orders differ (dot, softmax sum, P.K).
//   With RotorQuant ISO64 H64, both caches contain codec 11 rows. Q and the result are in the
//   same rotated basis; the FP16 QK and PV key fragments are centroid * scale rounded once.
//   One token per CTA (row-wise, batch invariant), deterministic.
#include <cuda_fp16.h>
#include <cstdint>
#include <limits>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "rotorquant_q4.cuh"

namespace ds41_sparse_attention {

constexpr int kHeads = 8, kDim = 512, kWindow = 128, kRing = 128, kWarps = 4, kThreads = 128;
constexpr int kQLd = kDim + 8;  // halves per shared query row
constexpr int kCodeBytes = kDim / 2, kLegacyRow = kCodeBytes + kDim / 16;
constexpr int kRotorCodec = 11, kRotorGroup = 128, kRotorHalf = kRotorGroup / 2, kRotorTable = 16;
constexpr int kRotorRow = kCodeBytes + sizeof(uint32_t) * (kDim / kRotorGroup);
static_assert(kRotorRow == 272 && kRotorRow % 16 == 0 && kLegacyRow == 288 && kLegacyRow % 16 == 0);

struct Args {
  const half* query;      // [N, 8, 512]
  const void* ring;       // [S + 1, 128, 512] FP16 or [S + 1, 128, 272] uint8
  const void* values;     // [N, 512] FP16 or [N, 272] uint8
  const int32_t* qsl;     // [R + 1]
  const int32_t* req_slot;
  const uint8_t* cache;   // pages
  const int32_t* table;   // [R, W]
  const int64_t* positions;
  const uint8_t* valid;   // bool
  const int32_t* token_req;
  const int32_t* topk;    // [N, 512] or null
  const float* sink;      // [8]
  half* out;              // [N, 8, 512]
  int64_t table_stride, page_bytes, pages, table_columns, requests;
  int block_size, ratio, offset, topk_count, slots, first;
  float scale;
};

template <int W>
struct Geo {
  static constexpr int kPLd = W + 8;  // halves per shared probability row
  static constexpr int kTilesPerWarp = W / 32 / kWarps;
  static constexpr int kQBytes = kHeads * kQLd * 2;
  static constexpr int kPBytes = kHeads * kPLd * 2;
  static constexpr int kBuf = kQBytes > kPBytes ? kQBytes : kPBytes;
  static constexpr int kRed = 2 * kWarps * kHeads * 4;
  static constexpr int kShared = kBuf + kRed + W * 8;
};
static_assert(Geo<640>::kShared + kRotorTable * sizeof(float) <= 48 * 1024,
              "no dynamic shared-memory opt-in needed");
static_assert(Geo<640>::kBuf % 16 == 0 && Geo<128>::kBuf % 16 == 0, "aligned reduction / descriptor areas");

__device__ __forceinline__ void mma_rc(float (&d)[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
      "{%0, %1, %2, %3, %4, %5, %6, %7};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_rr(float (&d)[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 "
      "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
      "{%0, %1, %2, %3, %4, %5, %6, %7};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// tl.exp / '/' as Triton emits them for _sink_softmax (FP32).
__device__ __forceinline__ float ex2_approx(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float div_full(float a, float b) {
  float y;
  asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}

// E4M3 byte -> FP16 bits (exact: 4-bit exponent, 3-bit mantissa fit FP16).
__device__ __forceinline__ uint32_t e4m3_half_bits(uint32_t b) {
  const uint32_t sign = (b & 0x80u) << 8, e = (b >> 3) & 15u, m = b & 7u;
  const uint32_t normal = ((e + 8u) << 10) | (m << 7);
  // The eight m * 2^-9 values have a zero low FP16 byte. Select their high byte.
  const uint32_t subnormal = (__byte_perm(0x1E1C1800u, 0x23222120u, m) & 0xFFu) << 8;
  const uint32_t finite = sign | (e == 0u ? subnormal : normal);
  return e == 15u && m == 7u ? 0x7e00u : finite;
}

// Four E2M1 codes (16 bits, code k in bits 4k..4k+3) -> 2 half2 (codes 0,1 / 2,3),
// times the FP16 scale (exact products), as the gather's FP32 product -> FP16.
__device__ __forceinline__ void dequant4(uint32_t c, uint32_t scale2, uint32_t& lo, uint32_t& hi) {
  const uint32_t mags = __byte_perm(0x3E3C3800u, 0x46444240u, c & 0x7777u);
  uint32_t h01 = __byte_perm(mags, 0u, 0x1404);
  uint32_t h23 = __byte_perm(mags, 0u, 0x3424);
  h01 |= ((c << 12) & 0x8000u) | ((c << 24) & 0x80000000u);
  h23 |= ((c << 4) & 0x8000u) | ((c << 16) & 0x80000000u);
  const __half2 s = *reinterpret_cast<const __half2*>(&scale2);
  const __half2 x = __hmul2(*reinterpret_cast<const __half2*>(&h01), s);
  const __half2 y = __hmul2(*reinterpret_cast<const __half2*>(&h23), s);
  lo = *reinterpret_cast<const uint32_t*>(&x);
  hi = *reinterpret_cast<const uint32_t*>(&y);
}

__device__ __forceinline__ uint32_t splat_scale(uint32_t code) {
  const uint32_t bits = e4m3_half_bits(code);
  return bits | (bits << 16);
}

// The binding and packed row stride guarantee alignment. These cache bytes
// are read-only during attention. Load four bytes directly so the compiler
// sees the indirect memory access without a C++ typed alias of byte storage.
__device__ __forceinline__ uint32_t load_scale_word(const uint8_t* bytes) {
  uint32_t result;
  asm volatile("ld.global.nc.b32 %0, [%1];" : "=r"(result) : "l"(__cvta_generic_to_global(bytes)) : "memory");
  return result;
}

// QK processes four coordinates at a time. PV uses four of the same groups
// to form its contiguous 16-coordinate fragment.
__device__ __forceinline__ uint32_t rotor_pair(uint32_t codes, float scale, const float* centroids) {
  const float lo = __fmul_rn(centroids[codes & 15u], scale);
  const float hi = __fmul_rn(centroids[(codes >> 4) & 15u], scale);
  const half2 pair = __floats2half2_rn(lo, hi);
  return uint32_t(__half_as_ushort(__low2half(pair))) | (uint32_t(__half_as_ushort(__high2half(pair))) << 16);
}

// Each packed chunk covers 16 coordinates at an 8-byte-aligned byte address.
__device__ __forceinline__ uint2 rotor_codes(const uint8_t* bytes) {
  uint2 result;
  asm volatile("ld.global.nc.v2.b32 {%0, %1}, [%2];"
               : "=r"(result.x), "=r"(result.y) : "l"(__cvta_generic_to_global(bytes)) : "memory");
  return result;
}

__device__ __forceinline__ float rotor_scale(const uint8_t* row, int dim) {
  const uint32_t word = load_scale_word(row + kCodeBytes + sizeof(uint32_t) * (dim / kRotorGroup));
  return tf_rq4::half_group_scale(__uint_as_float(word), (dim % kRotorGroup) / kRotorHalf);
}

// 0 = unused key; FP16 row pointer; packed row pointer | 1.
template <bool Rotor>
__device__ __forceinline__ uint64_t describe(const Args& a, int key, int64_t t, int64_t r,
                                             int64_t position, bool live, int64_t s0,
                                             int64_t start, int64_t slot) {
  if (!live) return 0;
  if (key < kWindow) {
    const int64_t at = position - kWindow + 1 + key;
    if (at < 0) return 0;
    if constexpr (Rotor) {
      const uint8_t* src = at >= start ? static_cast<const uint8_t*>(a.values) + (s0 + (at - start)) * kRotorRow
                                      : static_cast<const uint8_t*>(a.ring) + (slot * kRing + at % kRing) * kRotorRow;
      return reinterpret_cast<uint64_t>(src) | 1ull;
    } else {
      const half* src = at >= start ? static_cast<const half*>(a.values) + (s0 + at - start) * kDim
                                    : static_cast<const half*>(a.ring) + (slot * kRing + at % kRing) * kDim;
      return reinterpret_cast<uint64_t>(src);
    }
  }
  const int64_t index = a.topk[t * a.topk_count + key - kWindow];
  if (index < 0 || index >= (position + 1) / a.ratio) return 0;
  const int rows = a.block_size / a.ratio;
  const int64_t column = index / rows;
  if constexpr (Rotor) {
    if (column >= a.table_columns) return 0;
  }
  const int64_t page = a.table[r * a.table_stride + column];
  if (page < 0) return 0;
  if constexpr (Rotor) {
    if (page >= a.pages) return 0;
  }
  constexpr int row_bytes = Rotor ? kRotorRow : kLegacyRow;
  const uint8_t* row = a.cache + page * a.page_bytes + a.offset + (index % rows) * row_bytes;
  return reinterpret_cast<uint64_t>(row) | 1ull;
}

// Scores of one 32-key tile (8 heads x 32 keys, C layout) for this lane's key d.
template <bool Rotor>
__device__ __forceinline__ void tile_scores(bool window, uint64_t d, const half* qa, const float* centroids,
                                            float (&lo)[8]) {
#pragma unroll
  for (int e = 0; e < 8; ++e) lo[e] = 0.f;
  if constexpr (Rotor) {
    const uint8_t* row = reinterpret_cast<const uint8_t*>(d & ~1ull);
#pragma unroll 1
    for (int start = 0; start < kDim; start += kRotorHalf) {
      const float scale = d ? rotor_scale(row, start) : 0.0f;
#pragma unroll 2
      for (int dim = start; dim < start + kRotorHalf; dim += 16) {
        const uint2 v = d ? rotor_codes(row + dim / 2) : make_uint2(0, 0);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint32_t word = j < 2 ? v.x : v.y;
          const uint32_t codes = word >> (16 * (j & 1));
          const uint32_t k0 = d ? rotor_pair(codes, scale, centroids) : 0u;
          const uint32_t k1 = d ? rotor_pair(codes >> 8, scale, centroids) : 0u;
          const uint2 a0 = *reinterpret_cast<const uint2*>(qa + dim + 4 * j);
          mma_rc(lo, a0.x, a0.y, k0, k1);
        }
      }
    }
  } else if (window) {  // FP16 rows
    const uint4* src = reinterpret_cast<const uint4*>(d);
    uint4 nxt = d ? __ldg(src) : make_uint4(0, 0, 0, 0);
#pragma unroll 4
    for (int c = 0; c < kDim / 8; ++c) {
      const uint4 v = nxt;
      if (c + 1 < kDim / 8) nxt = d ? __ldg(src + c + 1) : make_uint4(0, 0, 0, 0);
      const uint2 a0 = *reinterpret_cast<const uint2*>(qa + 8 * c);
      const uint2 a1 = *reinterpret_cast<const uint2*>(qa + 8 * c + 4);
      mma_rc(lo, a0.x, a0.y, v.x, v.y);
      mma_rc(lo, a1.x, a1.y, v.z, v.w);
    }
  } else {  // packed rows (256 B of E2M1 codes, 32 E4M3 scales)
    const uint8_t* row = reinterpret_cast<const uint8_t*>(d & ~1ull);
    uint2 nxt = d ? __ldg(reinterpret_cast<const uint2*>(row)) : make_uint2(0, 0);
#pragma unroll 1
    for (int four = 0; four < kDim / 16; four += 4) {
      const uint32_t scales = d ? load_scale_word(row + 256 + four) : 0u;
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        const int c = four + g;
        const uint2 v = nxt;
        if (c + 1 < kDim / 16) nxt = d ? __ldg(reinterpret_cast<const uint2*>(row) + c + 1) : make_uint2(0, 0);
        const uint32_t s2 = splat_scale((scales >> (8 * g)) & 0xffu);
#pragma unroll
        for (int j = 0; j < 4; ++j) {  // k-step: dims 16c + 4j .. + 3
          const uint32_t word = j < 2 ? v.x : v.y;
          uint32_t k0, k1;
          dequant4((word >> (16 * (j & 1))) & 0xffffu, s2, k0, k1);
          const uint2 a0 = *reinterpret_cast<const uint2*>(qa + 16 * c + 4 * j);
          mma_rc(lo, a0.x, a0.y, k0, k1);
        }
      }
    }
  }
}

// out[8 x 128-dim warp slice] += P[:, keys 4 ks0 .. 4 ks1) . K (P columns pcol0-based).
template <bool Rotor>
__device__ __forceinline__ void pv_steps(const uint64_t* desc, int key0, const half* pa, int pcol0, int ks0,
                                         int ks1, int dbase, int kr, const float* centroids, float (&o)[4][8]) {
#pragma unroll 2
  for (int ks = ks0; ks < ks1; ++ks) {
    const int key = 4 * ks + kr;
    const uint64_t d = desc[key - key0];
    uint32_t b[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    if constexpr (Rotor) {
      if (d) {
        const uint8_t* row = reinterpret_cast<const uint8_t*>(d & ~1ull);
        const uint2 v = rotor_codes(row + dbase / 2);
        const float scale = rotor_scale(row, dbase);  // 16 aligned coordinates fit a single 64-value half.
#pragma unroll
        for (int j = 0; j < 8; ++j) b[j] = rotor_pair((j < 4 ? v.x : v.y) >> (8 * (j & 3)), scale, centroids);
      }
    } else if (key < kWindow) {
      if (d) {
        const uint4* src = reinterpret_cast<const uint4*>(reinterpret_cast<const half*>(d) + dbase);
        const uint4 v0 = __ldg(src), v1 = __ldg(src + 1);
        b[0] = v0.x; b[1] = v0.y; b[2] = v0.z; b[3] = v0.w;
        b[4] = v1.x; b[5] = v1.y; b[6] = v1.z; b[7] = v1.w;
      }
    } else if (d) {
      const uint8_t* row = reinterpret_cast<const uint8_t*>(d & ~1ull);
      const uint2 v = __ldg(reinterpret_cast<const uint2*>(row + dbase / 2));
      const uint32_t s2 = splat_scale(__ldg(row + 256 + dbase / 16));
      dequant4(v.x & 0xffffu, s2, b[0], b[1]);
      dequant4(v.x >> 16, s2, b[2], b[3]);
      dequant4(v.y & 0xffffu, s2, b[4], b[5]);
      dequant4(v.y >> 16, s2, b[6], b[7]);
    }
    const uint2 a0 = *reinterpret_cast<const uint2*>(pa + 4 * ks - pcol0);
#pragma unroll
    for (int j = 0; j < 4; ++j) mma_rr(o[j], a0.x, a0.y, b[2 * j], b[2 * j + 1]);
  }
}

struct Token {
  int64_t t, r, position, s0, start, slot;
  bool live;
};

template <bool Rotor>
__device__ __forceinline__ Token token_of(const Args& a, int64_t t) {
  Token k;
  k.t = t;
  k.r = a.token_req[t];
  k.position = a.positions[t];
  k.live = a.valid[t] != 0 && k.r >= 0;
  if constexpr (Rotor) {
    if (!k.live || k.position < 0 || k.position == INT64_MAX || k.r >= a.requests)
      return {k.t, k.r, k.position, 0, 0, a.slots, false};
  }
  const int64_t rr = k.r >= 0 ? k.r : 0;
  k.s0 = a.qsl[rr];
  if constexpr (Rotor) {
    if (k.s0 < 0 || k.s0 > t)
      return {k.t, k.r, k.position, 0, 0, a.slots, false};
  }
  k.start = a.positions[k.s0];
  if constexpr (Rotor) {
    // The current-row offset is nonnegative and no later than this token.
    if (k.start < 0 || k.position < k.start || k.position - k.start > k.t - k.s0)
      return {k.t, k.r, k.position, 0, 0, a.slots, false};
  }
  k.slot = k.live ? a.req_slot[rr] : a.slots;
  if (k.slot < 0 || k.slot > a.slots) k.slot = a.slots;
  return k;
}

__device__ __forceinline__ void load_query(const Args& a, int64_t t, half* qs) {
  for (int i = threadIdx.x; i < kHeads * kDim / 8; i += kThreads) {
    const int h = i / (kDim / 8), c = (i % (kDim / 8)) * 8;
    *reinterpret_cast<uint4*>(&qs[h * kQLd + c]) =
        __ldg(reinterpret_cast<const uint4*>(a.query + (t * kHeads + h) * kDim + c));
  }
}

// C element e (0..7) of a lane: head crow + ((e & 2) ? 2 : 0), key cbase + (e & 4) + (e & 1).
__device__ __forceinline__ int slot_of(int e) { return (e >> 1) & 1; }
constexpr float kLog2e = 1.44269502162933349609375f;  // 0x3FB8AA3B, Triton's tl.exp

// C element (head, dim) of P.K accumulator o[j][e] for this lane.
__device__ __forceinline__ int out_head(int lane, int e) { return (lane & 1) + ((lane >> 4) << 2) + (e & 2); }
__device__ __forceinline__ int out_dim(int lane, int warp, int j, int e) {
  // logical column 8 qp + (lane & 2) + (e & 4) + (e & 1) = B lane (qp, h' = (e & 4) >> 2), element
  // (lane & 2) + (e & 1)  ->  dim 128 w + 16 (2 qp + h') + 4 j + (lane & 2) + (e & 1).
  return 128 * warp + 16 * (2 * ((lane >> 2) & 3) + ((e & 4) >> 2)) + 4 * j + (lane & 2);
}

__device__ __forceinline__ void store_out(const Args& a, int64_t t, const float (&o)[4][8]) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  half* out = a.out + t * kHeads * kDim;
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int e = 0; e < 8; e += 2)
      *reinterpret_cast<__half2*>(out + out_head(lane, e) * kDim + out_dim(lane, warp, j, e)) =
          __floats2half2_rn(o[j][e], o[j][e + 1]);
}

template <int W, bool Rotor>
__global__ __launch_bounds__(kThreads, 6) void attend_kernel(Args a) {
  using G = Geo<W>;
  extern __shared__ __align__(16) unsigned char smem[];
  half* qs = reinterpret_cast<half*>(smem);   // [8][kQLd]   (phase 1)
  half* ps = reinterpret_cast<half*>(smem);   // [8][kPLd]   (phases 2-3, same bytes)
  float* red = reinterpret_cast<float*>(smem + G::kBuf);  // [2][kWarps][8]
  uint64_t* desc = reinterpret_cast<uint64_t*>(smem + G::kBuf + G::kRed);  // [W]
  float* centroids = nullptr;
  if constexpr (Rotor) {
    centroids = reinterpret_cast<float*>(smem + G::kShared);
    // Different lanes may read different codes; preload the constant table once per CTA.
    if (threadIdx.x < kRotorTable) centroids[threadIdx.x] = tf_rq4::selected_centroid<kRotorCodec>(threadIdx.x);
  }
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const Token k = token_of<Rotor>(a, a.first + blockIdx.x);
  load_query(a, k.t, qs);
  for (int i = threadIdx.x; i < W; i += kThreads)
    desc[i] = describe<Rotor>(a, i, k.t, k.r, k.position, k.live, k.s0, k.start, k.slot);
  __syncthreads();

  // ---------------- phase 1: scores, 8 heads x 32 keys per tile, in registers
  const int arow = (lane & 3) + ((lane >> 4) << 2);                      // A row = head
  const int bkey = ((lane & 12) << 1) + (lane & 3) + ((lane >> 4) << 2);  // B column = key in tile
  float sc[G::kTilesPerWarp][8];
#pragma unroll
  for (int i = 0; i < G::kTilesPerWarp; ++i) {
    const int tile = warp + i * kWarps;
    tile_scores<Rotor>(tile * 32 < kWindow, desc[tile * 32 + bkey], qs + arow * kQLd, centroids, sc[i]);
  }

  // ---------------- phase 2: _sink_softmax
  const int crow = (lane & 1) + ((lane >> 4) << 2);
  const int cbase = ((lane & 12) << 1) + (lane & 2);
  float mx[2] = {-INFINITY, -INFINITY};  // heads crow, crow + 2
#pragma unroll
  for (int i = 0; i < G::kTilesPerWarp; ++i) {
    const int tile = warp + i * kWarps;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int col = tile * 32 + cbase + (e & 4) + (e & 1);
      const float v = desc[col] != 0 ? __fmul_rn(sc[i][e], a.scale) : -INFINITY;
      sc[i][e] = v;
      mx[slot_of(e)] = fmaxf(mx[slot_of(e)], v);
    }
  }
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    mx[j] = fmaxf(mx[j], __shfl_xor_sync(0xffffffffu, mx[j], 2));
    mx[j] = fmaxf(mx[j], __shfl_xor_sync(0xffffffffu, mx[j], 4));
    mx[j] = fmaxf(mx[j], __shfl_xor_sync(0xffffffffu, mx[j], 8));
  }
  const int heads2[2] = {crow, crow + 2};
  if ((lane & 14) == 0) {
#pragma unroll
    for (int j = 0; j < 2; ++j) red[warp * kHeads + heads2[j]] = mx[j];
  }
  __syncthreads();  // also: every warp is done reading qs
  float m[2], sum[2] = {0.f, 0.f};
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    m[j] = a.sink[heads2[j]];
#pragma unroll
    for (int w = 0; w < kWarps; ++w) m[j] = fmaxf(m[j], red[w * kHeads + heads2[j]]);
  }
#pragma unroll
  for (int i = 0; i < G::kTilesPerWarp; ++i) {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float x = ex2_approx(__fmul_rn(__fsub_rn(sc[i][e], m[slot_of(e)]), kLog2e));
      sc[i][e] = x;
      sum[slot_of(e)] += x;
    }
  }
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    sum[j] += __shfl_xor_sync(0xffffffffu, sum[j], 2);
    sum[j] += __shfl_xor_sync(0xffffffffu, sum[j], 4);
    sum[j] += __shfl_xor_sync(0xffffffffu, sum[j], 8);
  }
  float* red2 = red + kWarps * kHeads;
  if ((lane & 14) == 0) {
#pragma unroll
    for (int j = 0; j < 2; ++j) red2[warp * kHeads + heads2[j]] = sum[j];
  }
  __syncthreads();
  float den[2];
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) s += red2[w * kHeads + heads2[j]];
    den[j] = __fadd_rn(s, ex2_approx(__fmul_rn(__fsub_rn(a.sink[heads2[j]], m[j]), kLog2e)));
  }
#pragma unroll
  for (int i = 0; i < G::kTilesPerWarp; ++i) {
    const int tile = warp + i * kWarps;
#pragma unroll
    for (int e = 0; e < 8; e += 2) {
      const int col = tile * 32 + cbase + (e & 4);
      const __half2 p = __halves2half2(__float2half_rn(div_full(sc[i][e], den[slot_of(e)])),
                                       __float2half_rn(div_full(sc[i][e + 1], den[slot_of(e)])));
      *reinterpret_cast<__half2*>(&ps[heads2[slot_of(e)] * G::kPLd + col]) = p;
    }
  }
  __syncthreads();

  // ---------------- phase 3: out = P . K (V = K), warp w: dims [128 w, 128 w + 128)
  const int qp = (lane >> 2) & 3, hh = lane >> 4, kr = lane & 3;
  const int dbase = 128 * warp + 16 * (2 * qp + hh);  // this lane's 16 contiguous B dims
  float o[4][8];
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int e = 0; e < 8; ++e) o[j][e] = 0.f;
  const half* pa = ps + arow * G::kPLd;
  pv_steps<Rotor>(desc, 0, pa, 0, 0, kWindow / 4, dbase, kr, centroids, o);
  pv_steps<Rotor>(desc, 0, pa, 0, kWindow / 4, W / 4, dbase, kr, centroids, o);
  store_out(a, k.t, o);
}

template <int W, bool Rotor>
void launch_w(const Args& a, int tokens, cudaStream_t stream) {
  static bool carved = false;
  if (!carved) {
    // Prefer the 96 KiB shared configuration (6 CTAs x ~16 KB per SM); a hint only.
    (void)cudaFuncSetAttribute(attend_kernel<W, Rotor>, cudaFuncAttributePreferredSharedMemoryCarveout,
                               int(cudaSharedmemCarveoutMaxShared));
    (void)cudaGetLastError();
    carved = true;
  }
  constexpr int shared = Geo<W>::kShared + (Rotor ? kRotorTable * sizeof(float) : 0);
  attend_kernel<W, Rotor><<<tokens, kThreads, shared, stream>>>(a);
}

template <bool Rotor>
void launch(const Args& a, int tokens, cudaStream_t stream) {
  if (a.topk) launch_w<kWindow + 512, Rotor>(a, tokens, stream);
  else launch_w<kWindow, Rotor>(a, tokens, stream);
}

void require(bool ok, const char* what) { TORCH_CHECK(ok, "ds41_sparse_attention: ", what); }

void attend(torch::Tensor out, torch::Tensor query, torch::Tensor ring, torch::Tensor values,
            torch::Tensor qsl, torch::Tensor req_slot, std::optional<torch::Tensor> cache,
            std::optional<torch::Tensor> table, torch::Tensor positions, torch::Tensor valid,
            torch::Tensor token_req, std::optional<torch::Tensor> topk, torch::Tensor sink,
            double scale, int64_t block_size, int64_t ratio, int64_t offset, int64_t first,
            int64_t last, int64_t compressed_row_bytes) {
  require(query.scalar_type() == torch::kFloat16 && query.is_contiguous() && query.dim() == 3 &&
              query.size(1) == kHeads && query.size(2) == kDim,
          "query must be FP16 [N, 8, 512]");
  require(out.sizes() == query.sizes() && out.scalar_type() == torch::kFloat16 && out.is_contiguous(),
          "out must match query");
  const bool rotor = ring.scalar_type() == torch::kUInt8;
  const int window_width = rotor ? kRotorRow : kDim;
  require((rotor || ring.scalar_type() == torch::kFloat16) && ring.is_contiguous() && ring.dim() == 3 &&
              ring.size(0) > 0 && ring.size(0) <= std::numeric_limits<int>::max() &&
              ring.size(1) == kRing && ring.size(2) == window_width,
          "ring must be FP16 [S + 1, 128, 512] or RotorQuant uint8 [S + 1, 128, 272]");
  require(values.scalar_type() == ring.scalar_type() && values.is_contiguous() && values.dim() == 2 &&
              values.size(0) == query.size(0) && values.size(1) == window_width,
          "values must match the window-ring format");
  require(reinterpret_cast<uintptr_t>(values.data_ptr()) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(ring.data_ptr()) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(query.data_ptr()) % 16 == 0,
          "16-byte aligned query / values / ring");
  if (rotor)
    require(reinterpret_cast<uintptr_t>(out.data_ptr()) % 4 == 0, "4-byte aligned RotorQuant output");
  require(positions.scalar_type() == torch::kInt64 && valid.scalar_type() == torch::kBool &&
              token_req.scalar_type() == torch::kInt32 && qsl.scalar_type() == torch::kInt32 &&
              req_slot.scalar_type() == torch::kInt32 && positions.is_contiguous() && valid.is_contiguous() &&
              token_req.is_contiguous() && qsl.is_contiguous() && req_slot.is_contiguous(),
          "contiguous index tensors (int64 positions, bool valid, int32 token_req/qsl/req_slot)");
  require(positions.numel() >= query.size(0) && valid.numel() >= query.size(0) &&
              token_req.numel() >= query.size(0),
          "per-token index tensors cover the query rows");
  if (rotor)
    require(qsl.numel() >= 1 && req_slot.numel() >= qsl.numel() - 1,
            "request indexing must cover every request");
  require(sink.scalar_type() == torch::kFloat32 && sink.is_contiguous() && sink.numel() == kHeads,
          "sink FP32 [8]");
  require(0 <= first && first <= last && last <= query.size(0) && last <= std::numeric_limits<int>::max(),
          "token range");
  const auto device = query.device();
  for (const torch::Tensor* t : {&out, &ring, &values, &qsl, &req_slot, &positions, &valid, &token_req, &sink})
    require(t->device() == device, "every tensor on the query's device");
  Args a{};
  a.query = reinterpret_cast<const half*>(query.data_ptr());
  a.ring = ring.data_ptr();
  a.values = values.data_ptr();
  a.qsl = qsl.data_ptr<int32_t>();
  a.req_slot = req_slot.data_ptr<int32_t>();
  a.positions = positions.data_ptr<int64_t>();
  a.valid = reinterpret_cast<const uint8_t*>(valid.data_ptr());
  a.token_req = token_req.data_ptr<int32_t>();
  a.sink = sink.data_ptr<float>();
  a.out = reinterpret_cast<half*>(out.data_ptr());
  a.slots = int(ring.size(0) - 1);
  a.first = int(first);
  a.scale = float(scale);
  a.requests = qsl.numel() - 1;
  require(cache.has_value() == topk.has_value() && table.has_value() == topk.has_value(),
          "compressed keys need cache, table and topk together");
  require(compressed_row_bytes == (topk ? (rotor ? kRotorRow : kLegacyRow) : 0),
          "compressed row format must match the window-ring mode");
  if (topk.has_value()) {
    require(block_size > 0 && block_size <= std::numeric_limits<int>::max() && ratio > 0 &&
                ratio <= block_size && block_size % ratio == 0 && offset >= 0 &&
                offset <= std::numeric_limits<int>::max(),
            "page geometry");
    a.block_size = int(block_size);
    a.ratio = int(ratio);
    a.offset = int(offset);
    require(topk->scalar_type() == torch::kInt32 && topk->is_contiguous() && topk->dim() == 2 &&
                topk->size(0) >= query.size(0) && topk->size(1) == 512 && topk->device() == device,
            "topk must be int32 [N, 512]");
    require(cache->scalar_type() == torch::kUInt8 && cache->is_contiguous() && cache->dim() == 2 &&
                cache->device() == device && reinterpret_cast<uintptr_t>(cache->data_ptr()) % 8 == 0 &&
                offset % 8 == 0,
            "cache must be 8-byte aligned uint8 pages [P, page_bytes]");
    require(table->scalar_type() == torch::kInt32 && table->dim() == 2 && table->stride(1) == 1 &&
                table->device() == device,
            "table int32 [R, pages]");
    a.cache = cache->data_ptr<uint8_t>();
    a.page_bytes = cache->size(1);
    require(a.page_bytes % 8 == 0 && offset + (block_size / ratio) * compressed_row_bytes <= a.page_bytes,
            "page geometry");
    if (rotor)
      require(table->size(0) >= a.requests, "page table must cover every request");
    a.pages = cache->size(0);
    a.table = table->data_ptr<int32_t>();
    a.table_stride = table->stride(0);
    a.table_columns = table->size(1);
    a.topk = topk->data_ptr<int32_t>();
    a.topk_count = 512;
  }
  if (last == first) return;
  const c10::cuda::CUDAGuard guard(device);
  const int tokens = int(last - first);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (rotor) launch<true>(a, tokens, stream);
  else launch<false>(a, tokens, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace ds41_sparse_attention

void ds41_sparse_attend(torch::Tensor out, torch::Tensor query, torch::Tensor ring, torch::Tensor values,
                        torch::Tensor qsl, torch::Tensor req_slot, std::optional<torch::Tensor> cache,
                        std::optional<torch::Tensor> table, torch::Tensor positions, torch::Tensor valid,
                        torch::Tensor token_req, std::optional<torch::Tensor> topk, torch::Tensor sink,
                        double scale, int64_t block_size, int64_t ratio, int64_t offset, int64_t first,
                        int64_t last, int64_t compressed_row_bytes) {
  ds41_sparse_attention::attend(out, query, ring, values, qsl, req_slot, cache, table, positions, valid, token_req,
                                topk, sink, scale, block_size, ratio, offset, first, last, compressed_row_bytes);
}
