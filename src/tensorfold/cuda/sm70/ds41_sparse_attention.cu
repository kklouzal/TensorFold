// SPDX-License-Identifier: Apache-2.0
// DeepSeek-V4.1 fused sparse attention over the patch-87 window rings and packed FP4 rows, SM70.
//
// Provenance: the friend's overlay patch 181 (ds41_pf_attn, SPDX Apache-2.0), snapshot
//   lp-snap-c1es8-numa-admit2-imgtext-share2-fix5-ram4-stepbud-ram3-memfix-ctx1m-ppovl-opt2-pfx-1-eng-pfxfix4,
//   patches/ds41mu_files/ds41mu_attn2.cu (648 lines; his TP4 = 16 query heads per rank).
// TensorFold changes (TP8, 2026-11-21):
//   - 8 query heads per rank (kHeads 16 -> 8): the m8n8k4 A operand is exactly the 8 heads, so the "high"
//     (heads 8..15) MMAs, accumulators and softmax slots are dropped; fragment layouts, key streaming, the
//     dequantization and the softmax arithmetic are his, unchanged.
//   - 80 registers and ~16 KB shared memory per CTA (ptxas, sm_70): 6 CTAs (24 warps) per SM instead of 3.
//   - pybind11 module (TensorFold JIT build) instead of TORCH_LIBRARY; the split mode's scratch is passed in.
//
// Contract (patch 87 _attend: gather + bmm Q.K^T FP32 + _sink_softmax + bmm P.K with V = K):
//   out[t, h, :] = FP16( sum_k p[t, h, k] * key[t, k, :] ),  p = FP16(softmax over the 128 window keys and the
//   512 selected compressed keys (unused keys -inf) with the per-head sink in the denominator), scores
//   FP32(q . key) * scale. Window keys come from this forward's FP16 latents (positions >= the request's first
//   position in this step) or the request's slot ring; compressed keys are the packed rows decoded exactly as
//   kv_pack.unpack_c_row (E2M1 x E4M3 products are exact in FP16). The FP16 key values and the FP16
//   probability boundary are the reference's; FP32 summation orders differ (dot, softmax sum, P.K).
//   One token per CTA (row-wise, batch invariant), deterministic.
#include <cuda_fp16.h>
#include <cstdint>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace ds41_sparse_attention {

constexpr int kHeads = 8, kDim = 512, kWindow = 128, kRing = 128, kWarps = 4, kThreads = 128;
constexpr int kQLd = kDim + 8;  // halves per shared query row

struct Args {
  const half* query;      // [N, 8, 512]
  const half* ring;       // [S + 1, 128, 512]
  const half* values;     // [N, 512] this forward's window latents
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
  int64_t table_stride, page_bytes;
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
static_assert(Geo<640>::kShared <= 48 * 1024, "no dynamic shared-memory opt-in needed");
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
  if (e == 15u && m == 7u) return 0x7e00u;
  if (e != 0u) return sign | ((e + 8u) << 10) | (m << 7);
  if (m == 0u) return sign;
  // subnormal E4M3: m * 2^-9 (normal in FP16)
  const half h = __float2half_rn(float(m) * 0.001953125f);
  return sign | __half_as_ushort(h);
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

__device__ __forceinline__ uint32_t splat_scale(const uint8_t* row, int group) {
  const uint32_t bits = e4m3_half_bits(__ldg(row + 256 + group));
  return bits | (bits << 16);
}

// 0 = unused key; FP16 row pointer; packed row pointer | 1.
__device__ __forceinline__ uint64_t describe(const Args& a, int key, int64_t t, int64_t r,
                                             int64_t position, bool live, int64_t s0,
                                             int64_t start, int64_t slot) {
  if (!live) return 0;
  if (key < kWindow) {
    const int64_t at = position - kWindow + 1 + key;
    if (at < 0) return 0;
    const half* src = at >= start ? a.values + (s0 + at - start) * kDim
                                  : a.ring + (slot * kRing + at % kRing) * kDim;
    return reinterpret_cast<uint64_t>(src);
  }
  const int64_t index = a.topk[t * a.topk_count + key - kWindow];
  if (index < 0 || index >= (position + 1) / a.ratio) return 0;
  const int rows = a.block_size / a.ratio;
  const int64_t page = a.table[r * a.table_stride + index / rows];
  if (page < 0) return 0;
  const uint8_t* row = a.cache + page * a.page_bytes + a.offset + (index % rows) * 288;
  return reinterpret_cast<uint64_t>(row) | 1ull;
}

// Scores of one 32-key tile (8 heads x 32 keys, C layout) for this lane's key d.
__device__ __forceinline__ void tile_scores(bool window, uint64_t d, const half* qa, float (&lo)[8]) {
#pragma unroll
  for (int e = 0; e < 8; ++e) lo[e] = 0.f;
  if (window) {  // FP16 rows
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
#pragma unroll 2
    for (int c = 0; c < kDim / 16; ++c) {  // 16 dims (one scale group) per chunk
      const uint2 v = nxt;
      if (c + 1 < kDim / 16) nxt = d ? __ldg(reinterpret_cast<const uint2*>(row) + c + 1) : make_uint2(0, 0);
      const uint32_t s2 = d ? splat_scale(row, c) : 0u;
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

// out[8 x 128-dim warp slice] += P[:, keys 4 ks0 .. 4 ks1) . K (P columns pcol0-based).
__device__ __forceinline__ void pv_steps(const uint64_t* desc, int key0, const half* pa, int pcol0, int ks0,
                                         int ks1, int dbase, int kr, float (&o)[4][8]) {
#pragma unroll 2
  for (int ks = ks0; ks < ks1; ++ks) {
    const int key = 4 * ks + kr;
    const uint64_t d = desc[key - key0];
    uint32_t b[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    if (key < kWindow) {
      if (d) {
        const uint4* src = reinterpret_cast<const uint4*>(reinterpret_cast<const half*>(d) + dbase);
        const uint4 v0 = __ldg(src), v1 = __ldg(src + 1);
        b[0] = v0.x; b[1] = v0.y; b[2] = v0.z; b[3] = v0.w;
        b[4] = v1.x; b[5] = v1.y; b[6] = v1.z; b[7] = v1.w;
      }
    } else if (d) {
      const uint8_t* row = reinterpret_cast<const uint8_t*>(d & ~1ull);
      const uint2 v = __ldg(reinterpret_cast<const uint2*>(row + dbase / 2));
      const uint32_t s2 = splat_scale(row, dbase / 16);
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

__device__ __forceinline__ Token token_of(const Args& a, int64_t t) {
  Token k;
  k.t = t;
  k.r = a.token_req[t];
  k.position = a.positions[t];
  k.live = a.valid[t] != 0 && k.r >= 0;
  const int64_t rr = k.r >= 0 ? k.r : 0;
  k.s0 = a.qsl[rr];
  k.start = a.positions[k.s0];
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

// C element (head, dim) of P.K accumulator o[j][e] for this lane (same mapping in the split partials).
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

template <int W>
__global__ __launch_bounds__(kThreads, 6) void attend_kernel(Args a) {
  using G = Geo<W>;
  extern __shared__ __align__(16) unsigned char smem[];
  half* qs = reinterpret_cast<half*>(smem);   // [8][kQLd]   (phase 1)
  half* ps = reinterpret_cast<half*>(smem);   // [8][kPLd]   (phases 2-3, same bytes)
  float* red = reinterpret_cast<float*>(smem + G::kBuf);  // [2][kWarps][8]
  uint64_t* desc = reinterpret_cast<uint64_t*>(smem + G::kBuf + G::kRed);  // [W]
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const Token k = token_of(a, a.first + blockIdx.x);
  load_query(a, k.t, qs);
  for (int i = threadIdx.x; i < W; i += kThreads)
    desc[i] = describe(a, i, k.t, k.r, k.position, k.live, k.s0, k.start, k.slot);
  __syncthreads();

  // ---------------- phase 1: scores, 8 heads x 32 keys per tile, in registers
  const int arow = (lane & 3) + ((lane >> 4) << 2);                      // A row = head
  const int bkey = ((lane & 12) << 1) + (lane & 3) + ((lane >> 4) << 2);  // B column = key in tile
  float sc[G::kTilesPerWarp][8];
#pragma unroll
  for (int i = 0; i < G::kTilesPerWarp; ++i) {
    const int tile = warp + i * kWarps;
    tile_scores(tile * 32 < kWindow, desc[tile * 32 + bkey], qs + arow * kQLd, sc[i]);
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
  pv_steps(desc, 0, pa, 0, 0, kWindow / 4, dbase, kr, o);
  pv_steps(desc, 0, pa, 0, kWindow / 4, W / 4, dbase, kr, o);
  store_out(a, k.t, o);
}

// ---------------- split mode (few tokens): S = W / 128 CTAs per token
constexpr int kSplitKeys = 128;
constexpr int kSPLd = kSplitKeys + 8;

// 1: raw scaled scores (-inf for unused keys) of this split's 128 keys -> scores[t][8][W].
template <int W>
__global__ __launch_bounds__(kThreads) void split_scores_kernel(Args a, float* scores) {
  __shared__ __align__(16) half qs[kHeads * kQLd];
  __shared__ uint64_t desc[kSplitKeys];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int split = blockIdx.y, key0 = split * kSplitKeys;
  const Token k = token_of(a, a.first + blockIdx.x);
  load_query(a, k.t, qs);
  for (int i = threadIdx.x; i < kSplitKeys; i += kThreads)
    desc[i] = describe(a, key0 + i, k.t, k.r, k.position, k.live, k.s0, k.start, k.slot);
  __syncthreads();
  const int arow = (lane & 3) + ((lane >> 4) << 2);
  const int bkey = ((lane & 12) << 1) + (lane & 3) + ((lane >> 4) << 2);
  const int tile = split * kWarps + warp;  // 32 keys
  float lo[8];
  tile_scores(tile * 32 < kWindow, desc[warp * 32 + bkey], qs + arow * kQLd, lo);
  const int crow = (lane & 1) + ((lane >> 4) << 2);
  const int cbase = ((lane & 12) << 1) + (lane & 2);
  float* row = scores + (int64_t(blockIdx.x) * kHeads) * W;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const int head = crow + ((e & 2) ? 2 : 0);
    const int col = cbase + (e & 4) + (e & 1);
    row[head * W + tile * 32 + col] = desc[warp * 32 + col] != 0 ? __fmul_rn(lo[e], a.scale) : -INFINITY;
  }
}

// 2: softmax over the token's W scores (every split recomputes m and the sum in the
// same order), FP16 probabilities of its own keys, P.K over them -> partial[t][split][8][512].
template <int W>
__global__ __launch_bounds__(kThreads) void split_pv_kernel(Args a, const float* scores, float* partial) {
  __shared__ __align__(16) half ps[kHeads * kSPLd];
  __shared__ uint64_t desc[kSplitKeys];
  __shared__ float stat[2][kHeads];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int split = blockIdx.y, key0 = split * kSplitKeys;
  const Token k = token_of(a, a.first + blockIdx.x);
  for (int i = threadIdx.x; i < kSplitKeys; i += kThreads)
    desc[i] = describe(a, key0 + i, k.t, k.r, k.position, k.live, k.s0, k.start, k.slot);
  const float* row = scores + (int64_t(blockIdx.x) * kHeads) * W;
  constexpr int kHeadsPerWarp = kHeads / kWarps;  // warp w: heads 2w, 2w + 1; lanes stride the W scores
  for (int h = kHeadsPerWarp * warp; h < kHeadsPerWarp * (warp + 1); ++h) {
    float m = -INFINITY;
    for (int c = lane; c < W; c += 32) m = fmaxf(m, row[h * W + c]);
#pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    m = fmaxf(m, a.sink[h]);
    float s = 0.f;
    for (int c = lane; c < W; c += 32) s += ex2_approx(__fmul_rn(__fsub_rn(row[h * W + c], m), kLog2e));
#pragma unroll
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0) {
      stat[0][h] = m;
      stat[1][h] = __fadd_rn(s, ex2_approx(__fmul_rn(__fsub_rn(a.sink[h], m), kLog2e)));
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < kHeads * kSplitKeys; i += kThreads) {
    const int h = i / kSplitKeys, c = i % kSplitKeys;
    const float x = ex2_approx(__fmul_rn(__fsub_rn(row[h * W + key0 + c], stat[0][h]), kLog2e));
    ps[h * kSPLd + c] = __float2half_rn(div_full(x, stat[1][h]));
  }
  __syncthreads();
  const int arow = (lane & 3) + ((lane >> 4) << 2);
  const int qp = (lane >> 2) & 3, hh = lane >> 4, kr = lane & 3;
  const int dbase = 128 * warp + 16 * (2 * qp + hh);
  float o[4][8];
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int e = 0; e < 8; ++e) o[j][e] = 0.f;
  pv_steps(desc, key0, ps + arow * kSPLd, key0, key0 / 4, (key0 + kSplitKeys) / 4, dbase, kr, o);
  // partial[t][split][head][dim], FP32 (same element mapping as store_out)
  float* dst = partial + ((int64_t(blockIdx.x) * gridDim.y + split) * kHeads) * kDim;
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int e = 0; e < 8; e += 2)
      *reinterpret_cast<float2*>(dst + out_head(lane, e) * kDim + out_dim(lane, warp, j, e)) =
          make_float2(o[j][e], o[j][e + 1]);
}

// 3: out = FP16(sum over splits in order).
__global__ __launch_bounds__(256) void split_combine_kernel(Args a, const float* partial, int splits) {
  const int64_t t = a.first + blockIdx.x;
  const float* src = partial + int64_t(blockIdx.x) * splits * kHeads * kDim;
  for (int i = threadIdx.x; i < kHeads * kDim / 2; i += blockDim.x) {
    float x = 0.f, y = 0.f;
    for (int s = 0; s < splits; ++s) {
      const float2 v = *reinterpret_cast<const float2*>(src + s * kHeads * kDim + 2 * i);
      x = s ? x + v.x : v.x;
      y = s ? y + v.y : v.y;
    }
    *reinterpret_cast<__half2*>(a.out + t * kHeads * kDim + 2 * i) = __floats2half2_rn(x, y);
  }
}

template <int W>
void launch_w(const Args& a, int tokens, cudaStream_t stream) {
  static bool carved = false;
  if (!carved) {
    // Prefer the 96 KiB shared configuration (6 CTAs x ~16 KB per SM); a hint only.
    (void)cudaFuncSetAttribute(attend_kernel<W>, cudaFuncAttributePreferredSharedMemoryCarveout,
                               int(cudaSharedmemCarveoutMaxShared));
    (void)cudaGetLastError();
    carved = true;
  }
  attend_kernel<W><<<tokens, kThreads, Geo<W>::kShared, stream>>>(a);
}

void launch(const Args& a, int tokens, cudaStream_t stream) {
  if (a.topk) launch_w<kWindow + 512>(a, tokens, stream);
  else launch_w<kWindow>(a, tokens, stream);
}

constexpr int kSplits = (kWindow + 512) / kSplitKeys;

void require(bool ok, const char* what) { TORCH_CHECK(ok, "ds41_sparse_attention: ", what); }

// split_scratch: FP32 >= tokens * (8 * 640 + 5 * 8 * 512) floats when split mode runs (else unused).
void attend(torch::Tensor out, torch::Tensor query, torch::Tensor ring, torch::Tensor values,
            torch::Tensor qsl, torch::Tensor req_slot, std::optional<torch::Tensor> cache,
            std::optional<torch::Tensor> table, torch::Tensor positions, torch::Tensor valid,
            torch::Tensor token_req, std::optional<torch::Tensor> topk, torch::Tensor sink,
            double scale, int64_t block_size, int64_t ratio, int64_t offset, int64_t first,
            int64_t last, bool split, std::optional<torch::Tensor> split_scratch) {
  require(query.scalar_type() == torch::kFloat16 && query.is_contiguous() && query.dim() == 3 &&
              query.size(1) == kHeads && query.size(2) == kDim,
          "query must be FP16 [N, 8, 512]");
  require(out.sizes() == query.sizes() && out.scalar_type() == torch::kFloat16 && out.is_contiguous(),
          "out must match query");
  require(ring.scalar_type() == torch::kFloat16 && ring.is_contiguous() && ring.dim() == 3 &&
              ring.size(1) == kRing && ring.size(2) == kDim,
          "ring must be FP16 [S + 1, 128, 512]");
  require(values.scalar_type() == torch::kFloat16 && values.is_contiguous() && values.dim() == 2 &&
              values.size(0) == query.size(0) && values.size(1) == kDim,
          "values must be FP16 [N, 512]");
  require(reinterpret_cast<uintptr_t>(values.data_ptr()) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(ring.data_ptr()) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(query.data_ptr()) % 16 == 0,
          "16-byte aligned query / values / ring");
  require(positions.scalar_type() == torch::kInt64 && valid.scalar_type() == torch::kBool &&
              token_req.scalar_type() == torch::kInt32 && qsl.scalar_type() == torch::kInt32 &&
              req_slot.scalar_type() == torch::kInt32 && positions.is_contiguous() && valid.is_contiguous() &&
              token_req.is_contiguous() && qsl.is_contiguous() && req_slot.is_contiguous(),
          "contiguous index tensors (int64 positions, bool valid, int32 token_req/qsl/req_slot)");
  require(positions.numel() >= query.size(0) && valid.numel() >= query.size(0) &&
              token_req.numel() >= query.size(0),
          "per-token index tensors cover the query rows");
  require(sink.scalar_type() == torch::kFloat32 && sink.is_contiguous() && sink.numel() == kHeads,
          "sink FP32 [8]");
  require(0 <= first && first <= last && last <= query.size(0), "token range");
  const auto device = query.device();
  for (const torch::Tensor* t : {&out, &ring, &values, &qsl, &req_slot, &positions, &valid, &token_req, &sink})
    require(t->device() == device, "every tensor on the query's device");
  Args a{};
  a.query = reinterpret_cast<const half*>(query.data_ptr());
  a.ring = reinterpret_cast<const half*>(ring.data_ptr());
  a.values = reinterpret_cast<const half*>(values.data_ptr());
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
  a.block_size = int(block_size);
  a.ratio = int(ratio);
  a.offset = int(offset);
  if (topk.has_value()) {
    require(cache.has_value() && table.has_value(), "compressed keys need cache and table");
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
    require(a.page_bytes % 8 == 0 && ratio > 0 && block_size % ratio == 0 &&
                offset + (block_size / ratio) * 288 <= a.page_bytes,
            "page geometry");
    a.table = table->data_ptr<int32_t>();
    a.table_stride = table->stride(0);
    a.topk = topk->data_ptr<int32_t>();
    a.topk_count = 512;
  } else {
    require(!split, "split mode needs compressed keys");
  }
  if (last == first) return;
  const c10::cuda::CUDAGuard guard(device);
  const int tokens = int(last - first);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (split) {
    constexpr int W = kWindow + 512;
    const int64_t need = int64_t(tokens) * (kHeads * W + kSplits * kHeads * kDim);
    require(split_scratch.has_value() && split_scratch->scalar_type() == torch::kFloat32 &&
                split_scratch->is_contiguous() && split_scratch->numel() >= need &&
                split_scratch->device() == device,
            "split scratch FP32 [tokens * (8 * 640 + 5 * 8 * 512)]");
    float* scores = split_scratch->data_ptr<float>();
    float* partial = scores + int64_t(tokens) * kHeads * W;
    const dim3 grid(tokens, kSplits);
    split_scores_kernel<W><<<grid, kThreads, 0, stream>>>(a, scores);
    split_pv_kernel<W><<<grid, kThreads, 0, stream>>>(a, scores, partial);
    split_combine_kernel<<<tokens, 256, 0, stream>>>(a, partial, kSplits);
  } else {
    launch(a, tokens, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace ds41_sparse_attention

void ds41_sparse_attend(torch::Tensor out, torch::Tensor query, torch::Tensor ring, torch::Tensor values,
                        torch::Tensor qsl, torch::Tensor req_slot, std::optional<torch::Tensor> cache,
                        std::optional<torch::Tensor> table, torch::Tensor positions, torch::Tensor valid,
                        torch::Tensor token_req, std::optional<torch::Tensor> topk, torch::Tensor sink,
                        double scale, int64_t block_size, int64_t ratio, int64_t offset, int64_t first,
                        int64_t last, bool split, std::optional<torch::Tensor> split_scratch) {
  ds41_sparse_attention::attend(out, query, ring, values, qsl, req_slot, cache, table, positions, valid, token_req,
                                topk, sink, scale, block_size, ratio, offset, first, last, split, split_scratch);
}
