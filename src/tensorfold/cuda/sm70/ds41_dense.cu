// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// DeepSeek-V4.1 dense projections on dmv13 segment planes (SM70).
//
// Provenance: 1Cat-vLLM 4d57473d (vllm/models/deepseek_v41/dense_tm.py,
// quantization/gguf_dmv13_dense.py pack(), csrc/sm70_turbomind/ops/
// gguf_dmv13_core.cuh = ./onecat_dmv13_core.cuh unchanged) and the friend's
// overlay patch 112 (ds41mu_files/ds41mu_dmv.cu: dense_mv with MT 8-row tiles
// per weight pass, every row bit-identical to its 8-row dmv13 piece).
//
// TensorFold changes, recorded:
//   * pack_planes: the plane layout of gguf_dmv13_dense.pack() built on the GPU
//     straight from the original Q8_0 / Q5_K row bytes, for any 32-column-
//     aligned column window (TP8 RowParallel slices such as the shared-expert
//     down's 288 columns start inside a Q5_K super-block).
//   * dense_rows: patch 112's kernel with MT in {1, 2, 4} (MT 1 is dmv13's
//     dense_mv order) and a K bound on the activation loads, so K % 128 != 0
//     (K = 288) reads zero activations against zero padded planes instead of
//     requiring DMV_FULLK's K % 128 == 0.
//   * restore_planes: planes -> FP16 [rows, K] for the prefill GEMMs: Q8_0 with
//     dmv13's own decode (exactly the ggml FP16 decode), Q5_K with the exact
//     FP32 sub-block coefficients of the kept super-block headers (exactly the
//     ggml decode; dmv13's FP16-rounded scale/min moved prefill NLL +0.015).
//   * q5k_rows: vocabulary-shard embedding lookup: Q5_K rows of the ids this
//     rank owns decoded with ggml-quants.c's FP32 product rounded once to FP16;
//     rows of foreign ids are +0.
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>

namespace dmvns {
#include "onecat_dmv13_core.cuh"
}

namespace ds41_dense {
using namespace dmvns;

constexpr int T_Q8_0 = 8, T_Q5_K = 13;

__device__ __forceinline__ int rowmap(int lane) {
  return ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
}

__device__ __forceinline__ float f16_bits(const uint8_t* p) {
  const unsigned short b = static_cast<unsigned short>(p[0] | (p[1] << 8));
  return __half2float(__ushort_as_half(b));
}

__device__ __forceinline__ unsigned short to_f16_bits(float v) {
  return __half_as_ushort(__float2half_rn(v));
}

// Q5_K sub-block j's 6-bit scale and min (ggml get_scale_min_k4).
__device__ __forceinline__ void k_scale_min(const uint8_t* q, int j, int& sc, int& mn) {
  if (j < 4) {
    sc = q[j] & 63;
    mn = q[j + 4] & 63;
  } else {
    sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    mn = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

// One thread per (tile, step, lane): codes of 32 weights of one row, plus the
// step's scale (and Q5_K min/high) words. Padded rows/steps are zero.
__global__ void pack_kernel(const uint8_t* __restrict__ raw, long long row_bytes, int rows,
                            int gtype, int first_step, int S, int G, int tiles,
                            uint32_t* __restrict__ codes, uint32_t* __restrict__ high,
                            uint32_t* __restrict__ scale) {
  const int Sp = 4 * G;
  const long long total = static_cast<long long>(tiles) * Sp * 32;
  for (long long id = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; id < total;
       id += static_cast<long long>(gridDim.x) * blockDim.x) {
    const int lane = static_cast<int>(id % 32);
    const int sidx = static_cast<int>((id / 32) % Sp);
    const int t = static_cast<int>(id / (32LL * Sp));
    const int row = t * 32 + rowmap(lane);
    const bool live = row < rows && sidx < S;
    uint8_t q[32];
    unsigned short s16 = 0, m16 = 0;
#pragma unroll
    for (int i = 0; i < 32; ++i) q[i] = 0;
    if (live) {
      const int gs = first_step + sidx;
      const uint8_t* r = raw + static_cast<long long>(row) * row_bytes;
      if (gtype == T_Q8_0) {
        const uint8_t* b = r + static_cast<long long>(gs) * 34;
        s16 = static_cast<unsigned short>(b[0] | (b[1] << 8));
#pragma unroll
        for (int i = 0; i < 32; ++i) q[i] = b[2 + i];
      } else {
        const uint8_t* b = r + static_cast<long long>(gs / 8) * 176;
        const int j = gs % 8, pair = j / 2, upper = j % 2;
        const float d = f16_bits(b), dmin = f16_bits(b + 2);
        int sc, mn;
        k_scale_min(b + 4, j, sc, mn);
        s16 = to_f16_bits(d * static_cast<float>(sc));
        m16 = to_f16_bits(-(dmin * static_cast<float>(mn)));
        const uint8_t* qh = b + 16;
        const uint8_t* qs = b + 48 + 32 * pair;
#pragma unroll
        for (int l = 0; l < 32; ++l) {
          const int lo = upper ? (qs[l] >> 4) : (qs[l] & 0xF);
          q[l] = static_cast<uint8_t>(lo | (((qh[l] >> j) & 1) << 4));
        }
      }
    }
    const int g = sidx / 4, st = sidx % 4;
    const long long sw = ((static_cast<long long>(t) * G + g) * 32 + lane) * 4 + st;
    if (gtype == T_Q8_0) {
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        const uint32_t w = q[4 * c] | (q[4 * c + 1] << 8) | (q[4 * c + 2] << 16) |
                           (static_cast<uint32_t>(q[4 * c + 3]) << 24);
        codes[(((static_cast<long long>(t) * Sp + sidx) * 2 + c / 4) * 32 + lane) * 4 + c % 4] = w;
      }
      scale[sw] = static_cast<uint32_t>(s16) | (static_cast<uint32_t>(s16) << 16);
    } else {
      uint32_t h = 0;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        uint32_t w = 0;
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
          const uint32_t a = q[8 * c + 2 * jj], b2 = q[8 * c + 2 * jj + 1];
          w |= (a & 0xF) << (4 * jj);
          w |= (b2 & 0xF) << (4 * jj + 16);
          h |= ((a >> 4) & 1) << (4 * c + jj);
          h |= ((b2 >> 4) & 1) << (16 + 4 * c + jj);
        }
        codes[((static_cast<long long>(t) * Sp + sidx) * 32 + lane) * 4 + c] = w;
      }
      high[sw] = h;
      scale[sw] = static_cast<uint32_t>(s16) | (static_cast<uint32_t>(m16) << 16);
    }
  }
}

// Q5_K step st of a loaded group with exact FP32 coefficients s = d * sc, m = dmin * mn (ggml
// dequantize_row_q5_K: y = s * q - m in FP32, q * s exact, rounded once to FP16); same bit layout and
// k order as decode<Q5K>, which instead multiplies FP16-rounded s and -m.
__device__ __forceinline__ void decode_q5k_exact(const Ld<Q5K>& L, int st, float s, float m,
                                                 uint32_t (&hw)[16]) {
  const uint32_t MAGIC = 0x64006400u;
  const uint32_t hb = word(L.hi[0], st);
  const uint32_t cr[4] = {L.c[st][0].x, L.c[st][0].y, L.c[st][0].z, L.c[st][0].w};
#pragma unroll
  for (int c = 0; c < 4; ++c) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int sft = 4 * c + j - 4;
      const uint32_t sh = sft >= 0 ? (hb >> (sft >= 0 ? sft : 0)) : (hb << (sft < 0 ? -sft : 0));
      const uint32_t h = lop_or(cr[c] >> (4 * j), 0x000f000fu, lop_or(sh, 0x00100010u, MAGIC));
      const float2 q = __half22float2(h2(h));  // 1024 + q, exact
      hw[4 * c + j] = u32(__floats2half2_rn(fmaf(q.x - 1024.0f, s, -m), fmaf(q.y - 1024.0f, s, -m)));
    }
  }
}

// One warp per (tile, group): dmv13's load + decode of 4 steps (lane L holds row rowmap(L)), transposed
// through shared memory so each store instruction writes two contiguous 256-byte row segments. Q5_K
// decodes with the exact coefficients of the original super-block headers (``head``: 16 bytes d, dmin,
// scales per row and super-block of the window, SB per row, the first at step first_step / 8 * 8).
constexpr int RESTORE_WARPS = 4;
template <int FMT>
__global__ void __launch_bounds__(32 * RESTORE_WARPS)
    restore_kernel(Seg sg, const uint8_t* __restrict__ head, int first_step, int SB, int tile0,
                   int tiles, int rows, int K, int S, int G, half* __restrict__ out, long long ld) {
  __shared__ uint4 buf[RESTORE_WARPS][32][17];  // 32 rows x 128 halves, padded against bank conflicts
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const long long items = static_cast<long long>(tiles) * G;
  for (long long it = static_cast<long long>(blockIdx.x) * RESTORE_WARPS + warp; it < items;
       it += static_cast<long long>(gridDim.x) * RESTORE_WARPS) {
    const int g = static_cast<int>(it % G);
    const int t = tile0 + static_cast<int>(it / G);
    Ld<FMT> A;
    load<FMT>(A, sg, t, g, S, G, lane);
    const int r = rowmap(lane);
#pragma unroll
    for (int st = 0; st < 4; ++st) {
      uint32_t hw[16];
      if constexpr (FMT == Q5K) {
        const int row = t * 32 + r, sidx = g * 4 + st;
        float s = 0.0f, m = 0.0f;
        if (row < rows && sidx < S) {
          const int gs = first_step + sidx;
          const uint8_t* hb =
              head + (static_cast<long long>(row) * SB + gs / 8 - first_step / 8) * 16;
          int sc, mn;
          k_scale_min(hb + 4, gs % 8, sc, mn);
          s = f16_bits(hb) * static_cast<float>(sc);
          m = f16_bits(hb + 2) * static_cast<float>(mn);
        }
        decode_q5k_exact(A, st, s, m, hw);
      } else {
        decode<FMT>(A, st, hw, nullptr);
      }
#pragma unroll
      for (int i = 0; i < 4; ++i)
        buf[warp][r][st * 4 + i] = make_uint4(hw[4 * i], hw[4 * i + 1], hw[4 * i + 2], hw[4 * i + 3]);
    }
    __syncwarp();
#pragma unroll 4
    for (int p = 0; p < 16; ++p) {
      const int rr = 2 * p + (lane >> 4), c = lane & 15;
      const int row = t * 32 + rr, col = g * 128 + c * 8;
      if (row < rows && col < K)
        *reinterpret_cast<uint4*>(out + static_cast<long long>(row - tile0 * 32) * ld + col) =
            buf[warp][rr][c];
    }
    __syncwarp();
  }
}

// ---------------------------------------------------------------- dense_rows (patch 112 + K bound)
template <int FMT, int KW, int TN, int MT>
__device__ __forceinline__ void body_rows(const Seg& sg, int t, bool on, int kslot, int pid, int g0,
                                          int g1, int S, int G, const half* __restrict__ x, int ldx,
                                          int M, int K, uint4* xs, half2* lut,
                                          float (&acc)[MT][8]) {
  constexpr int PARTS = 4 / TN;
  const int lane = threadIdx.x % 32;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  Ld<FMT> A;
  int g = g0 + kslot;
  uint4 X[MT][PARTS];
  auto xload = [&](int gg) {
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int jj = 0; jj < PARTS; ++jj) {
        const int idx = lane + 32 * (pid + TN * jj), c = idx >> 3, row = (idx & 7) + 8 * mt;
        const int col = gg * 128 + c * 8;
        X[mt][jj] = make_uint4(0, 0, 0, 0);
        if (row < M && col < K)
          X[mt][jj] = __ldg(reinterpret_cast<const uint4*>(x + row * ldx + col));
      }
  };
  auto xstore = [&](int b) {
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int jj = 0; jj < PARTS; ++jj)
        xs[(mt * KW + kslot) * 256 + b * 128 + lane + 32 * (pid + TN * jj)] = X[mt][jj];
  };
  auto xsync = [&]() {
    if constexpr (TN > 1)
      asm volatile("bar.sync %0, %1;" ::"r"(kslot + 1), "r"(TN * 32) : "memory");
    else
      __syncwarp();
  };
  if (g < g1) {
    if (on) load<FMT>(A, sg, t, g, S, G, lane);
    xload(g);
  }
  __syncthreads();
  if (g >= g1) return;
  xstore(0);
  xsync();
  int cur = 0;
  while (true) {
    Ld<FMT> B;
    const int gn = g + KW;
    const bool more = gn < g1;
    if (more) {
      if (on) load<FMT>(B, sg, t, gn, S, G, lane);
      xload(gn);
    }
    if (on) {
#pragma unroll
      for (int st = 0; st < 4; ++st) {
        uint4 xa[MT][4];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          const uint4* xp = xs + (mt * KW + kslot) * 256 + cur * 128 + (st * 4) * 8 + r;
#pragma unroll
          for (int j = 0; j < 4; ++j) xa[mt][j] = xp[j * 8];
        }
        uint32_t hw[16];
        decode<FMT>(A, st, hw, lut);
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int mt = 0; mt < MT; ++mt) {
            mma(acc[mt], xa[mt][j].x, xa[mt][j].y, hw[4 * j], hw[4 * j + 1]);
            mma(acc[mt], xa[mt][j].z, xa[mt][j].w, hw[4 * j + 2], hw[4 * j + 3]);
          }
      }
    }
    if (!more) break;
    cur ^= 1;
    xstore(cur);
    xsync();
    A = B;
    g = gn;
  }
}

__device__ __forceinline__ void write_tile(const Seg& sg, int mt, int t, int v, float val, int M) {
  Seg s = sg;
  s.out = sg.out + static_cast<long long>(8 * mt) * sg.out_ld;
  write_out(s, t, v, val, M - 8 * mt);
}

template <int KW, int TN, int FA, int FB, int MT>
__global__ void __launch_bounds__(32 * KW * TN)
    dense_rows(Segs segs, const half* __restrict__ x, int ldx, int M, int K, int S, int G, int split,
               float* ws, int* cnt) {
  constexpr int W = KW * TN;
  extern __shared__ uint4 smem[];
  __shared__ int last;
  const int tg = blockIdx.x, sp = blockIdx.y;
  int si = 0;
#pragma unroll
  for (int i = 1; i < MAXSEG; ++i)
    if (i < segs.nseg && tg >= segs.s[i].tile0) si = i;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int tin = warp % TN, kslot = warp / TN;
  const Seg& sg = segs.s[si];
  const int t = (tg - sg.tile0) * TN + tin;
  const bool on = t * 32 < sg.n;
  const int gps = (G + split - 1) / split;
  const int g0 = min(G, sp * gps), g1 = min(G, g0 + gps);
  uint4* xs = smem;
  half2* lut = reinterpret_cast<half2*>(smem + MT * KW * 256);
  float acc[MT][8] = {};
  if (FA == FB || sg.fmt == FA)
    body_rows<FA, KW, TN, MT>(sg, t, on, kslot, tin, g0, g1, S, G, x, ldx, M, K, xs, lut, acc);
  else
    body_rows<FB, KW, TN, MT>(sg, t, on, kslot, tin, g0, g1, S, G, x, ldx, M, K, xs, lut, acc);
  __syncthreads();
  float* red = reinterpret_cast<float*>(smem);
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int e = 0; e < 8; ++e) red[(mt * W + warp) * 256 + lane * 8 + e] = acc[mt][e];
  __syncthreads();
  const int tbase = (tg - sg.tile0) * TN;
  for (int v = threadIdx.x; v < MT * TN * 256; v += 32 * W) {
    const int mt = v / (TN * 256), vv = v % (TN * 256), i = vv / 256, e = vv % 256;
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < KW; ++k) s += red[(mt * W + k * TN + i) * 256 + e];
    if (split == 1)
      write_tile(sg, mt, tbase + i, e, s, M);
    else
      __stcg(ws + ((static_cast<size_t>(tg) * split + sp) * MT + mt) * TN * 256 + vv, s);
  }
  if (split == 1) return;
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    last = atomicAdd(cnt + tg, 1) == split - 1;
    if (last) __threadfence();
  }
  __syncthreads();
  if (!last) return;
  for (int v = threadIdx.x; v < MT * TN * 256; v += 32 * W) {
    const int mt = v / (TN * 256), vv = v % (TN * 256);
    float pv[8];
#pragma unroll
    for (int p = 0; p < 8; ++p)
      pv[p] = p < split
                  ? __ldcg(ws + ((static_cast<size_t>(tg) * split + p) * MT + mt) * TN * 256 + vv)
                  : 0.f;
    float s = 0.f;
#pragma unroll
    for (int p = 0; p < 8; ++p) s += pv[p];
    write_tile(sg, mt, tbase + vv / 256, vv % 256, s, M);
  }
  if (threadIdx.x == 0) cnt[tg] = 0;
}

template <int KW, int TN, int FA, int FB, int MT>
void launch_rows(const Segs& segs, const half* x, int ldx, int M, int K, int S, int G, int split,
                 float* ws, int* cnt, int tiles, cudaStream_t st) {
  constexpr int W = KW * TN;
  constexpr size_t xs_bytes = static_cast<size_t>(MT) * KW * 256 * 16 + TAB_VECS * 16;
  constexpr size_t red_bytes = static_cast<size_t>(MT) * W * 256 * 4;
  constexpr size_t smem = xs_bytes > red_bytes ? xs_bytes : red_bytes;
  static_assert(smem + 64 <= 96 * 1024, "dense_rows shared memory");
  static const cudaError_t attribute = cudaFuncSetAttribute(
      dense_rows<KW, TN, FA, FB, MT>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
  TORCH_CHECK(attribute == cudaSuccess, "ds41_dense: shared memory attribute: ",
              cudaGetErrorString(attribute));
  dense_rows<KW, TN, FA, FB, MT><<<dim3(tiles, split), 32 * W, smem, st>>>(segs, x, ldx, M, K, S,
                                                                           G, split, ws, cnt);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

bool fmt_ok(int64_t f) { return f == Q5K || f == Q8; }

void check_plane(const torch::Tensor& t, const char* what) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kUInt8 && t.is_contiguous(),
              "ds41_dense: ", what, " must be contiguous CUDA uint8");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, "ds41_dense: ", what,
              " must be 16-byte aligned");
}

int64_t plane_bytes(int64_t fmt, int64_t tiles, int64_t G, int which) {
  // which: 0 codes, 1 high, 2 scale (gguf_dmv13_dense.pack layout)
  if (which == 0) return tiles * G * 4 * 32 * 16 * (fmt == Q8 ? 2 : 1);
  if (which == 1) return fmt == Q5K ? tiles * G * 32 * 16 : 0;
  return tiles * G * 32 * 16;
}

}  // namespace ds41_dense

using namespace ds41_dense;

// Original GGUF rows [rows, row_bytes] -> planes for column steps [first_step, first_step + S).
void ds41_pack_planes(torch::Tensor raw, int64_t gtype, int64_t first_step, int64_t S,
                      torch::Tensor codes, torch::Tensor high, torch::Tensor scale) {
  const c10::cuda::CUDAGuard guard(raw.device());
  TORCH_CHECK(gtype == T_Q8_0 || gtype == T_Q5_K, "ds41_dense: pack admits Q8_0 / Q5_K");
  TORCH_CHECK(raw.is_cuda() && raw.scalar_type() == torch::kUInt8 && raw.dim() == 2 &&
                  raw.stride(1) == 1,
              "ds41_dense: raw rows must be CUDA uint8 [rows, bytes] with unit column stride");
  const int64_t rows = raw.size(0), row_bytes = raw.stride(0);
  TORCH_CHECK(first_step >= 0 && S >= 1, "ds41_dense: column window");
  // Q8_0: 34 bytes per 32-column step; Q5_K: 176 bytes per 8 steps
  const int64_t need = gtype == T_Q8_0 ? (first_step + S) * 34 : (first_step + S + 7) / 8 * 176;
  TORCH_CHECK(need <= raw.size(1), "ds41_dense: column window beyond the row bytes");
  const int64_t fmt = gtype == T_Q8_0 ? Q8 : Q5K;
  const int64_t tiles = (rows + 31) / 32, G = (S + 3) / 4;
  check_plane(codes, "codes");
  check_plane(scale, "scale");
  TORCH_CHECK(codes.numel() == plane_bytes(fmt, tiles, G, 0) &&
                  scale.numel() == plane_bytes(fmt, tiles, G, 2) &&
                  high.numel() == plane_bytes(fmt, tiles, G, 1),
              "ds41_dense: plane sizes do not match the rows/window");
  if (fmt == Q5K) check_plane(high, "high");
  const long long total = tiles * G * 4 * 32;
  const int threads = 256;
  const int blocks = static_cast<int>(std::min<long long>((total + threads - 1) / threads, 65535));
  pack_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      raw.data_ptr<uint8_t>(), row_bytes, static_cast<int>(rows), static_cast<int>(gtype),
      static_cast<int>(first_step), static_cast<int>(S), static_cast<int>(G),
      static_cast<int>(tiles), reinterpret_cast<uint32_t*>(codes.data_ptr()),
      fmt == Q5K ? reinterpret_cast<uint32_t*>(high.data_ptr()) : nullptr,
      reinterpret_cast<uint32_t*>(scale.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Planes rows [32 tile0, 32 tile0 + out.size(0)) -> FP16 out [count, K] (row stride out.stride(0)).
void ds41_restore_planes(torch::Tensor codes, torch::Tensor high, torch::Tensor scale,
                         torch::Tensor head, int64_t fmt, int64_t n, int64_t K, int64_t first_step,
                         int64_t tile0, torch::Tensor out) {
  const c10::cuda::CUDAGuard guard(codes.device());
  TORCH_CHECK(fmt_ok(fmt), "ds41_dense: restore admits Q5_K / Q8_0 planes");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat16 && out.dim() == 2 &&
                  out.stride(1) == 1 && out.size(1) == K && out.stride(0) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0 && K % 32 == 0,
              "ds41_dense: restore output must be 16-byte aligned FP16 [count, K] rows");
  const int64_t S = K / 32, G = (S + 3) / 4, tiles_all = (n + 31) / 32;
  const int64_t count = out.size(0);
  TORCH_CHECK(tile0 >= 0 && 32 * tile0 + count <= n && count >= 1, "ds41_dense: restore rows");
  TORCH_CHECK(codes.numel() == plane_bytes(fmt, tiles_all, G, 0) &&
                  scale.numel() == plane_bytes(fmt, tiles_all, G, 2) &&
                  high.numel() == plane_bytes(fmt, tiles_all, G, 1),
              "ds41_dense: plane sizes");
  const int64_t SB = (first_step + S + 7) / 8 - first_step / 8;
  if (fmt == Q5K)
    TORCH_CHECK(head.is_cuda() && head.scalar_type() == torch::kUInt8 && head.is_contiguous() &&
                    first_step >= 0 && head.numel() == n * SB * 16,
                "ds41_dense: Q5_K restore needs the [n, super-blocks, 16] header plane");
  Seg sg{};
  sg.codes = reinterpret_cast<const uint4*>(codes.data_ptr());
  sg.high = high.numel() ? reinterpret_cast<const uint4*>(high.data_ptr()) : nullptr;
  sg.scale = reinterpret_cast<const uint4*>(scale.data_ptr());
  sg.n = static_cast<int>(n);
  sg.fmt = static_cast<int>(fmt);
  const uint8_t* hp = fmt == Q5K ? head.data_ptr<uint8_t>() : nullptr;
  const int64_t tiles = (count + 31) / 32;
  const int rows = static_cast<int>(32 * tile0 + count);
  const long long items = tiles * G;
  const int threads = 32 * RESTORE_WARPS;
  const int blocks = static_cast<int>(
      std::min<long long>((items + RESTORE_WARPS - 1) / RESTORE_WARPS, 65535));
  auto st = at::cuda::getCurrentCUDAStream();
  half* o = reinterpret_cast<half*>(out.data_ptr());
  if (fmt == Q8)
    restore_kernel<Q8><<<blocks, threads, 0, st>>>(sg, hp, int(first_step), int(SB), int(tile0),
                                                  int(tiles), rows, int(K), int(S), int(G), o,
                                                  out.stride(0));
  else
    restore_kernel<Q5K><<<blocks, threads, 0, st>>>(sg, hp, int(first_step), int(SB), int(tile0),
                                                   int(tiles), rows, int(K), int(S), int(G), o,
                                                   out.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// out[i][:M] = x[:M] @ W_i.T for 1..4 same-input segments, M <= 8 * mt (mt 1, 2, 4).
void ds41_dense_rows(torch::Tensor x, std::vector<torch::Tensor> codes,
                     std::vector<torch::Tensor> high, std::vector<torch::Tensor> scale,
                     std::vector<torch::Tensor> out, std::vector<int64_t> fmt,
                     std::vector<int64_t> n, int64_t K, int64_t split, int64_t warps,
                     torch::Tensor ws, torch::Tensor cnt, int64_t tp, int64_t mt) {
  const c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(mt == 1 || mt == 2 || mt == 4, "ds41_dense: tiles must be 1, 2 or 4");
  const int M = static_cast<int>(x.size(0));
  TORCH_CHECK(x.dim() == 2 && M >= 1 && M <= 8 * mt, "ds41_dense: M must fit the tiles");
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16 && x.stride(1) == 1 &&
                  x.size(1) == K && x.stride(0) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "ds41_dense: x must be 16-byte aligned FP16 [M, K] rows");
  TORCH_CHECK(K % 32 == 0 && K >= 32, "ds41_dense: K must be a positive multiple of 32");
  Segs segs{};
  segs.nseg = static_cast<int>(codes.size());
  TORCH_CHECK(segs.nseg >= 1 && segs.nseg <= MAXSEG && high.size() == codes.size() &&
                  scale.size() == codes.size() && out.size() == codes.size() &&
                  fmt.size() == codes.size() && n.size() == codes.size(),
              "ds41_dense: 1..4 segments");
  const int S = static_cast<int>(K / 32);
  const int G = (S + 3) / 4;
  int tiles = 0;
  for (int i = 0; i < segs.nseg; ++i) {
    Seg& s = segs.s[i];
    TORCH_CHECK(fmt_ok(fmt[i]), "ds41_dense: segments are Q5_K / Q8_0 planes");
    const int64_t T = (n[i] + 31) / 32;
    TORCH_CHECK(codes[i].numel() == plane_bytes(fmt[i], T, G, 0) &&
                    scale[i].numel() == plane_bytes(fmt[i], T, G, 2) &&
                    high[i].numel() == plane_bytes(fmt[i], T, G, 1),
                "ds41_dense: segment plane sizes");
    TORCH_CHECK(out[i].is_cuda() && out[i].scalar_type() == torch::kFloat16 &&
                    out[i].dim() == 2 && out[i].size(0) >= M && out[i].size(1) == n[i] &&
                    out[i].stride(1) == 1,
                "ds41_dense: segment output [M, n] FP16");
    s.codes = reinterpret_cast<const uint4*>(codes[i].data_ptr());
    s.high = high[i].numel() ? reinterpret_cast<const uint4*>(high[i].data_ptr()) : nullptr;
    s.scale = reinterpret_cast<const uint4*>(scale[i].data_ptr());
    s.out = reinterpret_cast<half*>(out[i].data_ptr());
    s.out_ld = out[i].stride(0);
    s.n = static_cast<int>(n[i]);
    s.fmt = static_cast<int>(fmt[i]);
    s.tile0 = tiles;
    tiles += static_cast<int>((s.n + 32 * tp - 1) / (32 * tp));
  }
  segs.main_tiles = tiles;
  TORCH_CHECK(split >= 1 && split <= 8, "ds41_dense: split 1..8");
  TORCH_CHECK(split == 1 || (ws.scalar_type() == torch::kFloat32 &&
                             ws.numel() >= int64_t(tiles) * split * mt * tp * 256),
              "ds41_dense: workspace too small");
  TORCH_CHECK(cnt.scalar_type() == torch::kInt32 && cnt.numel() >= tiles, "ds41_dense: counters");
  int fa = segs.s[0].fmt, fb = fa;
  for (int i = 1; i < segs.nseg; ++i)
    if (segs.s[i].fmt != fa) fb = segs.s[i].fmt;
  auto st = at::cuda::getCurrentCUDAStream();
  const half* xp = reinterpret_cast<const half*>(x.data_ptr());
  const int ldx = static_cast<int>(x.stride(0));
  float* wsp = ws.data_ptr<float>();
  int* cp = cnt.data_ptr<int>();
#define CFG(A, B, X, Y, T)                                                                        \
  if (warps == A && tp == B && fa == X && fb == Y && mt == T)                                     \
    return launch_rows<A, B, X, Y, T>(segs, xp, ldx, M, int(K), S, G, int(split), wsp, cp, tiles, \
                                      st);
#define FMTS(A, B, T)     \
  CFG(A, B, Q5K, Q5K, T); \
  CFG(A, B, Q8, Q8, T);   \
  CFG(A, B, Q5K, Q8, T);  \
  CFG(A, B, Q8, Q5K, T);
  FMTS(4, 1, 1);
  FMTS(8, 1, 1);
  FMTS(4, 1, 2);
  FMTS(8, 1, 2);
  FMTS(4, 1, 4);
#undef FMTS
#undef CFG
  TORCH_CHECK(false, "ds41_dense: unsupported warps/tp/formats/tiles ", warps, "/", tp, "/", fa,
              "/", fb, "/", mt);
}

// Vocabulary-shard Q5_K lookup: out[i] = decode(table[ids[i] - first]) if owned, else +0.
__global__ void q5k_rows_kernel(const uint8_t* __restrict__ table, long long row_bytes, int rows,
                                const int64_t* __restrict__ ids, long long first, int cols,
                                half* __restrict__ out) {
  const int token = blockIdx.y;
  const long long local = ids[token] - first;
  const bool own = local >= 0 && local < rows;
  half2* o = reinterpret_cast<half2*>(out + static_cast<long long>(token) * cols);
  // each thread decodes 2 consecutive values of one 256-value super-block
  for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < cols / 2; p += gridDim.x * blockDim.x) {
    if (!own) {
      o[p] = __halves2half2(__ushort_as_half(0), __ushort_as_half(0));
      continue;
    }
    const int k = 2 * p, blk = k / 256, in = k % 256, j = in / 32, l = in % 32;
    const uint8_t* b = table + local * row_bytes + static_cast<long long>(blk) * 176;
    const float d = f16_bits(b), dmin = f16_bits(b + 2);
    int sc, mn;
    k_scale_min(b + 4, j, sc, mn);
    const float d1 = d * static_cast<float>(sc), m1 = dmin * static_cast<float>(mn);
    const uint8_t* qs = b + 48 + 32 * (j / 2);
    const uint8_t* qh = b + 16;
    float v[2];
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const int q = ((j & 1) ? (qs[l + e] >> 4) : (qs[l + e] & 0xF)) | (((qh[l + e] >> j) & 1) << 4);
      v[e] = d1 * static_cast<float>(q) - m1;  // ggml dequantize_row_q5_K
    }
    o[p] = __floats2half2_rn(v[0], v[1]);
  }
}

void ds41_q5k_rows(torch::Tensor table, torch::Tensor ids, int64_t first, torch::Tensor out) {
  const c10::cuda::CUDAGuard guard(table.device());
  TORCH_CHECK(table.is_cuda() && table.scalar_type() == torch::kUInt8 && table.dim() == 2 &&
                  table.is_contiguous(),
              "ds41_dense: Q5_K table must be contiguous CUDA uint8 [rows, bytes]");
  TORCH_CHECK(ids.is_cuda() && ids.scalar_type() == torch::kInt64 && ids.dim() == 1 &&
                  ids.is_contiguous(),
              "ds41_dense: ids must be contiguous CUDA int64 [T]");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat16 && out.dim() == 2 &&
                  out.is_contiguous() && out.size(0) == ids.size(0) && out.size(1) % 256 == 0 &&
                  table.size(1) == out.size(1) / 256 * 176,
              "ds41_dense: out must be contiguous FP16 [T, cols] matching the table rows");
  const int T = static_cast<int>(ids.size(0));
  if (T == 0) return;
  TORCH_CHECK(T <= 65535, "ds41_dense: at most 65535 tokens per lookup");
  const int cols = static_cast<int>(out.size(1));
  const int threads = 256;
  const dim3 grid((cols / 2 + threads - 1) / threads, T);
  q5k_rows_kernel<<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      table.data_ptr<uint8_t>(), table.size(1), static_cast<int>(table.size(0)),
      ids.data_ptr<int64_t>(), first, cols, reinterpret_cast<half*>(out.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
