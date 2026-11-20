// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// DeepSeek-V4.1 routed-expert integer-dot operators, adapted from
// 1CatAI/1Cat-vLLM 4d57473db67e csrc/sm70_turbomind/ops/gguf_dp4a_sm70.cu
// (quantize_q8, v41_swiglu_q8, v41_shared_swiglu, gate_up, v41_gate_reuse/v41_reuse_plan,
// raw_down_unroute). Changes, none in arithmetic:
//  * only the V4.1 instantiations (IQ2_XXS 16, IQ2_XS 17, IQ1_M 29; FP16
//    unactivated gate/up; Q8_1 tiled down; adaptive reuse for IQ1_M only);
//  * a route whose expert id is outside [0, E) traps instead of reading
//    outside the bank (upstream checked this only in v41_reuse_plan);
//  * down has unit route weights built in: upstream's caller passes ones, and
//    x * 1.f == x (also inside a contracted FMA), so the sums are identical;
//  * host checks also require the alignment the kernels' word loads assume.
// dequantize_kernel is TensorFold-owned, written from ggml-quants.c's
// reference dequantize_row_* (MIT) with explicit FP32 rounding per operation.
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdint>
#include "onecat_ds41_dp4a.cuh"

namespace {
using tensorfold::onecat_ds41::Q8_1;
using tensorfold::onecat_ds41::V41IntegerGroup;
template <int Type>
using LatticeDot = tensorfold::onecat_ds41::V41LatticeDot<Type>;
using tensorfold::onecat_ds41::quantize_q8_1_warp;

constexpr int kHidden = 5120, kRows = 576, kTop = 6;

__global__ void quantize_q8(Q8_1* out, const half* input, int k) {
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  if (column >= k) return;
  const float value = __half2float(input[int64_t(blockIdx.y) * k + column]);
  const int group = blockIdx.y * (k / 32) + column / 32;
  quantize_q8_1_warp(out + group, value);
}

__device__ float v41_swiglu(float gate, float up, float probability) {
  gate = gate > 10.f ? 10.f : gate;
  up = up > 10.f ? 10.f : up;
  up = up < -10.f ? -10.f : up;
  const float activated = __fdiv_rn(gate, __fadd_rn(1.f, expf(-gate)));
  return __fmul_rn(__fmul_rn(activated, up), probability);
}

__global__ void v41_swiglu_q8(Q8_1* out, const half* projection,
                              const float* probabilities) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= 576)
    return;  // Entire inactive warps; the 32-lane Q8 group stays full.
  const int route = blockIdx.y;
  const float gate = __half2float(projection[int64_t(route) * 1152 + col]);
  const float up = __half2float(projection[int64_t(route) * 1152 + 576 + col]);
  // Retain the FP16 down-input boundary before computing Q8_1 bytes.
  const float value =
      __half2float(__float2half_rn(v41_swiglu(gate, up, probabilities[route])));
  quantize_q8_1_warp(out + route * 18 + col / 32, value);
}

__global__ void v41_shared_swiglu(half* out, const half* gate, const half* up,
                                  int64_t count) {
  const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count)
    out[index] = __float2half_rn(
        v41_swiglu(__half2float(gate[index]), __half2float(up[index]), 1.f));
}

template <class Index>
__device__ __forceinline__ void require_expert(Index expert, int experts) {
  if (expert < 0 || expert >= experts) __trap();
}

// Upstream gate_up<Type, /*Activated=*/false, Index, /*Lanes=*/16> with the
// optional RouteGuard of the adaptive launch.
template <int Type, class Index, bool RouteGuard = false>
__global__ void gate_up(half* output, const Q8_1* activation, const Index* ids,
                        const uint8_t* gate, const uint8_t* up, int n, int k,
                        int stride, int top_k, int experts,
                        const int* reuse_mode) {
  constexpr int Lanes = 16;
  if constexpr (RouteGuard)
    if (*reuse_mode) return;
  using Dot = LatticeDot<Type>;
  __shared__ uint32_t book[Dot::kBookWords];
  __shared__ uint32_t masks[16];
  extern __shared__ Q8_1 shared_x[];
  const int groups = k / 32, route = blockIdx.y;
  const Q8_1* x = activation + (route / top_k) * groups;
  for (int i = threadIdx.x; i < groups * 9; i += blockDim.x)
    reinterpret_cast<uint32_t*>(shared_x)[i] =
        reinterpret_cast<const uint32_t*>(x)[i];
  Dot::initialize(book, masks);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  constexpr int Rows = 8;
  const int local_row = warp * (32 / Lanes) + lane / Lanes;
  const int row = blockIdx.x * Rows + local_row;
  if (row >= n) return;
  require_expert(ids[route], experts);
  const int64_t expert_row = ids[route] * n + row;
  const uint8_t* g = gate + expert_row * stride;
  const uint8_t* u = up + expert_row * stride;
  float gs = 0.f, us = 0.f;
  for (int group = lane % Lanes; group < groups; group += Lanes) {
    gs = __fadd_rn(gs, Dot::dot(g, group, shared_x[group], book, masks));
    us = __fadd_rn(us, Dot::dot(u, group, shared_x[group], book, masks));
  }
#pragma unroll
  for (int offset = Lanes / 2; offset; offset >>= 1) {
    gs += __shfl_down_sync(__activemask(), gs, offset, Lanes);
    us += __shfl_down_sync(__activemask(), us, offset, Lanes);
  }
  if (lane % Lanes == 0) {
    output[int64_t(route) * 2 * n + row] = __float2half_rn(gs);
    output[int64_t(route) * 2 * n + n + row] = __float2half_rn(us);
  }
}

// Concurrent and speculative rows route many tokens to the same expert. The
// CTA of the first route of each expert decodes the expert's groups once and
// applies them to every route selecting it; the other CTAs exit. Per-route
// arithmetic and reductions are those of gate_up.
template <int Type, class Index, bool Guarded = false>
__global__ void v41_gate_reuse(half* output, const Q8_1* activation,
                               const Index* ids, const uint8_t* gate,
                               const uint8_t* up, int stride,
                               const int* enabled) {
  if constexpr (Guarded)
    if (!*enabled) return;
  const Index expert = ids[blockIdx.y];
  // Every CTA of a repeated route exits before allocating/initializing its
  // codebook. The first occurrence computes all six tokens selecting it.
  for (int previous = 0; previous < blockIdx.y; ++previous)
    if (ids[previous] == expert) return;
  int members[6] = {};
#pragma unroll
  for (int token = 0; token < 6; ++token)
#pragma unroll
    for (int route = 0; route < 6; ++route)
      members[token] |= int(ids[token * 6 + route] == expert) << route;
  using Dot = LatticeDot<Type>;
  __shared__ uint32_t book[Dot::kBookWords], masks[16];
  Dot::initialize(book, masks);
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int row = blockIdx.x * 8 + warp * 2 + lane / 16;
  const auto* g = gate + (int64_t(expert) * 576 + row) * stride;
  const auto* u = up + (int64_t(expert) * 576 + row) * stride;
  float gs[6] = {}, us[6] = {};
  for (int group = lane % 16; group < 160; group += 16) {
    const V41IntegerGroup<Type> dg(g, group, book, masks);
    const V41IntegerGroup<Type> du(u, group, book, masks);
#pragma unroll
    for (int token = 0; token < 6; ++token)
      if (members[token]) {
        const auto& x = activation[token * 160 + group];
        gs[token] = __fadd_rn(gs[token], dg.apply(x));
        us[token] = __fadd_rn(us[token], du.apply(x));
      }
  }
#pragma unroll
  for (int token = 0; token < 6; ++token) {
#pragma unroll
    for (int offset = 8; offset; offset >>= 1) {
      gs[token] += __shfl_down_sync(0xffffffffU, gs[token], offset, 16);
      us[token] += __shfl_down_sync(0xffffffffU, us[token], offset, 16);
    }
    if (lane % 16 == 0) {
#pragma unroll
      for (int route = 0; route < 6; ++route)
        if (members[token] & (1 << route)) {
          output[(int64_t(token) * 6 + route) * 1152 + row] =
              __float2half_rn(gs[token]);
          output[(int64_t(token) * 6 + route) * 1152 + 576 + row] =
              __float2half_rn(us[token]);
        }
    }
  }
}

template <class Index>
__global__ void v41_reuse_plan(const Index* ids, int32_t* compact, int* enabled,
                               int threshold, int slots) {
  __shared__ int values[36], counts[2];
  const int i = threadIdx.x;
  if (i < 36) {
    const auto value = ids[i];
    if (value < 0 || value >= slots) asm volatile("trap;");
    compact[i] = values[i] = int(value);
  }
  __syncthreads();
  bool first = i < 36;
  if (first)
    for (int j = 0; j < i; ++j)
      if (values[j] == values[i]) first = false;
  const unsigned mask = __ballot_sync(0xffffffffU, first);
  if (i % 32 == 0) counts[i / 32] = __popc(mask);
  __syncthreads();
  if (!i) *enabled = counts[0] + counts[1] <= threshold;
}

// Upstream raw_down_unroute<Type, Index, /*Quantized=*/true, /*Lanes=*/4,
// Tiles> with unit route weights.
template <int Type, class Index, int Tiles>
__global__ void raw_down_unroute(half* output, const void* input,
                                 const Index* ids, const uint8_t* weights,
                                 int n, int row_stride, int left_groups,
                                 int top_k, int experts) {
  constexpr int Lanes = 4;
  extern __shared__ Q8_1 x[];
  __shared__ uint32_t book[LatticeDot<Type>::kBookWords], masks[16];
  __shared__ float partial[4][32 / Lanes];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int groups = 18, token = blockIdx.y;
  const auto* source = static_cast<const uint32_t*>(input) +
                       int64_t{token} * top_k * groups * 9;
  for (int i = threadIdx.x; i < top_k * groups * 9; i += blockDim.x)
    reinterpret_cast<uint32_t*>(x)[i] = source[i];
  LatticeDot<Type>::initialize(book, masks);
  // Reuse the activation and integer codebook across adjacent output tiles.
  // Each tile retains exactly the original K-group and route additions.
  for (int tile = 0; tile < Tiles; ++tile) {
    const int col = (blockIdx.x * Tiles + tile) * (32 / Lanes) + lane / Lanes;
    float total = 0.f;
    for (int route = warp; route < top_k; route += 4) {
      const int slot = token * top_k + route;
      require_expert(ids[slot], experts);
      const uint8_t* row =
          weights + (int64_t{ids[slot]} * n + col) * row_stride;
      float dot = 0.f;
      // Neighboring lanes fetch neighboring K groups of the same packed
      // row. Shuffles then restore the original ascending FP32 additions;
      // a parallel reduction would change the numerical boundary.
      for (int first = 0; first < groups; first += Lanes) {
        const int group = first + lane % Lanes;
        const float value =
            group < groups ? LatticeDot<Type>::dot(row, group + left_groups,
                                                   x[route * groups + group],
                                                   book, masks)
                           : 0.f;
#pragma unroll
        for (int i = 0; i < Lanes; ++i) {
          const float ordered =
              __shfl_sync(0xffffffff, value, (lane / Lanes) * Lanes + i);
          if (first + i < groups) dot = __fadd_rn(dot, ordered);
        }
      }
      total += __half2float(__float2half_rn(dot));
    }
    if (lane % Lanes == 0) partial[warp][lane / Lanes] = total;
    __syncthreads();
    if (!warp && lane % Lanes == 0) {
      float sum = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) sum += partial[i][lane / Lanes];
      output[int64_t{token} * n + col] = __float2half_rn(sum);
    }
    if constexpr (Tiles > 1) __syncthreads();
  }
}

// ---- GGML dequantization (ggml-quants.c reference order, FP32 then RN16) ----
__device__ __forceinline__ uint32_t byte_u16(const uint8_t* p) {
  return uint32_t(p[0]) | (uint32_t(p[1]) << 8);
}
__device__ __forceinline__ float byte_half(const uint8_t* p) {
  return __half2float(__ushort_as_half(uint16_t(byte_u16(p))));
}
template <int Book>
__device__ __forceinline__ float grid_value(int entry, int j) {
  return float(int(turbomind::gemm::LatticeCodebook<Book>::value(entry * 8 + j)) - 128);
}
__constant__ int8_t kIQ4NL[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                  1,    13,   25,  38,  53,  69,  89,  113};

// Decodes values [8 * g, 8 * g + 8) of one block starting at b.
template <int Type>
__device__ void decode8(const uint8_t* b, int g, float* y) {
  if constexpr (Type == 8) {  // Q8_0: x * d
    const float d = byte_half(b);
    for (int j = 0; j < 8; ++j) y[j] = __fmul_rn(float(int8_t(b[2 + 8 * g + j])), d);
  } else if constexpr (Type == 20) {  // IQ4_NL: d * kvalues[nibble]
    const float d = byte_half(b);
    for (int j = 0; j < 8; ++j) {
      const int q = g < 2 ? b[2 + 8 * g + j] & 15 : b[2 + 8 * (g - 2) + j] >> 4;
      y[j] = __fmul_rn(d, float(kIQ4NL[q]));
    }
  } else if constexpr (Type == 13) {  // Q5_K: (d * sc) * q - (dmin * m)
    const float d = byte_half(b), dmin = byte_half(b + 2);
    const uint8_t* scales = b + 4;
    const int j64 = g / 8, high = (g % 8) / 4, l0 = (g % 4) * 8;
    const int is = 2 * j64 + high;
    int sc, m;
    if (is < 4) {
      sc = scales[is] & 63;
      m = scales[is + 4] & 63;
    } else {
      sc = (scales[is + 4] & 0xF) | ((scales[is - 4] >> 6) << 4);
      m = (scales[is + 4] >> 4) | ((scales[is] >> 6) << 4);
    }
    const float d1 = __fmul_rn(d, float(sc)), m1 = __fmul_rn(dmin, float(m));
    const uint8_t* ql = b + 48 + 32 * j64;
    const uint8_t* qh = b + 16;
    const int bit = 1 << (2 * j64 + high);
    for (int j = 0; j < 8; ++j) {
      const int l = l0 + j;
      const int q = (high ? ql[l] >> 4 : ql[l] & 15) + (qh[l] & bit ? 16 : 0);
      y[j] = __fsub_rn(__fmul_rn(d1, float(q)), m1);
    }
  } else if constexpr (Type == 16) {  // IQ2_XXS
    const float d = byte_half(b);
    const int ib32 = g / 4, l = g % 4;
    const uint8_t* q = b + 2 + 8 * ib32;
    const uint32_t aux1 = byte_u16(q + 4) | (byte_u16(q + 6) << 16);
    const float db = __fmul_rn(__fmul_rn(d, __fadd_rn(0.5f, float(aux1 >> 28))), 0.25f);
    int signs = (aux1 >> (7 * l)) & 127;
    signs |= (__popc(signs) & 1) << 7;
    for (int j = 0; j < 8; ++j)
      y[j] = __fmul_rn(__fmul_rn(db, grid_value<16>(q[l], j)),
                       signs & (1 << j) ? -1.f : 1.f);
  } else if constexpr (Type == 17) {  // IQ2_XS
    const float d = byte_half(b);
    const int ib32 = g / 4, l = g % 4;
    const int scale = (b[66 + ib32] >> (4 * (l / 2))) & 15;
    const float db = __fmul_rn(__fmul_rn(d, __fadd_rn(0.5f, float(scale))), 0.25f);
    const uint32_t code = byte_u16(b + 2 + 2 * (4 * ib32 + l));
    int signs = code >> 9;
    signs |= (__popc(signs) & 1) << 7;
    for (int j = 0; j < 8; ++j)
      y[j] = __fmul_rn(__fmul_rn(db, grid_value<17>(code & 511, j)),
                       signs & (1 << j) ? -1.f : 1.f);
  } else {  // IQ1_M: dl * (grid + delta)
    static_assert(Type == 29);
    const uint8_t* sc = b + 48;
    const uint32_t s0 = byte_u16(sc), s1 = byte_u16(sc + 2), s2 = byte_u16(sc + 4),
                   s3 = byte_u16(sc + 6);
    const uint16_t bits = uint16_t((s0 >> 12) | ((s1 >> 8) & 0x00f0) |
                                   ((s2 >> 4) & 0x0f00) | (s3 & 0xf000));
    const float d = __half2float(__ushort_as_half(bits));
    const int ib = g / 4, l = g % 4;
    const uint32_t word = byte_u16(sc + 2 * (ib / 2));
    const int shift = 6 * (ib % 2) + (l < 2 ? 0 : 3);
    const float dl = __fmul_rn(d, float(2 * int((word >> shift) & 7) + 1));
    const int qh = b[32 + 2 * ib + l / 2];
    const int index = b[4 * ib + l] | ((l % 2 ? qh << 4 : qh << 8) & 0x700);
    const float delta = qh & (l % 2 ? 0x80 : 0x08) ? -0.125f : 0.125f;
    for (int j = 0; j < 8; ++j)
      y[j] = __fmul_rn(dl, __fadd_rn(grid_value<29>(index, j), delta));
  }
}

template <int Type, int Block, int Bytes>
__global__ void dequantize_kernel(const uint8_t* packed, half* out,
                                  int64_t rows, int64_t row_bytes, int cols) {
  const int64_t per_row = cols / 8;
  for (int64_t at = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       at < rows * per_row; at += int64_t(gridDim.x) * blockDim.x) {
    const int64_t row = at / per_row;
    const int col = int(at % per_row) * 8;
    float y[8];
    decode8<Type>(packed + row * row_bytes + int64_t(col / Block) * Bytes,
                  (col % Block) / 8, y);
    __align__(16) half h[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) h[j] = __float2half_rn(y[j]);
    reinterpret_cast<uint4*>(out + row * cols + col)[0] =
        *reinterpret_cast<const uint4*>(h);
  }
}

int block_bytes(int64_t type) {
  return type == 16 ? 66 : type == 17 ? 74 : 56;
}

void require_sm70(const torch::Tensor& t) {
  TORCH_CHECK(t.is_cuda(), "DeepSeek-V4.1 SM70 operators require CUDA tensors");
  const auto* props = at::cuda::getDeviceProperties(t.get_device());
  TORCH_CHECK(props->major == 7 && props->minor == 0, "GGUF dp4a requires SM70");
}

bool aligned(const torch::Tensor& t, int64_t bytes) {
  return reinterpret_cast<uintptr_t>(t.data_ptr()) % bytes == 0;
}

void check_ids(const torch::Tensor& ids, const torch::Tensor& like, int64_t m) {
  TORCH_CHECK(ids.device() == like.device() &&
                  (ids.scalar_type() == torch::kInt32 ||
                   ids.scalar_type() == torch::kInt64) &&
                  ids.is_contiguous() && ids.dim() == 2 && ids.size(0) == m &&
                  ids.size(1) == kTop,
              "V4.1 routes require contiguous int32/int64 ids [M,6]");
}

void check_gate_up(const torch::Tensor& out, const torch::Tensor& activation,
                   const torch::Tensor& ids, const torch::Tensor& gate,
                   const torch::Tensor& up, int64_t type) {
  TORCH_CHECK(type == 16 || type == 17 || type == 29,
              "V4.1 gate/up requires IQ2_XXS(16)/IQ2_XS(17)/IQ1_M(29)");
  require_sm70(activation);
  TORCH_CHECK(activation.scalar_type() == torch::kUInt8 &&
                  activation.dim() == 3 && activation.size(0) >= 1 &&
                  activation.size(0) <= 6 && activation.size(1) == kHidden / 32 &&
                  activation.size(2) == sizeof(Q8_1) &&
                  activation.is_contiguous() && aligned(activation, 4),
              "V4.1 gate/up requires Q8_1 [M<=6,160,36] activation");
  const int64_t m = activation.size(0);
  check_ids(ids, activation, m);
  const int64_t bytes = kHidden / 256 * block_bytes(type);
  TORCH_CHECK(gate.device() == activation.device() &&
                  up.device() == activation.device() &&
                  gate.scalar_type() == torch::kUInt8 &&
                  up.scalar_type() == torch::kUInt8 && gate.is_contiguous() &&
                  up.is_contiguous() && gate.dim() == 3 &&
                  gate.sizes() == up.sizes() && gate.size(0) > 0 &&
                  gate.size(0) <= 65535 && gate.size(1) == kRows &&
                  (gate.size(2) == bytes || gate.size(2) == (bytes + 7) / 8 * 8) &&
                  aligned(gate, 2) && aligned(up, 2),
              "V4.1 gate/up requires original expert rows [E,576,20*block]");
  TORCH_CHECK(out.device() == activation.device() && out.is_contiguous() &&
                  out.scalar_type() == torch::kFloat16 && out.dim() == 4 &&
                  out.size(0) == m && out.size(1) == kTop && out.size(2) == 2 &&
                  out.size(3) == kRows,
              "V4.1 gate/up requires FP16 output [M,6,2,576]");
}

template <int Type, class Index, bool Guarded>
void launch_gate_up(const torch::Tensor& out, const torch::Tensor& activation,
                    const torch::Tensor& ids, const torch::Tensor& gate,
                    const torch::Tensor& up, const int* mode,
                    cudaStream_t stream) {
  gate_up<Type, Index, Guarded>
      <<<dim3((kRows + 7) / 8, ids.numel()), 128, (kHidden / 32) * sizeof(Q8_1),
         stream>>>(
          reinterpret_cast<half*>(out.data_ptr()),
          reinterpret_cast<const Q8_1*>(activation.data_ptr()),
          ids.data_ptr<Index>(), gate.data_ptr<uint8_t>(),
          up.data_ptr<uint8_t>(), kRows, kHidden, gate.size(2), kTop,
          gate.size(0), mode);
}

template <int Type>
void dispatch_gate_up(const torch::Tensor& out, const torch::Tensor& activation,
                      const torch::Tensor& ids, const torch::Tensor& gate,
                      const torch::Tensor& up) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (ids.scalar_type() == torch::kInt32)
    launch_gate_up<Type, int32_t, false>(out, activation, ids, gate, up, nullptr, stream);
  else
    launch_gate_up<Type, int64_t, false>(out, activation, ids, gate, up, nullptr, stream);
}

template <class Index>
void launch_adaptive(const torch::Tensor& out, const torch::Tensor& activation,
                     const torch::Tensor& ids, const torch::Tensor& gate,
                     const torch::Tensor& up) {
  constexpr int Type = 29;
  const auto stream = at::cuda::getCurrentCUDAStream();
  auto compact = torch::empty({6, 6}, ids.options().dtype(torch::kInt32));
  auto mode = torch::empty({1}, ids.options().dtype(torch::kInt32));
  v41_reuse_plan<Index><<<1, 64, 0, stream>>>(
      ids.data_ptr<Index>(), compact.data_ptr<int32_t>(), mode.data_ptr<int>(),
      32, gate.size(0));
  // Both branches are captured. The device decision executes exactly one;
  // the fallback retains the caller's original index width and dot order.
  launch_gate_up<Type, Index, true>(out, activation, ids, gate, up,
                                    mode.data_ptr<int>(), stream);
  v41_gate_reuse<Type, int32_t, true><<<dim3(72, 36), 128, 0, stream>>>(
      reinterpret_cast<half*>(out.data_ptr()),
      reinterpret_cast<const Q8_1*>(activation.data_ptr()),
      compact.data_ptr<int32_t>(), gate.data_ptr<uint8_t>(),
      up.data_ptr<uint8_t>(), gate.size(2), mode.data_ptr<int>());
}

template <int Type, class Index, int Tiles>
void launch_down(const torch::Tensor& out, const torch::Tensor& input,
                 const torch::Tensor& ids, const torch::Tensor& weights,
                 int left) {
  raw_down_unroute<Type, Index, Tiles>
      <<<dim3(out.size(1) / 8 / Tiles, out.size(0)), 128,
         kTop * 18 * sizeof(Q8_1), at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<half*>(out.data_ptr()), input.data_ptr(),
          ids.data_ptr<Index>(), weights.data_ptr<uint8_t>(), out.size(1),
          weights.size(2), left / 32, kTop, weights.size(0));
}

template <int Type, class Index>
void dispatch_down(const torch::Tensor& out, const torch::Tensor& input,
                   const torch::Tensor& ids, const torch::Tensor& weights,
                   int left) {
  // Upstream's default tiles: IQ1_M 2 for M1 and 4 otherwise; IQ2 1.
  if constexpr (Type == 29) {
    if (input.size(0) == 1)
      launch_down<Type, Index, 2>(out, input, ids, weights, left);
    else
      launch_down<Type, Index, 4>(out, input, ids, weights, left);
  } else {
    launch_down<Type, Index, 1>(out, input, ids, weights, left);
  }
}
}  // namespace

void ds41_quantize_q8_1(torch::Tensor out, const torch::Tensor& input) {
  require_sm70(input);
  TORCH_CHECK(input.dim() == 2 && input.is_contiguous() &&
                  input.scalar_type() == torch::kFloat16 && input.size(0) >= 1 &&
                  input.size(0) <= 20 && input.size(1) > 0 &&
                  input.size(1) % 256 == 0 && out.device() == input.device() &&
                  out.scalar_type() == torch::kUInt8 && out.is_contiguous() &&
                  out.dim() == 3 && out.size(0) == input.size(0) &&
                  out.size(1) == input.size(1) / 32 &&
                  out.size(2) == sizeof(Q8_1) && aligned(out, 4),
              "Expected FP16 [M<=20,K%256] and Q8_1 [M,K/32,36]");
  const c10::cuda::CUDAGuard guard(input.device());
  const dim3 grid((input.size(1) + 255) / 256, input.size(0));
  quantize_q8<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<Q8_1*>(out.data_ptr()),
      reinterpret_cast<const half*>(input.data_ptr()), input.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ds41_swiglu_q8(torch::Tensor out, const torch::Tensor& projection,
                    const torch::Tensor& probabilities) {
  require_sm70(projection);
  TORCH_CHECK(projection.is_contiguous() &&
                  projection.scalar_type() == torch::kFloat16 &&
                  projection.dim() == 4 && projection.size(0) > 0 &&
                  projection.size(0) <= 6 && projection.size(1) == 6 &&
                  projection.size(2) == 2 && projection.size(3) == 576 &&
                  out.device() == projection.device() && out.is_contiguous() &&
                  out.scalar_type() == torch::kUInt8 && out.dim() == 4 &&
                  out.size(0) == projection.size(0) && out.size(1) == 6 &&
                  out.size(2) == 18 && out.size(3) == sizeof(Q8_1) &&
                  probabilities.device() == projection.device() &&
                  probabilities.is_contiguous() &&
                  probabilities.scalar_type() == torch::kFloat32 &&
                  probabilities.dim() == 2 &&
                  probabilities.size(0) == projection.size(0) &&
                  probabilities.size(1) == 6,
              "V4.1 SwiGLU Q8 requires FP16 M1..6 projections and FP32 routes");
  const c10::cuda::CUDAGuard guard(projection.device());
  v41_swiglu_q8<<<dim3(3, projection.size(0) * 6), 256, 0,
                  at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<Q8_1*>(out.data_ptr()),
      reinterpret_cast<const half*>(projection.data_ptr()),
      probabilities.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ds41_shared_swiglu(torch::Tensor out, const torch::Tensor& gate,
                        const torch::Tensor& up) {
  require_sm70(gate);
  TORCH_CHECK(gate.is_contiguous() && gate.dim() == 2 &&
                  gate.scalar_type() == torch::kFloat16 &&
                  up.device() == gate.device() && up.is_contiguous() &&
                  up.scalar_type() == torch::kFloat16 &&
                  up.sizes() == gate.sizes() && out.device() == gate.device() &&
                  out.is_contiguous() && out.scalar_type() == torch::kFloat16 &&
                  out.sizes() == gate.sizes(),
              "V4.1 shared SwiGLU requires matching contiguous FP16 matrices");
  if (!gate.numel()) return;
  const c10::cuda::CUDAGuard guard(gate.device());
  v41_shared_swiglu<<<(gate.numel() + 255) / 256, 256, 0,
                      at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<half*>(out.data_ptr()),
      reinterpret_cast<const half*>(gate.data_ptr()),
      reinterpret_cast<const half*>(up.data_ptr()), gate.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ds41_gate_up(torch::Tensor out, const torch::Tensor& activation,
                  const torch::Tensor& ids, const torch::Tensor& gate,
                  const torch::Tensor& up, int64_t type) {
  check_gate_up(out, activation, ids, gate, up, type);
  const c10::cuda::CUDAGuard guard(activation.device());
  if (type == 16)
    dispatch_gate_up<16>(out, activation, ids, gate, up);
  else if (type == 17)
    dispatch_gate_up<17>(out, activation, ids, gate, up);
  else
    dispatch_gate_up<29>(out, activation, ids, gate, up);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ds41_gate_adaptive(torch::Tensor out, const torch::Tensor& activation,
                        const torch::Tensor& ids, const torch::Tensor& gate,
                        const torch::Tensor& up, int64_t type) {
  check_gate_up(out, activation, ids, gate, up, type);
  TORCH_CHECK(type == 29 && activation.size(0) == 6,
              "V4.1 adaptive gate/up is the M6 IQ1_M schedule");
  const c10::cuda::CUDAGuard guard(activation.device());
  if (ids.scalar_type() == torch::kInt32)
    launch_adaptive<int32_t>(out, activation, ids, gate, up);
  else
    launch_adaptive<int64_t>(out, activation, ids, gate, up);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ds41_down(torch::Tensor out, const torch::Tensor& input,
               const torch::Tensor& ids, const torch::Tensor& weights,
               int64_t type, int64_t left) {
  TORCH_CHECK(type == 16 || type == 17 || type == 29,
              "V4.1 down requires IQ2_XXS(16)/IQ2_XS(17)/IQ1_M(29)");
  require_sm70(input);
  TORCH_CHECK(input.is_contiguous() && input.scalar_type() == torch::kUInt8 &&
                  input.dim() == 4 && input.size(0) >= 1 && input.size(0) <= 6 &&
                  input.size(1) == kTop && input.size(2) == 18 &&
                  input.size(3) == sizeof(Q8_1) && aligned(input, 4),
              "V4.1 down requires Q8_1 input [M<=6,6,18,36]");
  TORCH_CHECK(left >= 0 && left < 256 && left % 64 == 0,
              "V4.1 down left offset must be 0/64/128/192");
  check_ids(ids, input, input.size(0));
  TORCH_CHECK(out.device() == input.device() && out.is_contiguous() &&
                  out.scalar_type() == torch::kFloat16 && out.dim() == 2 &&
                  out.size(0) == input.size(0) && out.size(1) == kHidden,
              "V4.1 down requires FP16 output [M,5120]");
  TORCH_CHECK(weights.device() == input.device() && weights.is_contiguous() &&
                  weights.scalar_type() == torch::kUInt8 && weights.dim() == 3 &&
                  weights.size(0) > 0 && weights.size(0) <= 65535 &&
                  weights.size(1) == kHidden &&
                  weights.size(2) == ((left + 576 + 255) / 256) * block_bytes(type) &&
                  aligned(weights, 2),
              "V4.1 down requires original rows [E,5120,3*block]");
  const c10::cuda::CUDAGuard guard(input.device());
#define DS41_DOWN(TYPE)                                                    \
  if (ids.scalar_type() == torch::kInt32)                                  \
    dispatch_down<TYPE, int32_t>(out, input, ids, weights, int(left));     \
  else                                                                     \
    dispatch_down<TYPE, int64_t>(out, input, ids, weights, int(left))
  if (type == 16) {
    DS41_DOWN(16);
  } else if (type == 17) {
    DS41_DOWN(17);
  } else {
    DS41_DOWN(29);
  }
#undef DS41_DOWN
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ds41_dequantize(torch::Tensor out, const torch::Tensor& packed,
                     int64_t type) {
  require_sm70(packed);
  int block, bytes;
  switch (type) {
    case 8: block = 32; bytes = 34; break;
    case 13: block = 256; bytes = 176; break;
    case 16: block = 256; bytes = 66; break;
    case 17: block = 256; bytes = 74; break;
    case 20: block = 32; bytes = 18; break;
    case 29: block = 256; bytes = 56; break;
    default: TORCH_CHECK(false, "dequantize supports GGML types 8/13/16/17/20/29, got ", type);
  }
  TORCH_CHECK(packed.scalar_type() == torch::kUInt8 && packed.dim() == 2 &&
                  packed.is_contiguous() && out.device() == packed.device() &&
                  out.scalar_type() == torch::kFloat16 && out.dim() == 2 &&
                  out.is_contiguous() && aligned(out, 16) &&
                  out.size(0) == packed.size(0) && out.size(1) > 0 &&
                  out.size(1) <= INT32_MAX && out.size(1) % block == 0 &&
                  packed.size(1) == out.size(1) / block * bytes,
              "dequantize requires uint8 [rows, cols/block*bytes] and FP16 [rows, cols]");
  const int64_t rows = packed.size(0), cols = out.size(1);
  if (!rows) return;
  const c10::cuda::CUDAGuard guard(packed.device());
  const int64_t work = rows * (cols / 8);
  const int blocks = int(std::min<int64_t>((work + 255) / 256, 65535 * 8));
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto* source = packed.data_ptr<uint8_t>();
  auto* target = reinterpret_cast<half*>(out.data_ptr());
  switch (type) {
#define DS41_DQ(T, B, S) \
  case T: dequantize_kernel<T, B, S><<<blocks, 256, 0, stream>>>(source, target, rows, packed.size(1), int(cols)); break
    DS41_DQ(8, 32, 34);
    DS41_DQ(13, 256, 176);
    DS41_DQ(16, 256, 66);
    DS41_DQ(17, 256, 74);
    DS41_DQ(20, 32, 18);
    DS41_DQ(29, 256, 56);
#undef DS41_DQ
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
