// DeepSeek-V4.1 DSpark routed experts: MXFP4 (E2M1 pairs + one E8M0 scale per 32 inputs) on SM70.
// TensorFold-owned. Two activation contracts, one weight decode:
//  * every E2M1 code is decoded to its doubled value 2*e2m1 in -12..12 (an exact integer), and a block's
//    weights are those integers times 2^(s - 128) (s = the E8M0 byte), so no weight value is ever rounded;
//  * F16 mode (the reference contract of 1Cat-vLLM's TurboMind MXFP4 grouped GEMM, mxfp4_sm70_moe.py @4d57473d:
//    FP16 activations times the exact weights, FP32 accumulation): per block the 32 products int*x are exact in
//    FP32 and summed in FP32, then scaled by the power of two (exact) and accumulated in FP32;
//  * Q8 mode (the 1Cat dp4a style of onecat_ds41.cu): activations are quantized per 32 to int8 with an FP32 scale
//    d = amax/127 (an explicit activation quantization, NOT FP16-exact); dp4a sums are exact int32, and each
//    block adds float(dot) * (d * 2^(s - 128)) in FP32.
// Rounding boundaries follow the reference: FP16 gate/up per route; SwiGLU as vLLM silu_and_mul_with_clamp
// (gate = min(g, 10), up = clamp(u, -10, 10), fp16(silu) * up in FP16); FP16 per-route down output; the
// route-weighted sum in FP32 in route order, rounded once to FP16.
// Route ids are this rank's local expert indices; -1 marks a route owned by another rank (contributes nothing,
// reads nothing). An id outside [-1, E) traps the device.
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/MemoryOverlap.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <limits>

namespace {

struct Q8F {
  float d;
  int8_t qs[32];
};
static_assert(sizeof(Q8F) == 36);

__constant__ int8_t kE2M1x2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

constexpr int kWarps = 8;

__device__ __forceinline__ float block_scale(uint8_t s) {
  // 2^(s - 128) for the doubled E2M1 integers; s - 128 in [-128, 127] is exact in FP32 (s = 0: subnormal).
  return ldexpf(1.f, int(s) - 128);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int offset = 16; offset; offset >>= 1) v += __shfl_xor_sync(0xffffffff, v, offset);
  return v;
}

__device__ __forceinline__ void check_expert(int e, int experts) {
  if (e < -1 || e >= experts) __trap();
}

// Exact FP32 block dot of 16 code bytes with 32 FP32 activations (F16 mode).
__device__ __forceinline__ float dot_f16(const uint4 w, const float* x) {
  const uint32_t words[4] = {w.x, w.y, w.z, w.w};
  float acc = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t byte = (words[i] >> (8 * j)) & 0xFF;
      const int e = 8 * i + 2 * j;
      acc = fmaf(float(kE2M1x2[byte & 15]), x[e], acc);
      acc = fmaf(float(kE2M1x2[byte >> 4]), x[e + 1], acc);
    }
  }
  return acc;
}

// Exact int32 block dot of 16 code bytes with 32 int8 activations (Q8 mode).
__device__ __forceinline__ int dot_q8(const uint4 w, const int* q, const uint16_t* lut) {
  const uint32_t words[4] = {w.x, w.y, w.z, w.w};
  int acc = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t a = words[i];
    const int lo = int(uint32_t(lut[a & 0xFF]) | (uint32_t(lut[(a >> 8) & 0xFF]) << 16));
    const int hi = int(uint32_t(lut[(a >> 16) & 0xFF]) | (uint32_t(lut[a >> 24]) << 16));
    acc = __dp4a(lo, q[2 * i], acc);
    acc = __dp4a(hi, q[2 * i + 1], acc);
  }
  return acc;
}

__device__ __forceinline__ void fill_lut(uint16_t* lut) {
  for (int b = threadIdx.x; b < 256; b += blockDim.x)
    lut[b] = uint16_t(uint8_t(kE2M1x2[b & 15])) | uint16_t(uint16_t(uint8_t(kE2M1x2[b >> 4])) << 8);
}

// x [M, K] FP16 -> Q8F [M, K/32]; one warp per block of 32.
__global__ void quantize_kernel(Q8F* out, const half* x, int k) {
  const int lane = threadIdx.x % 32;
  const int64_t group = int64_t(blockIdx.x) * kWarps + threadIdx.x / 32;
  const int64_t groups = int64_t(gridDim.y) * (k / 32);
  const int64_t g = int64_t(blockIdx.y) * (k / 32) + group;
  if (group >= k / 32 || g >= groups) return;
  const float v = __half2float(x[g * 32 + lane]);
  float m = fabsf(v);
#pragma unroll
  for (int offset = 16; offset; offset >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, offset));
  const float d = m / 127.f;
  out[g].qs[lane] = m == 0.f ? int8_t(0) : int8_t(rintf(v / d));
  if (!lane) out[g].d = d;
}

// out [routes, 2, N] FP16 (gate, up); one warp per output row n of one route.
template <bool Q8>
__global__ void gate_up_kernel(half* out, const half* x, const Q8F* xq, const int* ids, const uint8_t* gate,
                               const uint8_t* gate_s, const uint8_t* up, const uint8_t* up_s, int n_rows, int k,
                               int top, int experts) {
  extern __shared__ float smem[];
  __shared__ uint16_t lut[256];
  const int route = blockIdx.y, m = route / top;
  const int e = ids[route];
  check_expert(e, experts);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int n = blockIdx.x * kWarps + warp;
  if (e < 0) {
    if (n < n_rows && lane == 0) {
      out[(int64_t(route) * 2) * n_rows + n] = __float2half(0.f);
      out[(int64_t(route) * 2 + 1) * n_rows + n] = __float2half(0.f);
    }
    return;
  }
  if (Q8) {
    fill_lut(lut);
  } else {
    for (int i = threadIdx.x; i < k; i += blockDim.x) smem[i] = __half2float(x[int64_t(m) * k + i]);
  }
  __syncthreads();
  if (n >= n_rows) return;
  const int blocks = k / 32;
  const int64_t row = int64_t(e) * n_rows + n;
  const uint4* gw = reinterpret_cast<const uint4*>(gate + row * (k / 2));
  const uint4* uw = reinterpret_cast<const uint4*>(up + row * (k / 2));
  const uint8_t* gs = gate_s + row * blocks;
  const uint8_t* us = up_s + row * blocks;
  float ga = 0.f, ua = 0.f;
  for (int b = lane; b < blocks; b += 32) {
    const uint4 g4 = __ldg(gw + b), u4 = __ldg(uw + b);
    if (Q8) {
      const Q8F& a = xq[int64_t(m) * blocks + b];
      int q[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) q[i] = reinterpret_cast<const int*>(a.qs)[i];
      ga += float(dot_q8(g4, q, lut)) * (a.d * block_scale(__ldg(gs + b)));
      ua += float(dot_q8(u4, q, lut)) * (a.d * block_scale(__ldg(us + b)));
    } else {
      ga += dot_f16(g4, smem + b * 32) * block_scale(__ldg(gs + b));
      ua += dot_f16(u4, smem + b * 32) * block_scale(__ldg(us + b));
    }
  }
  ga = warp_sum(ga);
  ua = warp_sum(ua);
  if (!lane) {
    out[(int64_t(route) * 2) * n_rows + n] = __float2half_rn(ga);
    out[(int64_t(route) * 2 + 1) * n_rows + n] = __float2half_rn(ua);
  }
}

__device__ __forceinline__ half swiglu_value(half g, half u) {
  // vLLM act_and_mul_kernel<silu_kernel, HAS_CLAMP>: clamp in FP32 then back to half, silu in FP32 to half,
  // and the half product.
  const half gc = __float2half_rn(fminf(__half2float(g), 10.f));
  const half uc = __float2half_rn(fmaxf(fminf(__half2float(u), 10.f), -10.f));
  const float gf = __half2float(gc);
  const half s = __float2half_rn(gf / (1.f + expf(-gf)));
  return __hmul(s, uc);
}

// gate_up [routes, 2, N] -> FP16 [routes, N] (and Q8F [routes, N/32] when q8 != nullptr).
__global__ void swiglu_kernel(half* out, Q8F* q8, const half* gate_up, int n_rows, int routes) {
  const int lane = threadIdx.x % 32;
  const int64_t group = int64_t(blockIdx.x) * kWarps + threadIdx.x / 32;
  const int groups = n_rows / 32;
  if (group >= int64_t(routes) * groups) return;
  const int64_t route = group / groups, col = (group % groups) * 32 + lane;
  const half v = swiglu_value(gate_up[(route * 2) * n_rows + col], gate_up[(route * 2 + 1) * n_rows + col]);
  out[route * n_rows + col] = v;
  if (q8 != nullptr) {
    const float f = __half2float(v);
    float m = fabsf(f);
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, offset));
    const float d = m / 127.f;
    q8[group].qs[lane] = m == 0.f ? int8_t(0) : int8_t(rintf(f / d));
    if (!lane) q8[group].d = d;
  }
}

// One warp per output dimension n of token m. The split ETP path keeps one
// unweighted FP32 plane per route until the complete expert has been reduced.
template <bool Q8, bool Partials = false>
__global__ void down_kernel(void* out, const half* inter, const Q8F* iq, const int* ids, const float* probs,
                            const uint8_t* down, const uint8_t* down_s, int hidden, int k, int top, int experts) {
  extern __shared__ float smem[];  // [top, k] FP32 activations (F16 mode)
  __shared__ uint16_t lut[256];
  const int m = blockIdx.y;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int n = blockIdx.x * kWarps + warp;
  if (Q8) {
    fill_lut(lut);
  } else {
    for (int i = threadIdx.x; i < top * k; i += blockDim.x) smem[i] = __half2float(inter[int64_t(m) * top * k + i]);
  }
  __syncthreads();
  if (n >= hidden) return;
  const int blocks = k / 32;
  float total = 0.f;
  for (int r = 0; r < top; ++r) {
    const int e = ids[m * top + r];
    if constexpr (Partials) {
      if (e < -1 || e >= experts) {
        __trap();
        return;
      }
    } else {
      check_expert(e, experts);
    }
    if (e < 0) {
      if constexpr (Partials) {
        if (!lane) static_cast<float*>(out)[(int64_t(r) * gridDim.y + m) * hidden + n] = 0.f;
      }
      continue;
    }
    const int64_t row = int64_t(e) * hidden + n;
    const uint4* w = reinterpret_cast<const uint4*>(down + row * (k / 2));
    const uint8_t* s = down_s + row * blocks;
    float acc = 0.f;
    for (int b = lane; b < blocks; b += 32) {
      const uint4 w4 = __ldg(w + b);
      if (Q8) {
        const Q8F& a = iq[(int64_t(m) * top + r) * blocks + b];
        int q[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) q[i] = reinterpret_cast<const int*>(a.qs)[i];
        acc += float(dot_q8(w4, q, lut)) * (a.d * block_scale(__ldg(s + b)));
      } else {
        acc += dot_f16(w4, smem + r * k + b * 32) * block_scale(__ldg(s + b));
      }
    }
    acc = warp_sum(acc);
    if constexpr (Partials) {
      if (!lane) static_cast<float*>(out)[(int64_t(r) * gridDim.y + m) * hidden + n] = acc;
    } else {
      total += __half2float(__float2half_rn(acc)) * probs[m * top + r];
    }
  }
  if constexpr (!Partials) {
    if (!lane) static_cast<half*>(out)[int64_t(m) * hidden + n] = __float2half_rn(total);
  }
}

__global__ void finish_partials_kernel(float* out, const float* planes, const int64_t* ids, const float* probs,
                                       int64_t elements, int64_t hidden, int top, int experts) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= elements) return;
  const int64_t row = i / hidden;
  float routed = 0.f;
  for (int r = 0; r < top; ++r) {
    const int64_t route = row * top + r;
    const int64_t expert = ids[route];
    if (expert < -1 || expert >= experts) {
      __trap();
      return;
    }
    if (expert == -1) continue;
    const float complete_expert = __half2float(__float2half_rn(planes[int64_t(r) * elements + i]));
    routed = __fadd_rn(routed, __fmul_rn(complete_expert, probs[route]));
  }
  const float complete_routed = __half2float(__float2half_rn(routed));
  const float complete_shared = __half2float(__float2half_rn(planes[int64_t(top) * elements + i]));
  out[i] = __fadd_rn(complete_routed, complete_shared);
}

void check_cuda(const torch::Tensor& t, const char* name, at::ScalarType type) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == type, name,
              " must be a contiguous CUDA tensor of the expected dtype");
}

}  // namespace

void mxfp4_quantize(torch::Tensor out, const torch::Tensor& x) {
  check_cuda(x, "x", at::kHalf);
  check_cuda(out, "out", at::kByte);
  const int64_t m = x.size(0), k = x.size(1);
  TORCH_CHECK(x.dim() == 2 && k % 32 == 0 && m <= 65535 && out.numel() == m * (k / 32) * 36, "quantize shapes");
  const c10::cuda::CUDAGuard guard(x.device());
  if (!m) return;
  dim3 grid((k / 32 + kWarps - 1) / kWarps, m);
  quantize_kernel<<<grid, 32 * kWarps, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<Q8F*>(out.data_ptr()), reinterpret_cast<const half*>(x.data_ptr()), int(k));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void mxfp4_gate_up(torch::Tensor out, const torch::Tensor& x, const torch::Tensor& xq, const torch::Tensor& ids,
                   const torch::Tensor& gate, const torch::Tensor& gate_s, const torch::Tensor& up,
                   const torch::Tensor& up_s, bool q8) {
  check_cuda(out, "out", at::kHalf);
  check_cuda(ids, "ids", at::kInt);
  for (const auto* t : {&gate, &gate_s, &up, &up_s}) check_cuda(*t, "expert bank", at::kByte);
  TORCH_CHECK(gate.dim() == 3 && gate_s.dim() == 3 && gate.sizes() == up.sizes() && gate_s.sizes() == up_s.sizes(),
              "gate/up banks [E, N, K/2] with scales [E, N, K/32]");
  const int64_t experts = gate.size(0), n = gate.size(1), k = gate.size(2) * 2;
  TORCH_CHECK(gate_s.size(0) == experts && gate_s.size(1) == n && gate_s.size(2) * 32 == k && k % 32 == 0,
              "gate scales do not cover the bank");
  TORCH_CHECK(ids.dim() == 2 && ids.size(1) >= 1 && ids.size(1) <= 8, "ids [M, top<=8]");
  const int64_t m = ids.size(0), top = ids.size(1), routes = m * top;
  TORCH_CHECK(out.numel() == routes * 2 * n && routes <= 65535 && n % kWarps == 0, "gate/up output shape");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(gate.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(up.data_ptr()) % 16 == 0
              && (k / 2) % 16 == 0, "16-byte aligned expert rows");
  if (q8) {
    check_cuda(xq, "xq", at::kByte);
    TORCH_CHECK(xq.numel() == m * (k / 32) * 36, "xq shape");
  } else {
    check_cuda(x, "x", at::kHalf);
    TORCH_CHECK(x.dim() == 2 && x.size(0) == m && x.size(1) == k && k * 4 <= 48 * 1024, "x shape");
  }
  const c10::cuda::CUDAGuard guard(ids.device());
  if (!routes) return;
  dim3 grid(n / kWarps, routes);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto* o = reinterpret_cast<half*>(out.data_ptr());
  if (q8) {
    gate_up_kernel<true><<<grid, 32 * kWarps, 0, stream>>>(
        o, nullptr, reinterpret_cast<const Q8F*>(xq.data_ptr()), ids.data_ptr<int>(), gate.data_ptr<uint8_t>(),
        gate_s.data_ptr<uint8_t>(), up.data_ptr<uint8_t>(), up_s.data_ptr<uint8_t>(), int(n), int(k), int(top),
        int(experts));
  } else {
    gate_up_kernel<false><<<grid, 32 * kWarps, k * 4, stream>>>(
        o, reinterpret_cast<const half*>(x.data_ptr()), nullptr, ids.data_ptr<int>(), gate.data_ptr<uint8_t>(),
        gate_s.data_ptr<uint8_t>(), up.data_ptr<uint8_t>(), up_s.data_ptr<uint8_t>(), int(n), int(k), int(top),
        int(experts));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void mxfp4_swiglu(torch::Tensor out, torch::Tensor q8, const torch::Tensor& gate_up, bool quantize) {
  check_cuda(gate_up, "gate_up", at::kHalf);
  check_cuda(out, "out", at::kHalf);
  TORCH_CHECK(gate_up.dim() == 3 && gate_up.size(1) == 2 && gate_up.size(2) % 32 == 0, "gate_up [routes, 2, N]");
  const int64_t routes = gate_up.size(0), n = gate_up.size(2);
  TORCH_CHECK(out.numel() == routes * n, "swiglu output shape");
  if (quantize) {
    check_cuda(q8, "q8", at::kByte);
    TORCH_CHECK(q8.numel() == routes * (n / 32) * 36, "swiglu q8 shape");
  }
  const c10::cuda::CUDAGuard guard(gate_up.device());
  const int64_t groups = routes * (n / 32);
  if (!groups) return;
  swiglu_kernel<<<(groups + kWarps - 1) / kWarps, 32 * kWarps, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<half*>(out.data_ptr()), quantize ? reinterpret_cast<Q8F*>(q8.data_ptr()) : nullptr,
      reinterpret_cast<const half*>(gate_up.data_ptr()), int(n), int(routes));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {
void mxfp4_down_impl(torch::Tensor out, const torch::Tensor& inter, const torch::Tensor& iq, const torch::Tensor& ids,
                     const torch::Tensor& probs, const torch::Tensor& down, const torch::Tensor& down_s,
                     bool q8, bool partials) {
  check_cuda(out, "out", partials ? at::kFloat : at::kHalf);
  check_cuda(ids, "ids", at::kInt);
  if (!partials) check_cuda(probs, "probs", at::kFloat);
  check_cuda(down, "down", at::kByte);
  check_cuda(down_s, "down_s", at::kByte);
  TORCH_CHECK(down.dim() == 3 && down_s.dim() == 3, "down bank [E, H, K/2] with scales [E, H, K/32]");
  const int64_t experts = down.size(0), hidden = down.size(1), k = down.size(2) * 2;
  TORCH_CHECK(down_s.size(0) == experts && down_s.size(1) == hidden && down_s.size(2) * 32 == k && k % 32 == 0,
              "down scales do not cover the bank");
  TORCH_CHECK(ids.dim() == 2 && (partials || probs.sizes() == ids.sizes()) && ids.size(1) <= 8, "ids/probs [M, top]");
  const int64_t m = ids.size(0), top = ids.size(1);
  TORCH_CHECK(m <= 65535 && hidden % kWarps == 0, "down launch geometry");
  if (partials) {
    TORCH_CHECK(top > 0 && experts > 0 && experts <= std::numeric_limits<int>::max()
                && hidden > 0 && hidden <= std::numeric_limits<int>::max() && k > 0 && k <= 48 * 1024 / (top * 4)
                && out.dim() == 3 && out.size(0) == top && out.size(1) == m && out.size(2) == hidden,
                "down partials require FP32 [top, M, H]");
    for (const auto* input : {&ids, &inter, &down, &down_s}) {
      TORCH_CHECK(input->device() == out.device(), "down partials require same-device inputs");
      at::assert_no_overlap(out, *input);
    }
  } else {
    TORCH_CHECK(out.dim() == 2 && out.size(0) == m && out.size(1) == hidden, "down output shape");
  }
  TORCH_CHECK(reinterpret_cast<uintptr_t>(down.data_ptr()) % 16 == 0 && (k / 2) % 16 == 0, "16-byte aligned rows");
  if (q8) {
    check_cuda(iq, "iq", at::kByte);
    TORCH_CHECK(iq.numel() == m * top * (k / 32) * 36, "down q8 shape");
  } else {
    check_cuda(inter, "inter", at::kHalf);
    TORCH_CHECK(inter.numel() == m * top * k && top * k * 4 <= 48 * 1024, "down activation shape");
  }
  const c10::cuda::CUDAGuard guard(ids.device());
  if (!m) return;
  dim3 grid(hidden / kWarps, m);
  auto stream = at::cuda::getCurrentCUDAStream();
  void* o = out.data_ptr();
  if (partials) {
    down_kernel<false, true><<<grid, 32 * kWarps, top * k * 4, stream>>>(
        o, reinterpret_cast<const half*>(inter.data_ptr()), nullptr, ids.data_ptr<int>(), nullptr,
        down.data_ptr<uint8_t>(), down_s.data_ptr<uint8_t>(), int(hidden), int(k), int(top), int(experts));
  } else if (q8) {
    down_kernel<true><<<grid, 32 * kWarps, 0, stream>>>(
        o, nullptr, reinterpret_cast<const Q8F*>(iq.data_ptr()), ids.data_ptr<int>(), probs.data_ptr<float>(),
        down.data_ptr<uint8_t>(), down_s.data_ptr<uint8_t>(), int(hidden), int(k), int(top), int(experts));
  } else {
    down_kernel<false><<<grid, 32 * kWarps, top * k * 4, stream>>>(
        o, reinterpret_cast<const half*>(inter.data_ptr()), nullptr, ids.data_ptr<int>(), probs.data_ptr<float>(),
        down.data_ptr<uint8_t>(), down_s.data_ptr<uint8_t>(), int(hidden), int(k), int(top), int(experts));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

void mxfp4_down(torch::Tensor out, const torch::Tensor& inter, const torch::Tensor& iq, const torch::Tensor& ids,
                const torch::Tensor& probs, const torch::Tensor& down, const torch::Tensor& down_s, bool q8) {
  mxfp4_down_impl(out, inter, iq, ids, probs, down, down_s, q8, false);
}

void mxfp4_down_partials(torch::Tensor out, const torch::Tensor& inter, const torch::Tensor& ids,
                         const torch::Tensor& down, const torch::Tensor& down_s) {
  mxfp4_down_impl(out, inter, {}, ids, {}, down, down_s, false, true);
}

void mxfp4_finish_partials(torch::Tensor out, const torch::Tensor& planes, const torch::Tensor& ids,
                           const torch::Tensor& probs, int experts) {
  check_cuda(out, "out", at::kFloat);
  check_cuda(planes, "complete branch planes", at::kFloat);
  check_cuda(ids, "global ids", at::kLong);
  check_cuda(probs, "route probabilities", at::kFloat);
  TORCH_CHECK(ids.dim() == 2 && ids.size(1) >= 1 && ids.size(1) <= 8 && probs.sizes() == ids.sizes()
              && out.dim() == 2 && out.size(0) == ids.size(0) && out.size(1) > 0 && experts > 0
              && planes.dim() == 3 && planes.size(0) == ids.size(1) + 1
              && planes.size(1) == out.size(0) && planes.size(2) == out.size(1),
              "complete partials require [top+1, M, H] planes and matching ids/probabilities");
  for (const auto* input : {&planes, &ids, &probs}) {
    TORCH_CHECK(input->device() == out.device(), "complete partials require same-device inputs");
    at::assert_no_overlap(out, *input);
  }
  if (!out.numel()) return;
  constexpr int threads = 256;
  const int64_t blocks = 1 + (out.numel() - 1) / threads;
  TORCH_CHECK(blocks <= std::numeric_limits<int>::max(), "complete partials launch is too large");
  const c10::cuda::CUDAGuard guard(out.device());
  finish_partials_kernel<<<int(blocks), threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      out.data_ptr<float>(), planes.data_ptr<float>(), ids.data_ptr<int64_t>(), probs.data_ptr<float>(), out.numel(),
      out.size(1), int(ids.size(1)), experts);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
