// SPDX-License-Identifier: Apache-2.0
// DeepSeek-V4.1 routed-expert prefill on the friend's TurboMind lattice path (1Cat-vLLM patch 140 + 340), the
// glue kernels. Ported from patch 140's Triton kernels (local_ds41_pf_moe.py: _pf_rows, _pf_tile_lists,
// _pf_swiglu, _pf_combine) with the same expressions and rounding points; TensorFold changes for the EP2 x ETP4
// layout: routes outside this rank's expert half (probability 0) are sorted behind a sentinel expert E, so
// offsets[E] = valid routes; the SwiGLU skips rows >= offsets[E] and the combine skips their positions (they
// contribute exactly +/-0 in the reference). No host synchronization.
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <initializer_list>

namespace {

// rows[p] = token of sorted route p; inverse[order[p]] = p (patch 140 _pf_rows).
__global__ void rows_kernel(const int64_t* order, int32_t* rows, int32_t* inverse, int64_t routes, int top) {
  const int64_t p = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (p >= routes) return;
  const int64_t o = order[p];
  rows[p] = int32_t(o / top);
  inverse[o] = int32_t(p);
}

// Per tile of `tile` experts: the local indices of its experts that have routes, ascending (patch 140
// _pf_tile_lists). One warp-sized sequential scan per tile (<= 384 experts).
__global__ void tile_lists_kernel(const int32_t* offsets, int32_t* lists, int32_t* counts, int experts, int tile) {
  const int t = blockIdx.x;
  if (threadIdx.x != 0) return;
  int used = 0;
  for (int local = 0; local < tile; ++local) {
    const int e = t * tile + local;
    if (e >= experts) break;
    if (offsets[e + 1] > offsets[e]) lists[t * tile + used++] = local;
  }
  counts[t] = used;
}

// hidden[p] = fp16((silu(min(g, L)) * clamp(u, -L, L)) * w[order[p]]) for p < offsets[E]; proj rows are gate|up
// interleaved (column 2c gate, 2c + 1 up). silu(g) = g / (1 + expf(-g)) with IEEE division; no FMA contraction
// (patch 140 _pf_swiglu, enable_fp_fusion=False, libdevice expf).
__global__ void swiglu_kernel(const half* proj, const float* weights, const int64_t* order, const int32_t* valid,
                              half* hidden, int width, float limit) {
  const int64_t p = blockIdx.x;
  if (p >= *valid) return;
  const float w = weights[order[p]];
  for (int c = blockIdx.y * blockDim.x + threadIdx.x; c < width; c += gridDim.y * blockDim.x) {
    float g = __half2float(proj[p * 2 * width + 2 * c]);
    float u = __half2float(proj[p * 2 * width + 2 * c + 1]);
    g = g > limit ? limit : g;
    u = u < -limit ? -limit : u;
    u = u > limit ? limit : u;
    const float s = __fdiv_rn(g, __fadd_rn(1.0f, expf(-g)));
    hidden[p * width + c] = __float2half_rn(__fmul_rn(__fmul_rn(s, u), w));
  }
}

// out[t] = fp16(sum_j fp32(down[inverse[t * top + j]])), j ascending, positions >= offsets[E] skipped
// (patch 140 _pf_combine). Eight columns per thread.
__global__ void combine_kernel(const half* down, const int32_t* inverse, const int32_t* valid, half* out,
                               int64_t tokens, int hidden, int top) {
  const int chunks = hidden / 8;
  const int32_t limit = *valid;
  for (int64_t at = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; at < tokens * chunks;
       at += int64_t(gridDim.x) * blockDim.x) {
    const int64_t t = at / chunks;
    const int c = int(at % chunks) * 8;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int j = 0; j < top; ++j) {
      const int32_t p = inverse[t * top + j];
      if (p >= limit) continue;
      const uint4 raw = *reinterpret_cast<const uint4*>(down + int64_t(p) * hidden + c);
      const half* v = reinterpret_cast<const half*>(&raw);
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[i] = __fadd_rn(acc[i], __half2float(v[i]));
    }
    __align__(16) half h[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) h[i] = __float2half_rn(acc[i]);
    *reinterpret_cast<uint4*>(out + t * hidden + c) = *reinterpret_cast<const uint4*>(h);
  }
}

void same_device(const torch::Tensor& a, const torch::Tensor& b) {
  TORCH_CHECK(a.is_cuda() && a.device() == b.device(), "ds41 lattice MoE: tensors must share one CUDA device");
}

}  // namespace

void ds41_lattice_rows(const torch::Tensor& order, torch::Tensor rows, torch::Tensor inverse, int64_t top) {
  same_device(order, rows);
  same_device(order, inverse);
  TORCH_CHECK(order.scalar_type() == torch::kLong && rows.scalar_type() == torch::kInt &&
                  inverse.scalar_type() == torch::kInt && order.is_contiguous() && rows.numel() == order.numel() &&
                  inverse.numel() == order.numel() && top > 0,
              "ds41 lattice MoE rows: int64 order, int32 rows/inverse of equal length");
  const int64_t routes = order.numel();
  if (!routes) return;
  const c10::cuda::CUDAGuard guard(order.device());
  rows_kernel<<<unsigned((routes + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      order.data_ptr<int64_t>(), rows.data_ptr<int32_t>(), inverse.data_ptr<int32_t>(), routes, int(top));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ds41_lattice_tile_lists(const torch::Tensor& offsets, torch::Tensor lists, torch::Tensor counts, int64_t experts,
                             int64_t tile) {
  same_device(offsets, lists);
  same_device(offsets, counts);
  const int64_t tiles = (experts + tile - 1) / tile;
  TORCH_CHECK(offsets.scalar_type() == torch::kInt && lists.scalar_type() == torch::kInt &&
                  counts.scalar_type() == torch::kInt && offsets.numel() >= experts + 1 &&
                  lists.numel() == tiles * tile && counts.numel() == tiles && tile > 0,
              "ds41 lattice MoE tile lists: geometry");
  const c10::cuda::CUDAGuard guard(offsets.device());
  tile_lists_kernel<<<unsigned(tiles), 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      offsets.data_ptr<int32_t>(), lists.data_ptr<int32_t>(), counts.data_ptr<int32_t>(), int(experts), int(tile));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ds41_lattice_swiglu(const torch::Tensor& proj, const torch::Tensor& weights, const torch::Tensor& order,
                         const torch::Tensor& valid, torch::Tensor hidden, double limit) {
  for (const torch::Tensor* t : std::initializer_list<const torch::Tensor*>{&weights, &order, &valid, &hidden})
    same_device(proj, *t);
  const int64_t routes = proj.size(0), width = hidden.size(1);
  TORCH_CHECK(proj.scalar_type() == torch::kHalf && hidden.scalar_type() == torch::kHalf &&
                  weights.scalar_type() == torch::kFloat && order.scalar_type() == torch::kLong &&
                  valid.scalar_type() == torch::kInt && proj.is_contiguous() && hidden.is_contiguous() &&
                  weights.is_contiguous() && order.numel() == routes && weights.numel() == routes &&
                  hidden.size(0) == routes && proj.size(1) == 2 * width,
              "ds41 lattice MoE SwiGLU: FP16 [R, 2W] -> [R, W], FP32 weights, int64 order");
  if (!routes) return;
  const c10::cuda::CUDAGuard guard(proj.device());
  const dim3 grid(unsigned(routes), unsigned((width + 255) / 256));
  swiglu_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(proj.data_ptr<at::Half>()), weights.data_ptr<float>(),
      order.data_ptr<int64_t>(), valid.data_ptr<int32_t>(), reinterpret_cast<half*>(hidden.data_ptr<at::Half>()),
      int(width), float(limit));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ds41_lattice_combine(const torch::Tensor& down, const torch::Tensor& inverse, const torch::Tensor& valid,
                          torch::Tensor out, int64_t top) {
  for (const torch::Tensor* t : std::initializer_list<const torch::Tensor*>{&inverse, &valid, &out})
    same_device(down, *t);
  const int64_t tokens = out.size(0), hidden = out.size(1);
  TORCH_CHECK(down.scalar_type() == torch::kHalf && out.scalar_type() == torch::kHalf &&
                  inverse.scalar_type() == torch::kInt && valid.scalar_type() == torch::kInt &&
                  down.is_contiguous() && out.is_contiguous() && down.size(1) == hidden && hidden % 8 == 0 &&
                  inverse.numel() == tokens * top && down.size(0) == tokens * top,
              "ds41 lattice MoE combine: FP16 [T * top, H] -> [T, H]");
  if (!tokens) return;
  const c10::cuda::CUDAGuard guard(down.device());
  const int64_t work = tokens * (hidden / 8);
  combine_kernel<<<unsigned(std::min<int64_t>((work + 255) / 256, 65535)), 256, 0,
                   at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(down.data_ptr<at::Half>()), inverse.data_ptr<int32_t>(),
      valid.data_ptr<int32_t>(), reinterpret_cast<half*>(out.data_ptr<at::Half>()), tokens, int(hidden), int(top));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
