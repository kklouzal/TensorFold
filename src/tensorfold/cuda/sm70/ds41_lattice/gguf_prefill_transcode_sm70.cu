// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <torch/all.h>
#include <algorithm>
#include <limits>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "gguf_prefill_transcode.cuh"

namespace {
using namespace vllm::sm70_gguf::transcode;

void require_disjoint(const torch::Tensor& a, const torch::Tensor& b) {
  if (!a.numel() || !b.numel()) return;
  const auto x = reinterpret_cast<uintptr_t>(a.data_ptr());
  const auto y = reinterpret_cast<uintptr_t>(b.data_ptr());
  TORCH_CHECK(x < y ? y - x >= a.nbytes() : x - y >= b.nbytes(),
              "prefill transcode: input and output spans must be disjoint");
}

template <int Type, bool Exact = false>
void launch(torch::Tensor& dst_w, torch::Tensor& dst_s, int64_t w_stride,
            int64_t s_stride, const torch::Tensor& src0,
            const std::optional<torch::Tensor>& src1,
            const torch::Tensor& experts, const torch::Tensor& count,
            int64_t ctas, int64_t n_out, int64_t k, int64_t k_offset,
            int64_t src_stride, torch::Tensor& error) {
  using F = Format<Type>;
  using Packet = PacketOf<Type>;
  const int64_t groups = k / F::kGroup;
  TORCH_CHECK(k % F::kGroup == 0 && k_offset % F::kGroup == 0 &&
                  groups <= std::numeric_limits<int>::max() / kPack,
              "prefill transcode: invalid packet group width or offset");
  TORCH_CHECK(w_stride >= (k / 8) * n_out * sizeof(Packet) &&
                  s_stride >= groups * n_out * sizeof(typename F::Stat) &&
                  dst_w.numel() / w_stride >= src0.size(0) &&
                  dst_s.numel() / s_stride >= src0.size(0) &&
                  w_stride % alignof(Packet) == 0 && s_stride % alignof(typename F::Stat) == 0 &&
                  reinterpret_cast<uintptr_t>(dst_w.data_ptr()) % alignof(Packet) == 0 &&
                  reinterpret_cast<uintptr_t>(dst_s.data_ptr()) % alignof(typename F::Stat) == 0,
              "prefill transcode: incompatible destination banks");
  const int packs = static_cast<int>(n_out / kPack);
  const int64_t items = experts.numel() * packs;
  if (items == 0) return;
  const int grid = static_cast<int>(std::min<int64_t>(ctas, items));
  const size_t smem = size_t(kPack) * src_stride;
  TORCH_CHECK(smem <= std::numeric_limits<int>::max(), "prefill transcode: source pack is too large");
  auto kernel = transcode_kernel<Type, Exact>;
  if (smem > 48 * 1024)
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)));
  kernel<<<grid, kThreads, smem, at::cuda::getCurrentCUDAStream()>>>(
      dst_w.data_ptr<uint8_t>(), dst_s.data_ptr<uint8_t>(), w_stride, s_stride,
      src0.data_ptr<uint8_t>(), src1 ? src1->data_ptr<uint8_t>() : nullptr, src0.stride(0),
      int(src0.size(0)), int(experts.numel()),
      experts.data_ptr<int32_t>(), count.data_ptr<int32_t>(), packs, int(n_out),
      int(k), int(k_offset), int(src_stride), error.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

// Converts the first `count` (device int32) experts listed in `experts`
// from their original rows into a stacked canonical bank, with at most
// `ctas` CTAs. src0/src1: CUDA uint8 [num_experts, rows, bytes] carriers.
// The worklist (from tile_lists) provides unique local expert IDs.
// src1 selects gate/up interleaving. Output
// columns are source columns [k_offset, k_offset + k) (a TP slice inside
// whole source blocks). dst_w/dst_s: uint8 banks, expert e at e * w_stride /
// e * s_stride bytes.
void gguf_prefill_transcode_sm70(torch::Tensor dst_w, torch::Tensor dst_s,
                                 int64_t w_stride, int64_t s_stride,
                                 torch::Tensor src0,
                                 std::optional<torch::Tensor> src1,
                                 torch::Tensor experts, torch::Tensor count,
                                 int64_t ctas, int64_t source_type,
                                 int64_t n_out, int64_t k, int64_t k_offset, torch::Tensor error) {
  const auto device = src0.device();
  const int64_t max_int = std::numeric_limits<int>::max();
  TORCH_CHECK(src0.is_cuda(), "prefill transcode: source must be CUDA");
  for (const auto* tensor : {&src0, &dst_w, &dst_s, &experts, &count, &error})
    TORCH_CHECK(tensor->device() == device && tensor->is_contiguous(),
                "prefill transcode: tensors must be contiguous on one CUDA device");
  TORCH_CHECK(
      dst_w.scalar_type() == at::kByte && dst_s.scalar_type() == at::kByte &&
          src0.scalar_type() == at::kByte && src0.dim() == 3 &&
          experts.scalar_type() == at::kInt && count.scalar_type() == at::kInt &&
          error.scalar_type() == at::kInt && count.numel() == 1 && error.numel() == 1,
      "prefill transcode: bad dtypes");
  TORCH_CHECK(!src1 || (src1->device() == device && src1->scalar_type() == at::kByte &&
                        src1->is_contiguous() && src1->sizes() == src0.sizes()),
              "prefill transcode: bad second source");
  const int64_t src_stride = src0.size(2);
  const int sources = src1 ? 2 : 1;
  TORCH_CHECK(n_out > 0 && n_out <= max_int && n_out % kPack == 0 && n_out / sources == src0.size(1) &&
                  k > 0 && k <= max_int && k_offset >= 0 && k_offset <= max_int - k &&
                  src_stride > 0 && src_stride <= max_int && ctas > 0 && ctas <= max_int &&
                  src0.size(0) > 0 && src0.size(0) <= max_int &&
                  w_stride > 0 && s_stride > 0 && experts.numel() <= max_int,
              "prefill transcode: bad geometry");
  // The persistent grid advances a signed work index by one grid size at a time.
  const int64_t items = experts.numel() * (n_out / kPack);
  TORCH_CHECK(items + std::min(ctas, items) <= max_int, "prefill transcode: worklist is too large");
  TORCH_CHECK(src0.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(src0.data_ptr()) % 8 == 0 &&
                  (!src1 || reinterpret_cast<uintptr_t>(src1->data_ptr()) % 8 == 0),
              "prefill transcode: expert bases must be 8-byte aligned");
  for (const auto* write : {&dst_w, &dst_s, &error}) {
    for (const auto* read : {&src0, &experts, &count}) require_disjoint(*write, *read);
    if (src1) require_disjoint(*write, *src1);
  }
  require_disjoint(dst_w, dst_s);
  require_disjoint(dst_w, error);
  require_disjoint(dst_s, error);
  const at::cuda::CUDAGuard guard(dst_w.device());
  // Source rows must hold whole blocks covering [k_offset, k_offset + k).
  auto rows_ok = [&](int block, int bytes) {
    return k % 32 == 0 && src_stride % bytes == 0 &&
           src_stride / bytes * block >= k_offset + k;
  };
  // TensorFold: only the exact-sub-scale stats encoding (GroupExact) is built, IQ2_XXS / IQ2_XS / IQ1_M (the
  // friend's FP16-scale encodings and its other GGUF types were removed with his FP16 arithmetic).
  switch (source_type) {
#define EXACT(TYPE, BLOCK, BYTES)                                              \
  case TYPE:                                                                   \
    TORCH_CHECK(rows_ok(BLOCK, BYTES), "prefill transcode: GGUF type ", TYPE,  \
                " rows do not cover the requested columns");                   \
    launch<TYPE, true>(dst_w, dst_s, w_stride, s_stride, src0, src1, experts,  \
                       count, ctas, n_out, k, k_offset, src_stride, error);    \
    return;
    EXACT(16, 256, 66)
    EXACT(17, 256, 74)
    EXACT(29, 256, 56)
#undef EXACT
  }
  TORCH_CHECK(false, "prefill transcode: no exact encoding for GGUF type ", source_type);
}
