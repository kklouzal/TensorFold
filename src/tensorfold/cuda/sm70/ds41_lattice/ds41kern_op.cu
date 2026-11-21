// SPDX-License-Identifier: Apache-2.0
// ds41kern (agent G, patch 340): torch ops around the TurboMind lattice grouped-GEMM variants.
//
//   torch.ops.ds41kern.lattice_gemm(out, input, offsets, weight_ptrs, stats_ptrs, decoder, n,
//                                   num_experts, group_size, a_indices, variant, swizzle)
//     Same contract as torch.ops._C.gguf_grouped_prefill_gemm_sm70_out(family 0, gated False)
//     (csrc/sm70_turbomind/ops/awq_sm70_gemm.cu:3229): out[m, n] FP16, rows = routes in expert
//     order (input rows, or input[a_indices]), canonical lattice banks given as per-expert
//     (pointer, ld) tables. Instead of TurboMind's dispatcher it launches ONE fixed kernel
//     (variant) with a fixed swizzle and splits = 1 (never split-K: keeps the K order).
//   torch.ops.ds41kern.variant_name(decoder, variant) -> str  (TurboMind kernel name + smem/occupancy)
//   torch.ops.ds41kern.max_swizzle(decoder, variant, m, n, k, num_experts) -> int
//   torch.ops.ds41kern.num_variants() -> int
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <array>
#include <memory>
#include <mutex>
#include <string>

#include "turbomind/kernels/gemm/kernel.h"
#include "turbomind/kernels/gemm/types.h"

namespace ds41kern {
using turbomind::gemm::Kernel;
constexpr int kVariants = 14;
template <int Type, int Group>
Kernel* make_variant(int v);
template <> Kernel* make_variant<16, 32>(int v);
template <> Kernel* make_variant<17, 16>(int v);
template <> Kernel* make_variant<29, 16>(int v);
}  // namespace ds41kern

namespace {

using namespace turbomind;
using namespace turbomind::gemm;

constexpr int kMaxDevices = 16;
constexpr int kTypes = 3;  // 16 (IQ2_XXS), 17 (IQ2_XS), 29 (IQ1_M)

int type_slot(int64_t decoder) {
  switch (decoder) {
    case 16: return 0;
    case 17: return 1;
    case 29: return 2;
  }
  TORCH_CHECK(false, "ds41kern: decoder ", decoder, " unsupported (16, 17, 29)");
}

int group_of(int64_t decoder) { return decoder == 16 ? 32 : 16; }

// Same values as awq_sm70_gemm.cu gguf_lattice_type_index / gguf_lattice_stat_type.
int lattice_type_index(int64_t decoder) { return decoder == 16 ? 0 : decoder == 17 ? 1 : 6; }
DataType stat_type(int64_t decoder) { return decoder == 29 ? kUint16 : kUint32; }

std::mutex g_mutex;
std::array<std::array<std::array<std::unique_ptr<Kernel>, ds41kern::kVariants>, kTypes>, kMaxDevices>
    g_kernels;

Kernel* get_kernel(int device, int64_t decoder, int64_t variant) {
  TORCH_CHECK(device >= 0 && device < kMaxDevices, "ds41kern: device index");
  TORCH_CHECK(variant >= 0 && variant < ds41kern::kVariants, "ds41kern: variant ", variant,
              " out of range [0, ", ds41kern::kVariants, ")");
  const int t = type_slot(decoder);
  std::lock_guard<std::mutex> lock(g_mutex);
  auto& slot = g_kernels[device][t][variant];
  if (!slot) {
    // KernelImpl's constructor queries / sets function attributes on the current device.
    Kernel* k = t == 0 ? ds41kern::make_variant<16, 32>(variant)
              : t == 1 ? ds41kern::make_variant<17, 16>(variant)
                       : ds41kern::make_variant<29, 16>(variant);
    TORCH_CHECK(k != nullptr, "ds41kern: variant ", variant, " not built");
    slot.reset(k);
  }
  return slot.get();
}

// gguf_lattice_layouts (awq_sm70_gemm.cu:3022) with GetGgufLatticeConverters' order/pack
// (convert_v3.cu:104): weights Cvt<uint16_t, uint2_t> kRowMajor, pack HMMA_884|OPERAND_B|1;
// stats kColMajor, pack HMMA_884|OPERAND_V|1.
std::array<MatrixLayout, 2> lattice_layouts(int n, int k, int64_t decoder, int group) {
  MatrixLayout w{kUint2, kRowMajor, n, k, k};
  std::swap(w.rows, w.cols);
  w.order = ~w.order;
  w.pack = (Pack)(HMMA_884 | OPERAND_B | 1);
  MatrixLayout s{stat_type(decoder), kColMajor, n, k / group, n};
  std::swap(s.rows, s.cols);
  s.order = ~s.order;
  s.pack = (Pack)(HMMA_884 | OPERAND_V | 1);
  return {w, s};
}

void lattice_gemm(torch::Tensor out, torch::Tensor input, torch::Tensor offsets,
                  torch::Tensor weight_ptrs, torch::Tensor stats_ptrs, int64_t decoder, int64_t n,
                  int64_t num_experts, int64_t group_size, std::optional<torch::Tensor> a_indices,
                  int64_t variant, int64_t swizzle) {
  TORCH_CHECK(input.is_cuda() && out.device() == input.device() &&
                  offsets.device() == input.device() && weight_ptrs.device() == input.device() &&
                  stats_ptrs.device() == input.device(),
              "ds41kern: tensors must share a CUDA device");
  // TensorFold: variants 9/10 (exact decode, FP32 C) write FP32; every other variant FP16.
  const bool float_out = variant == 9 || variant == 10;
  TORCH_CHECK(input.scalar_type() == torch::kFloat16 &&
                  out.scalar_type() == (float_out ? torch::kFloat32 : torch::kFloat16) &&
                  offsets.scalar_type() == torch::kInt32 &&
                  weight_ptrs.scalar_type() == torch::kUInt8 &&
                  stats_ptrs.scalar_type() == torch::kUInt8 && input.dim() == 2 &&
                  out.dim() == 2 && input.stride(1) == 1 && out.stride(1) == 1 &&
                  offsets.is_contiguous() && weight_ptrs.is_contiguous() &&
                  stats_ptrs.is_contiguous(),
              "ds41kern: FP16 rows and prepared pointer tables required");
  TORCH_CHECK(group_size == group_of(decoder), "ds41kern: group ", group_size,
              " wrong for decoder ", decoder);
  const bool indexed = a_indices.has_value();
  TORCH_CHECK(!indexed || (a_indices->is_cuda() && a_indices->scalar_type() == torch::kInt32 &&
                           a_indices->is_contiguous() && a_indices->device() == input.device()),
              "ds41kern: a_indices must be contiguous int32 CUDA");
  const int64_t m = indexed ? a_indices->numel() : input.size(0);
  const int64_t k = input.size(1);
  TORCH_CHECK(num_experts > 0 && num_experts <= INT_MAX && m <= INT_MAX && n > 0 && n <= INT_MAX &&
                  k > 0 && k <= INT_MAX && n % 32 == 0 && k % group_size == 0 &&
                  out.size(0) == m && out.size(1) == n && offsets.numel() == num_experts + 1 &&
                  weight_ptrs.numel() == num_experts * 16 && stats_ptrs.numel() == num_experts * 16,
              "ds41kern: descriptor shape mismatch");
  TORCH_CHECK(swizzle >= 0 && swizzle <= 7, "ds41kern: swizzle");
  if (m == 0) return;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const auto stream = at::cuda::getCurrentCUDAStream();
  Kernel* kernel = get_kernel(input.get_device(), decoder, variant);

  auto layouts = lattice_layouts((int)n, (int)k, decoder, (int)group_size);
  for (auto& layout : layouts) {
    layout.ld = 0;
    layout.num = (int)num_experts;
  }
  MatrixLayout a{kHalf, kRowMajor, (int)m, (int)k, (int)input.stride(0)};
  MatrixLayout d{float_out ? kFloat : kHalf, kRowMajor, (int)m, (int)n, (int)out.stride(0)};
  a.num = d.num = (int)num_experts;
  a.offsets = d.offsets = offsets.data_ptr<int>();
  a.idxs = indexed ? a_indices->data_ptr<int>() : nullptr;

  Operation operation{};
  operation.quant_b = {static_cast<QuantType>(8 + lattice_type_index(decoder)), (int)group_size};
  operation.epilogue = Epilogue::kNone;
  operation.quant_a = {QuantType::kNone, 0};
  operation.batch_dim = 0;

  Workspace workspace{};  // no partials / barriers: GetMaxSplits() == 1, never split-K
  const int max_swizzle = kernel->GetMaxSwizzle({(int)m, (int)n, (int)k, (int)num_experts});
  const int result = kernel->Launch(operation, 1.f, input.data_ptr(), a, nullptr, {},
                                    weight_ptrs.data_ptr(), layouts[0], stats_ptrs.data_ptr(),
                                    layouts[1], 0.f, out.data_ptr(), d, out.data_ptr(), d,
                                    std::min<int>((int)swizzle, max_swizzle), 1, workspace, stream);
  TORCH_CHECK(result == 0, "ds41kern: TurboMind launch failed");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::string variant_name(int64_t decoder, int64_t variant) {
  const int device = c10::cuda::current_device();
  Kernel* kernel = get_kernel(device, decoder, variant);
  const auto t = kernel->cta_tile_size();
  const auto w = kernel->warp_tile_size();
  return "cta " + std::to_string(t.x) + "x" + std::to_string(t.y) + "x" + std::to_string(t.z) +
         " warp " + std::to_string(w.x) + "x" + std::to_string(w.y) + "x" + std::to_string(w.z) +
         " smem " + std::to_string(kernel->smem_size()) + " ctas/sm " +
         std::to_string(kernel->info().max_active_ctas) + " regs " +
         std::to_string(kernel->info().attr.numRegs) + " | " + kernel->name();
}

int64_t max_swizzle(int64_t decoder, int64_t variant, int64_t m, int64_t n, int64_t k,
                    int64_t num_experts) {
  const int device = c10::cuda::current_device();
  return get_kernel(device, decoder, variant)
      ->GetMaxSwizzle({(int)m, (int)n, (int)k, (int)num_experts});
}

int64_t num_variants() { return ds41kern::kVariants; }

}  // namespace

// TensorFold (vendoring): the TORCH_LIBRARY(ds41kern) registration is replaced by these entry points for the
// ds41_lattice_moe pybind module; the functions above are unchanged.
namespace ds41kern {
void lattice_gemm_op(torch::Tensor out, torch::Tensor input, torch::Tensor offsets, torch::Tensor weight_ptrs,
                     torch::Tensor stats_ptrs, int64_t decoder, int64_t n, int64_t num_experts, int64_t group_size,
                     std::optional<torch::Tensor> a_indices, int64_t variant, int64_t swizzle) {
  lattice_gemm(out, input, offsets, weight_ptrs, stats_ptrs, decoder, n, num_experts, group_size, a_indices, variant,
               swizzle);
}
std::string variant_name_op(int64_t decoder, int64_t variant) { return variant_name(decoder, variant); }
}  // namespace ds41kern
