#include <torch/extension.h>
#include <vector>

#include "ds41_down_fragment_pack.h"

int64_t ds41_decode_moe_workspace_bytes(int64_t);
void ds41_decode_moe_init(torch::Tensor);
std::vector<std::vector<int64_t>> ds41_decode_moe_variants();
void ds41_decode_moe_experts(torch::Tensor, torch::Tensor, const torch::Tensor&, const torch::Tensor&,
                             const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                             const torch::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t);
void ds41_decode_moe_experts_fragments(torch::Tensor, torch::Tensor, const torch::Tensor&, const torch::Tensor&,
                                       const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                                       const torch::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t, bool);
void ds41_decode_moe_experts_fragments_shared(torch::Tensor, torch::Tensor, const torch::Tensor&,
                                              const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                                              const torch::Tensor&, const torch::Tensor&, int64_t, int64_t, int64_t,
                                              int64_t, int64_t, const torch::Tensor&, const torch::Tensor&);
void ds41_decode_moe_prep(torch::Tensor, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                          int64_t, int64_t);
void ds41_decode_moe_gate_up(torch::Tensor, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                             int64_t, int64_t);
void ds41_decode_moe_down(torch::Tensor, const torch::Tensor&, const torch::Tensor&, int64_t, int64_t, int64_t);
void ds41_decode_moe_combine(torch::Tensor, torch::Tensor, const torch::Tensor&);

namespace {
using tensorfold::ds41_down_fragments::Layout;
using tensorfold::ds41_down_fragments::groups_per_fragment;
using tensorfold::ds41_down_fragments::pack_rows;
using tensorfold::ds41_down_fragments::source_blocks;

int compact_down_row_bytes(int64_t kind) {
  switch (kind) {
    case 16: return Layout<16>::compact_row_bytes;
    case 17: return Layout<17>::compact_row_bytes;
    case 29: return Layout<29>::compact_row_bytes;
    default: TORCH_CHECK(false, "compact ETP4 packing requires an IQ2_XXS, IQ2_XS or IQ1_M carrier");
  }
}

template <int Type>
void pack_checked(const torch::Tensor& source, const torch::Tensor& destination, int start_group) {
  constexpr int input_stride = source_blocks * Layout<Type>::original_block_bytes;
  constexpr int output_stride = Layout<Type>::compact_row_bytes;
  const int64_t rows = source.numel() / input_stride;
  TORCH_CHECK(source.numel() % input_stride == 0 && destination.numel() == rows * output_stride,
              "compact ETP4 packing needs complete source rows and matching compact rows");
  if (rows == 0)
    return;

  const auto* input = source.data_ptr<uint8_t>();
  auto* output = destination.data_ptr<uint8_t>();
  const auto a = reinterpret_cast<uintptr_t>(input), b = reinterpret_cast<uintptr_t>(output);
  TORCH_CHECK(a < b ? b - a >= static_cast<uintptr_t>(source.numel())
                    : a - b >= static_cast<uintptr_t>(destination.numel()),
              "compact ETP4 packing needs disjoint source and destination spans");
  pack_rows<Type>(input, output, static_cast<std::size_t>(rows), start_group);
}

void pack_down_fragments(const torch::Tensor& source, const torch::Tensor& destination, int64_t kind,
                         int64_t start_group) {
  TORCH_CHECK(source.device().is_cpu() && destination.device().is_cpu() &&
                  source.scalar_type() == torch::kUInt8 && destination.scalar_type() == torch::kUInt8 &&
                  source.dim() == 1 && destination.dim() == 1 && source.is_contiguous() && destination.is_contiguous(),
              "compact ETP4 packing needs flat contiguous CPU uint8 buffers");
  TORCH_CHECK(start_group >= 0 && start_group <= 3 * groups_per_fragment && start_group % groups_per_fragment == 0,
              "compact ETP4 packing needs a legal rank cut");
  switch (kind) {
    case 16: return pack_checked<16>(source, destination, int(start_group));
    case 17: return pack_checked<17>(source, destination, int(start_group));
    case 29: return pack_checked<29>(source, destination, int(start_group));
    default: TORCH_CHECK(false, "compact ETP4 packing requires an IQ2_XXS, IQ2_XS or IQ1_M carrier");
  }
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("workspace_bytes", &ds41_decode_moe_workspace_bytes, pybind11::arg("intermediate") = 576);
  m.def("init", &ds41_decode_moe_init);
  m.def("variants", &ds41_decode_moe_variants);
  m.def("experts", &ds41_decode_moe_experts);
  m.def("experts_fragments", &ds41_decode_moe_experts_fragments);
  m.def("experts_fragments_shared", &ds41_decode_moe_experts_fragments_shared);
  m.def("compact_down_row_bytes", &compact_down_row_bytes);
  m.def("pack_down_fragments", &pack_down_fragments, pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("prep", &ds41_decode_moe_prep);
  m.def("gate_up", &ds41_decode_moe_gate_up);
  m.def("down", &ds41_decode_moe_down);
  m.def("combine", &ds41_decode_moe_combine);
}
