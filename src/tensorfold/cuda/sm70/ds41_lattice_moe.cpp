// SPDX-License-Identifier: Apache-2.0
// pybind entry points of the DeepSeek-V4.1 TurboMind lattice prefill MoE extension (see ds41_lattice_moe.py).
#include <torch/extension.h>
#include <optional>
#include <string>

namespace ds41kern {
void lattice_gemm_op(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t,
                     int64_t, int64_t, std::optional<torch::Tensor>, int64_t, int64_t);
std::string variant_name_op(int64_t, int64_t);
}  // namespace ds41kern
void gguf_prefill_transcode_sm70(torch::Tensor, torch::Tensor, int64_t, int64_t, torch::Tensor,
                                 std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, int64_t, int64_t,
                                 int64_t, int64_t, int64_t, int64_t, torch::Tensor, bool);
void ds41_lattice_rows(const torch::Tensor&, torch::Tensor, torch::Tensor, int64_t);
void ds41_lattice_tile_lists(const torch::Tensor&, torch::Tensor, torch::Tensor, int64_t, int64_t);
void ds41_lattice_swiglu(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                         torch::Tensor, double);
void ds41_lattice_combine(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, torch::Tensor, int64_t);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("lattice_gemm", &ds41kern::lattice_gemm_op);
  m.def("variant_name", &ds41kern::variant_name_op);
  m.def("transcode", &gguf_prefill_transcode_sm70);
  m.def("rows", &ds41_lattice_rows);
  m.def("tile_lists", &ds41_lattice_tile_lists);
  m.def("swiglu", &ds41_lattice_swiglu);
  m.def("combine", &ds41_lattice_combine);
}
