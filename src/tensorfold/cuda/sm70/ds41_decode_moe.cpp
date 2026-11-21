#include <torch/extension.h>
#include <vector>
int64_t ds41_decode_moe_workspace_bytes();
void ds41_decode_moe_init(torch::Tensor);
std::vector<std::vector<int64_t>> ds41_decode_moe_variants();
void ds41_decode_moe_experts(torch::Tensor, torch::Tensor, const torch::Tensor&, const torch::Tensor&,
                             const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                             const torch::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t);
void ds41_decode_moe_prep(torch::Tensor, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                          int64_t, int64_t);
void ds41_decode_moe_gate_up(torch::Tensor, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                             int64_t, int64_t);
void ds41_decode_moe_down(torch::Tensor, const torch::Tensor&, const torch::Tensor&, int64_t, int64_t, int64_t);
void ds41_decode_moe_combine(torch::Tensor, torch::Tensor, const torch::Tensor&);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("workspace_bytes", &ds41_decode_moe_workspace_bytes);
  m.def("init", &ds41_decode_moe_init);
  m.def("variants", &ds41_decode_moe_variants);
  m.def("experts", &ds41_decode_moe_experts);
  m.def("prep", &ds41_decode_moe_prep);
  m.def("gate_up", &ds41_decode_moe_gate_up);
  m.def("down", &ds41_decode_moe_down);
  m.def("combine", &ds41_decode_moe_combine);
}
