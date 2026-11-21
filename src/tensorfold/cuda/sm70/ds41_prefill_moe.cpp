#include <torch/extension.h>
void ds41_prefill_experts(torch::Tensor, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                          const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, int64_t, int64_t,
                          int64_t, int64_t, int64_t, bool);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("experts", &ds41_prefill_experts); }
