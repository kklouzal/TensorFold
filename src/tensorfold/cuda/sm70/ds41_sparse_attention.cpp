#include <torch/extension.h>
void ds41_sparse_attend(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                        std::optional<torch::Tensor>, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor,
                        torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, double, int64_t, int64_t, int64_t,
                        int64_t, int64_t);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("attend", &ds41_sparse_attend); }
