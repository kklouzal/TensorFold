// Bindings for the SM70 small-row weight-streaming dense GEMM (dense_small.cu).
#include <torch/extension.h>
#include <vector>

std::vector<int64_t> dense_small_requirement(int64_t, int64_t, int64_t);
std::vector<int64_t> dense_small_bound();
void dense_small(const at::Tensor&, const at::Tensor&, at::Tensor, const at::Tensor&, const at::Tensor&);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("requirement", &dense_small_requirement);
    m.def("dense_small", &dense_small);
    m.def("bound", &dense_small_bound);
}
