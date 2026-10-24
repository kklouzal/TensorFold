#include <torch/extension.h>

at::Tensor gemma_b16_head_cg(const at::Tensor& x, const at::Tensor& w, int64_t rows);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemma_b16_head_cg", &gemma_b16_head_cg, "prepared Gemma BF16 head with weight-only L2 cache loads");
}
