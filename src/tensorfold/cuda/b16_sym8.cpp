// Unconnected exact-word Sym8 prototype, no serving registration.
#include <torch/extension.h>
at::Tensor sym8_matmul(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, int64_t);
at::Tensor sym8_lookup(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, int64_t);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &sym8_matmul);
    m.def("lookup", &sym8_lookup);
}
