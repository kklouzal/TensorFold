#include <torch/extension.h>

at::Tensor sm70_rms_norm(const at::Tensor&, const at::Tensor&, double);
at::Tensor sm70_rope(const at::Tensor&, const at::Tensor&, const at::Tensor&);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm", &sm70_rms_norm);
    m.def("rope", &sm70_rope);
}
