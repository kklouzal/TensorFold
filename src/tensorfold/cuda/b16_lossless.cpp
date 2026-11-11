// Private lossless BF16 operator; original target/checkpoint paths do not import it.
#include <torch/extension.h>
at::Tensor lossless_unpack(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, int64_t);
at::Tensor lossless_rows(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, int64_t, int64_t);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("unpack", &lossless_unpack);
    module.def("matmul_rows", &lossless_rows);
}
