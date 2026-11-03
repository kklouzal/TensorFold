// Experimental adapter to the owned Delta4 v1 field ABI.
#include <torch/extension.h>

at::Tensor delta4_vector_unpack(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                               int64_t, int64_t);
at::Tensor delta4_vector_rows(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                             const at::Tensor&, int64_t, int64_t, int64_t);
at::Tensor delta4_vector_lookup(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                               const at::Tensor&, int64_t, int64_t, int64_t);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("unpack", &delta4_vector_unpack);
    module.def("matmul_rows", &delta4_vector_rows);
    module.def("lookup", &delta4_vector_lookup);
}
