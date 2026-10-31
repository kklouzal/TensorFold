#include <torch/extension.h>
#include "nvfp4_layout.cuh"
#include <cub/version.cuh>

void tf_sm70_nvfp4_decode(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                         at::Tensor, int64_t, int64_t, int64_t, int64_t);
void tf_sm70_nvfp4_matmul(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor);
void tf_sm70_nvfp4_experts(const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.attr("numerical_profile") = TF_SM70_NVFP4_PROFILE;
    m.attr("expert_abi") = TF_SM70_NVFP4_EXPERT_ABI;
    m.attr("expert_descriptor_words") = TF_SM70_NVFP4_EXPERT_WORDS;
    m.attr("cub_version") = CUB_VERSION;
    m.def("decode_tile", &tf_sm70_nvfp4_decode);
    m.def("matmul", &tf_sm70_nvfp4_matmul);
    m.def("experts", &tf_sm70_nvfp4_experts);
}
