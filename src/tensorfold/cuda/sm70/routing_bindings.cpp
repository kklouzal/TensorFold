#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "../kernels/tensor_contracts.cuh"

void routing_select_cuda(const at::Tensor&, at::Tensor&);
void routing_route_cuda(const at::Tensor&, at::Tensor&, at::Tensor&);
void routing_logits_cuda(const at::Tensor&, const at::Tensor&, at::Tensor&);

void select_experts(const at::Tensor& logits, at::Tensor indices) {
    TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == at::kFloat && logits.dim() == 2
                && logits.size(1) >= 512 && logits.stride(1) == 1 && logits.stride(0) >= logits.size(1),
                "SM70 router requires FP32 CUDA rows with 512 contiguous expert columns");
    TORCH_CHECK(indices.is_cuda() && indices.device() == logits.device()
                && indices.scalar_type() == at::kLong && indices.is_contiguous() && indices.dim() == 2
                && indices.size(0) == logits.size(0) && indices.size(1) > 0 && indices.size(1) <= 32,
                "SM70 router output requires same-device contiguous int64 [rows,1..32]");
    const auto* prop = at::cuda::getDeviceProperties(logits.get_device());
    TORCH_CHECK(prop->major == 7 && prop->minor == 0, "SM70 router requires Volta");
    tensorfold::check_disjoint(indices, {&logits}, "router indices must not overlap logits");
    c10::cuda::CUDAGuard guard(logits.device());
    routing_select_cuda(logits, indices);
}

void route_experts(const at::Tensor& logits, at::Tensor picks, at::Tensor weights) {
    TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == at::kFloat && logits.dim() == 2
                && logits.size(1) >= 513 && logits.stride(1) == 1 && logits.stride(0) >= logits.size(1),
                "SM70 router requires FP32 CUDA rows with contiguous expert/shared columns");
    TORCH_CHECK(picks.is_cuda() && picks.device() == logits.device() && picks.scalar_type() == at::kInt
                && picks.is_contiguous() && picks.dim() == 2 && picks.size(0) == logits.size(0)
                && picks.size(1) >= 2 && picks.size(1) <= 33,
                "SM70 router picks require same-device contiguous int32 [rows,2..33]");
    TORCH_CHECK(weights.is_cuda() && weights.device() == logits.device() && weights.scalar_type() == at::kFloat
                && weights.is_contiguous() && weights.sizes() == picks.sizes(),
                "SM70 router weights require matching contiguous FP32 slots");
    const auto* prop = at::cuda::getDeviceProperties(logits.get_device());
    TORCH_CHECK(prop->major == 7 && prop->minor == 0, "SM70 router requires Volta");
    tensorfold::check_disjoint(picks, {&logits, &weights}, "router picks must not overlap operands");
    tensorfold::check_disjoint(weights, {&logits}, "router weights must not overlap logits");
    c10::cuda::CUDAGuard guard(logits.device());
    routing_route_cuda(logits, picks, weights);
}

void router_logits(const at::Tensor& x, const at::Tensor& weight, at::Tensor output) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.dim() == 2 && x.size(0) <= 32
                && x.size(1) == 2560 && x.stride(1) == 1 && x.stride(0) >= 2560,
                "SM70 serial router requires at most32 channel-contiguous FP16 rows of2560");
    TORCH_CHECK(weight.device() == x.device() && weight.scalar_type() == at::kFloat
                && weight.sizes() == at::IntArrayRef({513, 2560}) && weight.is_contiguous(),
                "SM70 serial router requires contiguous FP32 weights513x2560");
    TORCH_CHECK(output.device() == x.device() && output.scalar_type() == at::kFloat
                && output.sizes() == at::IntArrayRef({x.size(0), 513}) && output.is_contiguous(),
                "SM70 serial router output requires contiguous matching FP32 rows");
    const auto* prop = at::cuda::getDeviceProperties(x.get_device());
    TORCH_CHECK(prop->major == 7 && prop->minor == 0, "SM70 serial router requires Volta");
    tensorfold::check_disjoint(output, {&x, &weight}, "serial router output must not overlap inputs");
    c10::cuda::CUDAGuard guard(x.device());
    routing_logits_cuda(x, weight, output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("logits", &router_logits, "FP16 decode rows with the qualified serial FP32 GEMV arithmetic");
    module.def("route", &route_experts, "Stable selection and Torch-rounded FP16 routing weights");
    module.def("select", &select_experts, "Stable integer-key top-k selection for 512 SM70 experts");
}
