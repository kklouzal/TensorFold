#include <torch/extension.h>
#include <ATen/MemoryOverlap.h>
#include <c10/cuda/CUDAGuard.h>
#include <limits>

using T = at::Tensor;
void compact_publish_cuda(const T&, T&, int64_t, const T&, int64_t, int64_t, T&, int);

void publish(const T& source, T destination, int64_t at, const T& generations,
             int64_t slot, int64_t owner, T status, int layer) {
    TORCH_CHECK(source.is_cuda() && source.is_contiguous() && destination.is_cuda()
        && destination.device() == source.device() && destination.is_contiguous()
        && source.scalar_type() == destination.scalar_type() && source.dim() >= 1
        && destination.dim() == source.dim() && source.sizes().slice(1) == destination.sizes().slice(1),
        "compact publication needs contiguous matching same-device rows");
    TORCH_CHECK(at >= 0 && source.size(0) <= destination.size(0)
        && at <= destination.size(0) - source.size(0), "compact publication exceeds its owned rows");
    TORCH_CHECK(generations.is_cuda() && generations.device() == source.device()
        && generations.is_contiguous() && generations.scalar_type() == at::kLong && generations.numel() == 2
        && slot >= 0 && owner >= 0 && status.is_cuda() && status.device() == source.device()
        && status.is_contiguous() && status.scalar_type() == at::kInt && status.numel() == 2,
        "compact publication needs its live generation and status owners");
    TORCH_CHECK(destination.stride(0) <= std::numeric_limits<int64_t>::max() / destination.element_size(),
        "compact publication byte stride exceeds signed64");
    const int64_t stride = destination.stride(0) * destination.element_size();
    TORCH_CHECK(!at || stride <= std::numeric_limits<int64_t>::max() / at,
        "compact publication byte offset exceeds signed64");
    TORCH_CHECK(source.numel() <= std::numeric_limits<int64_t>::max() / source.element_size(),
        "compact publication byte count exceeds signed64");
    at::assert_no_partial_overlap(source, destination);
    at::assert_no_overlap(destination, generations);
    at::assert_no_overlap(destination, status);
    c10::cuda::CUDAGuard guard(source.device());
    if (source.numel()) compact_publish_cuda(source, destination, at * stride, generations, slot, owner, status, layer);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) { module.def("publish", &publish); }
