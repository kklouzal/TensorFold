#include <torch/extension.h>
#include <ATen/MemoryOverlap.h>
#include <c10/cuda/CUDAGuard.h>
#include <limits>

using T = at::Tensor;
void compact_publish_cuda(const T&, T&, int64_t, const T&, int64_t, int64_t, T&, int);
void compact_round_stage_cuda(const T&, const T&, T&, T&, int);
void compact_round_publish_cuda(const T&, const T&, int);

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

// Round-table lanes: [lanes, 20] int64 words owned by the canonical round owner.
// Every pointer word references a retained State/staging tensor; capacity was
// validated on the host before the table was published.
void check_round(const T& tables, const T& owner) {
    TORCH_CHECK(tables.is_cuda() && tables.device() == owner.device() && tables.is_contiguous()
        && tables.scalar_type() == at::kLong && tables.dim() == 2 && tables.size(1) == 20,
        "compact round lane table geometry");
}

void round_stage(const T& ik, const T& tables, T raw, T coords, int layer) {
    check_round(tables, ik);
    TORCH_CHECK(ik.is_cuda() && ik.is_contiguous() && ik.scalar_type() == at::kHalf && ik.dim() == 2 && ik.size(1) == 128
        && raw.is_cuda() && raw.is_contiguous() && raw.scalar_type() == at::kHalf
        && raw.numel() == tables.size(0) * 16 * 4 * 128 && coords.is_cuda() && coords.is_contiguous()
        && coords.scalar_type() == at::kLong && coords.numel() == tables.size(0) * 16 * 3
        && raw.device() == ik.device() && coords.device() == ik.device(), "compact round stage geometry");
    at::assert_no_overlap(raw, ik);
    c10::cuda::CUDAGuard guard(ik.device());
    compact_round_stage_cuda(ik, tables, raw, coords, layer);
}

void round_publish(const T& values, const T& tables, int layer) {
    check_round(tables, values);
    TORCH_CHECK(values.is_cuda() && values.is_contiguous() && values.scalar_type() == at::kHalf
        && values.numel() == tables.size(0) * 16 * 128, "compact round publication geometry");
    c10::cuda::CUDAGuard guard(values.device());
    compact_round_publish_cuda(values, tables, layer);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("publish", &publish);
    module.def("round_stage", &round_stage);
    module.def("round_publish", &round_publish);
}
