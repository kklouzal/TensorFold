// Exact existing five-word MTP frame, on the current stream and original owners.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <limits>
#include "../kernels/tensor_contracts.cuh"

using T = at::Tensor;

void greedy_local_cuda(const T&, const T&, const T&, const T&, T&);
void greedy_global_cuda(const T&, T&, T&, int64_t);

namespace {
void check(const T& value, const T& owner, at::ScalarType dtype, const char* name) {
    TORCH_CHECK(value.is_cuda() && value.device() == owner.device() && value.scalar_type() == dtype
                && value.is_contiguous(), name, " requires contiguous same-device declared storage");
}
void device(const T& value) {
    TORCH_CHECK(value.is_cuda(), "greedy packet requires CUDA storage");
    const auto* prop = at::cuda::getDeviceProperties(value.get_device());
    TORCH_CHECK(prop->major == 7 && prop->minor == 0, "greedy packet requires SM70");
}
}  // namespace

void local(const T& logits, const T& ids, const T& norm, const T& status, T packet) {
    device(logits);
    check(logits, logits, at::kFloat, "original logits");
    check(ids, logits, at::kLong, "immutable IDs");
    check(norm, logits, at::kFloat, "original LSE");
    check(status, logits, at::kInt, "codec status");
    check(packet, logits, at::kLong, "rank packet");
    TORCH_CHECK(logits.dim() == 2 && logits.size(0) == 1 && logits.size(1) > 0
                && logits.size(1) <= std::numeric_limits<int>::max(), "one nonempty original shard row required");
    TORCH_CHECK(ids.dim() == 1 && ids.numel() == logits.size(1) && norm.numel() == 1
                && status.numel() == 2 && packet.dim() == 1 && packet.numel() == 5, "greedy packet geometry");
    tensorfold::check_disjoint(packet, {&logits, &ids, &norm, &status}, "greedy packet overlaps an input");
    c10::cuda::CUDAGuard guard(logits.device());
    greedy_local_cuda(logits, ids, norm, status, packet);
}

void global(const T& frame, T token, T input, int64_t vocabulary) {
    device(frame);
    check(frame, frame, at::kLong, "rank frame");
    check(token, frame, at::kLong, "historical token");
    check(input, frame, at::kInt, "safe next input");
    TORCH_CHECK(frame.dim() == 2 && frame.size(0) >= 1 && frame.size(0) <= 8 && frame.size(1) == 5
                && token.numel() == 1 && input.numel() == 1 && vocabulary > 0, "global greedy packet geometry");
    tensorfold::check_disjoint(token, {&frame, &input}, "historical token overlaps an input/output");
    tensorfold::check_disjoint(input, {&frame}, "safe next input overlaps the rank frame");
    c10::cuda::CUDAGuard guard(frame.device());
    greedy_global_cuda(frame, token, input, vocabulary);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("local", &local);
    module.def("global_winner", &global);
}
