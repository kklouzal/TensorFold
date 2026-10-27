// Exact existing five-word MTP frame, on the current stream and original owners.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <limits>
#include "../kernels/tensor_contracts.cuh"

using T = at::Tensor;

void greedy_local_cuda(const T&, const T&, const T&, const T&, T&);
void greedy_global_cuda(const T&, T&, T&, int64_t);
void target_local_cuda(const T&, T&, int64_t);
void target_accept_cuda(const T&, const T&, const T&, T&, int64_t, int64_t, bool);

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

void target_local(const T& logits, T packet, int64_t offset, int64_t vocabulary) {
    device(logits);
    check(logits, logits, at::kFloat, "original target logits");
    check(packet, logits, at::kLong, "target maxima packet");
    TORCH_CHECK(logits.dim() == 2 && logits.size(0) > 0 && logits.size(1) > 0
                && logits.size(0) <= std::numeric_limits<int>::max()
                && logits.size(1) <= std::numeric_limits<int>::max(), "nonempty bounded target rows required");
    TORCH_CHECK(vocabulary > 0 && vocabulary <= std::numeric_limits<int>::max()
                && offset >= 0 && logits.size(1) <= vocabulary && offset <= vocabulary - logits.size(1),
                "target offset must describe original contiguous global IDs");
    TORCH_CHECK(packet.dim() == 2 && packet.size(0) == logits.size(0) && packet.size(1) == 2,
                "target packet requires two int64 words per row");
    tensorfold::check_disjoint(packet, {&logits}, "target packet overlaps original logits");
    c10::cuda::CUDAGuard guard(logits.device());
    target_local_cuda(logits, packet, offset);
}

void target_accept(const T& frame, const T& inputs, const T& eos, T feedback,
                   int64_t room, int64_t vocabulary, bool stop_eos) {
    device(frame);
    check(frame, frame, at::kLong, "target rank frame");
    check(inputs, frame, at::kInt, "original pending/draft IDs");
    check(eos, frame, at::kInt, "generation EOS IDs");
    check(feedback, frame, at::kInt, "committed target feedback");
    TORCH_CHECK(frame.dim() == 3 && frame.size(0) >= 1 && frame.size(0) <= 8
                && frame.size(1) > 0 && frame.size(1) <= std::numeric_limits<int>::max() && frame.size(2) == 2,
                "target frame requires bounded rank/row/two-word geometry");
    TORCH_CHECK(inputs.dim() == 1 && inputs.numel() == frame.size(1)
                && eos.dim() == 1 && eos.numel() <= vocabulary
                && feedback.dim() == 1 && feedback.numel() == frame.size(1) + 2
                && room > 0 && vocabulary > 0 && vocabulary <= std::numeric_limits<int>::max(),
                "target feedback requires complete rows, EOS domain and positive output room");
    tensorfold::check_disjoint(feedback, {&frame, &inputs, &eos}, "target feedback overlaps an input");
    c10::cuda::CUDAGuard guard(frame.device());
    target_accept_cuda(frame, inputs, eos, feedback, room, vocabulary, stop_eos);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("local", &local);
    module.def("global_winner", &global);
    module.def("target_local", &target_local);
    module.def("target_accept", &target_accept);
}
