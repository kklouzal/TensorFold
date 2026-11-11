#pragma once
#include <ATen/ATen.h>
#include <c10/core/InferenceMode.h>
#include <cstdint>
#include <limits>

// Shared actual operand and exact strided-byte contracts for two owned Dense components.
namespace tensorfold::prefill {
inline void operand(const at::Tensor& t, at::ScalarType dtype, const c10::Device& device) {
    TORCH_CHECK(t.is_cuda() && t.device() == device && t.scalar_type() == dtype
        && !t.requires_grad() && !t.is_neg() && !t.is_conj(), "prefill Dense operand domain");
}
inline void writable(const at::Tensor& t) {
    TORCH_CHECK(!t.is_inference() || c10::InferenceMode::is_enabled(),
        "prefill Dense inference output requires InferenceMode");
}
struct ByteRows {
    uintptr_t begin, width, stride, rows;
};
inline ByteRows byte_rows(const at::Tensor& tensor) {
    if (!tensor.numel()) return {0, 0, 0, 0};
    constexpr auto maximum = std::numeric_limits<uintptr_t>::max();
    const auto begin = reinterpret_cast<uintptr_t>(tensor.data_ptr());
    const auto item = uintptr_t(tensor.element_size());
    uintptr_t rows = 1, width = tensor.numel(), stride = 0;
    if (!tensor.is_contiguous()) {
        TORCH_CHECK(tensor.dim() == 2 && tensor.stride(1) == 1
            && tensor.stride(0) >= tensor.size(1), "prefill Dense spans require disjoint contiguous rows");
        rows = tensor.size(0);
        width = tensor.size(1);
        TORCH_CHECK(uint64_t(tensor.stride(0)) <= maximum / item, "prefill Dense byte stride overflow");
        stride = uintptr_t(tensor.stride(0)) * item;
    }
    TORCH_CHECK(width <= (maximum - begin) / item, "prefill Dense byte width overflow");
    width *= item;
    TORCH_CHECK(!stride || rows - 1 <= (maximum - begin - width) / stride,
        "prefill Dense byte span overflow");
    return {begin, width, stride, rows};
}
inline bool overlaps(const at::Tensor& left, const at::Tensor& right) {
    if (left.device() != right.device()) return false;
    const auto a = byte_rows(left), b = byte_rows(right);
    if (!a.rows || !b.rows || a.begin >= b.begin + (b.rows - 1) * b.stride + b.width
            || b.begin >= a.begin + (a.rows - 1) * a.stride + a.width) return false;
    uintptr_t i = 0, j = 0;
    while (i < a.rows && j < b.rows) {
        const auto ab = a.begin + i * a.stride, bb = b.begin + j * b.stride;
        if (ab < bb + b.width && bb < ab + a.width) return true;
        if (ab + a.width <= bb) ++i;
        else ++j;
    }
    return false;
}

} // namespace tensorfold::prefill
