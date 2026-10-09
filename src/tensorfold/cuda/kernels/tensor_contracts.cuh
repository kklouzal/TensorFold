#pragma once

#include <ATen/ATen.h>
#include <cstdint>
#include <initializer_list>
#include <limits>

namespace tensorfold {

// Writes are contiguous; reads are contiguous, strided scalars, or contiguous
// rows. Compare the accessed bytes, not storage identity or a row-gap envelope.
inline bool overlaps(const at::Tensor& write, const at::Tensor& read, int64_t elements = -1,
                     int64_t read_elements = -1, int64_t read_columns = -1) {
    if (write.numel() == 0 || read.numel() == 0) return false;
    TORCH_CHECK(write.is_contiguous(), "overlap checks require contiguous writes");
    if (elements < 0) elements = write.numel();
    TORCH_CHECK(elements >= 0 && elements <= write.numel(), "invalid written prefix");
    if (elements == 0) return false;
    constexpr uintptr_t maximum = std::numeric_limits<uintptr_t>::max();
    const auto wb = reinterpret_cast<uintptr_t>(write.data_ptr());
    const auto rb = reinterpret_cast<uintptr_t>(read.data_ptr());
    TORCH_CHECK(static_cast<uint64_t>(elements) <= (maximum - wb) / write.element_size(),
                "written byte range overflows pointer addressing");
    const uintptr_t we = wb + static_cast<uintptr_t>(elements) * write.element_size();
    uintptr_t rows = 1, width = read.numel(), stride = 0;
    if (read_elements >= 0) {
        TORCH_CHECK(read.is_contiguous() && read_elements <= read.numel() && read_columns < 0, "invalid read prefix");
        if (read_elements == 0) return false;
        width = read_elements;
    } else if (!read.is_contiguous() || read_columns >= 0) {
        TORCH_CHECK((read.dim() == 1 || (read.dim() == 2 && read.stride(1) == 1)) && read.stride(0) >= 0,
                    "overlap checks require contiguous rows or nonnegative-stride scalars");
        rows = read.size(0);
        width = read.dim() == 1 ? 1 : read.size(1);
        if (read_columns >= 0) {
            TORCH_CHECK(read.dim() == 2 && read_columns <= read.size(1), "invalid read columns");
            if (read_columns == 0) return false;
            width = read_columns;
        }
        TORCH_CHECK(static_cast<uint64_t>(read.stride(0)) <= maximum / read.element_size(),
                    "read byte stride overflows pointer addressing");
        stride = static_cast<uintptr_t>(read.stride(0)) * read.element_size();
    }
    TORCH_CHECK(width <= (maximum - rb) / read.element_size(), "read byte width overflows pointer addressing");
    width *= read.element_size();
    TORCH_CHECK(stride == 0 || rows - 1 <= (maximum - rb - width) / stride,
                "read byte range overflows pointer addressing");
    if (rb >= we || rb + (rows - 1) * stride + width <= wb) return false;
    const uintptr_t row = stride > 0 && wb >= rb + width ? (wb - rb - width) / stride + 1 : 0;
    return row < rows && rb + row * stride < we;
}

inline void check_disjoint(const at::Tensor& write, std::initializer_list<const at::Tensor*> reads,
                           const char* message, int64_t elements = -1) {
    for (const auto* read : reads)
        if (read->defined()) TORCH_CHECK(!overlaps(write, *read, elements), message);
}

} // namespace tensorfold
