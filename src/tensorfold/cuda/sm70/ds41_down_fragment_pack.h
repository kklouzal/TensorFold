// CPU packing of raw GGUF down rows into the SM70 ETP4 fragment layout.
#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>

#include "ds41_down_fragments.cuh"

namespace tensorfold::ds41_down_fragments {

constexpr int source_blocks = 9;

namespace detail {

template <int Type, int StartGroup>
void pack_rows_at(const uint8_t* source, uint8_t* dest, std::size_t rows) noexcept {
  using Format = Layout<Type>;
  constexpr int block_bytes = Format::original_block_bytes;
  constexpr int source_row_bytes = source_blocks * block_bytes;
  constexpr int first_block = StartGroup / groups_per_block;
  constexpr int left = StartGroup % groups_per_block;
  static_assert(blocks_per_carrier == 3);
  static_assert((left + groups_per_fragment - 1) / groups_per_block == blocks_per_carrier - 1);
  static_assert(first_block + blocks_per_carrier <= source_blocks);

  const auto copy_block = []<int Block>(const uint8_t* raw, uint8_t* packed) {
    constexpr int first = Block == 0 ? left : 0;
    constexpr int ordinal = Block == 0 ? 0 : Block * groups_per_block - left;
    constexpr int end = (Block + 1) * groups_per_block - left;
    constexpr int count = (end < groups_per_fragment ? end : groups_per_fragment) - ordinal;
    static_assert(count > 0 && first + count <= groups_per_block);
    const uint8_t* input = raw + Block * block_bytes;
    if constexpr (Type == 29) {
      constexpr int low_bytes = 4, high_bytes = 2, scale_bytes = 8;
      constexpr int high_source = groups_per_block * low_bytes;
      constexpr int scale_source = high_source + groups_per_block * high_bytes;
      constexpr int low_dest = blocks_per_carrier * scale_bytes;
      constexpr int high_dest = low_dest + groups_per_fragment * low_bytes;
      static_assert(scale_source + scale_bytes == block_bytes);
      static_assert(high_dest + groups_per_fragment * high_bytes == Format::compact_row_bytes);
      std::memcpy(packed + Block * scale_bytes, input + scale_source, scale_bytes);
      std::memcpy(packed + low_dest + ordinal * low_bytes, input + first * low_bytes, count * low_bytes);
      std::memcpy(packed + high_dest + ordinal * high_bytes, input + high_source + first * high_bytes,
                  count * high_bytes);
    } else {
      constexpr int code_bytes = 8, scale_bytes = 2;
      constexpr int local_source = scale_bytes + groups_per_block * code_bytes;
      constexpr int scale_dest = groups_per_fragment * code_bytes;
      constexpr int local_dest = scale_dest + blocks_per_carrier * scale_bytes;
      static_assert(local_source + (Type == 17 ? groups_per_block : 0) == block_bytes);
      static_assert(local_dest + (Type == 17 ? groups_per_fragment : 0) == Format::compact_row_bytes);
      std::memcpy(packed + ordinal * code_bytes, input + scale_bytes + first * code_bytes, count * code_bytes);
      std::memcpy(packed + scale_dest + Block * scale_bytes, input, scale_bytes);
      if constexpr (Type == 17) {
        std::memcpy(packed + local_dest + ordinal, input + local_source + first, count);
      }
    }
  };

  for (std::size_t row = 0; row < rows; ++row) {
    const uint8_t* raw = source + row * source_row_bytes + first_block * block_bytes;
    uint8_t* packed = dest + row * Format::compact_row_bytes;
    copy_block.template operator()<0>(raw, packed);
    copy_block.template operator()<1>(raw, packed);
    copy_block.template operator()<2>(raw, packed);
  }
}

}  // namespace detail

// The caller validates disjoint contiguous buffers covering `rows` nine-block
// GGUF rows and `rows` tight compact rows, and a legal ETP4 rank cut. Resolve
// the cut once so the copies for each of the three intersecting blocks are fixed.
template <int Type>
void pack_rows(const uint8_t* source, uint8_t* dest, std::size_t rows, int start_group) noexcept {
  switch (start_group) {
    case 0: return detail::pack_rows_at<Type, 0>(source, dest, rows);
    case 18: return detail::pack_rows_at<Type, 18>(source, dest, rows);
    case 36: return detail::pack_rows_at<Type, 36>(source, dest, rows);
    case 54: return detail::pack_rows_at<Type, 54>(source, dest, rows);
  }
  std::abort();
}

}  // namespace tensorfold::ds41_down_fragments
