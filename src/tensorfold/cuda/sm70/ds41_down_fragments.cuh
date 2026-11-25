// Byte addressing for the DeepSeek-V4.1 SM70 ETP4 down fragments.
#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define TF_DS41_FRAGMENT_ADDRESS __host__ __device__ __forceinline__
#else
#define TF_DS41_FRAGMENT_ADDRESS constexpr
#endif

namespace tensorfold::ds41_down_fragments {

constexpr int groups_per_block = 8;
constexpr int blocks_per_carrier = 3;
constexpr int groups_per_fragment = 18;

struct Iq1GroupSource {
  const uint8_t* low_codes;    // Four low index bytes, four-byte aligned.
  const uint8_t* high_codes;   // Two bytes holding the high indices and polarity.
  const uint8_t* scale_block;  // Eight original scale bytes, four-byte aligned.
  int source_subgroup;
};

struct Iq2GroupSource {
  const uint8_t* codes;        // Eight original group bytes, two-byte aligned.
  const uint8_t* block_scale;  // Original halfword, two-byte aligned.
  const uint8_t* local_scale;  // IQ2_XS group-scale byte. IQ2_XXS embeds it in codes.
};

template <int Type>
struct Layout {
  static_assert(Type == 16 || Type == 17 || Type == 29);

  static constexpr int original_block_bytes = Type == 16 ? 66 : Type == 17 ? 74 : 56;
  static constexpr int original_row_bytes = blocks_per_carrier * original_block_bytes;
  static constexpr int compact_row_bytes = Type == 16 ? 150 : Type == 17 ? 168 : 132;
  static constexpr int padded_row_bytes = (compact_row_bytes + 15) / 16 * 16;

  // Before entering the kernel, validate the stride, the original rank cut
  // (0/2/4/6 groups), and the byte span of the complete row bank. The down
  // kernel supplies only ordinal 0..17, including its valid clamp for masked lanes.
  TF_DS41_FRAGMENT_ADDRESS static auto group(const uint8_t* row, int ordinal, int left_groups) {
    const int source_group = left_groups + ordinal;
    const int block = source_group / groups_per_block;
    if constexpr (Type == 29) {
      constexpr int scales = blocks_per_carrier * 8;
      constexpr int low_codes = groups_per_fragment * 4;
      return Iq1GroupSource{row + scales + 4 * ordinal, row + scales + low_codes + 2 * ordinal,
                            row + 8 * block, source_group % groups_per_block};
    } else {
      constexpr int codes = groups_per_fragment * 8;
      constexpr int scales = blocks_per_carrier * 2;
      // The scale suffix bounds EM3's code-word lookahead at the end of the
      // bank. Aligning down on odd compact IQ2_XXS rows can also read two
      // bytes from the previous row. The first row is aligned.
      const uint8_t* local = nullptr;
      if constexpr (Type == 17) local = row + codes + scales + ordinal;
      return Iq2GroupSource{row + 8 * ordinal, row + codes + 2 * block, local};
    }
  }

  static constexpr bool accepts_stride(int stride) {
    return stride == compact_row_bytes || stride == padded_row_bytes;
  }
};

}  // namespace tensorfold::ds41_down_fragments

#undef TF_DS41_FRAGMENT_ADDRESS
