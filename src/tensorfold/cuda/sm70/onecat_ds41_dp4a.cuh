// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from 1CatAI/1Cat-vLLM 4d57473db67e csrc/sm70_turbomind/ops/
// gguf_q8_1.cuh and gguf_dp4a.cuh. Integer-dot formulas and the Q8_1 layout
// follow ggml-org/llama.cpp ggml-cuda/vecdotq.cuh and quantize.cu (MIT).
// Changes: only the DeepSeek-V4.1 IQ2_XXS(16)/IQ2_XS(17)/IQ1_M(29) readers
// are kept (LatticeDot<22>::sign_mask became the free sign_mask); namespace
// renamed. Every arithmetic expression is unchanged.
#pragma once
#include <cuda_fp16.h>
#include <cstdint>
#include "onecat_ds41_codebooks.cuh"

namespace tensorfold::onecat_ds41 {
struct Q8_1 {
  half2 ds;
  int8_t qs[32];
};
static_assert(sizeof(Q8_1) == 36);

// One full warp quantizes one 32-value group (all 32 lanes participate).
__device__ __forceinline__ void quantize_q8_1_warp(Q8_1* out, float value) {
  const int lane = threadIdx.x % 32;
  float maximum = fabsf(value), sum = value;
#pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
    sum += __shfl_xor_sync(0xffffffff, sum, offset);
  }
  const float d = maximum / 127.f;
  out->qs[lane] = maximum == 0.f ? 0 : int8_t(roundf(value / d));
  if (!lane) out->ds = __floats2half2_rn(d, sum);
}

// Requires 2-byte alignment (original GGUF blocks are even-sized).
__device__ __forceinline__ uint32_t load_u32_2(const uint8_t* p) {
  return uint32_t(*reinterpret_cast<const uint16_t*>(p)) |
         (uint32_t(*reinterpret_cast<const uint16_t*>(p + 2)) << 16);
}

__device__ __forceinline__ uint32_t sign_mask(int nibble) {
  // Spread the four bits into byte sign positions, then sign-extend each
  // byte with PRMT. __byte_perm clears selector sign bits; use PTX here.
  const uint32_t bits = uint32_t(nibble) * 0x10204080U;
  uint32_t mask;
  asm("prmt.b32 %0, %1, 0, 0xba98;" : "=r"(mask) : "r"(bits));
  return mask;
}

// V4.1's IQ1_M/IQ2 rows consume one Q8_1 group (32 activations) per call.
// IQ1's eighth-step delta is folded into signed integer codebook values, so
// both local scale groups remain exact integer dots before the shared scale.
template <int Type>
struct V41LatticeDot {
  static_assert(Type == 16 || Type == 17 || Type == 29);
  using Book = turbomind::gemm::LatticeCodebook<Type>;
  static constexpr int kBookWords = Book::kBytes / 4;
  static constexpr int kBlockBytes = Type == 16 ? 66 : Type == 17 ? 74 : 56;

  __device__ static void initialize(uint32_t* book, uint32_t* masks) {
    for (int i = threadIdx.x; i < kBookWords; i += blockDim.x) {
      uint32_t word = Book::word(i) ^ 0x80808080U;
      if constexpr (Type == 29) {
        // IQ1's eighth-step grid is shared by every dot in this CTA.
        word = __vadd4(word, word);
        word = __vadd4(word, word);
        word = __vadd4(word, word);
      }
      book[i] = word;
    }
    if (threadIdx.x < 16) masks[threadIdx.x] = sign_mask(threadIdx.x);
    __syncthreads();
  }

  __device__ static float dot(const uint8_t* row, int group, const Q8_1& x,
                              const uint32_t* book, const uint32_t* masks) {
    const uint8_t* b = row + (group / 8) * kBlockBytes;
    const int sub = group % 8;
    const int* activation = reinterpret_cast<const int*>(x.qs);
    int sums[2] = {0, 0};
    const uint32_t meta = Type == 16 ? load_u32_2(b + 6 + sub * 8) : 0;
#pragma unroll
    for (int octet = 0; octet < 4; ++octet) {
      int index, signs = 0, delta = 0;
      if constexpr (Type == 16) {
        index = b[2 + sub * 8 + octet];
        signs = (meta >> (7 * octet)) & 127;
      } else if constexpr (Type == 17) {
        const uint16_t code =
            *reinterpret_cast<const uint16_t*>(b + 2 + (sub * 4 + octet) * 2);
        index = code & 511;
        signs = code >> 9;
      } else {
        const int t = sub * 4 + octet;
        const int high = (b[32 + t / 2] >> (4 * (t % 2))) & 15;
        index = b[t] | ((high & 7) << 8);
        delta = high >> 3;
      }
      uint32_t w0 = book[index * 2], w1 = book[index * 2 + 1];
      if constexpr (Type == 29) {
        const uint32_t bias = delta ? 0xffffffffU : 0x01010101U;
        w0 = __vadd4(w0, bias);
        w1 = __vadd4(w1, bias);
      } else {
        signs |= (__popc(signs) & 1) << 7;
        const auto s0 = masks[signs & 15], s1 = masks[signs >> 4];
        w0 = __vsub4(w0 ^ s0, s0);
        w1 = __vsub4(w1 ^ s1, s1);
      }
      sums[octet / 2] = __dp4a(int(w0), activation[2 * octet], sums[octet / 2]);
      sums[octet / 2] =
          __dp4a(int(w1), activation[2 * octet + 1], sums[octet / 2]);
    }
    float d;
    int mult0, mult1;
    if constexpr (Type == 29) {
      const auto* scales = reinterpret_cast<const uint16_t*>(b + 48);
      const uint16_t bits = (scales[0] >> 12) | ((scales[1] >> 8) & 0xf0) |
                            ((scales[2] >> 4) & 0xf00) | (scales[3] & 0xf000);
      d = __half2float(__ushort_as_half(bits));
      const int first = sub * 2, second = first + 1;
      mult0 = 2 * ((scales[first / 4] >> (3 * (first % 4))) & 7) + 1;
      mult1 = 2 * ((scales[second / 4] >> (3 * (second % 4))) & 7) + 1;
    } else {
      d = __half2float(*reinterpret_cast<const half*>(b));
      if constexpr (Type == 16)
        mult0 = mult1 = 1 + 2 * (meta >> 28);
      else {
        mult0 = 1 + 2 * (b[66 + sub] & 15);
        mult1 = 1 + 2 * (b[66 + sub] >> 4);
      }
    }
    const float scale = __fmul_rn(d, __low2float(x.ds));
    return __fmul_rn(__fmul_rn(scale, float(sums[0] * mult0 + sums[1] * mult1)),
                     0.125f);
  }
};

// The routed rows can share one expert. Decode the original integer words
// once, then apply each token's Q8_1 scale and exactly the original dot order.
template <int Type>
struct V41IntegerGroup {
  uint32_t words[8];
  float d;
  int mult0, mult1;

  __device__ V41IntegerGroup(const uint8_t* row, int group,
                             const uint32_t* book, const uint32_t* masks) {
    constexpr int bytes = V41LatticeDot<Type>::kBlockBytes;
    const uint8_t* b = row + (group / 8) * bytes;
    const int sub = group % 8;
    const uint32_t meta = Type == 16 ? load_u32_2(b + 6 + sub * 8) : 0;
#pragma unroll
    for (int octet = 0; octet < 4; ++octet) {
      int index, signs = 0, delta = 0;
      if constexpr (Type == 16) {
        index = b[2 + sub * 8 + octet];
        signs = (meta >> (7 * octet)) & 127;
      } else if constexpr (Type == 17) {
        const auto code =
            *reinterpret_cast<const uint16_t*>(b + 2 + (sub * 4 + octet) * 2);
        index = code & 511;
        signs = code >> 9;
      } else {
        const int t = sub * 4 + octet;
        const int high = (b[32 + t / 2] >> (4 * (t % 2))) & 15;
        index = b[t] | ((high & 7) << 8);
        delta = high >> 3;
      }
      uint32_t first = book[index * 2], second = book[index * 2 + 1];
      if constexpr (Type == 29) {
        const uint32_t bias = delta ? 0xffffffffU : 0x01010101U;
        first = __vadd4(first, bias);
        second = __vadd4(second, bias);
      } else {
        signs |= (__popc(signs) & 1) << 7;
        const auto low = masks[signs & 15], high = masks[signs >> 4];
        first = __vsub4(first ^ low, low);
        second = __vsub4(second ^ high, high);
      }
      words[2 * octet] = first;
      words[2 * octet + 1] = second;
    }
    if constexpr (Type == 29) {
      const auto* scales = reinterpret_cast<const uint16_t*>(b + 48);
      const uint16_t bits = (scales[0] >> 12) | ((scales[1] >> 8) & 0xf0) |
                            ((scales[2] >> 4) & 0xf00) | (scales[3] & 0xf000);
      d = __half2float(__ushort_as_half(bits));
      const int first = sub * 2, second = first + 1;
      mult0 = 2 * ((scales[first / 4] >> (3 * (first % 4))) & 7) + 1;
      mult1 = 2 * ((scales[second / 4] >> (3 * (second % 4))) & 7) + 1;
    } else {
      d = __half2float(*reinterpret_cast<const half*>(b));
      if constexpr (Type == 16)
        mult0 = mult1 = 1 + 2 * (meta >> 28);
      else {
        mult0 = 1 + 2 * (b[66 + sub] & 15);
        mult1 = 1 + 2 * (b[66 + sub] >> 4);
      }
    }
  }

  __device__ float apply(const Q8_1& x) const {
    const auto* a = reinterpret_cast<const int*>(x.qs);
    int sums[2] = {0, 0};
#pragma unroll
    for (int i = 0; i < 8; ++i)
      sums[i / 4] = __dp4a(int(words[i]), a[i], sums[i / 4]);
    const float scale = __fmul_rn(d, __low2float(x.ds));
    return __fmul_rn(__fmul_rn(scale, float(sums[0] * mult0 + sums[1] * mult1)),
                     0.125f);
  }
};
}  // namespace tensorfold::onecat_ds41
