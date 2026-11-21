// SPDX-License-Identifier: MIT
// GGML block decoding for DeepSeek-V4.1 original-GGUF carriers, TensorFold-owned,
// written from llama.cpp ggml-quants.c reference dequantize_row_* (MIT): every
// operation in FP32 with explicit rounding (no contraction), in reference order.
// The same functions as onecat_ds41.cu's dequantize_kernel helpers (that unit
// keeps its own copy until its owner switches it to this header).
#pragma once
#include <cuda_fp16.h>
#include <cstdint>
#include "onecat_ds41_codebooks.cuh"

namespace tensorfold::ds41_ggml {
__device__ __forceinline__ uint32_t byte_u16(const uint8_t* p) {
  return uint32_t(p[0]) | (uint32_t(p[1]) << 8);
}
__device__ __forceinline__ float byte_half(const uint8_t* p) {
  return __half2float(__ushort_as_half(uint16_t(byte_u16(p))));
}
template <int Book>
__device__ __forceinline__ float grid_value(int entry, int j) {
  return float(int(turbomind::gemm::LatticeCodebook<Book>::value(entry * 8 + j)) - 128);
}
static __constant__ int8_t kIQ4NL[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                  1,    13,   25,  38,  53,  69,  89,  113};

// Decodes values [8 * g, 8 * g + 8) of one block starting at b.
template <int Type>
__device__ void decode8(const uint8_t* b, int g, float* y) {
  if constexpr (Type == 8) {  // Q8_0: x * d
    const float d = byte_half(b);
    for (int j = 0; j < 8; ++j) y[j] = __fmul_rn(float(int8_t(b[2 + 8 * g + j])), d);
  } else if constexpr (Type == 20) {  // IQ4_NL: d * kvalues[nibble]
    const float d = byte_half(b);
    for (int j = 0; j < 8; ++j) {
      const int q = g < 2 ? b[2 + 8 * g + j] & 15 : b[2 + 8 * (g - 2) + j] >> 4;
      y[j] = __fmul_rn(d, float(kIQ4NL[q]));
    }
  } else if constexpr (Type == 13) {  // Q5_K: (d * sc) * q - (dmin * m)
    const float d = byte_half(b), dmin = byte_half(b + 2);
    const uint8_t* scales = b + 4;
    const int j64 = g / 8, high = (g % 8) / 4, l0 = (g % 4) * 8;
    const int is = 2 * j64 + high;
    int sc, m;
    if (is < 4) {
      sc = scales[is] & 63;
      m = scales[is + 4] & 63;
    } else {
      sc = (scales[is + 4] & 0xF) | ((scales[is - 4] >> 6) << 4);
      m = (scales[is + 4] >> 4) | ((scales[is] >> 6) << 4);
    }
    const float d1 = __fmul_rn(d, float(sc)), m1 = __fmul_rn(dmin, float(m));
    const uint8_t* ql = b + 48 + 32 * j64;
    const uint8_t* qh = b + 16;
    const int bit = 1 << (2 * j64 + high);
    for (int j = 0; j < 8; ++j) {
      const int l = l0 + j;
      const int q = (high ? ql[l] >> 4 : ql[l] & 15) + (qh[l] & bit ? 16 : 0);
      y[j] = __fsub_rn(__fmul_rn(d1, float(q)), m1);
    }
  } else if constexpr (Type == 16) {  // IQ2_XXS
    const float d = byte_half(b);
    const int ib32 = g / 4, l = g % 4;
    const uint8_t* q = b + 2 + 8 * ib32;
    const uint32_t aux1 = byte_u16(q + 4) | (byte_u16(q + 6) << 16);
    const float db = __fmul_rn(__fmul_rn(d, __fadd_rn(0.5f, float(aux1 >> 28))), 0.25f);
    int signs = (aux1 >> (7 * l)) & 127;
    signs |= (__popc(signs) & 1) << 7;
    for (int j = 0; j < 8; ++j)
      y[j] = __fmul_rn(__fmul_rn(db, grid_value<16>(q[l], j)),
                       signs & (1 << j) ? -1.f : 1.f);
  } else if constexpr (Type == 17) {  // IQ2_XS
    const float d = byte_half(b);
    const int ib32 = g / 4, l = g % 4;
    const int scale = (b[66 + ib32] >> (4 * (l / 2))) & 15;
    const float db = __fmul_rn(__fmul_rn(d, __fadd_rn(0.5f, float(scale))), 0.25f);
    const uint32_t code = byte_u16(b + 2 + 2 * (4 * ib32 + l));
    int signs = code >> 9;
    signs |= (__popc(signs) & 1) << 7;
    for (int j = 0; j < 8; ++j)
      y[j] = __fmul_rn(__fmul_rn(db, grid_value<17>(code & 511, j)),
                       signs & (1 << j) ? -1.f : 1.f);
  } else {  // IQ1_M: dl * (grid + delta)
    static_assert(Type == 29);
    const uint8_t* sc = b + 48;
    const uint32_t s0 = byte_u16(sc), s1 = byte_u16(sc + 2), s2 = byte_u16(sc + 4),
                   s3 = byte_u16(sc + 6);
    const uint16_t bits = uint16_t((s0 >> 12) | ((s1 >> 8) & 0x00f0) |
                                   ((s2 >> 4) & 0x0f00) | (s3 & 0xf000));
    const float d = __half2float(__ushort_as_half(bits));
    const int ib = g / 4, l = g % 4;
    const uint32_t word = byte_u16(sc + 2 * (ib / 2));
    const int shift = 6 * (ib % 2) + (l < 2 ? 0 : 3);
    const float dl = __fmul_rn(d, float(2 * int((word >> shift) & 7) + 1));
    const int qh = b[32 + 2 * ib + l / 2];
    const int index = b[4 * ib + l] | ((l % 2 ? qh << 4 : qh << 8) & 0x700);
    const float delta = qh & (l % 2 ? 0x80 : 0x08) ? -0.125f : 0.125f;
    for (int j = 0; j < 8; ++j)
      y[j] = __fmul_rn(dl, __fadd_rn(grid_value<29>(index, j), delta));
  }
}
}  // namespace tensorfold::ds41_ggml
