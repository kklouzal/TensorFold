// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
#include "lattice_codebooks.h"
#include "types.h"
#include "../core/array.h"
namespace turbomind::gemm {

// TensorFold (Exact = true, IQ2_XXS / IQ2_XS / IQ1_M only): the stats word carries the source's FP16 super-block
// scale d verbatim and the group's integer sub-scale s (bits 24..27; IQ1_M: packet bits 11..13), so the decoded
// weight is fp16(d * ((2s+1)/8 * signed grid)) [IQ1_M: fp16(d * ((2s+1) * (grid + delta)))]: the inner product is
// exact in FP16 (<= 11 significant bits), the one rounding is the HMUL by d, i.e. ggml's FP32 product rounded once
// to FP16. Exact = false keeps the friend's FP16 scale (d * (2s+1)/8 rounded at transcode) times grid, rounded
// again.
template<int Type, int GroupSize, int Replicas = 1, bool Exact = false>
struct Transform_HMMA_SM70_Lattice {
  static_assert(!Exact || Type == 16 || Type == 17 || Type == 29);
  using Codebook = LatticeCodebook<Type>;
  static constexpr int kReplicas = Replicas;
  static_assert(kReplicas == 1 || (Codebook::kWidth == 4 && kReplicas == 8));
  static constexpr int kCodebookBytes = Codebook::kBytes * kReplicas;
  static constexpr auto kQuantType = static_cast<QuantType>(8 + (
      Type == 16 ? 0 : Type == 17 ? 1 : Type == 18 ? 2 : Type == 19 ? 3 :
      Type == 21 ? 4 : Type == 22 ? 5 : 6));
  template<bool PublishAtOperandBarrier = false>
  __device__ static void initialize(uint8_t* shared) {
    if constexpr (PublishAtOperandBarrier && Exact && Type == 29 && kReplicas == 1) {
      // The source table and WithDecoderCodebook storage are both aligned
      // to 16 bytes. Each thread owns disjoint complete vectors. The caller
      // publishes these shared writes at its first operand barrier.
      constexpr int bytes = 16;
      static_assert(kCodebookBytes % bytes == 0);
      for (int i = threadIdx.x; i < kCodebookBytes / bytes; i += blockDim.x) {
        const auto src = __cvta_generic_to_global(Codebook::data() + bytes * i);
        const auto dst = static_cast<uint32_t>(__cvta_generic_to_shared(shared + bytes * i));
        asm volatile("{\n\t"
                     ".reg .b32 a, b, c, d;\n\t"
                     "ld.global.v4.b32 {a, b, c, d}, [%0];\n\t"
                     "st.shared.v4.b32 [%1], {a, b, c, d};\n\t"
                     "}" :: "l"(src), "r"(dst) : "memory");
      }
    } else {
      static_assert(kCodebookBytes % sizeof(uint32_t) == 0);
      auto* words = reinterpret_cast<uint32_t*>(shared);
      for (int i = threadIdx.x; i < kCodebookBytes / sizeof(uint32_t); i += blockDim.x)
        words[i] = Codebook::word(i / kReplicas);
      __syncthreads();
    }
  }
  template<class F, int Nf, int Mf, int K, class D, int Nd, int Md,
           class S, int Ns, int Ms, int Ks>
  __device__ static void apply(Array<F,Nf> (&frag)[K][Mf], int k,
                              Array<D,Nd> (&data)[K][Md],
                              Array<S,Ns> (&stat)[Ks][Ms], int div,
                              const uint8_t* grid) {
    static_assert(std::is_same_v<D,uint2_t> && std::is_same_v<F,half>);
    static_assert(Nd == 8 && Nf == 8 && Mf == Md);
    static_assert(sizeof(S) == (Codebook::kWidth == 4 ? 8 :
                  (Type == 19 || Type == 29 ? 2 : 4)));
    auto& dst = reinterpret_cast<Array<F,Nd> (&)[Md]>(frag[k]);
    auto& stats = reinterpret_cast<Array<S,1> (&)[Ns*Ms]>(stat[k/div]);
    const int base = (k * Nd) % GroupSize;
    PRAGMA_UNROLL
    for (int m = 0; m < Md; ++m) {
      const uint64_t metadata = stats[m][0];
      const uint16_t packet = (const uint16_t&)data[k][m];
      uint32_t scale = metadata & 65535U;
      scale |= scale << 16;
      half2 factor;  // Exact: the group's exact sub-scale factor
      if constexpr (Exact) {
        const float f = Type == 29 ? float(2 * ((packet >> 11) & 7) + 1)
                                   : float(2 * ((metadata >> 24) & 15) + 1) * 0.125f;
        factor = __float2half2_rn(f);
      }
      // A lattice row is exactly four/eight consecutive biased bytes.
      // Fetch the row as aligned words once, then unpack FP16 pairs in
      // registers instead of repeatedly reading adjacent shared halfwords.
      uint64_t table_values;
      if constexpr (Codebook::kWidth == 8) {
        const int index = Type == 19 || Type == 29 ? packet & 2047 :
            (packet & 255) | (((metadata >> (16+2*(base/8))) & 3) << 8);
        table_values = *reinterpret_cast<const uint64_t*>(grid+index*8);
      } else {
        const int first = (packet & 255) | (((metadata >> (48+base/4)) & 1) << 8);
        const int second = (packet >> 8) | (((metadata >> (49+base/4)) & 1) << 8);
        const int replica = threadIdx.x % kReplicas;
        const uint32_t low = *reinterpret_cast<const uint32_t*>(
            grid+(first*kReplicas+replica)*4);
        const uint32_t high = *reinterpret_cast<const uint32_t*>(
            grid+(second*kReplicas+replica)*4);
        table_values = low | (static_cast<uint64_t>(high) << 32);
      }
      Array<F,8> decoded;
      PRAGMA_UNROLL
      for (int i = 0; i < 8; i += 2) {
        constexpr uint32_t magic = 0x64006400U;
        constexpr uint32_t bias = 0x64806480U;
        const uint32_t bytes = static_cast<uint32_t>(table_values >> (i*8));
        const uint32_t pair = __byte_perm(bytes,magic,0x7170);
        half2 values = __hsub2((const half2&)pair,(const half2&)bias);
        if constexpr (Type == 19 || Type == 29) {
          const half delta = __float2half((packet & 32768) ? -0.125f : 0.125f);
          values = __hadd2(values,__halves2half2(delta,delta));
        } else {
          const uint32_t signs = Codebook::kWidth == 8 ? packet >> 8 :
                                 metadata >> (16+base);
          uint32_t sign_mask = ((signs >> i) & 1) << 15;
          sign_mask |= ((signs >> (i+1)) & 1) << 31;
          (uint32_t&)values ^= sign_mask;
        }
        if constexpr (Exact) values = __hmul2(values,factor);  // exact
        (half2&)decoded[i] = __hmul2(values,(const half2&)scale);
      }
      dst[m] = decoded;
    }
  }
};
} // namespace turbomind::gemm
