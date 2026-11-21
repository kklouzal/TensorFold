// SPDX-License-Identifier: Apache-2.0
// ds41kern (agent G, patch 340): TurboMind sm70 lattice grouped-GEMM tile variants for
// DeepSeek-V4.1 prefill routed experts, built OUTSIDE vllm/ against the fork's own
// TurboMind headers (1cat-vllm 4d57473db6, csrc/sm70_turbomind/lmdeploy).
//
// Variant 0 is the exact production instantiation (Config_GgufLattice<T, G, kColMajor, 0>
// ::Type<32, 128, 32, 1, 4, 1, Default, Stream, 2, true, 1, G>, the kernel seqncu-1 profiled).
// The others change ONLY the CTA/warp tiling or the shared-memory A layout
// (Operand_A_BatchPadded: +8 halves per activation row, the fork's own fix for the 4-row bank
// repeat of SmemCopy_MMA_884_A at K32). None changes the per-output K order (CTA_K multiples of
// the k4 MMA step, TG_K = 1, splits = 1), so every variant is expected bit-identical to the
// production kernel; the bench verifies that with torch.equal on real weights.
#pragma once

#include "turbomind/kernels/gemm/arch/config_sm70_s884.h"
#include "turbomind/kernels/gemm/kernel_impl.h"
#include "turbomind/kernels/gemm/types.h"

namespace ds41kern {

using namespace turbomind;
using namespace turbomind::gemm;
using namespace turbomind::gemm::sm70_s884;

template <class Config>
struct Padded;

template <class A, class TA, class U, class B, class TB, class V, Order oc, class Tc,
          Order ro, int ga>
struct Padded<Sm70_s884<A, TA, U, B, TB, V, oc, Tc, ro, ga>> {
  using type = Sm70_s884<Operand_A_BatchPadded<half>, TA, U, B, TB, V, oc, Tc, ro, ga>;
};

// Keep in sync with kVariants in ds41kern_op.cu. TensorFold: 9..13 are the exact-sub-scale decode
// (Transform_HMMA_SM70_Lattice Exact) at the tilings the patch-340 rule uses: 9/10 = 6/7 with FP32 output (gate|up
// into the SwiGLU unrounded), 11/12/13 = 3/6/7 with FP16 output (down).
constexpr int kVariants = 14;

template <int Type, int Group>
Kernel* make_variant(int v);

#define DS41KERN_DEFINE_VARIANTS(TYPE, GROUP)                                            \
  template <>                                                                            \
  Kernel* make_variant<TYPE, GROUP>(int v) {                                             \
    using G = Config_GgufLattice<TYPE, GROUP, kColMajor, 0>;                            \
    using P = typename Padded<G>::type;                                                  \
    using D = cache_policy::Default;                                                     \
    using S = cache_policy::Stream;                                                      \
    using EF = typename Padded<Config_GgufLattice<TYPE, GROUP, kColMajor, 0, true, float>>::type; \
    using EH = typename Padded<Config_GgufLattice<TYPE, GROUP, kColMajor, 0, true, half>>::type;  \
    switch (v) {                                                                         \
      case 0: /* production: 32x128x32, 1x4 warps (warp 32x32), plain A */             \
        return new KernelImpl<typename G::template Type<32, 128, 32, 1, 4, 1, D, S, 2,  \
                                                        true, 1, GROUP>::Kernel>();     \
      case 1: /* padded A, same tiling */                                               \
        return new KernelImpl<typename P::template Type<32, 128, 32, 1, 4, 1, D, S, 2,  \
                                                        true, 1, GROUP>::Kernel>();     \
      case 2: /* padded A, 1x2 warps (warp 32x64: A fragment reused for 2 N atoms) */   \
        return new KernelImpl<typename P::template Type<32, 128, 32, 1, 2, 1, D, S, 2,  \
                                                        true, 1, GROUP>::Kernel>();     \
      case 3: /* padded A, 32x256 CTA, 1x4 warps (warp 32x64) */                         \
        return new KernelImpl<typename P::template Type<32, 256, 32, 1, 4, 1, D, S, 2,  \
                                                        true, 1, GROUP>::Kernel>();     \
      case 4: /* padded A, 64x128, 2x2 warps (warp 32x64) */                             \
        return new KernelImpl<typename P::template Type<64, 128, 32, 2, 2, 1, D, S, 2,  \
                                                        true, 1, GROUP>::Kernel>();     \
      case 5: /* padded A, 64x128, 1x4 warps (warp 64x32) */                             \
        return new KernelImpl<typename P::template Type<64, 128, 32, 1, 4, 1, D, S, 2,  \
                                                        true, 1, GROUP>::Kernel>();     \
      case 6: /* padded A, 16x128 (less M padding at ~16-32 routes/expert) */           \
        return new KernelImpl<typename P::template Type<16, 128, 32, 1, 4, 1, D, S, 2,  \
                                                        true, 1, GROUP>::Kernel>();     \
      case 7: /* padded A, CTA_K 64 (half the barriers per K) */                         \
        return new KernelImpl<typename P::template Type<32, 128, 64, 1, 4, 1, D, S, 2,  \
                                                        true, 1, GROUP>::Kernel>();     \
      case 8: /* plain A, 1x2 warps (isolates the warp-shape effect) */                  \
        return new KernelImpl<typename G::template Type<32, 128, 32, 1, 2, 1, D, S, 2,  \
                                                        true, 1, GROUP>::Kernel>();     \
      case 9:                                                                            \
        return new KernelImpl<typename EF::template Type<16, 128, 32, 1, 4, 1, D, S, 2, \
                                                         true, 1, GROUP>::Kernel>();    \
      case 10:                                                                           \
        return new KernelImpl<typename EF::template Type<32, 128, 64, 1, 4, 1, D, S, 2, \
                                                         true, 1, GROUP>::Kernel>();    \
      case 11:                                                                           \
        return new KernelImpl<typename EH::template Type<32, 256, 32, 1, 4, 1, D, S, 2, \
                                                         true, 1, GROUP>::Kernel>();    \
      case 12:                                                                           \
        return new KernelImpl<typename EH::template Type<16, 128, 32, 1, 4, 1, D, S, 2, \
                                                         true, 1, GROUP>::Kernel>();    \
      case 13:                                                                           \
        return new KernelImpl<typename EH::template Type<32, 128, 64, 1, 4, 1, D, S, 2, \
                                                         true, 1, GROUP>::Kernel>();    \
    }                                                                                    \
    return nullptr;                                                                      \
  }

}  // namespace ds41kern
