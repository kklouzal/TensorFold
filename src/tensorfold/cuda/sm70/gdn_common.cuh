// Arithmetic shared by every SM70 GDN recurrence kernel (gdn.cu chain/replay, gdn_prefill_split.cu,
// gdn_replay_many.cu). Their bitwise agreement depends on these exact expression trees and xor-butterfly
// order, compiled with --fmad=false in every including translation unit.
#pragma once

#include <cuda_fp16.h>
#include "../kernels/gdn_recurrence.cuh"

namespace {

constexpr int DK = 128, DV = 128, TAPS = 4;

__device__ __forceinline__ float bf(float x) { return __half2float(__float2half_rn(x)); }


}  // namespace
