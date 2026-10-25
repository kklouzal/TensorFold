// Shared scalar-value decoding for the registered SM70 packed KV ABI.
// The caller proves logical/physical bounds and owns validated codec metadata.
// Preserve expression order and the single final FP16 working-operand rounding.
#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include "rotorquant_q4.cuh"

namespace tf_sm70 {
__device__ __forceinline__ __half decoded_half(const void* data, const void* scales, int bits, int codec, int64_t at) {
    if (!bits) return static_cast<const __half*>(data)[at];
    if (codec == 9 || codec == 10) {
        const unsigned byte = static_cast<const uint8_t*>(data)[at / 2];
        const int index = (byte >> (4 * (at & 1))) & 15;
        const float word = static_cast<const float*>(scales)[at / 128];
        return __float2half_rn(__fmul_rn(tf_rq4::selected_centroid<9>(index), tf_rq4::metadata_scale<9>(word)));
    }
    if (codec == 4 || codec == 5) {
        const unsigned byte = static_cast<const uint8_t*>(data)[at / 2];
        const int index = (byte >> (4 * (at & 1))) & 15;
        return __float2half_rn(__fmul_rn(tf_rq4::centroid(index), static_cast<const float*>(scales)[at / 128]));
    }
    float value = ((float(static_cast<const int8_t*>(data)[at]) + 0.5f) *
                   __half2float(static_cast<const __half*>(scales)[at / 32])) * (1.f / 128.f);
    return __float2half_rn(value);
}
}  // namespace tf_sm70
