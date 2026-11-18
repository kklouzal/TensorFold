// Shared scalar-value decoding for the registered SM70 packed KV ABI.
// The caller proves logical/physical bounds and owns validated codec metadata.
// Preserve expression order and the single final FP16 working-operand rounding.
#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include "rotorquant_q4.cuh"

namespace tf_sm70 {
// Element decoders: the one definition of each codec's expression.
__device__ __forceinline__ __half decoded_q8(int8_t q, __half scale) {
    float value = ((float(q) + 0.5f) * __half2float(scale)) * (1.f / 128.f);
    return __float2half_rn(value);
}

__device__ __forceinline__ __half decoded_q4(int codec, int index, float word) {
    if (codec == 9 || codec == 10)
        return __float2half_rn(__fmul_rn(tf_rq4::selected_centroid<9>(index), tf_rq4::metadata_scale<9>(word)));
    return __float2half_rn(__fmul_rn(tf_rq4::centroid(index), word));
}

// Eight-bit RotorQuant (kernel codecs 8/12): FP32 centroid times FP32 group scale, one final FP16 rounding.
// `table` is the 256-entry centroid table (a shared-memory copy in the RN kernels; the constant table otherwise).
__device__ __forceinline__ __half decoded_q8_rotor(int index, float scale, const float* table) {
    return __float2half_rn(__fmul_rn(table ? table[index] : tf_rq4::centroid8(index), scale));
}

// Bits of a kernel codec's stored value: 0 raw FP16 is handled by the callers' bits argument.
__host__ __device__ constexpr int codec_bits(int codec) {
    return codec == 0 || tf_rq4::eight_bit(codec) ? 8 : 4;
}

// Shared-memory centroid table for the eight-bit codecs: every CTA thread calls this (it ends with a barrier).
__device__ __forceinline__ const float* stage_centroids8(int codec, float* table) {
    if (!tf_rq4::eight_bit(codec)) return nullptr;
    for (int i = threadIdx.x; i < 256; i += blockDim.x) table[i] = tf_rq4::centroid8(i);
    __syncthreads();
    return table;
}

__device__ __forceinline__ __half decoded_half(const void* data, const void* scales, int bits, int codec, int64_t at,
                                               const float* table = nullptr) {
    if (!bits) return static_cast<const __half*>(data)[at];
    if (tf_rq4::eight_bit(codec))
        return decoded_q8_rotor(static_cast<const uint8_t*>(data)[at], static_cast<const float*>(scales)[at / 128], table);
    if (codec == 9 || codec == 10 || codec == 4 || codec == 5) {
        const unsigned byte = static_cast<const uint8_t*>(data)[at / 2];
        const int index = (byte >> (4 * (at & 1))) & 15;
        return decoded_q4(codec, index, static_cast<const float*>(scales)[at / 128]);
    }
    return decoded_q8(static_cast<const int8_t*>(data)[at], static_cast<const __half*>(scales)[at / 32]);
}

// Packed runs of N in {8, 16} consecutive values of one row for the Q8
// (codec 0), eight-bit RotorQuant (8, 12) and 4-bit (codecs 4, 5, 9, 10) formats: one vector payload load and
// one scale-word load (a run never crosses a 32-value Q8 or 128-value 4-bit
// scale group because at % N == 0). The payload base must be 16-byte aligned.
// decode_run reproduces decoded_half for every element.
template<int N>
__device__ __forceinline__ void load_run(const void* data, const void* scales, int codec, int64_t at,
                                         uint4& raw, unsigned& scale) {
    static_assert(N == 8 || N == 16, "packed runs hold 8 or 16 values");
    if (codec == 0 || tf_rq4::eight_bit(codec)) {
        const int8_t* source = static_cast<const int8_t*>(data) + at;
        if constexpr (N == 16) raw = *reinterpret_cast<const uint4*>(source);
        else { const uint2 v = *reinterpret_cast<const uint2*>(source); raw = make_uint4(v.x, v.y, 0u, 0u); }
        if (codec == 0) scale = __half_as_ushort(static_cast<const __half*>(scales)[at / 32]);
        else scale = __float_as_uint(static_cast<const float*>(scales)[at / 128]);
    } else {
        const uint8_t* source = static_cast<const uint8_t*>(data) + at / 2;
        if constexpr (N == 16) { const uint2 v = *reinterpret_cast<const uint2*>(source); raw = make_uint4(v.x, v.y, 0u, 0u); }
        else raw = make_uint4(*reinterpret_cast<const unsigned*>(source), 0u, 0u, 0u);
        scale = __float_as_uint(static_cast<const float*>(scales)[at / 128]);
    }
}

template<int N>
__device__ __forceinline__ __half decode_run_value(const uint4& raw, unsigned scale, int codec, int i,
                                                   const float* table = nullptr) {
    const unsigned words[4] = {raw.x, raw.y, raw.z, raw.w};
    if (codec == 0) {
        const int8_t q = int8_t((words[i / 4] >> (8 * (i % 4))) & 0xffu);
        return decoded_q8(q, __ushort_as_half((unsigned short)scale));
    }
    if (tf_rq4::eight_bit(codec))
        return decoded_q8_rotor(int((words[i / 4] >> (8 * (i % 4))) & 0xffu), __uint_as_float(scale), table);
    const unsigned byte = (words[i / 8] >> (8 * (i / 2 % 4))) & 0xffu;
    return decoded_q4(codec, (byte >> (4 * (i & 1))) & 15, __uint_as_float(scale));
}
}  // namespace tf_sm70
