// Staged IEEE FP32 RotorQuant basis; shared by the codec and SM70 attention.
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "rotorquant_q4_tables.cuh"

namespace tf_rq4 {

__device__ __forceinline__ float centroid(int code) {
    return __uint_as_float(centroid_bits[code]);
}

__device__ __forceinline__ int index(float value) {
    int code = 0;
    #pragma unroll
    for (int i = 0; i < 15; ++i) code += value > __uint_as_float(threshold_bits[i]);
    return code;  // Exact thresholds, including +/-zero, choose the lower bin.
}

template<int CODEC>
__device__ __forceinline__ float selected_centroid(int code) {
    if constexpr (CODEC == 9 || CODEC == 10) return __uint_as_float(compact_centroid_bits[code]);
    return centroid(code);
}

template<int CODEC>
__device__ __forceinline__ float metadata_scale(float word) {
    if constexpr (CODEC == 9 || CODEC == 10) return __half2float(__ushort_as_half(__float_as_uint(word) & 0xffffu));
    return word;
}

template<int CODEC>
__device__ __forceinline__ bool valid_metadata(float word) {
    const float scale = metadata_scale<CODEC>(word);
    if constexpr (CODEC == 9 || CODEC == 10) {
        if (__float_as_uint(word) >> 16) return false;
    }
    return isfinite(scale) && scale >= 0.0f;
}

template<int LOW, bool INVERSE>
__device__ __forceinline__ float quaternion(float value, int col, float* shared) {
    shared[col] = value;
    __syncthreads();
    const int low = col % LOW, base = col / (4 * LOW) * (4 * LOW);
    const int component = col / LOW % 4;
    const float a = shared[base + low], b = shared[base + LOW + low];
    const float c = shared[base + 2 * LOW + low], d = shared[base + 3 * LOW + low];
    float result;
    if constexpr (INVERSE) {
        if (component == 0) result = __fadd_rn(__fadd_rn(__fadd_rn(a, b), c), d);
        else if (component == 1) result = __fsub_rn(__fadd_rn(__fadd_rn(-a, b), c), d);
        else if (component == 2) result = __fadd_rn(__fadd_rn(__fsub_rn(-a, b), c), d);
        else result = __fadd_rn(__fsub_rn(__fadd_rn(-a, b), c), d);
    } else {
        if (component == 0) result = __fsub_rn(__fsub_rn(__fsub_rn(a, b), c), d);
        else if (component == 1) result = __fadd_rn(__fsub_rn(__fadd_rn(a, b), c), d);
        else if (component == 2) result = __fsub_rn(__fadd_rn(__fadd_rn(a, b), c), d);
        else result = __fadd_rn(__fadd_rn(__fsub_rn(a, b), c), d);
    }
    __syncthreads();  // No producer may overwrite a value another component still reads.
    return __fmul_rn(result, 0.5f);
}

template<bool INVERSE>
__device__ __forceinline__ float wide(float value, int col, float* shared) {
    shared[col] = value;
    __syncthreads();
    const int base = col / 128 * 128, low = col % 64;
    const float a = shared[base + low], b = shared[base + 64 + low];
    const float c = __uint_as_float(wide_cos_bits), s = __uint_as_float(wide_sin_bits);
    float result;
    if constexpr (INVERSE) {
        result = col % 128 < 64 ? __fadd_rn(__fmul_rn(c, a), __fmul_rn(s, b))
                               : __fadd_rn(__fmul_rn(-s, a), __fmul_rn(c, b));
    } else {
        result = col % 128 < 64 ? __fsub_rn(__fmul_rn(c, a), __fmul_rn(s, b))
                               : __fadd_rn(__fmul_rn(s, a), __fmul_rn(c, b));
    }
    __syncthreads();
    return result;
}

template<int CODEC, bool INVERSE>
__device__ __forceinline__ float rotate_component(float value, int col, float* shared) {
    static_assert((CODEC >= 1 && CODEC <= 5) || CODEC == 9 || CODEC == 10);
    // All CTA threads participate, col==threadIdx.x, blockDim.x is a multiple
    // of128 and shared has blockDim.x floats. There are no per-value barriers.
    if constexpr (CODEC == 1 || CODEC == 9) {
        shared[col] = value;
        __syncthreads();
        const int pair = col / 2 % (CODEC == 9 ? 64 : 4), base = col / 2 * 2;
        const float a = shared[base], b = shared[base + 1];
        const float c = __uint_as_float(CODEC == 9 ? compact_cos_bits[pair] : planar_cos_bits[pair]);
        const float s = __uint_as_float(CODEC == 9 ? compact_sin_bits[pair] : planar_sin_bits[pair]);
        if constexpr (INVERSE) {
            value = col % 2 == 0 ? __fadd_rn(__fmul_rn(c, a), __fmul_rn(s, b))
                                 : __fadd_rn(__fmul_rn(-s, a), __fmul_rn(c, b));
        } else {
            value = col % 2 == 0 ? __fsub_rn(__fmul_rn(c, a), __fmul_rn(s, b))
                                 : __fadd_rn(__fmul_rn(s, a), __fmul_rn(c, b));
        }
        __syncthreads();
        return value;
    }
    if constexpr (CODEC == 10) {
        shared[col] = value;
        __syncthreads();
        const int group = (col / 4) % 32, base = col / 4 * 4;
        const float a = shared[base], b = shared[base + 1];
        const float c = shared[base + 2], d = shared[base + 3];
        const float w = __uint_as_float(compact_qw_bits[group]);
        float x = __uint_as_float(compact_qx_bits[group]);
        float y = __uint_as_float(compact_qy_bits[group]);
        float z = __uint_as_float(compact_qz_bits[group]);
        if constexpr (INVERSE) { x = -x; y = -y; z = -z; }
        float result;
        if (col % 4 == 0)
            result = __fsub_rn(__fsub_rn(__fsub_rn(__fmul_rn(w, a), __fmul_rn(x, b)), __fmul_rn(y, c)), __fmul_rn(z, d));
        else if (col % 4 == 1)
            result = __fsub_rn(__fadd_rn(__fadd_rn(__fmul_rn(w, b), __fmul_rn(x, a)), __fmul_rn(y, d)), __fmul_rn(z, c));
        else if (col % 4 == 2)
            result = __fadd_rn(__fadd_rn(__fsub_rn(__fmul_rn(w, c), __fmul_rn(x, d)), __fmul_rn(y, a)), __fmul_rn(z, b));
        else
            result = __fadd_rn(__fsub_rn(__fadd_rn(__fmul_rn(w, d), __fmul_rn(x, c)), __fmul_rn(y, b)), __fmul_rn(z, a));
        __syncthreads();
        return result;
    }
    if constexpr ((CODEC == 4 || CODEC == 5) && !INVERSE)
        if ((sign_pair_mask >> ((col / 2) % 64)) & 1) value = -value;
    if constexpr (CODEC == 4 && INVERSE) value = wide<true>(value, col, shared);
    if constexpr (INVERSE) {
        if constexpr (CODEC > 2) {
            value = quaternion<16, true>(value, col, shared);
            value = quaternion<4, true>(value, col, shared);
        }
        value = quaternion<1, true>(value, col, shared);
    } else {
        value = quaternion<1, false>(value, col, shared);
        if constexpr (CODEC > 2) {
            value = quaternion<4, false>(value, col, shared);
            value = quaternion<16, false>(value, col, shared);
        }
    }
    if constexpr (CODEC == 4 && !INVERSE) value = wide<false>(value, col, shared);
    if constexpr ((CODEC == 4 || CODEC == 5) && INVERSE)
        if ((sign_pair_mask >> ((col / 2) % 64)) & 1) value = -value;
    return value;
}

__device__ __forceinline__ float inverse_component(float value, int col, float* shared) {
    return rotate_component<4, true>(value, col, shared);
}

template<bool MAXIMUM>
__device__ __forceinline__ float reduce_group(float value, int col, float* shared) {
    shared[col] = value;
    __syncthreads();
    const int lane = col % 128, base = col / 128 * 128;
    #pragma unroll
    for (int distance = 64; distance; distance /= 2) {
        if (lane < distance) {
            if constexpr (MAXIMUM) shared[col] = fmaxf(shared[col], shared[col + distance]);
            else shared[col] = __fadd_rn(shared[col], shared[col + distance]);
        }
        __syncthreads();
    }
    const float result = shared[base];
    __syncthreads();
    return result;
}

struct Encoded {
    int code;
    float scale;
    int errors;
};

template<int CODEC>
__device__ __forceinline__ Encoded encode_component(float value, int col, float* shared, int source_error) {
    // The encoder serves the FP16-working policy. A finite wire scale alone
    // does not prove that its selected centroid is representable as Half.
    const bool bad_source = __syncthreads_or(!isfinite(value));
    if (bad_source) value = 0.0f;
    if constexpr (CODEC == 9 || CODEC == 10) {
        const float energy = reduce_group<false>(__fmul_rn(value, value), col, shared);
        const float norm = __fsqrt_rn(energy);
        const float unit = norm > 1e-10f ? __fmul_rn(value, __fdiv_rn(1.0f, norm)) : 0.0f;
        const float rotated = rotate_component<CODEC, false>(unit, col, shared);
        int code = 0;
        float distance = fabsf(__fsub_rn(rotated, selected_centroid<CODEC>(0)));
        #pragma unroll
        for (int i = 1; i < 16; ++i) {
            const float candidate = fabsf(__fsub_rn(rotated, selected_centroid<CODEC>(i)));
            if (candidate < distance) { distance = candidate; code = i; }
        }
        const float selected = selected_centroid<CODEC>(code);
        const float reconstruction = __fsqrt_rn(reduce_group<false>(__fmul_rn(selected, selected), col, shared));
        const float corrected = reconstruction > 1e-10f ? __fdiv_rn(norm, reconstruction) : norm;
        const __half half_scale = __float2half_rn(corrected);
        const float scale = __half2float(half_scale);
        const bool bad = __syncthreads_or(!isfinite(norm) || !isfinite(scale)
            || (norm > 1e-10f && scale == 0.0f)
            || !isfinite(__half2float(__float2half_rn(__fmul_rn(selected, scale)))));
        const int errors = (bad_source ? source_error : 0) | (bad ? 8 : 0);
        return {errors ? 7 : code, errors ? 0.0f : __uint_as_float(__half_as_ushort(half_scale)), errors};
    }
    const float maximum = reduce_group<true>(fabsf(value), col, shared);
    const float scaled = __fdiv_rn(value, maximum > 0.0f ? maximum : 1.0f);
    const float energy = reduce_group<false>(__fmul_rn(scaled, scaled), col, shared);
    const float norm = __fsqrt_rn(__fmul_rn(energy, 0.0078125f));
    const float unit = __fdiv_rn(scaled, norm > 0.0f ? norm : 1.0f);
    float scale = __fmul_rn(maximum, norm);
    const float rotated = rotate_component<CODEC, false>(unit, col, shared);
    int code = index(rotated);
    if constexpr (CODEC == 3 || CODEC == 5) {
        const float center = centroid(code);
        const float coded_energy = reduce_group<false>(__fmul_rn(center, center), col, shared);
        scale = __fmul_rn(scale, __fsqrt_rn(__fdiv_rn(128.0f, coded_energy)));
    }
    const float reconstruction = __fmul_rn(centroid(code), scale);
    const bool bad_scale = (!isfinite(scale) || scale < 0.0f || (maximum > 0.0f && scale == 0.0f)
                            || !isfinite(reconstruction)
                            || !isfinite(__half2float(__float2half_rn(reconstruction))));
    const bool bad_working = __syncthreads_or(bad_scale);
    const int errors = (bad_source ? source_error : 0) | (bad_working ? 8 : 0);
    if (errors) { code = 7; scale = 0.0f; }
    return {code, scale, errors};
}

__device__ __forceinline__ void store_encoded(Encoded item, int col, float* shared,
                                             uint8_t* codes, float* scales,
                                             int64_t scalar_base, int dim, bool write) {
    shared[col] = float(item.code);
    __syncthreads();
    if (write && col < dim) {
        if (col % 2 == 0)
            codes[(scalar_base + col) / 2] = uint8_t(int(shared[col]) | (int(shared[col + 1]) << 4));
        if (col % 128 == 0) scales[(scalar_base + col) / 128] = item.scale;
    }
    __syncthreads();
}

}  // namespace tf_rq4
