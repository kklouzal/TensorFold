// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Adapted from 1CatAI/1Cat-vLLM csrc/qsa_lexicographic_topk.cuh,
// commit 1b57260a834d40faa32bfc6ac18f050414a8f042 (general 1024-thread body).
// https://github.com/1CatAI/1Cat-vLLM/blob/1b57260a834d40faa32bfc6ac18f050414a8f042/csrc/qsa_lexicographic_topk.cuh
// Full license: LICENSES/Apache-2.0.txt in the TensorFold source/distribution.
// TensorFold modifications: fixed K512; padded output stride; no length clamp;
// deterministic device status/invalid output; finite-prefix containment; bitwise
// zero canonicalization (no FTZ-dependent comparison). Decode fastpaths omitted.
// This is an isolated candidate; it changes no score arithmetic or dispatch.
#pragma once

#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>
#include <cstdint>

namespace tensorfold::sm70::qsa_select512 {
constexpr uint32_t kThreads = 1024;
constexpr uint32_t kBins = 256;
constexpr uint32_t kTop = 512;
// Each row: 0 success, 1 nonfinite valid-prefix score, 2 invalid causal length.
struct Shared {
    using Scan = cub::BlockScan<uint64_t, kThreads>;
    uint32_t histogram[kBins];
    typename Scan::TempStorage scan;
    uint32_t prefix, pivot, remaining, greater_seen, equal_seen;
    uint32_t chunk_greater_base, chunk_equal_base, invalid;
};

__device__ __forceinline__ uint32_t ordered_bits(uint32_t bits) {
    // Numeric ties include both signed zeros. Every nonzero finite bit pattern,
    // including either sign of subnormal, retains its exact numeric ordering.
    if ((bits & 0x7fffffffu) == 0) bits = 0;
    return bits & 0x80000000u ? ~bits : bits | 0x80000000u;
}
__device__ __forceinline__ bool finite_bits(uint32_t bits) {
    return (bits & 0x7f800000u) != 0x7f800000u;
}
__device__ __forceinline__ void invalid_ids(int32_t* output) {
    for (uint32_t index = threadIdx.x; index < kTop; index += kThreads) output[index] = -1;
}

__global__ __launch_bounds__(1024) void kernel(
        const float* __restrict__ scores, const int32_t* __restrict__ lengths,
        int32_t* __restrict__ output, int32_t* __restrict__ status,
        uint32_t columns, uint64_t score_stride, uint64_t output_stride) {
    __shared__ Shared shared;
    const uint32_t row = blockIdx.x, tx = threadIdx.x;
    const int32_t raw_length = lengths[row];
    const float* values = scores + uint64_t(row) * score_stride;
    int32_t* ids = output + uint64_t(row) * output_stride;
    if (raw_length < 0 || uint32_t(raw_length) > columns) {
        if (tx == 0) status[row] = 2;
        invalid_ids(ids);
        return;  // Uniform row decision; no score access outside the supplied prefix.
    }
    const uint32_t length = uint32_t(raw_length);
    if (tx == 0) {
        status[row] = 0;
        shared.invalid = shared.prefix = 0;
        shared.remaining = kTop;
    }
    __syncthreads();
    if (length <= kTop) {
        if (tx < length && !finite_bits(__float_as_uint(values[tx]))) atomicOr(&shared.invalid, 1u);
        __syncthreads();
        if (shared.invalid) {
            if (tx == 0) status[row] = 1;
            invalid_ids(ids);
        } else {
            for (uint32_t index = tx; index < kTop; index += kThreads)
                ids[index] = index < length ? int32_t(index) : -1;
        }
        return;
    }

    // Four exact radix passes find the kth numeric score. The final logical-ID
    // scan resolves pivot ties; no floating reduction/atomics or full-sort workspace.
#pragma unroll
    for (int pass = 0; pass < 4; ++pass) {
        if (tx < kBins) shared.histogram[tx] = 0;
        __syncthreads();
        const int shift = 24 - pass * 8;
        const uint32_t prefix = shared.prefix;
        const uint32_t mask = pass == 0 ? 0 : (~uint32_t{0} << (shift + 8));
        for (uint32_t index = tx; index < length; index += kThreads) {
            const uint32_t bits = __float_as_uint(values[index]);
            if (pass == 0 && !finite_bits(bits)) atomicOr(&shared.invalid, 1u);
            const uint32_t key = ordered_bits(bits);
            if ((key & mask) == prefix) atomicAdd(&shared.histogram[(key >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (shared.invalid) {
            if (tx == 0) status[row] = 1;
            invalid_ids(ids);
            return;
        }
        if (tx == 0) {
            uint32_t remaining = shared.remaining;
            for (int bin = kBins - 1; bin >= 0; --bin) {
                const uint32_t count = shared.histogram[bin];
                if (remaining > count) remaining -= count;
                else {
                    shared.prefix |= uint32_t(bin) << shift;
                    shared.remaining = remaining;
                    break;
                }
            }
        }
        __syncthreads();
    }
    if (tx == 0) {
        shared.pivot = shared.prefix;
        shared.greater_seen = shared.equal_seen = 0;
    }
    __syncthreads();
    // Packed integer scans track greater/pivot-equal cardinalities. A chunk has
    // <=1024 entries, so its low-word sum cannot carry into the greater count.
    for (uint32_t base = 0; base < length; base += kThreads) {
        const uint32_t index = base + tx;
        const uint32_t key = index < length ? ordered_bits(__float_as_uint(values[index])) : 0;
        const uint32_t greater = index < length && key > shared.pivot;
        const uint32_t equal = index < length && key == shared.pivot;
        uint64_t before = 0, total = 0;
        Shared::Scan(shared.scan).ExclusiveSum((uint64_t(greater) << 32) | equal, before, total);
        __syncthreads();
        if (tx == 0) {
            shared.chunk_greater_base = shared.greater_seen;
            shared.chunk_equal_base = shared.equal_seen;
            shared.greater_seen += uint32_t(total >> 32);
            shared.equal_seen += uint32_t(total);
        }
        __syncthreads();
        const uint32_t greater_before = shared.chunk_greater_base + uint32_t(before >> 32);
        const uint32_t equal_before = shared.chunk_equal_base + uint32_t(before);
        if (greater || (equal && equal_before < shared.remaining))
            ids[greater_before + min(equal_before, shared.remaining)] = int32_t(index);
        __syncthreads();
    }
}
}  // namespace tensorfold::sm70::qsa_select512
