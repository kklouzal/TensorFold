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
// A parallel bin search and contiguous-segment emission (one block scan) replace the single-thread bin walk
// and the per-chunk scans (same selection, bitwise; dec6 T0 ~2x faster at 4K..256K).
// select_greater is the general body; family.cu's captured owned selector
// (qsa_select_owned_kernel) and eager long-context rows (qsa_select_rows_kernel) use it,
// and the standalone kernel below keeps its contract.
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
    uint32_t bin_warp_total[kBins / 32];
    typename Scan::TempStorage scan;
    uint32_t prefix, remaining, invalid;
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

// The general (length > kTop) body: kTop indices of the largest finite scores in [0, length), numeric order
// with ties to the lower logical index, emitted as emit(rank, index) in ascending index order. Returns 1 (and
// emits nothing) when a score in the prefix is nonfinite, else 0. Every thread of the block must call it; the
// caller initialized nothing in ``shared``.
template <typename Emit>
__device__ __forceinline__ uint32_t select_greater(const float* __restrict__ values, uint32_t length, Shared& shared,
                                                   Emit emit) {
    const uint32_t tx = threadIdx.x;
    if (tx == 0) {
        shared.invalid = shared.prefix = 0;
        shared.remaining = kTop;
    }
    __syncthreads();
    // Four exact radix passes find the kth numeric score. The final logical-ID
    // scan resolves pivot ties; no floating reduction/atomics or full-sort workspace.
    const uint32_t lane = tx & 31;
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
        if (shared.invalid) return 1;
        // The highest bin whose count, accumulated from the top bin down, reaches ``remaining`` (exactly one bin
        // satisfies exclusive < remaining <= inclusive): a parallel scan of the 256 descending counts.
        // ``remaining`` is read before the barrier below: the winning thread rewrites it after that barrier.
        const uint32_t remaining = shared.remaining;
        uint32_t count = 0, inclusive = 0;
        if (tx < kBins) {
            count = inclusive = shared.histogram[kBins - 1 - tx];
#pragma unroll
            for (uint32_t offset = 1; offset < 32; offset <<= 1) {
                const uint32_t other = __shfl_up_sync(0xffffffffu, inclusive, offset);
                if (lane >= offset) inclusive += other;
            }
            if (lane == 31) shared.bin_warp_total[tx >> 5] = inclusive;
        }
        __syncthreads();
        if (tx < kBins) {
            for (uint32_t warp = 0; warp < (tx >> 5); ++warp) inclusive += shared.bin_warp_total[warp];
            const uint32_t exclusive = inclusive - count;
            if (exclusive < remaining && remaining <= inclusive) {
                shared.prefix = prefix | (uint32_t(kBins - 1 - tx) << shift);
                shared.remaining = remaining - exclusive;
            }
        }
        __syncthreads();
    }
    const uint32_t pivot = shared.prefix, remaining = shared.remaining;
    // Each thread owns one contiguous index segment, so a single block scan of the packed greater/pivot-equal
    // counts orders every emission by ascending index (length <= 2^31: the packed low word cannot carry).
    const uint32_t per = (length + kThreads - 1) / kThreads;
    const uint32_t begin = min(length, tx * per), end = min(length, begin + per);
    uint32_t greater = 0, equal = 0;
    for (uint32_t index = begin; index < end; ++index) {
        const uint32_t key = ordered_bits(__float_as_uint(values[index]));
        greater += key > pivot;
        equal += key == pivot;
    }
    uint64_t before = 0, total = 0;
    Shared::Scan(shared.scan).ExclusiveSum((uint64_t(greater) << 32) | equal, before, total);
    uint32_t greater_before = uint32_t(before >> 32), equal_before = uint32_t(before);
    for (uint32_t index = begin; index < end; ++index) {
        const uint32_t key = ordered_bits(__float_as_uint(values[index]));
        if (key > pivot) emit(greater_before++ + min(equal_before, remaining), index);
        else if (key == pivot) {
            if (equal_before < remaining) emit(greater_before + equal_before, index);
            ++equal_before;
        }
    }
    return 0;
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

    if (select_greater(values, length, shared, [ids](uint32_t rank, uint32_t index) { ids[rank] = int32_t(index); })) {
        if (tx == 0) status[row] = 1;
        invalid_ids(ids);
    }
}

}  // namespace tensorfold::sm70::qsa_select512
