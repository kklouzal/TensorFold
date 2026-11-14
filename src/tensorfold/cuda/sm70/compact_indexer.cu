#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>
#include <cstdint>
#include <cuda_fp16.h>

namespace {
__global__ void publish_kernel(const uint8_t* source, uint8_t* destination, int64_t bytes,
                               const int64_t* generations, int64_t slot, int64_t owner,
                               int* status, int layer) {
    if (generations[0] != slot || generations[1] != owner) {
        if (!blockIdx.x && !threadIdx.x) { atomicOr(status, 32); atomicCAS(status + 1, -2, layer); }
        return;
    }
    for (int64_t at = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         at < bytes; at += int64_t(gridDim.x) * blockDim.x)
        destination[at] = source[at];
}

// Round-table lane words (one State segment of one indexer ordinal). Produced
// by the canonical owner from retained State tensors; see compact_indexer.py.
namespace lane {
enum Word : int {
    ENABLE = 0, RING_RAW, RING_COORDS, PROTECTED, PROTECTED_COORDS, STAGE_RAW, STAGE_COORDS, POOLED,
    GENERATIONS, STATUS, EXPECTED_SLOT, EXPECTED_OWNER, POSITION, FIRST, ROWS, POOLED_ROWS,
    PROPOSAL_RAW, PROPOSAL_COORDS, PROPOSAL_OFFSET, RING_COMMIT, COUNT
};
}
constexpr int RATIO = 4, DIM = 128, GROUPS = 16;

template<typename V> __device__ __forceinline__ V* word(const int64_t* table, int index) {
    return reinterpret_cast<V*>(uintptr_t(table[index]));
}

__device__ __forceinline__ void latch(int* status, int bits, int layer) {
    atomicOr(status, bits);
    atomicCAS(status + 1, -2, layer);
}

// Stage new raw rows, protect the committed ring and gather the bounded
// [GROUPS, RATIO] pooling window. Rows outside complete groups gather zeros.
__global__ void round_stage_kernel(const __half* ik, const int64_t* tables, __half* raw_out,
                                   int64_t* coords_out, int layer) {
    const int64_t* t = tables + int64_t(blockIdx.x) * lane::COUNT;
    __half* raw = raw_out + int64_t(blockIdx.x) * GROUPS * RATIO * DIM;
    int64_t* coords = coords_out + int64_t(blockIdx.x) * GROUPS * 3;
    if (!t[lane::ENABLE]) {
        for (int at = threadIdx.x; at < GROUPS * RATIO * DIM; at += blockDim.x) raw[at] = __float2half_rn(0.f);
        for (int at = threadIdx.x; at < GROUPS * 3; at += blockDim.x) coords[at] = 0;
        return;
    }
    const int64_t* generations = word<const int64_t>(t, lane::GENERATIONS);
    int* status = word<int>(t, lane::STATUS);
    const int64_t first = t[lane::FIRST], rows = t[lane::ROWS], start = *word<const int>(t, lane::POSITION);
    const __half* ring = word<const __half>(t, lane::RING_RAW);
    const int64_t* ring_coords = word<const int64_t>(t, lane::RING_COORDS);
    const int64_t* stage_coords = word<const int64_t>(t, lane::STAGE_COORDS);
    __half* stage = word<__half>(t, lane::STAGE_RAW);
    for (int64_t at = threadIdx.x; at < rows * DIM; at += blockDim.x) stage[at] = ik[first * DIM + at];
    if (generations[0] != t[lane::EXPECTED_SLOT] || generations[1] != t[lane::EXPECTED_OWNER]) {
        if (!threadIdx.x) latch(status, 32, layer);
    } else {
        __half* protect = word<__half>(t, lane::PROTECTED);
        int64_t* protect_coords = word<int64_t>(t, lane::PROTECTED_COORDS);
        for (int at = threadIdx.x; at < RATIO * DIM; at += blockDim.x) protect[at] = ring[at];
        for (int at = threadIdx.x; at < RATIO * 3; at += blockDim.x) protect_coords[at] = ring_coords[at];
        // MTP proposal span: the new rows also enter the speculative archive.
        const int64_t offset = t[lane::PROPOSAL_OFFSET];
        if (offset >= 0) {
            __half* proposal = word<__half>(t, lane::PROPOSAL_RAW) + offset * DIM;
            int64_t* proposal_coords = word<int64_t>(t, lane::PROPOSAL_COORDS) + offset * 3;
            for (int64_t at = threadIdx.x; at < rows * DIM; at += blockDim.x) proposal[at] = ik[first * DIM + at];
            for (int64_t at = threadIdx.x; at < rows * 3; at += blockDim.x) proposal_coords[at] = stage_coords[at];
        }
    }
    const int64_t group0 = start / RATIO, groups = (start + rows) / RATIO - group0;
    for (int at = threadIdx.x; at < GROUPS * RATIO * DIM; at += blockDim.x) {
        const int group = at / (RATIO * DIM), member = at / DIM % RATIO, col = at % DIM;
        const int64_t position = (group0 + group) * RATIO + member;
        __half value = __float2half_rn(0.f);
        if (group < groups) {
            if (position < start) value = ring[(position % RATIO) * DIM + col];
            else value = ik[(first + min(position - start, rows - 1)) * DIM + col];
        }
        raw[at] = value;
    }
    for (int at = threadIdx.x; at < GROUPS * 3; at += blockDim.x) {
        const int group = at / 3, axis = at % 3;
        const int64_t position = (group0 + group) * RATIO;
        int64_t value = 0;
        if (group < groups)
            value = position < start ? ring_coords[(position % RATIO) * 3 + axis]
                                     : stage_coords[min(position - start, rows - 1) * 3 + axis];
        coords[at] = value;
    }
}

// Guard and generation-checked publication of the complete pooled groups.
__global__ void round_publish_kernel(const __half* values, const int64_t* tables, int layer) {
    const int64_t* t = tables + int64_t(blockIdx.x) * lane::COUNT;
    if (!t[lane::ENABLE]) return;
    int* status = word<int>(t, lane::STATUS);
    const int64_t start = *word<const int>(t, lane::POSITION), rows = t[lane::ROWS];
    const int64_t group0 = start / RATIO, groups = (start + rows) / RATIO - group0;
    const int64_t* generations = word<const int64_t>(t, lane::GENERATIONS);
    const bool current = generations[0] == t[lane::EXPECTED_SLOT] && generations[1] == t[lane::EXPECTED_OWNER];
    const bool fits = groups >= 0 && groups <= GROUPS && group0 >= 0 && group0 + groups <= t[lane::POOLED_ROWS];
    if ((!current || !fits) && groups > 0 && !threadIdx.x) latch(status, 32, layer);
    __half* pooled = word<__half>(t, lane::POOLED);
    values += int64_t(blockIdx.x) * GROUPS * DIM;
    const int count = int(groups < GROUPS ? groups : GROUPS);
    for (int group = 0; group < count; ++group) {
        int bad = 0;
        for (int col = threadIdx.x; col < DIM; col += blockDim.x) {
            const __half value = values[group * DIM + col];
            const float f = __half2float(value);
            const bool ok = isfinite(f) && fabsf(f) <= 1073741824.f;
            bad |= !ok;
            if (current && fits) pooled[(group0 + group) * DIM + col] = ok ? value : __float2half(0.f);
        }
        if (__syncthreads_or(bad) && !threadIdx.x) latch(status, 16, layer);
    }
    // MTP commit of the window end: the ring keeps only the incomplete last
    // group's raw rows (older rows from the protected ring), zeros elsewhere.
    if (t[lane::RING_COMMIT] && current) {
        const int64_t end = start + rows, count = end % RATIO;
        const __half* protect = word<const __half>(t, lane::PROTECTED);
        const int64_t* protect_coords = word<const int64_t>(t, lane::PROTECTED_COORDS);
        const __half* stage = word<const __half>(t, lane::STAGE_RAW);
        const int64_t* stage_coords = word<const int64_t>(t, lane::STAGE_COORDS);
        __half* ring = word<__half>(t, lane::RING_RAW);
        int64_t* ring_coords = word<int64_t>(t, lane::RING_COORDS);
        for (int at = threadIdx.x; at < RATIO * (DIM + 3); at += blockDim.x) {
            const bool coordinate = at >= RATIO * DIM;
            const int item = coordinate ? at - RATIO * DIM : at;
            const int slot = coordinate ? item / 3 : item / DIM, col = coordinate ? item % 3 : item % DIM;
            const int64_t position = end - count + ((slot - (end - count) % RATIO) % RATIO + RATIO) % RATIO;
            const bool live = position < end;
            if (coordinate) {
                ring_coords[item] = !live ? 0 : position < start ? protect_coords[slot * 3 + col]
                                                                  : stage_coords[(position - start) * 3 + col];
            } else {
                ring[item] = !live ? __float2half_rn(0.f) : position < start ? protect[slot * DIM + col]
                                                                              : stage[(position - start) * DIM + col];
            }
        }
    } else if (t[lane::RING_COMMIT] && !threadIdx.x) {
        latch(status, 32, layer);
    }
}
}

void compact_publish_cuda(const at::Tensor& source, at::Tensor& destination, int64_t offset,
                          const at::Tensor& generations, int64_t slot, int64_t owner, at::Tensor& status, int layer) {
    const int64_t bytes = source.numel() * source.element_size();
    const int blocks = int(std::min<int64_t>((bytes - 1) / 256 + 1, 65535));
    publish_kernel<<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint8_t*>(source.data_ptr()), reinterpret_cast<uint8_t*>(destination.data_ptr()) + offset,
        bytes, generations.data_ptr<int64_t>(), slot, owner, status.data_ptr<int>(), layer);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void compact_round_stage_cuda(const at::Tensor& ik, const at::Tensor& tables, at::Tensor& raw, at::Tensor& coords, int layer) {
    if (tables.size(0)) round_stage_kernel<<<int(tables.size(0)), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(ik.data_ptr()), tables.data_ptr<int64_t>(),
        reinterpret_cast<__half*>(raw.data_ptr()), coords.data_ptr<int64_t>(), layer);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void compact_round_publish_cuda(const at::Tensor& values, const at::Tensor& tables, int layer) {
    if (tables.size(0)) round_publish_kernel<<<int(tables.size(0)), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(values.data_ptr()), tables.data_ptr<int64_t>(), layer);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
