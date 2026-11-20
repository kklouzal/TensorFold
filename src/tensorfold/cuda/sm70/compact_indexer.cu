#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>
#include <cstdint>
#include <cuda_fp16.h>
#include "torch_reduce.cuh"

namespace {
__global__ void publish_kernel(const uint8_t* source, uint8_t* destination, int64_t bytes,
                               const int64_t* generations, int64_t slot,
                               int* status, int layer) {
    if (generations[0] != slot) {
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
    GENERATIONS, STATUS, EXPECTED_SLOT, POSITION, FIRST, ROWS, POOLED_ROWS,
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

// Stage new raw rows into the coordinate arena, protect the committed ring,
// extend the MTP proposal archive and (MTP) commit the window's ring tail.
// Pooling reads only the protected copy, so the ring may be rewritten here.
__global__ void round_stage_kernel(const __half* ik, const int64_t* tables, int layer) {
    const int64_t* t = tables + int64_t(blockIdx.x) * lane::COUNT;
    if (!t[lane::ENABLE]) return;
    const int64_t* generations = word<const int64_t>(t, lane::GENERATIONS);
    int* status = word<int>(t, lane::STATUS);
    const int64_t first = t[lane::FIRST], rows = t[lane::ROWS], start = *word<const int>(t, lane::POSITION);
    __half* ring = word<__half>(t, lane::RING_RAW);
    int64_t* ring_coords = word<int64_t>(t, lane::RING_COORDS);
    const int64_t* stage_coords = word<const int64_t>(t, lane::STAGE_COORDS);
    __half* stage = word<__half>(t, lane::STAGE_RAW);
    for (int64_t at = threadIdx.x; at < rows * DIM; at += blockDim.x) stage[at] = ik[first * DIM + at];
    if (generations[0] != t[lane::EXPECTED_SLOT]) {
        if (!threadIdx.x) latch(status, 32, layer);
        return;
    }
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
    if (!t[lane::RING_COMMIT]) return;
    // MTP commit of the window end: the ring keeps only the incomplete last
    // group's raw rows (older rows from the protected ring), zeros elsewhere.
    __syncthreads();
    const int64_t end = start + rows, count = end % RATIO;
    for (int at = threadIdx.x; at < RATIO * (DIM + 3); at += blockDim.x) {
        const bool coordinate = at >= RATIO * DIM;
        const int item = coordinate ? at - RATIO * DIM : at;
        const int slot = coordinate ? item / 3 : item / DIM, col = coordinate ? item % 3 : item % DIM;
        const int64_t position = end - count + slot;
        const bool live = slot < count;
        if (coordinate) {
            ring_coords[item] = !live ? 0 : position < start ? protect_coords[slot * 3 + col]
                                                              : stage_coords[(position - start) * 3 + col];
        } else {
            ring[item] = !live ? __float2half_rn(0.f) : position < start ? protect[slot * DIM + col]
                                                                          : ik[(first + position - start) * DIM + col];
        }
    }
}

struct SquareReader {
    const float* values;
    __device__ __forceinline__ float operator()(int index) const { return __fmul_rn(values[index], values[index]); }
};

// One complete pooled group per block (DIM threads): the eager owner's mean,
// 16-row RMSNorm, M-RoPE rotary prefix, row guard and generation-checked
// publication, with each Torch operator's FP32/FP16 rounding boundaries.
__global__ void round_pool_kernel(const __half* ik, const int64_t* tables, const float* weight, const float* freq,
                                  int rotary_half, float eps, float scale, int section1, int section2, int layer) {
    __shared__ float shared[DIM], normalized[DIM];
    const int64_t* t = tables + int64_t(blockIdx.x) * lane::COUNT;
    if (!t[lane::ENABLE]) return;
    const int64_t first = t[lane::FIRST], rows = t[lane::ROWS], start = *word<const int>(t, lane::POSITION);
    const int64_t group0 = start / RATIO, groups = (start + rows) / RATIO - group0, group = blockIdx.y;
    if (group >= groups) return;
    int* status = word<int>(t, lane::STATUS);
    const int64_t* generations = word<const int64_t>(t, lane::GENERATIONS);
    const __half* protect = word<const __half>(t, lane::PROTECTED);
    const int64_t* protect_coords = word<const int64_t>(t, lane::PROTECTED_COORDS);
    const int64_t* stage_coords = word<const int64_t>(t, lane::STAGE_COORDS);
    const int col = threadIdx.x;
    const int64_t base = (group0 + group) * RATIO;
    float total = 0.f;
    for (int member = 0; member < RATIO; ++member) {
        const int64_t position = base + member;
        const float value = __half2float(position < start ? protect[(position % RATIO) * DIM + col]
                                                            : ik[(first + position - start) * DIM + col]);
        total = member ? __fadd_rn(total, value) : value;
    }
    const float x = __half2float(__float2half_rn(__fdiv_rn(total, float(RATIO))));
    normalized[col] = x;
    __syncthreads();
    // Torch reduces the eager owner's padded [16, 1, DIM] squares (16 outputs).
    const float sum = tensorfold_sm70::torch_sum(SquareReader{normalized}, DIM, 16, shared);
    const float inverse = rsqrtf(__fadd_rn(__fmul_rn(sum, 1.f / float(DIM)), eps));
    const float y = __half2float(__float2half_rn(__fmul_rn(__fmul_rn(x, inverse), weight[col])));
    __syncthreads();
    normalized[col] = y;
    __syncthreads();
    float value = y;
    if (col < 2 * rotary_half) {
        const int pair = col % rotary_half;
        const int axis = pair % 3 == 1 && pair < 3 * section1 ? 1 : pair % 3 == 2 && pair < 3 * section2 ? 2 : 0;
        const int64_t coordinate = base < start ? protect_coords[(base % RATIO) * 3 + axis]
                                                : stage_coords[(base - start) * 3 + axis];
        const float angle = __fmul_rn(__ll2float_rn(coordinate), freq[pair]);
        const float sine = __fmul_rn(sinf(angle), scale), cosine = __fmul_rn(cosf(angle), scale);
        const float a = normalized[pair], b = normalized[pair + rotary_half];
        value = __half2float(__float2half_rn(col < rotary_half
            ? __fsub_rn(__fmul_rn(a, cosine), __fmul_rn(b, sine))
            : __fadd_rn(__fmul_rn(b, cosine), __fmul_rn(a, sine))));
    }
    const bool ok = isfinite(value) && fabsf(value) <= 1073741824.f;
    if (__syncthreads_or(!ok) && !col) latch(status, 16, layer);
    const bool current = generations[0] == t[lane::EXPECTED_SLOT];
    const bool fits = group0 + groups <= t[lane::POOLED_ROWS];
    if (!current || !fits) {
        if (!col) latch(status, 32, layer);
        return;
    }
    word<__half>(t, lane::POOLED)[(group0 + group) * DIM + col] = __float2half_rn(ok ? value : 0.f);
}
}

void compact_publish_cuda(const at::Tensor& source, at::Tensor& destination, int64_t offset,
                          const at::Tensor& generations, int64_t slot, at::Tensor& status, int layer) {
    const int64_t bytes = source.numel() * source.element_size();
    const int blocks = int(std::min<int64_t>((bytes - 1) / 256 + 1, 65535));
    publish_kernel<<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint8_t*>(source.data_ptr()), reinterpret_cast<uint8_t*>(destination.data_ptr()) + offset,
        bytes, generations.data_ptr<int64_t>(), slot, status.data_ptr<int>(), layer);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void compact_round_stage_cuda(const at::Tensor& ik, const at::Tensor& tables, int layer) {
    if (tables.size(0)) round_stage_kernel<<<int(tables.size(0)), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(ik.data_ptr()), tables.data_ptr<int64_t>(), layer);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void compact_round_pool_cuda(const at::Tensor& ik, const at::Tensor& tables, const at::Tensor& weight,
                             const at::Tensor& frequency, double eps, double scale, int section1, int section2, int layer) {
    if (tables.size(0)) round_pool_kernel<<<dim3(int(tables.size(0)), GROUPS), DIM, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(ik.data_ptr()), tables.data_ptr<int64_t>(), weight.data_ptr<float>(),
        frequency.data_ptr<float>(), int(frequency.numel()), float(eps), float(scale), section1, section2, layer);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
