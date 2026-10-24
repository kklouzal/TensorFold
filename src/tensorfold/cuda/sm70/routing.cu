// Stable 32-bit radix keys preserve FP32 order, including NaNs and signed zero.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cub/block/block_radix_sort.cuh>
#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_fp16.h>
#include <math_constants.h>

namespace {
__global__ void select512(const float* logits, int64_t stride, int64_t* indices, int count) {
    using Sort = cub::BlockRadixSort<uint32_t, 128, 4, uint32_t>;
    __shared__ typename Sort::TempStorage storage;
    uint32_t keys[4], ids[4];
    #pragma unroll
    for (int item = 0; item < 4; ++item) {
        const uint32_t index = threadIdx.x * 4 + item;
        const float value = logits[int64_t(blockIdx.x) * stride + index];
        const uint32_t bits = value == 0.f ? 0u : __float_as_uint(value);
        const uint32_t ordered = (bits & 0x80000000u) ? ~bits : bits ^ 0x80000000u;
        keys[item] = ordered;
        ids[item] = index;
    }
    // CUB BlockRadixSort is stable: the original blocked order is ascending ID.
    // Retaining IDs as values avoids sorting another 32 tie-break bits.
    Sort(storage).SortDescending(keys, ids);
    #pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int slot = threadIdx.x * 4 + item;
        if (slot < count)
            indices[int64_t(blockIdx.x) * count + slot] = ids[item];
    }
}

__global__ void route512(const float* logits, int64_t stride, int* picks,
                         float* weights, int count, int warp_width) {
    using Sort = cub::BlockRadixSort<uint32_t, 128, 4, uint32_t>;
    __shared__ typename Sort::TempStorage storage;
    __shared__ float selected[32];
    uint32_t keys[4], ids[4];
    const int row = blockIdx.x, slots = count + 1;
    #pragma unroll
    for (int item = 0; item < 4; ++item) {
        const uint32_t index = threadIdx.x * 4 + item;
        const float value = logits[int64_t(row) * stride + index];
        const uint32_t bits = value == 0.f ? 0u : __float_as_uint(value);
        keys[item] = (bits & 0x80000000u) ? ~bits : bits ^ 0x80000000u;
        ids[item] = index;
    }
    Sort(storage).SortDescending(keys, ids);
    #pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int slot = threadIdx.x * 4 + item;
        if (slot < count) {
            picks[int64_t(row) * slots + slot] = ids[item];
            selected[slot] = logits[int64_t(row) * stride + ids[item]];
        }
    }
    __syncthreads();
    if (threadIdx.x < warp_width) {
        const int lane = threadIdx.x;
        const unsigned mask = warp_width == 32 ? 0xffffffffu : (1u << warp_width) - 1u;
        const float value = lane < count ? selected[lane] : -CUDART_INF_F;
        float maximum = value;
        for (int offset = warp_width / 2; offset; offset /= 2) {
            const float other = __shfl_xor_sync(mask, maximum, offset, warp_width);
            maximum = maximum < other ? other : maximum;
        }
        const float numerator = expf(value - maximum);
        float sum = numerator;
        for (int offset = warp_width / 2; offset; offset /= 2)
            sum += __shfl_xor_sync(mask, sum, offset, warp_width);
        if (lane < count)
            weights[int64_t(row) * slots + lane] = __half2float(__float2half_rn(numerator / sum));
    }
    if (threadIdx.x == 0) {
        const float shared = __half2float(__float2half_rn(logits[int64_t(row) * stride + 512]));
        const float gate = 1.0f / (1.0f + expf(-shared));
        picks[int64_t(row) * slots + count] = 512;
        weights[int64_t(row) * slots + count] = __half2float(__float2half_rn(gate));
    }
}
}  // namespace

void routing_select_cuda(const at::Tensor& logits, at::Tensor& indices) {
    if (!logits.size(0)) return;
    select512<<<logits.size(0), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        logits.data_ptr<float>(), logits.stride(0), indices.data_ptr<int64_t>(), indices.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void routing_route_cuda(const at::Tensor& logits, at::Tensor& picks, at::Tensor& weights) {
    if (!logits.size(0)) return;
    const int count = picks.size(1) - 1;
    int width = 1;
    while (width < count) width *= 2;
    route512<<<logits.size(0), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        logits.data_ptr<float>(), logits.stride(0), picks.data_ptr<int>(), weights.data_ptr<float>(), count, width);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {
// Match the qualified pinned cuBLAS serial GEMV: eight outputs/CTA, sixteen
// lanes/output, five K512 tiles, then a left-fold of the sixteen lane totals.
__global__ void serial_logits(const __half* x, const float* weight, float* output,
                              int64_t input_stride) {
    __shared__ float input[512];
    __shared__ float partial[128];
    const int thread = threadIdx.x, lane = thread & 15, group = thread >> 4;
    const int expert = blockIdx.x * 8 + group, row = blockIdx.y;
    float total = 0.0f;
    #pragma unroll 1
    for (int tile = 0; tile < 2560; tile += 512) {
        #pragma unroll
        for (int offset = thread; offset < 512; offset += 128) {
            const float value = __half2float(x[int64_t(row) * input_stride + tile + offset]);
            float scaled;
            asm volatile("mul.rn.f32 %0, %1, 0f3f800000;" : "=f"(scaled) : "f"(value));
            input[offset] = scaled;
        }
        __syncthreads();
        float value = 0.0f;
        if (expert < 513) {
            #pragma unroll
            for (int step = 0; step < 32; ++step) {
                const int column = tile + lane + step * 16;
                value = __fmaf_rn(weight[expert * 2560 + column], input[lane + step * 16], value);
            }
        }
        total = __fadd_rn(total, value);
        __syncthreads();
    }
    partial[thread] = total;
    __syncthreads();
    if (lane == 0 && expert < 513) {
        float value = partial[group * 16];
        #pragma unroll
        for (int part = 1; part < 16; ++part)
            value = __fadd_rn(value, partial[group * 16 + part]);
        output[row * 513 + expert] = value;
    }
}
}  // namespace

void routing_logits_cuda(const at::Tensor& x, const at::Tensor& weight, at::Tensor& output) {
    if (!x.size(0)) return;
    serial_logits<<<dim3(65, x.size(0)), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), weight.data_ptr<float>(),
        output.data_ptr<float>(), x.stride(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
