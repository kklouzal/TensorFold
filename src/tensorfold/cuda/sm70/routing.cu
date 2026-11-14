// Match the pinned Torch512 medium radix path, not its small-sort NaN
// comparator. Signed zeros tie; other FP32 payloads retain radix ordering.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cub/block/block_radix_sort.cuh>
#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_fp16.h>
#include <math_constants.h>

namespace {
__device__ __forceinline__ uint32_t selection_key(float value) {
    uint32_t bits = __float_as_uint(value);
    const uint32_t magnitude = bits & 0x7fffffffu;
    if (magnitude == 0) bits = 0;
    return (bits & 0x80000000u) ? ~bits : bits ^ 0x80000000u;
}

__global__ void select512(const float* logits, int64_t stride, int64_t* indices, int count) {
    using Sort = cub::BlockRadixSort<uint32_t, 128, 4, uint32_t>;
    __shared__ typename Sort::TempStorage storage;
    uint32_t keys[4], ids[4];
    #pragma unroll
    for (int item = 0; item < 4; ++item) {
        const uint32_t index = threadIdx.x * 4 + item;
        const float value = logits[int64_t(blockIdx.x) * stride + index];
        keys[item] = selection_key(value);
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
        keys[item] = selection_key(value);
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
        // Pinned PersistentSoftmax initializes its accumulator before adding
        // the one exp element/lane; retain that operation for exceptional bits.
        float sum = 0.0f;
        sum += numerator;
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

void routing_prepare_route_cuda() {
    cudaFuncAttributes attributes;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, route512));
}

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
// Each CTA serves NR (<= 8) consecutive rows with one register copy of its weight
// tile, so the FP32 weights are read once per row group instead of once per row;
// the rows' independent FMA chains interleave. Every row's operation sequence is
// the serial one and does not depend on the call's row count.
template <int NR>
__global__ void __launch_bounds__(128) serial_logits(const __half* x, const float* weight, float* output,
                                                     int64_t input_stride, int first_row) {
    __shared__ float input[NR][512];
    __shared__ float partial[NR][128];
    const int thread = threadIdx.x, lane = thread & 15, group = thread >> 4;
    const int expert = blockIdx.x * 8 + group;
    const bool active = expert < 513;
    const int first = first_row + blockIdx.y * NR;
    const float* row_weight = weight + int64_t(active ? expert : 0) * 2560 + lane;
    float total[NR];
    #pragma unroll
    for (int r = 0; r < NR; ++r) total[r] = 0.0f;
    #pragma unroll 1
    for (int tile = 0; tile < 2560; tile += 512) {
        float w[32];
        #pragma unroll
        for (int step = 0; step < 32; ++step) w[step] = active ? row_weight[tile + step * 16] : 0.0f;
        #pragma unroll
        for (int r = 0; r < NR; ++r) {
            #pragma unroll
            for (int offset = thread; offset < 512; offset += 128) {
                const float value = __half2float(x[int64_t(first + r) * input_stride + tile + offset]);
                float scaled;
                asm volatile("mul.rn.f32 %0, %1, 0f3f800000;" : "=f"(scaled) : "f"(value));
                input[r][offset] = scaled;
            }
        }
        __syncthreads();
        float value[NR];
        #pragma unroll
        for (int r = 0; r < NR; ++r) value[r] = 0.0f;
        #pragma unroll
        for (int step = 0; step < 32; ++step) {
            #pragma unroll
            for (int r = 0; r < NR; ++r) value[r] = __fmaf_rn(w[step], input[r][lane + step * 16], value[r]);
        }
        #pragma unroll
        for (int r = 0; r < NR; ++r) total[r] = __fadd_rn(total[r], value[r]);
        __syncthreads();
    }
    #pragma unroll
    for (int r = 0; r < NR; ++r) partial[r][thread] = total[r];
    __syncthreads();
    if (lane == 0 && active) {
        #pragma unroll
        for (int r = 0; r < NR; ++r) {
            float value = partial[r][group * 16];
            #pragma unroll
            for (int part = 1; part < 16; ++part)
                value = __fadd_rn(value, partial[r][group * 16 + part]);
            output[int64_t(first + r) * 513 + expert] = value;
        }
    }
}

template <int NR>
void launch_logits(const __half* x, const float* weight, float* output, int64_t stride, int first, int groups,
                   cudaStream_t stream) {
    serial_logits<NR><<<dim3(65, groups), 128, 0, stream>>>(x, weight, output, stride, first);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_rows(const __half* x, const float* weight, float* output, int64_t stride, int first, int rows,
                 int groups, cudaStream_t stream) {
    switch (rows) {
        case 1: return launch_logits<1>(x, weight, output, stride, first, groups, stream);
        case 2: return launch_logits<2>(x, weight, output, stride, first, groups, stream);
        case 3: return launch_logits<3>(x, weight, output, stride, first, groups, stream);
        case 4: return launch_logits<4>(x, weight, output, stride, first, groups, stream);
        case 5: return launch_logits<5>(x, weight, output, stride, first, groups, stream);
        case 6: return launch_logits<6>(x, weight, output, stride, first, groups, stream);
        case 7: return launch_logits<7>(x, weight, output, stride, first, groups, stream);
        default: return launch_logits<8>(x, weight, output, stride, first, groups, stream);
    }
}
}  // namespace

void routing_logits_cuda(const at::Tensor& x, const at::Tensor& weight, at::Tensor& output) {
    const int rows = int(x.size(0));
    if (!rows) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    const auto* input = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
    // Up to eight rows: one launch; more: groups of eight, then one launch for the remainder.
    const int full = rows > 8 ? rows / 8 : 0, rest = rows - 8 * full;
    if (full) launch_rows(input, weight.data_ptr<float>(), output.data_ptr<float>(), x.stride(0), 0, 8, full, stream);
    if (rest) launch_rows(input, weight.data_ptr<float>(), output.data_ptr<float>(), x.stride(0), 8 * full, rest, 1,
                          stream);
}

