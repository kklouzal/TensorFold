#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/MemoryOverlap.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cmath>
#include <climits>
#include <array>
#include <mma.h>
#include <type_traits>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include "nvfp4_layout.cuh"
#include "nvfp4_decode.cuh"

namespace {
constexpr int M = TF_SM70_NVFP4_ROWS, N = TF_SM70_NVFP4_OUTPUTS, K = TF_SM70_NVFP4_INPUTS;
struct OperandStorage { half a[M * K]; half b[N * K]; };
union __align__(32) SharedStorage { OperandStorage operands; float result[M * N]; };
static_assert(std::is_trivially_default_constructible<OperandStorage>::value,
              "shared operand lifetime requires trivial CUDA half construction");
static_assert(std::is_trivially_copy_assignable<half>::value,
              "union activation requires trivial half copy assignment");
static_assert(sizeof(SharedStorage) == (M + N) * K * sizeof(half), "shared tile extent changed");

__device__ void output_value(float* output, int64_t at, float value) { output[at] = value; }
__device__ void output_value(half* output, int64_t at, float value) { output[at] = __float2half_rn(value); }

__device__ void require_internal(bool valid) {
    // Picks/descriptors are private products of the admitted router/importer.
    // Their corruption is a fatal cohort invariant, never a zero-output repair.
    if (!valid) asm volatile("trap;");
}

__global__ void expert_counts(const int* picks, int* counts, int pairs, int experts) {
    using Reduction = cub::BlockReduce<int, 256>;
    __shared__ typename Reduction::TempStorage temporary;
    int count = 0;
    for (int64_t index = threadIdx.x; index < pairs; index += 256) {
        const int value = picks[index];
        require_internal(value >= 0 && value < experts);
        count += value == int(blockIdx.x);
    }
    const int total = Reduction(temporary).Sum(count);
    if (threadIdx.x == 0) counts[blockIdx.x] = total;
}

__global__ void expert_tiles(const int* counts, int* starts, int* tile_expert, int* tile_begin,
                              int* work_count, int experts, int pairs, int capacity) {
    if (threadIdx.x) return;
    int offset = 0, tiles = 0;
    for (int expert = 0; expert < experts; ++expert) {
        const int count = counts[expert];
        require_internal(count >= 0 && count <= pairs - offset);
        starts[expert] = offset;
        offset += count;
        for (int64_t begin = 0; begin < count; begin += M) {
            require_internal(tiles < capacity);
            tile_expert[tiles] = expert;
            tile_begin[tiles++] = int(begin);
        }
    }
    require_internal(offset == pairs);
    starts[experts] = offset;
    *work_count = tiles;
}

__global__ void expert_members(const int* picks, const int* starts, int* members, int pairs) {
    using Scan = cub::BlockScan<int, 256>;
    __shared__ typename Scan::TempStorage temporary;
    __shared__ int base;
    if (threadIdx.x == 0) base = starts[blockIdx.x];
    __syncthreads();
    for (int64_t first = 0; first < pairs; first += 256) {
        const int64_t index = first + threadIdx.x;
        const int live = index < pairs && picks[index] == int(blockIdx.x);
        int prefix, total;
        Scan(temporary).ExclusiveSum(live, prefix, total);
        if (live) members[base + prefix] = int(index);
        __syncthreads();
        if (threadIdx.x == 0) base += total;
        __syncthreads(); // Retire CUB temporary and publish the next stable span.
    }
}

__device__ bool valid_matrix(const int64_t* descriptor, int64_t n, int64_t k) {
    if (n <= 0 || k <= 0 || k > INT64_MAX - K || descriptor[TF_NV_ROWS] != n
            || descriptor[TF_NV_COLUMNS] != k || descriptor[TF_NV_DATA] == 0) return false;
    if (descriptor[TF_NV_KIND] == 0) {
        if (k > INT64_MAX / 2 || n > INT64_MAX / (k * 2)) return false;
        return descriptor[TF_NV_DATA_BYTES] == n * (k * 2) && descriptor[TF_NV_SCALE_ELEMENTS] == 0
            && descriptor[TF_NV_SCALES] == 0 && descriptor[TF_NV_GLOBAL] == 0;
    }
    if (descriptor[TF_NV_KIND] != 1 || k % 16 || n > INT64_MAX / (k / 2)) return false;
    return descriptor[TF_NV_DATA_BYTES] == n * (k / 2) && descriptor[TF_NV_SCALE_ELEMENTS] == n * (k / 16)
        && descriptor[TF_NV_SCALES] != 0 && descriptor[TF_NV_GLOBAL] != 0;
}

__device__ half matrix_value(const int64_t* descriptor, int64_t row, int64_t column) {
    const int64_t k = descriptor[TF_NV_COLUMNS];
    if (descriptor[TF_NV_KIND] == 0)
        return reinterpret_cast<const half*>(descriptor[TF_NV_DATA])[row * k + column];
    const auto* codes = reinterpret_cast<const uint8_t*>(descriptor[TF_NV_DATA]);
    const auto* scales = reinterpret_cast<const uint8_t*>(descriptor[TF_NV_SCALES]);
    const auto* global = reinterpret_cast<const float*>(descriptor[TF_NV_GLOBAL]);
    unsigned code = (codes[row * (k / 2) + column / 2] >> (4 * (column & 1))) & 15;
    return __float2half_rn(tf_sm70_nvfp4::value(code, scales[row * (k / 16) + column / 16], global[0]));
}

using Accumulator = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>;

template<bool FromActivation>
__device__ void grouped_projection(const half* input, int64_t input_stride, const int* members,
    int group_start, int begin, int live, int slots, const int64_t* descriptor, int64_t first_column,
    SharedStorage* storage, Accumulator& total) {
    using namespace nvcuda;
    const int lane = threadIdx.x, warp = lane / 32, wm = warp / 4, wn = warp % 4;
    const int64_t n = descriptor[TF_NV_ROWS], k = descriptor[TF_NV_COLUMNS];
    if (lane == 0) storage[0].operands.a[0] = __float2half_rn(0.f);
    __syncthreads();
    wmma::fill_fragment(total, 0.f);
    for (int64_t start = 0; start < k; start += K) {
        for (int i = lane; i < M * K; i += 256) {
            const int r = i / K, c = i % K;
            half value = __float2half_rn(0.f);
            if (r < live && start + c < k) {
                const int pair = members[group_start + begin + r];
                const int row = FromActivation ? pair : pair / slots;
                value = input[int64_t(row) * input_stride + start + c];
            }
            storage[0].operands.a[i] = value;
        }
        for (int i = lane; i < N * K; i += 256) {
            const int r = i / K, c = i % K;
            storage[0].operands.b[i] = first_column + r < n && start + c < k
                ? matrix_value(descriptor, first_column + r, start + c) : __float2half_rn(0.f);
        }
        __syncthreads();
        Accumulator partial;
        wmma::fill_fragment(partial, 0.f);
        #pragma unroll
        for (int part = 0; part < K; part += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
            wmma::load_matrix_sync(a, storage[0].operands.a + wm * 16 * K + part, K);
            wmma::load_matrix_sync(b, storage[0].operands.b + wn * 16 * K + part, K);
            wmma::mma_sync(partial, a, b, partial);
        }
        #pragma unroll
        for (int i = 0; i < total.num_elements; ++i) total.x[i] = total.x[i] + partial.x[i];
        __syncthreads();
    }
}

__global__ void grouped_gate_up(const half* x, const int64_t* descriptors, const int* counts,
    const int* starts, const int* members, const int* tile_expert, const int* tile_begin,
    const int* work_count, half* activation, int64_t input_stride, int64_t activation_stride,
    int slots, int experts, int64_t dims, int64_t maximum_width, int64_t column_tiles) {
    using namespace nvcuda;
    const int work = int(int64_t(blockIdx.x) / column_tiles);
    if (work >= *work_count) return; // Uniform block predicate before any barrier/pointer read.
    const int expert = tile_expert[work], begin = tile_begin[work];
    require_internal(expert >= 0 && expert < experts);
    require_internal(begin >= 0 && begin < counts[expert]);
    const int live = min(M, counts[expert] - begin);
    const auto* gate = descriptors + int64_t(expert) * 3 * TF_SM70_NVFP4_EXPERT_WORDS;
    const auto* up = gate + TF_SM70_NVFP4_EXPERT_WORDS;
    const int64_t width = gate[TF_NV_ROWS];
    require_internal(width > 0 && width <= maximum_width && live > 0
        && valid_matrix(gate, width, dims) && valid_matrix(up, width, dims));
    const int64_t column = (int64_t(blockIdx.x) % column_tiles) * N;
    if (column >= width) return;
    extern __shared__ SharedStorage storage[];
    Accumulator gated, upward;
    grouped_projection<false>(x, input_stride, members, starts[expert], begin, live, slots, gate, column, storage, gated);
    grouped_projection<false>(x, input_stride, members, starts[expert], begin, live, slots, up, column, storage, upward);
    #pragma unroll
    for (int i = 0; i < gated.num_elements; ++i) {
        const float value = gated.x[i];
        gated.x[i] = (value / (1.f + expf(-value))) * upward.x[i];
    }
    if (threadIdx.x == 0) storage[0].result[0] = 0.f;
    __syncthreads();
    wmma::store_matrix_sync(storage[0].result + int(threadIdx.x / 128) * 16 * N
        + int(threadIdx.x / 32 % 4) * 16, gated, N, wmma::mem_row_major);
    __syncthreads();
    for (int i = threadIdx.x; i < M * N; i += 256) {
        const int row = i / N, c = i % N;
        if (row < live && column + c < width) {
            const int pair = members[starts[expert] + begin + row];
            activation[int64_t(pair) * activation_stride + column + c] = __float2half_rn(storage[0].result[i]);
        }
    }
}

template<class Output>
__global__ void grouped_down(const half* activation, const int64_t* descriptors, const int* counts,
    const int* starts, const int* members, const int* tile_expert, const int* tile_begin,
    const int* work_count, Output* output, int64_t activation_stride, int slots, int experts, int64_t dims,
    int64_t maximum_width, int64_t column_tiles) {
    using namespace nvcuda;
    const int work = int(int64_t(blockIdx.x) / column_tiles);
    if (work >= *work_count) return;
    const int expert = tile_expert[work], begin = tile_begin[work];
    require_internal(expert >= 0 && expert < experts);
    const auto* gate = descriptors + int64_t(expert) * 3 * TF_SM70_NVFP4_EXPERT_WORDS;
    const auto* down = gate + 2 * TF_SM70_NVFP4_EXPERT_WORDS;
    const int64_t width = gate[TF_NV_ROWS];
    require_internal(begin >= 0 && begin < counts[expert]);
    const int live = min(M, counts[expert] - begin);
    require_internal(width > 0 && width <= maximum_width && live > 0 && valid_matrix(down, dims, width));
    extern __shared__ SharedStorage storage[];
    Accumulator result;
    const int64_t column = (int64_t(blockIdx.x) % column_tiles) * N;
    grouped_projection<true>(activation, activation_stride, members, starts[expert], begin, live, slots,
                              down, column, storage, result);
    if (threadIdx.x == 0) storage[0].result[0] = 0.f;
    __syncthreads();
    wmma::store_matrix_sync(storage[0].result + int(threadIdx.x / 128) * 16 * N
        + int(threadIdx.x / 32 % 4) * 16, result, N, wmma::mem_row_major);
    __syncthreads();
    for (int i = threadIdx.x; i < M * N; i += 256) {
        const int row = i / N, c = i % N;
        if (row < live && column + c < dims) {
            const int pair = members[starts[expert] + begin + row];
            output_value(output, int64_t(pair) * dims + column + c, storage[0].result[i]);
        }
    }
}

template<class Output>
__global__ void fused_matmul(const half* x, const uint8_t* codes, const uint8_t* blocks,
                             const float* global, Output* output, int64_t rows, int64_t n, int64_t k,
                             int64_t input_stride, int64_t output_stride) {
    using namespace nvcuda;
    extern __shared__ SharedStorage storage[];
    const int lane = threadIdx.x, warp = lane / 32, wm = warp / 4, wn = warp % 4;
    const int64_t first_row = int64_t(blockIdx.y) * M, first_column = int64_t(blockIdx.x) * N;
    // Assignment through the named union member starts its trivial operand
    // lifetime. The barrier publishes that lifetime before any borrowed view.
    if (lane == 0) storage[0].operands.a[0] = __float2half_rn(0.f);
    __syncthreads();
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> total;
    wmma::fill_fragment(total, 0.f);
    for (int64_t start = 0; start < k; start += K) {
        for (int i = lane; i < M * K; i += 256) {
            const int r = i / K, c = i % K;
            storage[0].operands.a[i] = first_row + r < rows && start + c < k
                ? x[(first_row + r) * input_stride + start + c] : __float2half_rn(0.f);
        }
        for (int i = lane; i < N * K; i += 256) {
            const int r = i / K, c = i % K;
            half decoded = __float2half_rn(0.f);
            if (first_column + r < n && start + c < k) {
                const int64_t at = (first_column + r) * (k / 2) + (start + c) / 2;
                const unsigned code = (codes[at] >> (4 * ((start + c) & 1))) & 15;
                decoded = __float2half_rn(tf_sm70_nvfp4::value(code,
                    blocks[(first_column + r) * (k / 16) + (start + c) / 16], global[0]));
            }
            storage[0].operands.b[i] = decoded;
        }
        __syncthreads();
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> partial;
        wmma::fill_fragment(partial, 0.f);
        #pragma unroll
        for (int part = 0; part < K; part += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
            wmma::load_matrix_sync(a, storage[0].operands.a + wm * 16 * K + part, K);
            wmma::load_matrix_sync(b, storage[0].operands.b + wn * 16 * K + part, K);
            wmma::mma_sync(partial, a, b, partial);
        }
        #pragma unroll
        for (int i = 0; i < total.num_elements; ++i) total.x[i] = total.x[i] + partial.x[i];
        __syncthreads(); // Every operand reader retires before tile reuse.
    }
    // A named trivial-array assignment begins the other union member's
    // lifetime after the final operand-reader barrier. No old operand view is
    // used afterward; all warps publish complete disjoint result tiles.
    if (lane == 0) storage[0].result[0] = 0.f;
    __syncthreads();
    wmma::store_matrix_sync(storage[0].result + wm * 16 * N + wn * 16, total, N, wmma::mem_row_major);
    __syncthreads();
    for (int i = lane; i < M * N; i += 256) {
        const int r = i / N, c = i % N;
        if (first_row + r < rows && first_column + c < n)
            output_value(output, (first_row + r) * output_stride + first_column + c, storage[0].result[i]);
    }
}

__global__ void decode_tile(const uint8_t* codes, const uint8_t* blocks, const float* global,
                            half* out, int64_t k, int64_t row, int64_t column,
                            int64_t rows, int64_t columns, int64_t on, int64_t ok) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= on * ok) return;
    const int64_t r = i / ok, c = i % ok;
    float value = 0.f;
    if (r < rows && c < columns) {
        const int64_t ar = row + r, ac = column + c;
        unsigned pair = codes[ar * (k / 2) + ac / 2];
        unsigned code = (pair >> (4 * (ac & 1))) & 15;
        // The manifest fixes this FP32 multiplication order before FP16 RNE.
        value = tf_sm70_nvfp4::value(code, blocks[ar * (k / 16) + ac / 16], global[0]);
    }
    out[i] = __float2half_rn(value);
}
}

void tf_sm70_nvfp4_decode(const at::Tensor& codes, const at::Tensor& scales,
                         const at::Tensor& global, at::Tensor out, int64_t row,
                         int64_t column, int64_t rows, int64_t columns) {
    TORCH_CHECK(codes.is_cuda() && scales.device() == codes.device() && global.device() == codes.device()
                && out.device() == codes.device(), "SM70 NVFP4 tile tensors must share a CUDA device");
    c10::cuda::CUDAGuard guard(codes.device());
    TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major == 7
                && at::cuda::getCurrentDeviceProperties()->minor == 0, "NVFP4 software tile requires SM70");
    TORCH_CHECK(codes.scalar_type() == at::kByte && scales.scalar_type() == at::kByte
                && global.scalar_type() == at::kFloat && out.scalar_type() == at::kHalf,
                "NVFP4 tile requires original U8 codes/E4M3 bytes, FP32 global scale and FP16 output");
    TORCH_CHECK(codes.dim() == 2 && scales.dim() == 2 && global.numel() == 1 && out.dim() == 2
                && codes.is_contiguous() && scales.is_contiguous() && global.is_contiguous()
                && out.is_contiguous(), "invalid NVFP4 tile storage layout");
    TORCH_CHECK(codes.size(0) > 0 && codes.size(1) > 0 && codes.size(1) <= INT64_MAX / 2,
                "NVFP4 physical dimensions overflow");
    const int64_t n = codes.size(0), k = codes.size(1) * 2;
    TORCH_CHECK(n > 0 && k > 0 && k % 16 == 0 && scales.size(0) == n && scales.size(1) == k / 16,
                "NVFP4 codes/scales require exactly one E4M3 scale per16 input values");
    TORCH_CHECK(row >= 0 && column >= 0 && rows > 0 && columns > 0 && row <= n - rows
                && column <= k - columns && rows <= out.size(0) && columns <= out.size(1)
                && out.size(0) <= TF_SM70_NVFP4_OUTPUTS && out.size(1) <= TF_SM70_NVFP4_INPUTS,
                "NVFP4 decode exceeds its bounded tile or source");
    at::assert_no_overlap(out, codes);
    at::assert_no_overlap(out, scales);
    at::assert_no_overlap(out, global);
    const int64_t count = out.numel();
    decode_tile<<<(count + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(), global.data_ptr<float>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()), k, row, column, rows, columns,
        out.size(0), out.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void tf_sm70_nvfp4_matmul(const at::Tensor& x, const at::Tensor& codes, const at::Tensor& scales,
                         const at::Tensor& global, at::Tensor out) {
    TORCH_CHECK(codes.is_cuda() && x.device() == codes.device() && scales.device() == codes.device()
                && global.device() == codes.device() && out.device() == codes.device(),
                "SM70 NVFP4 fused tensors must share a CUDA device");
    c10::cuda::CUDAGuard guard(codes.device());
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(properties->major == 7 && properties->minor == 0, "NVFP4 fused WMMA requires SM70");
    TORCH_CHECK(codes.scalar_type() == at::kByte && scales.scalar_type() == at::kByte
                && global.scalar_type() == at::kFloat && x.scalar_type() == at::kHalf
                && (out.scalar_type() == at::kHalf || out.scalar_type() == at::kFloat), "invalid fused NVFP4 dtypes");
    TORCH_CHECK(codes.dim() == 2 && scales.dim() == 2 && global.numel() == 1 && x.dim() == 2 && out.dim() == 2
                && codes.is_contiguous() && scales.is_contiguous() && global.is_contiguous()
                && x.stride(1) == 1 && out.stride(1) == 1, "invalid fused NVFP4 physical layout");
    TORCH_CHECK(codes.size(0) > 0 && codes.size(1) > 0 && codes.size(1) <= INT64_MAX / 2,
                "fused NVFP4 dimensions overflow");
    const int64_t n = codes.size(0), k = codes.size(1) * 2, rows = x.size(0);
    TORCH_CHECK(k % 16 == 0 && k <= INT64_MAX - K && scales.sizes() == at::IntArrayRef({n, k / 16}) && x.size(1) == k
                && out.sizes() == at::IntArrayRef({rows, n}) && x.stride(0) >= k && out.stride(0) >= n,
                "fused NVFP4 dimensions/strides differ from the admitted projection");
    TORCH_CHECK(n <= int64_t(properties->maxGridSize[0]) * N && rows <= int64_t(properties->maxGridSize[1]) * M,
                "fused NVFP4 grid exceeds CUDA dimensions");
    for (const at::Tensor* input : {&x, &codes, &scales, &global}) at::assert_no_overlap(out, *input);
    if (!rows) return;
    const dim3 grid((n + N - 1) / N, (rows + M - 1) / M);
    const auto stream = at::cuda::getCurrentCUDAStream();
    if (out.scalar_type() == at::kFloat)
        fused_matmul<<<grid, 256, sizeof(SharedStorage), stream>>>(reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(), global.data_ptr<float>(), out.data_ptr<float>(),
            rows, n, k, x.stride(0), out.stride(0));
    else
        fused_matmul<<<grid, 256, sizeof(SharedStorage), stream>>>(reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(), global.data_ptr<float>(),
            reinterpret_cast<half*>(out.data_ptr<at::Half>()), rows, n, k, x.stride(0), out.stride(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void tf_sm70_nvfp4_experts(const at::Tensor& x, const at::Tensor& picks, const at::Tensor& descriptors,
    at::Tensor counts, at::Tensor starts, at::Tensor members, at::Tensor tile_expert, at::Tensor tile_begin,
    at::Tensor work_count, at::Tensor activation, at::Tensor output, int64_t width) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.dim() == 2 && x.size(1) > 0
                && x.stride(1) == 1 && x.stride(0) >= x.size(1), "device expert input requires FP16 row vectors");
    c10::cuda::CUDAGuard guard(x.device());
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(properties->major == 7 && properties->minor == 0, "device expert WMMA requires SM70");
    auto check = [&](const at::Tensor& tensor, at::ScalarType kind) {
        TORCH_CHECK(tensor.device() == x.device() && tensor.scalar_type() == kind && tensor.is_contiguous(),
                    "device expert operands require declared contiguous same-device tensors");
    };
    check(picks, at::kInt); check(descriptors, at::kLong);
    for (const at::Tensor* tensor : {&counts, &starts, &members, &tile_expert, &tile_begin, &work_count})
        check(*tensor, at::kInt);
    check(activation, at::kHalf);
    TORCH_CHECK(output.device() == x.device() && output.is_contiguous()
                && (output.scalar_type() == at::kHalf || output.scalar_type() == at::kFloat), "device expert output dtype/layout");
    const int64_t rows = x.size(0), dims = x.size(1);
    TORCH_CHECK(picks.dim() == 2 && picks.size(0) == rows && picks.size(1) > 0,
                "device expert picks require one fixed slot row per input");
    const int64_t slots = picks.size(1);
    TORCH_CHECK(rows <= INT_MAX / slots && descriptors.dim() == 3 && descriptors.size(0) > 0
                && descriptors.size(0) <= INT_MAX && descriptors.size(1) == 3
                && descriptors.size(2) == TF_SM70_NVFP4_EXPERT_WORDS, "device expert work/descriptor ABI overflow");
    const int64_t experts = descriptors.size(0), pairs = rows * slots, tiles = experts + (pairs + M - 1) / M;
    TORCH_CHECK(tiles <= INT_MAX && width > 0 && width <= INT64_MAX - K && dims <= INT64_MAX - K,
                "device expert dimensions exceed checked work arithmetic");
    TORCH_CHECK(counts.dim() == 1 && counts.numel() >= experts && starts.dim() == 1 && starts.numel() >= experts + 1
                && members.dim() == 1 && members.numel() >= pairs && tile_expert.dim() == 1
                && tile_expert.numel() >= tiles && tile_begin.sizes() == tile_expert.sizes()
                && work_count.dim() == 1 && work_count.numel() == 1
                && activation.dim() == 2 && activation.size(0) >= pairs && activation.size(1) >= width
                && output.sizes() == at::IntArrayRef({rows, slots, dims}), "device expert workspace exceeds admitted bounds");
    std::array<const at::Tensor*, 11> operands{&x, &picks, &descriptors, &counts, &starts, &members,
        &tile_expert, &tile_begin, &work_count, &activation, &output};
    for (size_t i = 0; i < operands.size(); ++i)
        for (size_t j = i + 1; j < operands.size(); ++j) at::assert_no_overlap(*operands[i], *operands[j]);
    if (!rows) return;
    const int64_t gate_columns = (width + N - 1) / N, down_columns = (dims + N - 1) / N;
    TORCH_CHECK(experts <= properties->maxGridSize[0] && tiles <= properties->maxGridSize[0] / gate_columns
                && tiles <= properties->maxGridSize[0] / down_columns, "device expert grid exceeds CUDA limits");
    auto stream = at::cuda::getCurrentCUDAStream();
    expert_counts<<<experts, 256, 0, stream>>>(picks.data_ptr<int>(), counts.data_ptr<int>(), int(pairs), int(experts));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    expert_tiles<<<1, 32, 0, stream>>>(counts.data_ptr<int>(), starts.data_ptr<int>(), tile_expert.data_ptr<int>(),
        tile_begin.data_ptr<int>(), work_count.data_ptr<int>(), int(experts), int(pairs), int(tiles));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    expert_members<<<experts, 256, 0, stream>>>(picks.data_ptr<int>(), starts.data_ptr<int>(), members.data_ptr<int>(), int(pairs));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    grouped_gate_up<<<tiles * gate_columns, 256, sizeof(SharedStorage), stream>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()), descriptors.data_ptr<int64_t>(), counts.data_ptr<int>(),
        starts.data_ptr<int>(), members.data_ptr<int>(), tile_expert.data_ptr<int>(), tile_begin.data_ptr<int>(),
        work_count.data_ptr<int>(), reinterpret_cast<half*>(activation.data_ptr<at::Half>()), x.stride(0),
        activation.size(1), int(slots), int(experts), dims, width, gate_columns);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (output.scalar_type() == at::kFloat)
        grouped_down<<<tiles * down_columns, 256, sizeof(SharedStorage), stream>>>(
            reinterpret_cast<const half*>(activation.data_ptr<at::Half>()), descriptors.data_ptr<int64_t>(), counts.data_ptr<int>(),
            starts.data_ptr<int>(), members.data_ptr<int>(), tile_expert.data_ptr<int>(), tile_begin.data_ptr<int>(),
            work_count.data_ptr<int>(), output.data_ptr<float>(), activation.size(1), int(slots), int(experts), dims, width, down_columns);
    else
        grouped_down<<<tiles * down_columns, 256, sizeof(SharedStorage), stream>>>(
            reinterpret_cast<const half*>(activation.data_ptr<at::Half>()), descriptors.data_ptr<int64_t>(), counts.data_ptr<int>(),
            starts.data_ptr<int>(), members.data_ptr<int>(), tile_expert.data_ptr<int>(), tile_begin.data_ptr<int>(),
            work_count.data_ptr<int>(), reinterpret_cast<half*>(output.data_ptr<at::Half>()), activation.size(1),
            int(slots), int(experts), dims, width, down_columns);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
