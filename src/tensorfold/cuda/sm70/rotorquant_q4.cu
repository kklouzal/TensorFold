// Current-stream, allocation-free packed Q4 codec for TensorFold's SM70 policy.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <optional>
#include <algorithm>
#include <type_traits>
#include "rotorquant_q4.cuh"

namespace {
using T = at::Tensor;
using O = std::optional<T>;

__device__ void latch(int* status, int errors, int layer) {
    if (status && errors) {
        atomicOr(status, errors);
        atomicCAS(status + 1, -2, layer);
    }
}

__device__ int live_rows(const int* control, int rows, int* status, int layer) {
    const int value = control ? *control : rows;
    if ((value < 0 || value > rows) && threadIdx.x == 0 && blockIdx.x == 0) latch(status, 16, layer);
    return value < 0 || value > rows ? 0 : value;
}

template<class V> __device__ float read(const V* source, int64_t index) { return source[index]; }
template<> __device__ float read(const __half* source, int64_t index) { return __half2float(source[index]); }
template<class V> __device__ void write(V* target, int64_t index, float value) { target[index] = value; }
template<> __device__ void write(__half* target, int64_t index, float value) {
    target[index] = __float2half_rn(value);
}
template<class V> __device__ bool representable(float value) { return isfinite(value); }
template<> __device__ bool representable<__half>(float value) {
    return isfinite(value) && isfinite(__half2float(__float2half_rn(value)));
}

template<class Input, int CODEC>
__global__ void encode_kernel(const Input* source, uint8_t* codes, float* scales,
                              const void* position, bool position64, int* status, const int* valid_rows,
                              int rows, int heads, int dim, int64_t source_stride, int64_t capacity,
                              int source_error, int layer) {
    __shared__ float shared[128];
    const int group_count = dim / 128, group = blockIdx.x % group_count;
    const int outer = blockIdx.x / group_count, row = outer / heads, head = outer % heads;
    const int live = live_rows(valid_rows, rows, status, layer);
    if (row >= live) return;                 // Dead graph lanes cannot mutate retained cache rows.
    const int64_t base = !position ? 0 : position64 ? *static_cast<const int64_t*>(position)
                                                          : *static_cast<const int*>(position);
    if (base < 0 || row >= capacity || base > capacity - 1 - row) {
        if (threadIdx.x == 0) latch(status, 16, layer);
        return;
    }
    const int col = threadIdx.x;
    const float value = read(source, int64_t(row) * source_stride + int64_t(head) * dim + group * 128 + col);
    const auto encoded = tf_rq4::encode_component<CODEC>(value, col, shared, source_error);
    if (col == 0) latch(status, encoded.errors, layer);
    const int64_t destination = ((base + row) * heads + head) * dim + group * 128;
    tf_rq4::store_encoded(encoded, col, shared, codes, scales, destination, 128, true);
}

template<class Input, class Output, int CODEC, bool INVERSE>
__global__ void rotate_kernel(const Input* source, Output* target, int* status, const int* valid_rows,
                              int rows, int heads, int dim, int64_t source_stride, int error, int layer) {
    __shared__ float shared[128];
    const int groups = dim / 128, group = blockIdx.x % groups;
    const int outer = blockIdx.x / groups, row = outer / heads, head = outer % heads;
    const int col = threadIdx.x;
    const int64_t output = (int64_t(row) * heads + head) * dim + group * 128 + col;
    if (row >= live_rows(valid_rows, rows, status, layer)) {
        write(target, output, 0.0f);
        return;
    }
    float value = read(source, int64_t(row) * source_stride + int64_t(head) * dim + group * 128 + col);
    const bool bad_source = __syncthreads_or(!isfinite(value));
    if (bad_source) value = 0.0f;
    value = tf_rq4::rotate_component<CODEC, INVERSE>(value, col, shared);
    const bool bad_output = __syncthreads_or(!representable<Output>(value));
    if (col == 0 && (bad_source || bad_output)) latch(status, error, layer);
    write(target, output, bad_source || bad_output ? 0.0f : value);
}

template<class Output, int CODEC, bool INVERSE>
__global__ void decode_kernel(const uint8_t* codes, const float* scales, Output* target,
                              int* status, const int* valid_rows,
                              int rows, int heads, int dim, int error, int layer) {
    __shared__ float shared[128];
    const int groups = dim / 128;
    const int outer = blockIdx.x / groups, row = outer / heads;
    const int col = threadIdx.x;
    const int64_t logical = int64_t(blockIdx.x) * 128 + col;
    if (row >= live_rows(valid_rows, rows, status, layer)) {
        write(target, logical, 0.0f);
        return;
    }
    const float word = scales[blockIdx.x];
    const float scale = tf_rq4::metadata_scale<CODEC>(word);
    const uint8_t packed = codes[logical / 2];
    const int code = (packed >> (4 * (col % 2))) & 15;
    const float value = __fmul_rn(tf_rq4::selected_centroid<CODEC>(code),
                                 tf_rq4::valid_metadata<CODEC>(word) ? scale : 0.0f);
    const bool bad_scale = __syncthreads_or(!tf_rq4::valid_metadata<CODEC>(word) || !isfinite(value));
    float output = bad_scale ? 0.0f : value;
    if constexpr (INVERSE) output = tf_rq4::rotate_component<CODEC, true>(output, col, shared);
    const bool bad_output = __syncthreads_or(!representable<Output>(output));
    if (col == 0 && (bad_scale || bad_output)) latch(status, error, layer);
    write(target, logical, bad_scale || bad_output ? 0.0f : output);
}

template<class Input, int CODEC>
void launch_encode(const T& x, T& codes, T& scales, const O& position, const O& status,
                   const O& valid, int rows, int error, int layer, cudaStream_t stream) {
    const int64_t blocks = int64_t(rows) * x.size(1) * (x.size(2) / 128);
    if (!blocks) return;
    encode_kernel<Input, CODEC><<<int(blocks), 128, 0, stream>>>(
        static_cast<const Input*>(x.data_ptr()), codes.data_ptr<uint8_t>(), scales.data_ptr<float>(),
        position ? position->data_ptr() : nullptr, position && position->scalar_type() == at::kLong,
        status ? status->data_ptr<int>() : nullptr, valid ? valid->data_ptr<int>() : nullptr,
        rows, x.size(1), x.size(2), x.stride(0), codes.size(0), error, layer);
}

template<class Input, class Output, int CODEC>
void launch_rotate(const T& x, T& out, const O& status, const O& valid,
                   int rows, bool inverse, int error, int layer, cudaStream_t stream) {
    const int64_t blocks = int64_t(rows) * x.size(1) * (x.size(2) / 128);
    if (!blocks) return;
    const auto invoke = [&](auto direction) {
        rotate_kernel<Input, Output, CODEC, decltype(direction)::value><<<int(blocks), 128, 0, stream>>>(
            static_cast<const Input*>(x.data_ptr()), static_cast<Output*>(out.data_ptr()),
            status ? status->data_ptr<int>() : nullptr, valid ? valid->data_ptr<int>() : nullptr,
            rows, x.size(1), x.size(2), x.stride(0), error, layer);
    };
    if (inverse) invoke(std::true_type{}); else invoke(std::false_type{});
}

template<class Output, int CODEC>
void launch_decode(const T& codes, const T& scales, T& out, const O& status, const O& valid,
                   int rows, bool inverse, int error, int layer, cudaStream_t stream) {
    const int64_t blocks = int64_t(rows) * out.size(1) * (out.size(2) / 128);
    if (!blocks) return;
    const auto invoke = [&](auto direction) {
        decode_kernel<Output, CODEC, decltype(direction)::value><<<int(blocks), 128, 0, stream>>>(
            codes.data_ptr<uint8_t>(), scales.data_ptr<float>(), static_cast<Output*>(out.data_ptr()),
            status ? status->data_ptr<int>() : nullptr, valid ? valid->data_ptr<int>() : nullptr,
            rows, out.size(1), out.size(2), error, layer);
    };
    if (inverse) invoke(std::true_type{}); else invoke(std::false_type{});
}

template<class Callable> void dispatch_codec(int codec, Callable callable) {
    switch (codec) {
        case 1: callable(std::integral_constant<int, 1>{}); break;
        case 2: callable(std::integral_constant<int, 2>{}); break;
        case 3: callable(std::integral_constant<int, 3>{}); break;
        case 4: callable(std::integral_constant<int, 4>{}); break;
        case 5: callable(std::integral_constant<int, 5>{}); break;
        case 9: callable(std::integral_constant<int, 9>{}); break;
    }
}

__global__ void copy_kernel(const uint8_t* source, uint8_t* target, const void* position, bool position64,
                            int* status, const int* valid, int rows, int64_t capacity,
                            int64_t source_stride, int64_t row_bytes, int layer) {
    const int row = blockIdx.x;
    if (row >= live_rows(valid, rows, status, layer)) return;
    const int64_t base = !position ? 0 : position64 ? *static_cast<const int64_t*>(position)
                                                 : *static_cast<const int*>(position);
    if (base < 0 || row >= capacity || base > capacity - 1 - row) {
        if (threadIdx.x == 0) latch(status, 16, layer);
        return;
    }
    for (int64_t col = threadIdx.x; col < row_bytes; col += blockDim.x)
        target[(base + row) * row_bytes + col] = source[int64_t(row) * source_stride + col];
}
}  // namespace

void sm70_rq4_encode_cuda(const T& x, T& codes, T& scales, const O& position, const O& status,
                          const O& valid, int codec, int rows, int error, int layer) {
    const auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    dispatch_codec(codec, [&](auto selected) {
        if (x.scalar_type() == at::kHalf)
            launch_encode<__half, decltype(selected)::value>(x, codes, scales, position, status, valid, rows, error, layer, stream);
        else launch_encode<float, decltype(selected)::value>(x, codes, scales, position, status, valid, rows, error, layer, stream);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sm70_rq4_rotate_cuda(const T& x, T& out, const O& status, const O& valid,
                          int codec, int rows, bool inverse, int error, int layer) {
    const auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    dispatch_codec(codec, [&](auto selected) {
        if (x.scalar_type() == at::kHalf) {
            if (out.scalar_type() == at::kHalf)
                launch_rotate<__half, __half, decltype(selected)::value>(x, out, status, valid, rows, inverse, error, layer, stream);
            else launch_rotate<__half, float, decltype(selected)::value>(x, out, status, valid, rows, inverse, error, layer, stream);
        } else if (out.scalar_type() == at::kHalf)
            launch_rotate<float, __half, decltype(selected)::value>(x, out, status, valid, rows, inverse, error, layer, stream);
        else launch_rotate<float, float, decltype(selected)::value>(x, out, status, valid, rows, inverse, error, layer, stream);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sm70_rq4_decode_cuda(const T& codes, const T& scales, T& out, const O& status, const O& valid,
                          int codec, int rows, bool inverse, int error, int layer) {
    const auto stream = at::cuda::getCurrentCUDAStream(out.get_device());
    dispatch_codec(codec, [&](auto selected) {
        if (out.scalar_type() == at::kHalf)
            launch_decode<__half, decltype(selected)::value>(codes, scales, out, status, valid, rows, inverse, error, layer, stream);
        else launch_decode<float, decltype(selected)::value>(codes, scales, out, status, valid, rows, inverse, error, layer, stream);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sm70_rq4_copy_cuda(const T& x, T& target, const O& position, const O& status, const O& valid, int layer) {
    if (!x.size(0)) return;
    const int64_t row_bytes = target.numel() / std::max<int64_t>(target.size(0), 1) * target.element_size();
    copy_kernel<<<int(x.size(0)), 256, 0, at::cuda::getCurrentCUDAStream(x.get_device())>>>(
        static_cast<const uint8_t*>(x.data_ptr()), static_cast<uint8_t*>(target.data_ptr()),
        position ? position->data_ptr() : nullptr, position && position->scalar_type() == at::kLong,
        status ? status->data_ptr<int>() : nullptr, valid ? valid->data_ptr<int>() : nullptr,
        x.size(0), target.size(0), x.stride(0) * x.element_size(), row_bytes, layer);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
