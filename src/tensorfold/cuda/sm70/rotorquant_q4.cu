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

template<bool TRAP_ON_ERROR>
__device__ __forceinline__ void report_error(int* status, int errors, int layer) {
    latch(status, errors, layer);
    if constexpr (TRAP_ON_ERROR) {
        if (errors) asm volatile("trap;");
    }
}

template<bool TRAP_ON_ERROR = false>
__device__ int live_rows(const int* control, int rows, int* status, int layer) {
    const int value = control ? *control : rows;
    const bool invalid = value < 0 || value > rows;
    if (invalid && threadIdx.x == 0 && blockIdx.x == 0)
        report_error<TRAP_ON_ERROR>(status, 16, layer);
    return invalid ? (TRAP_ON_ERROR ? -1 : 0) : value;
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

template<class Input, int CODEC, bool TRAP_ON_ERROR>
__global__ void encode_kernel(const Input* source, uint8_t* codes, float* scales,
                              const void* position, bool position64, int* status, const int* valid_rows,
                              int rows, int heads, int dim, int64_t source_stride, int64_t capacity,
                              int source_error, int layer) {
    __shared__ float shared[128];
    const int group_count = dim / 128, group = blockIdx.x % group_count;
    const int outer = blockIdx.x / group_count, row = outer / heads, head = outer % heads;
    const int live = live_rows<TRAP_ON_ERROR>(valid_rows, rows, status, layer);
    if (row >= live) return;                 // Dead graph lanes cannot mutate retained cache rows.
    const int64_t base = !position ? 0 : position64 ? *static_cast<const int64_t*>(position)
                                                          : *static_cast<const int*>(position);
    if (base < 0 || row >= capacity || base > capacity - 1 - row) {
        if (threadIdx.x == 0) report_error<TRAP_ON_ERROR>(status, 16, layer);
        return;
    }
    const int col = threadIdx.x;
    const float value = read(source, int64_t(row) * source_stride + int64_t(head) * dim + group * 128 + col);
    const auto encoded = tf_rq4::encode_component<CODEC>(value, col, shared, source_error);
    if (col == 0) report_error<TRAP_ON_ERROR>(status, encoded.errors, layer);
    if constexpr (TRAP_ON_ERROR) {
        if (encoded.errors) return;
    }
    const int64_t destination = ((base + row) * heads + head) * dim + group * 128;
    if constexpr (tf_rq4::eight_bit(CODEC)) tf_rq4::store_encoded8(encoded, col, codes, scales, destination, 128, true);
    else if constexpr (CODEC == 11) tf_rq4::store_encoded_halves(encoded, col, shared, codes, scales, destination, 128, true);
    else tf_rq4::store_encoded(encoded, col, shared, codes, scales, destination, 128, true);
}

template<class Input>
__global__ void pack_h64_kernel(const Input* source, const bool* valid, uint8_t* out, int* status,
                                int64_t source_row, int64_t source_col, int64_t valid_stride, int layer) {
    constexpr int dim = 512, group = 128, groups = dim / group;
    constexpr int code_bytes = dim / 2, row_bytes = code_bytes + groups * sizeof(uint32_t);
    __shared__ float shared[group];
    const int cluster = blockIdx.x % groups, col = threadIdx.x;
    const int64_t row = blockIdx.x / groups;
    // A masked lane has the exact wire representation of an all-positive-zero
    // input. Its projection storage may contain nonfinite or unwritten values.
    const float value = valid[row * valid_stride]
        ? read(source, row * source_row + (cluster * group + col) * source_col) : 0.0f;
    const auto encoded = tf_rq4::encode_component<11>(value, col, shared, 2);
    if (col == 0) report_error<true>(status, encoded.errors, layer);
    if (encoded.errors) return;

    uint8_t* destination = out + row * row_bytes;
    auto* group_scale = reinterpret_cast<float*>(destination + code_bytes + cluster * sizeof(uint32_t));
    tf_rq4::store_encoded_halves(encoded, col, shared, destination + cluster * (group / 2), group_scale,
                                 0, group, true);
}

template<class Output, int CODEC, bool INVERSE, bool TRAP_ON_ERROR>
__device__ __forceinline__ void rotate_group(float value, Output* target, int64_t output, int col,
                                            float* shared, int* status, int error, int layer) {
    const bool bad_source = __syncthreads_or(!isfinite(value));
    if (bad_source) value = 0.0f;
    value = tf_rq4::rotate_component<CODEC, INVERSE>(value, col, shared);
    const bool bad_output = __syncthreads_or(!representable<Output>(value));
    if (col == 0 && (bad_source || bad_output)) report_error<TRAP_ON_ERROR>(status, error, layer);
    if constexpr (TRAP_ON_ERROR) {
        if (bad_source || bad_output) return;
    }
    write(target, output, bad_source || bad_output ? 0.0f : value);
}

template<class Input, class Output, int CODEC, bool INVERSE, bool TRAP_ON_ERROR>
__global__ void rotate_kernel(const Input* source, Output* target, int* status, const int* valid_rows,
                              int rows, int heads, int dim, int64_t source_stride, int error, int layer) {
    __shared__ float shared[128];
    const int groups = dim / 128, group = blockIdx.x % groups;
    const int outer = blockIdx.x / groups, row = outer / heads, head = outer % heads;
    const int col = threadIdx.x;
    const int64_t output = (int64_t(row) * heads + head) * dim + group * 128 + col;
    const int live = live_rows<TRAP_ON_ERROR>(valid_rows, rows, status, layer);
    if (row >= live) {
        if constexpr (TRAP_ON_ERROR) {
            if (live < 0 || rows == 0) return;
        }
        write(target, output, 0.0f);
        return;
    }
    const float value = read(source, int64_t(row) * source_stride + int64_t(head) * dim + group * 128 + col);
    rotate_group<Output, CODEC, INVERSE, TRAP_ON_ERROR>(value, target, output, col, shared, status, error, layer);
}

template<class Value, bool INVERSE>
__global__ void rotate_h64_kernel(const Value* source, const bool* valid, Value* target, int* status,
                                  int heads, int dim, int64_t source_row, int64_t source_head,
                                  int64_t source_col, int64_t valid_stride, int layer) {
    __shared__ float shared[128];
    const int groups = dim / 128, group = blockIdx.x % groups;
    const int outer = blockIdx.x / groups, row = outer / heads, head = outer % heads;
    const int col = threadIdx.x;
    const int64_t output = int64_t(blockIdx.x) * 128 + col;
    // Run the original transform on positive zero for holes. A zero-filled
    // output would skip the transform's signed-zero behavior.
    const float value = valid[int64_t(row) * valid_stride]
        ? read(source, int64_t(row) * source_row + int64_t(head) * source_head + (group * 128 + col) * source_col)
        : 0.0f;
    rotate_group<Value, 11, INVERSE, true>(value, target, output, col, shared, status, 1, layer);
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
    float scale = tf_rq4::metadata_scale<CODEC>(word);
    bool valid = tf_rq4::valid_metadata<CODEC>(word);
    if constexpr (CODEC == 11) {
        scale = tf_rq4::half_group_scale(word, col / 64);
        valid = tf_rq4::valid_half_group_scale(scale);
    }
    int code;
    if constexpr (tf_rq4::eight_bit(CODEC)) code = codes[logical];
    else code = (codes[logical / 2] >> (4 * (col % 2))) & 15;
    const float value = __fmul_rn(tf_rq4::selected_centroid<CODEC>(code),
                                 valid ? scale : 0.0f);
    const bool bad_scale = __syncthreads_or(!valid || !isfinite(value));
    float output = bad_scale ? 0.0f : value;
    if constexpr (INVERSE) output = tf_rq4::rotate_component<CODEC, true>(output, col, shared);
    const bool bad_output = __syncthreads_or(!representable<Output>(output));
    if (col == 0 && (bad_scale || bad_output)) latch(status, error, layer);
    write(target, logical, bad_scale || bad_output ? 0.0f : output);
}

template<class Input, int CODEC, bool TRAP_ON_ERROR>
void launch_encode(const T& x, T& codes, T& scales, const O& position, const O& status,
                   const O& valid, int rows, int error, int layer, cudaStream_t stream) {
    const int64_t blocks = int64_t(rows) * x.size(1) * (x.size(2) / 128);
    // Just the device control is checked for an empty fatal-mode launch.
    if (!blocks && !(TRAP_ON_ERROR && valid)) return;
    encode_kernel<Input, CODEC, TRAP_ON_ERROR><<<int(std::max<int64_t>(blocks, 1)), 128, 0, stream>>>(
        static_cast<const Input*>(x.data_ptr()), codes.data_ptr<uint8_t>(), scales.data_ptr<float>(),
        position ? position->data_ptr() : nullptr, position && position->scalar_type() == at::kLong,
        status ? status->data_ptr<int>() : nullptr, valid ? valid->data_ptr<int>() : nullptr,
        rows, x.size(1), x.size(2), x.stride(0), codes.size(0), error, layer);
}

template<class Input, class Output, int CODEC, bool TRAP_ON_ERROR>
void launch_rotate(const T& x, T& out, const O& status, const O& valid,
                   int rows, bool inverse, int error, int layer, cudaStream_t stream) {
    const int64_t blocks = int64_t(rows) * x.size(1) * (x.size(2) / 128);
    if (!blocks && !(TRAP_ON_ERROR && valid)) return;
    const auto invoke = [&](auto direction) {
        rotate_kernel<Input, Output, CODEC, decltype(direction)::value, TRAP_ON_ERROR>
            <<<int(std::max<int64_t>(blocks, 1)), 128, 0, stream>>>(
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
        case 10: callable(std::integral_constant<int, 10>{}); break;
        case 11: callable(std::integral_constant<int, 11>{}); break;
        case 8: callable(std::integral_constant<int, 8>{}); break;
        case 12: callable(std::integral_constant<int, 12>{}); break;
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
                          const O& valid, int codec, int rows, int error, int layer, bool trap_on_error) {
    const auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    const auto invoke = [&](auto trap) {
        dispatch_codec(codec, [&](auto selected) {
            constexpr int format = decltype(selected)::value;
            constexpr bool fatal = decltype(trap)::value;
            if (x.scalar_type() == at::kHalf)
                launch_encode<__half, format, fatal>(x, codes, scales, position, status, valid, rows, error, layer, stream);
            else launch_encode<float, format, fatal>(x, codes, scales, position, status, valid, rows, error, layer, stream);
        });
    };
    if (trap_on_error) invoke(std::true_type{}); else invoke(std::false_type{});
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sm70_rq4_pack_h64_cuda(const T& x, const T& valid, T& out, T& status, int layer) {
    if (!x.size(0)) return;
    const auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    const int blocks = int(x.size(0) * 4);
    const auto launch = [&](auto input) {
        pack_h64_kernel<<<blocks, 128, 0, stream>>>(input, valid.data_ptr<bool>(), out.data_ptr<uint8_t>(),
            status.data_ptr<int>(), x.stride(0), x.stride(1), valid.stride(0), layer);
    };
    if (x.scalar_type() == at::kHalf) launch(static_cast<const __half*>(x.data_ptr()));
    else launch(static_cast<const float*>(x.data_ptr()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sm70_rq4_rotate_cuda(const T& x, T& out, const O& status, const O& valid,
                          int codec, int rows, bool inverse, int error, int layer, bool trap_on_error) {
    const auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    const auto invoke = [&](auto trap) {
        dispatch_codec(codec, [&](auto selected) {
            constexpr int format = decltype(selected)::value;
            constexpr bool fatal = decltype(trap)::value;
            if (x.scalar_type() == at::kHalf) {
                if (out.scalar_type() == at::kHalf)
                    launch_rotate<__half, __half, format, fatal>(x, out, status, valid, rows, inverse, error, layer, stream);
                else launch_rotate<__half, float, format, fatal>(x, out, status, valid, rows, inverse, error, layer, stream);
            } else if (out.scalar_type() == at::kHalf)
                launch_rotate<float, __half, format, fatal>(x, out, status, valid, rows, inverse, error, layer, stream);
            else launch_rotate<float, float, format, fatal>(x, out, status, valid, rows, inverse, error, layer, stream);
        });
    };
    if (trap_on_error) invoke(std::true_type{}); else invoke(std::false_type{});
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sm70_rq4_rotate_h64_cuda(const T& x, const T& valid, T& out, T& status, bool inverse, int layer) {
    if (!x.size(0)) return;
    const auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    const int blocks = int(x.size(0) * x.size(1) * (x.size(2) / 128));
    const auto typed = [&](auto output) {
        using Value = std::remove_pointer_t<decltype(output)>;
        const auto launch = [&](auto direction) {
            rotate_h64_kernel<Value, decltype(direction)::value><<<blocks, 128, 0, stream>>>(
                static_cast<const Value*>(x.data_ptr()), valid.data_ptr<bool>(), output, status.data_ptr<int>(),
                x.size(1), x.size(2), x.stride(0), x.stride(1), x.stride(2), valid.stride(0), layer);
        };
        if (inverse) launch(std::true_type{}); else launch(std::false_type{});
    };
    if (x.scalar_type() == at::kHalf) typed(static_cast<__half*>(out.data_ptr()));
    else typed(out.data_ptr<float>());
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
