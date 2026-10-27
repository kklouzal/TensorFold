#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <climits>

namespace {
constexpr int THREADS = 256;
constexpr int64_t SENTINEL = INT64_MAX;
struct Record { uint32_t bits; int first; int64_t id; };

__device__ __forceinline__ bool nan(uint32_t bits) {
    return (bits & 0x7fffffffU) > 0x7f800000U;
}
__device__ __forceinline__ uint32_t key(uint32_t bits) {
    // Preserve subnormal comparisons without depending on compiler FTZ policy.
    // Both signed zeros compare equal; raw bits remain a separate record field.
    if ((bits & 0x7fffffffU) == 0) return 0x80000000U;
    return (bits & 0x80000000U) ? ~bits : (bits | 0x80000000U);
}
__device__ __forceinline__ Record empty() { return {0xff800000U, INT_MAX, SENTINEL}; }

__device__ __forceinline__ Record combine(Record a, Record b) {
    const bool an = nan(a.bits), bn = nan(b.bits);
    if (an || bn) {
        Record result = bn && (!an || b.first < a.first) ? b : a;
        result.id = SENTINEL; // torch.eq(original, NaN maximum) has no true entry
        return result;
    }
    const uint32_t ak = key(a.bits), bk = key(b.bits);
    if (ak < bk) return b;
    if (bk < ak) return a;
    Record result = b.first < a.first ? b : a;
    result.id = a.id < b.id ? a.id : b.id;
    return result;
}

__device__ __forceinline__ Record warp_reduce(Record value) {
    for (int delta = 16; delta; delta >>= 1) {
        Record other{__shfl_down_sync(0xffffffffU, value.bits, delta),
                     __shfl_down_sync(0xffffffffU, value.first, delta),
                     static_cast<int64_t>(__shfl_down_sync(0xffffffffU,
                         static_cast<unsigned long long>(value.id), delta))};
        value = combine(value, other);
    }
    return value;
}

__global__ void local_kernel(const uint32_t* scores, const int64_t* ids, const int32_t* norm,
                             const int32_t* status, int64_t* packet, int width) {
    Record value = empty();
    for (int64_t column = threadIdx.x; column < width; column += THREADS) {
        Record item{scores[column], static_cast<int>(column), ids[column]};
        value = combine(value, item);
    }
    value = warp_reduce(value);
    __shared__ Record warps[THREADS / 32];
    if ((threadIdx.x & 31) == 0) warps[threadIdx.x / 32] = value;
    __syncthreads();
    if (threadIdx.x < 32) {
        value = threadIdx.x < THREADS/32 ? warps[threadIdx.x] : empty();
        value = warp_reduce(value);
        if (threadIdx.x == 0) {
            packet[0] = static_cast<int64_t>(static_cast<int32_t>(value.bits));
            packet[1] = value.id;
            packet[2] = static_cast<int64_t>(norm[0]);
            packet[3] = status[0];
            packet[4] = status[1];
        }
    }
}

__global__ void global_kernel(const int64_t* frame, int64_t* token, int32_t* input,
                              int world, int64_t vocabulary) {
    // Eight words fit one lane; avoiding block/shared synchronization is exact.
    Record value = empty();
    for (int rank = 0; rank < world; ++rank) {
        Record item{static_cast<uint32_t>(frame[rank*5]), rank, frame[rank*5+1]};
        value = combine(value, item);
    }
    token[0] = value.id;
    input[0] = value.id >= 0 && value.id < vocabulary && value.id <= INT_MAX ? static_cast<int32_t>(value.id) : 0;
}

__global__ void target_local_kernel(const float* scores, int64_t* packet, int width, int64_t offset) {
    const int row = blockIdx.x;
    Record value = empty();
    for (int64_t column = threadIdx.x; column < width; column += THREADS) {
        Record item{__float_as_uint(scores[int64_t(row) * width + column]), static_cast<int>(column), offset + column};
        value = combine(value, item);
    }
    value = warp_reduce(value);
    __shared__ Record warps[THREADS / 32];
    if ((threadIdx.x & 31) == 0) warps[threadIdx.x / 32] = value;
    __syncthreads();
    if (threadIdx.x < 32) {
        value = threadIdx.x < THREADS / 32 ? warps[threadIdx.x] : empty();
        value = warp_reduce(value);
        if (threadIdx.x == 0) {
            packet[int64_t(row) * 2] = static_cast<int64_t>(static_cast<int32_t>(value.bits));
            packet[int64_t(row) * 2 + 1] = value.id;
        }
    }
}

__global__ void target_accept_kernel(const int64_t* frame, const int32_t* inputs, const int32_t* eos,
                                     int32_t* feedback, int world, int rows, int eos_count,
                                     int64_t room, int64_t vocabulary, bool stop_eos) {
    int flags = 0;
    for (int row = 0; row < rows; ++row) {
        Record value = empty();
        for (int rank = 0; rank < world; ++rank) {
            const int64_t at = (int64_t(rank) * rows + row) * 2;
            for (int previous = 0; previous < rank; ++previous) {
                if (frame[(int64_t(previous) * rows + row) * 2 + 1] == frame[at + 1]) flags |= 16;
            }
            value = combine(value, Record{static_cast<uint32_t>(frame[at]), rank, frame[at + 1]});
        }
        if (nan(value.bits)) flags |= 1;
        if (value.bits == 0x7f800000U) flags |= 2;
        if (value.bits == 0xff800000U) flags |= 4;
        const bool valid = value.id >= 0 && value.id < vocabulary && value.id <= INT_MAX;
        if (!valid) flags |= 8;
        feedback[int64_t(row) + 2] = valid ? static_cast<int32_t>(value.id) : 0;
    }
    int keep = 1;
    while (keep < rows && keep < room) {
        bool ended = false;
        if (stop_eos) {
            for (int index = 0; index < eos_count; ++index) ended |= feedback[keep + 1] == eos[index];
        }
        if (ended || feedback[keep + 1] != inputs[keep]) break;
        ++keep;
    }
    feedback[0] = keep;
    feedback[1] = flags;  // Every produced row, including rejected branches, contributes its error.
    for (int row = keep; row < rows; ++row) feedback[int64_t(row) + 2] = 0;
}
} // namespace

void greedy_local_cuda(const at::Tensor& logits, const at::Tensor& ids, const at::Tensor& norm,
                       const at::Tensor& status, at::Tensor& packet) {
    local_kernel<<<1, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint32_t*>(logits.data_ptr<float>()), ids.data_ptr<int64_t>(),
        reinterpret_cast<const int32_t*>(norm.data_ptr<float>()), status.data_ptr<int32_t>(),
        packet.data_ptr<int64_t>(), static_cast<int>(logits.size(1)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void greedy_global_cuda(const at::Tensor& frame, at::Tensor& token, at::Tensor& input, int64_t vocabulary) {
    global_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(frame.data_ptr<int64_t>(), token.data_ptr<int64_t>(),
        input.data_ptr<int32_t>(), static_cast<int>(frame.size(0)), vocabulary);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void target_local_cuda(const at::Tensor& logits, at::Tensor& packet, int64_t offset) {
    target_local_kernel<<<logits.size(0), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        logits.data_ptr<float>(), packet.data_ptr<int64_t>(), static_cast<int>(logits.size(1)), offset);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void target_accept_cuda(const at::Tensor& frame, const at::Tensor& inputs, const at::Tensor& eos,
                        at::Tensor& feedback, int64_t room, int64_t vocabulary, bool stop_eos) {
    target_accept_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(frame.data_ptr<int64_t>(),
        inputs.data_ptr<int32_t>(), eos.data_ptr<int32_t>(), feedback.data_ptr<int32_t>(),
        static_cast<int>(frame.size(0)), static_cast<int>(frame.size(1)), static_cast<int>(eos.numel()),
        room, vocabulary, stop_eos);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
