#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <algorithm>

namespace {
__global__ void round_half(const float* input, __half* output, int64_t count,
                           int64_t columns, int64_t input_stride, int64_t output_stride) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += int64_t(blockDim.x) * gridDim.x) {
        const int64_t row = i / columns, column = i % columns;
        output[row * output_stride + column] = __float2half_rn(input[row * input_stride + column]);
    }
}
}

void prefill_dense_round_half(const at::Tensor& input, at::Tensor output) {
    const int64_t count = input.numel();
    if (!count) return;
    const unsigned blocks = unsigned(std::min<int64_t>((count + 255) / 256, 65535));
    round_half<<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(input.data_ptr<float>(),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()), count, input.size(1),
        input.stride(0), output.stride(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
