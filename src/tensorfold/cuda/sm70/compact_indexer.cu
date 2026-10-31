#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>
#include <cstdint>

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
