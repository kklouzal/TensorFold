#pragma once

// Programmatic dependent launch for the decode kernels (tensorfold/cuda/pdl.py): a kernel launched with the
// attribute may be scheduled while its stream predecessor finishes, so every block calls pdl_wait() as its first
// operation on every path, before any global access: the predecessor grid complete and its writes visible, then
// this grid's dependents may launch (they wait in turn). Without the attribute both instructions return at once.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <utility>

namespace tensorfold {

__device__ __forceinline__ void pdl_wait() {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
}

// ``kernel<<<grid, block, smem, current stream>>>(args...)``, with the programmatic serialization attribute when
// ``pdl`` (the kernel must call pdl_wait() first on every path).
template <typename... Params, typename... Args>
inline void launch(void (*kernel)(Params...), dim3 grid, dim3 block, size_t smem, bool pdl, Args&&... args) {
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem;
    config.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    config.attrs = attr;
    config.numAttrs = pdl ? 1 : 0;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, std::forward<Args>(args)...));
}

}  // namespace tensorfold
