#pragma once

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <memory>
#include <mutex>
#include <limits>

namespace tensorfold {

inline void check_grid(int64_t blocks, int z, const cudaDeviceProp* properties) {
    TORCH_CHECK(blocks >= 1 && blocks <= std::numeric_limits<int>::max() && blocks <= properties->maxGridSize[0] &&
                z >= 1 && z <= properties->maxGridSize[2], "QMM launch grid exceeds signed-int or device grid dimensions");
}

// One instance per compiled kernel specialization, owned by its loaded module.
// Torch's visible device count and primary contexts are stable for that module's
// lifetime; resetting those contexts requires process restart. The caller holds
// the device guard. Successful configuration is published once per device;
// a CUDA failure propagates and leaves that device's once flag uncommitted.
class KernelConfiguration {
public:
    KernelConfiguration() : devices_(c10::cuda::device_count()), flags_(std::make_unique<std::once_flag[]>(devices_)) {
        TORCH_CHECK(devices_ > 0, "kernel configuration requires a CUDA device");
    }

    template <typename Kernel>
    void configure(Kernel kernel, int bytes, int device) {
        TORCH_CHECK(device >= 0 && device < devices_, "kernel configuration device is outside the initialized device set");
        std::call_once(flags_[device], [&] {
            C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        });
    }

private:
    const int devices_;
    std::unique_ptr<std::once_flag[]> flags_;
};

} // namespace tensorfold
