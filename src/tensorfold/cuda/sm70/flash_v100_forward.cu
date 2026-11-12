// Owned adapter to ai-bond/flash-attention-v100 tag26.10, commit0ababf82.
// The foreign kernel definition remains unmodified in its separate source TU.
#include "flash_v100_forward.cuh"
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cstdint>
#include "kernel.h"
#include "forward.h"

// Exact foreign declaration from pinned kernel/forward.cu. Build input digest
// verification and actual separate-TU linking are required before publication.
template<int D, bool CAUSAL, bool ALIBI, bool SOFTCAP, bool WINDOW, bool DROPOUT>
__global__ void flash_attention_forward_kernel(
    const __half*, const __half*, const __half*, __half*, float*, __half*,
    int, int, int, int, int, float, float, const float*, int, int, int, float, uint64_t, uint64_t);

void flash_v100_prepare_kernel(FlashV100Info& info) {
    using Config = KernelConfig<256>;
    static_assert(Config::TOTAL_SMEM == 90368 && Config::THREADS_PER_BLOCK == 512,
                  "exact FlashAttention-V10026.10 head256 source configuration");
    auto kernel = flash_attention_forward_kernel<256, false, false, false, false, false>;
    info.shared_bytes = Config::TOTAL_SMEM;
    info.threads = Config::THREADS_PER_BLOCK;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, info.shared_bytes));
    C10_CUDA_CHECK(cudaFuncGetAttributes(&info.attributes, kernel));
}

void flash_v100_launch(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                      at::Tensor out, at::Tensor lse, float scale, cudaStream_t stream) {
    const int batch = q.size(0), rows = q.size(1), heads = q.size(2), kvheads = k.size(2), keys = k.size(1);
    const dim3 grid((int64_t(rows) + 31) / 32, 1, batch * heads);
    flash_attention_forward_kernel<256, false, false, false, false, false>
        <<<grid, 512, 90368, stream>>>(
            reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()), lse.data_ptr<float>(), nullptr,
            batch, heads, kvheads, rows, keys, scale, 0.f, nullptr, 0, -1, -1, 0.f, 0, 0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
