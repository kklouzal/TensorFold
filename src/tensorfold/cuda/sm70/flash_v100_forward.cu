// Owned adapter to ai-bond/flash-attention-v100 tag26.10, commit0ababf82.
// The foreign kernel definition remains unmodified in its separate source TU.
#include "flash_v100_forward.cuh"
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <math_constants.h>
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

// Same immutable upstream pin; the varlen definition is an unmodified extra TU.
template<int D, bool CAUSAL, bool ALIBI, bool SOFTCAP, bool WINDOW, bool DROPOUT, bool PAGED>
__global__ void flash_attention_forward_varlen_kernel(
    const __half*, const __half*, const __half*, __half*, float*, __half*,
    const int*, const int*, const int*, const int*, const int*,
    int, int, int, int, int, int, int, int, int, float, float, const float*,
    int, int, int, float, uint64_t, uint64_t);
namespace {
constexpr int CAPACITY = 2051;
__global__ void initialize_compact(__half* k, __half* v, int* cq, int* ck, int* used, int batch) {
    const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < int64_t(batch) * CAPACITY * 256) k[index] = v[index] = __float2half_rn(0.f);
    if (index <= batch) { cq[index] = int(index) * 12; ck[index] = int(index) * CAPACITY; }
    if (index < batch) used[index] = 1; // Initialized one-key cold warmup, not a serving mask.
}
__global__ void compact_kernel(const __half* k, const __half* v, const bool* valid,
                               __half* packed_k, __half* packed_v, int* used) {
    __shared__ int totals[256], destination[CAPACITY];
    const int tid = threadIdx.x, batch = blockIdx.x;
    constexpr int CHUNK = (CAPACITY + 255) / 256;
    int count = 0;
    for (int i = 0; i < CHUNK; ++i) {
        const int key = tid * CHUNK + i;
        if (key < CAPACITY && valid[int64_t(batch) * CAPACITY + key]) ++count;
    }
    totals[tid] = count;
    __syncthreads();
    // Inclusive scan over contiguous thread chunks; stable integer ranks do
    // not depend on CTA scheduling or payload values. Every thread participates.
    for (int distance = 1; distance < 256; distance *= 2) {
        const int add = tid >= distance ? totals[tid - distance] : 0;
        __syncthreads();
        totals[tid] += add;
        __syncthreads();
    }
    int next = tid ? totals[tid - 1] : 0;
    for (int i = 0; i < CHUNK; ++i) {
        const int key = tid * CHUNK + i;
        if (key < CAPACITY)
            destination[key] = valid[int64_t(batch) * CAPACITY + key] ? next++ : -1;
    }
    if (!tid) used[batch] = totals[255];
    __syncthreads();
    // One warp per key gives coalesced channel copies. Invalid payload bytes
    // are never read; out-of-place buffers prevent prefix overwrite races.
    for (int key = tid / 32; key < CAPACITY; key += 8) {
        const int target = destination[key];
        if (target < 0) continue;
        for (int col = tid % 32; col < 256; col += 32) {
            const int64_t source = (int64_t(batch) * CAPACITY + key) * 256 + col;
            const int64_t dest = (int64_t(batch) * CAPACITY + target) * 256 + col;
            packed_k[dest] = k[source]; packed_v[dest] = v[source];
        }
    }
}
__global__ void canonicalize_empty(__half* out, float* lse, const int* used, int batch) {
    const int row = blockIdx.x;
    if (row >= batch || used[row]) return;
    for (int at = threadIdx.x; at < 12 * 256; at += blockDim.x)
        out[int64_t(row) * 12 * 256 + at] = __float2half_rn(0.f);
    if (threadIdx.x < 12) lse[row * 12 + threadIdx.x] = -CUDART_INF_F;
}
}
void flash_v100_prepare_varlen_kernel(FlashV100Info& info) {
    auto kernel = flash_attention_forward_varlen_kernel<256, false, false, false, false, false, false>;
    info.shared_bytes = KernelConfig<256>::TOTAL_SMEM;
    info.threads = KernelConfig<256>::THREADS_PER_BLOCK;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, info.shared_bytes));
    C10_CUDA_CHECK(cudaFuncGetAttributes(&info.attributes, kernel));
}
void flash_v100_init_compact(at::Tensor k, at::Tensor v, at::Tensor cq, at::Tensor ck,
                            at::Tensor used, cudaStream_t stream) {
    const int batch = used.numel();
    initialize_compact<<<(k.numel() + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<__half*>(k.data_ptr()), reinterpret_cast<__half*>(v.data_ptr()),
        cq.data_ptr<int>(), ck.data_ptr<int>(), used.data_ptr<int>(), batch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void flash_v100_compact(const at::Tensor& k, const at::Tensor& v, const at::Tensor& valid,
                       at::Tensor pk, at::Tensor pv, at::Tensor used, cudaStream_t stream) {
    compact_kernel<<<used.numel(), 256, 0, stream>>>(
        reinterpret_cast<const __half*>(k.data_ptr()), reinterpret_cast<const __half*>(v.data_ptr()),
        valid.data_ptr<bool>(), reinterpret_cast<__half*>(pk.data_ptr()),
        reinterpret_cast<__half*>(pv.data_ptr()), used.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void flash_v100_launch_varlen(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& cq, const at::Tensor& ck, const at::Tensor& used,
    at::Tensor out, at::Tensor lse, cudaStream_t stream) {
    const int batch = used.numel(), total_queries = batch * 12;
    flash_attention_forward_varlen_kernel<256, false, false, false, false, false, false>
        <<<dim3(1, 1, batch), 512, 90368, stream>>>(
            reinterpret_cast<const __half*>(q.data_ptr()), reinterpret_cast<const __half*>(k.data_ptr()),
            reinterpret_cast<const __half*>(v.data_ptr()), reinterpret_cast<__half*>(out.data_ptr()),
            lse.data_ptr<float>(), nullptr, cq.data_ptr<int>(), ck.data_ptr<int>(), used.data_ptr<int>(),
            nullptr, nullptr, batch, 1, 1, total_queries, 12, CAPACITY, 1, 0, 0,
            .0625f, 0.f, nullptr, 0, -1, -1, 0.f, 0, 0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    canonicalize_empty<<<batch, 256, 0, stream>>>(reinterpret_cast<__half*>(out.data_ptr()),
        lse.data_ptr<float>(), used.data_ptr<int>(), batch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
