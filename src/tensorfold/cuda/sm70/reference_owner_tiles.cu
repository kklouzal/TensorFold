// Fused original reference index/gather/decode preparation; no attention reductions.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include "packed_kv_decode.cuh"

namespace {
__device__ void fail(int* status, int bit, int layer) {
    atomicOr(status, bit);
    atomicCAS(status + 1, -2, layer);
}
__device__ bool epoch(const int64_t* generations, int64_t slot, int64_t owner, int* status, int layer) {
    if (generations[0] == slot && generations[1] == owner) return true;
    if (!threadIdx.x) fail(status, 32, layer);
    return false;
}
__device__ __half decode(const void* data, const void* scales, int bits, int codec,
                        int64_t at, int* status, int layer) {
    bool valid;
    if (bits == 8) {
        const float scale = __half2float(static_cast<const __half*>(scales)[at / 32]);
        valid = isfinite(scale) && scale >= 0.f;
    } else {
        valid = tf_rq4::valid_metadata<9>(static_cast<const float*>(scales)[at / 128]);
    }
    if (!valid) { fail(status, 8, layer); return __float2half_rn(0.f); }
    const __half value = tf_sm70::decoded_half(data, scales, bits, codec, at);
    if (!isfinite(__half2float(value))) { fail(status, 8, layer); return __float2half_rn(0.f); }
    return value;
}
__global__ void keys_kernel(const void* data, const void* scales, const int64_t* pages,
    const int64_t* positions, const bool* visible, const int64_t* generations, int* status,
    int64_t* indices, bool* masks, __half* out, int logical, int physical, int page_tokens,
    int bits, int codec, int64_t slot, int64_t owner, int layer) {
    const int row = blockIdx.x, col = threadIdx.x;
    __shared__ int64_t offset;
    __shared__ bool mask;
    const bool live = epoch(generations, slot, owner, status, layer);
    if (!col) {
        mask = false;
        offset = 0;
        if (live) {
            const int64_t position = positions[row];
            const int safe = position < 0 ? 0 : position >= logical ? logical - 1 : int(position);
            const int64_t page = pages[safe / page_tokens];
            if (page < -1 || page > int64_t(physical)) fail(status, 32, layer);
            else {
                const int64_t candidate = page + safe % page_tokens;
                offset = candidate > 0 ? candidate : 0;
                mask = position >= 0 && position < logical && page >= 0 && visible[row];
                if (mask && offset >= physical) { fail(status, 32, layer); mask = false; }
            }
        }
        indices[row] = offset;
        masks[row] = mask;
    }
    __syncthreads();
    out[int64_t(row) * 256 + col] = mask
        ? decode(data, scales, bits, codec, offset * 256 + col, status, layer) : __float2half_rn(0.f);
}
__global__ void values_kernel(const void* data, const void* scales, const int64_t* indices,
    const bool* masks, const int64_t* generations, int* status, __half* out,
    int physical, int bits, int codec, int64_t slot, int64_t owner, int layer) {
    const int row = blockIdx.x, col = threadIdx.x;
    const bool live = epoch(generations, slot, owner, status, layer);
    const int64_t at = indices[row];
    const bool valid = live && masks[row] && at >= 0 && at < physical;
    if (live && masks[row] && !valid && !col) fail(status, 32, layer);
    out[int64_t(row) * 256 + col] = valid
        ? decode(data, scales, bits, codec, at * 256 + col, status, layer) : __float2half_rn(0.f);
}
}
void reference_keys_cuda(const at::Tensor& data, const at::Tensor& scales, const at::Tensor& pages,
    const at::Tensor& positions, const at::Tensor& visible, const at::Tensor& generations, at::Tensor& status,
    at::Tensor& indices, at::Tensor& masks, at::Tensor& out, int logical, int page_tokens,
    int bits, int codec, int64_t slot, int64_t owner, int layer) {
    if (!positions.numel()) return;
    keys_kernel<<<positions.numel(),256,0,at::cuda::getCurrentCUDAStream()>>>(data.data_ptr(),scales.data_ptr(),
        pages.data_ptr<int64_t>(),positions.data_ptr<int64_t>(),visible.data_ptr<bool>(),generations.data_ptr<int64_t>(),
        status.data_ptr<int>(),indices.data_ptr<int64_t>(),masks.data_ptr<bool>(),reinterpret_cast<__half*>(out.data_ptr()),
        logical,data.size(0),page_tokens,bits,codec,slot,owner,layer);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void reference_values_cuda(const at::Tensor& data, const at::Tensor& scales, const at::Tensor& indices,
    const at::Tensor& masks, const at::Tensor& generations, at::Tensor& status, at::Tensor& out,
    int count, int bits, int codec, int64_t slot, int64_t owner, int layer) {
    if (!count) return;
    values_kernel<<<count,256,0,at::cuda::getCurrentCUDAStream()>>>(data.data_ptr(),scales.data_ptr(),
        indices.data_ptr<int64_t>(),masks.data_ptr<bool>(),generations.data_ptr<int64_t>(),status.data_ptr<int>(),
        reinterpret_cast<__half*>(out.data_ptr()),data.size(0),bits,codec,slot,owner,layer);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Force both functions resident under CUDA lazy loading before capacity snapshots.
void reference_tiles_prepare_cuda() {
    cudaFuncAttributes attributes;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, keys_kernel));
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, values_kernel));
}
