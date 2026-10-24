// Original TensorFold SM70 primitives: FP16 storage, explicit FP32 arithmetic.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include "../kernels/tensor_contracts.cuh"

namespace {
constexpr int THREADS = 256;

void check_half(const at::Tensor& x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.is_contiguous(),
                "SM70 input must be contiguous CUDA FP16");
    TORCH_CHECK(at::cuda::getDeviceProperties(x.get_device())->major == 7 &&
                at::cuda::getDeviceProperties(x.get_device())->minor == 0, "SM70 requires Volta");
}

__global__ void norm_kernel(const __half* x, const float* w, __half* out, int width, float eps) {
    __shared__ float sums[THREADS];
    const int t = threadIdx.x;
    const int64_t base = int64_t(blockIdx.x) * width;
    float sum = 0;
    for (int64_t i = t; i < width; i += THREADS) {
        float f = __half2float(x[base + i]);
        sum += f * f;
    }
    sums[t] = sum;
    __syncthreads();
    for (int stride = THREADS / 2; stride; stride /= 2) {
        if (t < stride) sums[t] += sums[t + stride];
        __syncthreads();
    }
    float inv = rsqrtf(sums[0] / width + eps);
    for (int64_t i = t; i < width; i += THREADS)
        out[base + i] = __float2half_rn((__half2float(x[base + i]) * inv) * w[i]);
}

__global__ void rope_kernel(const __half* x, const float* freq, const int64_t* pos, __half* out,
                            int64_t count, int heads, int width, int rotary) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += int64_t(gridDim.x) * blockDim.x) {
        int col = i % width;
        float value = __half2float(x[i]);
        if (col < rotary) {
            int half = rotary / 2;
            int pair = col % half;
            float angle = float(pos[i / (int64_t(heads) * width)]) * freq[pair];
            float s, c;
            sincosf(angle, &s, &c);
            int64_t other = col < half ? i + half : i - half;
            float cross = __half2float(x[other]);
            value = value * c + (col < half ? -cross : cross) * s;
        }
        out[i] = __float2half_rn(value);
    }
}
}  // namespace

at::Tensor sm70_rms_norm(const at::Tensor& x, const at::Tensor& w, double eps) {
    check_half(x);
    TORCH_CHECK(x.dim() == 2 && x.size(1) > 0 && x.size(1) <= std::numeric_limits<int>::max(),
                "RMSNorm expects [rows, positive width]");
    TORCH_CHECK(w.device() == x.device() && w.scalar_type() == at::kFloat && w.is_contiguous()
                && w.dim() == 1 && w.size(0) == x.size(1), "RMSNorm weight must be FP32 [width] on input device");
    TORCH_CHECK(std::isfinite(eps) && eps > 0 && float(eps) > 0 && std::isfinite(float(eps)),
                "RMSNorm epsilon must be positive finite FP32");
    TORCH_CHECK(x.size(0) <= at::cuda::getDeviceProperties(x.get_device())->maxGridSize[0], "too many norm rows");
    c10::cuda::CUDAGuard guard(x.device());
    auto out = at::empty_like(x);
    if (x.numel()) {
        norm_kernel<<<x.size(0), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __half*>(x.data_ptr()), w.data_ptr<float>(),
            reinterpret_cast<__half*>(out.data_ptr()), x.size(1), float(eps));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

at::Tensor sm70_rope(const at::Tensor& x, const at::Tensor& freq, const at::Tensor& pos) {
    check_half(x);
    TORCH_CHECK(x.dim() == 3 && x.size(1) > 0 && x.size(2) > 0 && x.size(2) <= std::numeric_limits<int>::max(),
                "RoPE expects [rows, heads, width]");
    TORCH_CHECK(freq.device() == x.device() && freq.scalar_type() == at::kFloat && freq.is_contiguous()
                && freq.dim() == 1 && freq.numel() > 0 && freq.numel() <= x.size(2) / 2,
                "RoPE inverse frequencies must be FP32 with 2*count <= width");
    TORCH_CHECK(pos.device() == x.device() && pos.scalar_type() == at::kLong && pos.is_contiguous()
                && pos.dim() == 1 && pos.size(0) == x.size(0), "RoPE positions must be CUDA int64 [rows]");
    TORCH_CHECK(x.size(1) <= std::numeric_limits<int>::max(), "too many RoPE heads");
    c10::cuda::CUDAGuard guard(x.device());
    auto out = at::empty_like(x);
    if (x.numel()) {
        const int blocks = int(std::min<int64_t>((x.numel() + THREADS - 1) / THREADS, 65535));
        rope_kernel<<<blocks, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __half*>(x.data_ptr()), freq.data_ptr<float>(), pos.data_ptr<int64_t>(),
            reinterpret_cast<__half*>(out.data_ptr()), x.numel(), x.size(1), x.size(2), freq.numel() * 2);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}
