#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <limits>

void prefill_attention_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, int, float, int);

// Causal attention of (W, H, D) bf16 queries at positions [p0, p0 + W) over caches filled through p0 + W - 1.
void prefill_attention(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, at::Tensor& out, int64_t p0,
                       double scale, int64_t hpc) {
    TORCH_CHECK(q.is_cuda() && q.is_contiguous() && q.scalar_type() == at::kBFloat16 && q.dim() == 3 &&
                q.size(1) > 0 && (q.size(2) == 128 || q.size(2) == 256),
                "q: contiguous CUDA (W, positive H, 128 or 256) bf16");
    TORCH_CHECK(k.device() == q.device() && v.device() == q.device() &&
                k.is_contiguous() && v.is_contiguous() && k.scalar_type() == at::kBFloat16 &&
                v.scalar_type() == at::kBFloat16 && k.dim() == 3 && k.sizes() == v.sizes() &&
                k.size(1) > 0 && k.size(2) == q.size(2) && q.size(1) % k.size(1) == 0,
                "caches: contiguous bf16 (keys, positive HK, D) on q's CUDA device; HK divides H");
    // The kernel uses signed-int rows/positions, including padding a final 64-key tile and 128-row CTA.
    constexpr int64_t limit = std::numeric_limits<int>::max();
    TORCH_CHECK(q.size(0) <= limit - 127 && q.size(1) <= limit && k.size(1) <= limit &&
                p0 >= 0 && p0 <= limit - 63 - q.size(0), "prompt attention position/dimensions exceed int32");
    TORCH_CHECK(k.size(0) >= p0 + q.size(0), "caches must hold the chunk's keys");
    TORCH_CHECK(out.device() == q.device() && out.is_contiguous() && out.sizes() == q.sizes() &&
                out.scalar_type() == at::kBFloat16, "out: like q on q's CUDA device");
    TORCH_CHECK(hpc == 1 || hpc == 2 || hpc == 4 || hpc == 8, "heads a block must be 1, 2, 4 or 8");
    TORCH_CHECK((q.size(1) / k.size(1)) % hpc == 0, "heads a block must divide the query heads a KV head");
    c10::cuda::CUDAGuard guard(q.device());
    TORCH_CHECK(q.size(1) / hpc <= at::cuda::getDeviceProperties(q.get_device())->maxGridSize[1],
                "prompt attention head grid exceeds this CUDA device's limit");
    prefill_attention_cuda(q, k, v, out, static_cast<int>(p0), static_cast<float>(scale), static_cast<int>(hpc));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("prefill_attention", &prefill_attention); }
