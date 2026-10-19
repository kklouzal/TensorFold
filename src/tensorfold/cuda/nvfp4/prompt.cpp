#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void prompt16_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, double, const at::Tensor&, at::Tensor&,
                   int64_t, int64_t, int64_t, int64_t);

// out (M, n) = x (M, K) bf16 @ W for prompt rows: mode 0 NVFP4 (tiled words, e4m3 block scales [npad/64, K/64, 64, 4]),
// 1 FP8 (fragment-order bytes), 2 MXFP8 (those with e8m0 scales [npad/64, K/64, 64, 2]); ``scale`` the tensor factor,
// or for FP8 ``cs`` (fp32 [npad]) a factor an output channel in its place.
void prompt16(const at::Tensor& x, const at::Tensor& w, const c10::optional<at::Tensor>& bs, double scale,
              const c10::optional<at::Tensor>& cs, at::Tensor out, int64_t mode, int64_t n, int64_t npad,
              int64_t tile) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.stride(1) == 1 && x.size(0) >= 1,
                "x: (M, K) bf16 with contiguous rows");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && (x.size(0) == 1 || x.stride(0) % 8 == 0),
                "x rows must start on 16-byte boundaries");
    TORCH_CHECK(mode >= 0 && mode <= 2 && tile >= 0 && tile <= 5, "mode 0-2, tile 0-5");
    const int64_t m = x.size(0), k = x.size(1);
    TORCH_CHECK(k % 64 == 0 && npad % 128 == 0 && n <= npad, "K in groups of 64, n padded to 128");
    TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.numel() * w.element_size() == npad * k / (mode == 0 ? 2 : 1),
                "weight bytes do not match n and K");
    TORCH_CHECK(mode == 1 || (bs.has_value() && bs->is_contiguous() &&
                              bs->numel() == (k / 64) * npad * (mode == 0 ? 4 : 2)), "block scales [npad/64, K/64, 64, 4|2]");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.size(0) == m && out.size(1) == n &&
                out.scalar_type() == at::kBFloat16, "out: (M, n) bf16");
    TORCH_CHECK(!cs.has_value() || (mode == 1 && cs->device() == x.device() && cs->is_contiguous() &&
                                    cs->scalar_type() == at::kFloat && cs->numel() == npad),
                "cs: FP8's fp32 [npad] column scales");
    c10::cuda::CUDAGuard guard(x.device());
    prompt16_cuda(x, w, bs.has_value() ? *bs : at::Tensor(), scale, cs.has_value() ? *cs : at::Tensor(), out, mode, n,
                  npad, tile);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prompt16", &prompt16);
}
