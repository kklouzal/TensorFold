// Prefill: w = bf16(fma(q, s, b)), one fp32 chain over K; chunk-invariant bits, not decode's (replies prefill again).

#include "experts_prefill.cuh"

// Items of 16 pairs (experts_window.cu: its own translation unit, so its instantiations leave this one's code as is)
void experts_window_cuda(int64_t gs, int64_t epi, const at::Tensor& x, int64_t x_stride, int64_t slots,
                         const at::Tensor& w, int64_t kg, int64_t nb, const at::Tensor& items, const at::Tensor& counts,
                         const at::Tensor& members, at::Tensor& out, int64_t n, double limit, int64_t max_items,
                         bool pdl);

void experts_prefill_cuda(int64_t gs, int64_t epi, const at::Tensor& x, int64_t x_stride, int64_t slots,
                          const at::Tensor& w, int64_t kg, int64_t nb, const at::Tensor& items,
                          const at::Tensor& counts, const at::Tensor& members, at::Tensor& out, int64_t n,
                          double limit, int64_t max_items, int64_t tile, bool pdl) {
  if (tile == 16) {
    experts_window_cuda(gs, epi, x, x_stride, slots, w, kg, nb, items, counts, members, out, n, limit, max_items, pdl);
    return;
  }
  const c10::cuda::CUDAGuard guard(x.device());
  const int xs = static_cast<int>(x_stride), sl = static_cast<int>(slots), k = static_cast<int>(kg);
  const int b = static_cast<int>(nb), nn = static_cast<int>(n);
  const float lim = static_cast<float>(limit);
  // 8 warps a CTA, 64 pairs x 128 columns: two row tiles a warp, two warps down, four across
#define TF_PRE(GS_, M_, EPI_)                                                                                  \
  if (gs == GS_ && epi == EPI_) {                                                                              \
    launch_prefill<GS_, M_, EPI_, 2, 2, 4, 0>(x, xs, sl, w, k, b, items, counts, members, out, nn, lim,        \
                                              max_items, pdl);                                                 \
    return;                                                                                                    \
  }
  TF_PRE(32, 1, 0) TF_PRE(32, 1, 3) TF_PRE(32, 2, 2) TF_PRE(64, 1, 0) TF_PRE(64, 1, 3) TF_PRE(64, 1, 1)
  TF_PRE(64, 2, 2)
#undef TF_PRE
  TORCH_CHECK(false, "experts: no prefill kernel for group ", gs, " and epilogue ", epi);
}
