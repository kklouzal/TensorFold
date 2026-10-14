// Prefill form on items of 16 pairs (a decode window's plan where chosen, ``experts.window_tile``): one 16-row tile a
// CTA, four warps across and two stages for one matrix, eight across and three stages for two, so more CTAs an SM keep
// weights in flight than the 64-pair tiles (measured on dev-01). A pair's arithmetic is the prefill form's (one fp32
// mma chain over K in group order; mma rows are independent): the same bits as items of 64.

#include "experts_prefill.cuh"

void experts_window_cuda(int64_t gs, int64_t epi, const at::Tensor& x, int64_t x_stride, int64_t slots,
                         const at::Tensor& w, int64_t kg, int64_t nb, const at::Tensor& items, const at::Tensor& counts,
                         const at::Tensor& members, at::Tensor& out, int64_t n, double limit, int64_t max_items,
                         bool pdl) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int xs = static_cast<int>(x_stride), sl = static_cast<int>(slots), k = static_cast<int>(kg);
  const int b = static_cast<int>(nb), nn = static_cast<int>(n);
  const float lim = static_cast<float>(limit);
#define TF_WIN(GS_, M_, EPI_, WN_, ST_)                                                                        \
  if (gs == GS_ && epi == EPI_) {                                                                              \
    launch_prefill<GS_, M_, EPI_, 1, 1, WN_, ST_>(x, xs, sl, w, k, b, items, counts, members, out, nn, lim,    \
                                                  max_items, pdl);                                             \
    return;                                                                                                    \
  }
  TF_WIN(32, 1, 0, 4, 2) TF_WIN(32, 1, 3, 4, 2) TF_WIN(32, 2, 2, 8, 3) TF_WIN(64, 1, 0, 4, 2) TF_WIN(64, 1, 3, 4, 2)
  TF_WIN(64, 1, 1, 4, 2) TF_WIN(64, 2, 2, 8, 3)
#undef TF_WIN
  TORCH_CHECK(false, "experts: no 16-pair prefill kernel for group ", gs, " and epilogue ", epi);
}
