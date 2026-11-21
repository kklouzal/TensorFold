// SPDX-License-Identifier: Apache-2.0
// DeepSeek-V4.1 routed-expert prefill for SM70, TensorFold-owned.
//
// Same operands as onecat_ds41.v41_experts (original GGUF expert bytes of one EP/ETP rank) for any M.
// Per call: routes with probability 0 are dropped (they contribute exactly +/-0), the rest are
// stable-sorted by expert on the device, and the per-expert route counts reach the host once (one
// stream synchronization) to build the route tiles. Then two grouped GEMM launches decode the IQ rows
// inside their K loops (no FP16 weight bank): gate|up with the clamped SwiGLU times the route
// probability as epilogue (FP16 down input), and down over the rank's K window [left, left + 576)
// into per-route FP16 rows; the output is the FP32 sum of each token's routes in top-k order,
// rounded to FP16.
//
// Numerics versus the decode path: activations are FP16, not Q8_1; weights are ggml-quants.c FP32
// products rounded once to FP16; products accumulate in FP32, and gate/up stay FP32 into the SwiGLU
// (no FP16 boundary there); the other rounding points (FP16 down input, FP16 per-route down rows,
// FP32 route sum) are the reference's. Not graph-capturable (host synchronization, allocations).
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <algorithm>
#include <cstdint>
#include <vector>
#include "ds41_ggml_decode.cuh"

namespace {
using tensorfold::ds41_ggml::decode8;

constexpr int kHidden = 5120, kRows = 576, kTop = 6;

// out[t] = fp16(sum_j fp32(down[position[t * 6 + j]])), j ascending; position -1 = dropped route.
__global__ void combine_routes(half* out, const half* down, const int32_t* position, int64_t tokens) {
  constexpr int kChunks = kHidden / 8;
  for (int64_t at = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; at < tokens * kChunks;
       at += int64_t(gridDim.x) * blockDim.x) {
    const int64_t t = at / kChunks;
    const int c = int(at % kChunks) * 8;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int j = 0; j < kTop; ++j) {
      const int p = position[t * kTop + j];
      if (p < 0) continue;
      const uint4 raw = *reinterpret_cast<const uint4*>(down + int64_t(p) * kHidden + c);
      const half* v = reinterpret_cast<const half*>(&raw);
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[i] = __fadd_rn(acc[i], __half2float(v[i]));
    }
    __align__(16) half h[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) h[i] = __float2half_rn(acc[i]);
    *reinterpret_cast<uint4*>(out + t * kHidden + c) = *reinterpret_cast<const uint4*>(h);
  }
}

// ---- Fused grouped GEMM: IQ rows decoded into shared memory inside the K loop (no FP16 weight bank) ----
// One CTA = 64 sorted routes of one expert x 128 B rows, 256 threads; warps 2 (M) x 4 (N), each 32 x 32 of WMMA 16x16x16 FP16 products with FP32 accumulators. The decoded values are exactly
// the dequant kernel's (ggml FP32 products rounded once to FP16). K steps of 64: each thread fetches 2
// activation pieces and decodes 4 weight pieces of 8 values (independent loads in flight; the next
// step's are issued before this step's products), publishes them to shared memory, and the warps run
// the step's products. Measured alternatives (no gain, so not kept): 128-route tiles of 512 threads
// (slower below ~8K tokens, equal at 8K) and staging each GGML block in shared memory before decoding
// (within 2%).
//   Gated (gate|up): B rows 0..63 = gate rows c0..c0+63, rows 64..127 = the same up rows; warp n
//   covers gate and up columns 16n..16n+15, so its gate/up accumulators match element for element.
//   A rows are x[token[p]]. Epilogue: hidden[p][c0 + j] = fp16((silu(min(g,10)) * clamp(u)) * w[p]).
//   Down: B rows = down rows n0..n0+127, K window [left, left + 576); A rows = hidden[p];
//   epilogue: down_rows[p][n0 + j] = fp16(acc).
struct Work {
  int32_t expert, begin, rows, unused;
};
constexpr int kBM = 64, kThreads = 256, kBN = 128, kBK = 64, kPad = 8, kLd = kBK + kPad, kPieces = kBK / 8;
constexpr int kStage = 32 + 4;  // per-warp FP32 staging row stride (32 columns + pad)
static_assert(kHidden % kBK == 0 && kRows % kBK == 0 && 256 % kBK == 0);
using Accumulator = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>;

// One warp's 32 x 32 results through its FP32 staging tile [32][kStage].
template <bool Gated>
__device__ __forceinline__ void epilogue(Accumulator (&acc)[2][2], float* stage, const Work& job, int wm, int wn,
                                         int lane, int col0, const float* weights, half* out, int out_ld) {
  namespace wmma = nvcuda::wmma;
  if constexpr (Gated) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
      for (int e = 0; e < acc[i][0].num_elements; ++e) {
        float g = acc[i][0].x[e], u = acc[i][1].x[e];
        g = g > 10.f ? 10.f : g;
        u = u > 10.f ? 10.f : u;
        u = u < -10.f ? -10.f : u;
        acc[i][0].x[e] = __fmul_rn(__fdiv_rn(g, __fadd_rn(1.f, expf(-g))), u);
      }
      wmma::store_matrix_sync(stage + 16 * i * kStage, acc[i][0], kStage, wmma::mem_row_major);
    }
    __syncwarp();
    for (int idx = lane; idx < 32 * 16; idx += 32) {
      const int r = idx / 16, c = idx % 16, row = 32 * wm + r;
      if (row < job.rows) {
        const int64_t p = job.begin + row;
        out[p * out_ld + col0 + 16 * wn + c] = __float2half_rn(__fmul_rn(stage[r * kStage + c], weights[p]));
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 2; ++j)
        wmma::store_matrix_sync(stage + 16 * i * kStage + 16 * j, acc[i][j], kStage, wmma::mem_row_major);
    __syncwarp();
    for (int idx = lane; idx < 32 * 32; idx += 32) {
      const int r = idx / 32, c = idx % 32, row = 32 * wm + r;
      if (row < job.rows)
        out[int64_t(job.begin + row) * out_ld + col0 + 32 * wn + c] = __float2half_rn(stage[r * kStage + c]);
    }
  }
}

template <int Type, int Block, int Bytes, bool Gated>
__global__ void __launch_bounds__(kThreads, 2)
grouped_iq_gemm(const half* a_src, const int64_t* a_rows, const uint8_t* bank0, const uint8_t* bank1,
                int64_t expert_bytes, int64_t row_bytes, const Work* work, int k_total, int k_first,
                const float* weights, half* out, int out_ld) {
  namespace wmma = nvcuda::wmma;
  constexpr int kRowsPerPass = kThreads / kPieces, kAPieces = kBM / kRowsPerPass, kBPieces = kBN / kRowsPerPass;
  static_assert(kBN / 2 % kRowsPerPass == 0);
  constexpr int kLoopBytes = (kBM + kBN) * kLd * 2, kStageBytes = 8 * 32 * kStage * 4;
  __shared__ __align__(128) unsigned char smem[kLoopBytes > kStageBytes ? kLoopBytes : kStageBytes];
  half* sa = reinterpret_cast<half*>(smem);  // [kBM][kLd]
  half* sb = sa + kBM * kLd;                   // [kBN][kLd]
  const Work job = work[blockIdx.y];
  const int tid = threadIdx.x, warp = tid / 32, wm = warp / 4, wn = warp % 4;
  const int col0 = blockIdx.x * (Gated ? kBN / 2 : kBN);
  const int row0 = tid / kPieces, piece = 8 * (tid % kPieces);  // this thread's rows row0 + kRowsPerPass i

  const half* a_ptr[kAPieces];
#pragma unroll
  for (int i = 0; i < kAPieces; ++i) {
    const int r = row0 + kRowsPerPass * i;
    a_ptr[i] = nullptr;
    if (r < job.rows) {
      const int64_t p = job.begin + r;
      a_ptr[i] = a_src + (a_rows ? a_rows[p] : p) * int64_t(k_total) + piece;
    }
  }
  const uint8_t* b_ptr[kBPieces];
#pragma unroll
  for (int i = 0; i < kBPieces; ++i) {
    const int r = row0 + kRowsPerPass * i;  // B tile row
    const bool second = Gated && r >= kBN / 2;
    const int n = col0 + (second ? r - kBN / 2 : r);
    b_ptr[i] = (second ? bank1 : bank0) + int64_t(job.expert) * expert_bytes + int64_t(n) * row_bytes;
  }
  uint4 a_reg[kAPieces], b_reg[kBPieces];
  auto fetch = [&](int k) {
#pragma unroll
    for (int i = 0; i < kAPieces; ++i)
      a_reg[i] = a_ptr[i] ? *reinterpret_cast<const uint4*>(a_ptr[i] + k) : make_uint4(0, 0, 0, 0);
    const int kk = k_first + k + piece;
#pragma unroll
    for (int i = 0; i < kBPieces; ++i) {
      float y[8];
      decode8<Type>(b_ptr[i] + int64_t(kk / Block) * Bytes, (kk % Block) / 8, y);
      __align__(16) half h[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) h[j] = __float2half_rn(y[j]);
      b_reg[i] = *reinterpret_cast<const uint4*>(h);
    }
  };
  auto publish = [&]() {
#pragma unroll
    for (int i = 0; i < kAPieces; ++i)
      *reinterpret_cast<uint4*>(sa + (row0 + kRowsPerPass * i) * kLd + piece) = a_reg[i];
#pragma unroll
    for (int i = 0; i < kBPieces; ++i)
      *reinterpret_cast<uint4*>(sb + (row0 + kRowsPerPass * i) * kLd + piece) = b_reg[i];
  };

  Accumulator acc[2][2];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.f);
  // B fragment j of warp n: Gated -> rows 16n (gate) and 64 + 16n (up); down -> rows 32n + 16j.
  const int b_row[2] = {Gated ? 16 * wn : 32 * wn, Gated ? kBN / 2 + 16 * wn : 32 * wn + 16};

  const int steps = k_total / kBK;
  fetch(0);
  for (int step = 0; step < steps; ++step) {
    publish();
    __syncthreads();
    if (step + 1 < steps) fetch((step + 1) * kBK);
#pragma unroll
    for (int kk = 0; kk < kBK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa[2];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb[2];
#pragma unroll
      for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(fa[i], sa + (32 * wm + 16 * i) * kLd + kk, kLd);
#pragma unroll
      for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(fb[j], sb + b_row[j] * kLd + kk, kLd);
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
    }
    __syncthreads();
  }
  // The loop buffers are dead: each warp's staging tile.
  epilogue<Gated>(acc, reinterpret_cast<float*>(smem) + warp * 32 * kStage, job, wm, wn, tid % 32, col0, weights,
                  out, out_ld);
}

template <bool Gated>
void launch_grouped(int64_t type, const half* a_src, const int64_t* a_rows, const torch::Tensor& bank0,
                    const torch::Tensor* bank1, const Work* work, int jobs, int k_total, int k_first,
                    const float* weights, half* out, int out_ld, int n_tiles, cudaStream_t stream) {
  const int64_t row_bytes = bank0.size(2), expert_bytes = bank0.size(1) * row_bytes;
  const uint8_t* b0 = bank0.data_ptr<uint8_t>();
  const uint8_t* b1 = bank1 ? bank1->data_ptr<uint8_t>() : nullptr;
  const dim3 grid(n_tiles, jobs);
#define DS41_GROUPED(T, B, S)                                                                             \
  grouped_iq_gemm<T, B, S, Gated><<<grid, kThreads, 0, stream>>>(a_src, a_rows, b0, b1, expert_bytes,  \
                                                                   row_bytes, work, k_total, k_first,   \
                                                                   weights, out, out_ld)
  switch (type) {
    case 16: DS41_GROUPED(16, 256, 66); break;
    case 17: DS41_GROUPED(17, 256, 74); break;
    default: DS41_GROUPED(29, 256, 56); break;
  }
#undef DS41_GROUPED
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_experts(const torch::Tensor& x, const torch::Tensor& gate, const torch::Tensor& up,
                   const torch::Tensor& down, int64_t gate_type, int64_t down_type, int64_t left,
                   const std::vector<int32_t>& used, const std::vector<int64_t>& first, int64_t routes,
                   const torch::Tensor& tokens, const torch::Tensor& weights, cudaStream_t stream,
                   half* rows_out) {
  std::vector<Work> jobs;
  for (size_t i = 0; i < used.size(); ++i)
    for (int64_t at = first[i]; at < first[i + 1]; at += kBM)
      jobs.push_back({used[i], int32_t(at), int32_t(std::min<int64_t>(kBM, first[i + 1] - at)), 0});
  TORCH_CHECK(jobs.size() <= 65535 && routes <= INT32_MAX, "V4.1 prefill MoE: too many route tiles");
  const auto jobs_dev =
      torch::from_blob(jobs.data(), {int64_t(jobs.size() * sizeof(Work))}, torch::kUInt8).to(x.device());
  const auto* work = reinterpret_cast<const Work*>(jobs_dev.data_ptr());
  auto hidden = torch::empty({routes, kRows}, x.options());
  auto* hs = reinterpret_cast<half*>(hidden.data_ptr());
  launch_grouped<true>(gate_type, reinterpret_cast<const half*>(x.data_ptr()), tokens.data_ptr<int64_t>(),
                           gate, &up, work, int(jobs.size()), kHidden, 0, weights.data_ptr<float>(), hs, kRows,
                           kRows / (kBN / 2), stream);
  launch_grouped<false>(down_type, hs, nullptr, down, nullptr, work, int(jobs.size()), kRows, int(left),
                            nullptr, rows_out, kHidden, kHidden / kBN, stream);
}

int grid_for(int64_t work) {
  return int(std::max<int64_t>(1, std::min<int64_t>((work + 255) / 256, 65535 * 8)));
}

int block_bytes(int64_t type) { return type == 16 ? 66 : type == 17 ? 74 : 56; }

void check_bank(const torch::Tensor& bank, const torch::Tensor& x, int64_t rows, int64_t cols, int64_t type,
                const char* what) {
  const int64_t bytes = cols / 256 * block_bytes(type);
  TORCH_CHECK(bank.device() == x.device() && bank.scalar_type() == torch::kUInt8 && bank.is_contiguous() &&
                  bank.dim() == 3 && bank.size(0) > 0 && bank.size(1) == rows &&
                  (bank.size(2) == bytes || bank.size(2) == (bytes + 7) / 8 * 8),
              "V4.1 prefill MoE ", what, " requires original expert rows uint8 [E, ", rows, ", ", bytes, "]");
}
}  // namespace
void ds41_prefill_experts(torch::Tensor out, const torch::Tensor& x, const torch::Tensor& ids,
                          const torch::Tensor& probabilities, const torch::Tensor& gate, const torch::Tensor& up,
                          const torch::Tensor& down, int64_t gate_type, int64_t down_type, int64_t left) {
  TORCH_CHECK(x.is_cuda(), "V4.1 prefill MoE requires CUDA tensors");
  const auto* props = at::cuda::getDeviceProperties(x.get_device());
  TORCH_CHECK(props->major == 7 && props->minor == 0, "V4.1 prefill MoE requires SM70");
  TORCH_CHECK((gate_type == 16 || gate_type == 17 || gate_type == 29) &&
                  (down_type == 16 || down_type == 17 || down_type == 29),
              "V4.1 prefill MoE requires IQ2_XXS(16)/IQ2_XS(17)/IQ1_M(29) experts");
  TORCH_CHECK(left >= 0 && left % 64 == 0 && left + kRows <= 3 * 256, "V4.1 prefill MoE down window");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16 && x.is_contiguous() && x.dim() == 2 && x.size(0) >= 1 &&
                  x.size(1) == kHidden && x.size(0) <= (int64_t{1} << 28),
              "V4.1 prefill MoE requires FP16 x [M, 5120]");
  const int64_t m = x.size(0);
  TORCH_CHECK(ids.device() == x.device() && ids.scalar_type() == torch::kInt32 && ids.is_contiguous() &&
                  ids.dim() == 2 && ids.size(0) == m && ids.size(1) == kTop &&
                  probabilities.device() == x.device() && probabilities.scalar_type() == torch::kFloat32 &&
                  probabilities.is_contiguous() && probabilities.sizes() == ids.sizes(),
              "V4.1 prefill MoE requires int32 ids and FP32 probabilities [M, 6]");
  TORCH_CHECK(out.device() == x.device() && out.scalar_type() == torch::kFloat16 && out.is_contiguous() &&
                  out.sizes() == x.sizes() && !out.is_same(x),
              "V4.1 prefill MoE requires a separate FP16 out [M, 5120]");
  check_bank(gate, x, kRows, kHidden, gate_type, "gate");
  check_bank(up, x, kRows, kHidden, gate_type, "up");
  check_bank(down, x, kHidden, 3 * 256, down_type, "down");
  TORCH_CHECK(up.size(0) == gate.size(0) && down.size(0) == gate.size(0) && gate.sizes() == up.sizes(),
              "V4.1 prefill MoE banks must hold the same experts");
  const int64_t experts = gate.size(0);
  TORCH_CHECK(experts <= INT32_MAX - 1, "V4.1 prefill MoE expert count");
  const c10::cuda::CUDAGuard guard(x.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  cudaStreamCaptureStatus capture;
  C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
  TORCH_CHECK(capture == cudaStreamCaptureStatusNone, "V4.1 prefill MoE cannot be graph-captured");

  // Routes: probability-0 routes go to bucket E (dropped), routed ids outside [0, E) to bucket E + 1
  // (rejected after the one synchronization); the rest are stable-sorted by expert.
  const auto flat_w = probabilities.reshape({-1});
  const auto flat_ids = ids.reshape({-1});
  const auto live = flat_w != 0;
  const auto invalid = live & ((flat_ids < 0) | (flat_ids >= experts));
  auto bucket = torch::where(live, flat_ids, torch::full_like(flat_ids, int32_t(experts)));
  bucket.masked_fill_(invalid, int32_t(experts + 1));
  const auto counts_dev = torch::bincount(bucket.to(torch::kInt64), {}, experts + 2);
  const auto order = std::get<1>(torch::sort(bucket, /*stable=*/true, 0, false));
  const auto counts = counts_dev.cpu();  // the one host synchronization
  TORCH_CHECK(counts.size(0) == experts + 2 && counts.data_ptr<int64_t>()[experts + 1] == 0,
              "V4.1 prefill MoE: a routed expert id is outside [0, ", experts, ")");
  const auto* count = counts.data_ptr<int64_t>();
  std::vector<int32_t> used;
  std::vector<int64_t> first;  // first sorted position of each used expert, then the route total
  int64_t routes = 0;
  for (int64_t e = 0; e < experts; ++e) {
    if (count[e]) {
      used.push_back(int32_t(e));
      first.push_back(routes);
      routes += count[e];
    }
  }
  first.push_back(routes);
  if (!routes) {
    out.zero_();
    return;
  }
  const auto routed = order.slice(0, 0, routes);
  const auto tokens = torch::floor_divide(routed, kTop);
  const auto weights = flat_w.index_select(0, routed);
  auto position = torch::full({m * kTop}, -1, ids.options());
  position.index_put_({routed}, torch::arange(routes, ids.options()));

  auto down_rows = torch::empty({routes, kHidden}, x.options());
  auto* rows_out = reinterpret_cast<half*>(down_rows.data_ptr());
  fused_experts(x, gate, up, down, gate_type, down_type, left, used, first, routes, tokens, weights, stream,
                rows_out);
  combine_routes<<<grid_for(m * (kHidden / 8)), 256, 0, stream>>>(
      reinterpret_cast<half*>(out.data_ptr()), rows_out, position.data_ptr<int32_t>(), m);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
