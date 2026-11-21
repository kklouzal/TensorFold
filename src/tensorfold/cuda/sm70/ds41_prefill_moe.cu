// SPDX-License-Identifier: Apache-2.0
// DeepSeek-V4.1 routed-expert prefill for SM70, TensorFold-owned.
//
// Same operands as onecat_ds41.v41_experts (original GGUF expert bytes of one
// EP/ETP rank) for any M. Per call: routes with probability 0 are dropped (they
// contribute exactly +/-0), the rest are sorted by expert on the device, and the
// per-expert route counts reach the host once (one stream synchronization).
// Then per tile of routed experts: the gate|up and down rows (down: only the
// rank's K window [left, left + 576) of its 3-block carrier) are dequantized to
// FP16 (ggml-quants.c FP32 products rounded once), the token rows are gathered,
// and each expert runs two cuBLAS tensor-op GEMMs with FP32 accumulation:
// gate|up into FP32, the clamped SwiGLU times the route probability rounded to
// FP16 (the down input), down into per-route FP16 rows with FP32 reductions
// (CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION). The output is the FP32
// sum of each token's routes in top-k order, rounded to FP16.
//
// Numerics versus the decode path: activations are FP16, not Q8_1, and gate/up
// stay FP32 into the SwiGLU (no FP16 boundary there); the other rounding
// points (FP16 down input, FP16 per-route down rows, FP32 route sum) are the
// reference's. Not graph-capturable (host synchronization, allocations).
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <algorithm>
#include <cstdint>
#include <vector>
#include "ds41_ggml_decode.cuh"

namespace {
using tensorfold::ds41_ggml::decode8;

constexpr int kHidden = 5120, kRows = 576, kTop = 6;

// out[slot][row][c] = fp16(value of column col0 + c of row `row` of expert experts[slot]).
template <int Type, int Block, int Bytes>
__global__ void dequant_experts(const uint8_t* bank, int64_t expert_bytes, int64_t row_bytes,
                                const int32_t* experts, int slots, int rows, int col0, int cols,
                                half* out, int64_t out_expert, int64_t out_row) {
  const int64_t per_row = cols / 8, per_slot = rows * per_row;
  for (int64_t at = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; at < slots * per_slot;
       at += int64_t(gridDim.x) * blockDim.x) {
    const int slot = int(at / per_slot);
    const int64_t rest = at % per_slot;
    const int row = int(rest / per_row);
    const int c = int(rest % per_row) * 8;
    const int col = col0 + c;
    float y[8];
    decode8<Type>(bank + experts[slot] * expert_bytes + row * row_bytes + int64_t(col / Block) * Bytes,
                  (col % Block) / 8, y);
    __align__(16) half h[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) h[j] = __float2half_rn(y[j]);
    *reinterpret_cast<uint4*>(out + slot * out_expert + row * out_row + c) = *reinterpret_cast<const uint4*>(h);
  }
}

// h[p][c] = fp16((silu(min(g, 10)) * clamp(u, -10, 10)) * w[p]), g/u FP32 from proj [p][c] / [p][576 + c].
__global__ void routed_swiglu(half* hidden, const float* proj, const float* weights, int64_t routes) {
  for (int64_t at = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; at < routes * kRows;
       at += int64_t(gridDim.x) * blockDim.x) {
    const int64_t p = at / kRows;
    const int c = int(at % kRows);
    float gate = proj[p * (2 * kRows) + c], up = proj[p * (2 * kRows) + kRows + c];
    gate = gate > 10.f ? 10.f : gate;
    up = up > 10.f ? 10.f : up;
    up = up < -10.f ? -10.f : up;
    const float activated = __fdiv_rn(gate, __fadd_rn(1.f, expf(-gate)));
    hidden[at] = __float2half_rn(__fmul_rn(__fmul_rn(activated, up), weights[p]));
  }
}

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
// One CTA = 64 sorted routes of one expert x 128 B rows; 8 warps as 2 (M) x 4 (N), each 32 x 32 of
// WMMA 16x16x16 FP16 products with FP32 accumulators. The decoded values are exactly the dequant
// kernel's (ggml FP32 products rounded once to FP16). Double-buffered: the next K step's activations
// and decoded weights are fetched into registers while the current step's products run.
//   Gated (gate|up): B rows 0..63 = gate rows c0..c0+63, rows 64..127 = the same up rows; warp n
//   covers gate and up columns 16n..16n+15, so its gate/up accumulators match element for element.
//   A rows are x[token[p]]. Epilogue: hidden[p][c0 + j] = fp16((silu(min(g,10)) * clamp(u)) * w[p]).
//   Down: B rows = down rows n0..n0+127, K window [left, left + 576); A rows = hidden[p];
//   epilogue: down_rows[p][n0 + j] = fp16(acc).
struct Work {
  int32_t expert, begin, rows, unused;
};
constexpr int kBM = 64, kBN = 128, kBK = 32, kPad = 8, kLd = kBK + kPad, kThreads = 256;
constexpr int kStage = 32 + 4;  // per-warp FP32 staging row stride (32 columns + pad)

template <int Type, int Block, int Bytes, bool Gated>
__global__ void __launch_bounds__(kThreads, 2)
grouped_iq_gemm(const half* a_src, const int64_t* a_rows, const uint8_t* bank0, const uint8_t* bank1,
                int64_t expert_bytes, int64_t row_bytes, const Work* work, int k_total, int k_first,
                const float* weights, half* out, int out_ld) {
  namespace wmma = nvcuda::wmma;
  constexpr int kLoopHalves = 2 * (kBM + kBN) * kLd;
  constexpr int kStageFloats = 8 * 32 * kStage;
  constexpr int kBytes = (kLoopHalves * 2 > kStageFloats * 4) ? kLoopHalves * 2 : kStageFloats * 4;
  __shared__ __align__(128) unsigned char smem[kBytes];
  half* sa = reinterpret_cast<half*>(smem);           // [2][kBM][kLd]
  half* sb = sa + 2 * kBM * kLd;                       // [2][kBN][kLd]
  const Work job = work[blockIdx.y];
  const int tid = threadIdx.x, warp = tid / 32, wm = warp / 4, wn = warp % 4;
  const int col0 = blockIdx.x * (Gated ? kBN / 2 : kBN);

  // This thread's A piece: row tid / 4, halves 8 * (tid % 4) of the K step.
  const int a_row = tid / 4, a_col = 8 * (tid % 4);
  const half* a_ptr = nullptr;
  if (a_row < job.rows) {
    const int64_t p = job.begin + a_row;
    a_ptr = a_src + (a_rows ? a_rows[p] : p) * int64_t(k_total) + a_col;
  }
  // This thread's two B pieces: rows tid / 4 and 64 + tid / 4, values 8 * (tid % 4) of the K step.
  const uint8_t* b_ptr[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int r = tid / 4 + 64 * i;
    const uint8_t* bank = (Gated && i == 1) ? bank1 : bank0;
    const int n = Gated ? col0 + tid / 4 : col0 + r;
    b_ptr[i] = bank + int64_t(job.expert) * expert_bytes + int64_t(n) * row_bytes;
  }
  uint4 a_reg;
  uint4 b_reg[2];
  auto fetch = [&](int k) {
    a_reg = a_ptr ? *reinterpret_cast<const uint4*>(a_ptr + k) : make_uint4(0, 0, 0, 0);
    const int kk = k_first + k + a_col;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      float y[8];
      decode8<Type>(b_ptr[i] + int64_t(kk / Block) * Bytes, (kk % Block) / 8, y);
      __align__(16) half h[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) h[j] = __float2half_rn(y[j]);
      b_reg[i] = *reinterpret_cast<const uint4*>(h);
    }
  };
  auto publish = [&](int buffer) {
    *reinterpret_cast<uint4*>(sa + (buffer * kBM + a_row) * kLd + a_col) = a_reg;
#pragma unroll
    for (int i = 0; i < 2; ++i)
      *reinterpret_cast<uint4*>(sb + (buffer * kBN + tid / 4 + 64 * i) * kLd + a_col) = b_reg[i];
  };

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.f);
  // B fragment j of warp n: Gated -> rows 16n (gate) and 64 + 16n (up); down -> rows 32n + 16j.
  const int b_row[2] = {Gated ? 16 * wn : 32 * wn, Gated ? 64 + 16 * wn : 32 * wn + 16};

  fetch(0);
  publish(0);
  __syncthreads();
  const int steps = k_total / kBK;
  for (int step = 0; step < steps; ++step) {
    const int buffer = step & 1;
    if (step + 1 < steps) fetch((step + 1) * kBK);
#pragma unroll
    for (int kk = 0; kk < kBK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa[2];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb[2];
#pragma unroll
      for (int i = 0; i < 2; ++i)
        wmma::load_matrix_sync(fa[i], sa + (buffer * kBM + 32 * wm + 16 * i) * kLd + kk, kLd);
#pragma unroll
      for (int j = 0; j < 2; ++j)
        wmma::load_matrix_sync(fb[j], sb + (buffer * kBN + b_row[j]) * kLd + kk, kLd);
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
    }
    if (step + 1 < steps) publish(buffer ^ 1);
    __syncthreads();
  }

  // Epilogue through a per-warp FP32 staging tile [32][kStage] (the loop buffers are dead).
  float* stage = reinterpret_cast<float*>(smem) + warp * 32 * kStage;
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
    const int lane = tid % 32;
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
    const int lane = tid % 32;
    for (int idx = lane; idx < 32 * 32; idx += 32) {
      const int r = idx / 32, c = idx % 32, row = 32 * wm + r;
      if (row < job.rows)
        out[int64_t(job.begin + row) * out_ld + col0 + 32 * wn + c] = __float2half_rn(stage[r * kStage + c]);
    }
  }
}

template <bool Gated>
void launch_grouped(int64_t type, const half* a_src, const int64_t* a_rows, const torch::Tensor& bank0,
                    const torch::Tensor* bank1, const Work* work, int jobs, int k_total, int k_first,
                    const float* weights, half* out, int out_ld, int n_tiles, cudaStream_t stream) {
  const int64_t row_bytes = bank0.size(2), expert_bytes = bank0.size(1) * row_bytes;
  const uint8_t* b0 = bank0.data_ptr<uint8_t>();
  const uint8_t* b1 = bank1 ? bank1->data_ptr<uint8_t>() : nullptr;
  const dim3 grid(n_tiles, jobs);
#define DS41_GROUPED(T, B, S)                                                                              \
  grouped_iq_gemm<T, B, S, Gated><<<grid, kThreads, 0, stream>>>(a_src, a_rows, b0, b1, expert_bytes,     \
                                                                 row_bytes, work, k_total, k_first,      \
                                                                 weights, out, out_ld)
  switch (type) {
    case 16: DS41_GROUPED(16, 256, 66); break;
    case 17: DS41_GROUPED(17, 256, 74); break;
    default: DS41_GROUPED(29, 256, 56); break;
  }
#undef DS41_GROUPED
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int grid_for(int64_t work) {
  return int(std::max<int64_t>(1, std::min<int64_t>((work + 255) / 256, 65535 * 8)));
}

void dequant(int64_t type, const torch::Tensor& bank, const int32_t* experts, int slots, int rows,
             int col0, int cols, half* out, int64_t out_expert, int64_t out_row, cudaStream_t stream) {
  const int64_t row_bytes = bank.size(2), expert_bytes = bank.size(1) * row_bytes;
  const auto* source = bank.data_ptr<uint8_t>();
  const int grid = grid_for(int64_t(slots) * rows * (cols / 8));
  switch (type) {
    case 16: dequant_experts<16, 256, 66><<<grid, 256, 0, stream>>>(source, expert_bytes, row_bytes, experts,
                 slots, rows, col0, cols, out, out_expert, out_row); break;
    case 17: dequant_experts<17, 256, 74><<<grid, 256, 0, stream>>>(source, expert_bytes, row_bytes, experts,
                 slots, rows, col0, cols, out, out_expert, out_row); break;
    default: dequant_experts<29, 256, 56><<<grid, 256, 0, stream>>>(source, expert_bytes, row_bytes, experts,
                 slots, rows, col0, cols, out, out_expert, out_row); break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void blas(cublasStatus_t status) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "DeepSeek-V4.1 prefill MoE cuBLAS failure ", int(status));
}

// Row-major C[rows, n] = A[rows, k] * W[n, k]^T is column-major C^T = op_T(W) * A^T.
void gemm(cublasHandle_t handle, int n, int rows, int k, const half* w, int ldw, const half* a, int lda,
          void* c, cudaDataType_t c_type, int ldc) {
  const float one = 1.f, zero = 0.f;
  blas(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, rows, k, &one, w, CUDA_R_16F, ldw, a, CUDA_R_16F, lda,
                    &zero, c, c_type, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// Sets FP32 reductions on torch's handle; restores its previous mode.
struct MathMode {
  cublasHandle_t handle;
  cublasMath_t saved;
  bool active = true;
  explicit MathMode(cublasHandle_t h) : handle(h) {
    blas(cublasGetMathMode(handle, &saved));
    blas(cublasSetMathMode(handle, cublasMath_t(CUBLAS_DEFAULT_MATH |
                                                CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
  }
  void restore() {
    active = false;
    blas(cublasSetMathMode(handle, saved));
  }
  // Exception path only: the primary failure propagates; torch resets the mode at its next handle use.
  ~MathMode() {
    if (active) cublasSetMathMode(handle, saved);
  }
};

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
                          const torch::Tensor& down, int64_t gate_type, int64_t down_type, int64_t left,
                          int64_t tile, int64_t route_cap, bool fused) {
  TORCH_CHECK(x.is_cuda(), "V4.1 prefill MoE requires CUDA tensors");
  const auto* props = at::cuda::getDeviceProperties(x.get_device());
  TORCH_CHECK(props->major == 7 && props->minor == 0, "V4.1 prefill MoE requires SM70");
  TORCH_CHECK((gate_type == 16 || gate_type == 17 || gate_type == 29) &&
                  (down_type == 16 || down_type == 17 || down_type == 29),
              "V4.1 prefill MoE requires IQ2_XXS(16)/IQ2_XS(17)/IQ1_M(29) experts");
  TORCH_CHECK(left >= 0 && left % 64 == 0 && left + kRows <= 3 * 256, "V4.1 prefill MoE down window");
  TORCH_CHECK(tile >= 1 && route_cap >= 1, "V4.1 prefill MoE tile/route cap");
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
  const auto used_dev = torch::from_blob(used.data(), {int64_t(used.size())}, torch::kInt32).to(x.device());

  if (fused) {
    std::vector<Work> jobs;
    for (size_t i = 0; i < used.size(); ++i)
      for (int64_t at = first[i]; at < first[i + 1]; at += kBM)
        jobs.push_back({used[i], int32_t(at), int32_t(std::min<int64_t>(kBM, first[i + 1] - at)), 0});
    TORCH_CHECK(jobs.size() <= 65535 && routes <= INT32_MAX, "V4.1 prefill MoE: too many route tiles");
    auto jobs_dev = torch::from_blob(jobs.data(), {int64_t(jobs.size() * sizeof(Work))}, torch::kUInt8)
                        .to(x.device());
    const auto* work = reinterpret_cast<const Work*>(jobs_dev.data_ptr());
    auto hidden = torch::empty({routes, kRows}, x.options());
    auto down_rows = torch::empty({routes, kHidden}, x.options());
    auto* hs = reinterpret_cast<half*>(hidden.data_ptr());
    auto* rows_out = reinterpret_cast<half*>(down_rows.data_ptr());
    launch_grouped<true>(gate_type, reinterpret_cast<const half*>(x.data_ptr()), tokens.data_ptr<int64_t>(),
                         gate, &up, work, int(jobs.size()), kHidden, 0, weights.data_ptr<float>(), hs, kRows,
                         kRows / (kBN / 2), stream);
    launch_grouped<false>(down_type, hs, nullptr, down, nullptr, work, int(jobs.size()), kRows, int(left),
                          nullptr, rows_out, kHidden, kHidden / kBN, stream);
    combine_routes<<<grid_for(m * (kHidden / 8)), 256, 0, stream>>>(
        reinterpret_cast<half*>(out.data_ptr()), rows_out, position.data_ptr<int32_t>(), m);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

  // Tiles of consecutive used experts: <= tile experts and <= max(route_cap, one expert's routes).
  std::vector<std::pair<int64_t, int64_t>> tiles;  // [begin, end) indices into `used`
  int64_t tile_routes = 0;
  for (int64_t b = 0; b < int64_t(used.size());) {
    int64_t e = b + 1;
    while (e < int64_t(used.size()) && e - b < tile && first[e + 1] - first[b] <= route_cap) ++e;
    tiles.emplace_back(b, e);
    tile_routes = std::max(tile_routes, first[e] - first[b]);
    b = e;
  }
  const int64_t slots = std::min<int64_t>(tile, int64_t(used.size()));
  const auto half_options = x.options();
  auto gate_up_w = torch::empty({slots, 2 * kRows, kHidden}, half_options);
  auto down_w = torch::empty({slots, kHidden, kRows}, half_options);
  auto gathered = torch::empty({tile_routes, kHidden}, half_options);
  auto proj = torch::empty({tile_routes, 2 * kRows}, x.options().dtype(torch::kFloat32));
  auto hidden = torch::empty({tile_routes, kRows}, half_options);
  auto down_rows = torch::empty({routes, kHidden}, half_options);
  auto* gu = reinterpret_cast<half*>(gate_up_w.data_ptr());
  auto* dn = reinterpret_cast<half*>(down_w.data_ptr());
  auto* xs = reinterpret_cast<half*>(gathered.data_ptr());
  auto* hs = reinterpret_cast<half*>(hidden.data_ptr());
  auto* rows_out = reinterpret_cast<half*>(down_rows.data_ptr());
  float* pj = proj.data_ptr<float>();

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  MathMode mode(handle);
  for (const auto& [b, e] : tiles) {
    const int n = int(e - b);
    const int32_t* list = used_dev.data_ptr<int32_t>() + b;
    const int64_t p0 = first[b], tr = first[e] - p0;
    dequant(gate_type, gate, list, n, kRows, 0, kHidden, gu, 2 * kRows * kHidden, kHidden, stream);
    dequant(gate_type, up, list, n, kRows, 0, kHidden, gu + kRows * kHidden, 2 * kRows * kHidden, kHidden,
            stream);
    dequant(down_type, down, list, n, kHidden, int(left), kRows, dn, int64_t(kHidden) * kRows, kRows, stream);
    auto rows_in = gathered.slice(0, 0, tr);
    torch::index_select_out(rows_in, x, 0, tokens.slice(0, p0, p0 + tr));
    for (int s = 0; s < n; ++s) {
      const int64_t a = first[b + s] - p0, r = first[b + s + 1] - first[b + s];
      gemm(handle, 2 * kRows, int(r), kHidden, gu + s * int64_t(2 * kRows) * kHidden, kHidden,
           xs + a * kHidden, kHidden, pj + a * 2 * kRows, CUDA_R_32F, 2 * kRows);
    }
    routed_swiglu<<<grid_for(tr * kRows), 256, 0, stream>>>(hs, pj, weights.data_ptr<float>() + p0, tr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    for (int s = 0; s < n; ++s) {
      const int64_t a = first[b + s] - p0, r = first[b + s + 1] - first[b + s];
      gemm(handle, kHidden, int(r), kRows, dn + s * int64_t(kHidden) * kRows, kRows, hs + a * kRows, kRows,
           rows_out + (p0 + a) * kHidden, CUDA_R_16F, kHidden);
    }
  }
  mode.restore();
  combine_routes<<<grid_for(m * (kHidden / 8)), 256, 0, stream>>>(
      reinterpret_cast<half*>(out.data_ptr()), rows_out, position.data_ptr<int32_t>(), m);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
