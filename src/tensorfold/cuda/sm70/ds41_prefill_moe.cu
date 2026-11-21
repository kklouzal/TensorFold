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
                          int64_t tile, int64_t route_cap) {
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
