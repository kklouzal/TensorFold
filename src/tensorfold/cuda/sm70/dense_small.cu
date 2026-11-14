// Weight-streaming tensor-core GEMM for small-row FP16 dense projections on SM70.
//
// out[r, n] = sum_k x[r, k] * w[n, k] for r < rows <= 32, w row-major [N, K].
// Each CTA streams a BN x (K slice) slab of w exactly once with 128-bit loads
// (two register stages keep the next two BK chunks in flight during MMAs),
// stages the matching x chunk, and accumulates in FP32 WMMA fragments. Small-N
// shapes split K across CTAs; partial tiles go to a caller-owned FP32 workspace
// and the last-arriving CTA of a tile sums them in fixed split order, rounds
// once to the output dtype, and resets its arrival counter (graph-replayable).
// The launch configuration is a pure function of (rows, N, K), so a
// shape's reduction order is fixed; it differs from cuBLAS's order.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <algorithm>
#include <cstdint>
#include <limits>

namespace {
constexpr int MAX_ROWS = 32;
constexpr int MAX_BN = 64;
// About two CTAs per SM on the 80-SM V100. A fixed target keeps plans (and so
// reduction order) device-independent and bounds the split workspace: a split
// implies tiles < TARGET, hence tiles * splits < 2 * TARGET.
constexpr int TARGET_CTAS = 160;

struct Plan {
    int mt, warps, bk, tiles, splits, chunks_per_split;
    int64_t workspace;                  // floats; zero without a K split
    int counters;                       // ints; zero without a K split
};

// Tuned on V100-SXM2 (80 SMs) over the Flash Next TP8 projection shapes with
// cold-L2 weights: about two CTAs per SM, narrow tiles for long K.
Plan plan_for(int rows, int64_t n, int64_t k) {
    Plan p{};
    p.mt = rows <= 16 ? 16 : 32;
    if (p.mt == 16) {
        p.warps = k <= 512 ? 4 : 2;
        p.bk = k <= 512 ? 64 : 128;
    } else {
        p.warps = n >= 8192 ? 4 : 2;
        p.bk = 64;
    }
    const int bn = p.warps * 16;
    p.tiles = int((n + bn - 1) / bn);
    const int chunks = int((k + p.bk - 1) / p.bk);
    const int target = TARGET_CTAS;
    int splits = std::max(1, std::min(chunks, (target + p.tiles - 1) / p.tiles));
    p.chunks_per_split = (chunks + splits - 1) / splits;
    p.splits = (chunks + p.chunks_per_split - 1) / p.chunks_per_split;
    if (p.splits > 1) {
        p.workspace = int64_t(p.tiles) * p.splits * MAX_ROWS * MAX_BN;
        p.counters = p.tiles;
    }
    return p;
}

template <int MT, int WARPS, int BK, bool OUT_F32>
__global__ __launch_bounds__(WARPS * 32)
void small_gemm_kernel(const __half* __restrict__ x, int64_t ldx, const __half* __restrict__ w,
                       void* __restrict__ out, int64_t ldo, float* __restrict__ workspace,
                       unsigned* __restrict__ counters, int rows, int n_total, int k_total,
                       int chunks_per_split, int splits) {
    namespace wmma = nvcuda::wmma;
    constexpr int THREADS = WARPS * 32, BN = WARPS * 16, LD = BK + 8, VPR = BK / 8;
    constexpr int WV = BN * VPR / THREADS;
    constexpr int XV = (MT * VPR + THREADS - 1) / THREADS;
    static_assert(BN * VPR % THREADS == 0 && BN <= MAX_BN, "weight chunk must split evenly over threads");
    constexpr int STAGE_BYTES = (BN + MT) * LD * 2, RESULT_BYTES = MT * BN * 4;
    // The FP32 result tile reuses the staging bytes after the last chunk's MMAs.
    __shared__ __align__(32) unsigned char smem[STAGE_BYTES > RESULT_BYTES ? STAGE_BYTES : RESULT_BYTES];
    __half* sw = reinterpret_cast<__half*>(smem);
    __half* sx = sw + BN * LD;
    float* sc = reinterpret_cast<float*>(smem);
    __shared__ bool last_arrival;
    const int warp = threadIdx.x / 32;
    const int tile = blockIdx.x, split = blockIdx.y;
    const int n0 = tile * BN;
    const int chunks = (k_total + BK - 1) / BK;
    const int c0 = split * chunks_per_split, c1 = min(chunks, c0 + chunks_per_split);
    uint4 rw[WV], rx[XV], rw2[WV], rx2[XV];
    auto load = [&](int c, uint4 (&rw)[WV], uint4 (&rx)[XV]) {
        const int k0 = c * BK;
#pragma unroll
        for (int i = 0; i < WV; ++i) {
            const int at = threadIdx.x + i * THREADS;
            const int n = n0 + at / VPR, k = k0 + at % VPR * 8;
            rw[i] = n < n_total && k < k_total
                ? __ldg(reinterpret_cast<const uint4*>(w + int64_t(n) * k_total + k)) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int i = 0; i < XV; ++i) {
            const int at = threadIdx.x + i * THREADS;
            const int r = at / VPR, k = k0 + at % VPR * 8;
            rx[i] = at < MT * VPR && r < rows && k < k_total
                ? *reinterpret_cast<const uint4*>(x + int64_t(r) * ldx + k) : make_uint4(0, 0, 0, 0);
        }
    };
    auto stage = [&](const uint4 (&rw)[WV], const uint4 (&rx)[XV]) {
#pragma unroll
        for (int i = 0; i < WV; ++i) {
            const int at = threadIdx.x + i * THREADS;
            *reinterpret_cast<uint4*>(sw + (at / VPR) * LD + at % VPR * 8) = rw[i];
        }
#pragma unroll
        for (int i = 0; i < XV; ++i) {
            const int at = threadIdx.x + i * THREADS;
            if (at < MT * VPR) *reinterpret_cast<uint4*>(sx + (at / VPR) * LD + at % VPR * 8) = rx[i];
        }
    };
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[MT / 16];
#pragma unroll
    for (int m = 0; m < MT / 16; ++m) wmma::fill_fragment(acc[m], 0.0f);
    auto compute = [&]() {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::load_matrix_sync(b, sw + warp * 16 * LD + kk, LD);
#pragma unroll
            for (int m = 0; m < MT / 16; ++m) {
                wmma::load_matrix_sync(a, sx + m * 16 * LD + kk, LD);
                wmma::mma_sync(acc[m], a, b, acc[m]);
            }
        }
    };
    // Two register stages: chunks c+1 and c+2 are in flight while c computes.
    if (c0 < c1) load(c0, rw, rx);
    if (c0 + 1 < c1) load(c0 + 1, rw2, rx2);
    for (int c = c0; c < c1; c += 2) {
        stage(rw, rx);
        __syncthreads();
        if (c + 2 < c1) load(c + 2, rw, rx);
        compute();
        __syncthreads();
        if (c + 1 >= c1) break;
        stage(rw2, rx2);
        __syncthreads();
        if (c + 3 < c1) load(c + 3, rw2, rx2);
        compute();
        __syncthreads();
    }
#pragma unroll
    for (int m = 0; m < MT / 16; ++m)
        wmma::store_matrix_sync(sc + m * 16 * BN + warp * 16, acc[m], BN, wmma::mem_row_major);
    __syncthreads();
    auto publish = [&](int r, int n, float value) {
        const int64_t dest = int64_t(r) * ldo + n;
        if constexpr (OUT_F32) static_cast<float*>(out)[dest] = value;
        else static_cast<__half*>(out)[dest] = __float2half_rn(value);
    };
    if (splits == 1) {
        for (int at = threadIdx.x; at < rows * BN; at += THREADS) {
            const int n = n0 + at % BN;
            if (n < n_total) publish(at / BN, n, sc[at]);
        }
        return;
    }
    // Split K: publish this slice's FP32 tile, then the last arrival sums all
    // slices in split order. Workspace layout [tile][split][MAX_ROWS][MAX_BN].
    float* mine = workspace + (int64_t(tile) * splits + split) * MAX_ROWS * MAX_BN;
    for (int at = threadIdx.x; at < rows * BN; at += THREADS) mine[at] = sc[at];
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned arrived = atomicAdd(counters + tile, 1u);
        last_arrival = arrived == unsigned(splits - 1);
        if (last_arrival) counters[tile] = 0u;   // all slices arrived; reset for the next launch
    }
    __syncthreads();
    if (!last_arrival) return;
    __threadfence();
    const float* base = workspace + int64_t(tile) * splits * MAX_ROWS * MAX_BN;
    for (int at = threadIdx.x; at < rows * BN; at += THREADS) {
        const int n = n0 + at % BN;
        if (n >= n_total) continue;
        float value = __ldcg(base + at);
        for (int s = 1; s < splits; ++s) value += __ldcg(base + int64_t(s) * MAX_ROWS * MAX_BN + at);
        publish(at / BN, n, value);
    }
}

template <int MT, int WARPS, int BK, bool F32>
void launch(const Plan& p, const at::Tensor& x, const at::Tensor& w, at::Tensor& out, float* workspace,
            unsigned* counters, cudaStream_t stream) {
    small_gemm_kernel<MT, WARPS, BK, F32><<<dim3(p.tiles, p.splits), WARPS * 32, 0, stream>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), x.stride(0),
        reinterpret_cast<const __half*>(w.data_ptr<at::Half>()), out.data_ptr(), out.stride(0),
        workspace, counters, int(x.size(0)), int(w.size(0)), int(w.size(1)), p.chunks_per_split, p.splits);
}

template <bool F32>
void dispatch(const Plan& p, const at::Tensor& x, const at::Tensor& w, at::Tensor& out, float* workspace,
              unsigned* counters, cudaStream_t stream) {
    if (p.mt == 16 && p.warps == 4) launch<16, 4, 64, F32>(p, x, w, out, workspace, counters, stream);
    else if (p.mt == 16) launch<16, 2, 128, F32>(p, x, w, out, workspace, counters, stream);
    else if (p.warps == 4) launch<32, 4, 64, F32>(p, x, w, out, workspace, counters, stream);
    else launch<32, 2, 64, F32>(p, x, w, out, workspace, counters, stream);
}
} // namespace

// (workspace floats, counter ints) the fixed plan needs for this shape.
std::vector<int64_t> dense_small_requirement(int64_t rows, int64_t n, int64_t k) {
    TORCH_CHECK(rows >= 1 && rows <= MAX_ROWS && n >= 1 && n <= std::numeric_limits<int>::max()
                && k >= 8 && k % 8 == 0 && k <= std::numeric_limits<int>::max(),
                "small dense GEMM plans need 1..32 rows, positive N and K a multiple of 8");
    const Plan p = plan_for(int(rows), n, k);
    return {p.workspace, p.counters, p.tiles, p.splits, p.mt, p.warps * 16, p.bk};
}

void dense_small(const at::Tensor& x, const at::Tensor& w, at::Tensor out,
                 const at::Tensor& workspace, const at::Tensor& counters) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.dim() == 2 && x.stride(1) == 1
                && x.size(0) >= 1 && x.size(0) <= MAX_ROWS && x.size(1) >= 8 && x.size(1) % 8 == 0
                && x.size(1) <= std::numeric_limits<int>::max()
                && x.stride(0) >= x.size(1) && x.stride(0) % 8 == 0
                && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
                "small dense input must be 1..32 FP16 rows, contiguous 16-byte-aligned columns, K%8==0");
    const auto* prop = at::cuda::getDeviceProperties(x.get_device());
    TORCH_CHECK(prop->major == 7 && prop->minor == 0, "small dense GEMM requires SM70");
    TORCH_CHECK(w.device() == x.device() && w.scalar_type() == at::kHalf && w.dim() == 2 && w.is_contiguous()
                && w.size(1) == x.size(1) && w.size(0) >= 1 && w.size(0) <= std::numeric_limits<int>::max()
                && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
                "small dense weight must be contiguous 16-byte-aligned FP16 [N, K]");
    TORCH_CHECK(out.device() == x.device() && out.dim() == 2 && out.size(0) == x.size(0)
                && out.size(1) == w.size(0) && out.stride(1) == 1 && out.stride(0) >= out.size(1)
                && (out.scalar_type() == at::kHalf || out.scalar_type() == at::kFloat),
                "small dense output must be FP16/FP32 rows with contiguous columns and disjoint row strides");
    const int rows = int(x.size(0));
    const Plan p = plan_for(rows, w.size(0), w.size(1));
    float* scratch = nullptr;
    unsigned* arrivals = nullptr;
    if (p.splits > 1) {
        TORCH_CHECK(workspace.device() == x.device() && workspace.scalar_type() == at::kFloat
                    && workspace.is_contiguous() && workspace.numel() >= p.workspace,
                    "small dense split-K workspace is missing or too small");
        TORCH_CHECK(counters.device() == x.device() && counters.scalar_type() == at::kInt
                    && counters.is_contiguous() && counters.numel() >= p.counters,
                    "small dense split-K counters are missing or too small");
        scratch = workspace.data_ptr<float>();
        arrivals = reinterpret_cast<unsigned*>(counters.data_ptr<int>());
    }
    c10::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    if (out.scalar_type() == at::kFloat) dispatch<true>(p, x, w, out, scratch, arrivals, stream);
    else dispatch<false>(p, x, w, out, scratch, arrivals, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Workspace (floats, counters) sufficient for every plan.
std::vector<int64_t> dense_small_bound() {
    return {int64_t(2) * TARGET_CTAS * MAX_ROWS * MAX_BN, TARGET_CTAS};
}
