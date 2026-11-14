// Prefill form of gdn.cu's chain_kernel for SM70: the same contract, inputs and outputs, split into three kernels
// so a chunk's recurrence is not confined to one 1024-thread block per value head (6 blocks per TP8 rank).
//
//   front : per (row, head slot) block of 128 threads, the depthwise conv + SiLU (one fp16 rounding), the q/k
//           L2 norms and the decay/beta gates. Writes normalized k (fp32) and v (fp16), g and beta into the
//           caller's replay scratch (chain_kernel's k/v/g/b saves, same values) and normalized q into a workspace.
//   rows  : the delta rule. Each value row v of a head's state S[v][:] evolves independently given (k, q, g,
//           beta, v[v]), so rows are spread over warps across the whole GPU: one warp a value row (lane l owns
//           columns 4l..4l+3), W warps a block. The per-step inputs are read through the read-only cache D steps
//           ahead (register ring); no block barriers. Writes y = fp16(S q) into the output buffer.
//   back  : per (row, value head) block of 128 threads, the sigmoid-gated RMSNorm of y in place and the 32-group
//           sums, with chain_kernel's lane mapping and reduction trees.
//
// Every value is computed with chain_kernel's expression trees, lane mapping and xor-butterfly order, and this file
// is compiled with --fmad=false like gdn.cu, so outputs, sums, the final state and the saves are the same bits as
// chain_kernel. Shape (W = 4 warps, D = 4 prefetched steps) is the fastest of the measured bit-exact shapes on
// V100 (task ledger claude-perf-20261114: 0.655 ms vs chain_kernel 5.40 ms per 1024-row TP8 chunk; fewer lanes a
// row or fused multiply-adds regroup the sums and are not bit-exact).
// Design reference (no code taken): Strata-V100 (MIT) gdn_rec_cols_* column-split prefill recurrence.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>
#include <cstdint>
#include <optional>

namespace {

constexpr int DK = 128, DV = 128, TAPS = 4;

__device__ __forceinline__ float bf(float x) { return __half2float(__float2half_rn(x)); }

__device__ __forceinline__ float warp_sum(float x) {
    for (int o = 16; o; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float softplusf_(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

// Block (row r, slot): slots [0, nk) q heads, [nk, 2nk) k heads, [2nk, 2nk + nv) v heads; 128 threads.
template <bool NegativeDecay>
__global__ void __launch_bounds__(128) front_kernel(
        const __half* __restrict__ P, const __half* __restrict__ cs, const __half* __restrict__ cw,
        const float* __restrict__ decay_parameter, const float* __restrict__ dt_bias, int nk, int nv,
        float* __restrict__ q_out, float* __restrict__ k_out, __half* __restrict__ v_out,
        float* __restrict__ g_out, float* __restrict__ b_out) {
    const int C = 2 * nk * DK + nv * DV, PW = C + nv * DV + 2 * nv;
    const int r = blockIdx.x, slot = blockIdx.y, t = threadIdx.x, warp = t >> 5, lane = t & 31;
    __shared__ float xs[DK];
    __shared__ float inv_s;
    const int c = slot * DK + t;                      // q | k | v channels are consecutive 128-wide head slots
    float acc = 0.0f;
#pragma unroll
    for (int tap = 0; tap < TAPS; ++tap) {
        const int at = r + tap;
        const float x = at < TAPS - 1 ? __half2float(cs[at * C + c])
                                      : __half2float(P[(size_t)(at - (TAPS - 1)) * PW + c]);
        acc = acc + __half2float(cw[c * TAPS + tap]) * x;
    }
    const float act = bf(acc / (1.0f + expf(-acc)));
    if (slot >= 2 * nk) {
        const int hv = slot - 2 * nk;
        v_out[((size_t)r * nv + hv) * DV + t] = __float2half_rn(act);
        if (t == 0) {
            const float b = __half2float(P[(size_t)r * PW + C + nv * DV + hv]);
            const float a = __half2float(P[(size_t)r * PW + C + nv * DV + nv + hv]);
            const float rate = NegativeDecay ? decay_parameter[hv] : -expf(decay_parameter[hv]);
            g_out[(size_t)r * nv + hv] = expf(rate * softplusf_(a + dt_bias[hv]));
            b_out[(size_t)r * nv + hv] = bf(sigmoidf_(b));
        }
        return;
    }
    const bool is_q = slot < nk;
    xs[t] = act;
    __syncthreads();
    if (warp == 0) {                                  // chain_kernel's norm: lane l sums 4l..4l+3, then the butterfly
        float ss = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) { const float v = xs[lane * 4 + i]; ss = ss + v * v; }
        ss = warp_sum(ss);
        float inv = 1.0f / sqrtf(ss + 1e-6f);
        if (is_q) inv = inv * (1.0f / sqrtf((float)DK));
        if (lane == 0) inv_s = inv;
    }
    __syncthreads();
    const float x = act * inv_s;
    if (is_q) q_out[((size_t)r * nk + slot) * DK + t] = x;
    else k_out[((size_t)r * nk + slot - nk) * DK + t] = x;
}

// One warp a value row (E = 4 columns a lane), W warps (rows) a block, D steps prefetched.
constexpr int W = 4, D = 4;
__global__ void __launch_bounds__(32 * W) rows_kernel(
        const float* __restrict__ q, const float* __restrict__ k, const __half* __restrict__ v,
        const float* __restrict__ g, const float* __restrict__ beta, const float* __restrict__ state_in,
        float* __restrict__ state_out, __half* __restrict__ y, int rows, int nk, int nv,
        const int* __restrict__ keep_rows) {
    constexpr int L = 32, RPW = 1, E = DK / L, E4 = E / 4, G = 32 / L, ROWS_BLOCK = W * G * RPW;
    constexpr int BLOCKS_HEAD = DV / ROWS_BLOCK;
    static_assert(E == 4 && DV % ROWS_BLOCK == 0, "one float4 a lane, whole blocks a head");
    const int hv = blockIdx.x / BLOCKS_HEAD, hk = hv / (nv / nk);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, grp = lane / L, sub = lane % L;
    // row j of this lane group: base + grp + j * G (lane groups interleave within a warp's rows)
    const int base = (blockIdx.x % BLOCKS_HEAD) * ROWS_BLOCK + warp * G * RPW + grp;
    const size_t sbase = (size_t)hv * DV * DK;
    float s[RPW][E];
#pragma unroll
    for (int j = 0; j < RPW; ++j)
#pragma unroll
        for (int e4 = 0; e4 < E4; ++e4) {
            const float4 x = *reinterpret_cast<const float4*>(
                state_in + sbase + (size_t)(base + j * G) * DK + sub * E + 4 * e4);
            s[j][4 * e4] = x.x; s[j][4 * e4 + 1] = x.y; s[j][4 * e4 + 2] = x.z; s[j][4 * e4 + 3] = x.w;
        }
    const int live = keep_rows ? max(0, min(rows, *keep_rows)) : rows;
    auto store = [&]() {
#pragma unroll
        for (int j = 0; j < RPW; ++j)
#pragma unroll
            for (int e4 = 0; e4 < E4; ++e4)
                *reinterpret_cast<float4*>(state_out + sbase + (size_t)(base + j * G) * DK + sub * E + 4 * e4) =
                    make_float4(s[j][4 * e4], s[j][4 * e4 + 1], s[j][4 * e4 + 2], s[j][4 * e4 + 3]);
    };
    if (state_out != nullptr && !live) store();

    // the ring: step r's inputs live in slot r % D
    float4 kr[D][E4], qr[D][E4];
    float gr[D], br[D], vr[D][RPW];
    auto fetch = [&](float4 (&kd)[E4], float4 (&qd)[E4], float& gd, float& bd, float (&vd)[RPW], int r) {
        const float4* kp = reinterpret_cast<const float4*>(k + ((size_t)r * nk + hk) * DK + sub * E);
        const float4* qp = reinterpret_cast<const float4*>(q + ((size_t)r * nk + hk) * DK + sub * E);
#pragma unroll
        for (int e4 = 0; e4 < E4; ++e4) { kd[e4] = __ldg(kp + e4); qd[e4] = __ldg(qp + e4); }
        gd = __ldg(g + (size_t)r * nv + hv);
        bd = __ldg(beta + (size_t)r * nv + hv);
        const __half* vp = v + ((size_t)r * nv + hv) * DV + base;
#pragma unroll
        for (int j = 0; j < RPW; ++j) vd[j] = __half2float(__ldg(vp + j * G));
    };
#pragma unroll
    for (int d = 0; d < D; ++d)
        if (d < rows) fetch(kr[d], qr[d], gr[d], br[d], vr[d], d);

    for (int r0 = 0; r0 < rows; r0 += D) {
#pragma unroll
        for (int d = 0; d < D; ++d) {
            const int r = r0 + d;
            if (r < rows) {
                float kk[E], qq[E], vv[RPW];
#pragma unroll
                for (int e4 = 0; e4 < E4; ++e4) {
                    kk[4 * e4] = kr[d][e4].x; kk[4 * e4 + 1] = kr[d][e4].y;
                    kk[4 * e4 + 2] = kr[d][e4].z; kk[4 * e4 + 3] = kr[d][e4].w;
                    qq[4 * e4] = qr[d][e4].x; qq[4 * e4 + 1] = qr[d][e4].y;
                    qq[4 * e4 + 2] = qr[d][e4].z; qq[4 * e4 + 3] = qr[d][e4].w;
                }
                const float gt = gr[d], bt = br[d];
#pragma unroll
                for (int j = 0; j < RPW; ++j) vv[j] = vr[d][j];
                if (r + D < rows) fetch(kr[d], qr[d], gr[d], br[d], vr[d], r + D);
                // chain_kernel's update(): decay, read (k), correct toward v; then the read-out (q)
                float kv[RPW];
#pragma unroll
                for (int j = 0; j < RPW; ++j) {
                    kv[j] = 0.0f;
#pragma unroll
                    for (int e = 0; e < E; ++e) {
                        s[j][e] = s[j][e] * gt;
                        kv[j] = kv[j] + s[j][e] * kk[e];
                    }
                }
#pragma unroll
                for (int o = L / 2; o; o >>= 1)
#pragma unroll
                    for (int j = 0; j < RPW; ++j) kv[j] += __shfl_xor_sync(0xffffffffu, kv[j], o);
                float out[RPW];
#pragma unroll
                for (int j = 0; j < RPW; ++j) {
                    const float delta = (vv[j] - kv[j]) * bt;
                    out[j] = 0.0f;
#pragma unroll
                    for (int e = 0; e < E; ++e) {
                        s[j][e] = s[j][e] + kk[e] * delta;
                        out[j] = out[j] + s[j][e] * qq[e];
                    }
                }
#pragma unroll
                for (int o = L / 2; o; o >>= 1)
#pragma unroll
                    for (int j = 0; j < RPW; ++j) out[j] += __shfl_xor_sync(0xffffffffu, out[j], o);
                if (sub == 0) {
                    __half* yp = y + ((size_t)r * nv + hv) * DV + base;
#pragma unroll
                    for (int j = 0; j < RPW; ++j) yp[j * G] = __float2half_rn(out[j]);
                }
                if (state_out != nullptr && r + 1 == live) store();
            }
        }
    }
}

// Block (row r, value head hv), 128 threads: y (fp16 in out) -> gated RMSNorm in place, and the 32-group sums.
__global__ void __launch_bounds__(128) back_kernel(
        const __half* __restrict__ P, const __half* __restrict__ norm_w, float eps, int nk, int nv,
        __half* __restrict__ out, float* __restrict__ xs) {
    const int C = 2 * nk * DK + nv * DV, PW = C + nv * DV + 2 * nv;
    const int r = blockIdx.x, hv = blockIdx.y, t = threadIdx.x, warp = t >> 5, lane = t & 31;
    __shared__ float ys[DV];
    __shared__ float rinv;
    const size_t at = (size_t)r * nv * DV + hv * DV + t;
    const float yt = __half2float(out[at]);
    ys[t] = yt;
    __syncthreads();
    if (warp == 0) {
        float ss = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) { const float y = ys[lane * 4 + i]; ss = ss + y * y; }
        ss = warp_sum(ss);
        if (lane == 0) rinv = 1.0f / sqrtf(ss / (float)DV + eps);
    }
    __syncthreads();
    const float yn = bf(bf(yt * rinv) * __half2float(norm_w[t]));
    const float z = __half2float(P[(size_t)r * PW + C + hv * DV + t]);
    const float o = bf(yn * sigmoidf_(z));
    out[at] = __float2half_rn(o);
    const float gs = warp_sum(o);
    if (lane == 0) xs[(size_t)r * (nv * DV / 32) + hv * (DV / 32) + warp] = gs;
}

template <typename T>
T* ptr(const at::Tensor& x) { return x.defined() && x.numel() ? (T*)x.data_ptr() : nullptr; }

bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

}  // namespace

// Arguments as sm70_gdn_chain_cuda (validated by the binding).
void sm70_gdn_split_chain_cuda(const at::Tensor& P, const at::Tensor& cs, const at::Tensor& cw,
        const at::Tensor& state_in, const at::Tensor& decay_parameter, const at::Tensor& dt_bias,
        const at::Tensor& norm_w, double eps, int64_t rows, at::Tensor& out, at::Tensor& xs, at::Tensor& state_out,
        at::Tensor& k_save, at::Tensor& v_save, at::Tensor& g_save, at::Tensor& b_save,
        const std::optional<at::Tensor>& keep_rows, bool negative_decay) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const int nv = (int)decay_parameter.numel(), nk = nv / 3, n = (int)rows;
    TORCH_CHECK(nv == 6 || nv == 12 || nv == 24 || nv == 48, "gdn split: 48, 24, 12 or 6 value heads");
    // The replay saves (validated rows-capable by the binding) double as the recurrence's inputs; normalized q
    // takes a stream-ordered workspace from the caching allocator.
    const at::Tensor& k_ws = k_save;
    const at::Tensor& v_ws = v_save;
    const at::Tensor& g_ws = g_save;
    const at::Tensor& b_ws = b_save;
    at::Tensor q_ws = at::empty({std::max<int64_t>(rows, 1), nk, DK}, P.options().dtype(at::kFloat));
    TORCH_CHECK(n == 0 || (ptr<float>(k_ws) && ptr<__half>(v_ws) && ptr<float>(g_ws) && ptr<float>(b_ws)),
                "gdn split: the replay saves carry the recurrence inputs");
    TORCH_CHECK((n == 0 || aligned16(k_ws.data_ptr())) && aligned16(q_ws.data_ptr())
                && aligned16(state_in.data_ptr()) && (!ptr<float>(state_out) || aligned16(state_out.data_ptr())),
                "gdn split: 16-byte aligned keys and states");
    if (n > 0) {
        const dim3 front_grid(n, 2 * nk + nv);
        auto front = negative_decay ? front_kernel<true> : front_kernel<false>;
        front<<<front_grid, 128, 0, stream>>>(ptr<__half>(P), ptr<__half>(cs), ptr<__half>(cw),
            ptr<float>(decay_parameter), ptr<float>(dt_bias), nk, nv, ptr<float>(q_ws), ptr<float>(k_ws),
            ptr<__half>(v_ws), ptr<float>(g_ws), ptr<float>(b_ws));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    rows_kernel<<<nv * (DV / W), 32 * W, 0, stream>>>(
        ptr<float>(q_ws), ptr<float>(k_ws), ptr<__half>(v_ws), ptr<float>(g_ws), ptr<float>(b_ws),
        ptr<float>(state_in), ptr<float>(state_out), ptr<__half>(out), n, nk, nv,
        keep_rows ? ptr<int>(*keep_rows) : nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (n > 0) {
        back_kernel<<<dim3(n, nv), 128, 0, stream>>>(ptr<__half>(P), ptr<__half>(norm_w), (float)eps, nk, nv,
                                                    ptr<__half>(out), ptr<float>(xs));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}
