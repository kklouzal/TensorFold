// Gemma 4 SM70 (Volta) kernels: W4A16 NVFP4 GEMV, BF16 GEMV, routing, fused norm tails,
// q/k/v norm + RoPE + KV write, WMMA split-key flash attention. Profile gemma4-sm70-w4a16-v1.
//
// Numerical contract: original E2M1 codes and E4M3 block scales are decoded exactly (bit placement into
// FP16, see decode8); every 16-element group dot product accumulates in FP32 and is scaled once by its
// block scale, then by the projection's FP32 per-output-row multiplier (1 / weight_global_scale).
// Activations into decode GEMVs are FP32; tensor-core attention uses FP16 Q/K/V/P with FP32 accumulation.
// All reductions have a fixed order that does not depend on the number of rows in a call, except the
// attention key-split count, which the caller fixes per (rows, context bucket).
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>
#include <cmath>

namespace g4 {

using namespace nvcuda;

constexpr int WARPS = 8;  // 256-thread blocks for the GEMV family

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Four packed bytes (eight E2M1 codes, low nibble first) -> eight floats equal to E2M1 * 2^-14.
__device__ __forceinline__ void decode8(uint32_t v, float* f) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t u = v >> (8 * i);
        uint32_t w = ((u & 0x7u) << 9) | ((u & 0x8u) << 12) | ((u & 0x70u) << 21) | ((u & 0x80u) << 24);
        const float2 g = __half22float2(*reinterpret_cast<const __half2*>(&w));
        f[2 * i] = g.x;
        f[2 * i + 1] = g.y;
    }
}

// E4M3 byte (finite encodings only; validated at load) -> value * 2^14 (pairs with decode8's 2^-14).
__device__ __forceinline__ float scale_e4m3(uint32_t s) {
    const unsigned short bits = (unsigned short)(((s & 0x80u) << 8) | ((s & 0x7fu) << 7));
    return __half2float(__ushort_as_half(bits)) * 4194304.0f;  // e4m3 * 2^-8 * 2^22
}

__device__ __forceinline__ float bf16f(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T> __device__ __forceinline__ T cast_out(float v);
template <> __device__ __forceinline__ float cast_out<float>(float v) { return v; }
template <> __device__ __forceinline__ half cast_out<half>(float v) { return __float2half_rn(v); }

__device__ __forceinline__ float gelu_tanh(float g) {
    const float inner = 0.7978845608028654f * (g + 0.044715f * g * g * g);
    return 0.5f * g * (1.0f + tanhf(inner));
}

// ---------------------------------------------------------------------------------------------- W4 GEMV
// out[m, n] = rowmul[n] * sum_g scale[n,g] * dot(x[m, 16g:16g+16], E2M1[n, 16g:16g+16]), m < M <= MAXM.
// GEGLU: warp n computes gate row n and up row n + W (W = N), writes gelu_tanh(gate) * up.
template <int MAXM, bool GEGLU>
__global__ void __launch_bounds__(256) w4_rows(const float* __restrict__ x, int64_t ldx, int M,
        const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales, const float* __restrict__ rowmul,
        int N, int K, float* __restrict__ out, int64_t ldo) {
    const int n = blockIdx.x * WARPS + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (n >= N) return;
    const int groups = K >> 4;
    constexpr int PARTS = GEGLU ? 2 : 1;
    float acc[PARTS][MAXM];
#pragma unroll
    for (int p = 0; p < PARTS; ++p)
#pragma unroll
        for (int m = 0; m < MAXM; ++m) acc[p][m] = 0.f;
#pragma unroll
    for (int p = 0; p < PARTS; ++p) {
        const int row = n + p * N;
        const uint2* c = reinterpret_cast<const uint2*>(codes + (int64_t)row * (K >> 1));
        const uint8_t* s = scales + (int64_t)row * groups;
        for (int g = lane; g < groups; g += 32) {
            const uint2 w = __ldg(c + g);
            const float sc = scale_e4m3(__ldg(s + g));
            float f[16];
            decode8(w.x, f);
            decode8(w.y, f + 8);
#pragma unroll
            for (int m = 0; m < MAXM; ++m) {
                if (m < M) {
                    const float4* xp = reinterpret_cast<const float4*>(x + m * ldx + g * 16);
                    float d = 0.f;
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const float4 a = __ldg(xp + q);
                        d = fmaf(a.x, f[4 * q], d);
                        d = fmaf(a.y, f[4 * q + 1], d);
                        d = fmaf(a.z, f[4 * q + 2], d);
                        d = fmaf(a.w, f[4 * q + 3], d);
                    }
                    acc[p][m] = fmaf(d, sc, acc[p][m]);
                }
            }
        }
    }
#pragma unroll
    for (int m = 0; m < MAXM; ++m) {
        if (m < M) {
            float v0 = warp_sum(acc[0][m]) * __ldg(rowmul + n);
            if (GEGLU) {
                const float v1 = warp_sum(acc[PARTS - 1][m]) * __ldg(rowmul + n + N);
                v0 = gelu_tanh(v0) * v1;
            }
            if (lane == 0) out[m * ldo + n] = v0;
        }
    }
}

// Expert gate/up GEGLU over (row, slot) pairs: act[p, j] for local expert local[p] (>= 0), zeros skipped.
// Weights are stacked per local expert: codes [E, 2W, K/2], scales [E, 2W, K/16], rowmul [E, 2W].
__global__ void __launch_bounds__(256) w4_expert_gateup(const float* __restrict__ x, int64_t ldx, int slots,
        const int* __restrict__ local, const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales,
        const float* __restrict__ rowmul, int W, int K, float* __restrict__ act) {
    const int pair = blockIdx.y;
    const int e = local[pair];
    if (e < 0) return;
    const int j = blockIdx.x * WARPS + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (j >= W) return;
    const int groups = K >> 4;
    const float* xr = x + (pair / slots) * ldx;
    const int64_t base = (int64_t)e * 2 * W;
    float acc[2] = {0.f, 0.f};
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int64_t row = base + j + p * W;
        const uint2* c = reinterpret_cast<const uint2*>(codes + row * (K >> 1));
        const uint8_t* s = scales + row * groups;
        for (int g = lane; g < groups; g += 32) {
            const uint2 w = __ldg(c + g);
            const float sc = scale_e4m3(__ldg(s + g));
            float f[16];
            decode8(w.x, f);
            decode8(w.y, f + 8);
            const float4* xp = reinterpret_cast<const float4*>(xr + g * 16);
            float d = 0.f;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float4 a = __ldg(xp + q);
                d = fmaf(a.x, f[4 * q], d);
                d = fmaf(a.y, f[4 * q + 1], d);
                d = fmaf(a.z, f[4 * q + 2], d);
                d = fmaf(a.w, f[4 * q + 3], d);
            }
            acc[p] = fmaf(d, sc, acc[p]);
        }
    }
    const float g = warp_sum(acc[0]) * __ldg(rowmul + base + j);
    const float u = warp_sum(acc[1]) * __ldg(rowmul + base + j + W);
    if (lane == 0) act[(int64_t)pair * W + j] = gelu_tanh(g) * u;
}

// Expert down with the routing combine: out[m, n] = sum_s weight[m,s] * down_e(act[m*S+s])[n], slots in
// ascending order, non-local slots skipped. Down weights: codes [E, N, W/2], scales [E, N, W/16], rowmul [E, N].
__global__ void __launch_bounds__(256) w4_expert_down(const float* __restrict__ act, int slots,
        const int* __restrict__ local, const float* __restrict__ weight, const uint8_t* __restrict__ codes,
        const uint8_t* __restrict__ scales, const float* __restrict__ rowmul, int N, int W,
        float* __restrict__ out, int64_t ldo) {
    const int m = blockIdx.y;
    const int n = blockIdx.x * WARPS + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (n >= N) return;
    const int groups = W >> 4;
    float total = 0.f;
    for (int s = 0; s < slots; ++s) {
        const int pair = m * slots + s;
        const int e = local[pair];
        if (e < 0) continue;
        const int64_t row = (int64_t)e * N + n;
        const uint2* c = reinterpret_cast<const uint2*>(codes + row * (W >> 1));
        const uint8_t* sp = scales + row * groups;
        const float* xr = act + (int64_t)pair * W;
        float acc = 0.f;
        for (int g = lane; g < groups; g += 32) {
            const uint2 w = __ldg(c + g);
            const float sc = scale_e4m3(__ldg(sp + g));
            float f[16];
            decode8(w.x, f);
            decode8(w.y, f + 8);
            const float4* xp = reinterpret_cast<const float4*>(xr + g * 16);
            float d = 0.f;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float4 a = __ldg(xp + q);
                d = fmaf(a.x, f[4 * q], d);
                d = fmaf(a.y, f[4 * q + 1], d);
                d = fmaf(a.z, f[4 * q + 2], d);
                d = fmaf(a.w, f[4 * q + 3], d);
            }
            acc = fmaf(d, sc, acc);
        }
        total = fmaf(__ldg(weight + pair), warp_sum(acc) * __ldg(rowmul + row), total);
    }
    if (lane == 0) out[m * ldo + n] = total;
}

// Exact FP16 weights e2m1 * e4m3 (rowmul applied by the caller's GEMM epilogue): out[N, K] half.
__global__ void w4_dequant(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales, int64_t N, int K,
                           half* __restrict__ out) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;  // one group of 16 per thread
    const int groups = K >> 4;
    if (i >= N * groups) return;
    const int64_t row = i / groups;
    const int g = (int)(i % groups);
    const uint2 w = reinterpret_cast<const uint2*>(codes + row * (K >> 1))[g];
    const float sc = scale_e4m3(scales[i]);
    float f[16];
    decode8(w.x, f);
    decode8(w.y, f + 8);
    half* o = out + row * K + g * 16;
#pragma unroll
    for (int q = 0; q < 16; ++q) o[q] = __float2half_rn(f[q] * sc);  // exact: <= 7 significant bits in range
}

// ---------------------------------------------------------------------------------------------- BF16 GEMV
// out[m, n] = dot(x[m], W[n]) (FP32), optional final-logit softcap.  K % 8 == 0.
template <int MAXM>
__global__ void __launch_bounds__(256) bf16_rows(const float* __restrict__ x, int64_t ldx, int M,
        const __nv_bfloat16* __restrict__ w, int N, int K, float* __restrict__ out, int64_t ldo, float softcap) {
    const int n = blockIdx.x * WARPS + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (n >= N) return;
    const uint4* row = reinterpret_cast<const uint4*>(w + (int64_t)n * K);
    float acc[MAXM];
#pragma unroll
    for (int m = 0; m < MAXM; ++m) acc[m] = 0.f;
    for (int c = lane; c < (K >> 3); c += 32) {
        const uint4 v = __ldg(row + c);
        const uint32_t words[4] = {v.x, v.y, v.z, v.w};
        float f[8];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            f[2 * q] = __uint_as_float(words[q] << 16);
            f[2 * q + 1] = __uint_as_float(words[q] & 0xffff0000u);
        }
#pragma unroll
        for (int m = 0; m < MAXM; ++m) {
            if (m < M) {
                const float4* xp = reinterpret_cast<const float4*>(x + m * ldx + c * 8);
                const float4 a = __ldg(xp), b = __ldg(xp + 1);
                float d = acc[m];
                d = fmaf(a.x, f[0], d); d = fmaf(a.y, f[1], d); d = fmaf(a.z, f[2], d); d = fmaf(a.w, f[3], d);
                d = fmaf(b.x, f[4], d); d = fmaf(b.y, f[5], d); d = fmaf(b.z, f[6], d); d = fmaf(b.w, f[7], d);
                acc[m] = d;
            }
        }
    }
#pragma unroll
    for (int m = 0; m < MAXM; ++m) {
        if (m < M) {
            float v = warp_sum(acc[m]);
            if (softcap > 0.f) v = tanhf(v / softcap) * softcap;
            if (lane == 0) out[m * ldo + n] = v;
        }
    }
}

// ---------------------------------------------------------------------------------------------- routing
// One 128-thread block per row (E <= 128): softmax over all experts, stable top-k by probability,
// renormalized and multiplied by per_expert_scale. local = id - first when owned, else -1.
__global__ void route(const float* __restrict__ logits, int E, int topk, const __nv_bfloat16* __restrict__ scale,
                      int first, int count, int* __restrict__ ids, float* __restrict__ weights, int* __restrict__ local) {
    __shared__ float prob[128];
    __shared__ float red[4];
    __shared__ int redi[4];
    const int m = blockIdx.x, t = threadIdx.x;
    const float l = t < E ? logits[m * E + t] : -INFINITY;
    float mx = l;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    if ((t & 31) == 0) red[t >> 5] = mx;
    __syncthreads();
    mx = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
    __syncthreads();
    const float p = t < E ? expf(l - mx) : 0.f;
    float sum = warp_sum(p);
    if ((t & 31) == 0) red[t >> 5] = sum;
    __syncthreads();
    sum = red[0] + red[1] + red[2] + red[3];
    __syncthreads();
    prob[t] = t < E ? p / sum : -1.f;
    __syncthreads();
    float chosen = 0.f;
    int picked = -1;
    float selected_total = 0.f;
    // Sequential top-k (k <= 8): block argmax with lowest index on ties.
    for (int k = 0; k < topk; ++k) {
        float v = prob[t];
        int idx = t;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, o);
            const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
            if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
        }
        if ((t & 31) == 0) { red[t >> 5] = v; redi[t >> 5] = idx; }
        __syncthreads();
        float bv = red[0];
        int bi = redi[0];
        for (int w = 1; w < 4; ++w)
            if (red[w] > bv || (red[w] == bv && redi[w] < bi)) { bv = red[w]; bi = redi[w]; }
        selected_total += bv;
        if (t == k) { chosen = bv; picked = bi; }
        __syncthreads();
        if (t == bi) prob[t] = -2.f;
        __syncthreads();
    }
    if (t < topk) {
        const float w = chosen / selected_total * bf16f(scale[picked]);
        ids[m * topk + t] = picked;
        weights[m * topk + t] = w;
        local[m * topk + t] = (picked >= first && picked < first + count) ? picked - first : -1;
    }
}

// ---------------------------------------------------------------------------------------------- norms
template <int THREADS>
__device__ __forceinline__ float block_sum(float v, float* scratch) {
    v = warp_sum(v);
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    __syncthreads();
    if (lane == 0) scratch[w] = v;
    __syncthreads();
    float total = 0.f;
#pragma unroll
    for (int i = 0; i < THREADS / 32; ++i) total += scratch[i];
    return total;
}

// Sum of `world` rank partials in rank order (world == 1: the already reduced value).
__device__ __forceinline__ float parts_sum(const float* __restrict__ p, int world, int64_t stride) {
    float v = p[0];
    for (int r = 1; r < world; ++r) v += p[r * stride];
    return v;
}

constexpr int NT = 256;       // norm block threads
constexpr int MAXD = 2816;    // hidden width (11 values per thread)
constexpr int PER = (MAXD + NT - 1) / NT;

// out[m] = x[m] * rsqrt(mean(x^2) + eps) * (w ? w : 1) * mul
template <typename OT>
__global__ void __launch_bounds__(NT) rms(const float* __restrict__ x, int64_t ldx, const __nv_bfloat16* __restrict__ w,
                                         int D, float eps, float mul, OT* __restrict__ out, int64_t ldo) {
    __shared__ float scratch[NT / 32];
    const float* xr = x + blockIdx.x * ldx;
    float ss = 0.f;
    for (int d = threadIdx.x; d < D; d += NT) { const float v = xr[d]; ss = fmaf(v, v, ss); }
    const float inv = rsqrtf(block_sum<NT>(ss, scratch) / D + eps);
    for (int d = threadIdx.x; d < D; d += NT)
        out[blockIdx.x * ldo + d] = cast_out<OT>(xr[d] * inv * (w ? bf16f(w[d]) : 1.f) * mul);
}

// Attention tail: h += norm(o, w_pa); dense = norm(h, w_pf); expert = norm(h, w_pf2);
// router = rmsnorm_unweighted(h) * root * router_scale.  D == MAXD.
template <typename OT>
__global__ void __launch_bounds__(NT) attn_tail(const float* __restrict__ o, int64_t ldo_in, int world,
        int64_t pstride, float* __restrict__ h,
        const __nv_bfloat16* __restrict__ w_pa, const __nv_bfloat16* __restrict__ w_pf,
        const __nv_bfloat16* __restrict__ w_pf2, const __nv_bfloat16* __restrict__ w_router, float root, float eps,
        OT* __restrict__ dense, OT* __restrict__ expert, float* __restrict__ router) {
    __shared__ float scratch[NT / 32];
    const int m = blockIdx.x, D = MAXD;
    float v[PER];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int d = threadIdx.x + i * NT;
        v[i] = d < D ? parts_sum(o + m * ldo_in + d, world, pstride) : 0.f;
        ss = fmaf(v[i], v[i], ss);
    }
    float inv = rsqrtf(block_sum<NT>(ss, scratch) / D + eps);
    ss = 0.f;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int d = threadIdx.x + i * NT;
        if (d < D) {
            v[i] = h[(int64_t)m * D + d] + v[i] * inv * bf16f(w_pa[d]);
            h[(int64_t)m * D + d] = v[i];
        } else v[i] = 0.f;
        ss = fmaf(v[i], v[i], ss);
    }
    inv = rsqrtf(block_sum<NT>(ss, scratch) / D + eps);
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int d = threadIdx.x + i * NT;
        if (d < D) {
            const float y = v[i] * inv;
            dense[(int64_t)m * D + d] = cast_out<OT>(y * bf16f(w_pf[d]));
            expert[(int64_t)m * D + d] = cast_out<OT>(y * bf16f(w_pf2[d]));
            router[(int64_t)m * D + d] = y * root * bf16f(w_router[d]);
        }
    }
}

// FFN tail: s = norm(dense, w1) + norm(routed, w2); h = (h + norm(s, w_out)) * scalar; next = norm(h, w_next).
// red: [M, 2, D] (dense partial sums, routed partial sums). tap (nullable) receives h.
template <typename OT>
__global__ void __launch_bounds__(NT) ffn_tail(const float* __restrict__ red, int world, int64_t pstride,
        float* __restrict__ h,
        const __nv_bfloat16* __restrict__ w1, const __nv_bfloat16* __restrict__ w2,
        const __nv_bfloat16* __restrict__ w_out, const __nv_bfloat16* __restrict__ scalar,
        const __nv_bfloat16* __restrict__ w_next, float eps, OT* __restrict__ next, float* __restrict__ tap,
        int64_t ldtap) {
    __shared__ float scratch[NT / 32];
    const int m = blockIdx.x, D = MAXD;
    const float* dn = red + (int64_t)m * 2 * D;
    const float* rt = dn + D;
    float a[PER], b[PER];
    float sa = 0.f, sb = 0.f;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int d = threadIdx.x + i * NT;
        a[i] = d < D ? parts_sum(dn + d, world, pstride) : 0.f;
        b[i] = d < D ? parts_sum(rt + d, world, pstride) : 0.f;
        sa = fmaf(a[i], a[i], sa);
        sb = fmaf(b[i], b[i], sb);
    }
    const float ia = rsqrtf(block_sum<NT>(sa, scratch) / D + eps);
    const float ib = rsqrtf(block_sum<NT>(sb, scratch) / D + eps);
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int d = threadIdx.x + i * NT;
        a[i] = d < D ? a[i] * ia * bf16f(w1[d]) + b[i] * ib * bf16f(w2[d]) : 0.f;
        ss = fmaf(a[i], a[i], ss);
    }
    const float is = rsqrtf(block_sum<NT>(ss, scratch) / D + eps);
    const float sc = bf16f(scalar[0]);
    ss = 0.f;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int d = threadIdx.x + i * NT;
        if (d < D) {
            a[i] = (h[(int64_t)m * D + d] + a[i] * is * bf16f(w_out[d])) * sc;
            h[(int64_t)m * D + d] = a[i];
            if (tap) tap[m * ldtap + d] = a[i];
        } else a[i] = 0.f;
        ss = fmaf(a[i], a[i], ss);
    }
    const float in = rsqrtf(block_sum<NT>(ss, scratch) / D + eps);
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int d = threadIdx.x + i * NT;
        if (d < D) next[(int64_t)m * D + d] = cast_out<OT>(a[i] * in * bf16f(w_next[d]));
    }
}

// h[m] = float(table[id]) * mul
__global__ void embed(const int64_t* __restrict__ ids, const __nv_bfloat16* __restrict__ table, int D, float mul,
                      float* __restrict__ out) {
    const int64_t id = ids[blockIdx.x];
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        out[(int64_t)blockIdx.x * D + d] = bf16f(table[id * D + d]) * mul;
}

// ---------------------------------------------------------------------------------------------- q/k/v prep
// x: [M, ldx] FP32 projections: q heads at [0, Hq*D), k at [Hq*D, (Hq+1)*D), v (has_v) at [(Hq+1)*D, (Hq+2)*D).
// grid (M, Hq + 1), D/2 threads; thread i owns rotary pair (i, i + D/2).
// q -> learned RMS, RoPE -> qout[m, h, :] (FP16).  k -> learned RMS, RoPE -> kc[slot].
// v -> unweighted RMS of v (or of k when !has_v: K=V layers) -> vc[slot]. key_pos[slot] = position.
template <typename QT>
__global__ void qkv_prep(const float* __restrict__ x, int64_t ldx, int Hq, int D, int v_mode,
        const __nv_bfloat16* __restrict__ qw, const __nv_bfloat16* __restrict__ kw, const float* __restrict__ inv_freq,
        const int* __restrict__ pos0, QT* __restrict__ qout, half* __restrict__ kc, half* __restrict__ vc,
        int* __restrict__ key_pos, int ring, float eps) {
    __shared__ float scratch[16];
    const int m = blockIdx.x, h = blockIdx.y, i = threadIdx.x, half_d = D >> 1;
    const int pos = *pos0 + m;
    const float* xr = x + m * ldx + h * D;
    const float x1 = xr[i], x2 = xr[i + half_d];
    // block_sum over D/2 threads (128 or 256)
    float ss = fmaf(x1, x1, x2 * x2);
    ss = warp_sum(ss);
    const int warps = blockDim.x >> 5;
    if ((i & 31) == 0) scratch[i >> 5] = ss;
    __syncthreads();
    float total = 0.f;
    for (int w = 0; w < warps; ++w) total += scratch[w];
    const float inv = rsqrtf(total / D + eps);
    const __nv_bfloat16* wt = h < Hq ? qw : kw;
    const float a = x1 * inv * bf16f(wt[i]), b = x2 * inv * bf16f(wt[i + half_d]);
    const float angle = (float)pos * inv_freq[i];
    float sn, cs;
    sincosf(angle, &sn, &cs);
    const float r1 = a * cs - b * sn, r2 = b * cs + a * sn;
    if (h < Hq) {
        QT* q = qout + ((int64_t)m * Hq + h) * D;
        q[i] = cast_out<QT>(r1);
        q[i + half_d] = cast_out<QT>(r2);
        return;
    }
    const int slot = ring ? pos % ring : pos;
    kc[(int64_t)slot * D + i] = __float2half_rn(r1);
    kc[(int64_t)slot * D + i + half_d] = __float2half_rn(r2);
    if (i == 0 && key_pos) key_pos[slot] = pos;
    float v1 = x1, v2 = x2, vinv = inv;
    if (v_mode == 2) {
        v1 = xr[D + i];
        v2 = xr[D + i + half_d];
        vinv = 1.f;
    } else if (v_mode == 1) {
        const float* vr = xr + D;
        v1 = vr[i];
        v2 = vr[i + half_d];
        float vs = warp_sum(fmaf(v1, v1, v2 * v2));
        __syncthreads();
        if ((i & 31) == 0) scratch[i >> 5] = vs;
        __syncthreads();
        float vt = 0.f;
        for (int w = 0; w < warps; ++w) vt += scratch[w];
        vinv = rsqrtf(vt / D + eps);
    }
    vc[(int64_t)slot * D + i] = __float2half_rn(v1 * vinv);
    vc[(int64_t)slot * D + i + half_d] = __float2half_rn(v2 * vinv);
}

// ---------------------------------------------------------------------------------------------- attention
// Split-key flash attention over one KV head shared by every query row of a 16-row tile.
// q: [R, D] FP16, r = row * H + head (row position = *pos0 + r / H). K/V caches [S, D] FP16 with
// S % 256 == 0 and every slot initialized (masked keys must hold finite values).
// key_pos: slot -> absolute position (or nullptr: slot index is the position). A key is visible to a query
// at position p when 0 <= kp <= p and (window == 0 or kp > p - window).
// Block (split, tile) visits chunks [split*cps, min(chunks, (split+1)*cps)) of 256 keys with an online softmax
// and writes unnormalized partials po[split, r, :], pm/pl[split, r]; attn_merge folds splits in order.
constexpr int AC = 256;   // keys per chunk

template <int D>
struct AttnSmem {
    static constexpr int QLD = D + 8, PLD = AC + 8;
    static constexpr size_t bytes = 16 * QLD * 2 + 16 * AC * 4 + 16 * PLD * 2 + 16 * D * 4;
};

template <int D>
__global__ void __launch_bounds__(128) attn_split(const half* __restrict__ q, int R, int H, const half* __restrict__ kc,
        const half* __restrict__ vc, const int* __restrict__ key_pos, int S, const int* __restrict__ pos0, int window,
        int noncausal, int chunks, int cps, float* __restrict__ po, float* __restrict__ pm, float* __restrict__ pl, int Rpad) {
    constexpr int QLD = AttnSmem<D>::QLD, PLD = AttnSmem<D>::PLD;
    extern __shared__ __align__(32) unsigned char smem[];
    half* qs = reinterpret_cast<half*>(smem);
    float* ss = reinterpret_cast<float*>(qs + 16 * QLD);
    half* ps = reinterpret_cast<half*>(ss + 16 * AC);
    float* os = reinterpret_cast<float*>(ps + 16 * PLD);
    __shared__ int qpos[16];
    __shared__ float row_m[16], row_l[16], row_a[16];
    const int split = blockIdx.x, tile = blockIdx.y, t = threadIdx.x, warp = t >> 5;
    const int r0 = tile * 16;
    const int base = *pos0;
    if (t < 16) {
        const int r = r0 + t;
        qpos[t] = r < R ? base + (noncausal ? (R - 1) / H : r / H) : -1;
        row_m[t] = -INFINITY;
        row_l[t] = 0.f;
    }
    for (int i = t; i < 16 * D; i += 128) os[i] = 0.f;
    for (int i = t; i < 16 * (D / 8); i += 128) {
        const int row = i / (D / 8), c8 = i % (D / 8);
        uint4 v = make_uint4(0, 0, 0, 0);
        if (r0 + row < R) v = reinterpret_cast<const uint4*>(q + (int64_t)(r0 + row) * D)[c8];
        *reinterpret_cast<uint4*>(qs + row * QLD + c8 * 8) = v;
    }
    __syncthreads();
    int qmax = -1, qmin = 0x7fffffff;
    for (int r = 0; r < 16; ++r)
        if (qpos[r] >= 0) { qmax = max(qmax, qpos[r]); qmin = min(qmin, qpos[r]); }
    const int first = split * cps, last = min(chunks, first + cps);
    const int row = t >> 3, sub = t & 7;   // softmax: 8 threads per query row
    constexpr int COLS = D / 4, FR = COLS / 16;
    for (int c = first; c < last; ++c) {
        const int k0 = c * AC;
        bool ok = false;
        for (int i = t; i < AC; i += 128) {
            const int key = k0 + i;
            if (key < S) {
                const int kp = key_pos ? key_pos[key] : key;
                ok |= kp >= 0 && kp <= qmax && (window == 0 || kp > qmin - window);
            }
        }
        if (!__syncthreads_or(ok)) continue;
        // scores [16, AC]
        for (int kt = warp; kt < AC / 16; kt += 4) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.f);
            const half* kb = kc + (int64_t)(k0 + kt * 16) * D;
#pragma unroll 4
            for (int d = 0; d < D; d += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
                wmma::load_matrix_sync(a, qs + d, QLD);
                wmma::load_matrix_sync(b, kb + d, D);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(ss + kt * 16, acc, AC, wmma::mem_row_major);
        }
        __syncthreads();
        {
            const int qp = qpos[row];
            float mx = -INFINITY;
            for (int i = sub; i < AC; i += 8) {
                const int key = k0 + i;
                bool v = false;
                if (key < S && qp >= 0) {
                    const int kp = key_pos ? key_pos[key] : key;
                    v = kp >= 0 && kp <= qp && (window == 0 || kp > qp - window);
                }
                const float s = v ? ss[row * AC + i] : -INFINITY;
                ss[row * AC + i] = s;
                mx = fmaxf(mx, s);
            }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 4));
            const float old = row_m[row];
            const float nm = fmaxf(old, mx);
            float l = 0.f;
            for (int i = sub; i < AC; i += 8) {
                const float s = ss[row * AC + i];
                const half p = __float2half_rn(nm == -INFINITY ? 0.f : expf(s - nm));
                ps[row * PLD + i] = p;
                l += __half2float(p);
            }
            l += __shfl_xor_sync(0xffffffffu, l, 1);
            l += __shfl_xor_sync(0xffffffffu, l, 2);
            l += __shfl_xor_sync(0xffffffffu, l, 4);
            __syncwarp();
            if (sub == 0) {
                const float alpha = old == -INFINITY ? 0.f : expf(old - nm);
                row_a[row] = alpha;
                row_m[row] = nm;
                row_l[row] = row_l[row] * alpha + l;
            }
        }
        __syncthreads();
        for (int i = t; i < 16 * D; i += 128) os[i] *= row_a[i / D];
        __syncthreads();
        {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[FR];
#pragma unroll
            for (int f = 0; f < FR; ++f) wmma::load_matrix_sync(acc[f], os + warp * COLS + f * 16, D, wmma::mem_row_major);
            for (int kk = 0; kk < AC; kk += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
                wmma::load_matrix_sync(a, ps + kk, PLD);
                const half* vb = vc + (int64_t)(k0 + kk) * D + warp * COLS;
#pragma unroll
                for (int f = 0; f < FR; ++f) {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b;
                    wmma::load_matrix_sync(b, vb + f * 16, D);
                    wmma::mma_sync(acc[f], a, b, acc[f]);
                }
            }
#pragma unroll
            for (int f = 0; f < FR; ++f) wmma::store_matrix_sync(os + warp * COLS + f * 16, acc[f], D, wmma::mem_row_major);
        }
        __syncthreads();
    }
    float* out = po + ((int64_t)split * Rpad + r0) * D;
    for (int i = t; i < 16 * D; i += 128) out[i] = os[i];
    if (t < 16) {
        pm[(int64_t)split * Rpad + r0 + t] = row_m[t];
        pl[(int64_t)split * Rpad + r0 + t] = row_l[t];
    }
}

// out[r, :] = sum_s e^(m_s - M) o_s / sum_s e^(m_s - M) l_s, splits with l_s == 0 skipped.
template <typename OT>
__global__ void attn_merge(const float* __restrict__ po, const float* __restrict__ pm, const float* __restrict__ pl,
                           int splits, int Rpad, int D, OT* __restrict__ out) {
    extern __shared__ float wts[];
    const int r = blockIdx.x;
    __shared__ float denom;
    if (threadIdx.x == 0) {
        float M = -INFINITY;
        for (int s = 0; s < splits; ++s)
            if (pl[(int64_t)s * Rpad + r] > 0.f) M = fmaxf(M, pm[(int64_t)s * Rpad + r]);
        float L = 0.f;
        for (int s = 0; s < splits; ++s) {
            const float l = pl[(int64_t)s * Rpad + r];
            const float w = l > 0.f ? expf(pm[(int64_t)s * Rpad + r] - M) : 0.f;
            wts[s] = w;
            L += w * l;
        }
        denom = L;
    }
    __syncthreads();
    const float inv = 1.f / denom;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = 0.f;
        for (int s = 0; s < splits; ++s)
            if (wts[s] != 0.f) acc = fmaf(wts[s], po[((int64_t)s * Rpad + r) * D + d], acc);
        out[(int64_t)r * D + d] = cast_out<OT>(acc * inv);
    }
}

}  // namespace g4

// ================================================================================================ launchers
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

using at::Tensor;

namespace {
cudaStream_t stream() { return at::cuda::getCurrentCUDAStream(); }
template <typename T> T* ptr(const Tensor& t) { return reinterpret_cast<T*>(t.data_ptr()); }
template <typename T> const T* cptr(const Tensor& t) { return reinterpret_cast<const T*>(t.data_ptr()); }
const __nv_bfloat16* bptr(const c10::optional<Tensor>& t) {
    return t.has_value() ? reinterpret_cast<const __nv_bfloat16*>(t->data_ptr()) : nullptr;
}
}  // namespace

void w4_rows_cuda(const Tensor& x, int64_t M, const Tensor& codes, const Tensor& scales, const Tensor& rowmul,
                  int64_t N, int64_t K, Tensor& out, int64_t ldo, bool geglu) {
    const dim3 grid((N + g4::WARPS - 1) / g4::WARPS);
    const int64_t ldx = x.stride(0);
#define G4_ROWS(MM, GG) g4::w4_rows<MM, GG><<<grid, 256, 0, stream()>>>(cptr<float>(x), ldx, (int)M, \
        cptr<uint8_t>(codes), cptr<uint8_t>(scales), cptr<float>(rowmul), (int)N, (int)K, ptr<float>(out), ldo)
    if (geglu) {
        if (M <= 1) G4_ROWS(1, true); else if (M <= 4) G4_ROWS(4, true); else if (M <= 8) G4_ROWS(8, true);
        else G4_ROWS(16, true);
    } else {
        if (M <= 1) G4_ROWS(1, false); else if (M <= 4) G4_ROWS(4, false); else if (M <= 8) G4_ROWS(8, false);
        else G4_ROWS(16, false);
    }
#undef G4_ROWS
}

void w4_expert_gateup_cuda(const Tensor& x, int64_t pairs, int64_t slots, const Tensor& local, const Tensor& codes,
                           const Tensor& scales, const Tensor& rowmul, int64_t W, int64_t K, Tensor& act) {
    const dim3 grid((W + g4::WARPS - 1) / g4::WARPS, pairs);
    g4::w4_expert_gateup<<<grid, 256, 0, stream()>>>(cptr<float>(x), x.stride(0), (int)slots, cptr<int>(local),
        cptr<uint8_t>(codes), cptr<uint8_t>(scales), cptr<float>(rowmul), (int)W, (int)K, ptr<float>(act));
}

void w4_expert_down_cuda(const Tensor& act, int64_t M, int64_t slots, const Tensor& local, const Tensor& weight,
                         const Tensor& codes, const Tensor& scales, const Tensor& rowmul, int64_t N, int64_t W,
                         Tensor& out, int64_t ldo) {
    const dim3 grid((N + g4::WARPS - 1) / g4::WARPS, M);
    g4::w4_expert_down<<<grid, 256, 0, stream()>>>(cptr<float>(act), (int)slots, cptr<int>(local),
        cptr<float>(weight), cptr<uint8_t>(codes), cptr<uint8_t>(scales), cptr<float>(rowmul), (int)N, (int)W,
        ptr<float>(out), ldo);
}

void w4_dequant_cuda(const Tensor& codes, const Tensor& scales, int64_t N, int64_t K, Tensor& out) {
    const int64_t total = N * (K / 16);
    g4::w4_dequant<<<(total + 255) / 256, 256, 0, stream()>>>(cptr<uint8_t>(codes), cptr<uint8_t>(scales), N,
        (int)K, ptr<half>(out));
}

void bf16_rows_cuda(const Tensor& x, int64_t M, const Tensor& w, int64_t N, int64_t K, Tensor& out, int64_t ldo,
                    double softcap) {
    const dim3 grid((N + g4::WARPS - 1) / g4::WARPS);
    const int64_t ldx = x.stride(0);
#define G4_BF(MM) g4::bf16_rows<MM><<<grid, 256, 0, stream()>>>(cptr<float>(x), ldx, (int)M, \
        cptr<__nv_bfloat16>(w), (int)N, (int)K, ptr<float>(out), ldo, (float)softcap)
    if (M <= 1) G4_BF(1); else if (M <= 4) G4_BF(4); else if (M <= 8) G4_BF(8); else G4_BF(16);
#undef G4_BF
}

void route_cuda(const Tensor& logits, int64_t M, int64_t E, int64_t topk, const Tensor& scale, int64_t first,
                int64_t count, Tensor& ids, Tensor& weights, Tensor& local) {
    g4::route<<<M, 128, 0, stream()>>>(cptr<float>(logits), (int)E, (int)topk, cptr<__nv_bfloat16>(scale),
        (int)first, (int)count, ptr<int>(ids), ptr<float>(weights), ptr<int>(local));
}

void rms_cuda(const Tensor& x, int64_t M, const c10::optional<Tensor>& w, int64_t D, double eps, double mul,
              Tensor& out) {
    if (out.scalar_type() == at::kHalf)
        g4::rms<half><<<M, g4::NT, 0, stream()>>>(cptr<float>(x), x.stride(0), bptr(w), (int)D, (float)eps,
            (float)mul, ptr<half>(out), out.stride(0));
    else
        g4::rms<float><<<M, g4::NT, 0, stream()>>>(cptr<float>(x), x.stride(0), bptr(w), (int)D, (float)eps,
            (float)mul, ptr<float>(out), out.stride(0));
}

void attn_tail_cuda(const Tensor& o, int64_t M, int64_t world, int64_t pstride, Tensor& h, const Tensor& w_pa, const Tensor& w_pf,
                    const Tensor& w_pf2, const Tensor& w_router, double root, double eps, Tensor& dense,
                    Tensor& expert, Tensor& router) {
    TORCH_CHECK(h.size(1) == g4::MAXD, "attn_tail is compiled for hidden 2816");
#define G4_AT(T) g4::attn_tail<T><<<M, g4::NT, 0, stream()>>>(cptr<float>(o), o.stride(-2), (int)world, pstride, \
        ptr<float>(h), \
        cptr<__nv_bfloat16>(w_pa), cptr<__nv_bfloat16>(w_pf), cptr<__nv_bfloat16>(w_pf2), \
        cptr<__nv_bfloat16>(w_router), (float)root, (float)eps, ptr<T>(dense), ptr<T>(expert), ptr<float>(router))
    if (dense.scalar_type() == at::kHalf) G4_AT(half); else G4_AT(float);
#undef G4_AT
}

void ffn_tail_cuda(const Tensor& red, int64_t M, int64_t world, int64_t pstride, Tensor& h, const Tensor& w1, const Tensor& w2, const Tensor& w_out,
                   const Tensor& scalar, const Tensor& w_next, double eps, Tensor& next,
                   const c10::optional<Tensor>& tap) {
    TORCH_CHECK(h.size(1) == g4::MAXD, "ffn_tail is compiled for hidden 2816");
    float* tp = tap.has_value() ? reinterpret_cast<float*>(tap->data_ptr()) : nullptr;
    const int64_t ldt = tap.has_value() ? tap->stride(0) : 0;
#define G4_FT(T) g4::ffn_tail<T><<<M, g4::NT, 0, stream()>>>(cptr<float>(red), (int)world, pstride, ptr<float>(h), \
        cptr<__nv_bfloat16>(w1), cptr<__nv_bfloat16>(w2), cptr<__nv_bfloat16>(w_out), cptr<__nv_bfloat16>(scalar), \
        cptr<__nv_bfloat16>(w_next), (float)eps, ptr<T>(next), tp, ldt)
    if (next.scalar_type() == at::kHalf) G4_FT(half); else G4_FT(float);
#undef G4_FT
}

void embed_cuda(const Tensor& ids, int64_t M, const Tensor& table, int64_t D, double mul, Tensor& out) {
    g4::embed<<<M, 256, 0, stream()>>>(cptr<int64_t>(ids), cptr<__nv_bfloat16>(table), (int)D, (float)mul,
        ptr<float>(out));
}

void qkv_prep_cuda(const Tensor& x, int64_t M, int64_t Hq, int64_t D, int64_t v_mode, const Tensor& qw, const Tensor& kw,
                   const Tensor& inv_freq, const Tensor& pos0, Tensor& qout, Tensor& kc, Tensor& vc,
                   const c10::optional<Tensor>& key_pos, int64_t ring, double eps) {
    const dim3 grid(M, Hq + 1);
#define G4_QP(T) g4::qkv_prep<T><<<grid, D / 2, 0, stream()>>>(cptr<float>(x), x.stride(0), (int)Hq, (int)D, \
        (int)v_mode, cptr<__nv_bfloat16>(qw), cptr<__nv_bfloat16>(kw), cptr<float>(inv_freq), cptr<int>(pos0), \
        ptr<T>(qout), ptr<half>(kc), ptr<half>(vc), \
        key_pos.has_value() ? reinterpret_cast<int*>(key_pos->data_ptr()) : nullptr, (int)ring, (float)eps)
    if (qout.scalar_type() == at::kHalf) G4_QP(half); else G4_QP(float);
#undef G4_QP
}

namespace {
template <int D>
void attn_launch(const Tensor& q, int64_t R, int64_t H, const Tensor& kc, const Tensor& vc,
                 const c10::optional<Tensor>& key_pos, int64_t S, const Tensor& pos0, int64_t window, int64_t noncausal,
                 int64_t chunks, int64_t splits, Tensor& po, Tensor& pm, Tensor& pl, int64_t Rpad) {
    static bool configured = false;
    const size_t bytes = g4::AttnSmem<D>::bytes;
    if (!configured) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(g4::attn_split<D>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes));
        configured = true;
    }
    const int64_t cps = (chunks + splits - 1) / splits;
    const dim3 grid(splits, (R + 15) / 16);
    g4::attn_split<D><<<grid, 128, bytes, stream()>>>(cptr<half>(q), (int)R, (int)H, cptr<half>(kc), cptr<half>(vc),
        key_pos.has_value() ? cptr<int>(*key_pos) : nullptr, (int)S, cptr<int>(pos0), (int)window, (int)noncausal,
        (int)chunks,
        (int)cps, ptr<float>(po), ptr<float>(pm), ptr<float>(pl), (int)Rpad);
}
}  // namespace

void attn_cuda(const Tensor& q, int64_t R, int64_t H, int64_t D, const Tensor& kc, const Tensor& vc,
               const c10::optional<Tensor>& key_pos, int64_t S, const Tensor& pos0, int64_t window, int64_t noncausal,
               int64_t chunks, int64_t splits, Tensor& po, Tensor& pm, Tensor& pl, int64_t Rpad, Tensor& out) {
    if (D == 128) attn_launch<128>(q, R, H, kc, vc, key_pos, S, pos0, window, noncausal, chunks, splits, po, pm, pl, Rpad);
    else if (D == 256) attn_launch<256>(q, R, H, kc, vc, key_pos, S, pos0, window, noncausal, chunks, splits, po, pm, pl, Rpad);
    else if (D == 512) attn_launch<512>(q, R, H, kc, vc, key_pos, S, pos0, window, noncausal, chunks, splits, po, pm, pl, Rpad);
    else TORCH_CHECK(false, "attention head width must be 128, 256 or 512");
    const size_t bytes = splits * sizeof(float);
    if (out.scalar_type() == at::kHalf)
        g4::attn_merge<half><<<R, 128, bytes, stream()>>>(cptr<float>(po), cptr<float>(pm), cptr<float>(pl),
            (int)splits, (int)Rpad, (int)D, ptr<half>(out));
    else
        g4::attn_merge<float><<<R, 128, bytes, stream()>>>(cptr<float>(po), cptr<float>(pm), cptr<float>(pl),
            (int)splits, (int)Rpad, (int)D, ptr<float>(out));
}
