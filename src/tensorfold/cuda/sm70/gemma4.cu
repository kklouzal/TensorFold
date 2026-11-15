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
// Lanes split into 32 / L subgroups; a subgroup owns P physical weight rows and strides their 16-element groups
// by L, so every activation load is reused across the P rows it multiplies. L and P depend only on the
// matrix shape, never on the number of activation rows M, so each row's arithmetic is row-count invariant.
template <int L>
__device__ __forceinline__ float sub_sum(float v) {
#pragma unroll
    for (int o = L / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

template <int MAXM, int P, int L>
__device__ __forceinline__ void w4_dot(const float* __restrict__ x, int64_t ldx, int M,
        const uint8_t* const (&codes)[P], const uint8_t* const (&scales)[P], int groups, int sl,
        float (&acc)[P][MAXM]) {
#pragma unroll
    for (int p = 0; p < P; ++p)
#pragma unroll
        for (int m = 0; m < MAXM; ++m) acc[p][m] = 0.f;
    for (int g = sl; g < groups; g += L) {
        float f[P][16], sc[P];
#pragma unroll
        for (int p = 0; p < P; ++p) {
            const uint2 w = __ldg(reinterpret_cast<const uint2*>(codes[p]) + g);
            sc[p] = scale_e4m3(__ldg(scales[p] + g));
            decode8(w.x, f[p]);
            decode8(w.y, f[p] + 8);
        }
#pragma unroll
        for (int m = 0; m < MAXM; ++m) {
            if (m < M) {
                const float4* xp = reinterpret_cast<const float4*>(x + m * ldx + g * 16);
                float4 a[4];
#pragma unroll
                for (int q = 0; q < 4; ++q) a[q] = __ldg(xp + q);
#pragma unroll
                for (int p = 0; p < P; ++p) {
                    float d = 0.f;
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        d = fmaf(a[q].x, f[p][4 * q], d);
                        d = fmaf(a[q].y, f[p][4 * q + 1], d);
                        d = fmaf(a[q].z, f[p][4 * q + 2], d);
                        d = fmaf(a[q].w, f[p][4 * q + 3], d);
                    }
                    acc[p][m] = fmaf(d, sc[p], acc[p][m]);
                }
            }
        }
    }
}

// out[m, n] = rowmul[n] * sum_g scale[n,g] * dot(x[m, 16g:16g+16], E2M1[n, 16g:16g+16]), m < M <= MAXM.
// GEGLU: output n reads gate row n and up row n + N and writes gelu_tanh(gate) * up.
// A subgroup owns R outputs (P = R or 2R physical rows).
template <int MAXM, bool GEGLU, int R, int L>
__global__ void __launch_bounds__(256) w4_rows(const float* __restrict__ x, int64_t ldx, int M,
        const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales, const float* __restrict__ rowmul,
        int N, int K, float* __restrict__ out, int64_t ldo) {
    constexpr int SUBS = 32 / L, P = GEGLU ? 2 * R : R;
    const int lane = threadIdx.x & 31, sl = lane % L;
    const int n0 = ((blockIdx.x * WARPS + (threadIdx.x >> 5)) * SUBS + lane / L) * R;
    // No early exit: half-warp subgroups share shuffles. Out-of-range outputs read row N - 1 and store nothing.
    const int groups = K >> 4;
    const uint8_t* cr[P];
    const uint8_t* sr[P];
#pragma unroll
    for (int p = 0; p < P; ++p) {
        const int n = min(n0 + (GEGLU ? p % R : p), N - 1) + (GEGLU && p >= R ? N : 0);
        cr[p] = codes + (int64_t)n * (K >> 1);
        sr[p] = scales + (int64_t)n * groups;
    }
    float acc[P][MAXM];
    w4_dot<MAXM, P, L>(x, ldx, M, cr, sr, groups, sl, acc);
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int n = n0 + r;
#pragma unroll
        for (int m = 0; m < MAXM; ++m) {
            if (m < M) {
                float v0 = sub_sum<L>(acc[r][m]);
                if (GEGLU) {
                    const float v1 = sub_sum<L>(acc[r + R][m]);
                    if (n < N) v0 = gelu_tanh(v0 * __ldg(rowmul + n)) * (v1 * __ldg(rowmul + n + N));
                } else if (n < N) {
                    v0 *= __ldg(rowmul + n);
                }
                if (sl == 0 && n < N) out[m * ldo + n] = v0;
            }
        }
    }
}

// Expert gate/up GEGLU over (row, slot) pairs: act[p, j] for local expert local[p] (>= 0), zeros skipped.
// Weights are stacked per local expert: codes [E, 2W, K/2], scales [E, 2W, K/16], rowmul [E, 2W].
template <int R>
__global__ void __launch_bounds__(256) w4_expert_gateup(const float* __restrict__ x, int64_t ldx, int slots,
        const int* __restrict__ local, const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales,
        const float* __restrict__ rowmul, int W, int K, float* __restrict__ act) {
    const int pair = blockIdx.y;
    const int e = local[pair];
    if (e < 0) return;
    const int lane = threadIdx.x & 31;
    const int j0 = (blockIdx.x * WARPS + (threadIdx.x >> 5)) * R;
    if (j0 >= W) return;
    const int groups = K >> 4;
    const int64_t base = (int64_t)e * 2 * W;
    const uint8_t* cr[2 * R];
    const uint8_t* sr[2 * R];
#pragma unroll
    for (int p = 0; p < 2 * R; ++p) {
        const int64_t row = base + min(j0 + p % R, W - 1) + (p >= R ? W : 0);
        cr[p] = codes + row * (K >> 1);
        sr[p] = scales + row * groups;
    }
    float acc[2 * R][1];
    w4_dot<1, 2 * R, 32>(x + (pair / slots) * ldx, ldx, 1, cr, sr, groups, lane, acc);
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int j = j0 + r;
        const float g = sub_sum<32>(acc[r][0]), u = sub_sum<32>(acc[r + R][0]);
        if (lane == 0 && j < W)
            act[(int64_t)pair * W + j] = gelu_tanh(g * __ldg(rowmul + base + j)) * (u * __ldg(rowmul + base + j + W));
    }
}

// Expert down with the routing combine: out[m, n] = sum_s weight[m,s] * down_e(act[m*S+s])[n], slots in
// ascending order, non-local slots skipped. Down weights: codes [E, N, W/2], scales [E, N, W/16], rowmul [E, N].
// Half-warps own one output each (W / 16 <= 16 groups).
__global__ void __launch_bounds__(256) w4_expert_down(const float* __restrict__ act, int slots,
        const int* __restrict__ local, const float* __restrict__ weight, const uint8_t* __restrict__ codes,
        const uint8_t* __restrict__ scales, const float* __restrict__ rowmul, int N, int W,
        float* __restrict__ out, int64_t ldo) {
    const int m = blockIdx.y;
    const int lane = threadIdx.x & 31, sl = lane & 15;
    const int out_n = (blockIdx.x * WARPS + (threadIdx.x >> 5)) * 2 + (lane >> 4);
    const int n = min(out_n, N - 1);      // both half-warps stay converged for the shuffles
    const int groups = W >> 4;
    float total = 0.f;
    for (int s = 0; s < slots; ++s) {
        const int pair = m * slots + s;
        const int e = local[pair];
        if (e < 0) continue;
        const int64_t row = (int64_t)e * N + n;
        const uint8_t* cr[1] = {codes + row * (W >> 1)};
        const uint8_t* sr[1] = {scales + row * groups};
        float acc[1][1];
        w4_dot<1, 1, 16>(act + (int64_t)pair * W, W, 1, cr, sr, groups, sl, acc);
        total = fmaf(__ldg(weight + pair), sub_sum<16>(acc[0][0]) * __ldg(rowmul + row), total);
    }
    if (sl == 0 && out_n < N) out[m * ldo + out_n] = total;
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
// out[m, n] = dot(x[m], W[n]) (FP32), optional final-logit softcap.  K % 8 == 0. A warp owns R outputs and reuses
// each activation load across them (R from the shape only: row-count invariant).
template <int MAXM, int R>
__global__ void __launch_bounds__(256) bf16_rows(const float* __restrict__ x, int64_t ldx, int M,
        const __nv_bfloat16* __restrict__ w, int N, int K, float* __restrict__ out, int64_t ldo, float softcap) {
    const int n0 = (blockIdx.x * WARPS + (threadIdx.x >> 5)) * R, lane = threadIdx.x & 31;
    if (n0 >= N) return;
    const uint4* rows[R];
#pragma unroll
    for (int r = 0; r < R; ++r) rows[r] = reinterpret_cast<const uint4*>(w + (int64_t)min(n0 + r, N - 1) * K);
    float acc[R][MAXM];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < MAXM; ++m) acc[r][m] = 0.f;
    for (int c = lane; c < (K >> 3); c += 32) {
        float f[R][8];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint4 v = __ldg(rows[r] + c);
            const uint32_t words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                f[r][2 * q] = __uint_as_float(words[q] << 16);
                f[r][2 * q + 1] = __uint_as_float(words[q] & 0xffff0000u);
            }
        }
#pragma unroll
        for (int m = 0; m < MAXM; ++m) {
            if (m < M) {
                const float4* xp = reinterpret_cast<const float4*>(x + m * ldx + c * 8);
                const float4 a = __ldg(xp), b = __ldg(xp + 1);
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    float d = acc[r][m];
                    d = fmaf(a.x, f[r][0], d); d = fmaf(a.y, f[r][1], d); d = fmaf(a.z, f[r][2], d);
                    d = fmaf(a.w, f[r][3], d); d = fmaf(b.x, f[r][4], d); d = fmaf(b.y, f[r][5], d);
                    d = fmaf(b.z, f[r][6], d); d = fmaf(b.w, f[r][7], d);
                    acc[r][m] = d;
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int m = 0; m < MAXM; ++m) {
            if (m < M) {
                float v = warp_sum(acc[r][m]);
                if (softcap > 0.f) v = tanhf(v / softcap) * softcap;
                if (lane == 0 && n0 + r < N) out[m * ldo + n0 + r] = v;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------- INT8 GEMV
// out[m, n] = scale[n] * dot(x[m], q[n]) with symmetric per-row INT8 q (private W8A16 proposal copies). K % 16 == 0.
template <int MAXM, int R>
__global__ void __launch_bounds__(256) i8_rows(const float* __restrict__ x, int64_t ldx, int M,
        const int8_t* __restrict__ w, const float* __restrict__ scale, int N, int K, float* __restrict__ out,
        int64_t ldo) {
    const int n0 = (blockIdx.x * WARPS + (threadIdx.x >> 5)) * R, lane = threadIdx.x & 31;
    if (n0 >= N) return;
    const uint4* rows[R];
#pragma unroll
    for (int r = 0; r < R; ++r) rows[r] = reinterpret_cast<const uint4*>(w + (int64_t)min(n0 + r, N - 1) * K);
    float acc[R][MAXM];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < MAXM; ++m) acc[r][m] = 0.f;
    for (int c = lane; c < (K >> 4); c += 32) {
        float f[R][16];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint4 v = __ldg(rows[r] + c);
            const uint32_t words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int q = 0; q < 16; ++q) f[r][q] = (float)(int8_t)(words[q >> 2] >> (8 * (q & 3)));
        }
#pragma unroll
        for (int m = 0; m < MAXM; ++m) {
            if (m < M) {
                const float4* xp = reinterpret_cast<const float4*>(x + m * ldx + c * 16);
                float4 a[4];
#pragma unroll
                for (int q = 0; q < 4; ++q) a[q] = __ldg(xp + q);
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    float d = acc[r][m];
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        d = fmaf(a[q].x, f[r][4 * q], d); d = fmaf(a[q].y, f[r][4 * q + 1], d);
                        d = fmaf(a[q].z, f[r][4 * q + 2], d); d = fmaf(a[q].w, f[r][4 * q + 3], d);
                    }
                    acc[r][m] = d;
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int m = 0; m < MAXM; ++m) {
            if (m < M) {
                const float v = warp_sum(acc[r][m]);
                if (lane == 0 && n0 + r < N) out[m * ldo + n0 + r] = v * __ldg(scale + n0 + r);
            }
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

// Fused router: logits[m, e] = dot(x[m], W[e]) (BF16 weights, FP32), then the last block to finish routes every
// row exactly as `route` does. Block b owns experts 2b, 2b+1; warp w covers K quarter (w & 3) of expert 2b + (w >> 2).
// `counter` is a zeroed device int the last block resets (graph-replay safe).
template <int MAXM>
__global__ void __launch_bounds__(256) router(const float* __restrict__ x, int64_t ldx, int M,
        const __nv_bfloat16* __restrict__ w, int E, int K, float* __restrict__ logits, int* __restrict__ counter,
        int topk, const __nv_bfloat16* __restrict__ scale, int first, int count, int* __restrict__ ids,
        float* __restrict__ weights, int* __restrict__ local) {
    __shared__ float part[8][MAXM];
    __shared__ bool last;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, t = threadIdx.x;
    const int e = blockIdx.x * 2 + (warp >> 2), quarter = warp & 3;
    const int chunks = K >> 3, per = (chunks + 3) / 4;
    const int c0 = quarter * per, c1 = min(chunks, c0 + per);
    float acc[MAXM];
#pragma unroll
    for (int m = 0; m < MAXM; ++m) acc[m] = 0.f;
    if (e < E) {
        const uint4* row = reinterpret_cast<const uint4*>(w + (int64_t)e * K);
        for (int c = c0 + lane; c < c1; c += 32) {
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
    }
#pragma unroll
    for (int m = 0; m < MAXM; ++m) {
        const float v = warp_sum(acc[m]);
        if (lane == 0) part[warp][m] = v;
    }
    __syncthreads();
    if (t < 2 * MAXM) {
        const int half = t / MAXM, m = t % MAXM, ee = blockIdx.x * 2 + half;
        if (m < M && ee < E) {
            logits[m * E + ee] = ((part[half * 4][m] + part[half * 4 + 1][m]) + part[half * 4 + 2][m])
                                 + part[half * 4 + 3][m];
        }
    }
    __threadfence();
    __syncthreads();
    if (t == 0) last = atomicAdd(counter, 1) == (int)gridDim.x - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    // Route each row: softmax over E, stable top-k by probability, renormalized, times per_expert_scale.
    __shared__ float prob[256];
    __shared__ float red[8];
    __shared__ int redi[8];
    for (int m = 0; m < M; ++m) {
        const float l = t < E ? __ldcg(logits + m * E + t) : -INFINITY;
        float mx = l;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        if (lane == 0) red[warp] = mx;
        __syncthreads();
        mx = red[0];
        for (int i = 1; i < 8; ++i) mx = fmaxf(mx, red[i]);
        __syncthreads();
        const float pe = t < E ? expf(l - mx) : 0.f;
        float sum = warp_sum(pe);
        if (lane == 0) red[warp] = sum;
        __syncthreads();
        sum = 0.f;
        for (int i = 0; i < 8; ++i) sum += red[i];
        __syncthreads();
        prob[t] = t < E ? pe / sum : -1.f;
        __syncthreads();
        float chosen = 0.f, selected_total = 0.f;
        int picked = -1;
        for (int k = 0; k < topk; ++k) {
            float v = prob[t];
            int idx = t;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, v, o);
                const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
                if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
            }
            if (lane == 0) { red[warp] = v; redi[warp] = idx; }
            __syncthreads();
            float bv = red[0];
            int bi = redi[0];
            for (int i = 1; i < 8; ++i)
                if (red[i] > bv || (red[i] == bv && redi[i] < bi)) { bv = red[i]; bi = redi[i]; }
            selected_total += bv;
            if (t == k) { chosen = bv; picked = bi; }
            __syncthreads();
            if (t == bi) prob[t] = -2.f;
            __syncthreads();
        }
        if (t < topk) {
            ids[m * topk + t] = picked;
            weights[m * topk + t] = chosen / selected_total * bf16f(scale[picked]);
            local[m * topk + t] = (picked >= first && picked < first + count) ? picked - first : -1;
        }
        __syncthreads();
    }
    if (t == 0) *counter = 0;
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

constexpr int NT = 256;       // norm block threads
constexpr int MAXD = 2816;    // hidden width (11 values per thread)

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

// Fused tails, one 704-thread block per row: thread t owns hidden elements [4t, 4t + 4) (D == MAXD). Rank
// partials are summed in rank order with all `world` loads issued before the adds.
constexpr int TT = MAXD / 4;   // 704 threads

__device__ __forceinline__ float4 parts4(const float* __restrict__ p, int world, int64_t stride) {
    float4 v[8];
#pragma unroll
    for (int r = 0; r < 8; ++r)
        if (r < world) v[r] = __ldg(reinterpret_cast<const float4*>(p + r * stride));
    float4 acc = v[0];
#pragma unroll
    for (int r = 1; r < 8; ++r)
        if (r < world) { acc.x += v[r].x; acc.y += v[r].y; acc.z += v[r].z; acc.w += v[r].w; }
    return acc;
}

__device__ __forceinline__ float sumsq4(float4 v) { return v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w; }

__device__ __forceinline__ float4 w4(const __nv_bfloat16* __restrict__ w, int d) {
    const uint2 raw = __ldg(reinterpret_cast<const uint2*>(w + d));
    return make_float4(__uint_as_float(raw.x << 16), __uint_as_float(raw.x & 0xffff0000u),
                       __uint_as_float(raw.y << 16), __uint_as_float(raw.y & 0xffff0000u));
}

__device__ __forceinline__ float4 mul4(float4 a, float s, float4 w) {
    return make_float4(a.x * s * w.x, a.y * s * w.y, a.z * s * w.z, a.w * s * w.w);
}

template <typename OT> __device__ __forceinline__ void store4(OT* p, float4 v);
template <> __device__ __forceinline__ void store4<float>(float* p, float4 v) { *reinterpret_cast<float4*>(p) = v; }
template <> __device__ __forceinline__ void store4<half>(half* p, float4 v) {
    __half2 a = __floats2half2_rn(v.x, v.y), b = __floats2half2_rn(v.z, v.w);
    uint2 raw = make_uint2(*reinterpret_cast<uint32_t*>(&a), *reinterpret_cast<uint32_t*>(&b));
    *reinterpret_cast<uint2*>(p) = raw;
}

__device__ __forceinline__ float tsum(float v, float* scratch) {
    v = warp_sum(v);
    const int w = threadIdx.x >> 5;
    __syncthreads();
    if ((threadIdx.x & 31) == 0) scratch[w] = v;
    __syncthreads();
    float total = 0.f;
#pragma unroll
    for (int i = 0; i < TT / 32; ++i) total += scratch[i];
    return total;
}

// Attention tail: h += norm(o, w_pa); dense = norm(h, w_pf); expert = norm(h, w_pf2);
// router = rmsnorm_unweighted(h) * root * router_scale.
template <typename OT>
__global__ void __launch_bounds__(TT) attn_tail(const float* __restrict__ o, int64_t ldo_in, int world,
        int64_t pstride, float* __restrict__ h,
        const __nv_bfloat16* __restrict__ w_pa, const __nv_bfloat16* __restrict__ w_pf,
        const __nv_bfloat16* __restrict__ w_pf2, const __nv_bfloat16* __restrict__ w_router, float root, float eps,
        OT* __restrict__ dense, OT* __restrict__ expert, float* __restrict__ router) {
    __shared__ float scratch[TT / 32];
    const int m = blockIdx.x, d = threadIdx.x * 4;
    const float4 v = parts4(o + m * ldo_in + d, world, pstride);
    float* hr = h + (int64_t)m * MAXD + d;
    const float4 h0 = *reinterpret_cast<const float4*>(hr);
    const float inv = rsqrtf(tsum(sumsq4(v), scratch) / MAXD + eps);
    const float4 t = mul4(v, inv, w4(w_pa, d));
    const float4 hn = make_float4(h0.x + t.x, h0.y + t.y, h0.z + t.z, h0.w + t.w);
    *reinterpret_cast<float4*>(hr) = hn;
    const float inv2 = rsqrtf(tsum(sumsq4(hn), scratch) / MAXD + eps);
    store4<OT>(dense + (int64_t)m * MAXD + d, mul4(hn, inv2, w4(w_pf, d)));
    store4<OT>(expert + (int64_t)m * MAXD + d, mul4(hn, inv2, w4(w_pf2, d)));
    store4<float>(router + (int64_t)m * MAXD + d, mul4(hn, inv2 * root, w4(w_router, d)));
}

// FFN tail: s = norm(dense, w1) + norm(routed, w2); h = (h + norm(s, w_out)) * scalar; next = norm(h, w_next).
// red: [M, 2, D] (dense partial sums, routed partial sums). tap (nullable) receives h.
template <typename OT>
__global__ void __launch_bounds__(TT) ffn_tail(const float* __restrict__ red, int world, int64_t pstride,
        float* __restrict__ h,
        const __nv_bfloat16* __restrict__ w1, const __nv_bfloat16* __restrict__ w2,
        const __nv_bfloat16* __restrict__ w_out, const __nv_bfloat16* __restrict__ scalar,
        const __nv_bfloat16* __restrict__ w_next, float eps, OT* __restrict__ next, float* __restrict__ tap,
        int64_t ldtap) {
    __shared__ float scratch[TT / 32];
    const int m = blockIdx.x, d = threadIdx.x * 4;
    const float* dn = red + (int64_t)m * 2 * MAXD + d;
    const float4 a = parts4(dn, world, pstride), b = parts4(dn + MAXD, world, pstride);
    float* hr = h + (int64_t)m * MAXD + d;
    const float4 h0 = *reinterpret_cast<const float4*>(hr);
    const float ia = rsqrtf(tsum(sumsq4(a), scratch) / MAXD + eps);
    const float ib = rsqrtf(tsum(sumsq4(b), scratch) / MAXD + eps);
    const float4 x = mul4(a, ia, w4(w1, d)), y = mul4(b, ib, w4(w2, d));
    const float4 sv = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
    const float is = rsqrtf(tsum(sumsq4(sv), scratch) / MAXD + eps);
    const float sc = bf16f(scalar[0]);
    const float4 u = mul4(sv, is, w4(w_out, d));
    const float4 hn = make_float4((h0.x + u.x) * sc, (h0.y + u.y) * sc, (h0.z + u.z) * sc, (h0.w + u.w) * sc);
    *reinterpret_cast<float4*>(hr) = hn;
    if (tap) *reinterpret_cast<float4*>(tap + m * ldtap + d) = hn;
    const float in = rsqrtf(tsum(sumsq4(hn), scratch) / MAXD + eps);
    store4<OT>(next + (int64_t)m * MAXD + d, mul4(hn, in, w4(w_next, d)));
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
// at position p when 0 <= kp <= p and (window == 0 or kp > p - window). Scores are scaled by `scale` (FP32)
// before masking (Gemma 4 target: 1.0; Qwen3-style DFlash companion: head_dim^-0.5).
// Block (split, tile) visits chunks [split*cps, min(chunks, (split+1)*cps)) of AC keys (64 for one 16-row
// tile, else 256) with an online softmax
// and writes unnormalized partials po[split, r, :], pm/pl[split, r]; attn_merge folds splits in order.

template <int D, int AC>
struct AttnSmem {
    static constexpr int QLD = D + 8, PLD = AC + 8;
    static constexpr size_t bytes = 16 * QLD * 2 + 16 * AC * 4 + 16 * PLD * 2 + 16 * D * 4;
};

template <int D, int AC>
__global__ void __launch_bounds__(128) attn_split(const half* __restrict__ q, int R, int H, const half* __restrict__ kc,
        const half* __restrict__ vc, const int* __restrict__ key_pos, int S, const int* __restrict__ pos0, int window,
        int noncausal, float scale, int chunks, int cps, float* __restrict__ po, float* __restrict__ pm, float* __restrict__ pl, int Rpad) {
    constexpr int QLD = AttnSmem<D, AC>::QLD, PLD = AttnSmem<D, AC>::PLD;
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
    if (!key_pos && split * cps * AC > base + (R - 1) / H) {
        // Linear cache: every key of this split is in the future of every query row in the call.
        if (t < 16) {
            pm[(int64_t)split * Rpad + r0 + t] = -INFINITY;
            pl[(int64_t)split * Rpad + r0 + t] = 0.f;
        }
        return;
    }
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
    bool processed = false;
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
        processed = true;
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
                const float s = v ? ss[row * AC + i] * scale : -INFINITY;
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
    if (processed) {        // otherwise l == 0 and the merge never reads this split's o
        float* out = po + ((int64_t)split * Rpad + r0) * D;
        for (int i = t; i < 16 * D; i += 128) out[i] = os[i];
    }
    if (t < 16) {
        pm[(int64_t)split * Rpad + r0 + t] = row_m[t];
        pl[(int64_t)split * Rpad + r0 + t] = row_l[t];
    }
}

// out[r, :] = sum_s e^(m_s - M) o_s / sum_s e^(m_s - M) l_s, splits with l_s == 0 skipped.
template <typename OT>
__global__ void attn_merge(const float* __restrict__ po, const float* __restrict__ pm, const float* __restrict__ pl,
                           int splits, int Rpad, int D, OT* __restrict__ out) {
    extern __shared__ float wts[];          // [splits] fold weights (0: an empty split)
    __shared__ float red[4];
    const int r = blockIdx.x, t = threadIdx.x;
    float M = -INFINITY;
    for (int s = t; s < splits; s += blockDim.x)
        if (pl[(int64_t)s * Rpad + r] > 0.f) M = fmaxf(M, pm[(int64_t)s * Rpad + r]);
    M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, 16)); M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, 8));
    M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, 4)); M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, 2));
    M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, 1));
    if ((t & 31) == 0) red[t >> 5] = M;
    __syncthreads();
    M = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
    for (int s = t; s < splits; s += blockDim.x) {
        const float l = pl[(int64_t)s * Rpad + r];
        wts[s] = l > 0.f ? expf(pm[(int64_t)s * Rpad + r] - M) : 0.f;
    }
    __syncthreads();
    float L = 0.f;                         // split order, every thread the same
    for (int s = 0; s < splits; ++s)
        if (wts[s] != 0.f) L = fmaf(wts[s], pl[(int64_t)s * Rpad + r], L);
    const float inv = 1.f / L;
    for (int d = t; d < D; d += blockDim.x) {
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

namespace {
template <bool GEGLU, int R, int L>
void w4_rows_launch(const Tensor& x, int64_t M, const Tensor& codes, const Tensor& scales, const Tensor& rowmul,
                    int64_t N, int64_t K, Tensor& out, int64_t ldo) {
    const int64_t per_block = g4::WARPS * (32 / L) * R;
    const dim3 grid((N + per_block - 1) / per_block);
    const int64_t ldx = x.stride(0);
#define G4_ROWS(MM) g4::w4_rows<MM, GEGLU, R, L><<<grid, 256, 0, stream()>>>(cptr<float>(x), ldx, (int)M, \
        cptr<uint8_t>(codes), cptr<uint8_t>(scales), cptr<float>(rowmul), (int)N, (int)K, ptr<float>(out), ldo)
    if (M <= 1) G4_ROWS(1); else if (M <= 4) G4_ROWS(4); else if (M <= 8) G4_ROWS(8); else G4_ROWS(16);
#undef G4_ROWS
}
}  // namespace

// Shape-only tiling (row-count invariant): half-warp subgroups for <= 16 groups, else full warps; outputs per
// subgroup reuse each activation load (4 plain, 2 GEGLU pairs), dropping to 2/1 for narrow matrices.
void w4_rows_cuda(const Tensor& x, int64_t M, const Tensor& codes, const Tensor& scales, const Tensor& rowmul,
                  int64_t N, int64_t K, Tensor& out, int64_t ldo, bool geglu) {
    const bool half_warp = K / 16 <= 16;
    const bool wide = N >= 1024;
    if (geglu) {
        if (half_warp) w4_rows_launch<true, 1, 16>(x, M, codes, scales, rowmul, N, K, out, ldo);
        else if (wide) w4_rows_launch<true, 2, 32>(x, M, codes, scales, rowmul, N, K, out, ldo);
        else w4_rows_launch<true, 1, 32>(x, M, codes, scales, rowmul, N, K, out, ldo);
    } else {
        if (half_warp) w4_rows_launch<false, 2, 16>(x, M, codes, scales, rowmul, N, K, out, ldo);
        else if (wide) w4_rows_launch<false, 4, 32>(x, M, codes, scales, rowmul, N, K, out, ldo);
        else w4_rows_launch<false, 2, 32>(x, M, codes, scales, rowmul, N, K, out, ldo);
    }
}

void w4_expert_gateup_cuda(const Tensor& x, int64_t pairs, int64_t slots, const Tensor& local, const Tensor& codes,
                           const Tensor& scales, const Tensor& rowmul, int64_t W, int64_t K, Tensor& act) {
    constexpr int R = 2;
    const dim3 grid((W + g4::WARPS * R - 1) / (g4::WARPS * R), pairs);
    g4::w4_expert_gateup<R><<<grid, 256, 0, stream()>>>(cptr<float>(x), x.stride(0), (int)slots, cptr<int>(local),
        cptr<uint8_t>(codes), cptr<uint8_t>(scales), cptr<float>(rowmul), (int)W, (int)K, ptr<float>(act));
}

void w4_expert_down_cuda(const Tensor& act, int64_t M, int64_t slots, const Tensor& local, const Tensor& weight,
                         const Tensor& codes, const Tensor& scales, const Tensor& rowmul, int64_t N, int64_t W,
                         Tensor& out, int64_t ldo) {
    TORCH_CHECK(W / 16 <= 16, "expert down keeps at most 16 input groups per half-warp");
    const dim3 grid((N + g4::WARPS * 2 - 1) / (g4::WARPS * 2), M);
    g4::w4_expert_down<<<grid, 256, 0, stream()>>>(cptr<float>(act), (int)slots, cptr<int>(local),
        cptr<float>(weight), cptr<uint8_t>(codes), cptr<uint8_t>(scales), cptr<float>(rowmul), (int)N, (int)W,
        ptr<float>(out), ldo);
}

void w4_dequant_cuda(const Tensor& codes, const Tensor& scales, int64_t N, int64_t K, Tensor& out) {
    const int64_t total = N * (K / 16);
    g4::w4_dequant<<<(total + 255) / 256, 256, 0, stream()>>>(cptr<uint8_t>(codes), cptr<uint8_t>(scales), N,
        (int)K, ptr<half>(out));
}

namespace {
template <int R>
void bf16_launch(const Tensor& x, int64_t M, const Tensor& w, int64_t N, int64_t K, Tensor& out, int64_t ldo,
                 double softcap) {
    const dim3 grid((N + g4::WARPS * R - 1) / (g4::WARPS * R));
    const int64_t ldx = x.stride(0);
#define G4_BF(MM) g4::bf16_rows<MM, R><<<grid, 256, 0, stream()>>>(cptr<float>(x), ldx, (int)M, \
        cptr<__nv_bfloat16>(w), (int)N, (int)K, ptr<float>(out), ldo, (float)softcap)
    if (M <= 1) G4_BF(1); else if (M <= 4) G4_BF(4); else if (M <= 8) G4_BF(8); else G4_BF(16);
#undef G4_BF
}
template <int R>
void i8_launch(const Tensor& x, int64_t M, const Tensor& w, const Tensor& scale, int64_t N, int64_t K, Tensor& out,
               int64_t ldo) {
    const dim3 grid((N + g4::WARPS * R - 1) / (g4::WARPS * R));
    const int64_t ldx = x.stride(0);
#define G4_I8(MM) g4::i8_rows<MM, R><<<grid, 256, 0, stream()>>>(cptr<float>(x), ldx, (int)M, cptr<int8_t>(w), \
        cptr<float>(scale), (int)N, (int)K, ptr<float>(out), ldo)
    if (M <= 1) G4_I8(1); else if (M <= 4) G4_I8(4); else if (M <= 8) G4_I8(8); else G4_I8(16);
#undef G4_I8
}
}  // namespace

void bf16_rows_cuda(const Tensor& x, int64_t M, const Tensor& w, int64_t N, int64_t K, Tensor& out, int64_t ldo,
                    double softcap) {
    if (N >= 1024) bf16_launch<4>(x, M, w, N, K, out, ldo, softcap);
    else bf16_launch<1>(x, M, w, N, K, out, ldo, softcap);
}

void i8_rows_cuda(const Tensor& x, int64_t M, const Tensor& w, const Tensor& scale, int64_t N, int64_t K, Tensor& out,
                  int64_t ldo) {
    if (N >= 1024) i8_launch<4>(x, M, w, scale, N, K, out, ldo);
    else i8_launch<2>(x, M, w, scale, N, K, out, ldo);
}

void router_cuda(const Tensor& x, int64_t M, const Tensor& w, int64_t E, int64_t K, Tensor& logits, Tensor& counter,
                 int64_t topk, const Tensor& scale, int64_t first, int64_t count, Tensor& ids, Tensor& weights,
                 Tensor& local) {
    TORCH_CHECK(E <= 256 && M <= 16 && topk <= 8, "fused router serves <= 256 experts, <= 16 rows, top-k <= 8");
    const dim3 grid((E + 1) / 2);
#define G4_RT(MM) g4::router<MM><<<grid, 256, 0, stream()>>>(cptr<float>(x), x.stride(0), (int)M, \
        cptr<__nv_bfloat16>(w), (int)E, (int)K, ptr<float>(logits), ptr<int>(counter), (int)topk, \
        cptr<__nv_bfloat16>(scale), (int)first, (int)count, ptr<int>(ids), ptr<float>(weights), ptr<int>(local))
    if (M <= 1) G4_RT(1); else if (M <= 4) G4_RT(4); else if (M <= 8) G4_RT(8); else G4_RT(16);
#undef G4_RT
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
#define G4_AT(T) g4::attn_tail<T><<<M, g4::TT, 0, stream()>>>(cptr<float>(o), o.stride(-2), (int)world, pstride, \
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
#define G4_FT(T) g4::ffn_tail<T><<<M, g4::TT, 0, stream()>>>(cptr<float>(red), (int)world, pstride, ptr<float>(h), \
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
template <int D, int AC>
void attn_launch(const Tensor& q, int64_t R, int64_t H, const Tensor& kc, const Tensor& vc,
                 const c10::optional<Tensor>& key_pos, int64_t S, const Tensor& pos0, int64_t window, int64_t noncausal,
                 double scale, int64_t chunks, int64_t splits, Tensor& po, Tensor& pm, Tensor& pl, int64_t Rpad) {
    static bool configured = false;
    const size_t bytes = g4::AttnSmem<D, AC>::bytes;
    if (!configured) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(g4::attn_split<D, AC>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes));
        configured = true;
    }
    const int64_t cps = (chunks + splits - 1) / splits;
    const dim3 grid(splits, (R + 15) / 16);
    g4::attn_split<D, AC><<<grid, 128, bytes, stream()>>>(cptr<half>(q), (int)R, (int)H, cptr<half>(kc), cptr<half>(vc),
        key_pos.has_value() ? cptr<int>(*key_pos) : nullptr, (int)S, cptr<int>(pos0), (int)window, (int)noncausal,
        (float)scale, (int)chunks,
        (int)cps, ptr<float>(po), ptr<float>(pm), ptr<float>(pl), (int)Rpad);
}
}  // namespace

void attn_cuda(const Tensor& q, int64_t R, int64_t H, int64_t D, const Tensor& kc, const Tensor& vc,
               const c10::optional<Tensor>& key_pos, int64_t S, const Tensor& pos0, int64_t window, int64_t noncausal,
               double scale, int64_t chunks, int64_t splits, Tensor& po, Tensor& pm, Tensor& pl, int64_t Rpad,
               Tensor& out) {
    if (D == 128) (R <= 16 ? attn_launch<128, 64> : attn_launch<128, 256>)(q, R, H, kc, vc, key_pos, S, pos0, window, noncausal, scale, chunks, splits, po, pm, pl, Rpad);
    else if (D == 256) (R <= 16 ? attn_launch<256, 64> : attn_launch<256, 256>)(q, R, H, kc, vc, key_pos, S, pos0, window, noncausal, scale, chunks, splits, po, pm, pl, Rpad);
    else if (D == 512) (R <= 16 ? attn_launch<512, 64> : attn_launch<512, 256>)(q, R, H, kc, vc, key_pos, S, pos0, window, noncausal, scale, chunks, splits, po, pm, pl, Rpad);
    else TORCH_CHECK(false, "attention head width must be 128, 256 or 512");
    const size_t bytes = splits * sizeof(float);
    if (out.scalar_type() == at::kHalf)
        g4::attn_merge<half><<<R, 128, bytes, stream()>>>(cptr<float>(po), cptr<float>(pm), cptr<float>(pl),
            (int)splits, (int)Rpad, (int)D, ptr<half>(out));
    else
        g4::attn_merge<float><<<R, 128, bytes, stream()>>>(cptr<float>(po), cptr<float>(pm), cptr<float>(pl),
            (int)splits, (int)Rpad, (int)D, ptr<float>(out));
}
