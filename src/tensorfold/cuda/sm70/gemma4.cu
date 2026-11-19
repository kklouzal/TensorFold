// Gemma 4 SM70 (Volta) kernels: W4A16 NVFP4 GEMV, BF16 GEMV, routing, fused norm tails,
// q/k/v norm + RoPE + KV write, WMMA split-key flash attention. Profile gemma4-sm70-w4a16-v1.
//
// Numerical contract: original E2M1 codes and E4M3 block scales are decoded exactly (bit placement into
// FP16, see decode8); every 16-element group dot product accumulates in FP32 and is scaled once by its
// block scale, then by the projection's FP32 per-output-row multiplier (1 / weight_global_scale).
// Activations into decode GEMVs are FP32; tensor-core attention uses FP16 Q/K/V/P with FP32 accumulation.
// GEMV/norm/routing reductions have a fixed order that does not depend on the number of rows in a call.
// Calls with more than one row tile (MAXM rows) run tiles on grid.y with the same per-row arithmetic.
// Attention runs per segment (one stream's consecutive rows, its own KV slot and start position) and depends on
// the segment shape: 64-key chunks when the launch's segments have rows x local heads <= 16 (one 16-row tile),
// else 256-key chunks, and the caller's split count. Target decode/verify segments of <= 8 rows (2 local heads)
// therefore share serial-decode arithmetic; wider verify blocks and prompt forwards do not. KV slots are
// addressed through per-layer device pointer tables indexed by slot (int64 addresses).
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
// by L, so every activation load is reused across the P rows it multiplies. L depends only on the matrix shape, never
// on the number of activation rows M; each output's arithmetic depends on L alone (not on P or the row tile), so each
// row's result is row-count invariant.
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
            if (MAXM == 5 || m < M) {
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
    const int m0 = blockIdx.y * MAXM;     // row tile
    x += m0 * ldx;
    out += m0 * ldo;
    M -= m0;
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
            if (MAXM == 5 || m < M) {
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

// ---------------------------------------------------------------------------------------------- wide decode (MMA)
// Decode forwards wider than one 16-row tile use tensor cores: each FP32 activation is split into FP16 hi + lo
// (hi = rn(x), lo = rn(x - hi): about 22 significant bits), weights decode exactly to FP16 (E2M1 x E4M3), and
// both products accumulate in FP32 on the MMA (rowmul then applied in FP32). Not the GEMV's summation order.

// xh/xl[m, k] = split of x[m, k] (K % 4 == 0).
__global__ void split_half(const float* __restrict__ x, int64_t ldx, int K, half* __restrict__ xh,
                           half* __restrict__ xl) {
    const int m = blockIdx.y;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < K; k += gridDim.x * blockDim.x) {
        const float v = x[m * ldx + k];
        const half h = __float2half_rn(v);
        xh[(int64_t)m * K + k] = h;
        xl[(int64_t)m * K + k] = __float2half_rn(v - __half2float(h));
    }
}

// lists[e, i]: the i-th pair (ascending) routed to local expert e; counts[e]. One warp per expert.
__global__ void expert_lists(const int* __restrict__ local, int pairs, int cap, int* __restrict__ counts,
                             int* __restrict__ lists) {
    const int e = blockIdx.x, lane = threadIdx.x;
    int n = 0;
    for (int base = 0; base < pairs; base += 32) {
        const int p = base + lane;
        const bool hit = p < pairs && local[p] == e;
        const unsigned mask = __ballot_sync(0xffffffffu, hit);
        if (hit) lists[(int64_t)e * cap + n + __popc(mask & ((1u << lane) - 1))] = p;
        n += __popc(mask);
    }
    if (lane == 0) counts[e] = n;
}

// out[dest(i), n] = rowmul[n] * sum_k x[src(i), k] * W_g[n, k] for the rows i of group g (blockIdx.y):
// groups with lists: rows i < counts[g], src = lists[g, i] / xdiv, dest = lists[g, i]; otherwise rows i < M,
// src = dest = i. Weights of group g start at row g * wrows. GEGLU: out = gelu_tanh(gate n) * up (n + N).
// Block: 4 warps = 64 rows (one 16-row tile each) x 32 outputs over K range blockIdx.z % ksplit (kper wide).
// ksplit > 1: raw sums go to part[ks, dest, col] (col: gate n, up N + n) and mma_finish adds the splits in
// order and applies the epilogue (deterministic).
constexpr int MT_ROWS = 64, MT_N = 32, MT_K = 64, MT_LD = MT_K + 16;

template <bool GEGLU>
__global__ void __launch_bounds__(128) w4_mma(const half* __restrict__ xh, const half* __restrict__ xl, int M,
        const int* __restrict__ counts, const int* __restrict__ lists, int cap, int xdiv,
        const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales, const float* __restrict__ rowmul,
        int64_t wrows, int N, int K, int ksplit, int kper, float* __restrict__ part, int64_t pstride,
        float* __restrict__ out, int64_t ldo) {
    constexpr int WR = GEGLU ? 2 * MT_N : MT_N;       // weight rows decoded per block
    __shared__ __align__(32) half ws[WR * MT_LD];
    __shared__ __align__(32) half as[2 * MT_ROWS * MT_LD];   // hi rows, then lo rows; reused for the C tile
    __shared__ int src[MT_ROWS], dst[MT_ROWS];
    const int g = blockIdx.y, ks = blockIdx.z % ksplit, r0 = (blockIdx.z / ksplit) * MT_ROWS, n0 = blockIdx.x * MT_N;
    const int rows = lists ? counts[g] : M;
    if (r0 >= rows) return;
    const int R = min(MT_ROWS, rows - r0);
    const int t = threadIdx.x, warp = t >> 5;
    if (t < MT_ROWS) {
        int s_ = 0, d_ = 0;
        if (t < R) {
            const int i = lists ? lists[(int64_t)g * cap + r0 + t] : r0 + t;
            s_ = lists ? i / xdiv : i;
            d_ = i;
        }
        src[t] = s_;
        dst[t] = d_;
    }
    const int64_t wbase = (int64_t)g * wrows;
    const int groups = K >> 4;
    constexpr int FR = WR / 16;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[FR];
#pragma unroll
    for (int f = 0; f < FR; ++f) wmma::fill_fragment(acc[f], 0.f);
    const bool active = warp * 16 < R;
    const int kend = min(K, (ks + 1) * kper);
    for (int k0 = ks * kper; k0 < kend; k0 += MT_K) {
        const int kc = min(MT_K, kend - k0);        // multiple of 16
        __syncthreads();
        // weights: WR rows x kc -> FP16 (exact), eight codes per item, one 16-byte store
        for (int item = t; item < WR * (kc >> 3); item += 128) {
            const int wr = item / (kc >> 3), c8 = item % (kc >> 3);
            const int n = min(n0 + (GEGLU ? wr % MT_N : wr), N - 1) + (GEGLU && wr >= MT_N ? N : 0);
            const int64_t row = wbase + n;
            const int kk = k0 + c8 * 8;
            const uint32_t w = __ldg(reinterpret_cast<const uint32_t*>(codes + row * (K >> 1) + (kk >> 1)));
            const float sc = scale_e4m3(__ldg(scales + row * groups + (kk >> 4)));
            float f[8];
            decode8(w, f);
            uint32_t words[4];
#pragma unroll
            for (int q = 0; q < 4; ++q)
                words[q] = uint32_t(__half_as_ushort(__float2half_rn(f[2 * q] * sc)))
                    | (uint32_t(__half_as_ushort(__float2half_rn(f[2 * q + 1] * sc))) << 16);
            *reinterpret_cast<uint4*>(ws + wr * MT_LD + c8 * 8) = make_uint4(words[0], words[1], words[2], words[3]);
        }
        // activations: R rows x kc (hi, lo), zero rows beyond R
        for (int item = t; item < MT_ROWS * (kc >> 3); item += 128) {
            const int r = item / (kc >> 3), c8 = item % (kc >> 3);
            uint4 vh = make_uint4(0, 0, 0, 0), vl = make_uint4(0, 0, 0, 0);
            if (r < R) {
                const int64_t at = (int64_t)src[r] * K + k0 + c8 * 8;
                vh = *reinterpret_cast<const uint4*>(xh + at);
                vl = *reinterpret_cast<const uint4*>(xl + at);
            }
            *reinterpret_cast<uint4*>(as + r * MT_LD + c8 * 8) = vh;
            *reinterpret_cast<uint4*>(as + (MT_ROWS + r) * MT_LD + c8 * 8) = vl;
        }
        __syncthreads();
        if (active) {
            for (int kk = 0; kk < kc; kk += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> ah, al;
                wmma::load_matrix_sync(ah, as + warp * 16 * MT_LD + kk, MT_LD);
                wmma::load_matrix_sync(al, as + (MT_ROWS + warp * 16) * MT_LD + kk, MT_LD);
#pragma unroll
                for (int f = 0; f < FR; ++f) {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
                    wmma::load_matrix_sync(b, ws + f * 16 * MT_LD + kk, MT_LD);
                    wmma::mma_sync(acc[f], ah, b, acc[f]);
                    wmma::mma_sync(acc[f], al, b, acc[f]);
                }
            }
        }
    }
    __syncthreads();
    float* cs = reinterpret_cast<float*>(as);      // [64 rows][WR] FP32 (64 * 64 * 4 <= sizeof as)
    if (active) {
#pragma unroll
        for (int f = 0; f < FR; ++f) wmma::store_matrix_sync(cs + warp * 16 * WR + f * 16, acc[f], WR, wmma::mem_row_major);
    }
    __syncthreads();
    for (int item = t; item < R * MT_N; item += 128) {
        const int r = item / MT_N, j = item % MT_N, n = n0 + j;
        if (n >= N) continue;
        if (ksplit > 1) {
            float* p = part + ks * pstride + (int64_t)dst[r] * (GEGLU ? 2 * N : N);
            p[n] = cs[r * WR + j];
            if (GEGLU) p[N + n] = cs[r * WR + MT_N + j];
            continue;
        }
        float v = cs[r * WR + j] * __ldg(rowmul + wbase + n);
        if (GEGLU) v = gelu_tanh(v) * (cs[r * WR + MT_N + j] * __ldg(rowmul + wbase + n + N));
        out[(int64_t)dst[r] * ldo + n] = v;
    }
}

// Split-K finish: out[r, n] = epilogue(sum over ks ascending of part[ks, r, col]) for rows r (grid.y) whose
// group is local[r] (>= 0; nullptr: group 0).
template <bool GEGLU>
__global__ void mma_finish(const float* __restrict__ part, int ksplit, int64_t pstride, const int* __restrict__ local,
                           int64_t wrows, const float* __restrict__ rowmul, int N, float* __restrict__ out,
                           int64_t ldo) {
    const int r = blockIdx.y, n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const int e = local ? local[r] : 0;
    if (e < 0) return;
    const int64_t wbase = (int64_t)e * wrows, width = GEGLU ? 2 * N : N;
    float a = 0.f, b = 0.f;
    for (int ks = 0; ks < ksplit; ++ks) {
        const float* p = part + ks * pstride + (int64_t)r * width;
        a += p[n];
        if (GEGLU) b += p[N + n];
    }
    float v = a * __ldg(rowmul + wbase + n);
    if (GEGLU) v = gelu_tanh(v) * (b * __ldg(rowmul + wbase + n + N));
    out[(int64_t)r * ldo + n] = v;
}

// out[m, n] = sum over slots s ascending with local >= 0 of weight[m, s] * y[m * S + s, n].
__global__ void expert_combine(const float* __restrict__ y, int slots, const int* __restrict__ local,
                               const float* __restrict__ weight, int N, float* __restrict__ out, int64_t ldo) {
    const int m = blockIdx.y, n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float total = 0.f;
    for (int s = 0; s < slots; ++s) {
        const int pair = m * slots + s;
        if (local[pair] < 0) continue;
        total = fmaf(__ldg(weight + pair), y[(int64_t)pair * N + n], total);
    }
    out[m * ldo + n] = total;
}

// Exact FP16 weights e2m1 * e4m3 (rowmul applied by the caller's GEMM epilogue): out[N, K] half.
template <bool VECTOR_STORE>
__global__ void w4_dequant(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales, int64_t N, int K,
                           half* __restrict__ out) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;  // eight coefficients per thread
    if (i >= N * (K >> 3)) return;
    const uint32_t w = reinterpret_cast<const uint32_t*>(codes)[i];
    const float sc = scale_e4m3(scales[i >> 1]);
    float f[8];
    decode8(w, f);
    half* o = out + i * 8;
    if constexpr (VECTOR_STORE) {
        // Each instruction writes adjacent 16-byte groups across the warp.
        uint32_t words[4];
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const int q = pair * 2;
            words[pair] = uint32_t(__half_as_ushort(__float2half_rn(f[q] * sc)))
                | (uint32_t(__half_as_ushort(__float2half_rn(f[q + 1] * sc))) << 16);
        }
        *reinterpret_cast<uint4*>(o) = make_uint4(words[0], words[1], words[2], words[3]);
    } else {
        // Contiguous output views may begin at any half offset; retain that contract.
#pragma unroll
        for (int q = 0; q < 8; ++q) o[q] = __float2half_rn(f[q] * sc);  // exact: <= 7 significant bits in range
    }
}

// ---------------------------------------------------------------------------------------------- BF16 GEMV
// out[m, n] = dot(x[m], W[n]) (FP32), optional final-logit softcap.  K % 8 == 0. A warp owns R outputs and reuses
// each activation load across them (R from the shape only: row-count invariant).
template <int MAXM, int R>
__global__ void __launch_bounds__(256) bf16_rows(const float* __restrict__ x, int64_t ldx, int M,
        const __nv_bfloat16* __restrict__ w, int N, int K, float* __restrict__ out, int64_t ldo, float softcap) {
    // MAXM5 is an exact-five specialization, dispatched only for whole 5-row tiles.
    // Other templates retain their original runtime row bounds.
    const int m0 = blockIdx.y * MAXM;     // row tile
    x += m0 * ldx;
    out += m0 * ldo;
    M -= m0;
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
            if (MAXM == 5 || m < M) {
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
            if (MAXM == 5 || m < M) {
                float v = warp_sum(acc[r][m]);
                if (softcap > 0.f) v = tanhf(v / softcap) * softcap;
                if (lane == 0 && n0 + r < N) out[m * ldo + n0 + r] = v;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------- INT8 GEMV
// out[m, n] = scale[n] * dot(x[m], q[n]) with symmetric per-row INT8 q (private W8A16 proposal copies). K % 16 == 0.
template <int MAXM, int R, bool EXACT_FIVE>
__global__ void __launch_bounds__(256) i8_rows(const float* __restrict__ x, int64_t ldx, int M,
        const int8_t* __restrict__ w, const float* __restrict__ scale, int N, int K, float* __restrict__ out,
        int64_t ldo, float softcap) {
    static_assert(!EXACT_FIVE || MAXM == 5, "exact-five requires its five-row tile");
    const int m0 = blockIdx.y * MAXM;     // row tile
    x += m0 * ldx;
    out += m0 * ldo;
    M -= m0;
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
            if (EXACT_FIVE || m < M) {
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
            if (EXACT_FIVE || m < M) {
                float v = warp_sum(acc[r][m]);
                if (lane == 0 && n0 + r < N) {
                    v *= __ldg(scale + n0 + r);
                    if (softcap > 0.f) v = tanhf(v / softcap) * softcap;
                    out[m * ldo + n0 + r] = v;
                }
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
// `counter` holds one zeroed device int per row tile (grid.y); each tile's last block resets its own (graph-replay
// safe).
template <int MAXM>
__global__ void __launch_bounds__(256) router(const float* __restrict__ x, int64_t ldx, int M,
        const __nv_bfloat16* __restrict__ w, int E, int K, float* __restrict__ logits, int* __restrict__ counter,
        int topk, const __nv_bfloat16* __restrict__ scale, int first, int count, int* __restrict__ ids,
        float* __restrict__ weights, int* __restrict__ local) {
    __shared__ float part[8][MAXM];
    __shared__ bool last;
    {
        const int m0 = blockIdx.y * MAXM;     // row tile
        x += m0 * ldx;
        logits += m0 * E;
        ids += m0 * topk;
        weights += m0 * topk;
        local += m0 * topk;
        counter += blockIdx.y;
        M = min(M - m0, MAXM);
    }
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
                if (MAXM == 5 || m < M) {
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
        if ((MAXM == 5 || m < M) && ee < E) {
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
    // Route each row with one warp (rows in parallel): softmax over E <= 256, stable top-k by probability
    // (lowest expert on ties), renormalized, times per_expert_scale. Lane l holds experts l, l + 32, ...
    for (int m = warp; m < (MAXM == 5 ? 5 : M); m += 8) {
        float pr[8];
        float mx = -INFINITY;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int ee = lane + 32 * j;
            pr[j] = ee < E ? __ldcg(logits + m * E + ee) : -INFINITY;
            mx = fmaxf(mx, pr[j]);
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        float sum = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            pr[j] = lane + 32 * j < E ? expf(pr[j] - mx) : 0.f;
            sum += pr[j];
        }
        sum = warp_sum(sum);
#pragma unroll
        for (int j = 0; j < 8; ++j) pr[j] = lane + 32 * j < E ? pr[j] / sum : -1.f;
        float chosen = 0.f, selected_total = 0.f;
        int picked = -1;
        for (int k = 0; k < topk; ++k) {
            float v = pr[0];
            int idx = lane;
#pragma unroll
            for (int j = 1; j < 8; ++j)
                if (pr[j] > v) { v = pr[j]; idx = lane + 32 * j; }   // ascending index: ties keep the lower one
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, v, o);
                const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
                if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
            }
            selected_total += v;
            if (lane == k) { chosen = v; picked = idx; }
            if ((idx & 31) == lane) pr[idx >> 5] = -2.f;
        }
        if (lane < topk) {
            ids[m * topk + lane] = picked;
            weights[m * topk + lane] = chosen / selected_total * bf16f(scale[picked]);
            local[m * topk + lane] = (picked >= first && picked < first + count) ? picked - first : -1;
        }
    }
    __syncthreads();
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
// grid (M, Hq + 1), D/2 threads; thread i owns rotary pair (i, i + D/2). Row m sits at position row_pos[m] of
// KV slot row_slot[m]; the slot's caches are ktab/vtab[slot] (and its ring positions ptab[slot], nullable).
// q -> learned RMS, RoPE -> qout[m, h, :] (FP16).  k -> learned RMS, RoPE -> kc[cache row].
// v -> unweighted RMS of v (or of k when !has_v: K=V layers) -> vc[cache row]. key_pos[cache row] = position.
template <typename QT>
__global__ void qkv_prep(const float* __restrict__ x, int64_t ldx, int Hq, int D, int v_mode,
        const __nv_bfloat16* __restrict__ qw, const __nv_bfloat16* __restrict__ kw, const float* __restrict__ inv_freq,
        const int* __restrict__ row_pos, const int* __restrict__ row_slot, QT* __restrict__ qout,
        const int64_t* __restrict__ ktab, const int64_t* __restrict__ vtab, const int64_t* __restrict__ ptab,
        int ring, float eps) {
    __shared__ float scratch[16];
    const int m = blockIdx.x, h = blockIdx.y, i = threadIdx.x, half_d = D >> 1;
    const int pos = row_pos[m];
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
    const int cache = row_slot[m];
    half* __restrict__ kc = reinterpret_cast<half*>(ktab[cache]);
    half* __restrict__ vc = reinterpret_cast<half*>(vtab[cache]);
    int* __restrict__ key_pos = ptab ? reinterpret_cast<int*>(ptab[cache]) : nullptr;
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
// Segment g (one stream) owns rows [row0[g], row0[g] + rows[g]) at positions pos[g] + local row, and KV slot
// slot[g]. q: [rows, H, D] FP16; segment-local attention row r = local row * H + head. Tile t covers local
// rows tile_r0[t] .. + 16 of segment tile_seg[t] (a tile never mixes segments) and owns padded rows
// [16 t, 16 t + 16) of po/pm/pl. A slot's K/V caches (ktab/vtab[slot]) hold every key a visited chunk covers,
// all initialized (masked keys must hold finite values): linear caches a multiple of 256 rows beyond
// the segment's last position, rings S rows.
// key_pos: slot -> absolute position (or nullptr: slot index is the position). A key is visible to a query
// at position p when 0 <= kp <= p and (window == 0 or kp > p - window). Scores are scaled by `scale` (FP32)
// before masking (Gemma 4 target: 1.0; Qwen3-style DFlash companion: head_dim^-0.5).
// Block (split, tile) visits chunks split, split + splits, ... of AC keys (64 for one 16-row
// tile, else 256) with an online softmax
// and writes unnormalized partials po[split, p, :], pm/pl[split, p]; attn_merge folds splits in order.

struct Segs {
    const int* slot;
    const int* pos;
    const int* rows;
    const int* row0;
    const int* tile_seg;
    const int* tile_r0;
};

template <int D, int AC>
struct AttnSmem {
    static constexpr int QLD = D + 8, PLD = AC + 8;
    static constexpr size_t bytes = 16 * QLD * 2 + 16 * AC * 4 + 16 * PLD * 2 + 16 * D * 4;
};

// RING: the slot's caches are rings with per-row positions (ptab); else linear (slot row = position).
template <int D, int AC, bool RING>
__global__ void __launch_bounds__(128) attn_split(const half* __restrict__ q, int H, const int64_t* __restrict__ ktab,
        const int64_t* __restrict__ vtab, const int64_t* __restrict__ ptab, Segs sg, int S, int window,
        int noncausal, float scale, int chunks, float* __restrict__ po, float* __restrict__ pm, float* __restrict__ pl, int Rpad) {
    constexpr int QLD = AttnSmem<D, AC>::QLD, PLD = AttnSmem<D, AC>::PLD;
    extern __shared__ __align__(32) unsigned char smem[];
    half* qs = reinterpret_cast<half*>(smem);
    float* ss = reinterpret_cast<float*>(qs + 16 * QLD);
    half* ps = reinterpret_cast<half*>(ss + 16 * AC);
    float* os = reinterpret_cast<float*>(ps + 16 * PLD);
    __shared__ int qpos[16];
    __shared__ float row_m[16], row_l[16], row_a[16];
    const int split = blockIdx.x, tile = blockIdx.y, t = threadIdx.x, warp = t >> 5;
    const int p0 = tile * 16;                       // padded partial rows
    const int seg = sg.tile_seg[tile], r0 = sg.tile_r0[tile], R = sg.rows[seg] * H, base = sg.pos[seg];
    const int cache = sg.slot[seg];
    q += (int64_t)sg.row0[seg] * H * D;
    // The slot's cache addresses stay in (volatile) shared memory and are read where used: held in registers
    // across the loop they made ptxas spill the 256-key variants (the earlier kernel-parameter pointers lived in
    // the constant bank).
    __shared__ const half* volatile kcs;
    __shared__ const half* volatile vcs;
    __shared__ const int* volatile kps;
    if (t == 0) {
        kcs = reinterpret_cast<const half*>(ktab[cache]);
        vcs = reinterpret_cast<const half*>(vtab[cache]);
        kps = RING ? reinterpret_cast<const int*>(ptab[cache]) : nullptr;
    }
    __syncthreads();
    if (!RING && split * AC > base + (R - 1) / H) {
        // Linear cache: every key of this split is in the future of every query row of the segment.
        if (t < 16) {
            pm[(int64_t)split * Rpad + p0 + t] = -INFINITY;
            pl[(int64_t)split * Rpad + p0 + t] = 0.f;
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
    const int row = t >> 3, sub = t & 7;   // softmax: 8 threads per query row
    constexpr int COLS = D / 4, FR = COLS / 16;
    bool processed = false;
    for (int c = split; c < chunks; c += gridDim.x) {     // interleaved: live keys spread over splits
        const int k0 = c * AC;
        const int* key_pos = RING ? kps : nullptr;  // one shared read per chunk (see kcs)
        bool ok = false;
        for (int i = t; i < AC; i += 128) {
            const int key = k0 + i;
            if (key < S) {
                const int kp = RING ? key_pos[key] : key;
                ok |= kp >= 0 && kp <= qmax && (window == 0 || kp > qmin - window);
            }
        }
        if (!__syncthreads_or(ok)) continue;
        processed = true;
        const half* kc = kcs;
        const half* vc = vcs;
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
                    const int kp = RING ? key_pos[key] : key;
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
        float* out = po + ((int64_t)split * Rpad + p0) * D;
        for (int i = t; i < 16 * D; i += 128) out[i] = os[i];
    }
    if (t < 16) {
        pm[(int64_t)split * Rpad + p0 + t] = row_m[t];
        pl[(int64_t)split * Rpad + p0 + t] = row_l[t];
    }
}

// out[row, :] = sum_s e^(m_s - M) o_s / sum_s e^(m_s - M) l_s, splits with l_s == 0 skipped. One block per
// padded row r (pad rows exit); out row = (segment row0) * H + segment-local attention row.
template <typename OT>
__global__ void attn_merge(const float* __restrict__ po, const float* __restrict__ pm, const float* __restrict__ pl,
                           int splits, int Rpad, int D, int H, Segs sg, OT* __restrict__ out) {
    extern __shared__ float wts[];          // [splits] fold weights (0: an empty split)
    __shared__ float red[4];
    const int r = blockIdx.x, t = threadIdx.x;
    const int seg = sg.tile_seg[r >> 4], local = sg.tile_r0[r >> 4] + (r & 15);
    if (local >= sg.rows[seg] * H) return;
    out += ((int64_t)sg.row0[seg] * H + local) * D;
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
        out[d] = cast_out<OT>(acc * inv);
    }
}

// The owner constructs a unique pair->padded-row map with -1 for absent local experts.
// Preserve the original separate Torch multiply/add operations, including signed zeros
// and exceptional weights for absent slots; neither FMA nor skipping absent slots is valid.
__global__ void expert_prompt_combine(const float* y, const int64_t* dest, const float* weights,
                                      const int* local, const float* rowmul,
                                      int D, int slots, float* out, int64_t ldo) {
    const int row = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= D) return;
    const int64_t pair = int64_t(row) * slots;
    const int64_t first = dest[pair];
    // Scale only real gathered pairs, with a separately rounded multiplication
    // exactly as the former full padded tensor mul_. Missing pairs were +0 after
    // that operation, so they must not read or apply an expert row multiplier.
    const float first_value = first < 0 ? 0.0f
        : __fmul_rn(y[first * D + col], rowmul[int64_t(local[pair]) * D + col]);
    float acc = __fmul_rn(first_value, weights[pair]);
    for (int s = 1; s < slots; ++s) {
        const int64_t at = dest[pair + s];
        const float value = at < 0 ? 0.0f
            : __fmul_rn(y[at * D + col], rowmul[int64_t(local[pair + s]) * D + col]);
        acc = __fadd_rn(acc, __fmul_rn(value, weights[pair + s]));
    }
    out[int64_t(row) * ldo + col] = acc;
}

}  // namespace g4

// ================================================================================================ launchers
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
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
// Row tiles. A call of M rows runs ceil(M / T) tiles of T rows on grid.y (instance MAXM = T; the last tile may be
// partial, except T = 5, chosen only for whole tiles). A row's arithmetic never depends on T (MAXM only bounds the
// rows one thread walks), so the tile is a pure speed choice, bitwise row-count invariant. Tiles measured on V100 at
// the Gemma 4 26B rank shapes (single GPU, 1..40 rows, build-cache/g4p/rows kb sweep 2026-11-19): 8- and 16-row
// tiles lose to 5/6-row tiles from 6 rows on (register pressure, fewer blocks); GEGLU gate/up prefers one row per
// tile up to 16 rows (2x faster at 2..16 rows), the router two.
int w4_tile(int64_t M, bool geglu) {
    if (geglu) return M <= 16 ? 1 : 3;
    if (M <= 1) return 1;
    if (M <= 4) return 4;
    return M % 5 == 0 ? 5 : 6;
}

template <bool GEGLU, int R, int L>
void w4_rows_launch(const Tensor& x, int64_t M, const Tensor& codes, const Tensor& scales, const Tensor& rowmul,
                    int64_t N, int64_t K, Tensor& out, int64_t ldo) {
    if (M <= 0) return;                   // an empty call writes nothing (a zero-height grid is a launch error)
    const int64_t per_block = g4::WARPS * (32 / L) * R;
    const int T = w4_tile(M, GEGLU);
    const dim3 grid((N + per_block - 1) / per_block, (M + T - 1) / T);
    const int64_t ldx = x.stride(0);
#define G4_ROWS(MM) g4::w4_rows<MM, GEGLU, R, L><<<grid, 256, 0, stream()>>>(cptr<float>(x), ldx, (int)M, \
        cptr<uint8_t>(codes), cptr<uint8_t>(scales), cptr<float>(rowmul), (int)N, (int)K, ptr<float>(out), ldo)
    switch (T) {
        case 1: G4_ROWS(1); break;
        case 3: G4_ROWS(3); break;
        case 4: G4_ROWS(4); break;
        case 5: G4_ROWS(5); break;
        default: G4_ROWS(6); break;
    }
#undef G4_ROWS
}
}  // namespace

// Half-warp subgroups for <= 16 groups, else full warps; outputs per subgroup reuse each activation load (4 plain,
// 2 GEGLU pairs), dropping to 2/1 for narrow matrices. One-row calls take 2 outputs on wide matrices (more blocks:
// 1.25x faster) and narrow GEGLU 2 pairs from 32 rows (measured). Outputs per subgroup never change an output's
// arithmetic (each output's group stride and reduction depend on L only), so every choice is bitwise row-count
// invariant.
void w4_rows_cuda(const Tensor& x, int64_t M, const Tensor& codes, const Tensor& scales, const Tensor& rowmul,
                  int64_t N, int64_t K, Tensor& out, int64_t ldo, bool geglu) {
    const bool half_warp = K / 16 <= 16;
    const bool wide = N >= 1024;
    if (geglu) {
        if (half_warp) w4_rows_launch<true, 1, 16>(x, M, codes, scales, rowmul, N, K, out, ldo);
        else if (wide || M >= 32) w4_rows_launch<true, 2, 32>(x, M, codes, scales, rowmul, N, K, out, ldo);
        else w4_rows_launch<true, 1, 32>(x, M, codes, scales, rowmul, N, K, out, ldo);
    } else {
        if (half_warp) w4_rows_launch<false, 2, 16>(x, M, codes, scales, rowmul, N, K, out, ldo);
        else if (wide && M > 1) w4_rows_launch<false, 4, 32>(x, M, codes, scales, rowmul, N, K, out, ldo);
        else w4_rows_launch<false, 2, 32>(x, M, codes, scales, rowmul, N, K, out, ldo);
    }
}

void w4_expert_gateup_cuda(const Tensor& x, int64_t pairs, int64_t slots, const Tensor& local, const Tensor& codes,
                           const Tensor& scales, const Tensor& rowmul, int64_t W, int64_t K, Tensor& act) {
    // One output (gate + up row) per warp: twice the blocks of two, 1.07-1.7x faster at 1..40 rows (measured).
    constexpr int R = 1;
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
    const int64_t total = N * (K / 8);
    if (reinterpret_cast<uintptr_t>(out.data_ptr()) % alignof(uint4) == 0)
        g4::w4_dequant<true><<<(total + 255) / 256, 256, 0, stream()>>>(cptr<uint8_t>(codes), cptr<uint8_t>(scales), N,
            (int)K, ptr<half>(out));
    else
        g4::w4_dequant<false><<<(total + 255) / 256, 256, 0, stream()>>>(cptr<uint8_t>(codes), cptr<uint8_t>(scales), N,
            (int)K, ptr<half>(out));
}

void expert_prompt_combine_cuda(const Tensor& y, const Tensor& dest, const Tensor& weights,
                               const Tensor& local, const Tensor& rowmul, Tensor& out) {
    TORCH_CHECK(y.is_cuda() && dest.device() == y.device() && weights.device() == y.device()
                && local.device() == y.device() && rowmul.device() == y.device() && out.device() == y.device(),
                "expert prompt combine requires tensors on one CUDA device");
    TORCH_CHECK(y.scalar_type() == at::kFloat && weights.scalar_type() == at::kFloat
                && out.scalar_type() == at::kFloat && rowmul.scalar_type() == at::kFloat
                && dest.scalar_type() == at::kLong && local.scalar_type() == at::kInt,
                "expert prompt combine requires FP32 values, int64 destinations and int32 local experts");
    TORCH_CHECK(y.dim() == 2 && dest.dim() == 2 && weights.sizes() == dest.sizes()
                && local.sizes() == dest.sizes() && rowmul.dim() == 2 && out.dim() == 2,
                "expert prompt combine requires matrices and matching destination/weight shapes");
    TORCH_CHECK(y.is_contiguous() && dest.is_contiguous() && weights.is_contiguous()
                && local.is_contiguous() && rowmul.is_contiguous()
                && out.stride(1) == 1 && out.stride(0) >= out.size(1),
                "expert prompt combine requires contiguous inputs and disjoint contiguous output rows");
    TORCH_CHECK(y.size(0) > 0 && y.size(1) == g4::MAXD && dest.size(0) > 0 && dest.size(0) <= 1024
                && dest.size(1) == 8 && out.size(0) == dest.size(0) && out.size(1) == y.size(1),
                "expert prompt combine requires 1..1024 rows, 8 slots and hidden width 2816");
    TORCH_CHECK(rowmul.size(0) > 0 && rowmul.size(1) == y.size(1),
                "expert prompt combine requires one multiplier row per original expert");
    // Destination values are owned routing output: -1 or a valid row of y, proved when
    // building dest from the unique sorted pairs. Every valid pair's local ID selects
    // that packed row's original expert, even with unequal group widths; absent IDs are
    // never dereferenced. An all-absent map may use an unread dummy y row. No host sync is needed.
    const dim3 grid((y.size(1) + 255) / 256, dest.size(0));
    g4::expert_prompt_combine<<<grid, 256, 0, stream()>>>(cptr<float>(y), cptr<int64_t>(dest),
        cptr<float>(weights), cptr<int>(local), cptr<float>(rowmul), (int)y.size(1), (int)dest.size(1),
        ptr<float>(out), out.stride(0));
}

namespace {
// Row tiles as for w4_rows (see w4_tile). BF16 head: 4-row tiles from 6 to 16 rows, 5 on whole 5-row tiles, else
// the 16-row tile (32 rows: 16 measured fastest).
int bf16_tile(int64_t M) {
    if (M <= 1) return 1;
    if (M <= 4) return 4;
    if (M % 5 == 0) return 5;
    return M <= 16 ? 4 : 16;
}

template <int R>
void bf16_launch(const Tensor& x, int64_t M, const Tensor& w, int64_t N, int64_t K, Tensor& out, int64_t ldo,
                 double softcap) {
    if (M <= 0) return;
    const int T = bf16_tile(M);
    const dim3 grid((N + g4::WARPS * R - 1) / (g4::WARPS * R), (M + T - 1) / T);
    const int64_t ldx = x.stride(0);
#define G4_BF(MM) g4::bf16_rows<MM, R><<<grid, 256, 0, stream()>>>(cptr<float>(x), ldx, (int)M, \
        cptr<__nv_bfloat16>(w), (int)N, (int)K, ptr<float>(out), ldo, (float)softcap)
    switch (T) {
        case 1: G4_BF(1); break;
        case 4: G4_BF(4); break;
        case 5: G4_BF(5); break;
        default: G4_BF(16); break;
    }
#undef G4_BF
}

// DFlash INT8 rows (private proposal copies, at most 16 rows): the vocabulary head takes 4-row tiles beyond five
// rows; narrow (R = 2) and short-K (o_proj) matrices one row per tile (two-row tiles of three for narrow ones past
// eight rows); the MLP matrices 6-row tiles beyond five rows.
int i8_tile(int64_t M, int64_t N, int64_t K, int R) {
    if (M <= 1) return 1;
    if (N >= 16384) return M == 5 ? 5 : 4;
    if (R == 2) return M <= 8 ? 1 : 3;
    if (K <= 512) return 1;
    if (M <= 4) return 4;
    return M == 5 ? 5 : 6;
}

template <int R>
void i8_launch(const Tensor& x, int64_t M, const Tensor& w, const Tensor& scale, int64_t N, int64_t K, Tensor& out,
               int64_t ldo, double softcap) {
    if (M <= 0) return;
    const int T = i8_tile(M, N, K, R);
    const dim3 grid((N + g4::WARPS * R - 1) / (g4::WARPS * R), (M + T - 1) / T);
    const int64_t ldx = x.stride(0);
#define G4_I8(MM, EXACT) g4::i8_rows<MM, R, EXACT><<<grid, 256, 0, stream()>>>(cptr<float>(x), ldx, (int)M, cptr<int8_t>(w), \
        cptr<float>(scale), (int)N, (int)K, ptr<float>(out), ldo, (float)softcap)
    switch (T) {
        case 1: G4_I8(1, false); break;
        case 3: G4_I8(3, false); break;
        case 4: G4_I8(4, false); break;
        case 5:
            // The two measured wide projection shapes favor fixed-row execution; other
            // R4 shapes retain the measured bounded tile with runtime row guards.
            if (R == 2 || (K == 2816 && (N == 1408 || N == 1280))) G4_I8(5, true);
            else G4_I8(5, false);
            break;
        default: G4_I8(6, false); break;
    }
#undef G4_I8
}
}  // namespace

void bf16_rows_cuda(const Tensor& x, int64_t M, const Tensor& w, int64_t N, int64_t K, Tensor& out, int64_t ldo,
                    double softcap) {
    if (N >= 1024) bf16_launch<4>(x, M, w, N, K, out, ldo, softcap);
    else bf16_launch<1>(x, M, w, N, K, out, ldo, softcap);
}

void i8_rows_cuda(const Tensor& x, int64_t M, const Tensor& w, const Tensor& scale, int64_t N, int64_t K, Tensor& out,
                  int64_t ldo, double softcap) {
    if (N >= 1024) i8_launch<4>(x, M, w, scale, N, K, out, ldo, softcap);
    else i8_launch<2>(x, M, w, scale, N, K, out, ldo, softcap);
}

// Router: two-row tiles beyond one row (measured 1.0-1.6x faster than 4..16-row tiles at 3..40 rows).
void router_cuda(const Tensor& x, int64_t M, const Tensor& w, int64_t E, int64_t K, Tensor& logits, Tensor& counter,
                 int64_t topk, const Tensor& scale, int64_t first, int64_t count, Tensor& ids, Tensor& weights,
                 Tensor& local) {
    if (M <= 0) return;
    const int T = M <= 1 ? 1 : 2;
    TORCH_CHECK(E <= 256 && topk <= 8 && counter.numel() >= (M + T - 1) / T,
                "fused router serves <= 256 experts, top-k <= 8 and needs one counter per 2-row tile");
    const dim3 grid((E + 1) / 2, (M + T - 1) / T);
#define G4_RT(MM) g4::router<MM><<<grid, 256, 0, stream()>>>(cptr<float>(x), x.stride(0), (int)M, \
        cptr<__nv_bfloat16>(w), (int)E, (int)K, ptr<float>(logits), ptr<int>(counter), (int)topk, \
        cptr<__nv_bfloat16>(scale), (int)first, (int)count, ptr<int>(ids), ptr<float>(weights), ptr<int>(local))
    if (T == 1) G4_RT(1); else G4_RT(2);
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
                   const Tensor& inv_freq, const Tensor& row_pos, const Tensor& row_slot, Tensor& qout,
                   const Tensor& ktab, const Tensor& vtab, const c10::optional<Tensor>& ptab, int64_t ring, double eps) {
    const dim3 grid(M, Hq + 1);
#define G4_QP(T) g4::qkv_prep<T><<<grid, D / 2, 0, stream()>>>(cptr<float>(x), x.stride(0), (int)Hq, (int)D, \
        (int)v_mode, cptr<__nv_bfloat16>(qw), cptr<__nv_bfloat16>(kw), cptr<float>(inv_freq), cptr<int>(row_pos), \
        cptr<int>(row_slot), ptr<T>(qout), cptr<int64_t>(ktab), cptr<int64_t>(vtab), \
        ptab.has_value() ? cptr<int64_t>(*ptab) : nullptr, (int)ring, (float)eps)
    if (qout.scalar_type() == at::kHalf) G4_QP(half); else G4_QP(float);
#undef G4_QP
}

namespace {
g4::Segs segs(const Tensor& seg) {
    // seg: int32 [6, ...] rows: slot, pos, rows, row0 (per segment), tile_seg, tile_r0 (per tile); one stride.
    const int* b = cptr<int>(seg);
    const int64_t s = seg.stride(0);
    return g4::Segs{b, b + s, b + 2 * s, b + 3 * s, b + 4 * s, b + 5 * s};
}

template <int D, int AC, bool RING>
void attn_launch(const Tensor& q, int64_t H, const Tensor& ktab, const Tensor& vtab,
                 const c10::optional<Tensor>& ptab, const g4::Segs& sg, int64_t tiles, int64_t S, int64_t window,
                 int64_t noncausal, double scale, int64_t chunks, int64_t splits, Tensor& po, Tensor& pm, Tensor& pl) {
    static bool configured = false;
    const size_t bytes = g4::AttnSmem<D, AC>::bytes;
    if (!configured) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(g4::attn_split<D, AC, RING>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes));
        configured = true;
    }
    const dim3 grid(splits, tiles);
    g4::attn_split<D, AC, RING><<<grid, 128, bytes, stream()>>>(cptr<half>(q), (int)H, cptr<int64_t>(ktab),
        cptr<int64_t>(vtab), ptab.has_value() ? cptr<int64_t>(*ptab) : nullptr, sg, (int)S, (int)window,
        (int)noncausal, (float)scale, (int)chunks, ptr<float>(po), ptr<float>(pm), ptr<float>(pl), (int)(tiles * 16));
}
}  // namespace

// seg: int32 [6, >= max(segments, tiles)] (see segs). small: every segment has rows x H <= 16 (64-key chunks).
void attn_cuda(const Tensor& q, int64_t H, int64_t D, const Tensor& ktab, const Tensor& vtab,
               const c10::optional<Tensor>& ptab, const Tensor& seg, int64_t tiles, bool small, int64_t S,
               int64_t window, int64_t noncausal, double scale, int64_t chunks, int64_t splits, Tensor& po, Tensor& pm,
               Tensor& pl, Tensor& out) {
    const g4::Segs sg = segs(seg);
    const bool ring = ptab.has_value();
#define G4_ATT(DD) (small ? (ring ? attn_launch<DD, 64, true> : attn_launch<DD, 64, false>) \
                          : (ring ? attn_launch<DD, 256, true> : attn_launch<DD, 256, false>)) \
        (q, H, ktab, vtab, ptab, sg, tiles, S, window, noncausal, scale, chunks, splits, po, pm, pl)
    if (D == 128) G4_ATT(128);
    else if (D == 256) G4_ATT(256);
    else if (D == 512) G4_ATT(512);
    else TORCH_CHECK(false, "attention head width must be 128, 256 or 512");
#undef G4_ATT
    const size_t bytes = splits * sizeof(float);
    const int Rpad = (int)(tiles * 16);
    if (out.scalar_type() == at::kHalf)
        g4::attn_merge<half><<<Rpad, 128, bytes, stream()>>>(cptr<float>(po), cptr<float>(pm), cptr<float>(pl),
            (int)splits, Rpad, (int)D, (int)H, sg, ptr<half>(out));
    else
        g4::attn_merge<float><<<Rpad, 128, bytes, stream()>>>(cptr<float>(po), cptr<float>(pm), cptr<float>(pl),
            (int)splits, Rpad, (int)D, (int)H, sg, ptr<float>(out));
}

namespace {
void split(const Tensor& x, int64_t M, int64_t K, Tensor& xh, Tensor& xl) {
    TORCH_CHECK(K % 16 == 0 && xh.numel() >= M * K && xl.numel() >= M * K, "wide split needs K % 16 == 0 and room");
    g4::split_half<<<dim3((K + 255) / 256, M), 256, 0, stream()>>>(cptr<float>(x), x.stride(0), (int)K, ptr<half>(xh),
                                                                   ptr<half>(xl));
}

// K splits: about four 64-wide chunks each, at most 8.
int ksplits(int64_t K, int64_t* kper) {
    const int64_t chunks = (K + g4::MT_K - 1) / g4::MT_K;
    int64_t ks = std::max<int64_t>(1, std::min<int64_t>(8, chunks / 4));
    *kper = ((chunks + ks - 1) / ks) * g4::MT_K;
    return (int)((K + *kper - 1) / *kper);
}

// One wide projection (all groups): rows by lists (or 0..M-1), split-K partials in part (floats >= ksplit * rows
// * width) and the finish over dest rows [0, dest_rows) with groups local (or nullptr).
template <bool G>
void mma_launch(const Tensor& xh, const Tensor& xl, int64_t M, const int* counts, const int* lists, int cap, int xdiv,
                int64_t groups, int64_t rchunks, const Tensor& codes, const Tensor& scales, const Tensor& rowmul,
                int64_t wrows, int64_t N, int64_t K, Tensor& part, int64_t dest_rows, const int* local, float* out,
                int64_t ldo) {
    int64_t kper = 0;
    const int ks = ksplits(K, &kper);
    const int64_t width = G ? 2 * N : N, pstride = dest_rows * width;
    TORCH_CHECK(ks == 1 || part.numel() >= ks * pstride, "wide split-K scratch is too small");
    const dim3 grid((N + g4::MT_N - 1) / g4::MT_N, groups, rchunks * ks);
    g4::w4_mma<G><<<grid, 128, 0, stream()>>>(cptr<half>(xh), cptr<half>(xl), (int)M, counts, lists, cap, xdiv,
        cptr<uint8_t>(codes), cptr<uint8_t>(scales), cptr<float>(rowmul), wrows, (int)N, (int)K, ks, (int)kper,
        ks > 1 ? ptr<float>(part) : nullptr, pstride, out, ldo);
    if (ks > 1)
        g4::mma_finish<G><<<dim3((N + 255) / 256, dest_rows), 256, 0, stream()>>>(cptr<float>(part), ks, pstride, local,
            wrows, cptr<float>(rowmul), (int)N, out, ldo);
}
}  // namespace

// Wide decode projection: out[M, N] (ldo) = x[M, K] (FP32) x W4^T (x rowmul; GEGLU pairs gate/up rows).
// xh/xl: FP16 scratch >= M * K each; part: FP32 split-K scratch >= 8 * M * N (x2 GEGLU).
void w4_mma_cuda(const Tensor& x, int64_t M, const Tensor& codes, const Tensor& scales, const Tensor& rowmul,
                 int64_t N, int64_t K, Tensor& out, int64_t ldo, bool geglu, Tensor& xh, Tensor& xl, Tensor& part) {
    split(x, M, K, xh, xl);
    const int64_t rchunks = (M + g4::MT_ROWS - 1) / g4::MT_ROWS;
    if (geglu) mma_launch<true>(xh, xl, M, nullptr, nullptr, 0, 1, 1, rchunks, codes, scales, rowmul, 0, N, K, part, M,
                                nullptr, ptr<float>(out), ldo);
    else mma_launch<false>(xh, xl, M, nullptr, nullptr, 0, 1, 1, rchunks, codes, scales, rowmul, 0, N, K, part, M,
                           nullptr, ptr<float>(out), ldo);
}

// Wide decode routed experts: pairs grouped by local expert (lists/counts int32 scratch [E, M] / [E]); gate/up
// GEGLU into act [M * slots, W], down into y [M * slots, D], then the slot-ordered combine into out (ldo).
// xh/xl: FP16 scratch >= M * max(D, slots * W) each; part: FP32 split-K scratch.
void w4_experts_mma_cuda(const Tensor& x, int64_t M, int64_t slots, const Tensor& local, const Tensor& weight,
                         const Tensor& gu_codes, const Tensor& gu_scales, const Tensor& gu_rowmul,
                         const Tensor& dn_codes, const Tensor& dn_scales, const Tensor& dn_rowmul, int64_t E,
                         int64_t W, int64_t D, Tensor& counts, Tensor& lists, Tensor& act, Tensor& y, Tensor& out,
                         int64_t ldo, Tensor& xh, Tensor& xl, Tensor& part) {
    TORCH_CHECK(counts.numel() >= E && lists.numel() >= E * M && act.numel() >= M * slots * W
                && y.numel() >= M * slots * D, "wide expert scratch is too small");
    const int pairs = (int)(M * slots), cap = (int)M;
    const int64_t rchunks = (M + g4::MT_ROWS - 1) / g4::MT_ROWS;
    g4::expert_lists<<<E, 32, 0, stream()>>>(cptr<int>(local), pairs, cap, ptr<int>(counts), ptr<int>(lists));
    split(x, M, D, xh, xl);
    mma_launch<true>(xh, xl, M, cptr<int>(counts), cptr<int>(lists), cap, (int)slots, E, rchunks, gu_codes, gu_scales,
                     gu_rowmul, 2 * W, W, D, part, pairs, cptr<int>(local), ptr<float>(act), W);
    split(act, pairs, W, xh, xl);
    mma_launch<false>(xh, xl, M, cptr<int>(counts), cptr<int>(lists), cap, 1, E, rchunks, dn_codes, dn_scales,
                      dn_rowmul, D, D, W, part, pairs, cptr<int>(local), ptr<float>(y), D);
    const dim3 g3((D + 255) / 256, M);
    g4::expert_combine<<<g3, 256, 0, stream()>>>(cptr<float>(y), (int)slots, cptr<int>(local), cptr<float>(weight),
        (int)D, ptr<float>(out), ldo);
}
