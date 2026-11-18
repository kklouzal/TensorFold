// Tensor-core (mma.sync.m8n8k4) local phases for SM70 unique-history prompt attention.
//
// Structural port: one CTA per query row sharing every K/V tile across the
// twelve GQA heads of its KV head, online softmax over 32-key chunks and the
// Volta m8n8k4 fragment mapping follow Strata-V100 prompt_attn_v70_kernel
// (src/kernels/cuda/qsa_prompt_attn.cu, MIT license, llama.cpp-derived).
// Everything else (descriptor/selection domain, failure status, unique-page
// compaction, the Q8 exact-code arithmetic and the output formats) belongs to
// TensorFold's owner contract and mirrors paged_attention.cu's RN family
// (shared device helpers: paged_attention_device.cuh):
//
//   fused   one pass: (m, l) [rows,12,2] FP32 with m = max local score,
//           l = sum exp(s - m), s = dot(q, k) * 2^-4, empty local set =
//           (-inf, 0); plus the UNNORMALIZED local sum_i exp(s_i - m) v_i
//           [rows,12,256] FP32.
//   weight  after the statistics gather/merge, scales it by exp(m - M) / L
//           into the globally normalized local partial of the RN family.
//
// Arithmetic: Q is the gathered FP16 query (exact in the tensor core). K and V
// enter the tensor core as their exact Q8 codes plus one half, (c + 0.5),
// exact in FP16; the per-32 FP16 group scale (/128) is applied in FP32 to each
// group's partial dot (K) or folded with a power-of-two range into FP16 P (V).
// This is the unrounded Q8 value, not paged_attention.cu's FP16-rounded
// decode; both are approximations of the same dequantized operand.
// Exponentials use the RN family's RN(x*log2e) -> ex2.approx.f32 (non-FTZ).
// RotorQuant caches (kernel template KIND 1..3) enter the tensor core as their
// FP16-rounded centroids (a shared-memory table) with the codec's FP32 or FP16
// per-128 group scale applied exactly where the Q8 scale is (no /128); their
// local sums stay in the stored basis (merge_values applies the inverse basis).
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <cstdint>
#include "paged_attention_descriptor.cuh"
#include "paged_attention_device.cuh"
#include "rotorquant_q4.cuh"

using T = at::Tensor;

namespace {
constexpr int HEADS = 12, DIM = 256, CH = 32, THREADS = 128, MAX_SELECTED = 2051;
constexpr int WORDS = tf_sm70_paged::WORD_COUNT;
constexpr int QW = DIM + 16;   // halves per query row: 136 words == 8 mod 32 (conflict-free A loads)
constexpr int KW = DIM + 16;   // bytes per K row: 68 words == 4 mod 32 (conflict-free B loads)
constexpr int SW = CH + 4;     // floats per score row
constexpr unsigned FULL = 0xffffffffu;
namespace meta = tf_sm70_paged;
using namespace tf_sm70_paged_device;

// Shared owner domain plus this family's envelope: Q8 tiles move 16-byte
// payload/scale rows (misaligned storage is a descriptor fault) and the
// compacted list holds at most MAX_SELECTED keys (never truncated).
__device__ __forceinline__ bool tc_domain(const int64_t* descriptor, const int* selections, int rows, int width,
                                         int row, int page_tokens, int layer, int& end, int& count, const int*& ids) {
    if (!domain(descriptor, selections, rows, width, row, page_tokens, layer, end, count, ids)) return false;
    const uint64_t alignment = uint64_t(descriptor[meta::K_DATA] | descriptor[meta::V_DATA]
                                        | descriptor[meta::K_METADATA] | descriptor[meta::V_METADATA]) & 15u;
    if (alignment || count > MAX_SELECTED) {
        latch(descriptor, 32, layer);
        return false;
    }
    return true;
}

// ---- Volta tensor-core primitives -------------------------------------------
// m8n8k4 (PTX ISA "Matrix Fragments for mma.m8n8k4"): four independent 8x8x4
// products per warp, one per quad-pair QP = lanes {4q..4q+3, 4q+16..4q+19}.
// Thread j = (lane&3) + 4*(lane>>4) holds A row j (k0..3) and B column j
// (k0..3); C register i is row (lane&1) + 4*(lane>>4) + 2*((i>>1)&1), column
// (lane&2) + 4*(i>>2) + (i&1).
__device__ __forceinline__ void mma884(float* c, uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                 "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
#else
    asm volatile("trap;");  // Built and dispatched only for SM70.
#endif
}

// Four int8 codes -> two FP16x2 of (code + 0.5), exact: 0x6400|(c^0x80) is
// 1024 + c + 128; subtracting 1152 gives c, adding 0.5 is exact below 2^8.
__device__ __forceinline__ void codes(uint32_t raw, uint32_t& lo, uint32_t& hi) {
    const uint32_t biased = raw ^ 0x80808080u;
    uint32_t a = __byte_perm(biased, 0x64646464u, 0x5140), b = __byte_perm(biased, 0x64646464u, 0x7362);
    const __half2 offset = __floats2half2_rn(1152.f, 1152.f), half = __floats2half2_rn(.5f, .5f);
    __half2 x = __hadd2(__hsub2(*reinterpret_cast<__half2*>(&a), offset), half);
    __half2 y = __hadd2(__hsub2(*reinterpret_cast<__half2*>(&b), offset), half);
    lo = *reinterpret_cast<uint32_t*>(&x);
    hi = *reinterpret_cast<uint32_t*>(&y);
}

// KIND: 0 ordinary Q8; 1 eight-bit RotorQuant (kernel codecs 8/12: index bytes, FP32 scale per 128);
// 2 FP32-scale Q4 (codecs 1..5: nibbles, FP32 scale per 128); 3 compact Q4 (codecs 9/10: nibbles, FP16 norm word).
// Four index bytes -> two FP16x2 of their FP16 centroids.
__device__ __forceinline__ void centroids(uint32_t raw, const __half* table, uint32_t& lo, uint32_t& hi) {
    __half2 x = __halves2half2(table[raw & 0xffu], table[(raw >> 8) & 0xffu]);
    __half2 y = __halves2half2(table[(raw >> 16) & 0xffu], table[raw >> 24]);
    lo = *reinterpret_cast<uint32_t*>(&x);
    hi = *reinterpret_cast<uint32_t*>(&y);
}

template<int KIND>
__device__ __forceinline__ void tile_codes(uint32_t raw, const __half* table, uint32_t& lo, uint32_t& hi) {
    if constexpr (KIND == 0) codes(raw, lo, hi);
    else centroids(raw, table, lo, hi);
}

// Eight nibble-packed codes (even value in the low nibble) -> eight index bytes.
__device__ __forceinline__ uint2 nibbles(uint32_t packed) {
    uint32_t low = 0, high = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        low |= ((packed >> (4 * i)) & 15u) << (8 * i);
        high |= ((packed >> (16 + 4 * i)) & 15u) << (8 * i);
    }
    return make_uint2(low, high);
}

__device__ __forceinline__ uint4 widen(uint2 packed) {
    const uint2 a = nibbles(packed.x), b = nibbles(packed.y);
    return make_uint4(a.x, a.y, b.x, b.y);
}

__device__ __forceinline__ uint32_t pack(float a, float b) {
    __half2 value = __floats2half2_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&value);
}

struct __align__(16) Shared {
    __half q[16][QW];             // 12 heads + 4 zero rows
    int8_t k[CH][KW];             // row-major key codes
    uint32_t v[CH / 4][DIM];      // value codes, 4 consecutive keys per word; 8-dim halves swizzled by bit 5
    float ks[CH][8];              // K group scale / 128 (0 for masked keys)
    float vs[CH][8];              // V group scale (0 for masked keys)
    float s[16][SW];              // scores, then probabilities
    float alpha[16];
    int list[MAX_SELECTED + 1];   // compacted local physical rows, selection order
    int warp_total[THREADS / 32];
};

// RotorQuant FP16 centroids (KIND > 0): dynamic shared memory, launched only for RotorQuant caches so the
// ordinary Q8 kernel keeps its static footprint.
extern __shared__ __half prompt_centroids[];

struct Prefetch {
    uint4 k[4], v[4], scale;
};

// Q4 kinds load the 8 nibble bytes of a thread's 16 dims (widened in stage); metadata is two words per row.
template<int KIND>
__device__ __forceinline__ void fetch(Prefetch& f, const Shared& S, int first, int n, const uint8_t* kdata,
                                      const uint8_t* vdata, const void* kmeta, const void* vmeta) {
    constexpr int ROW = KIND >= 2 ? DIM / 2 : DIM;
    const int t = threadIdx.x;
    const uint4 zero = make_uint4(0, 0, 0, 0);
    auto load = [&](const uint8_t* data, int at) {
        if (at >= n) return zero;
        const uint8_t* row = data + int64_t(S.list[at]) * ROW;
        if constexpr (KIND >= 2) {
            const uint2 v = __ldg(reinterpret_cast<const uint2*>(row) + (t & 15));
            return make_uint4(v.x, v.y, 0u, 0u);
        } else return __ldg(reinterpret_cast<const uint4*>(row) + (t & 15));
    };
    #pragma unroll
    for (int s = 0; s < 4; ++s) f.k[s] = load(kdata, first + (t >> 4) + 8 * s);
    #pragma unroll
    for (int e = 0; e < 4; ++e) f.v[e] = load(vdata, first + 4 * (t >> 4) + e);
    f.scale = zero;
    if (t < 2 * CH) {
        const int at = first + (t & (CH - 1));
        if (at < n) {
            const void* meta = t < CH ? kmeta : vmeta;
            if constexpr (KIND == 0)
                f.scale = __ldg(reinterpret_cast<const uint4*>(static_cast<const __half*>(meta) + int64_t(S.list[at]) * 8));
            else {
                const uint2 v = __ldg(reinterpret_cast<const uint2*>(static_cast<const float*>(meta) + int64_t(S.list[at]) * 2));
                f.scale = make_uint4(v.x, v.y, 0u, 0u);
            }
        }
    }
}

template<int KIND>
__device__ __forceinline__ void stage(Shared& S, const Prefetch& f, int first, int n,
                                      const int64_t* descriptor, int layer) {
    const int t = threadIdx.x;
    #pragma unroll
    for (int s = 0; s < 4; ++s) {
        uint4 k = f.k[s];
        if constexpr (KIND >= 2) k = widen(make_uint2(k.x, k.y));
        *reinterpret_cast<uint4*>(&S.k[(t >> 4) + 8 * s][(t & 15) * 16]) = k;
    }
    {
        // Thread owns keys 4cq..4cq+3 x dims 16dg..16dg+15: 4x4 byte transposes.
        const int cq = t >> 4, dg = t & 15;
        uint4 wide[4] = {f.v[0], f.v[1], f.v[2], f.v[3]};
        if constexpr (KIND >= 2) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) wide[e] = widen(make_uint2(f.v[e].x, f.v[e].y));
        }
        const uint32_t* in0 = reinterpret_cast<const uint32_t*>(&wide[0]);
        const uint32_t* in1 = reinterpret_cast<const uint32_t*>(&wide[1]);
        const uint32_t* in2 = reinterpret_cast<const uint32_t*>(&wide[2]);
        const uint32_t* in3 = reinterpret_cast<const uint32_t*>(&wide[3]);
        uint32_t out[16];
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            const uint32_t a = __byte_perm(in0[w], in1[w], 0x5140), b = __byte_perm(in0[w], in1[w], 0x7362);
            const uint32_t c = __byte_perm(in2[w], in3[w], 0x5140), d = __byte_perm(in2[w], in3[w], 0x7362);
            out[4 * w + 0] = __byte_perm(a, c, 0x5410);
            out[4 * w + 1] = __byte_perm(a, c, 0x7632);
            out[4 * w + 2] = __byte_perm(b, d, 0x5410);
            out[4 * w + 3] = __byte_perm(b, d, 0x7632);
        }
        uint4* destination = reinterpret_cast<uint4*>(&S.v[cq][16 * dg]);
        const int swizzle = ((dg >> 1) & 1) * 2;  // dim bit 5 swaps the 8-dim halves
        #pragma unroll
        for (int part = 0; part < 4; ++part)
            destination[part ^ swizzle] = make_uint4(out[4 * part], out[4 * part + 1], out[4 * part + 2], out[4 * part + 3]);
    }
    if (t < 2 * CH) {
        const int cell = t & (CH - 1);
        const bool key = t < CH, live = first + cell < n;
        const __half* scale = reinterpret_cast<const __half*>(&f.scale);
        const float* word = reinterpret_cast<const float*>(&f.scale);
        #pragma unroll
        for (int g = 0; g < 8; ++g) {
            // Q8: one FP16 scale per 32 values. RotorQuant: one metadata word per 128 values (groups 4w..4w+3).
            float value;
            bool valid;
            if constexpr (KIND == 0) { value = __half2float(scale[g]); valid = isfinite(value); }
            else if constexpr (KIND == 3) {
                valid = tf_rq4::valid_metadata<9>(word[g / 4]);
                value = tf_rq4::metadata_scale<9>(word[g / 4]);
            } else { valid = tf_rq4::valid_metadata<4>(word[g / 4]); value = word[g / 4]; }
            if (live && !valid) {
                latch(descriptor, key ? 2 : 4, layer);  // Every decoded element of the group is invalid.
                value = 0.f;
            }
            if (!live) value = 0.f;
            if (key) S.ks[cell][g] = KIND == 0 ? __fmul_rn(value, .0078125f) : value;
            else S.vs[cell][g] = value;
        }
    }
}

template<int KIND>
__global__ void __launch_bounds__(THREADS, 2) fused_kernel(const __half* queries, const int* selections,
        const int64_t* table, float* statistics, float* values, int rows, int width,
        int group, int page_tokens, const int* layer_word) {
    __shared__ Shared S;
    __shared__ int end, count, enabled, invalid;
    __shared__ const int* ids;
    const int row = blockIdx.x, t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int layer = *layer_word;
    const int64_t* descriptor = table + int64_t(row) * WORDS;
    float* row_values = values + int64_t(row) * HEADS * DIM;
    if (t < HEADS * 2)
        statistics[int64_t(row) * HEADS * 2 + t] = t % 2 ? 0.f : -CUDART_INF_F;
    if (!t) {
        invalid = 0;
        enabled = tc_domain(descriptor, selections, rows, width, row, page_tokens, layer, end, count, ids);
    }
    __syncthreads();
    if (!enabled) {
        for (int i = t; i < HEADS * DIM / 4; i += THREADS)
            reinterpret_cast<float4*>(row_values)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
        return;  // CTA-uniform.
    }

    // Queries: twelve heads of this KV head from four peers, plus four zero rows.
    for (int i = t; i < 16 * DIM / 8; i += THREADS) {
        const int head = i / (DIM / 8), part = i % (DIM / 8);
        uint4 value = make_uint4(0, 0, 0, 0);
        if (head < HEADS) {
            value = __ldg(reinterpret_cast<const uint4*>(
                queries + ((int64_t(group + head / 3) * rows + row) * 3 + head % 3) * DIM) + part);
            __half* h = reinterpret_cast<__half*>(&value);
            #pragma unroll
            for (int e = 0; e < 8; ++e)
                if (!isfinite(__half2float(h[e]))) {
                    latch(descriptor, 1, layer);
                    h[e] = __float2half_rn(0.f);
                }
        }
        *reinterpret_cast<uint4*>(&S.q[head][part * 8]) = value;
    }
    if (t < 16) S.alpha[t] = 0.f;
    if constexpr (KIND > 0) {
        // FP16 centroids of the stored basis; codec is the RotorQuant kernel codec.
        for (int i = t; i < 256; i += THREADS)
            prompt_centroids[i] = __float2half_rn(KIND == 1 ? tf_rq4::centroid8(i) : i >= 16 ? 0.f
                                         : KIND == 3 ? tf_rq4::selected_centroid<9>(i) : tf_rq4::centroid(i));
    }

    // Compact local selected keys (selection order) into physical row ids.
    int n = 0;
    for (int first = 0; first < count; first += THREADS) {
        const int at = first + t;
        int64_t physical = -1;
        if (at < count) {
            physical = local_row(descriptor, ids == nullptr ? at : ids[at], end, page_tokens, layer);
            if (physical == -2) atomicExch(&invalid, 1);
        }
        const unsigned ballot = __ballot_sync(FULL, physical >= 0);
        if (!lane) S.warp_total[warp] = __popc(ballot);
        __syncthreads();
        int offset = n;
        #pragma unroll
        for (int w = 0; w < THREADS / 32; ++w) {
            if (w < warp) offset += S.warp_total[w];
            n += S.warp_total[w];
        }
        if (physical >= 0) S.list[offset + __popc(ballot & ((1u << lane) - 1u))] = int(physical);
        __syncthreads();
    }
    if (invalid) {  // Uniform: read after the final barrier above.
        for (int i = t; i < HEADS * DIM / 4; i += THREADS)
            reinterpret_cast<float4*>(row_values)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
        return;
    }

    const uint8_t* kdata = pointer<const uint8_t>(descriptor[meta::K_DATA]);
    const uint8_t* vdata = pointer<const uint8_t>(descriptor[meta::V_DATA]);
    const void* kmeta = pointer<const void>(descriptor[meta::K_METADATA]);
    const void* vmeta = pointer<const void>(descriptor[meta::V_METADATA]);

    const int qp = (lane >> 2) & 3, jj = (lane & 3) + 4 * (lane >> 4);
    const int crow = (lane & 1) + 4 * (lane >> 4), ccol = lane & 2;
    // Softmax ownership: row r = t / 8 (rows 12..15 = warp 3, padding), 4 keys per lane.
    const int srow = t >> 3, sub = t & 7;
    float m = -CUDART_INF_F, l = 0.f;
    float acc[2][2][8];
    #pragma unroll
    for (int rb = 0; rb < 2; ++rb)
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt)
            #pragma unroll
            for (int i = 0; i < 8; ++i) acc[rb][nt][i] = 0.f;

    Prefetch f;
    if (n) fetch<KIND>(f, S, 0, n, kdata, vdata, kmeta, vmeta);
    for (int first = 0; first < n; first += CH) {
        __syncthreads();  // Previous chunk's readers have retired every tile.
        stage<KIND>(S, f, first, n, descriptor, layer);
        __syncthreads();
        if (first + CH < n) fetch<KIND>(f, S, first + CH, n, kdata, vdata, kmeta, vmeta);

        // Scores: warp w = keys 8w..8w+7, QP = dims 64qp..64qp+63 (two scale groups,
        // separate chains), k order rotated by qp for conflict-free shared loads.
        {
            float chain[2][2][8];
            #pragma unroll
            for (int g = 0; g < 2; ++g)
                #pragma unroll
                for (int rb = 0; rb < 2; ++rb)
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) chain[g][rb][i] = 0.f;
            const int cell = warp * 8 + jj;
            #pragma unroll
            for (int g = 0; g < 2; ++g)
                #pragma unroll
                for (int kk = 0; kk < 8; ++kk) {
                    const int d0 = 64 * qp + 32 * g + 4 * ((kk + qp) & 7);
                    uint32_t b0, b1;
                    tile_codes<KIND>(*reinterpret_cast<const uint32_t*>(&S.k[cell][d0]), prompt_centroids, b0, b1);
                    #pragma unroll
                    for (int rb = 0; rb < 2; ++rb) {
                        const uint2 a = *reinterpret_cast<const uint2*>(&S.q[rb * 8 + jj][d0]);
                        mma884(chain[g][rb], a.x, a.y, b0, b1);
                    }
                }
            #pragma unroll
            for (int rb = 0; rb < 2; ++rb)
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int col = ccol + 4 * (i >> 2) + (i & 1), key = warp * 8 + col;
                    float part = __fadd_rn(__fmul_rn(chain[0][rb][i], S.ks[key][2 * qp]),
                                           __fmul_rn(chain[1][rb][i], S.ks[key][2 * qp + 1]));
                    part = __fadd_rn(part, __shfl_xor_sync(FULL, part, 4));
                    part = __fadd_rn(part, __shfl_xor_sync(FULL, part, 8));
                    if (rb == (qp >> 1) && (i >> 2) == (qp & 1))
                        S.s[rb * 8 + crow + 2 * ((i >> 1) & 1)][key] = __fmul_rn(part, .0625f);
                }
        }
        __syncthreads();

        // Softmax: eight lanes per head row; warp 3 holds only padding rows.
        const int live = min(CH, n - first);
        if (warp < 3) {
            float x[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int key = sub * 4 + j;
                x[j] = key < live ? S.s[srow][key] : -CUDART_INF_F;
            }
            {
                float tile = fmaxf(fmaxf(x[0], x[1]), fmaxf(x[2], x[3]));
                #pragma unroll
                for (int o = 1; o < 8; o <<= 1) tile = fmaxf(tile, __shfl_xor_sync(FULL, tile, o));
                const float next = fmaxf(m, tile);
                float e[4] = {0.f, 0.f, 0.f, 0.f}, mass = 0.f;
                if (isfinite(next)) {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        e[j] = rn_exp(__fsub_rn(x[j], next), descriptor, layer);
                        mass = __fadd_rn(mass, e[j]);
                    }
                }
                #pragma unroll
                for (int o = 1; o < 8; o <<= 1) mass = __fadd_rn(mass, __shfl_xor_sync(FULL, mass, o));
                float alpha = 1.f;
                if (isfinite(next)) {
                    alpha = isfinite(m) ? rn_exp(__fsub_rn(m, next), descriptor, layer) : 0.f;
                    l = __fadd_rn(__fmul_rn(l, alpha), mass);
                }
                m = next;
                #pragma unroll
                for (int j = 0; j < 4; ++j) S.s[srow][sub * 4 + j] = e[j];
                if (!sub) S.alpha[srow] = alpha;
            }
        } else {
            #pragma unroll
            for (int j = 0; j < 4; ++j) S.s[srow][sub * 4 + j] = 0.f;
        }

        {
            __syncthreads();
            // P.V: warp w = dims 64w.., QP = 16 dims 64w+16qp.. (one scale group g),
            // k = 32 keys in steps of 4. P carries V scale * 2^(14-e), exact range.
            const int g = 2 * warp + (qp >> 1);
            // Range from the largest scale magnitude of each of the warp's two groups.
            float r0 = fabsf(S.vs[lane][2 * warp]), r1 = fabsf(S.vs[lane][2 * warp + 1]);
            #pragma unroll
            for (int o = 16; o; o >>= 1) {
                r0 = fmaxf(r0, __shfl_xor_sync(FULL, r0, o));
                r1 = fmaxf(r1, __shfl_xor_sync(FULL, r1, o));
            }
            const float range = qp >> 1 ? r1 : r0;
            int exponent = 0;
            if (range > 0.f) frexpf(range, &exponent);  // range < 2^exponent
            const float up = range > 0.f ? ldexpf(1.f, 14 - exponent) : 0.f;
            const float down = ldexpf(1.f, exponent - 14 - (KIND == 0 ? 7 : 0));  // Q8 includes its /128
            float tmp[2][2][8];
            #pragma unroll
            for (int rb = 0; rb < 2; ++rb)
                #pragma unroll
                for (int nt = 0; nt < 2; ++nt)
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) tmp[rb][nt][i] = 0.f;
            #pragma unroll
            for (int ks = 0; ks < CH / 4; ++ks) {
                const int c0 = 4 * ks;
                float w[4];
                #pragma unroll
                for (int e = 0; e < 4; ++e) w[e] = __fmul_rn(S.vs[c0 + e][g], up);
                uint32_t a[2][2];
                #pragma unroll
                for (int rb = 0; rb < 2; ++rb) {
                    const float4 p = *reinterpret_cast<const float4*>(&S.s[rb * 8 + jj][c0]);
                    a[rb][0] = pack(__fmul_rn(p.x, w[0]), __fmul_rn(p.y, w[1]));
                    a[rb][1] = pack(__fmul_rn(p.z, w[2]), __fmul_rn(p.w, w[3]));
                }
                #pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    const int d = 64 * warp + 16 * qp + 8 * nt + jj;
                    uint32_t b0, b1;
                    tile_codes<KIND>(S.v[ks][d ^ ((qp >> 1) << 3)], prompt_centroids, b0, b1);
                    #pragma unroll
                    for (int rb = 0; rb < 2; ++rb) mma884(tmp[rb][nt], a[rb][0], a[rb][1], b0, b1);
                }
            }
            #pragma unroll
            for (int rb = 0; rb < 2; ++rb) {
                const float scale[2] = {S.alpha[rb * 8 + crow], S.alpha[rb * 8 + crow + 2]};
                #pragma unroll
                for (int nt = 0; nt < 2; ++nt)
                    #pragma unroll
                    for (int i = 0; i < 8; ++i)
                        acc[rb][nt][i] = __fmaf_rn(acc[rb][nt][i], scale[(i >> 1) & 1], __fmul_rn(tmp[rb][nt][i], down));
            }
        }
    }

    if (warp < 3 && !sub) {
        statistics[(int64_t(row) * HEADS + srow) * 2] = m;
        statistics[(int64_t(row) * HEADS + srow) * 2 + 1] = l;
    }
    {
        bool bad = false;
        #pragma unroll
        for (int rb = 0; rb < 2; ++rb)
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int head = rb * 8 + crow + 2 * h;
                if (head >= HEADS) continue;
                #pragma unroll
                for (int nt = 0; nt < 2; ++nt)
                    #pragma unroll
                    for (int hh = 0; hh < 2; ++hh) {
                        const int i = 4 * hh + 2 * h;
                        const int d = 64 * warp + 16 * qp + 8 * nt + ccol + 4 * hh;
                        const float2 value = make_float2(acc[rb][nt][i], acc[rb][nt][i + 1]);
                        bad = bad || !isfinite(value.x) || !isfinite(value.y);
                        *reinterpret_cast<float2*>(row_values + head * DIM + d) = value;
                    }
            }
        if (bad) latch(descriptor, 4, layer);
    }
}

// local_values = unnormalized * exp(m - M) / L; empty local sets publish zero.
__global__ void weight_kernel(const float* unnormalized, const float* local, const float* global,
                              const int64_t* table, float* output, const int* layer_word) {
    const int row = blockIdx.x, head = blockIdx.y, t = threadIdx.x;
    const int layer = *layer_word;
    const int64_t* descriptor = table + int64_t(row) * WORDS;
    const int64_t at = int64_t(row) * HEADS + head;
    __shared__ float factor;
    __shared__ int enabled;
    if (!t) {
        enabled = active(descriptor, layer);
        factor = 0.f;
        if (enabled) {
            const float m = local[at * 2];
            if (m != -CUDART_INF_F)
                factor = __fdiv_rn(rn_exp(__fsub_rn(m, global[at * 2]), descriptor, layer), global[at * 2 + 1]);
        }
    }
    __syncthreads();
    float4 value = make_float4(0.f, 0.f, 0.f, 0.f);
    if (enabled && factor != 0.f) {
        value = reinterpret_cast<const float4*>(unnormalized + at * DIM)[t];
        value = make_float4(__fmul_rn(value.x, factor), __fmul_rn(value.y, factor),
                            __fmul_rn(value.z, factor), __fmul_rn(value.w, factor));
        if (!isfinite(value.x) || !isfinite(value.y) || !isfinite(value.z) || !isfinite(value.w))
            latch(descriptor, 4, layer);
    }
    reinterpret_cast<float4*>(output + at * DIM)[t] = value;
}
}  // namespace

void sm70_prompt_attention_fused_cuda(const T& queries, const T& selected, const T& table, T& statistics,
                                      T& unnormalized, int group, int page_tokens, const T& layer, int codec) {
    const int rows = int(queries.size(1));
    if (!rows) return;
    auto launch = [&](auto kind) {
        fused_kernel<decltype(kind)::value><<<rows, THREADS, decltype(kind)::value ? 256 * sizeof(__half) : 0,
                                              at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __half*>(queries.data_ptr()), selected.data_ptr<int>(), table.data_ptr<int64_t>(),
            statistics.data_ptr<float>(), unnormalized.data_ptr<float>(), rows,
            int(selected.size(2)), group, page_tokens, layer.data_ptr<int>());
    };
    if (codec == 0) launch(std::integral_constant<int, 0>{});
    else if (codec == 8 || codec == 12) launch(std::integral_constant<int, 1>{});
    else if (codec == 9 || codec == 10) launch(std::integral_constant<int, 3>{});
    else launch(std::integral_constant<int, 2>{});
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sm70_prompt_attention_weight_cuda(const T& unnormalized, const T& local, const T& global, const T& table,
                                       T& output, const T& layer) {
    const int rows = int(unnormalized.size(0));
    if (!rows) return;
    weight_kernel<<<dim3(rows, HEADS), DIM / 4, 0, at::cuda::getCurrentCUDAStream()>>>(
        unnormalized.data_ptr<float>(), local.data_ptr<float>(), global.data_ptr<float>(),
        table.data_ptr<int64_t>(), output.data_ptr<float>(), layer.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
