// Grouped checkpoint A4 experts. Same original packed bytes, per-projection
// globals, block quantizer and m16n8k64 MMA as checkpoint.py. Shape-only decode
// K slices accumulate separately, then add in original slice order before alpha.
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <type_traits>
#include <torch/extension.h>

#include "mma4.cuh"
#include "nvfp4q.cuh"

namespace {

// ModelOpt's stored direct input scale feeds alpha unchanged. Quantization
// separately rounds its reciprocal and every FP32 scale operation. Keep this
// independent of compiler contraction/fast-div options and the CT convention.
__device__ __forceinline__ nvfp4q::Scale direct_block_scale(float amax, float inverse) {
    const float value = __fmul_rn(inverse, __fmul_rn(amax, 0x1.555556p-3f));
    const auto sf8 = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
    const float sf = __half2float(__half(__nv_cvt_fp8_to_halfraw(sf8, __NV_E4M3)));
    return {static_cast<uint32_t>(sf8), sf != 0.f ? __fdiv_rn(inverse, sf) : 0.f};
}

template <bool GROUPED, bool DIRECT>
__global__ void quant_kernel(const __nv_bfloat16* x, int64_t ldx, int rows, int k,
                              const int* plan, int pairs, const float* input, int cells, int64_t input_stride,
                              uint8_t* codes, uint8_t* scales, bool paired, const int* items = nullptr,
                              const int* counts = nullptr, const int* members = nullptr, int slots = 0,
                              int tile = 16) {
    const int blk = blockIdx.x * blockDim.x + threadIdx.x;
    const int projection = GROUPED ? blockIdx.z / tile : blockIdx.z;
    if (blk >= k / 16) return;
    int pair = GROUPED ? blockIdx.z % tile : blockIdx.y, row, cell;
    if constexpr (GROUPED) {
        const int item = blockIdx.y, local = pair;
        if (counts[0] < 0 || counts[0] > static_cast<int>(gridDim.y)) { __trap(); return; }
        if (item >= counts[0]) return;
        cell = items[3LL * item];
        if (cell == cells) return;  // the resident shared expert has its own higher-precision operator
        const int first = items[3LL * item + 1], count = items[3LL * item + 2];
        if (first < 0 || count < 1 || count > tile || first > pairs - count) { __trap(); return; }
        if (local >= count) return;
        pair = first + local;
        const int member = members[pair];
        if (member < 0 || member >= pairs) { __trap(); return; }
        row = member / slots;
    } else {
        row = paired ? pair : plan[3LL * pair];
        cell = plan[3LL * pair + 2];
    }
    if (row < 0 || row >= rows || cell < 0 || cell >= cells) { __trap(); return; }
    const float calibration = input[static_cast<int64_t>(cell) * input_stride + projection];
    const float inverse = DIRECT ? __fdiv_rn(1.f, calibration) : calibration;
    float values[16], amax = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] = __bfloat162float(x[static_cast<int64_t>(row) * ldx + blk * 16 + i]);
        amax = fmaxf(amax, fabsf(values[i]));
    }
    const auto sc = DIRECT ? direct_block_scale(amax, inverse) : nvfp4q::block_scale(amax, inverse);
    uint32_t words[2] = {0u, 0u};
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const float value = DIRECT ? __fmul_rn(values[i], sc.mul) : values[i] * sc.mul;
        words[i / 8] |= nvfp4q::e2m1(value) << (4 * (i % 8));
    }
    const int64_t base = static_cast<int64_t>(projection) * pairs + pair;
    *reinterpret_cast<uint2*>(codes + base * (k / 2) + blk * 8) = make_uint2(words[0], words[1]);
    scales[base * (k / 16) + blk] = static_cast<uint8_t>(sc.sf8);
}

// Host-window form (leased/cached banks, expert-parallel picks): one warp owns one same-expert m16 x n8 tile. A's
// four registers are (row g, row g+8) x (inputs 8t..8t+7, 32+8t..39+8t), matching lane4's ldmatrix. Each pair has
// independent activation scale words, even when the expert's input globals differ from the other projection or the
// next expert.
template <bool DOWN, bool F32, bool DIRECT>
__global__ void __launch_bounds__(128) gemm_kernel(const uint8_t* codes, const uint8_t* scales,
    const uint8_t* blob, int64_t blob_bytes, const float* weight, const float* input,
    int64_t blob_stride, int64_t weight_stride, int64_t input_stride, int cells,
    const int* items, int item_count, const int* plan, int pairs, void* out, int64_t out_row, int64_t out_slot,
    int output_rows, int output_slots, int n, int k, int slices) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int64_t unit = static_cast<int64_t>(blockIdx.x) * 4 + (threadIdx.x >> 5);
    const int nb = n / 8;
    if (unit >= static_cast<int64_t>(item_count) * nb) return;
    const int item = unit / nb, cb = unit % nb, projection = blockIdx.y;
    const int cell = items[3LL * item], first = items[3LL * item + 1], count = items[3LL * item + 2];
    if (cell < 0 || cell >= cells || first < 0 || count < 1 || count > 16 || first > pairs - count) {
        __trap(); return;
    }
    const int64_t r0 = static_cast<int64_t>(first) + g, r1 = r0 + 8;
    const int kg_count = k / 64, per = kg_count / slices;
    const bool valid0 = g < count, valid1 = g + 8 < count;
    const int64_t scale_row = static_cast<int64_t>(first) + g + 8 * (t & 1);
    const bool scale_valid = scale_row < first + count;
    const uint8_t* w = blob + static_cast<int64_t>(cell) * blob_stride + projection * blob_bytes;
    const uint8_t* ws = w + static_cast<int64_t>(n) * k / 2;
    float total[4] = {0.f, 0.f, 0.f, 0.f};
    for (int slice = 0; slice < slices; ++slice) {
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        for (int kg = slice * per; kg < (slice + 1) * per; ++kg) {
            const int64_t a0at = (static_cast<int64_t>(projection) * pairs + r0) * (k / 2) + kg * 32 + t * 4;
            const int64_t a1at = (static_cast<int64_t>(projection) * pairs + r1) * (k / 2) + kg * 32 + t * 4;
            uint32_t a[4] = {
                valid0 ? *reinterpret_cast<const uint32_t*>(codes + a0at) : 0u,
                valid1 ? *reinterpret_cast<const uint32_t*>(codes + a1at) : 0u,
                valid0 ? *reinterpret_cast<const uint32_t*>(codes + a0at + 16) : 0u,
                valid1 ? *reinterpret_cast<const uint32_t*>(codes + a1at + 16) : 0u};
            const uint32_t sa = scale_valid ? *reinterpret_cast<const uint32_t*>(
                scales + (static_cast<int64_t>(projection) * pairs + scale_row) * (k / 16) + kg * 4) : 0u;
            const int64_t tile = static_cast<int64_t>(cb / 8) * kg_count + kg;
            const uint2 b = reinterpret_cast<const uint2*>(w)[(tile * 8 + cb % 8) * 32 + lane];
            const uint32_t sb = reinterpret_cast<const uint32_t*>(ws)[tile * 64 + (cb % 8) * 8 + g];
            mma4::mma_fp4(acc, a, b.x, b.y, sa, sb);
        }
#pragma unroll
        for (int v = 0; v < 4; ++v) total[v] = slice == 0 ? acc[v] : __fadd_rn(total[v], acc[v]);
    }
    const int64_t input_at = static_cast<int64_t>(cell) * input_stride + projection;
    const int64_t weight_at = static_cast<int64_t>(cell) * weight_stride + projection;
    const float act = DIRECT ? input[input_at] : __fdiv_rn(1.f, input[input_at]);
    const float alpha = __fmul_rn(act, weight[weight_at]);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int64_t r = static_cast<int64_t>(first) + g + h * 8;
        if (r >= first + count) continue;
        int64_t at;
        if constexpr (DOWN) {
            const int row = plan[3LL * r], slot = plan[3LL * r + 1];
            if (row < 0 || row >= output_rows || slot < 0 || slot >= output_slots || plan[3LL * r + 2] != cell) {
                __trap(); return;
            }
            at = static_cast<int64_t>(row) * out_row + static_cast<int64_t>(slot) * out_slot + cb * 8 + t * 2;
        } else {
            at = (static_cast<int64_t>(projection) * pairs + r) * n + cb * 8 + t * 2;
        }
        const float v0 = __fmul_rn(total[2 * h], alpha), v1 = __fmul_rn(total[2 * h + 1], alpha);
        if constexpr (F32) {
            reinterpret_cast<float*>(out)[at] = v0;
            reinterpret_cast<float*>(out)[at + 1] = v1;
        } else {
            reinterpret_cast<__nv_bfloat16*>(out)[at] = __float2bfloat16_rn(v0);
            reinterpret_cast<__nv_bfloat16*>(out)[at + 1] = __float2bfloat16_rn(v1);
        }
    }
}

constexpr int TILE_WARPS = 4;
constexpr int STAGES = 4;
constexpr int BTILE = 2048 + 256;   // one 64-column block's packed words and E4M3 scales for one 64-input group

__device__ __forceinline__ void stage_copy(uint32_t dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void stage_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void stage_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// Down projection of a device plan. Grid (plan items, n/64 column blocks). One item (<= 16*MT same-expert pairs) and
// one 64-column block a CTA: each 64-input group's packed weight words and E4M3 scales are staged once in shared
// memory (a STAGES-deep cp.async ring) and feed every m16 tile of the item, so an expert's weights stream once per
// item. Each output element runs one fixed chain: A/B/scale fragments as lane4's ldmatrix order, input groups
// ascending, shape-only slices added in order; a pair's bits never depend on MT, its item or the other pairs.
template <bool F32, bool DIRECT, int MT>
__global__ void __launch_bounds__(TILE_WARPS * 32) down_kernel(const uint8_t* codes, const uint8_t* scales,
    const uint8_t* blob, const float* weight, const float* input,
    int64_t blob_stride, int64_t weight_stride, int64_t input_stride, int cells,
    const int* items, int item_count, int pairs, void* out, int64_t out_row, int64_t out_slot,
    int output_rows, int output_slots, int n, int k, int slices, const int* counts, const int* members, int slots) {
    constexpr int WPM = TILE_WARPS / MT;       // warps sharing one m16 tile
    constexpr int SPW = 8 / WPM;              // n8 tiles a warp
    static_assert(TILE_WARPS % MT == 0 && 8 % WPM == 0, "staged A4 warp layout");
    __shared__ __align__(16) uint8_t stage[STAGES][BTILE];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const int item = blockIdx.x, block = blockIdx.y;
    const int active_items = counts[0];
    if (active_items < 0 || active_items > item_count) { __trap(); return; }
    if (item >= active_items) return;
    const int cell = items[3LL * item], first = items[3LL * item + 1], count = items[3LL * item + 2];
    if (cell == cells) return;                 // a resident shared expert with its own operator
    if (cell < 0 || cell >= cells || first < 0 || count < 1 || count > 16 * MT || first > pairs - count) {
        __trap(); return;
    }
    const int mt = warp / WPM, sub0 = (warp % WPM) * SPW;
    const int kg_count = k / 64, per = kg_count / slices;
    const uint8_t* w = blob + static_cast<int64_t>(cell) * blob_stride;
    const uint8_t* ws = w + static_cast<int64_t>(n) * k / 2;
    const auto load = [&](int kg, int buffer) {
        const int64_t tile = static_cast<int64_t>(block) * kg_count + kg;
        const uint8_t* words = w + tile * 2048;
        const uint8_t* bscales = ws + tile * 256;
        const uint32_t base = static_cast<uint32_t>(__cvta_generic_to_shared(stage[buffer]));
        for (int c = threadIdx.x; c < BTILE / 16; c += TILE_WARPS * 32)
            stage_copy(base + c * 16, c < 128 ? words + c * 16 : bscales + (c - 128) * 16);
    };
    const int lo = mt * 16 + g;
    const bool valid0 = lo < count, valid1 = lo + 8 < count, live = mt * 16 < count;
    const int64_t input0 = static_cast<int64_t>(first) + lo, input1 = input0 + 8;
    const int64_t scale_row = static_cast<int64_t>(first) + lo + 8 * (t & 1);
    const bool scale_valid = lo + 8 * (t & 1) < count;
    float total[SPW][4], acc[SPW][4];
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < kg_count) load(s, s);
        stage_commit();
    }
    for (int kg = 0; kg < kg_count; ++kg) {
        stage_wait<STAGES - 2>();
        __syncthreads();                       // group kg landed; every warp is done with the buffer reloaded next
        if (kg + STAGES - 1 < kg_count) load(kg + STAGES - 1, (kg + STAGES - 1) % STAGES);
        stage_commit();
        if (kg % per == 0) {
#pragma unroll
            for (int j = 0; j < SPW; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
        }
        if (live) {
            const int64_t a0at = input0 * (k / 2) + kg * 32 + t * 4, a1at = input1 * (k / 2) + kg * 32 + t * 4;
            const uint32_t a[4] = {
                valid0 ? *reinterpret_cast<const uint32_t*>(codes + a0at) : 0u,
                valid1 ? *reinterpret_cast<const uint32_t*>(codes + a1at) : 0u,
                valid0 ? *reinterpret_cast<const uint32_t*>(codes + a0at + 16) : 0u,
                valid1 ? *reinterpret_cast<const uint32_t*>(codes + a1at + 16) : 0u};
            const uint32_t sa = scale_valid ? *reinterpret_cast<const uint32_t*>(
                scales + scale_row * (k / 16) + kg * 4) : 0u;
            const uint8_t* staged = stage[kg % STAGES];
#pragma unroll
            for (int j = 0; j < SPW; ++j) {
                const int s = sub0 + j;
                const uint2 b = reinterpret_cast<const uint2*>(staged)[s * 32 + lane];
                const uint32_t sb = reinterpret_cast<const uint32_t*>(staged + 2048)[s * 8 + g];
                mma4::mma_fp4(acc[j], a, b.x, b.y, sa, sb);
            }
        }
        if ((kg + 1) % per == 0) {
#pragma unroll
            for (int j = 0; j < SPW; ++j)
#pragma unroll
                for (int v = 0; v < 4; ++v) total[j][v] = kg + 1 == per ? acc[j][v] : __fadd_rn(total[j][v], acc[j][v]);
        }
    }
    stage_wait<0>();
    if (!live) return;
    const float act = DIRECT ? input[static_cast<int64_t>(cell) * input_stride]
                             : __fdiv_rn(1.f, input[static_cast<int64_t>(cell) * input_stride]);
    const float alpha = __fmul_rn(act, weight[static_cast<int64_t>(cell) * weight_stride]);
#pragma unroll
    for (int j = 0; j < SPW; ++j) {
        const int cb = block * 8 + sub0 + j;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int local = lo + h * 8;
            if (local >= count) continue;
            const int member = members[first + local];
            const int row = member / slots, slot = member % slots;
            if (member < 0 || row >= output_rows || slot >= output_slots) { __trap(); return; }
            const int64_t at = static_cast<int64_t>(row) * out_row + static_cast<int64_t>(slot) * out_slot + cb * 8 + t * 2;
            const float v0 = __fmul_rn(total[j][2 * h], alpha), v1 = __fmul_rn(total[j][2 * h + 1], alpha);
            if constexpr (F32) {
                reinterpret_cast<float*>(out)[at] = v0;
                reinterpret_cast<float*>(out)[at + 1] = v1;
            } else {
                reinterpret_cast<__nv_bfloat16*>(out)[at] = __float2bfloat16_rn(v0);
                reinterpret_cast<__nv_bfloat16*>(out)[at + 1] = __float2bfloat16_rn(v1);
            }
        }
    }
}

// The checkpoints' expert activations on BF16 gate/up outputs, each a BF16 tensor operation in the references:
// SiLU (Qwen: x / (1 + exp(-x)), rounded, then times up, rounded) and tanh GELU (Gemma:
// 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))), rounded, then times up, rounded), in FP32 as PyTorch computes.
template <int ACT>
__device__ __forceinline__ float activate(float gate, float up) {
    float a;
    if constexpr (ACT == 0) {
        a = __fdiv_rn(gate, __fadd_rn(1.f, expf(-gate)));
    } else {
        const float beta = 0.7978845608028654f, kappa = 0.044715f;
        const float cube = __fmul_rn(__fmul_rn(gate, gate), gate);
        const float inner = __fmul_rn(beta, __fmaf_rn(kappa, cube, gate));
        a = __fmul_rn(__fmul_rn(0.5f, gate), __fadd_rn(1.f, tanhf(inner)));
    }
    const float act = __bfloat162float(__float2bfloat16_rn(a));
    return __bfloat162float(__float2bfloat16_rn(__fmul_rn(act, up)));
}

// down_kernel's chain form for the gate and up projections of one 64-column block in one CTA (both staged per group),
// then the BF16 activation/product and the down projection's FP4 block quantization in the epilogue: each warp's
// 16-column groups give one down block per row, written as the paired codes/scales the down GEMM reads.
template <bool DIRECT, bool COMMON, int MT, int ACT>
__global__ void __launch_bounds__(TILE_WARPS * 32) fused_up_kernel(const uint8_t* codes, const uint8_t* scales,
    const uint8_t* blob, int64_t blob_bytes, const float* weight, const float* input, const float* down_input,
    int64_t blob_stride, int64_t weight_stride, int64_t input_stride, int64_t down_stride, int cells,
    const int* items, int item_count, int pairs, uint8_t* down_codes, uint8_t* down_scales, int n, int k, int slices,
    const int* counts, const int* members, int slots, int code_rows) {
    constexpr int WPM = TILE_WARPS / MT;
    constexpr int SPW = 8 / WPM;
    static_assert(SPW % 2 == 0, "a warp owns whole 16-column down blocks");
    __shared__ __align__(16) uint8_t stage[STAGES][2][BTILE];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const int item = blockIdx.x, block = blockIdx.y;
    const int active_items = counts[0];
    if (active_items < 0 || active_items > item_count) { __trap(); return; }
    if (item >= active_items) return;
    const int cell = items[3LL * item], first = items[3LL * item + 1], count = items[3LL * item + 2];
    if (cell == cells) return;
    if (cell < 0 || cell >= cells || first < 0 || count < 1 || count > 16 * MT || first > pairs - count) {
        __trap(); return;
    }
    const int mt = warp / WPM, sub0 = (warp % WPM) * SPW;
    const int kg_count = k / 64, per = kg_count / slices;
    const uint8_t* w = blob + static_cast<int64_t>(cell) * blob_stride;
    const auto load = [&](int kg, int buffer) {
        const int64_t tile = static_cast<int64_t>(block) * kg_count + kg;
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const uint8_t* wp = w + p * blob_bytes;
            const uint8_t* words = wp + tile * 2048;
            const uint8_t* bscales = wp + static_cast<int64_t>(n) * k / 2 + tile * 256;
            const uint32_t base = static_cast<uint32_t>(__cvta_generic_to_shared(stage[buffer][p]));
            for (int c = threadIdx.x; c < BTILE / 16; c += TILE_WARPS * 32)
                stage_copy(base + c * 16, c < 128 ? words + c * 16 : bscales + (c - 128) * 16);
        }
    };
    const int lo = mt * 16 + g;
    const bool valid0 = lo < count, valid1 = lo + 8 < count, live = mt * 16 < count;
    int64_t input0 = static_cast<int64_t>(first) + lo, input1 = input0 + 8;
    int64_t scale_row = static_cast<int64_t>(first) + lo + 8 * (t & 1);
    const bool scale_valid = lo + 8 * (t & 1) < count;
    if constexpr (COMMON) {
        if (valid0) {
            const int member = members[input0];
            if (member < 0 || member >= pairs) { __trap(); return; }
            input0 = member / slots;
        }
        if (valid1) {
            const int member = members[input1];
            if (member < 0 || member >= pairs) { __trap(); return; }
            input1 = member / slots;
        }
        if (scale_valid) {
            const int member = members[scale_row];
            if (member < 0 || member >= pairs) { __trap(); return; }
            scale_row = member / slots;
        }
    }
    const int64_t code_count = COMMON ? code_rows : pairs;
    float total[2][SPW][4], acc[2][SPW][4];
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < kg_count) load(s, s);
        stage_commit();
    }
    for (int kg = 0; kg < kg_count; ++kg) {
        stage_wait<STAGES - 2>();
        __syncthreads();
        if (kg + STAGES - 1 < kg_count) load(kg + STAGES - 1, (kg + STAGES - 1) % STAGES);
        stage_commit();
        if (kg % per == 0) {
#pragma unroll
            for (int p = 0; p < 2; ++p)
#pragma unroll
                for (int j = 0; j < SPW; ++j) acc[p][j][0] = acc[p][j][1] = acc[p][j][2] = acc[p][j][3] = 0.f;
        }
        if (live) {
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                const int64_t code_projection = COMMON ? 0 : p;
                const int64_t a0at = (code_projection * code_count + input0) * (k / 2) + kg * 32 + t * 4;
                const int64_t a1at = (code_projection * code_count + input1) * (k / 2) + kg * 32 + t * 4;
                const uint32_t a[4] = {
                    valid0 ? *reinterpret_cast<const uint32_t*>(codes + a0at) : 0u,
                    valid1 ? *reinterpret_cast<const uint32_t*>(codes + a1at) : 0u,
                    valid0 ? *reinterpret_cast<const uint32_t*>(codes + a0at + 16) : 0u,
                    valid1 ? *reinterpret_cast<const uint32_t*>(codes + a1at + 16) : 0u};
                const uint32_t sa = scale_valid ? *reinterpret_cast<const uint32_t*>(
                    scales + (code_projection * code_count + scale_row) * (k / 16) + kg * 4) : 0u;
                const uint8_t* staged = stage[kg % STAGES][p];
#pragma unroll
                for (int j = 0; j < SPW; ++j) {
                    const int s = sub0 + j;
                    const uint2 b = reinterpret_cast<const uint2*>(staged)[s * 32 + lane];
                    const uint32_t sb = reinterpret_cast<const uint32_t*>(staged + 2048)[s * 8 + g];
                    mma4::mma_fp4(acc[p][j], a, b.x, b.y, sa, sb);
                }
            }
        }
        if ((kg + 1) % per == 0) {
#pragma unroll
            for (int p = 0; p < 2; ++p)
#pragma unroll
                for (int j = 0; j < SPW; ++j)
#pragma unroll
                    for (int v = 0; v < 4; ++v)
                        total[p][j][v] = kg + 1 == per ? acc[p][j][v] : __fadd_rn(total[p][j][v], acc[p][j][v]);
        }
    }
    stage_wait<0>();
    if (!live) return;
    float alpha[2];
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        const float in = input[static_cast<int64_t>(cell) * input_stride + p];
        alpha[p] = __fmul_rn(DIRECT ? in : __fdiv_rn(1.f, in), weight[static_cast<int64_t>(cell) * weight_stride + p]);
    }
    const float calibration = down_input[static_cast<int64_t>(cell) * down_stride];
    const float inverse = DIRECT ? __fdiv_rn(1.f, calibration) : calibration;
#pragma unroll
    for (int j = 0; j < SPW; j += 2) {
        const int column = (block * 8 + sub0 + j) * 8;      // first of this warp's 16 columns: one down block
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            // Lane t holds columns {2t, 2t+1} and {8+2t, 9+2t} of row lo + 8h: four of the block's sixteen.
            float values[4];
#pragma unroll
            for (int q = 0; q < 2; ++q)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const float gate = __bfloat162float(__float2bfloat16_rn(__fmul_rn(total[0][j + q][2 * h + e], alpha[0])));
                    const float up = __bfloat162float(__float2bfloat16_rn(__fmul_rn(total[1][j + q][2 * h + e], alpha[1])));
                    values[2 * q + e] = activate<ACT>(gate, up);
                }
            float amax = fmaxf(fmaxf(fabsf(values[0]), fabsf(values[1])), fmaxf(fabsf(values[2]), fabsf(values[3])));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
            const auto sc = DIRECT ? direct_block_scale(amax, inverse) : nvfp4q::block_scale(amax, inverse);
            uint32_t nib[4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
                nib[i] = nvfp4q::e2m1(DIRECT ? __fmul_rn(values[i], sc.mul) : values[i] * sc.mul);
            const int local = lo + 8 * h;
            if (local >= count) continue;
            const int64_t pair = static_cast<int64_t>(first) + local;
            uint8_t* row_codes = down_codes + pair * (n / 2) + column / 2;
            row_codes[t] = static_cast<uint8_t>(nib[0] | (nib[1] << 4));
            row_codes[4 + t] = static_cast<uint8_t>(nib[2] | (nib[3] << 4));
            if (t == 0) down_scales[pair * (n / 16) + column / 16] = static_cast<uint8_t>(sc.sf8);
        }
    }
}

}  // namespace

void expert_a4_quant_cuda(const at::Tensor& x, const at::Tensor& plan, const at::Tensor& input,
                           at::Tensor& codes, at::Tensor& scales, bool paired, bool direct) {
    const auto launch = [&](auto mode) {
    quant_kernel<false, decltype(mode)::value><<<dim3((x.size(1) / 16 + 127) / 128, plan.size(0), input.size(1)), 128, 0,
                    at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        x.stride(0), x.size(0), x.size(1), plan.data_ptr<int>(), plan.size(0), input.data_ptr<float>(), input.size(0),
        input.stride(0), codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(), paired);
    };
    if (direct) launch(std::true_type{}); else launch(std::false_type{});
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool DOWN, bool F32, bool DIRECT>
void launch_gemm(const at::Tensor& codes, const at::Tensor& scales, const at::Tensor& blob,
    const at::Tensor& weight, const at::Tensor& input, const at::Tensor& items, const at::Tensor& plan,
    at::Tensor& out, int64_t n, int64_t k, int64_t sk) {
    const dim3 grid((items.size(0) * (n / 8) + 3) / 4, blob.size(1));
    gemm_kernel<DOWN, F32, DIRECT><<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(), blob.data_ptr<uint8_t>(), blob.size(2),
        weight.data_ptr<float>(), input.data_ptr<float>(), blob.stride(0), weight.stride(0), input.stride(0),
        blob.size(0), items.data_ptr<int>(),
        items.size(0), plan.data_ptr<int>(), plan.size(0), out.data_ptr(), out.stride(0), out.stride(1),
        out.size(0), out.size(1), n, k, sk);
}

void expert_a4_gemm_cuda(const at::Tensor& codes, const at::Tensor& scales, const at::Tensor& blob,
    const at::Tensor& weight, const at::Tensor& input, const at::Tensor& items, const at::Tensor& plan,
    at::Tensor& out, int64_t n, int64_t k, int64_t sk, bool down, bool direct) {
    const auto launch = [&](auto mode) {
    constexpr bool DIRECT = decltype(mode)::value;
    if (!down) launch_gemm<false, false, DIRECT>(codes, scales, blob, weight, input, items, plan, out, n, k, sk);
    else if (out.scalar_type() == at::kFloat)
        launch_gemm<true, true, DIRECT>(codes, scales, blob, weight, input, items, plan, out, n, k, sk);
    else launch_gemm<true, false, DIRECT>(codes, scales, blob, weight, input, items, plan, out, n, k, sk);
    };
    if (direct) launch(std::true_type{}); else launch(std::false_type{});
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Device plans: members hold the original flat (row, slot) IDs and items/counts remain live GPU routing data
// during graph capture/replay. ``tile`` is the plan's pairs an item (16 or 64).
void expert_a4_quant_plan_cuda(const at::Tensor& x, const at::Tensor& items, const at::Tensor& counts,
    const at::Tensor& members, const at::Tensor& input, at::Tensor& codes, at::Tensor& scales,
    int64_t slots, bool direct, int64_t tile) {
    const auto launch = [&](auto mode) {
    quant_kernel<true, decltype(mode)::value><<<dim3((x.size(1) / 16 + 127) / 128, items.size(0),
        input.size(1) * tile), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        x.stride(0), x.size(0), x.size(1), nullptr, codes.size(1), input.data_ptr<float>(), input.size(0),
        input.stride(0), codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(), false,
        items.data_ptr<int>(), counts.data_ptr<int>(), members.data_ptr<int>(), slots, static_cast<int>(tile));
    };
    if (direct) launch(std::true_type{}); else launch(std::false_type{});
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Gate/up GEMMs, activation and down quantization in one launch (fused_up_kernel); ``act`` 0 SiLU, 1 tanh GELU.
void expert_a4_fused_up_cuda(const at::Tensor& codes, const at::Tensor& scales, const at::Tensor& blob,
    const at::Tensor& weight, const at::Tensor& input, const at::Tensor& down_input, const at::Tensor& items,
    const at::Tensor& counts, const at::Tensor& members, at::Tensor& down_codes, at::Tensor& down_scales, int64_t n,
    int64_t k, int64_t sk, int64_t slots, bool common, bool direct, int64_t tile, int64_t act) {
    const auto launch = [&](auto mode, auto shared, auto mt, auto activation) {
        fused_up_kernel<decltype(mode)::value, decltype(shared)::value, decltype(mt)::value,
                        decltype(activation)::value><<<dim3(items.size(0), n / 64), TILE_WARPS * 32, 0,
            at::cuda::getCurrentCUDAStream()>>>(codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),
            blob.data_ptr<uint8_t>(), blob.size(2), weight.data_ptr<float>(), input.data_ptr<float>(),
            down_input.data_ptr<float>(), blob.stride(0), weight.stride(0), input.stride(0), down_input.stride(0),
            blob.size(0), items.data_ptr<int>(), items.size(0), members.size(0), down_codes.data_ptr<uint8_t>(),
            down_scales.data_ptr<uint8_t>(), n, k, sk, counts.data_ptr<int>(), members.data_ptr<int>(), slots,
            codes.size(1));
    };
    using T = std::true_type;
    using F = std::false_type;
    const auto by_act = [&](auto mode, auto shared, auto mt) {
        if (act == 0) launch(mode, shared, mt, std::integral_constant<int, 0>{});
        else launch(mode, shared, mt, std::integral_constant<int, 1>{});
    };
    const auto by_tile = [&](auto mode, auto shared) {
        if (tile == 16) by_act(mode, shared, std::integral_constant<int, 1>{});
        else by_act(mode, shared, std::integral_constant<int, 4>{});
    };
    const auto by_common = [&](auto mode) { if (common) by_tile(mode, T{}); else by_tile(mode, F{}); };
    if (direct) by_common(T{}); else by_common(F{});
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Down GEMM of a device plan into the original (row, slot) outputs: FP32 decode or BF16 prompt slots.
void expert_a4_down_plan_cuda(const at::Tensor& codes, const at::Tensor& scales, const at::Tensor& blob,
    const at::Tensor& weight, const at::Tensor& input, const at::Tensor& items, const at::Tensor& counts,
    const at::Tensor& members, at::Tensor& out, int64_t n, int64_t k, int64_t sk, int64_t slots, bool direct,
    int64_t tile) {
    const auto launch = [&](auto f32, auto mode, auto mt) {
        down_kernel<decltype(f32)::value, decltype(mode)::value, decltype(mt)::value><<<dim3(items.size(0), n / 64),
            TILE_WARPS * 32, 0, at::cuda::getCurrentCUDAStream()>>>(codes.data_ptr<uint8_t>(),
            scales.data_ptr<uint8_t>(), blob.data_ptr<uint8_t>(), weight.data_ptr<float>(), input.data_ptr<float>(),
            blob.stride(0), weight.stride(0), input.stride(0), blob.size(0), items.data_ptr<int>(), items.size(0),
            members.size(0), out.data_ptr(), out.stride(0), out.stride(1), out.size(0), out.size(1), n, k, sk,
            counts.data_ptr<int>(), members.data_ptr<int>(), slots);
    };
    using T = std::true_type;
    using F = std::false_type;
    const auto by_tile = [&](auto f32, auto mode) {
        if (tile == 16) launch(f32, mode, std::integral_constant<int, 1>{});
        else launch(f32, mode, std::integral_constant<int, 4>{});
    };
    const auto by_mode = [&](auto f32) { if (direct) by_tile(f32, T{}); else by_tile(f32, F{}); };
    if (out.scalar_type() == at::kFloat) by_mode(T{}); else by_mode(F{});
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
