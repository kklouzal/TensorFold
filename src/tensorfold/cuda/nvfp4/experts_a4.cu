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

template <bool GROUPED, bool DIRECT>
__global__ void quant_kernel(const __nv_bfloat16* x, int64_t ldx, int rows, int k,
                              const int* plan, int pairs, const float* input, int cells, int64_t input_stride,
                              uint8_t* codes, uint8_t* scales, bool paired, const int* items = nullptr,
                              const int* counts = nullptr, const int* members = nullptr, int slots = 0) {
    const int blk = blockIdx.x * blockDim.x + threadIdx.x;
    const int projection = GROUPED ? blockIdx.z / 16 : blockIdx.z;
    if (blk >= k / 16) return;
    int pair = GROUPED ? blockIdx.y * 16 + blockIdx.z % 16 : blockIdx.y, row, cell;
    if constexpr (GROUPED) {
        const int item = pair / 16, local = pair % 16;
        if (counts[0] < 0 || counts[0] > static_cast<int>(gridDim.y)) { __trap(); return; }
        if (item >= counts[0]) return;
        cell = items[3 * item];
        if (cell == cells) return;  // the resident shared expert has its own higher-precision operator
        const int first = items[3 * item + 1], count = items[3 * item + 2];
        if (first < 0 || count < 1 || count > 16 || first > pairs - count) { __trap(); return; }
        if (local >= count) return;
        pair = first + local;
        const int member = members[pair];
        if (member < 0 || member >= pairs) { __trap(); return; }
        row = paired ? pair : member / slots;
    } else {
        row = paired ? pair : plan[3 * pair];
        cell = plan[3 * pair + 2];
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
    const auto sc = nvfp4q::block_scale(amax, inverse);
    uint32_t words[2] = {0u, 0u};
#pragma unroll
    for (int i = 0; i < 16; ++i) words[i / 8] |= nvfp4q::e2m1(values[i] * sc.mul) << (4 * (i % 8));
    const int64_t base = static_cast<int64_t>(projection) * pairs + pair;
    *reinterpret_cast<uint2*>(codes + base * (k / 2) + blk * 8) = make_uint2(words[0], words[1]);
    scales[base * (k / 16) + blk] = static_cast<uint8_t>(sc.sf8);
}

// One warp owns one same-expert m16 x n8 tile. A's four registers are
// (row g, row g+8) x (inputs8t..8t+7,32+8t..39+8t), matching lane4's ldmatrix.
// Each pair has independent activation scale words, even when the expert's
// input globals differ from the other projection or the next expert.
template <bool DOWN, bool F32, bool GROUPED = false, bool DIRECT = false>
__global__ void __launch_bounds__(128) gemm_kernel(const uint8_t* codes, const uint8_t* scales,
    const uint8_t* blob, int64_t blob_bytes, const float* weight, const float* input,
    int64_t blob_stride, int64_t weight_stride, int64_t input_stride, int cells,
    const int* items, int item_count, const int* plan, int pairs, void* out, int64_t out_row, int64_t out_slot,
    int output_rows, int output_slots, int n, int k, int slices,
    const int* counts = nullptr, const int* members = nullptr, int slots = 0) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int64_t unit = static_cast<int64_t>(blockIdx.x) * 4 + (threadIdx.x >> 5);
    const int nb = n / 8;
    const int active_items = GROUPED ? counts[0] : item_count;
    if (active_items < 0 || active_items > item_count) { __trap(); return; }
    if (unit >= static_cast<int64_t>(active_items) * nb) return;
    const int item = unit / nb, cb = unit % nb, projection = blockIdx.y;
    const int cell = items[3 * item], first = items[3 * item + 1], count = items[3 * item + 2];
    if constexpr (GROUPED) { if (cell == cells) return; }
    if (cell < 0 || cell >= cells || first < 0 || count < 1 || count > 16 || first > pairs - count) {
        __trap(); return;
    }
    const int r0 = first + g, r1 = r0 + 8, kg_count = k / 64, per = kg_count / slices;
    const bool valid0 = g < count, valid1 = g + 8 < count;
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
            const int sr = first + g + 8 * (t & 1);
            const uint32_t sa = sr < first + count ? *reinterpret_cast<const uint32_t*>(
                scales + (static_cast<int64_t>(projection) * pairs + sr) * (k / 16) + kg * 4) : 0u;
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
        const int r = first + g + h * 8;
        if (r >= first + count) continue;
        int64_t at;
        if constexpr (DOWN) {
            const int member = GROUPED ? members[r] : 0;
            const int row = GROUPED ? member / slots : plan[3 * r];
            const int slot = GROUPED ? member % slots : plan[3 * r + 1];
            if (row < 0 || row >= output_rows || slot < 0 || slot >= output_slots ||
                    (!GROUPED && plan[3 * r + 2] != cell)) {
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

template <bool DOWN, bool F32, bool DIRECT = false>
void launch_gemm(const at::Tensor& codes, const at::Tensor& scales, const at::Tensor& blob,
    const at::Tensor& weight, const at::Tensor& input, const at::Tensor& items, const at::Tensor& plan,
    at::Tensor& out, int64_t n, int64_t k, int64_t sk) {
    const dim3 grid((items.size(0) * (n / 8) + 3) / 4, blob.size(1));
    gemm_kernel<DOWN, F32, false, DIRECT><<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
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

// Device-only Flash plan: members contain original flat (row,slot) IDs and
// items/counts remain live GPU routing data during graph capture/replay.
void expert_a4_quant_plan_cuda(const at::Tensor& x, const at::Tensor& items, const at::Tensor& counts,
    const at::Tensor& members, const at::Tensor& input, at::Tensor& codes, at::Tensor& scales,
    int64_t slots, bool paired) {
    quant_kernel<true, true><<<dim3((x.size(1) / 16 + 127) / 128, items.size(0), input.size(1) * 16),
        128, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        x.stride(0), x.size(0), x.size(1), nullptr, codes.size(1), input.data_ptr<float>(), input.size(0),
        input.stride(0), codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(), paired,
        items.data_ptr<int>(), counts.data_ptr<int>(), members.data_ptr<int>(), slots);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool DOWN, bool F32>
void launch_plan(const at::Tensor& codes, const at::Tensor& scales, const at::Tensor& blob,
    const at::Tensor& weight, const at::Tensor& input, const at::Tensor& items, const at::Tensor& counts,
    const at::Tensor& members, at::Tensor& out, int64_t n, int64_t k, int64_t sk, int64_t slots) {
    gemm_kernel<DOWN, F32, true, true><<<dim3((items.size(0) * (n / 8) + 3) / 4, blob.size(1)),
        128, 0, at::cuda::getCurrentCUDAStream()>>>(codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),
        blob.data_ptr<uint8_t>(), blob.size(2), weight.data_ptr<float>(), input.data_ptr<float>(),
        blob.stride(0), weight.stride(0), input.stride(0), blob.size(0), items.data_ptr<int>(), items.size(0),
        nullptr, codes.size(1), out.data_ptr(), out.stride(0), out.stride(1), out.size(0), out.size(1), n, k, sk,
        counts.data_ptr<int>(), members.data_ptr<int>(), slots);
}

void expert_a4_gemm_plan_cuda(const at::Tensor& codes, const at::Tensor& scales, const at::Tensor& blob,
    const at::Tensor& weight, const at::Tensor& input, const at::Tensor& items, const at::Tensor& counts,
    const at::Tensor& members, at::Tensor& out, int64_t n, int64_t k, int64_t sk, int64_t slots, bool down) {
    if (!down) launch_plan<false, false>(codes, scales, blob, weight, input, items, counts, members, out, n, k, sk, slots);
    else if (out.scalar_type() == at::kFloat)
        launch_plan<true, true>(codes, scales, blob, weight, input, items, counts, members, out, n, k, sk, slots);
    else launch_plan<true, false>(codes, scales, blob, weight, input, items, counts, members, out, n, k, sk, slots);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
