// Device-side owner-descriptor domain shared by the SM70 paged attention
// families (paged_attention.cu, prompt_attention.cu). One definition keeps the
// descriptor validation, failure-status latch and RN exponential identical
// across families; each including translation unit is its own extension.
#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>
#include "paged_attention_descriptor.cuh"

namespace tf_sm70_paged_device {

// Pointer words are produced only by the canonical owner, which retains every
// backing tensor through stream/capture retirement. Device data is checked
// before dereferencing page offsets or payloads. A zero live-count pointer is
// the explicit inactive-row representation and never touches a State.
template<typename V> __device__ __forceinline__ V* pointer(int64_t word) {
    return reinterpret_cast<V*>(uintptr_t(word));
}

__device__ __forceinline__ void latch(const int64_t* descriptor, int bits, int layer) {
    int* status = pointer<int>(descriptor[tf_sm70_paged::STATUS]);
    if (status == nullptr) {
        asm volatile("trap;");  // A live descriptor lost its required failure owner.
        return;
    }
    atomicOr(status, bits);
    atomicCAS(status + 1, -2, layer);
}

__device__ __forceinline__ bool active(const int64_t* descriptor, int layer) {
    namespace meta = tf_sm70_paged;
    const int* live = pointer<const int>(descriptor[meta::LIVE_ROWS]);
    if (live == nullptr) return false;
    const int count = *live;
    const int64_t within = descriptor[meta::ROW_OFFSET];
    if (count < 0 || within < 0) {
        latch(descriptor, 32, layer);
        return false;
    }
    if (within >= count) return false;
    const int64_t* generations = pointer<const int64_t>(descriptor[meta::GENERATIONS]);
    if (generations == nullptr || descriptor[meta::EXPECTED_SLOT] < 0 || descriptor[meta::EXPECTED_OWNER] < 0
            || generations[0] != descriptor[meta::EXPECTED_SLOT] || generations[1] != descriptor[meta::EXPECTED_OWNER]) {
        latch(descriptor, 32, layer);
        return false;
    }
    return true;
}

// Active row's visible end, selected count and sparse ids (nullptr = dense 0..end-1).
__device__ __forceinline__ bool domain(const int64_t* descriptor, const int* selections,
                                      int rows, int width, int row, int page_tokens,
                                      int layer, int& end, int& count, const int*& ids) {
    namespace meta = tf_sm70_paged;
    if (!active(descriptor, layer)) return false;
    const int* position = pointer<const int>(descriptor[meta::POSITION]);
    const int64_t logical = descriptor[meta::LOGICAL_CAPACITY], physical = descriptor[meta::PHYSICAL_ROWS], pages = descriptor[meta::PAGE_COUNT];
    const int64_t owner = descriptor[meta::INDEXER_OWNER];
    if (position == nullptr || logical <= 0 || logical > INT32_MAX || physical < 0 || physical > INT32_MAX
            || pages != (logical + page_tokens - 1) / page_tokens || owner < 0 || owner >= 8
            || descriptor[meta::PAGE_OFFSETS] == 0 || (physical && (descriptor[meta::K_DATA] == 0 || descriptor[meta::V_DATA] == 0
                                                 || descriptor[meta::K_METADATA] == 0 || descriptor[meta::V_METADATA] == 0))) {
        latch(descriptor, 32, layer);
        return false;
    }
    const int64_t visible = int64_t(*position) + descriptor[meta::ROW_OFFSET] + 1;
    if (visible <= 0 || visible > logical) {
        latch(descriptor, 32, layer);
        return false;
    }
    const int* selection = selections + (owner * rows + row) * int64_t(width);
    const int sparse = selection[width - 1], selected = selection[width - 2];
    if ((sparse != 0 && sparse != 1) || selected < 0 || selected > width - 2) {
        latch(descriptor, 32, layer);
        return false;
    }
    end = int(visible);
    count = sparse ? selected : end;
    ids = sparse ? selection : nullptr;
    return true;
}

// Local physical row of a logical position: -1 = not on this rank or outside
// [0, end), -2 = descriptor fault (latched).
__device__ __forceinline__ int64_t local_row(const int64_t* descriptor, int logical,
                                           int end, int page_tokens, int layer) {
    namespace meta = tf_sm70_paged;
    if (logical < 0 || logical >= end) return -1;
    const int page = logical / page_tokens, within = logical % page_tokens;
    if (page >= descriptor[meta::PAGE_COUNT]) {
        latch(descriptor, 32, layer);
        return -2;
    }
    const int64_t offset = pointer<const int64_t>(descriptor[meta::PAGE_OFFSETS])[page];
    if (offset == -1) return -1;
    if (offset < -1 || descriptor[meta::PHYSICAL_ROWS] <= within || offset > descriptor[meta::PHYSICAL_ROWS] - 1 - within) {
        latch(descriptor, 32, layer);
        return -2;
    }
    return offset + within;
}

// RN family exponential, x <= 0: exp(x) = ex2.approx(RN(x * log2e)) (non-FTZ).
__device__ __forceinline__ float rn_exp(float x, const int64_t* descriptor, int layer) {
    if (x == -CUDART_INF_F) return 0.f;
    if (!isfinite(x) || x > 0.f) {
        latch(descriptor, 16, layer);
        return 0.f;  // Defined failed-row scratch; status forbids accepted publication.
    }
    if (x == 0.f) return 1.f;
    const float exponent = __fmul_rn(x, __int_as_float(0x3fb8aa3b));
    float result;
    asm volatile("ex2.approx.f32 %0, %1;" : "=f"(result) : "f"(exponent));
    return result;
}

}  // namespace tf_sm70_paged_device
