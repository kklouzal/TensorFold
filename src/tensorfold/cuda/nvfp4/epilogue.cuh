// The output factor of an NVFP4 / FP8 matmul's fp32 sums, shared by the lane matmuls, the prompt GEMMs and their
// split-K reduces: one tensor factor ``alpha``, or per-output-channel weight scales ``cs`` ([npad] fp32) and, for
// rows quantized one scale a token, their row scales ``rs`` ([M] fp32). Per-channel products are (sum * cs) * rs,
// vLLM's / CUTLASS's scaled-mm epilogue order, in fp32; ``alpha`` then plays no part. ``rs`` without ``cs`` is never
// passed (the bindings refuse it).
#pragma once

#include <cuda_runtime.h>

namespace epilogue {

struct Scales {
    float alpha;
    const float* __restrict__ rs;
    const float* __restrict__ cs;
};

__device__ __forceinline__ float one(const Scales& e, int row, int col, float v) {
    if (e.cs == nullptr) return v * e.alpha;
    const float c = v * e.cs[col];
    return e.rs == nullptr ? c : c * e.rs[row];
}

// Columns col and col + 1 (col even; ``cs`` holds npad entries, so col + 1 is always inside it).
__device__ __forceinline__ float2 pair(const Scales& e, int row, int col, float v0, float v1) {
    return make_float2(one(e, row, col, v0), one(e, row, col + 1, v1));
}

}  // namespace epilogue
