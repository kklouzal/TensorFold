#pragma once
#include <cmath>

#ifdef __CUDACC__
#define TF_NVFP4_HD __host__ __device__ __forceinline__
#else
#define TF_NVFP4_HD inline
#endif

namespace tf_sm70_nvfp4 {
// Only admitted finite E4M3 encodings reach this decode operation. The import
// checks every actual payload value and its FP16 reconstruction range.
TF_NVFP4_HD float e4m3(unsigned b) {
    unsigned e = (b >> 3) & 15, m = b & 7;
    float v = e ? ldexpf(1.f + float(m) * .125f, int(e) - 7) : ldexpf(float(m), -9);
    return b & 128 ? -v : v;
}

TF_NVFP4_HD float value(unsigned code, unsigned block_scale, float global) {
    unsigned magnitude = code & 7;
    float v = magnitude < 2 ? float(magnitude) * .5f
              : ldexpf(1.f + float(magnitude & 1) * .5f, int(magnitude >> 1) - 1);
    if (code & 8) v = -v;
    return (v * e4m3(block_scale)) * global;
}
}

#undef TF_NVFP4_HD
