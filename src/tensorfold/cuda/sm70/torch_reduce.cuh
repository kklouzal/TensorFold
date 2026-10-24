// Reproduce the pinned ATen contiguous FP32 Reduce.cuh input-vector4 arithmetic.
// The virtual x-lanes permit the same reduction tree with a256-thread block.
#pragma once

namespace tensorfold_sm70 {
__device__ __forceinline__ int lower_power(int value) {
    return 1 << (31 - __clz(max(1, value)));
}

__device__ __forceinline__ int torch_reduce_width(int width, int outputs) {
    int dim0 = min(512, lower_power(width >= 128 ? width / 4 : width));
    int dim1 = min(512, lower_power(outputs));
    int x = min(dim0, 32);
    int y = min(dim1, 512 / x);
    return min(dim0, 512 / y);
}

template<class Reader>
__device__ __forceinline__ float torch_sum(Reader reader, int width, int outputs, float* shared) {
    const int block_width = torch_reduce_width(width, outputs);
    for (int logical = threadIdx.x; logical < block_width; logical += blockDim.x) {
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        if (width >= 128) {
            for (int vector = logical; vector < width / 4; vector += block_width) {
                a0 = a0 + reader(vector * 4);
                a1 = a1 + reader(vector * 4 + 1);
                a2 = a2 + reader(vector * 4 + 2);
                a3 = a3 + reader(vector * 4 + 3);
            }
        } else {
            if (logical < width) a0 = a0 + reader(logical);
            if (logical + block_width < width) a1 = a1 + reader(logical + block_width);
            if (logical + 2 * block_width < width) a2 = a2 + reader(logical + 2 * block_width);
            if (logical + 3 * block_width < width) a3 = a3 + reader(logical + 3 * block_width);
        }
        float value = a0 + a1;
        value = value + a2;
        shared[logical] = value + a3;
    }
    __syncthreads();
    for (int offset = block_width / 2; offset >= 32; offset >>= 1) {
        if (threadIdx.x < offset) shared[threadIdx.x] = shared[threadIdx.x] + shared[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x < 32) {
        float value = shared[threadIdx.x];
        for (int offset = 16; offset; offset >>= 1) value = value + __shfl_down_sync(0xffffffffu, value, offset);
        if (!threadIdx.x) shared[0] = value;
    }
    __syncthreads();
    return shared[0];
}

struct FloatReader {
    const float* values;
    __device__ __forceinline__ float operator()(int index) const { return values[index]; }
};

struct HalfSquareReader {
    const __half* values;
    __device__ __forceinline__ float operator()(int index) const {
        float value = __half2float(values[index]);
        return value * value;
    }
};
}  // namespace tensorfold_sm70
