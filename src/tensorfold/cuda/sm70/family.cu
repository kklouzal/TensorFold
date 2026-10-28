// FP16 stored boundaries and FP32 ordered reductions; no allocations/readbacks in launches.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <optional>
#include <algorithm>
#include <type_traits>
#include <cub/block/block_radix_sort.cuh>
#include "torch_reduce.cuh"
#include "rotorquant_q4.cuh"

using T = at::Tensor;
using O = std::optional<T>;

namespace {
constexpr int THREADS = 256;
__device__ __forceinline__ float half_round(float x) { return __half2float(__float2half_rn(x)); }
__device__ __forceinline__ float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }
__device__ __forceinline__ float silu(float x) { return x / (1.f + expf(-x)); }
__device__ __forceinline__ float warp_sum(float x) {
    for (int offset = 16; offset; offset >>= 1) x += __shfl_xor_sync(0xffffffffu, x, offset);
    return x;
}
__device__ __forceinline__ float block_sum(float x, float* shared) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    x = warp_sum(x);
    if (!lane) shared[warp] = x;
    __syncthreads();
    x = threadIdx.x < 8 ? shared[lane] : 0.f;
    if (!warp) x = warp_sum(x);
    if (!threadIdx.x) shared[0] = x;
    __syncthreads();
    return shared[0];
}
__device__ __forceinline__ float read(const void* pointer, bool is_float, int64_t at) {
    return is_float ? static_cast<const float*>(pointer)[at] : __half2float(static_cast<const __half*>(pointer)[at]);
}

using tensorfold_sm70::torch_sum;
using tensorfold_sm70::HalfSquareReader;

struct PleProductReader {
    const __half* keys;
    const __half* query;
    const float* key_weight;
    const float* query_weight;
    float key_inverse, query_inverse;
    __device__ __forceinline__ float operator()(int col) const {
        float key = half_round((__half2float(keys[col]) * key_inverse) * key_weight[col]);
        float value = half_round((__half2float(query[col]) * query_inverse) * query_weight[col]);
        return half_round(key * value);
    }
};

struct PleGatedSquareReader {
    const __half* values;
    float gate;
    __device__ __forceinline__ float operator()(int col) const {
        float value = half_round(gate * __half2float(values[col]));
        return value * value;
    }
};

__global__ void writeback_kernel(const __half* h, __half* out, float* pss, int rows, int dims, int streams,
                                 int mode, const void* branch, bool bf, int ranks, const __half* inject,
                                 const void* y, bool yf, const float* weights, int slots) {
    __shared__ float shared[512];
    const int chunk = blockIdx.x % (dims / 256);
    const int stream = (blockIdx.x / (dims / 256)) % streams;
    const int row = blockIdx.x / ((dims / 256) * streams);
    const int col = chunk * 256 + threadIdx.x;
    const int64_t at = int64_t(row) * streams * dims + stream * dims + col;
    float value = __half2float(h[at]);
    if (mode) {
        float update = 0.f;
        if (mode == 1) update = read(branch, bf, int64_t(row) * dims + col);
        else if (mode == 2) {
            for (int slot = 0; slot < slots; ++slot)
                update = update + read(y, yf, (int64_t(row) * slots + slot) * dims + col) * weights[row * slots + slot];
        } else {
            update = read(branch, bf, int64_t(row) * dims + col);
            for (int rank = 1; rank < ranks; ++rank)
                update = update + read(branch, bf, (int64_t(rank) * rows + row) * dims + col);
        }
        float addition = half_round(half_round(update) * __half2float(inject[row * streams + stream]));
        value = half_round(value + addition);
    }
    out[at] = __float2half_rn(value);
    __syncthreads();
    const int64_t group_base = int64_t(row) * streams * dims + stream * dims + chunk * 256;
    float sum = torch_sum(HalfSquareReader{out + group_base}, 256, rows * streams * (dims / 256), shared);
    if (!threadIdx.x) pss[(row * (dims / 256) + chunk) * streams + stream] = sum;
}

__global__ void normed_kernel(const __half* h, const float* pss, const float* scale, __half* out, float* xs,
                              int rows, int dims, int streams, float eps) {
    const int group = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int wide = dims * streams, groups = wide / 32;
    const int row = group / groups, col = (group % groups) * 32 + lane;
    if (row >= rows) return;
    const int stream = col / dims;
    float sum = pss[row * (dims / 256) * streams + stream];
    for (int chunk = 1; chunk < dims / 256; ++chunk)
        sum += pss[(row * (dims / 256) + chunk) * streams + stream];
    float value = half_round((__half2float(h[int64_t(row) * wide + col]) * rsqrtf(sum * (1.f / float(dims)) + eps)) * scale[col]);
    out[int64_t(row) * wide + col] = __float2half_rn(value);
    float total = warp_sum(value);
    if (!lane) xs[group] = total;
}

__global__ void act_kernel(const void* input, bool in_float, __half* out, float* xs, __half* inject,
                           int rows, int low, int streams, int splits, int columns, bool reduced) {
    const int row = blockIdx.x, lane = threadIdx.x & 31;
    for (int col = threadIdx.x; col < columns; col += blockDim.x) {
        float x = read(input, in_float, int64_t(row) * columns + col);
        for (int split = 1; split < splits; ++split)
            x += read(input, in_float, (int64_t(split) * rows + row) * columns + col);
        if (reduced) x = half_round(x);
        x = half_round(x / streams);
        if (col < low) {
            float value = half_round(silu(x));
            out[int64_t(row) * low + col] = __float2half_rn(value);
            float sum = warp_sum(value);
            if (!lane) xs[int64_t(row) * (low / 32) + col / 32] = sum;
        } else if (inject) inject[row * streams + col - low] = __float2half_rn(2.f * half_round(sigmoid(x)));
    }
}

__global__ void mix_kernel(const __half* up, const __half* normed, __half* out, float* xs,
                           int rows, int dims, int streams) {
    const int group = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int row = group / (dims / 32), col = (group % (dims / 32)) * 32 + lane;
    if (row >= rows) return;
    float sum = 0.f;
    for (int stream = 0; stream < streams; ++stream) {
        int64_t at = int64_t(row) * streams * dims + stream * dims + col;
        float gate = half_round(sigmoid(__half2float(up[at])));
        sum = sum + half_round(gate * __half2float(normed[at]));
    }
    float value = half_round(sum / streams);
    out[int64_t(row) * dims + col] = __float2half_rn(value);
    sum = warp_sum(value);
    if (!lane) xs[group] = sum;
}

__global__ void rms_kernel(const __half* x, const float* weight, __half* out, float* xs,
                           int wide, int width, float eps) {
    __shared__ float shared[8];
    const int group = blockIdx.x, per_row = wide / width;
    const int offset = (group % per_row) * width;
    const int64_t base = int64_t(group / per_row) * wide + offset;
    float sum = 0.f;
    for (int col = threadIdx.x; col < width; col += blockDim.x) {
        float value = __half2float(x[base + col]); sum += value * value;
    }
    float inv = rsqrtf(block_sum(sum, shared) / width + eps);
    for (int col = threadIdx.x; col < width; col += blockDim.x) {
        float value = half_round((__half2float(x[base + col]) * inv) * weight[offset + col]);
        out[base + col] = __float2half_rn(value);
        float total = warp_sum(value);
        if (!(threadIdx.x & 31)) xs[(base + col) / 32] = total;
    }
}

__global__ void gate_kernel(const __half* value, const __half* proj, __half* out, float* xs,
                            int rows, int heads, int dims, int projection_width) {
    const int group = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int width = heads * dims, row = group / (width / 32), col = (group % (width / 32)) * 32 + lane;
    if (row >= rows) return;
    int source = (col / dims) * 2 * dims + dims + col % dims;
    float result = half_round(__half2float(value[int64_t(row) * width + col]) *
                              sigmoid(__half2float(proj[int64_t(row) * projection_width + source])));
    out[int64_t(row) * width + col] = __float2half_rn(result);
    float sum = warp_sum(result);
    if (!lane) xs[group] = sum;
}

__global__ void embed_kernel(const __half* values, __half* out, float* xs, int64_t groups, float scale) {
    const int64_t group = int64_t(blockIdx.x) * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (group >= groups) return;
    float value = half_round(__half2float(values[group * 32 + lane]) * scale);
    out[group * 32 + lane] = __float2half_rn(value);
    float sum = warp_sum(value);
    if (!lane) xs[group] = sum;
}

__global__ void ple_gate_kernel(const __half* keys, const __half* values, const __half* h,
                                const float* nk, const float* nq, __half* gated, float* pss,
                                int dims, int streams, int rows, const int* live_rows, bool independent_rows,
                                float eps, float inverse_root) {
    __shared__ float shared[512], norms[2];
    const int stream = blockIdx.x % streams, row = blockIdx.x / streams;
    const int64_t base = int64_t(row) * streams * dims + stream * dims;
    const int outputs = independent_rows ? streams : max(1, (live_rows ? *live_rows : rows) * streams);
    const float inverse_dim = 1.f / float(dims);
    float kk = torch_sum(HalfSquareReader{keys + base}, dims, outputs, shared);
    float inv_k = rsqrtf(kk * inverse_dim + eps);
    if (!threadIdx.x) norms[0] = inv_k;
    __syncthreads();
    float qq = torch_sum(HalfSquareReader{h + base}, dims, outputs, shared);
    float inv_q = rsqrtf(qq * inverse_dim + eps);
    if (!threadIdx.x) norms[1] = inv_q;
    __syncthreads();
    float dot = torch_sum(PleProductReader{keys + base, h + base, nk + stream * dims, nq + stream * dims,
                                         norms[0], norms[1]}, dims, outputs, shared);
    float score = half_round(dot * inverse_root);
    float signed_root = half_round((score > 0.f ? 1.f : score < 0.f ? -1.f : 0.f) * sqrtf(fmaxf(fabsf(score), 1e-6f)));
    float gate = half_round(sigmoid(signed_root));
    for (int col = threadIdx.x; col < dims; col += blockDim.x) {
        float value = half_round(gate * __half2float(values[int64_t(row) * dims + col]));
        gated[base + col] = __float2half_rn(value);
    }
    float ss = torch_sum(PleGatedSquareReader{values + int64_t(row) * dims, gate}, dims, outputs, shared);
    if (!threadIdx.x) pss[row * streams + stream] = ss;
}

__global__ void ple_norm_kernel(const __half* gated, const float* pss, const float* norm, __half* out,
                                int64_t count, int dims, int streams, float eps) {
    for (int64_t at = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; at < count; at += int64_t(gridDim.x) * blockDim.x) {
        int col = at % (dims * streams), row = at / (dims * streams), stream = col / dims;
        float value = (__half2float(gated[at]) * rsqrtf(pss[row * streams + stream] * (1.f / float(dims)) + eps)) * norm[col];
        out[at] = __float2half_rn(value);
    }
}

__global__ void ple_conv_kernel(const __half* gated, const __half* tail, const __half* weight,
                                const __half* h, __half* out, const __half* normalized,
                                int64_t count, int width, int taps, int dilation) {
    const int tail_rows = (taps - 1) * dilation;
    for (int64_t at = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; at < count; at += int64_t(gridDim.x) * blockDim.x) {
        int row = at / width, col = at % width;
        float acc = 0.f;
        for (int tap = 0; tap < taps; ++tap) {
            int source_row = row + tap * dilation;
            float value = __half2float(source_row < tail_rows ? tail[int64_t(source_row) * width + col]
                : normalized[int64_t(source_row - tail_rows) * width + col]);
            acc = acc + value * __half2float(weight[col * taps + tap]);
        }
        float activation = half_round(silu(half_round(acc)));
        float branch = half_round(__half2float(gated[at]) + activation);
        out[at] = __float2half_rn(__half2float(h[at]) + branch);
    }
}

__global__ void add_kernel(const __half* embedding, const __half* hidden, __half* out,
                           int64_t count, int dims, int streams) {
    for (int64_t at = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; at < count; at += int64_t(gridDim.x) * blockDim.x) {
        int row = at / (dims * streams), col = at % dims;
        out[at] = __float2half_rn(__half2float(embedding[int64_t(row) * dims + col]) + __half2float(hidden[at]));
    }
}

__global__ void partial_kernel(const void* values, bool is_float, const float* weights, float* out,
                               int64_t count, int dims, int slots) {
    for (int64_t at = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; at < count; at += int64_t(gridDim.x) * blockDim.x) {
        int row = at / dims, col = at % dims;
        float sum = 0.f;
        for (int slot = 0; slot < slots; ++slot)
            sum = sum + read(values, is_float, (int64_t(row) * slots + slot) * dims + col) * weights[row * slots + slot];
        out[at] = sum;
    }
}

__global__ void shift_kernel(__half* old, const __half* rows, int layers, int taps, int channels,
                             int64_t layer_stride, int64_t row_stride, int keep) {
    int layer = blockIdx.y, col = blockIdx.x * blockDim.x + threadIdx.x;
    if (layer >= layers || col >= channels) return;
    __half saved[16];
    for (int tap = 0; tap < taps; ++tap) {
        int source = keep + tap;
        saved[tap] = source < taps ? old[(int64_t(layer) * taps + source) * channels + col]
            : rows[int64_t(layer) * layer_stride + int64_t(source - taps) * row_stride + col];
    }
    for (int tap = 0; tap < taps; ++tap) old[(int64_t(layer) * taps + tap) * channels + col] = saved[tap];
}

__device__ __forceinline__ int64_t position_value(const void* position, bool is_long) {
    return is_long ? *static_cast<const int64_t*>(position) : *static_cast<const int*>(position);
}
__device__ __forceinline__ float guarded(float value, int* status, int flag, int layer, int row, const int* live) {
    if (!status || (isfinite(value) && fabsf(value) <= 1073741824.f)) return value;
    if (!live || row < *live) { atomicOr(status, flag); atomicCAS(status + 1, -2, layer); }
    return 0.f;
}


__global__ void value_guard_kernel(const void* input, void* output, bool is_float, int64_t elements,
    int64_t heads, int64_t dims, int64_t in_row, int64_t in_head, int64_t out_row, int64_t out_head,
    int* status, int flag, int layer, const int* live) {
    for (int64_t at = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; at < elements;
         at += int64_t(gridDim.x) * blockDim.x) {
        const int64_t row = at / (heads * dims), within = at % (heads * dims);
        const int64_t head = within / dims, col = within % dims;
        const float value = guarded(read(input, is_float, row * in_row + head * in_head + col),
                                    status, flag, layer, int(row), live);
        const int64_t to = row * out_row + head * out_head + col;
        if (is_float) static_cast<float*>(output)[to] = value;
        else static_cast<__half*>(output)[to] = __float2half_rn(value);
    }
}

struct FloatSquareReader {
    const float* values;
    __device__ __forceinline__ float operator()(int col) const {
        float value = values[col];
        return value * value;
    }
};
__device__ __forceinline__ float hadamard(float value) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int offset = 1; offset <= 16; offset *= 2) {
        float cross = __shfl_xor_sync(0xffffffffu, value, offset);
        value = lane & offset ? cross - value : value + cross;
    }
    return value * 0.1767766952966369f;
}
__device__ __forceinline__ void quantized_store(float value, int8_t* out, __half* scales,
                                               int64_t base, int col, int* status, int layer, int row, const int* live) {
    const int lane = threadIdx.x & 31;
    value = hadamard(value);
    float scale = fabsf(value);
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        scale = fmaxf(scale, __shfl_xor_sync(0xffffffffu, scale, offset));
    scale += 1e-10f;
    float stored = half_round(scale);
    if (status && (!isfinite(stored) || stored < 0.f)) {
        if (!lane && (!live || row < *live)) { atomicOr(status, 8); atomicCAS(status + 1, -2, layer); }
        stored = 0.f;
    }
    float code = floorf((value * (1.f / scale)) * 128.f) + 128.f;
    code = fminf(255.f, fmaxf(0.f, code));
    out[base + col] = static_cast<int8_t>(static_cast<int>(code) - 128);
    if (!lane) scales[(base + col) / 32] = __float2half_rn(stored);
}

__device__ __forceinline__ void prep_unit(const __half* p, const float* qw, const float* kw,
    const float* iw, const float* freq, __half* queries, void* kc, void* vc, __half* iq, __half* ikc,
    __half* ks, __half* vs, int* status, const int* live, int qheads, int kvheads, int dim,
    int index_heads, int index_dim, int kbits, int vbits, int layer, int rotary_half, int rows,
    float eps, float rope_scale, bool independent_rows, int unit, int row, int64_t cache_row,
    int64_t coordinate_row, float* shared, float* normalized) {
    const int units = qheads + 2 * kvheads + index_heads + 1, col = threadIdx.x;
    int kind, head, width, start, flag;
    if (unit < qheads) { kind = 0; head = unit; width = dim; start = head * 2 * dim; flag = 1; }
    else if (unit < qheads + kvheads) { kind = 1; head = unit - qheads; width = dim; start = 2 * qheads * dim + head * dim; flag = 2; }
    else if (unit < qheads + 2 * kvheads) { kind = 2; head = unit - qheads - kvheads; width = dim;
        start = 2 * qheads * dim + kvheads * dim + head * dim; flag = 4; }
    else if (unit < units - 1) { kind = 3; head = unit - qheads - 2 * kvheads; width = index_dim;
        start = 2 * qheads * dim + 2 * kvheads * dim + head * index_dim; flag = 16; }
    else { kind = 4; head = 0; width = index_dim; start = 2 * qheads * dim + 2 * kvheads * dim + index_heads * index_dim; flag = 16; }
    const int packed = 2 * qheads * dim + 2 * kvheads * dim + (index_heads + 1) * index_dim;
    float value = col < width ? guarded(__half2float(p[int64_t(row) * packed + start + col]), status, flag, layer, row, live) : 0.f;
    if (kind == 0 || kind == 1 || kind == 3) {
        normalized[col] = value;
        __syncthreads();
        const int norm_heads = kind == 0 ? qheads : kind == 1 ? kvheads : index_heads;
        const int outputs = independent_rows ? norm_heads : max(1, (live ? *live : rows) * norm_heads);
        float total = torch_sum(FloatSquareReader{normalized}, width, outputs, shared);
        float inv = rsqrtf(total * (1.f / float(width)) + eps);
        const float* gamma = kind == 0 ? qw : kind == 1 ? kw : iw;
        normalized[col] = col < width ? half_round((value * inv) * gamma[col]) : 0.f;
        __syncthreads();
        value = normalized[col];
        if (col < 2 * rotary_half) {
            const int pair = col % rotary_half;
            float angle = float(coordinate_row) * freq[pair];
            float sine = sinf(angle), cosine = cosf(angle);
            sine *= rope_scale; cosine *= rope_scale;
            float cross = normalized[col < rotary_half ? col + rotary_half : col - rotary_half];
            value = half_round(value * cosine + (col < rotary_half ? -cross : cross) * sine);
        }
        value = guarded(value, status, flag, layer, row, live);
    }
    if (col >= width) return;
    if (kind == 0) {
        if (kbits) value = half_round(hadamard(value));
        queries[(int64_t(row) * qheads + head) * dim + col] = __float2half_rn(value);
    } else if (kind == 1) {
        const int64_t base = (cache_row * kvheads + head) * dim;
        if (kbits) quantized_store(value, static_cast<int8_t*>(kc), ks, base, col, status, layer, row, live);
        else static_cast<__half*>(kc)[base + col] = __float2half_rn(value);
    } else if (kind == 2) {
        const int64_t base = (cache_row * kvheads + head) * dim;
        if (vbits) quantized_store(value, static_cast<int8_t*>(vc), vs, base, col, status, layer, row, live);
        else static_cast<__half*>(vc)[base + col] = __float2half_rn(value);
    } else if (kind == 3) iq[(int64_t(row) * index_heads + head) * index_dim + col] = __float2half_rn(value);
    else ikc[cache_row * index_dim + col] = __float2half_rn(value);
}

__global__ void prep_kernel(const __half* p, const void* position, bool pos_long, const float* qw,
                            const float* kw, const float* iw, const float* freq, __half* queries,
                            void* kc, void* vc, __half* iq, __half* ikc, __half* ks, __half* vs,
                            int* status, const int* delta, const int* live, int qheads, int kvheads,
                            int dim, int index_heads, int index_dim, int kbits, int vbits, int layer,
                            int rotary_half, int capacity, int rows, float eps, float rope_scale, bool independent_rows) {
    __shared__ float shared[512], normalized[256];
    const int units = qheads + 2 * kvheads + index_heads + 1;
    const int unit = blockIdx.x % units, row = blockIdx.x / units;
    const int64_t cache_row = position_value(position, pos_long) + row;
    if (cache_row < 0 || cache_row >= capacity) return;
    prep_unit(p, qw, kw, iw, freq, queries, kc, vc, iq, ikc, ks, vs, status, live,
        qheads, kvheads, dim, index_heads, index_dim, kbits, vbits, layer, rotary_half, rows,
        eps, rope_scale, independent_rows, unit, row, cache_row,
        cache_row + (delta ? *delta : 0), shared, normalized);
}

// Descriptors are actor-owned rows [State position pointer, State status pointer,
// segment first, segment count]. Python validates canonical State ownership and
// retention before upload; this entry never accepts external pointer records.
__global__ void prep_batch_kernel(const __half* p, const int64_t* descriptor,
    const float* qw, const float* kw, const float* iw, const float* freq, __half* q,
    int8_t* key, int8_t* value, __half* iq, __half* raw_index, __half* ks, __half* vs,
    int total_rows, int layer, int rotary_half, float eps, float rope_scale, bool independent_rows) {
    __shared__ float shared[512], normalized[256];
    constexpr int qheads = 3, kvheads = 1, dim = 256, index_heads = 4, index_dim = 128;
    constexpr int units = qheads + 2 * kvheads + index_heads + 1;
    const int row = blockIdx.x / units, unit = blockIdx.x % units;
    const int64_t* record = descriptor + int64_t(row) * 4;
    const auto* position = reinterpret_cast<const int*>(record[0]);
    auto* status = reinterpret_cast<int*>(record[1]);
    const int64_t first = record[2], count = record[3];
    if (!position || !status || first < 0 || count <= 0 || first > row
            || count > int64_t(total_rows) - first || row >= first + count) {
        asm volatile("trap;");
        return;
    }
    const int64_t coordinate = int64_t(*position) + row - first;
    if (coordinate < 0 || coordinate >= 262144) {
        if (!threadIdx.x) { atomicOr(status, 16); atomicCAS(status + 1, -2, layer); }
        return;
    }
    // Every segment retains the scalar producer's row-dependent Torch tree.
    prep_unit(p, qw, kw, iw, freq, q, key, value, iq, raw_index, ks, vs, status, nullptr,
        qheads, kvheads, dim, index_heads, index_dim, 8, 8, layer, rotary_half, int(count),
        eps, rope_scale, independent_rows, unit, row, row, coordinate, shared, normalized);
}

// Keep the qualified raw/INT8 prep_kernel above intact. This separate Q4
// specialization preserves its stored Half boundaries without adding branches
// or codec registers to the existing launch. The codec owns its staged basis,
// normalization, codebook and packed stores in the shared Q4 header.
template<bool NORM5_POSSIBLE, bool INDEX_STAGED = false, bool COMPACT_POSSIBLE = false>
__global__ void prep_q4_kernel(const __half* p, const void* position, bool pos_long, const float* qw,
                               const float* kw, const float* iw, const float* freq, __half* queries,
                               void* kc, void* vc, __half* iq, __half* ikc, void* ks, void* vs,
                               int* status, const int* delta, const int* live, int qheads, int kvheads,
                               int dim, int index_heads, int index_dim, int kbits, int vbits, int layer,
                               int rotary_half, int capacity, int rows, float eps, float rope_scale,
                               bool independent_rows, int kcodec, int vcodec, const int64_t* stage_origin, int stage_rows) {
    __shared__ float shared[512], normalized[256];
    const int units = qheads + 2 * kvheads + index_heads + 1;
    const int unit = blockIdx.x % units, row = blockIdx.x / units, col = threadIdx.x;
    const int count = live ? *live : rows;
    if (count < 0 || count > rows) {
        if (!col && status) { atomicOr(status, 16); atomicCAS(status + 1, -2, layer); }
        return;
    }
    if (row >= count) return;  // Padded rows do not read source or mutate cache.
    const int64_t start_row = position_value(position, pos_long);
    if (start_row < 0 || start_row >= capacity
            || (INDEX_STAGED ? count > int64_t(capacity) - start_row : row >= int64_t(capacity) - start_row)) {
        if (!col && status) { atomicOr(status, 16); atomicCAS(status + 1, -2, layer); }
        return;
    }
    const int64_t cache_row = start_row + row;
    int64_t stage_offset = 0;
    if constexpr (INDEX_STAGED) {
        const int64_t origin = stage_origin ? *stage_origin : start_row;
        if (origin < 0 || origin > start_row || count > int64_t(stage_rows) - (start_row - origin)) {
            if (!col && status) { atomicOr(status, 16); atomicCAS(status + 1, -2, layer); }
            return;
        }
        stage_offset = start_row - origin;
    }
    int kind, head, width, start, flag;
    if (unit < qheads) { kind = 0; head = unit; width = dim; start = head * 2 * dim; flag = 1; }
    else if (unit < qheads + kvheads) { kind = 1; head = unit - qheads; width = dim; start = 2 * qheads * dim + head * dim; flag = 2; }
    else if (unit < qheads + 2 * kvheads) { kind = 2; head = unit - qheads - kvheads; width = dim;
        start = 2 * qheads * dim + kvheads * dim + head * dim; flag = 4; }
    else if (unit < units - 1) { kind = 3; head = unit - qheads - 2 * kvheads; width = index_dim;
        start = 2 * qheads * dim + 2 * kvheads * dim + head * index_dim; flag = 16; }
    else { kind = 4; head = 0; width = index_dim; start = 2 * qheads * dim + 2 * kvheads * dim + index_heads * index_dim; flag = 16; }
    const int packed = 2 * qheads * dim + 2 * kvheads * dim + (index_heads + 1) * index_dim;
    float value = col < width ? guarded(__half2float(p[int64_t(row) * packed + start + col]), status, flag, layer, row, live) : 0.f;
    if (kind == 0 || kind == 1 || kind == 3) {
        normalized[col] = value;
        __syncthreads();
        const int norm_heads = kind == 0 ? qheads : kind == 1 ? kvheads : index_heads;
        const int outputs = independent_rows ? norm_heads : max(1, count * norm_heads);
        float total = torch_sum(FloatSquareReader{normalized}, width, outputs, shared);
        float inv = rsqrtf(total * (1.f / float(width)) + eps);
        const float* gamma = kind == 0 ? qw : kind == 1 ? kw : iw;
        normalized[col] = col < width ? half_round((value * inv) * gamma[col]) : 0.f;
        __syncthreads();
        value = normalized[col];
        if (col < 2 * rotary_half) {
            const int pair = col % rotary_half;
            float angle = float(cache_row + (delta ? *delta : 0)) * freq[pair];
            float sine = sinf(angle), cosine = cosf(angle);
            sine *= rope_scale; cosine *= rope_scale;
            float cross = normalized[col < rotary_half ? col + rotary_half : col - rotary_half];
            value = half_round(value * cosine + (col < rotary_half ? -cross : cross) * sine);
        }
        value = guarded(value, status, flag, layer, row, live);
    }
    if constexpr (COMPACT_POSSIBLE) {
        if ((kind == 1 && (kcodec == 9 || kcodec == 10)) || (kind == 2 && (vcodec == 9 || vcodec == 10))) {
            const int selected = kind == 1 ? kcodec : vcodec;
            const auto item = selected == 10 ? tf_rq4::encode_component<10>(value, col, shared, flag)
                                              : tf_rq4::encode_component<9>(value, col, shared, flag);
            if (item.errors && status) { atomicOr(status, item.errors); atomicCAS(status + 1, -2, layer); }
            const int64_t base = (cache_row * kvheads + head) * dim;
            tf_rq4::store_encoded(item, col, shared,
                static_cast<uint8_t*>(kind == 1 ? kc : vc), static_cast<float*>(kind == 1 ? ks : vs),
                base, dim, true);
            return;
        }
        if (kind == 0 && (kcodec == 9 || kcodec == 10)) {
            value = half_round(kcodec == 10 ? tf_rq4::rotate_component<10, false>(value, col, shared)
                                           : tf_rq4::rotate_component<9, false>(value, col, shared));
            value = guarded(value, status, flag, layer, row, live);
        }
    }
    if ((kind == 1 && (kcodec == 4 || (NORM5_POSSIBLE && kcodec == 5)))
            || (kind == 2 && (vcodec == 4 || (NORM5_POSSIBLE && vcodec == 5)))) {
        tf_rq4::Encoded item;
        if constexpr (NORM5_POSSIBLE) {
            const int selected = kind == 1 ? kcodec : vcodec;
            item = selected == 5 ? tf_rq4::encode_component<5>(value, col, shared, flag)
                                 : tf_rq4::encode_component<4>(value, col, shared, flag);
        } else item = tf_rq4::encode_component<4>(value, col, shared, flag);
        if (item.errors && status) { atomicOr(status, item.errors); atomicCAS(status + 1, -2, layer); }
        const int64_t base = (cache_row * kvheads + head) * dim;
        tf_rq4::store_encoded(item, col, shared,
            static_cast<uint8_t*>(kind == 1 ? kc : vc), static_cast<float*>(kind == 1 ? ks : vs),
            base, dim, true);
        return;
    }
    if (kind == 0 && (kcodec == 4 || (NORM5_POSSIBLE && kcodec == 5))) {
        if constexpr (NORM5_POSSIBLE) {
            value = kcodec == 5 ? tf_rq4::rotate_component<5, false>(value, col, shared)
                                : tf_rq4::rotate_component<4, false>(value, col, shared);
        } else value = tf_rq4::rotate_component<4, false>(value, col, shared);
        value = half_round(value);
        value = guarded(value, status, flag, layer, row, live);
    }
    if (col >= width) return;
    if (kind == 0) {
        if (kbits == 8) value = half_round(hadamard(value));
        queries[(int64_t(row) * qheads + head) * dim + col] = __float2half_rn(value);
    } else if (kind == 1) {
        const int64_t base = (cache_row * kvheads + head) * dim;
        if (kbits == 8) quantized_store(value, static_cast<int8_t*>(kc), static_cast<__half*>(ks), base, col, status, layer, row, live);
        else static_cast<__half*>(kc)[base + col] = __float2half_rn(value);
    } else if (kind == 2) {
        const int64_t base = (cache_row * kvheads + head) * dim;
        if (vbits == 8) quantized_store(value, static_cast<int8_t*>(vc), static_cast<__half*>(vs), base, col, status, layer, row, live);
        else static_cast<__half*>(vc)[base + col] = __float2half_rn(value);
    } else if (kind == 3) iq[(int64_t(row) * index_heads + head) * index_dim + col] = __float2half_rn(value);
    else ikc[(INDEX_STAGED ? stage_offset + row : cache_row) * index_dim + col] = __float2half_rn(value);
}

__global__ void pool_kernel(const __half* cache, __half* pooled, const void* position, bool pos_long,
                            const float* weight, const float* freq, const int* delta,
                            int ratio, int rows, int dim, int rotary_half, float eps, float rope_scale) {
    __shared__ float shared[512], normalized[256];
    const int block = blockIdx.x, col = threadIdx.x;
    const int64_t start = position_value(position, pos_long);
    if (block < start / ratio || int64_t(block + 1) * ratio > start + rows) return;
    float value = 0.f;
    if (col < dim) {
        value = __half2float(cache[int64_t(block) * ratio * dim + col]);
        for (int row = 1; row < ratio; ++row) value += __half2float(cache[(int64_t(block) * ratio + row) * dim + col]);
        value = half_round(value * (1.f / float(ratio)));
    }
    normalized[col] = value;
    __syncthreads();
    float total = torch_sum(FloatSquareReader{normalized}, dim, gridDim.x, shared);
    float inv = rsqrtf(total * (1.f / float(dim)) + eps);
    normalized[col] = col < dim ? half_round((value * inv) * weight[col]) : 0.f;
    __syncthreads();
    value = normalized[col];
    if (col < rotary_half * 2) {
        float angle = float(int64_t(block) * ratio + (delta ? *delta : 0)) * freq[col % rotary_half];
        float sine = sinf(angle), cosine = cosf(angle); sine *= rope_scale; cosine *= rope_scale;
        float cross = normalized[col < rotary_half ? col + rotary_half : col - rotary_half];
        value = half_round(value * cosine + (col < rotary_half ? -cross : cross) * sine);
    }
    if (col < dim) pooled[int64_t(block) * dim + col] = __float2half_rn(value);
}

__device__ __forceinline__ bool staged_index_frame(const void* position, bool pos_long, const int* live,
                                                  int rows, int capacity, const int64_t* origin, int stage_rows,
                                                  int* status, int layer, int64_t& start, int& count, int64_t& offset) {
    count = live ? *live : rows;
    if (count < 0 || count > rows) {
        if (!threadIdx.x && status) { atomicOr(status, 16); atomicCAS(status + 1, -2, layer); }
        return false;
    }
    if (!count) return false;
    start = position_value(position, pos_long);
    if (start < 0 || start >= capacity || count > int64_t(capacity) - start) {
        if (!threadIdx.x && status) { atomicOr(status, 16); atomicCAS(status + 1, -2, layer); }
        return false;
    }
    const int64_t first = origin ? *origin : start;
    if (first < 0 || first > start || count > int64_t(stage_rows) - (start - first)) {
        if (!threadIdx.x && status) { atomicOr(status, 16); atomicCAS(status + 1, -2, layer); }
        return false;
    }
    offset = start - first;
    return true;
}

__global__ void pool_staged_kernel(const __half* stage, const __half* ring, __half* pooled,
                                   const void* position, bool pos_long, const float* weight,
                                   const float* freq, const int* delta, const int* live, int* status,
                                   int ratio, int rows, int dim, int ring_rows, int capacity,
                                   int logical_outputs, int rotary_half, int layer, float eps, float rope_scale,
                                   const int64_t* stage_origin, int stage_rows) {
    __shared__ float shared[512], normalized[256];
    int64_t start, stage_offset;
    int count;
    if (!staged_index_frame(position, pos_long, live, rows, capacity, stage_origin, stage_rows,
                            status, layer, start, count, stage_offset)) return;
    const int64_t block = start / ratio + blockIdx.x;
    if (block >= logical_outputs || (block + 1) * ratio > start + count) return;
    const int col = threadIdx.x;
    float value = 0.f;
    if (col < dim) {
        for (int tap = 0; tap < ratio; ++tap) {
            const int64_t source = block * ratio + tap;
            const float next = guarded(__half2float(source < start
                ? ring[(source % ring_rows) * dim + col]
                : stage[(stage_offset + source - start) * dim + col]), status, 16, layer, 0, nullptr);
            if (!tap) value = next;
            else value = value + next;
        }
        value = half_round(value * (1.f / float(ratio)));
    }
    normalized[col] = value;
    __syncthreads();
    // The old full-cache launch supplied capacity/ratio via gridDim.x.
    // A bounded launch must retain that logical reduction-mode input.
    float total = torch_sum(FloatSquareReader{normalized}, dim, logical_outputs, shared);
    float inv = rsqrtf(total * (1.f / float(dim)) + eps);
    normalized[col] = col < dim ? half_round((value * inv) * weight[col]) : 0.f;
    __syncthreads();
    value = normalized[col];
    if (col < rotary_half * 2) {
        float angle = float(block * ratio + (delta ? *delta : 0)) * freq[col % rotary_half];
        float sine = sinf(angle), cosine = cosf(angle); sine *= rope_scale; cosine *= rope_scale;
        float cross = normalized[col < rotary_half ? col + rotary_half : col - rotary_half];
        value = half_round(value * cosine + (col < rotary_half ? -cross : cross) * sine);
    }
    if (col < dim) pooled[block * dim + col] = __float2half_rn(guarded(value, status, 16, layer, 0, nullptr));
}

__global__ void ring_publish_kernel(const __half* stage, __half* ring, const void* position, bool pos_long,
                                    const int* live, int* status, int rows, int dim, int ring_rows,
                                    int capacity, int layer, const int64_t* stage_origin, int stage_rows) {
    int64_t start, stage_offset;
    int count;
    if (!staged_index_frame(position, pos_long, live, rows, capacity, stage_origin, stage_rows,
                            status, layer, start, count, stage_offset)) return;
    const int retained = min(count, ring_rows), first = count - retained;
    for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         index < int64_t(retained) * dim; index += int64_t(gridDim.x) * blockDim.x) {
        const int source = first + index / dim, col = index % dim;
        // At most ring_rows distinct logical rows publish after pooling consumed
        // all staging/carry inputs. No modulo destination has multiple writers.
        ring[((start + source) % ring_rows) * dim + col] = stage[(stage_offset + source) * dim + col];
    }
}

__global__ void qsa_dense_kernel(const void* position, bool pos_long, int* sparse, int* counts, int rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) { sparse[row] = 0; counts[row] = position_value(position, pos_long) + row + 1; }
}

template<int ITEMS>
__global__ void qsa_rows_kernel(const __half* query, const __half* pooled, const void* position, bool pos_long,
                               int* sparse, int* counts, int* ids, float* scores,
                               int heads, int dim, int ratio, int budget, int blocks, int id_stride, int score_stride) {
    using Sort = cub::BlockRadixSort<unsigned long long, THREADS, ITEMS>;
    __shared__ union Storage {
        unsigned long long keys[THREADS * ITEMS];
        typename Sort::TempStorage sort;
    } storage;
    const int row = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int end = position_value(position, pos_long) + row + 1;
    const int complete = end / ratio, top = budget / ratio;
    if (!threadIdx.x) { sparse[row] = complete > top; counts[row] = complete > top ? budget + end % ratio : end; }
    // This test uses the device position, so a captured context bucket remains
    // cheap until a request actually crosses the sparse-attention threshold.
    if (complete <= top) return;
    for (int block = warp; block < THREADS * ITEMS; block += THREADS / 32) {
        float score = 0.f;
        if (block < blocks && block < complete) {
            for (int head = 0; head < heads; ++head) {
                float dot = 0.f;
                for (int col = lane; col < dim; col += 32)
                    dot = dot + __half2float(query[(int64_t(row) * heads + head) * dim + col]) *
                                __half2float(pooled[int64_t(block) * dim + col]);
                dot = warp_sum(dot);
                score = score + (dot < 0.f ? 0.f : dot);
            }
            score /= sqrtf(float(dim));
        }
        if (!lane) {
            const bool valid = block < blocks && block < complete;
            // Under the finite-score contract, +0 initialization and the
            // ordered additions of nonnegative ReLU terms produce +0 for a
            // zero score (default round-to-nearest, no fast math). Its bits
            // therefore preserve numeric FP32 order; the inverse logical
            // pooled-group ID breaks ties, never a physical owner/page index.
            // Nonfinite/range qualification is a separate numerical gate.
            storage.keys[block] = valid ? ((static_cast<unsigned long long>(__float_as_uint(score)) + 1ULL) << 32)
                                          | (0xffffffffu - static_cast<unsigned int>(block)) : 0ULL;
            if (block < blocks) scores[int64_t(row) * score_stride + block] = valid ? score : -CUDART_INF_F;
        }
    }
    __syncthreads();
    unsigned long long keys[ITEMS];
#pragma unroll
    for (int item = 0; item < ITEMS; ++item) keys[item] = storage.keys[threadIdx.x * ITEMS + item];
    __syncthreads();
    Sort(storage.sort).SortDescending(keys);
    __syncthreads();
#pragma unroll
    for (int item = 0; item < ITEMS; ++item) {
        int rank = threadIdx.x * ITEMS + item;
        keys[item] = rank < top ? 0xffffffffu - static_cast<unsigned int>(keys[item]) : ~0ULL;
    }
    Sort(storage.sort).Sort(keys);
#pragma unroll
    for (int item = 0; item < ITEMS; ++item) {
        const int rank = threadIdx.x * ITEMS + item;
        if (rank < top) for (int offset = 0; offset < ratio; ++offset)
            ids[int64_t(row) * id_stride + rank * ratio + offset] = static_cast<int>(keys[item]) * ratio + offset;
    }
    for (int offset = threadIdx.x; offset < ratio; offset += blockDim.x)
        ids[int64_t(row) * id_stride + budget + offset] = complete * ratio + offset;
}

template<typename V> V* ptr(const T& x) { return reinterpret_cast<V*>(x.data_ptr()); }
const void* optional_ptr(const O& x) { return x ? x->data_ptr() : nullptr; }
int blocks(int64_t count) { return int(std::min<int64_t>((count + THREADS - 1) / THREADS, 65535)); }
}  // namespace

void family_writeback_cuda(const T& h, T& out, T& pss, int streams, int mode,
                           const O& branch, const O& inject, const O& y, const O& weights) {
    int rows = h.size(0), dims = h.size(1) / streams;
    if (rows) writeback_kernel<<<rows * streams * (dims / 256), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        ptr<__half>(h), ptr<__half>(out), ptr<float>(pss), rows, dims, streams, mode, optional_ptr(branch),
        branch && branch->scalar_type() == at::kFloat, branch && branch->dim() == 3 ? branch->size(0) : 1,
        static_cast<const __half*>(optional_ptr(inject)), optional_ptr(y), y && y->scalar_type() == at::kFloat,
        static_cast<const float*>(optional_ptr(weights)), y ? y->size(1) : 0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void family_normed_cuda(const T& h, const T& pss, const T& scale, T& out, T& xs, int streams, float eps) {
    if (h.numel()) normed_kernel<<<blocks(h.numel()), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        ptr<__half>(h), ptr<float>(pss), ptr<float>(scale), ptr<__half>(out), ptr<float>(xs),
        h.size(0), h.size(1) / streams, streams, eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void family_act_cuda(const T& input, T& out, T& xs, const O& inject, int streams, int low) {
    if (out.size(0)) act_kernel<<<out.size(0), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr(), input.scalar_type() == at::kFloat, ptr<__half>(out), ptr<float>(xs),
        inject ? ptr<__half>(*inject) : nullptr, out.size(0), low, streams,
        input.dim() == 3 ? input.size(0) : 1, input.size(-1), input.dim() == 3);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void family_mix_cuda(const T& up, const T& normalized, T& out, T& xs, int streams) {
    if (out.numel()) mix_kernel<<<blocks(out.numel()), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        ptr<__half>(up), ptr<__half>(normalized), ptr<__half>(out), ptr<float>(xs), out.size(0), out.size(1), streams);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void family_rms_cuda(const T& x, const T& weight, T& out, T& xs, int group, float eps) {
    if (x.numel()) rms_kernel<<<x.numel() / group, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        ptr<__half>(x), ptr<float>(weight), ptr<__half>(out), ptr<float>(xs), x.size(1), group, eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void family_gate_cuda(const T& value, const T& projection, T& out, T& xs, int heads, int dims) {
    if (out.numel()) gate_kernel<<<blocks(out.numel()), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        ptr<__half>(value), ptr<__half>(projection), ptr<__half>(out), ptr<float>(xs), value.size(0), heads, dims, projection.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void family_ple_embed_cuda(const T& values, T& out, T& xs, int rows, int heads, int dim, float scale) {
    if (out.numel()) embed_kernel<<<blocks(out.numel()), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        ptr<__half>(values), ptr<__half>(out), ptr<float>(xs), out.numel() / 32, scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void family_ple_gate_cuda(const T& keys, const T& values, const T& h, const T& nk, const T& nq,
                          T& gated, T& pss, const O& live_rows, int streams, float eps, bool independent_rows) {
    const float inverse_root = 1.f / float(std::sqrt(double(h.size(1) / streams)));
    if (h.size(0)) ple_gate_kernel<<<h.size(0) * streams, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        ptr<__half>(keys), ptr<__half>(values), ptr<__half>(h), ptr<float>(nk), ptr<float>(nq),
        ptr<__half>(gated), ptr<float>(pss), h.size(1) / streams, streams, h.size(0),
        live_rows ? ptr<int>(*live_rows) : nullptr, independent_rows, eps, inverse_root);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void family_ple_conv_cuda(const T& gated, const T& pss, const T& norm, const T& tail,
                          const T& weight, const T& h, T& out, T& nrow, int streams, int dilation, float eps) {
    if (h.numel()) {
        auto stream = at::cuda::getCurrentCUDAStream();
        ple_norm_kernel<<<blocks(h.numel()), THREADS, 0, stream>>>(ptr<__half>(gated), ptr<float>(pss), ptr<float>(norm),
            ptr<__half>(nrow), h.numel(), h.size(1) / streams, streams, eps);
        ple_conv_kernel<<<blocks(h.numel()), THREADS, 0, stream>>>(ptr<__half>(gated), ptr<__half>(tail), ptr<__half>(weight),
            ptr<__half>(h), ptr<__half>(out), ptr<__half>(nrow), h.numel(), h.size(1), weight.size(1), dilation);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void family_add_cuda(const T& embedding, const T& hidden, T& out, int streams) {
    if (out.numel()) add_kernel<<<blocks(out.numel()), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        ptr<__half>(embedding), ptr<__half>(hidden), ptr<__half>(out), out.numel(), embedding.size(1), streams);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void family_partial_cuda(const T& values, const T& weights, T& out, int rows) {
    if (rows) partial_kernel<<<blocks(int64_t(rows) * out.size(1)), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        values.data_ptr(), values.scalar_type() == at::kFloat, ptr<float>(weights), ptr<float>(out),
        int64_t(rows) * out.size(1), out.size(1), values.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void family_shift_cuda(T& old, const T& rows, int keep, int channels) {
    if (old.numel()) shift_kernel<<<dim3((channels + THREADS - 1) / THREADS, old.size(0)), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        ptr<__half>(old), ptr<__half>(rows), old.size(0), old.size(1), channels, rows.stride(0), rows.stride(1), keep);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void family_prep_cuda(const T& p, const T& position, const T& qw, const T& kw, const T& iw, const T& frequency,
                      T& q, T& kc, T& vc, T& iq, T& ikc, const O& ks, const O& vs, const O& status,
                      const O& delta, const O& live, int qheads, int kvheads, int dim, int index_heads,
                      int index_dim, int kbits, int vbits, int layer, float eps, float rope_scale, bool independent_rows) {
    if (p.size(0)) prep_kernel<<<p.size(0) * (qheads + 2 * kvheads + index_heads + 1), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        ptr<__half>(p), position.data_ptr(), position.scalar_type() == at::kLong, ptr<float>(qw), ptr<float>(kw), ptr<float>(iw),
        ptr<float>(frequency), ptr<__half>(q), kc.data_ptr(), vc.data_ptr(), ptr<__half>(iq), ptr<__half>(ikc),
        ks ? ptr<__half>(*ks) : nullptr, vs ? ptr<__half>(*vs) : nullptr, status ? ptr<int>(*status) : nullptr,
        delta ? ptr<int>(*delta) : nullptr, live ? ptr<int>(*live) : nullptr, qheads, kvheads, dim, index_heads,
        index_dim, kbits, vbits, layer, frequency.numel(), kc.size(0), p.size(0), eps, rope_scale, independent_rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template<bool INDEX_STAGED>
void prep_format_cuda(const T& p, const T& position, const T& qw, const T& kw, const T& iw, const T& frequency,
                         T& q, T& kc, T& vc, T& iq, T& ikc, const O& ks, const O& vs, const O& status,
                         const O& delta, const O& live, int qheads, int kvheads, int dim, int index_heads,
                         int index_dim, int kbits, int vbits, int layer, float eps, float rope_scale,
                         bool independent_rows, int kcodec, int vcodec, const O& stage_origin) {
    if (p.size(0)) {
        // CODEC4 keeps a separate compiled path without CODEC5's norm work or
        // register lifetimes; mixed CODEC4/5 follows the extended specialization.
        auto launch = [&](auto norm5, auto compact) {
            prep_q4_kernel<decltype(norm5)::value, INDEX_STAGED, decltype(compact)::value><<<p.size(0) * (qheads + 2 * kvheads + index_heads + 1),
                                                    THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                ptr<__half>(p), position.data_ptr(), position.scalar_type() == at::kLong,
                ptr<float>(qw), ptr<float>(kw), ptr<float>(iw), ptr<float>(frequency), ptr<__half>(q),
                kc.data_ptr(), vc.data_ptr(), ptr<__half>(iq), ptr<__half>(ikc),
                ks ? ks->data_ptr() : nullptr, vs ? vs->data_ptr() : nullptr, status ? ptr<int>(*status) : nullptr,
                delta ? ptr<int>(*delta) : nullptr, live ? ptr<int>(*live) : nullptr, qheads, kvheads, dim, index_heads,
                index_dim, kbits, vbits, layer, frequency.numel(), kc.size(0), p.size(0), eps, rope_scale,
                independent_rows, kcodec, vcodec, stage_origin ? ptr<int64_t>(*stage_origin) : nullptr, ikc.size(0));
        };
        auto dispatch = [&](auto compact) {
            if (kcodec == 5 || vcodec == 5) launch(std::true_type{}, compact);
            else launch(std::false_type{}, compact);
        };
        if (kcodec == 9 || vcodec == 9 || kcodec == 10 || vcodec == 10) dispatch(std::true_type{});
        else dispatch(std::false_type{});
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void family_prep_q4_cuda(const T& p, const T& position, const T& qw, const T& kw, const T& iw, const T& frequency,
                         T& q, T& kc, T& vc, T& iq, T& ikc, const O& ks, const O& vs, const O& status,
                         const O& delta, const O& live, int qheads, int kvheads, int dim, int index_heads,
                         int index_dim, int kbits, int vbits, int layer, float eps, float rope_scale,
                         bool independent_rows, int kcodec, int vcodec) {
    prep_format_cuda<false>(p, position, qw, kw, iw, frequency, q, kc, vc, iq, ikc, ks, vs, status, delta, live,
                             qheads, kvheads, dim, index_heads, index_dim, kbits, vbits, layer, eps, rope_scale,
                             independent_rows, kcodec, vcodec, {});
}
void family_prep_staged_cuda(const T& p, const T& position, const T& qw, const T& kw, const T& iw, const T& frequency,
                             T& q, T& kc, T& vc, T& iq, T& ikc, const O& ks, const O& vs, const O& status,
                             const O& delta, const O& live, int qheads, int kvheads, int dim, int index_heads,
                             int index_dim, int kbits, int vbits, int layer, float eps, float rope_scale,
                             bool independent_rows, int kcodec, int vcodec, const O& stage_origin) {
    prep_format_cuda<true>(p, position, qw, kw, iw, frequency, q, kc, vc, iq, ikc, ks, vs, status, delta, live,
                            qheads, kvheads, dim, index_heads, index_dim, kbits, vbits, layer, eps, rope_scale,
                            independent_rows, kcodec, vcodec, stage_origin);
}
void family_pool_cuda(const T& cache, T& pooled, const T& position, const T& weight, const T& frequency,
                      const O& delta, int ratio, int rows, float eps, float rope_scale) {
    const int count = cache.size(0) / ratio;
    if (count) pool_kernel<<<count, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(ptr<__half>(cache), ptr<__half>(pooled),
        position.data_ptr(), position.scalar_type() == at::kLong, ptr<float>(weight), ptr<float>(frequency),
        delta ? ptr<int>(*delta) : nullptr, ratio, rows, cache.size(1), frequency.numel(), eps, rope_scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void family_pool_staged_cuda(const T& stage, const T& ring, T& pooled, const T& position, const T& weight,
                             const T& frequency, const O& delta, int ratio, int rows, int capacity,
                             float eps, float rope_scale, const O& status, const O& live, int layer,
                             const O& stage_origin) {
    const int count = int((int64_t(rows) + ratio - 1) / ratio);
    if (count) pool_staged_kernel<<<count, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        ptr<__half>(stage), ptr<__half>(ring), ptr<__half>(pooled), position.data_ptr(),
        position.scalar_type() == at::kLong, ptr<float>(weight), ptr<float>(frequency),
        delta ? ptr<int>(*delta) : nullptr, live ? ptr<int>(*live) : nullptr, status ? ptr<int>(*status) : nullptr,
        ratio, rows, stage.size(1), ring.size(0), capacity, capacity / ratio, frequency.numel(), layer, eps, rope_scale,
        stage_origin ? ptr<int64_t>(*stage_origin) : nullptr, stage.size(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void family_ring_publish_cuda(const T& stage, T& ring, const T& position, int rows, int capacity,
                              const O& status, const O& live, int layer, const O& stage_origin) {
    const int64_t count = int64_t(std::min<int64_t>(rows, ring.size(0))) * stage.size(1);
    if (count) ring_publish_kernel<<<blocks(count), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        ptr<__half>(stage), ptr<__half>(ring), position.data_ptr(), position.scalar_type() == at::kLong,
        live ? ptr<int>(*live) : nullptr, status ? ptr<int>(*status) : nullptr,
        rows, stage.size(1), ring.size(0), capacity, layer,
        stage_origin ? ptr<int64_t>(*stage_origin) : nullptr, stage.size(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void family_qsa_dense_cuda(const T& position, T& sparse, T& counts, int rows) {
    if (rows) qsa_dense_kernel<<<(rows + THREADS - 1) / THREADS, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        position.data_ptr(), position.scalar_type() == at::kLong, ptr<int>(sparse), ptr<int>(counts), rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void family_qsa_rows_cuda(const T& query, const T& pooled, const T& position, T& sparse, T& counts,
                          T& ids, T& scores, int rows, int ratio, int budget, int count) {
    auto launch = [&](auto kernel) {
        kernel<<<rows, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            ptr<__half>(query), ptr<__half>(pooled), position.data_ptr(), position.scalar_type() == at::kLong,
            ptr<int>(sparse), ptr<int>(counts), ptr<int>(ids), ptr<float>(scores), query.size(1), query.size(2),
            ratio, budget, count, ids.size(1), scores.size(1));
    };
    if (rows) {
        if (count <= 1024) launch(qsa_rows_kernel<4>);
        else if (count <= 2048) launch(qsa_rows_kernel<8>);
        else launch(qsa_rows_kernel<16>);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// The output is an existing exclusive actor buffer. No allocation or host read.
void family_value_guard_cuda(const T& input, T& output, const T& status, const O& live, int flag, int layer) {
    const int64_t elements = input.numel();
    if (!elements) return;
    const int64_t heads = input.dim() == 3 ? input.size(1) : 1;
    const int64_t dims = input.size(-1);
    const int blocks = int(std::min<int64_t>((elements - 1) / THREADS + 1, 4096));
    value_guard_kernel<<<blocks, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr(), output.data_ptr(), input.scalar_type() == at::kFloat, elements, heads, dims,
        input.stride(0), input.dim() == 3 ? input.stride(1) : 0,
        output.stride(0), output.dim() == 3 ? output.stride(1) : 0,
        status.data_ptr<int>(), flag, layer, live ? live->data_ptr<int>() : nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int family_prepare_value_guard_cuda() {
    cudaFuncAttributes attributes;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, value_guard_kernel));
    return 1;
}

int family_prepare_owner_prelude_cuda() {
    cudaFuncAttributes attributes;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, prep_kernel));
    return 1;
}

void family_owner_prep_batch_cuda(const T& p, const T& descriptor, const T& qw, const T& kw,
    const T& iw, const T& frequency, T& q, T& key, T& value, T& iq, T& raw_index, T& ks, T& vs,
    int layer, float eps, float rope_scale, bool independent_rows) {
    if (p.size(0)) prep_batch_kernel<<<p.size(0) * 10, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        ptr<__half>(p), ptr<int64_t>(descriptor), ptr<float>(qw), ptr<float>(kw), ptr<float>(iw),
        ptr<float>(frequency), ptr<__half>(q), ptr<int8_t>(key), ptr<int8_t>(value),
        ptr<__half>(iq), ptr<__half>(raw_index), ptr<__half>(ks), ptr<__half>(vs),
        p.size(0), layer, frequency.numel(), eps, rope_scale, independent_rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int family_prepare_owner_batch_cuda() {
    cudaFuncAttributes attributes;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, prep_batch_kernel));
    return 1;
}
