// Fixed absolute K tiles, FP16 operands/probabilities, FP32 partials and ordered PV reduction.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <math_constants.h>
#include <optional>
#include "pv_dispatch.cuh"
#include "rotorquant_q4.cuh"
#include "packed_kv_decode.cuh"

using T = at::Tensor;
using O = std::optional<T>;

namespace {
constexpr int SCORE_TILE = 32, VALUE_TILE = 128, VALUE_DIMS = 64;
namespace wmma = nvcuda::wmma;

__device__ __forceinline__ int base_position(const void* position, bool is_long) {
    return is_long ? int(*static_cast<const int64_t*>(position)) : *static_cast<const int*>(position);
}
__device__ __forceinline__ bool visible(int key, int row, int end, bool qsa,
                                       const int* sparse, const int* counts, const int* ids, int id_stride) {
    if (key >= end || key < 0) return false;
    if (!qsa || !sparse[row]) return true;
    int low = 0, high = counts[row];
    while (low < high) {
        int mid = (low + high) / 2;
        if (ids[int64_t(row) * id_stride + mid] < key) low = mid + 1; else high = mid;
    }
    return low < counts[row] && ids[int64_t(row) * id_stride + low] == key;
}
__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset; offset >>= 1) value += __shfl_xor_sync(0xffffffffu, value, offset);
    return value;
}
__device__ __forceinline__ float block_sum(float value, float* shared) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    value = warp_sum(value);
    if (!lane) shared[warp] = value;
    __syncthreads();
    value = threadIdx.x < 8 ? shared[lane] : 0.f;
    if (!warp) value = warp_sum(value);
    if (!threadIdx.x) shared[0] = value;
    __syncthreads();
    return shared[0];
}
__device__ __forceinline__ float block_max(float value, float* shared) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int offset = 16; offset; offset >>= 1) value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    if (!lane) shared[warp] = value;
    __syncthreads();
    value = threadIdx.x < 8 ? shared[lane] : -CUDART_INF_F;
    if (!warp) for (int offset = 16; offset; offset >>= 1) value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    if (!threadIdx.x) shared[0] = value;
    __syncthreads();
    return shared[0];
}

template<bool SUM>
__device__ __forceinline__ float register_softmax_reduce(const float* scores, int end,
                                                         float maximum, float* shared) {
    // SoftMaxForwardReg uses1024 logical lanes and down-shuffle reductions.
    // Four serial virtual groups preserve that exact tree with256 threads.
    const int lane = threadIdx.x & 31;
    __syncthreads();  // All lanes consume the previous reduction before reuse.
    for (int group = 0; group < 4; ++group) {
        const int logical = threadIdx.x + group * 256;
        float value = SUM ? 0.f : -CUDART_INF_F;
        for (int key = logical; key < end; key += 1024) {
            if constexpr (SUM) value += expf(scores[key] - maximum);
            else value = fmaxf(value, scores[key]);
        }
        for (int offset = 16; offset; offset >>= 1) {
            float cross = __shfl_down_sync(0xffffffffu, value, offset);
            if constexpr (SUM) value += cross;
            else value = fmaxf(value, cross);
        }
        if (!lane) shared[logical >> 5] = value;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        float value = shared[threadIdx.x];
        for (int offset = 16; offset; offset >>= 1) {
            float cross = __shfl_down_sync(0xffffffffu, value, offset);
            if constexpr (SUM) value += cross;
            else value = fmaxf(value, cross);
        }
        if (!threadIdx.x) shared[0] = value;
    }
    __syncthreads();
    return shared[0];
}

template<int DIM>
__global__ void scores_kernel(const __half* q, const void* keys, const void* scales, int bits, int codec,
                               const void* position, bool pos_long, const int* sparse, const int* counts,
                               const int* ids, int id_stride, bool qsa, float* scores,
                               int score_stride, int rows, int heads, int kvheads, int context, float scale) {
    __shared__ __align__(32) __half query_tile[16 * DIM];
    __shared__ __align__(32) __half key_tile[SCORE_TILE * DIM];
    __shared__ __align__(32) __half serial_keys[DIM * (SCORE_TILE + 1)];
    __shared__ __align__(32) float output[16 * SCORE_TILE];
    const int tile = blockIdx.x, query_group = blockIdx.y, kvhead = blockIdx.z;
    const int group_heads = heads / kvheads, query_count = rows * group_heads;
    const int first_query = query_group * 16, first_key = tile * SCORE_TILE;
    const int position0 = base_position(position, pos_long);
    const int max_row = min(rows - 1, (first_query + 15) / group_heads);
    if (first_query >= query_count || first_key >= min(context, position0 + max_row + 1)) return;
    for (int at = threadIdx.x; at < 16 * DIM; at += blockDim.x) {
        int m = first_query + at / DIM, col = at % DIM;
        int row = m / group_heads, head = kvhead * group_heads + m % group_heads;
        query_tile[at] = m < query_count ? q[(int64_t(row) * heads + head) * DIM + col] : __float2half_rn(0.f);
    }
    for (int at = threadIdx.x; at < SCORE_TILE * DIM; at += blockDim.x) {
        int key = first_key + at / DIM, col = at % DIM;
        key_tile[at] = key < min(context, position0 + max_row + 1)
            ? tf_sm70::decoded_half(keys, scales, bits, codec, (int64_t(key) * kvheads + kvhead) * DIM + col) : __float2half_rn(0.f);
        serial_keys[col * (SCORE_TILE + 1) + at / DIM] = key_tile[at];
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5;
    if (warp < SCORE_TILE / 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
        wmma::fill_fragment(c, 0.f);
#pragma unroll
        for (int offset = 0; offset < DIM; offset += 16) {
            wmma::load_matrix_sync(a, query_tile + offset, DIM);
            wmma::load_matrix_sync(b, key_tile + warp * 16 * DIM + offset, DIM);
            wmma::mma_sync(c, a, b, c);
        }
        wmma::store_matrix_sync(output + warp * 16, c, SCORE_TILE, wmma::mem_row_major);
    }
    __syncthreads();
    for (int at = threadIdx.x; at < 16 * SCORE_TILE; at += blockDim.x) {
        int m = first_query + at / SCORE_TILE, key = first_key + at % SCORE_TILE;
        if (m >= query_count || key >= context) continue;
        int row = m / group_heads, head = kvhead * group_heads + m % group_heads;
        int end = min(context, position0 + row + 1);
        float value = -CUDART_INF_F;
        if (visible(key, row, end, qsa, sparse, counts, ids, id_stride)) {
            if (end <= 8192 && qk_modes[end] == 1) {
                value = output[at] * scale;
            } else {
                float accum[16] = {0.f};
#pragma unroll
                for (int offset = 0; offset < DIM; offset += 16) {
#pragma unroll
                    for (int reg = 0; reg < 16; ++reg) {
                        float a = __half2float(query_tile[(at / SCORE_TILE) * DIM + offset + reg]);
                        float b = __half2float(serial_keys[(offset + reg) * (SCORE_TILE + 1) + at % SCORE_TILE]);
                        accum[reg] = __fadd_rn(accum[reg], __fmul_rn(a, b));
                    }
                }
                value = accum[0];
#pragma unroll
                for (int reg = 1; reg < 16; ++reg) value = __fadd_rn(value, accum[reg]);
                value = __fmul_rn(value, scale);
            }
        }
        scores[(int64_t(row) * heads + head) * score_stride + key] = value;
    }
    // The pinned one-key QK path uses adjacent Half pairs per each of128
    // threads, then a shared-memory half-block reduction (64,32,...,1).
    // The reused output tile is no longer needed after the score stores.
    if (position0 == 0 && first_key == 0 && first_query == 0) {
        __syncthreads();
        for (int head = 0; head < min(group_heads, 16); ++head) {
            const int col = threadIdx.x * 2;
            float lo = 0.f, hi = 0.f;
            if (col < DIM) {
                lo = __fadd_rn(0.f, __fmul_rn(__half2float(query_tile[head * DIM + col]),
                                            __half2float(serial_keys[col * (SCORE_TILE + 1)])));
                hi = __fadd_rn(0.f, __fmul_rn(__half2float(query_tile[head * DIM + col + 1]),
                                            __half2float(serial_keys[(col + 1) * (SCORE_TILE + 1)])));
            }
            output[threadIdx.x] = __fadd_rn(__fadd_rn(0.f, lo), hi);
            __syncthreads();
            for (int offset = 64; offset; offset /= 2) {
                if (threadIdx.x < offset)
                    output[threadIdx.x] = __fadd_rn(output[threadIdx.x], output[threadIdx.x + offset]);
                __syncthreads();
            }
            if (!threadIdx.x) scores[int64_t(kvhead * group_heads + head) * score_stride] = __fmul_rn(output[0], scale);
            __syncthreads();
        }
    }
}

__global__ void probabilities_kernel(const float* scores, __half* probabilities,
                                      const void* position, bool pos_long, int stride, int heads, int context) {
    __shared__ float shared[32];
    const int query = blockIdx.x, row = query / heads;
    const int end = min(context, base_position(position, pos_long) + row + 1);
    const int64_t base = int64_t(query) * stride;
    // Torch's ordinary contiguous FP32 softmax uses PersistentSoftmax for
    // up to2048 elements: one warp accumulates lane+32*i, then XOR-reduces.
    // Using the true causal end keeps this tree independent of graph padding.
    if (end <= 2048) {
        if (threadIdx.x >= 32) return;
        float maximum = -CUDART_INF_F;
        for (int key = threadIdx.x; key < end; key += 32) maximum = fmaxf(maximum, scores[base + key]);
        for (int offset = 16; offset; offset >>= 1)
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, offset));
        float sum = 0.f;
        for (int key = threadIdx.x; key < end; key += 32) sum += expf(scores[base + key] - maximum);
        sum = warp_sum(sum);
        for (int key = threadIdx.x; key < end; key += 32)
            probabilities[base + key] = __float2half_rn(expf(scores[base + key] - maximum) / sum);
        return;
    }
    if (end <= 9216) {
        // The pinned ordinary FP32 path selects the register kernel while
        // ceil(end/1024)<10. Choose by causal end, not a padded graph bucket.
        float maximum = register_softmax_reduce<false>(scores + base, end, 0.f, shared);
        float sum = register_softmax_reduce<true>(scores + base, end, maximum, shared);
        for (int key = threadIdx.x; key < end; key += blockDim.x)
            probabilities[base + key] = __float2half_rn(expf(scores[base + key] - maximum) / sum);
        return;
    }
    float maximum = -CUDART_INF_F;
    for (int key = threadIdx.x; key < end; key += blockDim.x) maximum = fmaxf(maximum, scores[base + key]);
    maximum = block_max(maximum, shared);
    float sum = 0.f;
    for (int key = threadIdx.x; key < end; key += blockDim.x) sum += expf(scores[base + key] - maximum);
    sum = block_sum(sum, shared);
    for (int key = threadIdx.x; key < end; key += blockDim.x)
        probabilities[base + key] = __float2half_rn(expf(scores[base + key] - maximum) / sum);
}


__device__ __forceinline__ int pv_chunk(int end, int splits) {
    return splits == 1 ? end : ((end + splits - 1) / splits + 7) / 8 * 8;
}

template<int DIM>
__global__ void values_kernel(const __half* probabilities, int probability_stride,
                               const void* values, const void* scales, int bits, int codec,
                               const void* position, bool pos_long, const int* sparse, const int* counts,
                               const int* ids, int id_stride, bool qsa, float* partials,
                               int partial_stride, int heads, int kvheads, int context) {
    __shared__ __align__(32) __half probability_tile[16 * 16];
    __shared__ __align__(32) __half value_tile[16 * VALUE_DIMS];
    __shared__ __align__(32) float output[16 * VALUE_DIMS];
    const int tile = blockIdx.x, dim_tile = blockIdx.y, query = blockIdx.z;
    const int row = query / heads, head = query % heads, kvhead = head / (heads / kvheads);
    const int end = min(context, base_position(position, pos_long) + row + 1);
    const int first_col = dim_tile * VALUE_DIMS;
    if (end < 1 || end > 8192) return;  // The host eligibility rejects this scope before staging.
    const int mode = pv_modes[end], splits = pv_splits[end];
    if (tile >= splits) return;
    if (mode < 2) {
        if (!tile && threadIdx.x < VALUE_DIMS) {
            const int col = first_col + threadIdx.x, parts = mode == 1 ? 2 : 4;
            const int chunk = (end + parts - 1) / parts;
            float accum[4] = {0.f};
            for (int part = 0; part < parts; ++part) {
                for (int key = part * chunk; key < min(end, (part + 1) * chunk); ++key) {
                    bool keep = visible(key, row, end, qsa, sparse, counts, ids, id_stride);
                    float p = __half2float(probabilities[int64_t(query) * probability_stride + key]);
                    float v = keep ? __half2float(tf_sm70::decoded_half(values, scales, bits, codec,
                                (int64_t(key) * kvheads + kvhead) * DIM + col)) : 0.f;
                    accum[part] = __fadd_rn(accum[part], __fmul_rn(p, v));
                }
            }
            float result = accum[0];
            for (int part = 1; part < parts; ++part) result = __fadd_rn(result, accum[part]);
            partials[int64_t(query) * partial_stride * DIM + col] = result;
        }
        return;
    }
    const int chunk = pv_chunk(end, splits);
    const int first = tile * chunk, last = min(end, first + chunk);
    const int residual_tile = mode == 2 ? 64 : 32;
    const int residue = (last - first) % residual_tile;
    const int warp = threadIdx.x >> 5;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(c, 0.f);
    // The partial prologue and full tiles share the same FP32 accumulator.
    // Each phase has its own bounds; no tail value leaks into the next phase.
    for (int phase = 0; phase < 2; ++phase) {
        const int low = phase ? first + residue : first;
        const int high = phase ? last : first + residue;
        for (int first_key = low; first_key < high; first_key += 16) {
            for (int at = threadIdx.x; at < 16 * 16; at += blockDim.x) {
                const int key = first_key + at % 16;
                const bool keep = at < 16 && key < high && visible(key, row, end, qsa, sparse, counts, ids, id_stride);
                probability_tile[at] = keep ? probabilities[int64_t(query) * probability_stride + key] : __float2half_rn(0.f);
            }
            for (int at = threadIdx.x; at < 16 * VALUE_DIMS; at += blockDim.x) {
                const int key = first_key + at / VALUE_DIMS, col = at % VALUE_DIMS;
                const bool keep = key < high && visible(key, row, end, qsa, sparse, counts, ids, id_stride);
                value_tile[at] = keep ? tf_sm70::decoded_half(values, scales, bits, codec,
                    (int64_t(key) * kvheads + kvhead) * DIM + first_col + col) : __float2half_rn(0.f);
            }
            __syncthreads();
            wmma::load_matrix_sync(a, probability_tile, 16);
            wmma::load_matrix_sync(b, value_tile + warp * 16, VALUE_DIMS);
            wmma::mma_sync(c, a, b, c);
            __syncthreads();
        }
    }
    wmma::store_matrix_sync(output + warp * 16, c, VALUE_DIMS, wmma::mem_row_major);
    __syncthreads();
    if (threadIdx.x < VALUE_DIMS)
        partials[(int64_t(query) * partial_stride + tile) * DIM + first_col + threadIdx.x] = output[threadIdx.x];
}

template<int DIM>
__global__ void reduce_kernel(const float* partials, __half* out, const void* position, bool pos_long,
                              int partial_stride, int heads, int context, int bits, int codec) {
    __shared__ float rotor[DIM];
    const int query = blockIdx.x, row = query / heads, col = threadIdx.x;
    const int end = min(context, base_position(position, pos_long) + row + 1);
    if (end < 1 || end > 8192) return;
    const int tiles = pv_splits[end];
    float value = 0.f;
    if (col < DIM) for (int tile = 0; tile < tiles; ++tile)
        value += partials[(int64_t(query) * partial_stride + tile) * DIM + col];
    if (col >= DIM) return;
    if (codec == 4) {
        value = tf_rq4::inverse_component(value, col, rotor);
    } else if (codec == 5) {
        value = tf_rq4::rotate_component<5, true>(value, col, rotor);
    } else if (codec == 9) {
        value = tf_rq4::rotate_component<9, true>(value, col, rotor);
    } else if (codec == 10) {
        value = tf_rq4::rotate_component<10, true>(value, col, rotor);
    } else if (bits) {
#pragma unroll
        for (int offset = 1; offset <= 16; offset *= 2) {
            float cross = __shfl_xor_sync(0xffffffffu, value, offset);
            value = (col & offset) ? cross - value : cross + value;
        }
        value *= 0.1767766952966369f;
    }
    out[int64_t(query) * DIM + col] = __float2half_rn(value);
}

template<typename V> V* ptr(const T& tensor) { return reinterpret_cast<V*>(tensor.data_ptr()); }
}  // namespace

void sm70_attention_cuda(const T& q, const T& k, const T& v, const T& position, const T& sparse,
                         const T& counts, const T& ids, T& scores, T& probabilities, T& partials, T& out,
                         const O& ks, const O& vs, int rows, int context, int kbits, int vbits, bool qsa, float scale,
                         int kcodec, int vcodec) {
    const int heads = q.size(1), kvheads = k.size(1), dim = q.size(2), stride = scores.size(2), pstride = partials.size(2);
    const bool pos_long = position.scalar_type() == at::kLong;
    auto stream = at::cuda::getCurrentCUDAStream();
    auto launch_scores = [&](auto kernel) {
        kernel<<<dim3((context + SCORE_TILE - 1) / SCORE_TILE, (rows * (heads / kvheads) + 15) / 16, kvheads), 128, 0, stream>>>(
            ptr<__half>(q), k.data_ptr(), ks ? ks->data_ptr() : nullptr, kbits, kcodec, position.data_ptr(), pos_long,
            ptr<int>(sparse), ptr<int>(counts), ptr<int>(ids), ids.size(1), qsa, ptr<float>(scores), stride,
            rows, heads, kvheads, context, scale);
    };
    auto launch_values = [&](auto kernel) {
        kernel<<<dim3(pstride, dim / VALUE_DIMS, rows * heads), 128, 0, stream>>>(
            ptr<__half>(probabilities), stride, v.data_ptr(), vs ? vs->data_ptr() : nullptr, vbits, vcodec,
            position.data_ptr(), pos_long, ptr<int>(sparse), ptr<int>(counts), ptr<int>(ids), ids.size(1), qsa,
            ptr<float>(partials), pstride, heads, kvheads, context);
    };
    auto launch_reduce = [&](auto kernel) {
        kernel<<<rows * heads, 256, 0, stream>>>(ptr<float>(partials), ptr<__half>(out), position.data_ptr(), pos_long,
            pstride, heads, context, vbits, vcodec);
    };
    if (rows) {
        launch_scores(scores_kernel<256>);
        probabilities_kernel<<<rows * heads, 256, 0, stream>>>(ptr<float>(scores), ptr<__half>(probabilities),
            position.data_ptr(), pos_long, stride, heads, context);
        launch_values(values_kernel<256>);
        launch_reduce(reduce_kernel<256>);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
