// Unique packed pages, bounded WMMA tiles, and globally normalized FP16 PV.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <mma.h>
#include <cstdint>
#include "packed_kv_decode.cuh"
#include "paged_attention_descriptor.cuh"
#include "paged_attention_device.cuh"

using T = at::Tensor;

namespace {
constexpr int HEADS = 12, PAD_HEADS = 16, DIM = 256, TILE = 32, THREADS = 128, WORDS = tf_sm70_paged::WORD_COUNT;
namespace wmma = nvcuda::wmma;
namespace meta = tf_sm70_paged;

using namespace tf_sm70_paged_device;

struct __align__(32) Storage {
    __half query[PAD_HEADS * DIM];
    union {
        __half kv[TILE * DIM];
        float output[PAD_HEADS * DIM];
    } tile;
    float scores[PAD_HEADS * TILE];
    __half probability[PAD_HEADS * TILE];
    int64_t physical[TILE];
};

__device__ __forceinline__ void load_queries(Storage& storage, const __half* queries,
                                            int rows, int row, int group,
                                            const int64_t* descriptor, int layer) {
    for (int item = threadIdx.x; item < PAD_HEADS * DIM; item += blockDim.x) {
        const int head = item / DIM, col = item % DIM;
        __half value = __float2half_rn(0.f);
        if (head < HEADS) {
            const int peer = group + head / 3, local_head = head % 3;
            value = queries[((int64_t(peer) * rows + row) * 3 + local_head) * DIM + col];
            if (!isfinite(__half2float(value))) {
                latch(descriptor, 1, layer);
                value = __float2half_rn(0.f);
            }
        }
        storage.query[item] = value;
    }
    __syncthreads();
}

__device__ __forceinline__ bool load_keys(Storage& storage, const int64_t* descriptor,
                                         const int* ids, int64_t first, int count, int end,
                                         int page_tokens, int layer, int codec, int* invalid) {
    if (!threadIdx.x) *invalid = 0;
    __syncthreads();
    if (threadIdx.x < TILE) {
        const int64_t at = first + threadIdx.x;
        const int logical = at < count ? (ids == nullptr ? int(at) : ids[at]) : -1;
        const int64_t physical = local_row(descriptor, logical, end, page_tokens, layer);
        storage.physical[threadIdx.x] = physical;
        if (physical == -2) atomicExch(invalid, 1);
    }
    __syncthreads();
    if (*invalid) return false;  // Uniform CTA exit before any payload access.
    for (int item = threadIdx.x; item < TILE * DIM; item += blockDim.x) {
        const int key = item / DIM, col = item % DIM;
        const int64_t physical = storage.physical[key];
        storage.tile.kv[item] = physical < 0 ? __float2half_rn(0.f)
            : tf_sm70::decoded_half(pointer<const void>(descriptor[meta::K_DATA]), pointer<const void>(descriptor[meta::K_METADATA]),
                                   codec ? 4 : 8, codec, physical * DIM + col);
    }
    __syncthreads();
    return true;
}

__device__ __forceinline__ void qk(Storage& storage) {
    const int warp = threadIdx.x >> 5;
    if (warp < 2) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> left;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> right;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> product;
        wmma::fill_fragment(product, 0.f);
        #pragma unroll
        for (int col = 0; col < DIM; col += 16) {
            wmma::load_matrix_sync(left, storage.query + col, DIM);
            wmma::load_matrix_sync(right, storage.tile.kv + warp * 16 * DIM + col, DIM);
            wmma::mma_sync(product, left, right, product);
        }
        wmma::store_matrix_sync(storage.scores + warp * 16, product, TILE, wmma::mem_row_major);
    }
    __syncthreads();
}

__global__ void statistics_kernel(const __half* queries, const int* selections, const int64_t* table,
                                  float* output, int rows, int width, int group,
                                  int page_tokens, const int* layer_word, int codec, float scale) {
    __shared__ Storage storage;
    __shared__ int end, count, enabled, invalid;
    __shared__ const int* ids;
    const int row = blockIdx.x;
    const int layer = *layer_word;
    const int64_t* descriptor = table + int64_t(row) * WORDS;
    for (int item = threadIdx.x; item < HEADS * 2; item += blockDim.x)
        output[int64_t(row) * HEADS * 2 + item] = item % 2 ? 0.f : -CUDART_INF_F;
    if (!threadIdx.x) {
        storage.tile.kv[0] = __float2half_rn(0.f);
        enabled = domain(descriptor, selections, rows, width, row, page_tokens, layer, end, count, ids);
    }
    __syncthreads();
    if (!enabled) return;
    load_queries(storage, queries, rows, row, group, descriptor, layer);
    float maximum = -CUDART_INF_F, denominator = 0.f;
    for (int64_t first = 0; first < count; first += TILE) {
        if (!load_keys(storage, descriptor, ids, first, count, end, page_tokens, layer, codec, &invalid)) return;
        qk(storage);
        if (threadIdx.x < HEADS) {
            float tile_max = -CUDART_INF_F;
            #pragma unroll
            for (int key = 0; key < TILE; ++key)
                if (storage.physical[key] >= 0)
                    tile_max = fmaxf(tile_max, __fmul_rn(storage.scores[threadIdx.x * TILE + key], scale));
            const float next_max = fmaxf(maximum, tile_max);
            if (isfinite(next_max)) {
                const float factor = isfinite(maximum) ? expf(__fsub_rn(maximum, next_max)) : 0.f;
                float mass = 0.f;
                #pragma unroll
                for (int key = 0; key < TILE; ++key)
                    if (storage.physical[key] >= 0)
                        mass = __fadd_rn(mass, expf(__fsub_rn(
                            __fmul_rn(storage.scores[threadIdx.x * TILE + key], scale), next_max)));
                denominator = __fadd_rn(__fmul_rn(denominator, factor), mass);
            }
            maximum = next_max;
        }
        __syncthreads();
    }
    if (threadIdx.x < HEADS) {
        output[(int64_t(row) * HEADS + threadIdx.x) * 2] = maximum;
        output[(int64_t(row) * HEADS + threadIdx.x) * 2 + 1] = denominator;
    }
}

__global__ void merge_kernel(const float* gathered, const int64_t* table, float* output,
                             int rows, int group, const int* layer_word) {
    __shared__ int enabled;
    const int row = blockIdx.x, head = threadIdx.x;
    const int layer = *layer_word;
    const int64_t* descriptor = table + int64_t(row) * WORDS;
    if (!head) enabled = active(descriptor, layer);
    __syncthreads();
    if (head >= HEADS) return;
    float maximum = -CUDART_INF_F, denominator = 0.f;
    int peers[4];
    bool valid = !enabled || merge_order(descriptor, layer, peers);
    if (enabled && valid) {
        #pragma unroll
        for (int peer = 0; peer < 4; ++peer) {
            const int64_t at = ((int64_t(group + peer) * rows + row) * HEADS + head) * 2;
            const float m = gathered[at], l = gathered[at + 1];
            valid = valid && ((isfinite(m) && isfinite(l) && l > 0.f)
                               || (m == -CUDART_INF_F && l == 0.f));
            maximum = fmaxf(maximum, m);
        }
        if (isfinite(maximum)) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int64_t at = ((int64_t(group + peers[j]) * rows + row) * HEADS + head) * 2;
                const float m = gathered[at], l = gathered[at + 1];
                if (isfinite(m)) denominator = __fadd_rn(denominator, __fmul_rn(expf(__fsub_rn(m, maximum)), l));
            }
        }
        valid = valid && isfinite(maximum) && isfinite(denominator) && denominator > 0.f;
        if (!valid) latch(descriptor, 16, layer);
    }
    // Empty padded rows are not a model error; downstream PV stays zero.
    output[(int64_t(row) * HEADS + head) * 2] = enabled && valid ? maximum : 0.f;
    output[(int64_t(row) * HEADS + head) * 2 + 1] = enabled && valid ? denominator : 1.f;
}

__global__ void values_kernel(const __half* queries, const int* selections, const int64_t* table,
                              const float* global, float* output, int rows, int width, int group,
                              int page_tokens, const int* layer_word, int key_codec, int value_codec, float scale) {
    __shared__ Storage storage;
    __shared__ int end, count, enabled, invalid;
    __shared__ const int* ids;
    const int row = blockIdx.x, warp = threadIdx.x >> 5;
    const int layer = *layer_word;
    const int64_t* descriptor = table + int64_t(row) * WORDS;
    for (int item = threadIdx.x; item < HEADS * DIM; item += blockDim.x)
        output[int64_t(row) * HEADS * DIM + item] = 0.f;
    if (!threadIdx.x) {
        storage.tile.kv[0] = __float2half_rn(0.f);
        enabled = domain(descriptor, selections, rows, width, row, page_tokens, layer, end, count, ids);
    }
    __syncthreads();
    if (!enabled) return;
    load_queries(storage, queries, rows, row, group, descriptor, layer);
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> probability;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> value;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulators[4];
    #pragma unroll
    for (int part = 0; part < 4; ++part) wmma::fill_fragment(accumulators[part], 0.f);
    for (int64_t first = 0; first < count; first += TILE) {
        if (!load_keys(storage, descriptor, ids, first, count, end, page_tokens, layer, key_codec, &invalid)) return;
        qk(storage);
        for (int item = threadIdx.x; item < PAD_HEADS * TILE; item += blockDim.x) {
            const int head = item / TILE, key = item % TILE;
            float p = 0.f;
            if (head < HEADS && storage.physical[key] >= 0) {
                const float maximum = global[(int64_t(row) * HEADS + head) * 2];
                const float denominator = global[(int64_t(row) * HEADS + head) * 2 + 1];
                if (isfinite(maximum) && isfinite(denominator) && denominator > 0.f)
                    p = __fdiv_rn(expf(__fsub_rn(__fmul_rn(storage.scores[item], scale), maximum)), denominator);
                else latch(descriptor, 16, layer);
            }
            storage.probability[item] = __float2half_rn(p);
        }
        __syncthreads();
        for (int item = threadIdx.x; item < TILE * DIM; item += blockDim.x) {
            const int key = item / DIM, col = item % DIM;
            const int64_t physical = storage.physical[key];
            storage.tile.kv[item] = physical < 0 ? __float2half_rn(0.f)
                : tf_sm70::decoded_half(pointer<const void>(descriptor[meta::V_DATA]), pointer<const void>(descriptor[meta::V_METADATA]),
                                       value_codec ? 4 : 8, value_codec, physical * DIM + col);
        }
        __syncthreads();
        #pragma unroll
        for (int key = 0; key < TILE; key += 16) {
            wmma::load_matrix_sync(probability, storage.probability + key, TILE);
            #pragma unroll
            for (int part = 0; part < 4; ++part) {
                wmma::load_matrix_sync(value, storage.tile.kv + key * DIM + warp * 64 + part * 16, DIM);
                wmma::mma_sync(accumulators[part], probability, value, accumulators[part]);
            }
        }
        __syncthreads();
    }
    // The key/value tile is retired; reuse its exact bytes to spill all16heads,
    // then publish only the12 real heads. No padded-head store reaches output.
    // This assignment begins the trivial union member's array lifetime before
    // the WMMA intrinsic writes through its pointer. Every reader retired above.
    if (!threadIdx.x) storage.tile.output[0] = 0.f;
    __syncthreads();
    #pragma unroll
    for (int part = 0; part < 4; ++part)
        wmma::store_matrix_sync(storage.tile.output + warp * 64 + part * 16,
                                accumulators[part], DIM, wmma::mem_row_major);
    __syncthreads();
    for (int item = threadIdx.x; item < HEADS * DIM; item += blockDim.x)
        output[int64_t(row) * HEADS * DIM + item] = storage.tile.output[item];
}

__global__ void merge_values_kernel(const float* gathered, const int64_t* table, __half* output,
                                    int rows, int rank, const int* layer_word, int codec) {
    __shared__ float rotated[DIM];
    __shared__ int enabled;
    const int row = blockIdx.x, local_head = blockIdx.y, col = threadIdx.x;
    const int layer = *layer_word;
    const int64_t* descriptor = table + int64_t(row) * WORDS;
    if (!col) enabled = active(descriptor, layer);
    __syncthreads();
    const int64_t destination = (int64_t(row) * 3 + local_head) * DIM + col;
    if (!enabled) {
        output[destination] = __float2half_rn(0.f);
        return;
    }
    int peers[4];
    if (!merge_order(descriptor, layer, peers)) {  // latched; status forbids publication
        output[destination] = __float2half_rn(0.f);
        return;
    }
    const int group = rank / 4 * 4, head = rank % 4 * 3 + local_head;
    float value = gathered[((int64_t(group + peers[0]) * rows + row) * HEADS + head) * DIM + col];
    #pragma unroll
    for (int j = 1; j < 4; ++j)
        value = __fadd_rn(value, gathered[((int64_t(group + peers[j]) * rows + row) * HEADS + head) * DIM + col]);
    if (codec == 9) {
        value = tf_rq4::rotate_component<9, true>(value, col, rotated);
    } else if (codec == 10) {
        value = tf_rq4::rotate_component<10, true>(value, col, rotated);
    } else {
        #pragma unroll
        for (int offset = 1; offset <= 16; offset *= 2) {
            const float cross = __shfl_xor_sync(0xffffffffu, value, offset);
            value = col & offset ? __fsub_rn(cross, value) : __fadd_rn(cross, value);
        }
        value = __fmul_rn(value, 0.1767766952966369f);
    }
    const __half result = __float2half_rn(value);
    if (!isfinite(value) || !isfinite(__half2float(result))) latch(descriptor, 4, layer);
    output[destination] = result;
}

// Distinct experimental RN/ex2 numerical family. The incumbent WMMA kernels
// above remain independent controls; no contraction/reduction order is shared
// by implication. Constants/ELF/PTX and analytical oracle require qualification.
constexpr int RN_MAX_SELECTED = 2051;

__device__ __forceinline__ bool rn_domain(const int64_t* descriptor, const int* selections,
        int rows, int width, int row, int page_tokens, int layer, int& end, int& count, const int*& ids) {
    if (!domain(descriptor, selections, rows, width, row, page_tokens, layer, end, count, ids)) return false;
    if (count > RN_MAX_SELECTED) {
        latch(descriptor, 32, layer);
        return false;  // Never truncate or address outside the fixed Half-P pitch.
    }
    return true;
}

__device__ __forceinline__ float rn_warp_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, offset));
    return value;  // Only lane zero's full tree is consumed.
}

// Each head has one FULL warp: lane l consumes l+32*t, t=0..7 in
// increasing order. K decode is cooperatively reused across H heads; changing
// H never changes an individual dot's eight FMAs/five RN shuffle additions.
//
// Key tiles are spread over RN_SPLITS CTAs per (row, head block). A key's score
// and a probability are per-key values; the order-dependent online (m, l)
// recurrence of the serial form (tile maxima, then per-tile masses summed in key
// order, then denominator = denominator * alpha + mass tile by tile) runs in
// rn_fold_kernel over the stored scores with the same operations in the same
// order. A descriptor fault (-2) in tile t stopped the serial kernel before tile
// t: every CTA finds the first faulting tile among the tiles up to its own and
// never decodes, scores or publishes from it or any later tile.
constexpr int RN_TILES = (RN_MAX_SELECTED + TILE - 1) / TILE, RN_SPLITS = 16;

template<int H> struct RNStorage {
    __half query[H * DIM];
    __half key[TILE * DIM];
    int64_t physical[TILE];
};

// First tile (< limit) holding a descriptor fault, else limit; local_row latches the fault.
__device__ __forceinline__ int rn_first_fault(const int64_t* descriptor, const int* ids, int count, int end,
                                             int page_tokens, int layer, int limit, int* shared) {
    if (!threadIdx.x) *shared = limit;
    __syncthreads();
    for (int at = threadIdx.x; at < min(count, limit * TILE); at += blockDim.x) {
        if (local_row(descriptor, ids == nullptr ? at : ids[at], end, page_tokens, layer) == -2)
            atomicMin(shared, at / TILE);
    }
    __syncthreads();
    return *shared;
}

// One tile's 32 scores per head (warp) into scores[warp * TILE + key]; NaN marks keys without a local row.
template<int H>
__device__ __forceinline__ void rn_tile_scores(RNStorage<H>& storage, const int64_t* descriptor, const int* ids,
        int first, int count, int end, int page_tokens, int layer, int codec, float* scores) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (threadIdx.x < TILE) {
        const int at = first + threadIdx.x;
        const int logical = at < count ? (ids == nullptr ? at : ids[at]) : -1;
        storage.physical[threadIdx.x] = local_row(descriptor, logical, end, page_tokens, layer);
    }
    // A tile without a local key (another owner's pages) has only NaN scores: no K decode or dots.
    if (!__syncthreads_or(threadIdx.x < TILE && storage.physical[threadIdx.x] >= 0)) {
        scores[threadIdx.x & 31] = CUDART_NAN_F;  // each warp its own head's row (TILE == 32 keys)
        __syncthreads();
        return;
    }
    for (int at = threadIdx.x; at < TILE * DIM; at += H * 32) {
        const int64_t physical = storage.physical[at / DIM];
        __half value = __float2half_rn(0.f);
        if (physical >= 0) {
            value = tf_sm70::decoded_half(pointer<const void>(descriptor[meta::K_DATA]),
                pointer<const void>(descriptor[meta::K_METADATA]), codec ? 4 : 8, codec,
                physical * DIM + at % DIM);
            if (!isfinite(__half2float(value))) {
                latch(descriptor, 2, layer);
                value = __float2half_rn(0.f);
            }
        }
        storage.key[at] = value;
    }
    __syncthreads();
    #pragma unroll 4
    for (int key = 0; key < TILE; ++key) {
        float value = 0.f;
        const bool valid = storage.physical[key] >= 0;
        if (valid) {
            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                const int col = lane + t * 32;
                value = __fmaf_rn(__half2float(storage.query[warp * DIM + col]),
                                 __half2float(storage.key[key * DIM + col]), value);
            }
            value = rn_warp_sum(value);
        }
        if (!lane) scores[key] = valid ? __fmul_rn(value, .0625f) : CUDART_NAN_F;
    }
    __syncthreads();  // storage.key/physical are reused by the next tile
}

template<int H>
__device__ __forceinline__ void rn_load_query(RNStorage<H>& storage, const __half* queries, int rows, int row,
                                              int group, int head_base, const int64_t* descriptor, int layer) {
    for (int at = threadIdx.x; at < H * DIM; at += H * 32) {
        const int h = head_base + at / DIM, col = at % DIM;
        __half value = queries[((int64_t(group + h / 3) * rows + row) * 3 + h % 3) * DIM + col];
        if (!isfinite(__half2float(value))) {
            latch(descriptor, 1, layer);
            value = __float2half_rn(0.f);
        }
        storage.query[at] = value;
    }
    __syncthreads();
}

// Statistics pass, part 1: scores of this CTA's tiles into scores[row][head][key].
template<int H>
__global__ void rn_scores_kernel(const __half* queries, const int* selections, const int64_t* table,
        float* scores, int rows, int width, int group, int page_tokens, const int* layer_word, int codec) {
    __shared__ RNStorage<H> storage;
    __shared__ int end, count, enabled, fault;
    __shared__ const int* ids;
    const int row = blockIdx.x, head_base = blockIdx.y * H, split = blockIdx.z;
    const int warp = threadIdx.x >> 5;
    const int layer = *layer_word;
    const int64_t* descriptor = table + int64_t(row) * WORDS;
    if (!threadIdx.x)
        enabled = rn_domain(descriptor, selections, rows, width, row, page_tokens, layer, end, count, ids);
    __syncthreads();
    if (!enabled) return;
    rn_load_query(storage, queries, rows, row, group, head_base, descriptor, layer);  // latches as the serial form
    const int tiles = (count + TILE - 1) / TILE;
    if (split >= tiles) return;
    const int last = split + (tiles - 1 - split) / RN_SPLITS * RN_SPLITS;
    const int stop = rn_first_fault(descriptor, ids, count, end, page_tokens, layer, last + 1, &fault);
    if (split >= stop) return;
    float* head_scores = scores + (int64_t(row) * HEADS + head_base + warp) * RN_TILES * TILE;
    for (int tile = split; tile < stop; tile += RN_SPLITS)
        rn_tile_scores(storage, descriptor, ids, tile * TILE, count, end, page_tokens, layer, codec,
                       head_scores + tile * TILE);
}

// Statistics pass, part 2: one warp per head folds the stored scores in the serial kernel's order.
__global__ void rn_fold_kernel(const int* selections, const int64_t* table, const float* scores, float* statistics,
        int rows, int width, int page_tokens, const int* layer_word) {
    __shared__ int end, count, enabled, fault;
    __shared__ const int* ids;
    __shared__ float tile_mass[HEADS][RN_TILES], tile_alpha[HEADS][RN_TILES], tile_next[HEADS][RN_TILES];
    const int row = blockIdx.x, head = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int layer = *layer_word;
    const int64_t* descriptor = table + int64_t(row) * WORDS;
    float* out = statistics + (int64_t(row) * HEADS + head) * 2;
    if (!threadIdx.x)
        enabled = rn_domain(descriptor, selections, rows, width, row, page_tokens, layer, end, count, ids);
    __syncthreads();
    const int tiles = enabled ? (count + TILE - 1) / TILE : 0;
    const int stop = enabled ? rn_first_fault(descriptor, ids, count, end, page_tokens, layer, tiles, &fault) : 0;
    if (!enabled || stop < tiles) {  // the serial kernel's initial values (no tile ran, or it returned at a fault)
        if (!lane) { out[0] = -CUDART_INF_F; out[1] = 0.f; }
        return;
    }
    const float* head_scores = scores + (int64_t(row) * HEADS + head) * RN_TILES * TILE;
    float carry = -CUDART_INF_F;  // running maximum before this group of 32 tiles
    for (int base = 0; base < tiles; base += 32) {
        const int tile = base + lane;
        float tile_max = -CUDART_INF_F;
        if (tile < tiles) {
            for (int key = 0; key < TILE; ++key) {
                const float s = head_scores[tile * TILE + key];
                if (!isnan(s)) tile_max = fmaxf(tile_max, s);
            }
        }
        float next = fmaxf(carry, tile_max);  // inclusive max scan: maxima are exact in any order
        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            const float other = __shfl_up_sync(0xffffffffu, next, offset);
            if (lane >= offset) next = fmaxf(next, other);
        }
        float before = __shfl_up_sync(0xffffffffu, next, 1);
        if (!lane) before = carry;
        if (tile < tiles) {
            float mass = 0.f, alpha = 0.f;
            if (isfinite(next)) {
                alpha = isfinite(before) ? rn_exp(__fsub_rn(before, next), descriptor, layer) : 0.f;
                for (int key = 0; key < TILE; ++key) {
                    const float s = head_scores[tile * TILE + key];
                    if (!isnan(s)) mass = __fadd_rn(mass, rn_exp(__fsub_rn(s, next), descriptor, layer));
                }
            }
            tile_mass[head][tile] = mass;
            tile_alpha[head][tile] = alpha;
            tile_next[head][tile] = next;
        }
        carry = __shfl_sync(0xffffffffu, next, 31);
    }
    __syncwarp();
    if (!lane) {
        float maximum = -CUDART_INF_F, denominator = 0.f;
        for (int tile = 0; tile < tiles; ++tile) {
            const float next = tile_next[head][tile];
            if (isfinite(next)) denominator = __fadd_rn(__fmul_rn(denominator, tile_alpha[head][tile]), tile_mass[head][tile]);
            maximum = next;
        }
        out[0] = maximum;
        out[1] = denominator;
    }
}

// Probability pass: each CTA zeroes and then publishes its own tiles' Half P.
template<int H>
__global__ void rn_probability_kernel(const __half* queries, const int* selections, const int64_t* table,
        const float* global, __half* probability, int rows, int width, int group, int page_tokens,
        const int* layer_word, int codec) {
    __shared__ RNStorage<H> storage;
    __shared__ float scores[H * TILE];
    __shared__ int end, count, enabled, fault;
    __shared__ const int* ids;
    const int row = blockIdx.x, head_base = blockIdx.y * H, split = blockIdx.z;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, head = head_base + warp;
    const int layer = *layer_word;
    const int64_t* descriptor = table + int64_t(row) * WORDS;
    for (int tile = split; tile < RN_TILES; tile += RN_SPLITS)
        for (int at = threadIdx.x; at < H * TILE; at += H * 32) {
            const int key = tile * TILE + at % TILE;
            if (key < RN_MAX_SELECTED)
                probability[(int64_t(row) * HEADS + head_base + at / TILE) * RN_MAX_SELECTED + key] = __float2half_rn(0.f);
        }
    if (!threadIdx.x)
        enabled = rn_domain(descriptor, selections, rows, width, row, page_tokens, layer, end, count, ids);
    __syncthreads();
    if (!enabled) return;
    rn_load_query(storage, queries, rows, row, group, head_base, descriptor, layer);  // latches as the serial form
    const int tiles = (count + TILE - 1) / TILE;
    if (split >= tiles) return;
    const int last = split + (tiles - 1 - split) / RN_SPLITS * RN_SPLITS;
    const int stop = rn_first_fault(descriptor, ids, count, end, page_tokens, layer, last + 1, &fault);
    if (split >= stop) return;
    const float m = global[(int64_t(row) * HEADS + head) * 2];
    const float l = global[(int64_t(row) * HEADS + head) * 2 + 1];
    const bool usable = isfinite(m) && isfinite(l) && l > 0.f;
    for (int tile = split; tile < stop; tile += RN_SPLITS) {
        rn_tile_scores(storage, descriptor, ids, tile * TILE, count, end, page_tokens, layer, codec,
                       scores + warp * TILE);
        if (!usable) {
            if (!lane) latch(descriptor, 16, layer);
        } else {
            const int key = tile * TILE + lane;
            const float s = scores[warp * TILE + lane];
            if (key < count && !isnan(s)) {
                const float p = __fdiv_rn(rn_exp(__fsub_rn(s, m), descriptor, layer), l);
                const __half result = __float2half_rn(p);
                if (!isfinite(p) || !isfinite(__half2float(result))) latch(descriptor, 16, layer);
                probability[(int64_t(row) * HEADS + head) * RN_MAX_SELECTED + key] = result;
            }
        }
    }
}

__global__ void rn_merge_kernel(const float* gathered, const int64_t* table, float* output,
                                int rows, int group, const int* layer_word) {
    const int row = blockIdx.x, head = threadIdx.x;
    const int layer = *layer_word;
    const int64_t* descriptor = table + int64_t(row) * WORDS;
    __shared__ int enabled;
    if (!head) enabled = active(descriptor, layer);
    __syncthreads();
    if (head >= HEADS) return;
    float maximum = -CUDART_INF_F, denominator = 0.f;
    int peers[4];
    bool valid = !enabled || merge_order(descriptor, layer, peers);
    if (enabled && valid) {
        #pragma unroll
        for (int peer = 0; peer < 4; ++peer) {
            const int64_t at = ((int64_t(group + peer) * rows + row) * HEADS + head) * 2;
            const float m = gathered[at], l = gathered[at + 1];
            valid = valid && ((isfinite(m) && isfinite(l) && l > 0.f) || (m == -CUDART_INF_F && l == 0.f));
            maximum = fmaxf(maximum, m);
        }
        if (isfinite(maximum)) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int64_t at = ((int64_t(group + peers[j]) * rows + row) * HEADS + head) * 2;
                if (isfinite(gathered[at]))
                    denominator = __fadd_rn(denominator, __fmul_rn(rn_exp(__fsub_rn(gathered[at], maximum), descriptor, layer), gathered[at + 1]));
            }
        }
        valid = valid && isfinite(maximum) && isfinite(denominator) && denominator > 0.f;
        if (!valid) latch(descriptor, 16, layer);
    }
    output[(int64_t(row) * HEADS + head) * 2] = enabled && valid ? maximum : 0.f;
    output[(int64_t(row) * HEADS + head) * 2 + 1] = enabled && valid ? denominator : 1.f;
}

// Four full warps, 32 output columns/block/head: warp w owns base+w+4*t,
// t=0..7. Every thread keeps exactly eight independent FP32 accumulators.
// Lane l visits selected i=l+32*t monotonically; lane0 publishes after five
// explicit RN shuffle additions. P is globally normalized Half, not local-P.
// All of a row's local rows and P faults are resolved up front by the whole
// block (one round of dependent loads, not one per key tile); then keys go 32
// at a time: the block decodes the tile's 32 x 32 V values into shared memory
// (each thread 8 consecutive columns of one key) and lane l of every warp
// consumes its key from there. Tiles without a local key (another owner's
// pages) are skipped. Per lane the FMA sequence, and so every output bit and
// status bit, is the per-key form's.
constexpr int RN_PV_COLUMNS = 32;
__global__ void rn_pv_kernel(const __half* probability, const int* selections, const int64_t* table,
        float* output, int rows, int width, int page_tokens, const int* layer_word, int codec) {
    constexpr int PARTS = RN_PV_COLUMNS / 4;
    const int row = blockIdx.x, head = blockIdx.y, column_base = blockIdx.z * RN_PV_COLUMNS;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int layer = *layer_word;
    const int64_t* descriptor = table + int64_t(row) * WORDS;
    __shared__ int enabled, end, count;
    __shared__ const int* ids;
    __shared__ int physical[RN_TILES * TILE];      // local row, or -1: foreign/masked/fault/bad P
    __shared__ float weight[RN_TILES * TILE];
    __shared__ int tile_live[RN_TILES];
    __shared__ __half tile[TILE][RN_PV_COLUMNS + 2];  // odd 17-word pitch: lane l's key row in its own bank
    if (!threadIdx.x) enabled = rn_domain(descriptor, selections, rows, width, row, page_tokens, layer, end, count, ids);
    __syncthreads();
    float result[PARTS];
    #pragma unroll
    for (int part = 0; part < PARTS; ++part) result[part] = 0.f;
    const __half* p_row = probability + (int64_t(row) * HEADS + head) * RN_MAX_SELECTED;
    if (enabled) {
        const int tiles = (count + TILE - 1) / TILE;
        for (int t = threadIdx.x; t < tiles; t += blockDim.x) tile_live[t] = 0;
        __syncthreads();
        for (int at = threadIdx.x; at < tiles * TILE; at += blockDim.x) {
            int64_t local = -1;
            float p = 0.f;
            if (at < count) {
                local = local_row(descriptor, ids == nullptr ? at : ids[at], end, page_tokens, layer);
                if (local >= 0) {  // Never read foreign/masked/error V payloads.
                    const __half half_p = p_row[at];
                    p = __half2float(half_p);
                    if (!isfinite(p) || p < 0.f) {
                        latch(descriptor, 16, layer);
                        local = -1;
                    }
                }
            }
            physical[at] = local >= 0 ? int(local) : -1;  // PHYSICAL_ROWS <= INT32_MAX (rn_domain)
            weight[at] = p;
            if (local >= 0) tile_live[at / TILE] = 1;
        }
        __syncthreads();
        const void* v_data = pointer<const void>(descriptor[meta::V_DATA]);
        const void* v_meta = pointer<const void>(descriptor[meta::V_METADATA]);
        for (int t = 0; t < tiles; ++t) {
            if (!tile_live[t]) continue;  // uniform: written before the barrier above
            const int base = t * TILE;
            const int key = threadIdx.x >> 2, first = (threadIdx.x & 3) * (RN_PV_COLUMNS / 4);
            const int source = physical[base + key];
            if (source >= 0) {
                #pragma unroll
                for (int c = 0; c < RN_PV_COLUMNS / 4; ++c) {
                    __half value = tf_sm70::decoded_half(v_data, v_meta, codec ? 4 : 8, codec,
                                                         int64_t(source) * DIM + column_base + first + c);
                    if (!isfinite(__half2float(value))) {
                        latch(descriptor, 4, layer);
                        value = __float2half_rn(0.f);
                    }
                    tile[key][first + c] = value;
                }
            }
            __syncthreads();
            if (physical[base + lane] >= 0) {
                const float p = weight[base + lane];
                #pragma unroll
                for (int part = 0; part < PARTS; ++part)
                    result[part] = __fmaf_rn(p, __half2float(tile[lane][warp + 4 * part]), result[part]);
            }
            __syncthreads();  // the next live tile rewrites tile[][]
        }
    }
    #pragma unroll
    for (int part = 0; part < PARTS; ++part) {
        const float value = rn_warp_sum(result[part]);
        if (!lane) {
            if (!isfinite(value)) latch(descriptor, 4, layer);
            output[(int64_t(row) * HEADS + head) * DIM + column_base + warp + 4 * part] = value;
        }
    }
}

// The ATen owner family retains the same descriptor, Q8 decoder and failure
// authority. Only its local normalized Half/LSE arithmetic is different.
__global__ void aten_pack_queries_kernel(const __half* input, const int64_t* table, __half* output,
                                         int rows, int group, const int* layer_word) {
    const int row = blockIdx.x, layer = *layer_word;
    const int64_t* descriptor = table + int64_t(row) * WORDS;
    const bool enabled = active(descriptor, layer);
    for (int at = threadIdx.x; at < HEADS * DIM; at += blockDim.x) {
        const int head = at / DIM, col = at % DIM;
        __half value = __float2half_rn(0.f);
        if (enabled) {
            value = input[((int64_t(group + head / 3) * rows + row) * 3 + head % 3) * DIM + col];
            if (!isfinite(__half2float(value))) {
                latch(descriptor, 1, layer);
                value = __float2half_rn(0.f); // Failed-row scratch; status forbids publication.
            }
        }
        output[int64_t(row) * HEADS * DIM + at] = value;
    }
}

__global__ void aten_gather_q8_kernel(const int* selections, const int64_t* table,
        __half* keys, __half* values, bool* valid, int rows, int width, int first_row,
        int key_capacity, int page_tokens, const int* layer_word) {
    constexpr int KEYS_PER_BLOCK = 8;
    const int panel_row = blockIdx.y, row = first_row + panel_row, layer = *layer_word;
    const int64_t* descriptor = table + int64_t(row) * WORDS;
    __shared__ int end, count, enabled;
    __shared__ const int* ids;
    if (!threadIdx.x) {
        enabled = domain(descriptor, selections, rows, width, row, page_tokens, layer, end, count, ids);
        if (enabled && count > key_capacity) {
            latch(descriptor, 32, layer);
            enabled = false;
        }
    }
    __syncthreads();
    const int tile = int(blockIdx.x) * KEYS_PER_BLOCK;
    for (int item = threadIdx.x; item < KEYS_PER_BLOCK * DIM; item += blockDim.x) {
        const int key = tile + item / DIM, col = item % DIM;
        if (key >= key_capacity) continue;
        const int logical = enabled && key < count ? (ids == nullptr ? key : ids[key]) : -1;
        const int64_t physical = enabled ? local_row(descriptor, logical, end, page_tokens, layer) : -1;
        __half k = __float2half_rn(0.f), v = __float2half_rn(0.f);
        if (physical >= 0) {
            k = tf_sm70::decoded_half(pointer<const void>(descriptor[meta::K_DATA]),
                pointer<const void>(descriptor[meta::K_METADATA]), 8, 0, physical * DIM + col);
            v = tf_sm70::decoded_half(pointer<const void>(descriptor[meta::V_DATA]),
                pointer<const void>(descriptor[meta::V_METADATA]), 8, 0, physical * DIM + col);
            if (!isfinite(__half2float(k))) { latch(descriptor, 2, layer); k = __float2half_rn(0.f); }
            if (!isfinite(__half2float(v))) { latch(descriptor, 4, layer); v = __float2half_rn(0.f); }
        }
        const int64_t at = (int64_t(panel_row) * key_capacity + key) * DIM + col;
        keys[at] = k;
        values[at] = v;
        if (!col) valid[int64_t(panel_row) * key_capacity + key] = physical >= 0;
    }
}

__global__ void aten_weight_values_kernel(const __half* normalized, const float* local_lse,
        const float* global, const int64_t* table, float* output, const int* layer_word) {
    const int row = blockIdx.x, head = blockIdx.y, col = threadIdx.x, layer = *layer_word;
    const int64_t* descriptor = table + int64_t(row) * WORDS;
    const int64_t at = int64_t(row) * HEADS + head;
    float value = 0.f;
    if (active(descriptor, layer)) {
        const float lse = local_lse[at];
        if (lse != -CUDART_INF_F) {
            const float maximum = global[at * 2], denominator = global[at * 2 + 1];
            const float weight = __fdiv_rn(expf(__fsub_rn(lse, maximum)), denominator);
            value = __fmul_rn(weight, __half2float(normalized[at * DIM + col]));
        }
        // Nonempty NaNs remain NaN through the ordinary final publication
        // guard. Only the explicit local-empty (-inf) identity skips the read.
    }
    output[at * DIM + col] = value;
}

}  // namespace

void sm70_paged_statistics_cuda(const T& queries, const T& selected, const T& table, T& output,
                                int group, int page_tokens, const T& layer, int codec, float scale) {
    const int rows = int(queries.size(1));
    if (!rows) return;
    statistics_kernel<<<rows, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(queries.data_ptr()), selected.data_ptr<int>(), table.data_ptr<int64_t>(),
        output.data_ptr<float>(), rows, int(selected.size(2)), group, page_tokens, layer.data_ptr<int>(), codec, scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sm70_paged_merge_cuda(const T& gathered, const T& table, T& output, int group, const T& layer) {
    const int rows = int(gathered.size(1));
    if (!rows) return;
    merge_kernel<<<rows, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        gathered.data_ptr<float>(), table.data_ptr<int64_t>(), output.data_ptr<float>(), rows, group, layer.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sm70_paged_values_cuda(const T& queries, const T& selected, const T& table, const T& global, T& output,
                           int group, int page_tokens, const T& layer, int key_codec, int value_codec, float scale) {
    const int rows = int(queries.size(1));
    if (!rows) return;
    values_kernel<<<rows, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(queries.data_ptr()), selected.data_ptr<int>(), table.data_ptr<int64_t>(),
        global.data_ptr<float>(), output.data_ptr<float>(), rows, int(selected.size(2)), group, page_tokens,
        layer.data_ptr<int>(), key_codec, value_codec, scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sm70_paged_merge_values_cuda(const T& gathered, const T& table, T& output, int rank, const T& layer, int codec) {
    const int rows = int(gathered.size(1));
    if (!rows) return;
    merge_values_kernel<<<dim3(rows, 3), DIM, 0, at::cuda::getCurrentCUDAStream()>>>(
        gathered.data_ptr<float>(), table.data_ptr<int64_t>(), reinterpret_cast<__half*>(output.data_ptr()),
        rows, rank, layer.data_ptr<int>(), codec);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template<int H, bool P>
void launch_rn_qk(const T& queries, const T& selected, const T& table, const T& global, T& output,
                  int group, int page_tokens, const T& layer, int codec) {
    const int rows = int(queries.size(1));
    if (!rows) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 grid(rows, HEADS / H, RN_SPLITS);
    const auto* q = reinterpret_cast<const __half*>(queries.data_ptr());
    if (P) {
        rn_probability_kernel<H><<<grid, H * 32, 0, stream>>>(q, selected.data_ptr<int>(), table.data_ptr<int64_t>(),
            global.data_ptr<float>(), reinterpret_cast<__half*>(output.data_ptr()), rows, int(selected.size(2)), group,
            page_tokens, layer.data_ptr<int>(), codec);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }
    // Stream-ordered scratch (a capture's private pool inside graphs): rows x heads x 2051-key scores.
    auto scores = at::empty({rows, HEADS, RN_TILES * TILE}, queries.options().dtype(at::kFloat));
    rn_scores_kernel<H><<<grid, H * 32, 0, stream>>>(q, selected.data_ptr<int>(), table.data_ptr<int64_t>(),
        scores.data_ptr<float>(), rows, int(selected.size(2)), group, page_tokens, layer.data_ptr<int>(), codec);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    rn_fold_kernel<<<rows, HEADS * 32, 0, stream>>>(selected.data_ptr<int>(), table.data_ptr<int64_t>(),
        scores.data_ptr<float>(), output.data_ptr<float>(), rows, int(selected.size(2)), page_tokens,
        layer.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sm70_paged_rn_qk_cuda(const T& queries, const T& selected, const T& table, const T& global, T& output,
                           int group, int page_tokens, const T& layer, int codec, int heads, bool probability) {
    #define RN_LAUNCH(H) if (heads == H) { if (probability) launch_rn_qk<H, true>(queries, selected, table, global, output, group, page_tokens, layer, codec); \
        else launch_rn_qk<H, false>(queries, selected, table, global, output, group, page_tokens, layer, codec); return; }
    RN_LAUNCH(1)
    RN_LAUNCH(4)
    RN_LAUNCH(12)
    #undef RN_LAUNCH
    TORCH_CHECK(false, "RN QK heads per block must be explicitly1,4,12");
}

void sm70_paged_rn_merge_cuda(const T& gathered, const T& table, T& output, int group, const T& layer) {
    const int rows = int(gathered.size(1));
    if (!rows) return;
    rn_merge_kernel<<<rows, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(gathered.data_ptr<float>(),
        table.data_ptr<int64_t>(), output.data_ptr<float>(), rows, group, layer.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sm70_paged_rn_pv_cuda(const T& probability, const T& selected, const T& table, T& output,
                          int rows, int page_tokens, const T& layer, int codec) {
    if (!rows) return;
    rn_pv_kernel<<<dim3(rows, HEADS, DIM / RN_PV_COLUMNS), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(probability.data_ptr()), selected.data_ptr<int>(), table.data_ptr<int64_t>(),
        output.data_ptr<float>(), rows, int(selected.size(2)), page_tokens, layer.data_ptr<int>(), codec);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sm70_paged_aten_pack_cuda(const T& queries, const T& table, T& output, int group, const T& layer) {
    const int rows = int(queries.size(1));
    if (!rows) return;
    aten_pack_queries_kernel<<<rows, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(queries.data_ptr()), table.data_ptr<int64_t>(),
        reinterpret_cast<__half*>(output.data_ptr()), rows, group, layer.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sm70_paged_aten_gather_cuda(const T& selected, const T& table, T& keys, T& values, T& valid,
        int first, int page_tokens, const T& layer) {
    const int count = int(keys.size(0)), capacity = int(keys.size(1));
    if (!count) return;
    aten_gather_q8_kernel<<<dim3((capacity + 7) / 8, count), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        selected.data_ptr<int>(), table.data_ptr<int64_t>(), reinterpret_cast<__half*>(keys.data_ptr()),
        reinterpret_cast<__half*>(values.data_ptr()), valid.data_ptr<bool>(), int(selected.size(1)),
        int(selected.size(2)), first, capacity, page_tokens, layer.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sm70_paged_aten_weight_cuda(const T& normalized, const T& local_lse, const T& global,
        const T& table, T& output, const T& layer) {
    const int rows = int(normalized.size(0));
    if (!rows) return;
    aten_weight_values_kernel<<<dim3(rows, HEADS), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(normalized.data_ptr()), local_lse.data_ptr<float>(),
        global.data_ptr<float>(), table.data_ptr<int64_t>(), output.data_ptr<float>(), layer.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
