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
                                   tf_sm70::codec_bits(codec), codec, physical * DIM + col);
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
                                       tf_sm70::codec_bits(value_codec), value_codec, physical * DIM + col);
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
    } else if (codec == 4 || tf_rq4::eight_bit(codec)) {
        value = tf_rq4::rotate_component<4, true>(value, col, rotated);  // signed Iso128 basis
    } else if (codec == 5) {
        value = tf_rq4::rotate_component<5, true>(value, col, rotated);
    } else if (codec == 1) {
        value = tf_rq4::rotate_component<1, true>(value, col, rotated);
    } else if (codec == 2) {
        value = tf_rq4::rotate_component<2, true>(value, col, rotated);
    } else if (codec == 3) {
        value = tf_rq4::rotate_component<3, true>(value, col, rotated);
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
// Key tiles are spread over gridDim.z CTAs per (row, head block). A key's score
// and a probability are per-key values; the order-dependent online (m, l)
// recurrence of the serial form (tile maxima, then per-tile masses summed in key
// order, then denominator = denominator * alpha + mass tile by tile) runs in
// rn_fold_kernel over the stored scores with the same operations in the same
// order. A descriptor fault (-2) in tile t stopped the serial kernel before tile
// t: every CTA finds the first faulting tile among the tiles up to its own and
// never decodes, scores or publishes from it or any later tile.
//
// Latency structure (no arithmetic change): page-offset loads of RN_SCAN_BATCH
// keys per thread are issued before any is evaluated; the fault scan records
// the local rows of the CTA's own tiles, so a tile needs no further page loads;
// K and V are read in packed runs (one payload and one scale load per run).
// The split count is the launch's gridDim.z (per-key values: any count gives the same bits). Few-row
// calls use RN_SPLITS_FEW for more CTAs; RN_SPLITS (the minimum) sizes a CTA's own-tile slab.
constexpr int RN_TILES = (RN_MAX_SELECTED + TILE - 1) / TILE, RN_SPLITS = 16, RN_SPLITS_FEW = 32, RN_FEW_ROWS = 8;
constexpr int RN_OWN_TILES = (RN_TILES + RN_SPLITS - 1) / RN_SPLITS;  // tiles of one (row, split) CTA
constexpr int RN_RUN = 16, RN_SCAN_BATCH = 4;

template<int H> struct __align__(16) RNStorage {
    __half query[H * DIM];
    __half key[TILE * DIM];
    int64_t physical[RN_OWN_TILES][TILE];  // own tiles' local rows (-1 foreign/masked/beyond count)
};

// RotorQuant centroid table (stage_centroids): dynamic shared memory, launched with RN_TABLE_BYTES only for
// RotorQuant codecs, so ordinary Q8 launches keep their static footprint and occupancy.
extern __shared__ float rn_centroid_table[];
constexpr size_t RN_TABLE_BYTES = 256 * sizeof(float);
__host__ __forceinline__ size_t rn_table_bytes(int codec) { return codec ? RN_TABLE_BYTES : 0; }

// First tile (< limit) holding a descriptor fault, else limit; local_row_from latches the fault.
// With `own`, also records the local rows of this CTA's tiles split, split + splits, ... (splits = gridDim.z) .
__device__ __forceinline__ int rn_first_fault(const int64_t* descriptor, const int* ids, int count, int end,
        int page_tokens, int layer, int limit, int* shared, int split = -1, int64_t (*own)[TILE] = nullptr) {
    if (!threadIdx.x) *shared = limit;
    if (own != nullptr)
        for (int at = threadIdx.x; at < RN_OWN_TILES * TILE; at += blockDim.x) own[at / TILE][at % TILE] = -1;
    __syncthreads();
    const int n = min(count, limit * TILE);
    for (int base = threadIdx.x; base < n; base += RN_SCAN_BATCH * blockDim.x) {
        int logical[RN_SCAN_BATCH];
        int64_t offset[RN_SCAN_BATCH];
        #pragma unroll
        for (int j = 0; j < RN_SCAN_BATCH; ++j) {
            const int at = base + j * blockDim.x;
            logical[j] = at < n ? (ids == nullptr ? at : ids[at]) : -1;
        }
        #pragma unroll
        for (int j = 0; j < RN_SCAN_BATCH; ++j) offset[j] = page_offset_word(descriptor, logical[j], end, page_tokens);
        #pragma unroll
        for (int j = 0; j < RN_SCAN_BATCH; ++j) {
            const int at = base + j * blockDim.x;
            if (at >= n) break;
            const int64_t local = local_row_from(descriptor, logical[j], end, page_tokens, layer, offset[j]);
            if (local == -2) atomicMin(shared, at / TILE);
            if (own != nullptr && at / TILE % int(gridDim.z) == split) own[at / TILE / int(gridDim.z)][at % TILE] = local;
        }
    }
    __syncthreads();
    return *shared;
}

// One tile's 32 scores per head (warp) into scores[warp * TILE + key]; NaN marks keys without a local row.
// `physical` holds the tile's 32 local rows. A CTA walks its tiles software-pipelined: the next live
// tile's packed K runs (one payload and one scale load each) are issued before the current tile's dots,
// then decoded with decoded_half's element expressions.
template<int H> struct RNTileRuns {
    static constexpr int RUNS_PER_KEY = DIM / RN_RUN, RUNS = TILE * RUNS_PER_KEY;
    static constexpr int ITEMS = (RUNS + H * 32 - 1) / (H * 32);
    uint4 raw[ITEMS];
    unsigned scale[ITEMS];
    // A tile without a local key (another owner's pages) has only NaN scores: no K decode or dots.
    __device__ __forceinline__ static bool live(const int64_t* physical) {
        return __syncthreads_or(threadIdx.x < TILE && physical[threadIdx.x] >= 0);
    }
    __device__ __forceinline__ void load(const int64_t* physical, const int64_t* descriptor, int codec) {
        const void* data = pointer<const void>(descriptor[meta::K_DATA]);
        const void* scales = pointer<const void>(descriptor[meta::K_METADATA]);
        #pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            const int task = threadIdx.x + i * H * 32;
            raw[i] = make_uint4(0u, 0u, 0u, 0u);
            scale[i] = 0u;
            if (task < RUNS && physical[task / RUNS_PER_KEY] >= 0)
                tf_sm70::load_run<RN_RUN>(data, scales, codec,
                    physical[task / RUNS_PER_KEY] * DIM + task % RUNS_PER_KEY * RN_RUN, raw[i], scale[i]);
        }
    }
    // Decoded K into storage.key; ends with a barrier.
    __device__ __forceinline__ void stage(RNStorage<H>& storage, const float* table, const int64_t* physical,
                                          const int64_t* descriptor,
                                          int layer, int codec) const {
        #pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            const int task = threadIdx.x + i * H * 32;
            if (task >= RUNS) continue;
            const int key = task / RUNS_PER_KEY, col = task % RUNS_PER_KEY * RN_RUN;
            const bool local = physical[key] >= 0;
            unsigned packed[RN_RUN / 2];
            #pragma unroll
            for (int c = 0; c < RN_RUN; ++c) {
                __half value = __float2half_rn(0.f);
                if (local) {
                    value = tf_sm70::decode_run_value<RN_RUN>(raw[i], scale[i], codec, c, table);
                    if (!isfinite(__half2float(value))) {
                        latch(descriptor, 2, layer);
                        value = __float2half_rn(0.f);
                    }
                }
                const unsigned bits = __half_as_ushort(value);
                packed[c / 2] = c % 2 ? packed[c / 2] | bits << 16 : bits;
            }
            #pragma unroll
            for (int v = 0; v < RN_RUN / 8; ++v)
                *reinterpret_cast<uint4*>(storage.key + key * DIM + col + 8 * v) =
                    make_uint4(packed[4 * v], packed[4 * v + 1], packed[4 * v + 2], packed[4 * v + 3]);
        }
        __syncthreads();
    }
};

// Dots of a staged tile: scores[key] per head (warp); ends with a barrier (storage.key is reused).
template<int H>
__device__ __forceinline__ void rn_tile_dots(const RNStorage<H>& storage, const int64_t* physical, float* scores) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    #pragma unroll 4
    for (int key = 0; key < TILE; ++key) {
        float value = 0.f;
        const bool valid = physical[key] >= 0;
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
    __syncthreads();
}

// Scores of the CTA's tiles split, split + splits, ... < stop: after each tile's barrier,
// visit(tile, scores) consumes it (uniformly called by every thread).
template<int H, typename Scores, typename Visit>
__device__ __forceinline__ void rn_own_tiles(RNStorage<H>& storage, const float* table, int split, int stop,
                                            const int64_t* descriptor,
                                            int layer, int codec, Scores scores_of, Visit visit) {
    RNTileRuns<H> runs;
    bool live = RNTileRuns<H>::live(storage.physical[0]);
    if (live) runs.load(storage.physical[0], descriptor, codec);
    for (int tile = split; tile < stop; tile += int(gridDim.z)) {
        const int64_t* physical = storage.physical[tile / int(gridDim.z)];
        float* scores = scores_of(tile);
        if (live) runs.stage(storage, table, physical, descriptor, layer, codec);
        const bool next = tile + int(gridDim.z) < stop && RNTileRuns<H>::live(storage.physical[tile / int(gridDim.z) + 1]);
        if (next) runs.load(storage.physical[tile / int(gridDim.z) + 1], descriptor, codec);
        if (live) rn_tile_dots(storage, physical, scores);
        else {
            scores[threadIdx.x & 31] = CUDART_NAN_F;  // each warp its own head's row (TILE == 32 keys)
            __syncthreads();
        }
        visit(tile, scores);
        live = next;
    }
}

template<int H>
__device__ __forceinline__ void rn_load_query(RNStorage<H>& storage, const __half* queries, int rows, int row,
                                              int group, int head_base, const int64_t* descriptor, int layer) {
    constexpr int ITEMS = DIM / 32;  // H * DIM values over H * 32 threads; all loads issue first
    __half value[ITEMS];
    #pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        const int at = threadIdx.x + i * H * 32;
        const int h = head_base + at / DIM, col = at % DIM;
        value[i] = queries[((int64_t(group + h / 3) * rows + row) * 3 + h % 3) * DIM + col];
    }
    #pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        if (!isfinite(__half2float(value[i]))) {
            latch(descriptor, 1, layer);
            value[i] = __float2half_rn(0.f);
        }
        storage.query[threadIdx.x + i * H * 32] = value[i];
    }
    __syncthreads();
}

// Statistics pass, part 1: scores of this CTA's tiles into scores[row][head][key].
// ROTOR false: ordinary Q8 only (codec folds to 0, so the Q8 build carries no RotorQuant decode branches).
template<int H, bool ROTOR>
__global__ void rn_scores_kernel(const __half* queries, const int* selections, const int64_t* table,
        float* scores, int rows, int width, int group, int page_tokens, const int* layer_word, int kv_codec) {
    const int codec = ROTOR ? kv_codec : 0;
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
    const float* centroids = tf_sm70::stage_centroids(codec, rn_centroid_table);
    rn_load_query(storage, queries, rows, row, group, head_base, descriptor, layer);  // latches as the serial form
    const int tiles = (count + TILE - 1) / TILE;
    if (split >= tiles) return;
    const int last = split + (tiles - 1 - split) / int(gridDim.z) * int(gridDim.z);
    const int stop = rn_first_fault(descriptor, ids, count, end, page_tokens, layer, last + 1, &fault,
                                    split, storage.physical);
    if (split >= stop) return;
    float* head_scores = scores + (int64_t(row) * HEADS + head_base + warp) * RN_TILES * TILE;
    rn_own_tiles(storage, centroids, split, stop, descriptor, layer, codec,
                 [&](int tile) { return head_scores + tile * TILE; }, [](int, const float*) {});
}

// Statistics pass, part 2: one warp per head folds the stored scores in the serial kernel's order.
// Lane l holds tile l's 32 scores in registers (eight 16-byte loads).
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
        float s[TILE];
        float tile_max = -CUDART_INF_F;
        if (tile < tiles) {
            const float4* source = reinterpret_cast<const float4*>(head_scores + tile * TILE);
            #pragma unroll
            for (int q = 0; q < TILE / 4; ++q) {
                const float4 v = source[q];
                s[4 * q] = v.x; s[4 * q + 1] = v.y; s[4 * q + 2] = v.z; s[4 * q + 3] = v.w;
            }
            #pragma unroll
            for (int key = 0; key < TILE; ++key)
                if (!isnan(s[key])) tile_max = fmaxf(tile_max, s[key]);
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
                #pragma unroll
                for (int key = 0; key < TILE; ++key)
                    if (!isnan(s[key])) mass = __fadd_rn(mass, rn_exp(__fsub_rn(s[key], next), descriptor, layer));
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
template<int H, bool ROTOR>
__global__ void rn_probability_kernel(const __half* queries, const int* selections, const int64_t* table,
        const float* global, __half* probability, int rows, int width, int group, int page_tokens,
        const int* layer_word, int kv_codec) {
    const int codec = ROTOR ? kv_codec : 0;
    __shared__ RNStorage<H> storage;
    __shared__ float scores[H * TILE];
    __shared__ int end, count, enabled, fault;
    __shared__ const int* ids;
    const int row = blockIdx.x, head_base = blockIdx.y * H, split = blockIdx.z;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, head = head_base + warp;
    const int layer = *layer_word;
    const int64_t* descriptor = table + int64_t(row) * WORDS;
    for (int tile = split; tile < RN_TILES; tile += int(gridDim.z))
        for (int at = threadIdx.x; at < H * TILE; at += H * 32) {
            const int key = tile * TILE + at % TILE;
            if (key < RN_MAX_SELECTED)
                probability[(int64_t(row) * HEADS + head_base + at / TILE) * RN_MAX_SELECTED + key] = __float2half_rn(0.f);
        }
    if (!threadIdx.x)
        enabled = rn_domain(descriptor, selections, rows, width, row, page_tokens, layer, end, count, ids);
    __syncthreads();
    if (!enabled) return;
    const float* centroids = tf_sm70::stage_centroids(codec, rn_centroid_table);
    rn_load_query(storage, queries, rows, row, group, head_base, descriptor, layer);  // latches as the serial form
    const int tiles = (count + TILE - 1) / TILE;
    if (split >= tiles) return;
    const int last = split + (tiles - 1 - split) / int(gridDim.z) * int(gridDim.z);
    const int stop = rn_first_fault(descriptor, ids, count, end, page_tokens, layer, last + 1, &fault,
                                    split, storage.physical);
    if (split >= stop) return;
    const float m = global[(int64_t(row) * HEADS + head) * 2];
    const float l = global[(int64_t(row) * HEADS + head) * 2 + 1];
    const bool usable = isfinite(m) && isfinite(l) && l > 0.f;
    rn_own_tiles(storage, centroids, split, stop, descriptor, layer, codec, [&](int) { return scores + warp * TILE; },
                 [&](int tile, const float* tile_scores) {
        if (!usable) {
            if (!lane) latch(descriptor, 16, layer);
        } else {
            const int key = tile * TILE + lane;
            const float s = tile_scores[lane];
            if (key < count && !isnan(s)) {
                const float p = __fdiv_rn(rn_exp(__fsub_rn(s, m), descriptor, layer), l);
                const __half result = __float2half_rn(p);
                if (!isfinite(p) || !isfinite(__half2float(result))) latch(descriptor, 16, layer);
                probability[(int64_t(row) * HEADS + head) * RN_MAX_SELECTED + key] = result;
            }
        }
    });
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

// PV: one CTA per (row, block of RN_PV_COLUMNS columns) for all 12 heads. Warp
// h is head h; lane l visits selected i=l+32*t monotonically with one FP32
// accumulator per column, and lane0 publishes each column after five explicit
// RN shuffle additions. P is globally normalized Half, not local-P. The row's
// local rows are resolved once per CTA (page-offset loads batched); the tiles
// holding a local key are listed in increasing order and processed
// RN_PV_GROUP at a time, double-buffered: the next group's packed V runs (one
// per thread, decoded once for all heads) and P values are loaded into
// registers while the current group's FMAs read shared memory. Per lane the
// FMA sequence, and so every output bit and status bit, is the per-key form's:
// a local key whose P is nonfinite or negative is skipped for that head
// (status 16); a nonfinite V value is latched (status 4) and consumed as zero
// by every head that uses the key. Foreign/masked/error V is never read.
constexpr int RN_PV_COLUMNS = 16, RN_PV_THREADS = HEADS * 32;
constexpr int RN_PV_RUNS = RN_PV_COLUMNS / RN_RUN;                   // packed runs per key and block
constexpr int RN_PV_GROUP = RN_PV_THREADS / (TILE * RN_PV_RUNS);     // tiles per group: one run per thread
constexpr int RN_PV_PITCH = RN_PV_COLUMNS + 2;  // odd word pitch: lane l's key row in its own bank
static_assert(RN_PV_COLUMNS % RN_RUN == 0 && DIM % RN_PV_COLUMNS == 0 && RN_PV_GROUP >= 1
              && RN_PV_THREADS % (TILE * RN_PV_RUNS) == 0 && RN_PV_PITCH / 2 % 2 == 1 && RN_TILES <= 255,
              "RN PV block geometry");
template<bool ROTOR>
__global__ void __launch_bounds__(RN_PV_THREADS, 2) rn_pv_kernel(const __half* probability, const int* selections,
        const int64_t* table, float* output, int rows, int width, int page_tokens, const int* layer_word, int kv_codec) {
    const int codec = ROTOR ? kv_codec : 0;
    const int row = blockIdx.x, column_base = blockIdx.y * RN_PV_COLUMNS;
    const int head = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int layer = *layer_word;
    const int64_t* descriptor = table + int64_t(row) * WORDS;
    __shared__ int enabled, end, count, live_count;
    __shared__ const int* ids;
    __shared__ int physical[RN_TILES * TILE];      // local row, or -1: foreign/masked/fault/beyond count
    __shared__ int tile_live[RN_TILES];
    __shared__ unsigned char live[RN_TILES];       // tiles with a local key, increasing
    __shared__ __align__(16) __half staged[2][RN_PV_GROUP][TILE][RN_PV_PITCH];
    if (!threadIdx.x) enabled = rn_domain(descriptor, selections, rows, width, row, page_tokens, layer, end, count, ids);
    __syncthreads();
    const float* centroid_table = tf_sm70::stage_centroids(codec, rn_centroid_table);
    float result[RN_PV_COLUMNS];
    #pragma unroll
    for (int c = 0; c < RN_PV_COLUMNS; ++c) result[c] = 0.f;
    const __half* p_row = probability + (int64_t(row) * HEADS + head) * RN_MAX_SELECTED;
    if (enabled) {
        const int tiles = (count + TILE - 1) / TILE, keys = tiles * TILE;
        for (int t = threadIdx.x; t < tiles; t += blockDim.x) tile_live[t] = 0;
        __syncthreads();
        for (int base = threadIdx.x; base < keys; base += RN_SCAN_BATCH * blockDim.x) {
            int logical[RN_SCAN_BATCH];
            int64_t offset[RN_SCAN_BATCH];
            #pragma unroll
            for (int j = 0; j < RN_SCAN_BATCH; ++j) {
                const int at = base + j * blockDim.x;
                logical[j] = at < count ? (ids == nullptr ? at : ids[at]) : -1;
            }
            #pragma unroll
            for (int j = 0; j < RN_SCAN_BATCH; ++j) offset[j] = page_offset_word(descriptor, logical[j], end, page_tokens);
            #pragma unroll
            for (int j = 0; j < RN_SCAN_BATCH; ++j) {
                const int at = base + j * blockDim.x;
                if (at >= keys) break;
                const int64_t local = local_row_from(descriptor, logical[j], end, page_tokens, layer, offset[j]);
                physical[at] = local >= 0 ? int(local) : -1;  // PHYSICAL_ROWS <= INT32_MAX (rn_domain)
                if (local >= 0) tile_live[at / TILE] = 1;
            }
        }
        __syncthreads();
        if (head == 0) {
            int n = 0;
            for (int base = 0; base < tiles; base += 32) {
                const bool flag = base + lane < tiles && tile_live[base + lane];
                const unsigned mask = __ballot_sync(0xffffffffu, flag);
                if (flag) live[n + __popc(mask & ((1u << lane) - 1u))] = (unsigned char)(base + lane);
                n += __popc(mask);
            }
            if (!lane) live_count = n;
        }
        __syncthreads();
        const int n_live = live_count, groups = (n_live + RN_PV_GROUP - 1) / RN_PV_GROUP;
        const void* v_data = pointer<const void>(descriptor[meta::V_DATA]);
        const void* v_meta = pointer<const void>(descriptor[meta::V_METADATA]);
        // This thread's loader task within a group: (slot, key, run).
        const int slot = threadIdx.x / (TILE * RN_PV_RUNS), task_key = threadIdx.x / RN_PV_RUNS % TILE;
        const int run = threadIdx.x % RN_PV_RUNS;
        uint4 raw = make_uint4(0u, 0u, 0u, 0u);
        unsigned scale = 0u;
        bool loaded = false;
        __half p_next[RN_PV_GROUP];
        auto fetch = [&](int g) {
            const int at = g * RN_PV_GROUP + slot;
            const int source = at < n_live ? physical[live[at] * TILE + task_key] : -1;
            loaded = source >= 0;
            if (loaded)
                tf_sm70::load_run<RN_RUN>(v_data, v_meta, codec, int64_t(source) * DIM + column_base + run * RN_RUN,
                                          raw, scale);
            #pragma unroll
            for (int i = 0; i < RN_PV_GROUP; ++i) {
                const int k = g * RN_PV_GROUP + i;
                p_next[i] = __float2half_rn(0.f);
                if (k < n_live && physical[live[k] * TILE + lane] >= 0) p_next[i] = p_row[live[k] * TILE + lane];
            }
        };
        if (groups) fetch(0);
        for (int g = 0; g < groups; ++g) {
            const int buffer = g & 1;
            if (loaded) {
                __half2* destination = reinterpret_cast<__half2*>(&staged[buffer][slot][task_key][run * RN_RUN]);
                #pragma unroll
                for (int c = 0; c < RN_RUN; c += 2)
                    destination[c / 2] = __halves2half2(tf_sm70::decode_run_value<RN_RUN>(raw, scale, codec, c, centroid_table),
                                                        tf_sm70::decode_run_value<RN_RUN>(raw, scale, codec, c + 1, centroid_table));
            }
            __half p_current[RN_PV_GROUP];
            #pragma unroll
            for (int i = 0; i < RN_PV_GROUP; ++i) p_current[i] = p_next[i];
            // Group g is staged; group g-1's readers of staged[buffer ^ 1] finished before this barrier.
            __syncthreads();
            if (g + 1 < groups) fetch(g + 1);
            #pragma unroll
            for (int i = 0; i < RN_PV_GROUP; ++i) {
                const int k = g * RN_PV_GROUP + i;
                if (k >= n_live) break;  // uniform
                if (physical[live[k] * TILE + lane] < 0) continue;
                const float p = __half2float(p_current[i]);
                if (!isfinite(p) || p < 0.f) {
                    latch(descriptor, 16, layer);
                    continue;
                }
                const __half2* values = reinterpret_cast<const __half2*>(&staged[buffer][i][lane][0]);
                #pragma unroll
                for (int c = 0; c < RN_PV_COLUMNS; c += 2) {
                    const float2 pair = __half22float2(values[c / 2]);
                    float v0 = pair.x, v1 = pair.y;
                    if (!isfinite(v0)) { latch(descriptor, 4, layer); v0 = 0.f; }
                    if (!isfinite(v1)) { latch(descriptor, 4, layer); v1 = 0.f; }
                    result[c] = __fmaf_rn(p, v0, result[c]);
                    result[c + 1] = __fmaf_rn(p, v1, result[c + 1]);
                }
            }
        }
    }
    #pragma unroll
    for (int c = 0; c < RN_PV_COLUMNS; ++c) {
        const float value = rn_warp_sum(result[c]);
        if (!lane) {
            if (!isfinite(value)) latch(descriptor, 4, layer);
            output[(int64_t(row) * HEADS + head) * DIM + column_base + c] = value;
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
    const dim3 grid(rows, HEADS / H, rows <= RN_FEW_ROWS ? RN_SPLITS_FEW : RN_SPLITS);
    const auto* q = reinterpret_cast<const __half*>(queries.data_ptr());
    if (P) {
        (codec ? rn_probability_kernel<H, true> : rn_probability_kernel<H, false>)
            <<<grid, H * 32, rn_table_bytes(codec), stream>>>(q, selected.data_ptr<int>(), table.data_ptr<int64_t>(),
            global.data_ptr<float>(), reinterpret_cast<__half*>(output.data_ptr()), rows, int(selected.size(2)), group,
            page_tokens, layer.data_ptr<int>(), codec);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }
    // Stream-ordered scratch (a capture's private pool inside graphs): rows x heads x 2051-key scores.
    auto scores = at::empty({rows, HEADS, RN_TILES * TILE}, queries.options().dtype(at::kFloat));
    (codec ? rn_scores_kernel<H, true> : rn_scores_kernel<H, false>)
        <<<grid, H * 32, rn_table_bytes(codec), stream>>>(q, selected.data_ptr<int>(), table.data_ptr<int64_t>(),
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
    (codec ? rn_pv_kernel<true> : rn_pv_kernel<false>)
        <<<dim3(rows, DIM / RN_PV_COLUMNS), RN_PV_THREADS, rn_table_bytes(codec), at::cuda::getCurrentCUDAStream()>>>(
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
