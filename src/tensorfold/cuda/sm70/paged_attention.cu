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

using T = at::Tensor;

namespace {
constexpr int HEADS = 12, PAD_HEADS = 16, DIM = 256, TILE = 32, THREADS = 128, WORDS = tf_sm70_paged::WORD_COUNT;
namespace wmma = nvcuda::wmma;
namespace meta = tf_sm70_paged;

// Pointer words are produced only by the canonical owner, which retains every
// backing tensor through stream/capture retirement. Device data is checked
// before dereferencing page offsets or payloads. A zero live-count pointer is
// the explicit inactive-row representation and never touches a State.
template<typename V> __device__ __forceinline__ V* pointer(int64_t word) {
    return reinterpret_cast<V*>(uintptr_t(word));
}

__device__ __forceinline__ void latch(const int64_t* descriptor, int bits, int layer) {
    int* status = pointer<int>(descriptor[meta::STATUS]);
    if (status == nullptr) {
        asm volatile("trap;");  // A live descriptor lost its required failure owner.
        return;
    }
    atomicOr(status, bits);
    atomicCAS(status + 1, -2, layer);
}

__device__ __forceinline__ bool active(const int64_t* descriptor, int layer) {
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

__device__ __forceinline__ bool domain(const int64_t* descriptor, const int* selections,
                                      int rows, int width, int row, int page_tokens,
                                      int layer, int& end, int& count, const int*& ids) {
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

__device__ __forceinline__ int64_t local_row(const int64_t* descriptor, int logical,
                                           int end, int page_tokens, int layer) {
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
    bool valid = true;
    if (enabled) {
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
            for (int peer = 0; peer < 4; ++peer) {
                const int64_t at = ((int64_t(group + peer) * rows + row) * HEADS + head) * 2;
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
    const int group = rank / 4 * 4, head = rank % 4 * 3 + local_head;
    float value = gathered[((int64_t(group) * rows + row) * HEADS + head) * DIM + col];
    #pragma unroll
    for (int peer = 1; peer < 4; ++peer)
        value = __fadd_rn(value, gathered[((int64_t(group + peer) * rows + row) * HEADS + head) * DIM + col]);
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
