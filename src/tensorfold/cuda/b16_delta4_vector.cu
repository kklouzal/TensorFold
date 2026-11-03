// Experimental owned Delta4 v1 projection/lookup. No checkpoint mutation.
// PTX ISA: vector loads access object-representation bytes with natural8/16B
// alignment, checked before launch. No C++ BF16/vector pointer type punning.
// https://docs.nvidia.com/cuda/parallel-thread-execution/#addresses-as-operands
// https://docs.nvidia.com/cuda/parallel-thread-execution/#vectors-as-operands
// CUDA Math13.4.2: __halves2bfloat162 preserves halves; __bfloat1622float2
// widens them. The FMA order below is families/qwen3_5/cuda/b16.cu's oracle.
// https://docs.nvidia.com/cuda/archive/13.4.2/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__MISC.html
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <limits>

namespace {
constexpr uint32_t compressed = 0xffffff;

__device__ __forceinline__ uint32_t load32(const void* address) {
    uint32_t value;
    asm volatile("ld.global.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
    return value;
}

__device__ __forceinline__ uint2 load64(const void* address) {
    uint2 value;
    asm volatile("ld.global.v2.u32 {%0, %1}, [%2];"
                 : "=r"(value.x), "=r"(value.y) : "l"(address) : "memory");
    return value;
}

__device__ __forceinline__ uint4 load128(const void* address) {
    uint4 value;
    asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                 : "l"(address) : "memory");
    return value;
}

__device__ __forceinline__ uint4 eight_words(const uint8_t* low, const uint8_t* delta, const int32_t* meta,
                                            const int16_t* raw, size_t block, int offset, uint32_t raw_blocks) {
    const uint32_t info = load32(meta + block), index = info >> 8;
    if (index != compressed) {
        if (index >= raw_blocks) {
            __assert_fail("raw escape index outside owned pool", "b16_delta4_vector.cu", __LINE__, "eight_words");
            __trap();
            return make_uint4(0, 0, 0, 0);  // unreachable; no unsafe pointer is formed
        }
        return load128(raw + size_t(index) * 64 + offset);
    }
    const uint2 lows = load64(low + block * 64 + offset);
    const uint32_t differences = load32(delta + block * 32 + offset / 2), maximum = info & 255;
    uint32_t pairs[4] = {};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint32_t difference = (differences >> (4 * i)) & 15;
        if (maximum < difference) {
            __assert_fail("compressed exponent delta exceeds maximum", "b16_delta4_vector.cu", __LINE__, "eight_words");
            __trap();
            return make_uint4(0, 0, 0, 0);  // unreachable; no invalid exponent is consumed
        }
        const uint32_t lo = ((i < 4 ? lows.x : lows.y) >> (8 * (i & 3))) & 255;
        const uint32_t bits = ((lo & 128) << 8) | ((maximum - difference) << 7) | (lo & 127);
        pairs[i / 2] |= bits << ((i & 1) * 16);
    }
    return make_uint4(pairs[0], pairs[1], pairs[2], pairs[3]);
}

__device__ __forceinline__ void floats8(uint4 words, float (&values)[8]) {
    const uint32_t pairs[4] = {words.x, words.y, words.z, words.w};
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const __nv_bfloat162 pair = __halves2bfloat162(__ushort_as_bfloat16(uint16_t(pairs[p])),
                                                      __ushort_as_bfloat16(uint16_t(pairs[p] >> 16)));
        const float2 value = __bfloat1622float2(pair);
        values[2 * p] = value.x;
        values[2 * p + 1] = value.y;
    }
}

__global__ void decode(const uint8_t* low, const uint8_t* delta, const int32_t* meta, const int16_t* raw,
                       at::BFloat16* out, size_t groups, uint32_t raw_blocks) {
    const size_t group = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (group >= groups) return;
    const size_t word_at = group * 8;
    const uint4 words = eight_words(low, delta, meta, raw, word_at / 64, int(word_at % 64), raw_blocks);
    const uint32_t pairs[4] = {words.x, words.y, words.z, words.w};
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        out[word_at + 2 * p] = at::BFloat16(uint16_t(pairs[p]), at::BFloat16::from_bits());
        out[word_at + 2 * p + 1] = at::BFloat16(uint16_t(pairs[p] >> 16), at::BFloat16::from_bits());
    }
}

template <typename ID>
__global__ void gather(const ID* ids, const uint8_t* low, const uint8_t* delta, const int32_t* meta,
                       const int16_t* raw, at::BFloat16* out, size_t groups, int N, int K, uint32_t raw_blocks) {
    const size_t group = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (group >= groups) return;
    const size_t row = group / (K / 8), column = (group % (K / 8)) * 8;
    const int64_t token = ids[row];
    if (token < 0 || token >= N) {
        __assert_fail("embedding ID outside original vocabulary", "b16_delta4_vector.cu", __LINE__, "gather");
        __trap();
        return;  // unreachable; no table pointer is formed for an invalid ID
    }
    const uint4 words = eight_words(low, delta, meta, raw, size_t(token) * (K / 64) + column / 64,
                                   int(column % 64), raw_blocks);
    const uint32_t pairs[4] = {words.x, words.y, words.z, words.w};
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        out[row * K + column + 2 * p] = at::BFloat16(uint16_t(pairs[p]), at::BFloat16::from_bits());
        out[row * K + column + 2 * p + 1] = at::BFloat16(uint16_t(pairs[p] >> 16), at::BFloat16::from_bits());
    }
}

template <int R>
__global__ void project(const at::BFloat16* __restrict__ x, const uint8_t* __restrict__ low,
                        const uint8_t* __restrict__ delta, const int32_t* __restrict__ meta,
                        const int16_t* __restrict__ raw, at::BFloat16* __restrict__ y,
                        int M, int N, int K, uint32_t raw_blocks) {
    constexpr int warps = 4;
    const int lane = threadIdx.x & 31, row0 = blockIdx.y * R;
    const int col = blockIdx.x * warps + (threadIdx.x >> 5);
    if (col >= N) return;
    float acc[R] = {};
    // k starts at8*lane and advances256, exactly the independent raw oracle.
    // K%64 and pointer checks make every8-value tile wholly owned/aligned.
    for (int k = 8 * lane; k + 8 <= K; k += 256) {
        float weights[8];
        floats8(eight_words(low, delta, meta, raw, size_t(col) * (K / 64) + k / 64,
                           k % 64, raw_blocks), weights);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (row0 + r >= M) break;
            float values[8];
            floats8(load128(x + size_t(row0 + r) * K + k), values);
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[r] = fmaf(values[i], weights[i], acc[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (row0 + r >= M) break;
        float value = acc[r];
#pragma unroll
        for (int mask = 16; mask >= 1; mask >>= 1) value += __shfl_xor_sync(0xffffffffu, value, mask);
        if (lane == 0)
            y[size_t(row0 + r) * N + col] = at::BFloat16(__bfloat16_as_ushort(__float2bfloat16_rn(value)),
                                                       at::BFloat16::from_bits());
    }
}

void checked(const at::Tensor& low, const at::Tensor& delta, const at::Tensor& meta, const at::Tensor& raw,
             int64_t N, int64_t K) {
    constexpr int64_t maximum = int64_t(std::numeric_limits<int>::max()) - 256;
    TORCH_CHECK(N > 0 && N <= maximum && K > 0 && K <= maximum && K % 64 == 0, "invalid encoded dimensions");
    TORCH_CHECK(N <= std::numeric_limits<int64_t>::max() / K / 2, "encoded byte product overflow");
    const int64_t blocks = N * (K / 64);
    TORCH_CHECK(low.is_cuda() && low.is_contiguous() && delta.is_contiguous() && meta.is_contiguous() && raw.is_contiguous(),
                "contiguous CUDA encoding required");
    TORCH_CHECK(delta.device() == low.device() && meta.device() == low.device() && raw.device() == low.device(),
                "encoded device mismatch");
    TORCH_CHECK(low.scalar_type() == at::kByte && low.dim() == 2 && low.size(0) == blocks && low.size(1) == 64,
                "invalid sign/mantissa payload");
    TORCH_CHECK(delta.scalar_type() == at::kByte && delta.dim() == 2 && delta.size(0) == blocks && delta.size(1) == 32,
                "invalid delta payload");
    TORCH_CHECK(meta.scalar_type() == at::kInt && meta.dim() == 1 && meta.size(0) == blocks, "invalid metadata payload");
    TORCH_CHECK(raw.scalar_type() == at::kShort && raw.dim() == 2 && raw.size(1) == 64 && raw.size(0) <= compressed,
                "invalid raw escape pool");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(low.data_ptr()) % 8 == 0 &&
                reinterpret_cast<uintptr_t>(delta.data_ptr()) % 4 == 0 &&
                reinterpret_cast<uintptr_t>(meta.data_ptr()) % 4 == 0 &&
                reinterpret_cast<uintptr_t>(raw.data_ptr()) % 16 == 0,
                "encoded vector bases require low8/delta4/metadata4/raw16-byte alignment");
}
}  // namespace

at::Tensor delta4_vector_unpack(const at::Tensor& low, const at::Tensor& delta, const at::Tensor& meta,
                               const at::Tensor& raw, int64_t N, int64_t K) {
    checked(low, delta, meta, raw, N, K);
    const size_t groups = size_t(N) * (K / 8);
    TORCH_CHECK((groups + 255) / 256 <= std::numeric_limits<int>::max(), "decode grid exceeds CUDA x-axis limit");
    c10::cuda::CUDAGuard guard(low.device());
    auto out = at::empty({N, K}, low.options().dtype(at::kBFloat16));
    decode<<<unsigned((groups + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        low.data_ptr<uint8_t>(), delta.data_ptr<uint8_t>(), meta.data_ptr<int32_t>(), raw.data_ptr<int16_t>(),
        out.data_ptr<at::BFloat16>(), groups, uint32_t(raw.size(0)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

at::Tensor delta4_vector_rows(const at::Tensor& x, const at::Tensor& low, const at::Tensor& delta,
                             const at::Tensor& meta, const at::Tensor& raw, int64_t N, int64_t K, int64_t R) {
    checked(low, delta, meta, raw, N, K);
    TORCH_CHECK(R == 1 || R == 2 || R == 4 || R == 8 || R == 16, "reuse rows must be1/2/4/8/16");
    TORCH_CHECK(x.device() == low.device() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() &&
                x.dim() == 2 && x.size(1) == K && x.size(0) >= 1 &&
                x.size(0) <= int64_t(std::numeric_limits<int>::max()) - 16,
                "input requires positive signed32 contiguous same-device BF16 rows of encoded K");
    TORCH_CHECK((x.size(0) + R - 1) / R <= 65535 &&
                x.size(0) <= std::numeric_limits<int64_t>::max() / N / 2,
                "projection grid or output byte product exceeds native bounds");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "input requires16-byte vector alignment");
    c10::cuda::CUDAGuard guard(low.device());
    auto out = at::empty({x.size(0), N}, x.options());
    const dim3 grid(unsigned((N + 3) / 4), unsigned((x.size(0) + R - 1) / R));
#define LAUNCH(REUSE) project<REUSE><<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>( \
        x.data_ptr<at::BFloat16>(), low.data_ptr<uint8_t>(), delta.data_ptr<uint8_t>(), meta.data_ptr<int32_t>(), \
        raw.data_ptr<int16_t>(), out.data_ptr<at::BFloat16>(), int(x.size(0)), int(N), int(K), uint32_t(raw.size(0)))
    if (R == 1) { LAUNCH(1); }
    else if (R == 2) { LAUNCH(2); }
    else if (R == 4) { LAUNCH(4); }
    else if (R == 8) { LAUNCH(8); }
    else { LAUNCH(16); }
#undef LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

at::Tensor delta4_vector_lookup(const at::Tensor& ids, const at::Tensor& low, const at::Tensor& delta,
                               const at::Tensor& meta, const at::Tensor& raw, int64_t N, int64_t K, int64_t maximum) {
    checked(low, delta, meta, raw, N, K);
    TORCH_CHECK(maximum > 0 && maximum <= std::numeric_limits<int64_t>::max() / K / 2,
                "invalid admitted lookup row bound");
    TORCH_CHECK(ids.is_cuda() && ids.device() == low.device() && ids.is_contiguous() &&
                (ids.scalar_type() == at::kInt || ids.scalar_type() == at::kLong),
                "lookup IDs require contiguous same-device INT32/INT64 storage");
    TORCH_CHECK(ids.numel() <= maximum &&
                reinterpret_cast<uintptr_t>(ids.data_ptr()) % ids.element_size() == 0,
                "lookup ID count or alignment exceeds admitted bounds");
    const size_t groups = size_t(ids.numel()) * (K / 8);
    TORCH_CHECK((groups + 255) / 256 <= std::numeric_limits<int>::max(), "lookup grid exceeds CUDA x-axis limit");
    auto shape = ids.sizes().vec();
    shape.push_back(K);
    c10::cuda::CUDAGuard guard(low.device());
    auto out = at::empty(shape, low.options().dtype(at::kBFloat16));
    if (groups == 0) return out;
#define GATHER(ID) gather<ID><<<unsigned((groups + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>( \
        ids.data_ptr<ID>(), low.data_ptr<uint8_t>(), delta.data_ptr<uint8_t>(), meta.data_ptr<int32_t>(), \
        raw.data_ptr<int16_t>(), out.data_ptr<at::BFloat16>(), groups, int(N), int(K), uint32_t(raw.size(0)))
    if (ids.scalar_type() == at::kInt) { GATHER(int32_t); }
    else { GATHER(int64_t); }
#undef GATHER
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
