// Sym8 v1 bit operands. GEMV order matches qwen3_5/cuda/b16.cu:
// eight elements/lane, ordered FP32 fmaf, XOR16..1, BF16 RN store.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {
constexpr uint32_t sentinel = 0xffffff;

__device__ __forceinline__ bool require_word(bool condition, const char* message) {
    // Shipping validation survives NDEBUG. Explicit caller guards also prevent
    // invalid accesses if the compiler cannot prove assertion non-return.
    if (condition) return true;
    __assert_fail(message, "b16_sym8.cu", __LINE__, "Sym8 word");
    return false;
}

__device__ __forceinline__ bool word(const uint16_t* palette, const uint8_t* codes,
        const uint32_t* metadata, const uint16_t* raw, size_t group, int offset, uint32_t raw_groups, uint16_t& result) {
    const uint32_t info = metadata[group], index = info >> 8, flags = info & 255;
    if (index != sentinel) {
        if (!require_word(flags == 0 && index < raw_groups, "Sym8 raw index/flags outside owned pool")) return false;
        result = raw[size_t(index) * 32 + offset];
        return true;
    }
    const bool zero = flags & 1, extra = flags & 2;
    const uint32_t count = flags >> 2;
    if (!require_word(!(flags & 0xc0) && count <= 8 && (zero || count > 0) &&
                 (!extra || (zero && count == 8)), "Sym8 invalid compressed metadata")) return false;
    const uint32_t code = (codes[group * 16 + offset / 2] >> (4 * (offset & 1))) & 15;
    if (code == 0 && zero) { result = 0; return true; }
    uint32_t slot;
    bool sign;
    if (code == 0 || code == 8) {
        if (!require_word(count == 8, "Sym8 unused last palette code")) return false;
        slot = 7;
        sign = zero ? extra : code == 8;
    } else {
        slot = (code & 7) - 1;
        sign = code >= 8;
        if (!require_word(slot < (count < 7 ? count : 7), "Sym8 code outside active palette")) return false;
    }
    const uint16_t magnitude = palette[group * 8 + slot];
    if (!require_word(magnitude > 0 && magnitude < 0x8000, "Sym8 invalid absolute BF16 palette word")) return false;
    result = uint16_t(magnitude | (sign ? 0x8000 : 0));
    return true;
}

template <int R>
__global__ void __launch_bounds__(128) project(const at::BFloat16* __restrict__ x,
        const uint16_t* __restrict__ palette, const uint8_t* __restrict__ codes,
        const uint32_t* __restrict__ metadata, const uint16_t* __restrict__ raw,
        at::BFloat16* __restrict__ out, int M, int N, int K, uint32_t raw_groups) {
    const int lane = threadIdx.x & 31;
    const int first = blockIdx.y * R, col = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (col >= N) return;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.f;
    for (int k = 8 * lane; k + 8 <= K; k += 256) {
        float weights[8];
        const size_t group = size_t(col) * (K / 32) + k / 32;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint16_t bits = 0;
            if (!word(palette, codes, metadata, raw, group, k % 32 + i, raw_groups, bits)) return;
            weights[i] = __bfloat162float(__ushort_as_bfloat16(bits));
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (first + r >= M) break;
            // Object-representation copy, not an unproved BF16/vector alias.
            uint4 packed{};
            memcpy(&packed, x + size_t(first + r) * K + k, sizeof(packed));
            const uint32_t pairs[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const float a = __bfloat162float(__ushort_as_bfloat16(uint16_t(pairs[p] & 65535)));
                const float b = __bfloat162float(__ushort_as_bfloat16(uint16_t(pairs[p] >> 16)));
                acc[r] = fmaf(a, weights[2 * p], acc[r]);
                acc[r] = fmaf(b, weights[2 * p + 1], acc[r]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (first + r >= M) break;
        float sum = acc[r];
#pragma unroll
        for (int mask = 16; mask >= 1; mask >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, mask);
        if (lane == 0)
            out[size_t(first + r) * N + col] = at::BFloat16(
                __bfloat16_as_ushort(__float2bfloat16_rn(sum)), at::BFloat16::from_bits());
    }
}

template <typename I>
__global__ void gather(const I* ids, const uint16_t* palette, const uint8_t* codes,
        const uint32_t* metadata, const uint16_t* raw, at::BFloat16* out,
        size_t count, int N, int K, uint32_t raw_groups) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const I token = ids[index / K];
    if (!require_word(token >= 0 && uint64_t(token) < uint64_t(N), "Sym8 embedding ID outside vocabulary")) return;
    const size_t at = size_t(token) * K + index % K;
    uint16_t bits = 0;
    if (!word(palette, codes, metadata, raw, at / 32, int(at % 32), raw_groups, bits)) return;
    out[index] = at::BFloat16(bits, at::BFloat16::from_bits());
}

void checked(const at::Tensor& palette, const at::Tensor& codes, const at::Tensor& metadata,
             const at::Tensor& raw, int64_t N, int64_t K) {
    constexpr int64_t limit = int64_t(std::numeric_limits<int>::max()) - 256;
    TORCH_CHECK(N > 0 && N <= limit && K > 0 && K <= limit && K % 32 == 0, "invalid Sym8 dimensions");
    TORCH_CHECK(N <= std::numeric_limits<int64_t>::max() / K / 2, "Sym8 decoded byte product overflow");
    const int64_t groups = N * (K / 32);
    TORCH_CHECK(palette.is_cuda() && palette.is_contiguous() && codes.is_contiguous() && metadata.is_contiguous() && raw.is_contiguous(), "Sym8 contiguous CUDA fields required");
    TORCH_CHECK(codes.device() == palette.device() && metadata.device() == palette.device() && raw.device() == palette.device(), "Sym8 field device mismatch");
    TORCH_CHECK(palette.scalar_type() == at::kShort && palette.dim() == 2 && palette.size(0) == groups && palette.size(1) == 8, "Sym8 palette schema mismatch");
    TORCH_CHECK(codes.scalar_type() == at::kByte && codes.dim() == 2 && codes.size(0) == groups && codes.size(1) == 16, "Sym8 index schema mismatch");
    TORCH_CHECK(metadata.scalar_type() == at::kInt && metadata.dim() == 1 && metadata.size(0) == groups, "Sym8 metadata schema mismatch");
    TORCH_CHECK(raw.scalar_type() == at::kShort && raw.dim() == 2 && raw.size(1) == 32 && raw.size(0) <= sentinel, "Sym8 raw pool schema mismatch");
}
}

at::Tensor sym8_matmul(const at::Tensor& x, const at::Tensor& palette, const at::Tensor& codes,
                       const at::Tensor& metadata, const at::Tensor& raw, int64_t N, int64_t K) {
    checked(palette, codes, metadata, raw, N, K);
    TORCH_CHECK(x.device() == palette.device() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2 && x.size(1) == K, "Sym8 contiguous same-device BF16[M,K] required");
    TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 8, "Sym8 GEMV admits M1..8");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "Sym8 input requires16-byte alignment");
    c10::cuda::CUDAGuard guard(x.device());
    const auto* props = at::cuda::getDeviceProperties(x.get_device());
    TORCH_CHECK((N + 3) / 4 <= props->maxGridSize[0], "Sym8 GEMV launch grid exceeds device");
    const int M = int(x.size(0)), reuse = M == 1 ? 1 : M == 2 ? 2 : M <= 4 ? 4 : 8;
    auto out = at::empty({M, N}, x.options());
    const dim3 grid(unsigned((N + 3) / 4), unsigned((M + reuse - 1) / reuse));
#define RUN(R) project<R><<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<at::BFloat16>(), reinterpret_cast<const uint16_t*>(palette.data_ptr<int16_t>()), codes.data_ptr<uint8_t>(), reinterpret_cast<const uint32_t*>(metadata.data_ptr<int32_t>()), reinterpret_cast<const uint16_t*>(raw.data_ptr<int16_t>()), out.data_ptr<at::BFloat16>(), M, int(N), int(K), uint32_t(raw.size(0)))
    if (reuse == 1) { RUN(1); } else if (reuse == 2) { RUN(2); }
    else if (reuse == 4) { RUN(4); } else { RUN(8); }
#undef RUN
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

at::Tensor sym8_lookup(const at::Tensor& ids, const at::Tensor& palette, const at::Tensor& codes,
                       const at::Tensor& metadata, const at::Tensor& raw, int64_t N, int64_t K) {
    checked(palette, codes, metadata, raw, N, K);
    TORCH_CHECK(ids.device() == palette.device() && ids.is_contiguous() && (ids.scalar_type() == at::kInt || ids.scalar_type() == at::kLong), "Sym8 contiguous same-device int32/int64 IDs required");
    TORCH_CHECK(ids.numel() <= std::numeric_limits<int64_t>::max() / K / 2, "Sym8 lookup output byte product overflow");
    c10::cuda::CUDAGuard guard(ids.device());
    auto shape = ids.sizes().vec();
    shape.push_back(K);
    const size_t count = size_t(ids.numel()) * K;
    const auto* props = at::cuda::getDeviceProperties(ids.get_device());
    TORCH_CHECK((count + 255) / 256 <= uint64_t(props->maxGridSize[0]), "Sym8 lookup grid exceeds device");
    auto out = at::empty(shape, palette.options().dtype(at::kBFloat16));
    if (count == 0) return out;
#define GATHER(I) gather<I><<<unsigned((count + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(ids.data_ptr<I>(), reinterpret_cast<const uint16_t*>(palette.data_ptr<int16_t>()), codes.data_ptr<uint8_t>(), reinterpret_cast<const uint32_t*>(metadata.data_ptr<int32_t>()), reinterpret_cast<const uint16_t*>(raw.data_ptr<int16_t>()), out.data_ptr<at::BFloat16>(), count, int(N), int(K), uint32_t(raw.size(0)))
    if (ids.scalar_type() == at::kInt) { GATHER(int32_t); } else { GATHER(int64_t); }
#undef GATHER
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
