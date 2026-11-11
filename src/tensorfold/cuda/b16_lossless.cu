// Private exact BF16 operand codec. Arithmetic loop/reduction is the owned
// families/qwen3_5/cuda/b16.cu matmul_rows oracle, with fused bit reconstruction.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {
constexpr uint32_t compressed = 0xffffff;

__device__ __forceinline__ uint16_t word(const uint8_t* low, const uint8_t* delta, const uint32_t* meta,
                                        const uint16_t* raw, size_t block, int offset, uint32_t raw_blocks) {
    uint32_t info = meta[block], index = info >> 8;
    if (index != compressed) {
        // Required shipping validation must survive NDEBUG. Device assertion
        // failure aborts this isolated process's CUDA work; no context reuse.
        if (index >= raw_blocks) __assert_fail("raw escape index outside owned pool", "b16_lossless.cu", __LINE__, "word");
        return raw[size_t(index) * 64 + offset];
    }
    uint32_t difference = (delta[block * 32 + offset / 2] >> ((offset & 1) * 4)) & 15;
    uint32_t maximum = info & 255;
    if (maximum < difference) __assert_fail("compressed exponent delta exceeds maximum", "b16_lossless.cu", __LINE__, "word");
    uint32_t value = low[block * 64 + offset];
    return uint16_t(((value & 128) << 8) | ((maximum - difference) << 7) | (value & 127));
}

__global__ void decode(const uint8_t* low, const uint8_t* delta, const uint32_t* meta, const uint16_t* raw,
                        at::BFloat16* out, size_t count, uint32_t raw_blocks) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) out[i] = at::BFloat16(word(low, delta, meta, raw, i / 64, int(i % 64), raw_blocks), at::BFloat16::from_bits());
}

template <int R>
__global__ void project(const at::BFloat16* __restrict__ x, const uint8_t* __restrict__ low,
                       const uint8_t* __restrict__ delta, const uint32_t* __restrict__ meta,
                       const uint16_t* __restrict__ raw, at::BFloat16* __restrict__ y,
                       int M, int N, int K, uint32_t raw_blocks) {
    constexpr int warps = 4;
    int lane = threadIdx.x & 31;
    int row0 = blockIdx.y * R, col = blockIdx.x * warps + (threadIdx.x >> 5);
    if (col >= N) return;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.f;
    // Same per-lane eight-element ordered FP32 FMA chain as existing b16.
    for (int k = 8 * lane; k + 8 <= K; k += 256) {
        float weights[8];
        size_t block = size_t(col) * (K / 64) + k / 64;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint16_t bits = word(low, delta, meta, raw, block, (k % 64) + i, raw_blocks);
            weights[i] = __bfloat162float(__ushort_as_bfloat16(bits));
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (row0 + r >= M) break;
            const at::BFloat16* input = x + size_t(row0 + r) * K + k;
            // CUDA13.4 declares device memcpy. Copy the permitted object
            // representation into an initialized vector; no BF16/vector alias.
            uint4 packed{};
            memcpy(&packed, input, sizeof(packed));
            const uint32_t pairs[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                float first = __bfloat162float(__ushort_as_bfloat16(uint16_t(pairs[p] & 65535)));
                float second = __bfloat162float(__ushort_as_bfloat16(uint16_t(pairs[p] >> 16)));
                acc[r] = fmaf(first, weights[2 * p], acc[r]);
                acc[r] = fmaf(second, weights[2 * p + 1], acc[r]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (row0 + r >= M) break;
        float value = acc[r];
#pragma unroll
        for (int mask = 16; mask >= 1; mask >>= 1) value += __shfl_xor_sync(0xffffffffu, value, mask);
        if (lane == 0) y[size_t(row0 + r) * N + col] = at::BFloat16(__bfloat16_as_ushort(__float2bfloat16_rn(value)), at::BFloat16::from_bits());
    }
}

void checked(const at::Tensor& low, const at::Tensor& delta, const at::Tensor& meta, const at::Tensor& raw,
             int64_t N, int64_t K) {
    constexpr int64_t max_dim = int64_t(std::numeric_limits<int>::max()) - 256;
    TORCH_CHECK(N > 0 && N <= max_dim && K > 0 && K <= max_dim && K % 64 == 0, "invalid encoded dimensions");
    TORCH_CHECK(N <= std::numeric_limits<int64_t>::max() / K, "encoded shape product overflow");
    int64_t blocks = N * (K / 64);
    TORCH_CHECK(low.is_cuda() && low.is_contiguous() && delta.is_contiguous() && meta.is_contiguous() && raw.is_contiguous(), "contiguous CUDA encoding required");
    TORCH_CHECK(delta.device() == low.device() && meta.device() == low.device() && raw.device() == low.device(), "encoded device mismatch");
    TORCH_CHECK(low.scalar_type() == at::kByte && low.dim() == 2 && low.size(0) == blocks && low.size(1) == 64, "invalid sign/mantissa payload");
    TORCH_CHECK(delta.scalar_type() == at::kByte && delta.dim() == 2 && delta.size(0) == blocks && delta.size(1) == 32, "invalid delta payload");
    TORCH_CHECK(meta.scalar_type() == at::kInt && meta.dim() == 1 && meta.size(0) == blocks, "invalid metadata payload");
    TORCH_CHECK(raw.scalar_type() == at::kShort && raw.dim() == 2 && raw.size(1) == 64 && raw.size(0) <= compressed, "invalid raw escape pool");
}
}

at::Tensor lossless_unpack(const at::Tensor& low, const at::Tensor& delta, const at::Tensor& meta,
                           const at::Tensor& raw, int64_t N, int64_t K) {
    checked(low, delta, meta, raw, N, K);
    c10::cuda::CUDAGuard guard(low.device());
    size_t count = size_t(N) * K;
    TORCH_CHECK((count + 255) / 256 <= std::numeric_limits<int>::max(), "decode grid exceeds CUDA x-axis limit");
    auto out = at::empty({N, K}, low.options().dtype(at::kBFloat16));
    decode<<<unsigned((count + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        low.data_ptr<uint8_t>(), delta.data_ptr<uint8_t>(), reinterpret_cast<const uint32_t*>(meta.data_ptr<int32_t>()),
        reinterpret_cast<const uint16_t*>(raw.data_ptr<int16_t>()), out.data_ptr<at::BFloat16>(), count, uint32_t(raw.size(0)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

at::Tensor lossless_rows(const at::Tensor& x, const at::Tensor& low, const at::Tensor& delta,
                         const at::Tensor& meta, const at::Tensor& raw, int64_t N, int64_t K, int64_t R) {
    checked(low, delta, meta, raw, N, K);
    TORCH_CHECK(R == 1 || R == 2 || R == 4 || R == 8, "reuse rows must be1/2/4/8");
    TORCH_CHECK(x.device() == low.device() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2 && x.size(1) == K, "contiguous same-device BF16 input required");
    TORCH_CHECK(x.size(0) > 0 && x.size(0) <= 8, "invalid input row count");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "input requires16-byte vector alignment");
    c10::cuda::CUDAGuard guard(low.device());
    auto out = at::empty({x.size(0), N}, x.options());
    dim3 grid(unsigned((N + 3) / 4), unsigned((x.size(0) + R - 1) / R));
    const auto* input = x.data_ptr<at::BFloat16>();
    auto* output = out.data_ptr<at::BFloat16>();
#define LAUNCH(ROWS) project<ROWS><<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(input, low.data_ptr<uint8_t>(), delta.data_ptr<uint8_t>(), reinterpret_cast<const uint32_t*>(meta.data_ptr<int32_t>()), reinterpret_cast<const uint16_t*>(raw.data_ptr<int16_t>()), output, int(x.size(0)), int(N), int(K), uint32_t(raw.size(0)))
    if (R == 1) { LAUNCH(1); }
    else if (R == 2) { LAUNCH(2); }
    else if (R == 4) { LAUNCH(4); }
    else { LAUNCH(8); }
#undef LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
