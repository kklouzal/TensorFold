// Original GGML byte-format implementation for TensorFold.
// Format facts and IQ4_NL codebook follow ggml-org/ggml (MIT); no dependency
// on ggml/vLLM layouts, operator registrations, allocator or scheduling code.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/core/InferenceMode.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>
#include "../kernels/tensor_contracts.cuh"
#include "gguf_decode_tables.cuh"

namespace {
struct BlockLayout { int values, bytes; };
BlockLayout block_layout(int64_t kind, const char* domain) {
    switch (kind) {
        case 0: return {1, 4};
        case 1: case 30: return {1, 2};
        case 7: return {32, 24};
        case 8: return {32, 34};
        case 12: return {256, 144};
        case 13: return {256, 176};
        case 14: return {256, 210};
        case 18: return {256, 98};
        case 20: return {32, 18};
        case 21: return {256, 110};
        case 22: return {256, 82};
        case 23: return {256, 136};
        case 42: return {64, 18};
        default: TORCH_CHECK(false, "unsupported ", domain, " encoding ", kind);
    }
}

__device__ __forceinline__ uint16_t u16(const uint8_t* p) {
    return uint16_t(p[0]) | (uint16_t(p[1]) << 8);
}
__device__ __forceinline__ uint32_t u32(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
__device__ __forceinline__ float half_value(const uint8_t* p) {
    return __half2float(__ushort_as_half(u16(p)));
}
__device__ __forceinline__ float value(const uint8_t* p, int type, int i) {
    if (type == 0) return __uint_as_float(u32(p));
    if (type == 1) return half_value(p);
    if (type == 30) return __uint_as_float(uint32_t(u16(p)) << 16);
    if (type == 8) return half_value(p) * float(int8_t(p[2 + i]));
    if (type == 42) return float(int((p[2 + i / 4] >> (2 * (i % 4))) & 3) - 1) * half_value(p);
    if (type == 14) {
        const int half = i / 128, quarter = (i % 128) / 32, lane = i % 32;
        const int lo = (p[half * 64 + (quarter % 2) * 32 + lane] >> (4 * (quarter / 2))) & 15;
        const int hi = (p[128 + half * 32 + lane] >> (2 * quarter)) & 3;
        const int scale = int(int8_t(p[192 + half * 8 + quarter * 2 + lane / 16]));
        return (half_value(p + 208) * float(scale)) * float((lo | (hi << 4)) - 32);
    }
    if (type == 23) {
        const int group = i / 32, element = i % 32;
        const int scale = int(((p[4 + group / 2] >> (4 * (group % 2))) & 15)
                             | (((u16(p + 2) >> (2 * group)) & 3) << 4)) - 32;
        const int code = (p[8 + group * 16 + element % 16] >> (4 * (element / 16))) & 15;
        return (half_value(p) * float(scale)) * float(tf_kvalues_iq4nl[code]);
    }
    if (type == 22) {
        const int group = i / 32, sub = (i % 32) / 8, element = i % 8;
        const int code = p[2 + group * 4 + sub] | ((uint32_t(p[66 + group]) << (8 - 2 * sub)) & 0x300);
        const uint64_t grid = tf_iq2s_grid[code];
        const int magnitude = int((grid >> (8 * element)) & 255);
        const int scale = (p[74 + group] >> (4 * (sub / 2))) & 15;
        const float d = (half_value(p) * (0.5f + float(scale))) * 0.25f;
        return (d * float(magnitude)) * ((p[34 + group * 4 + sub] & (1 << element)) ? -1.0f : 1.0f);
    }
    if (type == 18) {
        const int group = i / 32, sub = (i % 32) / 8, element = i % 8;
        const uint32_t auxiliary = u32(p + 66 + group * 4);
        const uint8_t signs = tf_ksigns_iq2xs[(auxiliary >> (7 * sub)) & 127];
        const uint32_t grid = tf_iq3xxs_grid[p[2 + group * 8 + sub * 2 + element / 4]];
        const float d = (half_value(p) * (0.5f + float(auxiliary >> 28))) * 0.5f;
        return (d * float((grid >> (8 * (element % 4))) & 255)) * ((signs & (1 << element)) ? -1.0f : 1.0f);
    }
    if (type == 21) {
        const int group = i / 32, sub = (i % 32) / 8, element = i % 8;
        const int code = p[2 + group * 8 + sub * 2 + element / 4]
            | ((uint32_t(p[66 + group]) << (8 - 2 * sub - element / 4)) & 256);
        const uint32_t grid = tf_iq3s_grid[code];
        const int scale = (p[106 + group / 2] >> (4 * (group % 2))) & 15;
        const float d = half_value(p) * float(1 + 2 * scale);
        return (d * float((grid >> (8 * (element % 4))) & 255))
            * ((p[74 + group * 4 + sub] & (1 << element)) ? -1.0f : 1.0f);
    }
    if (type == 20) {
        // The same exact signed integers already live in module read-only storage.
        const int q = (p[2 + i % 16] >> (4 * (i / 16))) & 15;
        return half_value(p) * float(tf_kvalues_iq4nl[q]);
    }
    if (type == 7) {
        const int q = ((p[8 + i % 16] >> (4 * (i / 16))) & 15) | (((u32(p + 4) >> i) & 1) << 4);
        return half_value(p) * float(q) + half_value(p + 2);
    }
    const int group = i / 32;
    const uint8_t* scales = p + 4;
    const int sc = group < 4 ? scales[group] & 63
        : (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4);
    const int minimum = group < 4 ? scales[group + 4] & 63
        : (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4);
    const uint8_t* qs = p + (type == 12 ? 16 : 48);
    int q = (qs[(i / 64) * 32 + i % 32] >> (4 * (group % 2))) & 15;
    if (type == 13) q |= ((p[16 + i % 32] >> group) & 1) << 4;
    return (half_value(p) * float(sc)) * float(q) - half_value(p + 2) * float(minimum);
}

template <bool F32>
__global__ void decode_kernel(const uint8_t* raw, const int64_t* ids, bool gather,
                              int64_t rows, int64_t stored_rows, int64_t row_bytes,
                              int type, int block, int block_bytes, int offset, int columns, void* out) {
    for (int64_t at = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; at < rows * columns;
         at += int64_t(gridDim.x) * blockDim.x) {
        const int64_t row = at / columns;
        const int col = at % columns + offset;
        const int64_t source = gather ? ids[row] : row;
        float result = __int_as_float(0x7fffffff); // Invalid trusted IDs never read unrelated memory.
        if (source >= 0 && source < stored_rows) {
            const uint8_t* p = raw + source * row_bytes + int64_t(col / block) * block_bytes;
            result = value(p, type, col % block);
        }
        if constexpr (F32) static_cast<float*>(out)[at] = result;
        else static_cast<__half*>(out)[at] = __float2half_rn(result);
    }
}
} // namespace

at::Tensor gguf_decode(const at::Tensor& raw, int64_t type, int64_t offset, int64_t columns,
                      const at::Tensor& ids, bool gather, bool f32) {
    TORCH_CHECK(raw.is_cuda() && raw.scalar_type() == at::kByte && raw.dim() == 2 && raw.is_contiguous(),
                "GGUF bytes must be contiguous CUDA uint8 [rows, packed_row_bytes]");
    const auto* properties = at::cuda::getDeviceProperties(raw.get_device());
    TORCH_CHECK(properties->major == 7 && properties->minor == 0, "GGUF SM70 operators require Volta");
    const auto [block, bytes] = block_layout(type, "GGML");
    TORCH_CHECK(raw.size(1) > 0 && raw.size(1) % bytes == 0, "GGUF row must contain complete source blocks");
    TORCH_CHECK(offset >= 0 && columns > 0 && offset <= std::numeric_limits<int>::max()
                && columns <= std::numeric_limits<int>::max() - offset
                && (offset + columns - 1) / block < raw.size(1) / bytes,
                "GGUF visible columns exceed the original packed row");
    TORCH_CHECK(ids.device() == raw.device() && ids.scalar_type() == at::kLong && ids.dim() == 1
                && ids.is_contiguous(), "GGUF row IDs must be contiguous CUDA int64 on the bytes' device");
    const int64_t rows = gather ? ids.numel() : raw.size(0);
    TORCH_CHECK(rows <= std::numeric_limits<int64_t>::max() / columns, "GGUF decoded extent overflows");
    c10::cuda::CUDAGuard guard(raw.device());
    auto out = at::empty({rows, columns}, raw.options().dtype(f32 ? at::kFloat : at::kHalf));
    if (rows) {
        const int blocks = int(std::min<int64_t>(1 + (rows * columns - 1) / 256, 65535));
        auto stream = at::cuda::getCurrentCUDAStream();
        if (f32) decode_kernel<true><<<blocks, 256, 0, stream>>>(raw.data_ptr<uint8_t>(), ids.data_ptr<int64_t>(), gather,
            rows, raw.size(0), raw.size(1), type, block, bytes, offset, columns, out.data_ptr());
        else decode_kernel<false><<<blocks, 256, 0, stream>>>(raw.data_ptr<uint8_t>(), ids.data_ptr<int64_t>(), gather,
            rows, raw.size(0), raw.size(1), type, block, bytes, offset, columns, out.data_ptr());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

namespace {
constexpr int EXPERT_WARPS = 4;

struct PackedView {
    const uint8_t* data;
    int64_t stride;
    int kind, block, bytes, offset;
};

template <int Kind = -1>
__device__ __forceinline__ float coefficient(PackedView bank, int64_t row, int column) {
    constexpr int fixed_block = Kind == 12 || Kind == 13 ? 256 : Kind == 7 ? 32 : 0;
    constexpr int fixed_bytes = Kind == 12 ? 144 : Kind == 13 ? 176 : Kind == 7 ? 24 : 0;
    const int block = fixed_block ? fixed_block : bank.block;
    const int bytes = fixed_bytes ? fixed_bytes : bank.bytes;
    const int col = column + bank.offset;
    const auto* raw = bank.data + row * bank.stride + int64_t(col / block) * bytes;
    // Preserve the reference decoder's stored coefficient boundary before
    // multiplying. FP16 operands have an exactly representable FP32 product.
    return __half2float(__float2half_rn(value(raw, Kind >= 0 ? Kind : bank.kind, col % block)));
}

template <typename Index, int GateKind = -1, int UpKind = -1, int Splits = 1>
__global__ void expert_gate_up(const __half* x, const Index* picks, PackedView gate, PackedView up,
                                const __half* shared_gate, const __half* shared_up,
                                int dims, int width, int slots, int experts, void* intermediate) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const unsigned output = blockIdx.x * EXPERT_WARPS + warp;
    const int pair = blockIdx.y;
    const auto expert = picks[pair];
    const int split = Splits == 1 ? 0 : blockIdx.z;
    if (output >= width) return;
    const bool valid = expert >= 0 && expert <= experts;
    float g = 0.0f, u = 0.0f;
    if (valid) {
        const int64_t source = int64_t(expert) * width + output;
        const __half* input = x + int64_t(pair / slots) * dims;
        for (unsigned k = lane + split * 32; k < unsigned(dims); k += 32 * Splits) {
            const float xv = __half2float(input[k]);
            const float gv = expert == experts ? __half2float(shared_gate[int64_t(output) * dims + k])
                                               : coefficient<GateKind>(gate, source, k);
            const float uv = expert == experts ? __half2float(shared_up[int64_t(output) * dims + k])
                                               : coefficient<UpKind>(up, source, k);
            g = fmaf(xv, gv, g);
            u = fmaf(xv, uv, u);
        }
        for (int offset = 16; offset; offset /= 2) {
            g += __shfl_down_sync(0xffffffff, g, offset);
            u += __shfl_down_sync(0xffffffff, u, offset);
        }
    }
    if (lane == 0) {
        if constexpr (Splits == 1) {
            const float result = valid ? (g / (1.0f + expf(-g))) * u : __int_as_float(0x7fffffff);
            static_cast<__half*>(intermediate)[int64_t(pair) * width + output] = __float2half_rn(result);
        } else {
            const int64_t at = ((int64_t(pair) * width + output) * Splits + split) * 2;
            static_cast<float*>(intermediate)[at] = valid ? g : __int_as_float(0x7fffffff);
            static_cast<float*>(intermediate)[at + 1] = valid ? u : __int_as_float(0x7fffffff);
        }
    }
}

template <int Byte>
__device__ __forceinline__ int scale_byte(const uint4& header) {
    static_assert(Byte >= 0 && Byte < 12, "K-quant scale byte");
    if constexpr (Byte < 4) return (header.y >> (Byte * 8)) & 255;
    else if constexpr (Byte < 8) return (header.z >> ((Byte - 4) * 8)) & 255;
    else return (header.w >> ((Byte - 8) * 8)) & 255;
}

template <int Group>
__device__ __forceinline__ float block_coefficient(const uint4& header, int quant) {
    int scale, minimum;
    if constexpr (Group < 4) {
        scale = scale_byte<Group>(header) & 63;
        minimum = scale_byte<Group + 4>(header) & 63;
    } else {
        scale = (scale_byte<Group + 4>(header) & 15) | ((scale_byte<Group - 4>(header) >> 6) << 4);
        minimum = (scale_byte<Group + 4>(header) >> 4) | ((scale_byte<Group>(header) >> 6) << 4);
    }
    const float d = __half2float(__ushort_as_half(uint16_t(header.x)));
    const float dmin = __half2float(__ushort_as_half(uint16_t(header.x >> 16)));
    // Keep the generic decoder's two multiplies/subtraction and FP16 boundary.
    return __half2float(__float2half_rn((d * float(scale)) * float(quant) - dmin * float(minimum)));
}

template <int Pair, int GateKind, int UpKind>
__device__ __forceinline__ void accumulate_block_pair(const __half* input, int lane,
    const uint8_t* gp, const uint8_t* up, const uint4& gh, const uint4& uh,
    int ghigh, int uhigh, float& g, float& u) {
    constexpr int group = Pair * 2;
    const int gbyte = gp[(GateKind == 12 ? 16 : 48) + Pair * 32 + lane];
    const int ubyte = up[(UpKind == 12 ? 16 : 48) + Pair * 32 + lane];
    const int glo = (gbyte & 15) | (((ghigh >> group) & 1) << 4);
    const int ulo = (ubyte & 15) | (((uhigh >> group) & 1) << 4);
    const int ghi = (gbyte >> 4) | (((ghigh >> (group + 1)) & 1) << 4);
    const int uhi = (ubyte >> 4) | (((uhigh >> (group + 1)) & 1) << 4);
    const float xlo = __half2float(input[lane + group * 32]);
    const float xhi = __half2float(input[lane + (group + 1) * 32]);
    g = fmaf(xlo, block_coefficient<group>(gh, glo), g);
    u = fmaf(xlo, block_coefficient<group>(uh, ulo), u);
    g = fmaf(xhi, block_coefficient<group + 1>(gh, ghi), g);
    u = fmaf(xhi, block_coefficient<group + 1>(uh, uhi), u);
}

template <typename Index, int GateKind, int UpKind>
__global__ void expert_gate_up_blocks(const __half* x, const Index* picks, PackedView gate, PackedView up,
    const __half* shared_gate, const __half* shared_up,
    int dims, int width, int slots, int experts, __half* activation) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const unsigned output = blockIdx.x * EXPERT_WARPS + warp;
    const int pair = blockIdx.y;
    if (output >= width) return;
    const auto expert = picks[pair];
    const bool valid = expert >= 0 && expert <= experts;
    float g = 0.0f, u = 0.0f;
    if (valid) {
        const __half* input = x + int64_t(pair / slots) * dims;
        if (expert == experts) {
            // Shared FP16 rows retain the qualified provider's exact loop.
            for (unsigned k = lane; k < unsigned(dims); k += 32) {
                const float xv = __half2float(input[k]);
                g = fmaf(xv, __half2float(shared_gate[int64_t(output) * dims + k]), g);
                u = fmaf(xv, __half2float(shared_up[int64_t(output) * dims + k]), u);
            }
        } else {
            const int64_t source = int64_t(expert) * width + output;
            for (int block = 0; block < dims / 256; ++block) {
                const auto* gp = gate.data + source * gate.stride + int64_t(block) * (GateKind == 12 ? 144 : 176);
                const auto* uptr = up.data + source * up.stride + int64_t(block) * (UpKind == 12 ? 144 : 176);
                const uint4 gh = *reinterpret_cast<const uint4*>(gp), uh = *reinterpret_cast<const uint4*>(uptr);
                const int ghigh = GateKind == 13 ? gp[16 + lane] : 0;
                const int uhigh = UpKind == 13 ? uptr[16 + lane] : 0;
                const __half* xx = input + block * 256;
                // Each lane keeps k=lane, lane+32, ... exactly as packed v1.
                accumulate_block_pair<0, GateKind, UpKind>(xx, lane, gp, uptr, gh, uh, ghigh, uhigh, g, u);
                accumulate_block_pair<1, GateKind, UpKind>(xx, lane, gp, uptr, gh, uh, ghigh, uhigh, g, u);
                accumulate_block_pair<2, GateKind, UpKind>(xx, lane, gp, uptr, gh, uh, ghigh, uhigh, g, u);
                accumulate_block_pair<3, GateKind, UpKind>(xx, lane, gp, uptr, gh, uh, ghigh, uhigh, g, u);
            }
        }
        for (int offset = 16; offset; offset /= 2) {
            g += __shfl_down_sync(0xffffffff, g, offset);
            u += __shfl_down_sync(0xffffffff, u, offset);
        }
    }
    if (lane == 0) {
        const float result = valid ? (g / (1.0f + expf(-g))) * u : __int_as_float(0x7fffffff);
        activation[int64_t(pair) * width + output] = __float2half_rn(result);
    }
}

template <int Pair, int Kind>
__device__ __forceinline__ void accumulate_projection_pair(const __half* input, int lane,
    const uint8_t* packed, const uint4& header, int high, float& sum) {
    constexpr int group = Pair * 2;
    const int byte = packed[(Kind == 12 ? 16 : 48) + Pair * 32 + lane];
    const int low = (byte & 15) | (((high >> group) & 1) << 4);
    const int upper = (byte >> 4) | (((high >> (group + 1)) & 1) << 4);
    sum = fmaf(__half2float(input[lane + group * 32]), block_coefficient<group>(header, low), sum);
    sum = fmaf(__half2float(input[lane + (group + 1) * 32]), block_coefficient<group + 1>(header, upper), sum);
}

template <int Kind>
__device__ __forceinline__ float projection_sum(const __half* input, PackedView bank,
    const __half* shared, int64_t source, int output, int dims, int lane, bool is_shared) {
    float sum = 0.0f;
    if (is_shared) {
        for (unsigned k = lane; k < unsigned(dims); k += 32)
            sum = fmaf(__half2float(input[k]), __half2float(shared[int64_t(output) * dims + k]), sum);
    } else {
        for (int block = 0; block < dims / 256; ++block) {
            const auto* packed = bank.data + source * bank.stride + int64_t(block) * (Kind == 12 ? 144 : 176);
            const uint4 header = *reinterpret_cast<const uint4*>(packed);
            const int high = Kind == 13 ? packed[16 + lane] : 0;
            const __half* x = input + block * 256;
            accumulate_projection_pair<0, Kind>(x, lane, packed, header, high, sum);
            accumulate_projection_pair<1, Kind>(x, lane, packed, header, high, sum);
            accumulate_projection_pair<2, Kind>(x, lane, packed, header, high, sum);
            accumulate_projection_pair<3, Kind>(x, lane, packed, header, high, sum);
        }
    }
    for (int offset = 16; offset; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    return sum;
}

template <typename Index, int GateKind, int UpKind>
__global__ void expert_gate_up_blocks2w(const __half* x, const Index* picks, PackedView gate, PackedView up,
    const __half* shared_gate, const __half* shared_up,
    int dims, int width, int slots, int experts, __half* activation) {
    __shared__ float projections[EXPERT_WARPS];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const unsigned output = blockIdx.x * (EXPERT_WARPS / 2) + warp / 2;
    const int pair = blockIdx.y;
    const auto expert = picks[pair];
    const bool live = output < width, valid = live && expert >= 0 && expert <= experts;
    float sum = __int_as_float(0x7fffffff);
    if (valid) {
        const __half* input = x + int64_t(pair / slots) * dims;
        const int64_t source = int64_t(expert) * width + output;
        if constexpr (GateKind == UpKind) {
            sum = projection_sum<GateKind>(input, warp % 2 ? up : gate,
                warp % 2 ? shared_up : shared_gate, source, output, dims, lane, expert == experts);
        } else if (warp % 2 == 0) {
            sum = projection_sum<GateKind>(input, gate, shared_gate, source, output, dims, lane, expert == experts);
        } else {
            sum = projection_sum<UpKind>(input, up, shared_up, source, output, dims, lane, expert == experts);
        }
    }
    if (lane == 0) projections[warp] = sum;
    // All four warps participate, including the final inactive output pair.
    // FP32 stores/loads retain the original g/u scalar bits before SwiGLU.
    __syncthreads();
    if (live && warp % 2 == 0 && lane == 0) {
        const float g = projections[warp], u = projections[warp + 1];
        const float result = valid ? (g / (1.0f + expf(-g))) * u : __int_as_float(0x7fffffff);
        activation[int64_t(pair) * width + output] = __float2half_rn(result);
    }
}

__global__ void expert_reduce_splitk4(const float* partials, __half* activation, int64_t count) {
    for (int64_t at = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; at < count;
         at += int64_t(gridDim.x) * blockDim.x) {
        const float* p = partials + at * 8;
        float g = p[0], u = p[1];
        #pragma unroll
        for (int split = 1; split < 4; ++split) {
            g += p[split * 2];
            u += p[split * 2 + 1];
        }
        activation[at] = __float2half_rn((g / (1.0f + expf(-g))) * u);
    }
}

template <typename Index, bool F32, int DownKind = -1>
__global__ void expert_down(const __half* activation, const Index* picks, PackedView down,
                             const __half* shared_down, int dims, int width, int experts, void* out) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    // Four independent eight-lane reductions share a warp for the narrow
    // TP8 K=80 down projection. Wider shards keep the same row-local order.
    const int part = lane % 8;
    const unsigned output = blockIdx.x * (EXPERT_WARPS * 4) + warp * 4 + lane / 8;
    const int pair = blockIdx.y;
    const auto expert = picks[pair];
    const unsigned mask = __ballot_sync(0xffffffff, output < dims);
    if (output >= dims) return;
    const bool valid = expert >= 0 && expert <= experts;
    float sum = 0.0f;
    if (valid) {
        const int64_t source = int64_t(expert) * dims + output;
        const __half* input = activation + int64_t(pair) * width;
        for (unsigned k = part; k < unsigned(width); k += 8) {
            const float w = expert == experts ? __half2float(shared_down[int64_t(output) * width + k])
                                               : coefficient<DownKind>(down, source, k);
            sum = fmaf(__half2float(input[k]), w, sum);
        }
        for (int offset = 4; offset; offset /= 2)
            sum += __shfl_down_sync(mask, sum, offset, 8);
    } else {
        sum = __int_as_float(0x7fffffff);
    }
    if (part == 0) {
        const int64_t at = int64_t(pair) * dims + output;
        if constexpr (F32) static_cast<float*>(out)[at] = sum;
        else static_cast<__half*>(out)[at] = __float2half_rn(sum);
    }
}

void operand(const at::Tensor& x, at::ScalarType dtype, at::Device device, const char* name) {
    TORCH_CHECK(x.is_cuda() && x.device() == device && x.scalar_type() == dtype && x.is_contiguous(),
                name, " must be contiguous on the input CUDA device with its declared dtype");
}

PackedView packed_view(const at::Tensor& raw, int64_t kind, int64_t offset, int columns, at::Device device) {
    operand(raw, at::kByte, device, "packed expert bank");
    TORCH_CHECK(raw.dim() == 2 && raw.size(0) > 0, "packed expert bank must be a nonempty matrix");
    const auto [block, bytes] = block_layout(kind, "GGML expert");
    TORCH_CHECK(raw.size(1) > 0 && raw.size(1) % bytes == 0 && offset >= 0 && columns > 0
                && offset <= std::numeric_limits<int>::max() - columns
                && (offset + columns - 1) / block < raw.size(1) / bytes,
                "packed expert columns must fit complete source blocks");
    return {raw.data_ptr<uint8_t>(), raw.size(1), int(kind), block, bytes, int(offset)};
}

void run_exact_down(const at::Tensor& activation, const at::Tensor& picks, PackedView view,
                    const at::Tensor& shared, at::Tensor& result, int dims, int experts);

template <typename Index, int Splits = 1>
void run_experts(const at::Tensor& x, const at::Tensor& picks, PackedView gate, PackedView up, PackedView down,
                   const at::Tensor& shared_gate, const at::Tensor& shared_up, const at::Tensor& shared_down,
                   at::Tensor& output, at::Tensor& activation, int dims, int width, int slots, int experts,
                   bool blockwise = false, bool split_projections = false, bool exact_down = false, bool gate_only = false) {
    const int pairs = int(picks.numel());
    const auto* xx = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
    const auto* ids = picks.data_ptr<Index>();
    const auto* sg = reinterpret_cast<const __half*>(shared_gate.data_ptr<at::Half>());
    const auto* su = reinterpret_cast<const __half*>(shared_up.data_ptr<at::Half>());
    const auto* sd = reinterpret_cast<const __half*>(shared_down.data_ptr<at::Half>());
    auto* act = reinterpret_cast<__half*>(activation.data_ptr<at::Half>());
    auto intermediate = activation;
    if constexpr (Splits == 4)
        intermediate = at::empty({pairs, width, Splits, 2}, x.options().dtype(at::kFloat));
    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 gu_grid(1 + (width - 1) / EXPERT_WARPS, pairs, Splits);
    const dim3 down_grid(1 + (dims - 1) / (EXPERT_WARPS * 4), pairs);
    bool launched = false;
    if constexpr (Splits == 1) {
        const bool kquants = (gate.kind == 12 || gate.kind == 13) && (up.kind == 12 || up.kind == 13);
        const bool aligned = (reinterpret_cast<uintptr_t>(gate.data) & 15) == 0
            && (reinterpret_cast<uintptr_t>(up.data) & 15) == 0 && gate.stride % 16 == 0 && up.stride % 16 == 0;
        if (blockwise && kquants && aligned && gate.offset == 0 && up.offset == 0 && dims % 256 == 0) {
            const dim3 projection_grid(1 + (width - 1) / (EXPERT_WARPS / 2), pairs);
            if (split_projections && gate.kind == 12 && up.kind == 12)
                expert_gate_up_blocks2w<Index, 12, 12><<<projection_grid, EXPERT_WARPS * 32, 0, stream>>>(
                    xx, ids, gate, up, sg, su, dims, width, slots, experts, act);
            else if (split_projections && gate.kind == 13 && up.kind == 13)
                expert_gate_up_blocks2w<Index, 13, 13><<<projection_grid, EXPERT_WARPS * 32, 0, stream>>>(
                    xx, ids, gate, up, sg, su, dims, width, slots, experts, act);
            else if (split_projections && gate.kind == 12)
                expert_gate_up_blocks2w<Index, 12, 13><<<projection_grid, EXPERT_WARPS * 32, 0, stream>>>(
                    xx, ids, gate, up, sg, su, dims, width, slots, experts, act);
            else if (split_projections)
                expert_gate_up_blocks2w<Index, 13, 12><<<projection_grid, EXPERT_WARPS * 32, 0, stream>>>(
                    xx, ids, gate, up, sg, su, dims, width, slots, experts, act);
            else if (gate.kind == 12 && up.kind == 12)
                expert_gate_up_blocks<Index, 12, 12><<<gu_grid, EXPERT_WARPS * 32, 0, stream>>>(
                    xx, ids, gate, up, sg, su, dims, width, slots, experts, act);
            else if (gate.kind == 13 && up.kind == 13)
                expert_gate_up_blocks<Index, 13, 13><<<gu_grid, EXPERT_WARPS * 32, 0, stream>>>(
                    xx, ids, gate, up, sg, su, dims, width, slots, experts, act);
            else if (gate.kind == 12)
                expert_gate_up_blocks<Index, 12, 13><<<gu_grid, EXPERT_WARPS * 32, 0, stream>>>(
                    xx, ids, gate, up, sg, su, dims, width, slots, experts, act);
            else
                expert_gate_up_blocks<Index, 13, 12><<<gu_grid, EXPERT_WARPS * 32, 0, stream>>>(
                    xx, ids, gate, up, sg, su, dims, width, slots, experts, act);
            launched = true;
        }
    }
    if (!launched && gate.kind == 12 && up.kind == 12)
        expert_gate_up<Index, 12, 12, Splits><<<gu_grid, EXPERT_WARPS * 32, 0, stream>>>(xx, ids, gate, up, sg, su,
            dims, width, slots, experts, intermediate.data_ptr());
    else if (!launched && gate.kind == 13 && up.kind == 13)
        expert_gate_up<Index, 13, 13, Splits><<<gu_grid, EXPERT_WARPS * 32, 0, stream>>>(xx, ids, gate, up, sg, su,
            dims, width, slots, experts, intermediate.data_ptr());
    else if (!launched)
        expert_gate_up<Index, -1, -1, Splits><<<gu_grid, EXPERT_WARPS * 32, 0, stream>>>(xx, ids, gate, up, sg, su,
            dims, width, slots, experts, intermediate.data_ptr());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if constexpr (Splits == 4) {
        const int64_t count = int64_t(pairs) * width;
        expert_reduce_splitk4<<<std::min<int64_t>((count + 127) / 128, 65535), 128, 0, stream>>>(
            intermediate.data_ptr<float>(), act, count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (gate_only) return;
    if (exact_down) {
        run_exact_down(activation, picks, down, shared_down, output, dims, experts);
        return;
    }
    if (down.kind == 7) {
        if (output.scalar_type() == at::kFloat)
            expert_down<Index, true, 7><<<down_grid, EXPERT_WARPS * 32, 0, stream>>>(act, ids, down, sd,
                dims, width, experts, output.data_ptr());
        else
            expert_down<Index, false, 7><<<down_grid, EXPERT_WARPS * 32, 0, stream>>>(act, ids, down, sd,
                dims, width, experts, output.data_ptr());
    } else {
        if (output.scalar_type() == at::kFloat)
            expert_down<Index, true><<<down_grid, EXPERT_WARPS * 32, 0, stream>>>(act, ids, down, sd,
                dims, width, experts, output.data_ptr());
        else
            expert_down<Index, false><<<down_grid, EXPERT_WARPS * 32, 0, stream>>>(act, ids, down, sd,
                dims, width, experts, output.data_ptr());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace

void gguf_experts_impl(const at::Tensor& x, const at::Tensor& picks,
                  const at::Tensor& gate, int64_t gate_kind, int64_t gate_offset,
                  const at::Tensor& up, int64_t up_kind, int64_t up_offset,
                  const at::Tensor& down, int64_t down_kind, int64_t down_offset,
                  const at::Tensor& shared_gate, const at::Tensor& shared_up,
                  const at::Tensor& shared_down, at::Tensor output, bool split_k4, bool blockwise = false,
                  bool split_projections = false, bool exact_down = false) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.dim() == 2 && x.is_contiguous()
                && x.size(1) > 0 && x.size(1) <= std::numeric_limits<int>::max(),
                "packed experts require contiguous FP16 CUDA [rows,hidden] inputs");
    const auto device = x.device();
    const auto* prop = at::cuda::getDeviceProperties(x.get_device());
    TORCH_CHECK(prop->major == 7 && prop->minor == 0, "packed expert provider requires SM70");
    TORCH_CHECK(picks.scalar_type() == at::kInt || picks.scalar_type() == at::kLong,
                "packed expert picks must be int32 or int64");
    operand(picks, picks.scalar_type(), device, "expert picks");
    TORCH_CHECK(picks.dim() == 2 && picks.size(0) == x.size(0) && picks.size(1) > 0
                && picks.numel() <= 65535, "packed expert slot geometry exceeds its bounded launch");
    for (const auto* shared : {&shared_gate, &shared_up, &shared_down})
        operand(*shared, at::kHalf, device, "shared expert weights");
    TORCH_CHECK(shared_gate.dim() == 2 && shared_gate.size(0) > 0
                && shared_gate.size(0) <= std::numeric_limits<int>::max()
                && shared_gate.size(1) == x.size(1) && shared_up.sizes() == shared_gate.sizes()
                && shared_down.sizes() == at::IntArrayRef({x.size(1), shared_gate.size(0)}),
                "shared expert dimensions disagree with input or local expert width");
    const int dims = int(x.size(1)), width = int(shared_gate.size(0));
    auto g = packed_view(gate, gate_kind, gate_offset, dims, device);
    auto u = packed_view(up, up_kind, up_offset, dims, device);
    auto d = packed_view(down, down_kind, down_offset, width, device);
    TORCH_CHECK(gate.size(0) % width == 0 && gate.size(0) / width <= std::numeric_limits<int>::max()
                && up.size(0) == gate.size(0) && down.size(0) % dims == 0
                && down.size(0) / dims == gate.size(0) / width, "packed expert bank row counts disagree");
    TORCH_CHECK(output.scalar_type() == at::kHalf || output.scalar_type() == at::kFloat,
                "packed expert output must be FP16 or FP32");
    operand(output, output.scalar_type(), device, "expert output");
    TORCH_CHECK(output.sizes() == at::IntArrayRef({x.size(0), picks.size(1), x.size(1)}),
                "packed expert output must be [rows,slots,hidden]");
    tensorfold::check_disjoint(output, {&x, &picks, &gate, &up, &down, &shared_gate, &shared_up, &shared_down},
                               "packed expert output must not overlap its inputs");
    c10::cuda::CUDAGuard guard(device);
    if (picks.numel() == 0) return;
    auto activation = at::empty({picks.numel(), width}, x.options());
    if (picks.scalar_type() == at::kInt) {
        if (split_k4)
            run_experts<int32_t, 4>(x, picks, g, u, d, shared_gate, shared_up, shared_down,
                output, activation, dims, width, int(picks.size(1)), int(gate.size(0) / width));
        else
            run_experts<int32_t>(x, picks, g, u, d, shared_gate, shared_up, shared_down,
                output, activation, dims, width, int(picks.size(1)), int(gate.size(0) / width),
                blockwise, split_projections, exact_down);
    } else {
        if (split_k4)
            run_experts<int64_t, 4>(x, picks, g, u, d, shared_gate, shared_up, shared_down,
                output, activation, dims, width, int(picks.size(1)), int(gate.size(0) / width));
        else
            run_experts<int64_t>(x, picks, g, u, d, shared_gate, shared_up, shared_down,
                output, activation, dims, width, int(picks.size(1)), int(gate.size(0) / width),
                blockwise, split_projections, exact_down);
    }
}

void gguf_experts(const at::Tensor& x, const at::Tensor& picks,
                  const at::Tensor& gate, int64_t gate_kind, int64_t gate_offset,
                  const at::Tensor& up, int64_t up_kind, int64_t up_offset,
                  const at::Tensor& down, int64_t down_kind, int64_t down_offset,
                  const at::Tensor& shared_gate, const at::Tensor& shared_up,
                  const at::Tensor& shared_down, at::Tensor output) {
    gguf_experts_impl(x, picks, gate, gate_kind, gate_offset, up, up_kind, up_offset,
                      down, down_kind, down_offset, shared_gate, shared_up, shared_down, output, false);
}

void gguf_experts_splitk4(const at::Tensor& x, const at::Tensor& picks,
                         const at::Tensor& gate, int64_t gate_kind, int64_t gate_offset,
                         const at::Tensor& up, int64_t up_kind, int64_t up_offset,
                         const at::Tensor& down, int64_t down_kind, int64_t down_offset,
                         const at::Tensor& shared_gate, const at::Tensor& shared_up,
                         const at::Tensor& shared_down, at::Tensor output) {
    gguf_experts_impl(x, picks, gate, gate_kind, gate_offset, up, up_kind, up_offset,
                      down, down_kind, down_offset, shared_gate, shared_up, shared_down, output, true);
}

void gguf_experts_blocks(const at::Tensor& x, const at::Tensor& picks,
                         const at::Tensor& gate, int64_t gate_kind, int64_t gate_offset,
                         const at::Tensor& up, int64_t up_kind, int64_t up_offset,
                         const at::Tensor& down, int64_t down_kind, int64_t down_offset,
                         const at::Tensor& shared_gate, const at::Tensor& shared_up,
                         const at::Tensor& shared_down, at::Tensor output) {
    gguf_experts_impl(x, picks, gate, gate_kind, gate_offset, up, up_kind, up_offset,
                      down, down_kind, down_offset, shared_gate, shared_up, shared_down, output, false, true);
}

void gguf_experts_blocks2w(const at::Tensor& x, const at::Tensor& picks,
                         const at::Tensor& gate, int64_t gate_kind, int64_t gate_offset,
                         const at::Tensor& up, int64_t up_kind, int64_t up_offset,
                         const at::Tensor& down, int64_t down_kind, int64_t down_offset,
                         const at::Tensor& shared_gate, const at::Tensor& shared_up,
                         const at::Tensor& shared_down, at::Tensor output) {
    gguf_experts_impl(x, picks, gate, gate_kind, gate_offset, up, up_kind, up_offset,
                      down, down_kind, down_offset, shared_gate, shared_up, shared_down, output, false, true, true);
}

namespace {
__global__ void dense_pad32(const __half* input, __half* padded, int rows, int columns,
                             int64_t row_stride, int64_t column_stride) {
    for (int64_t at = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; at < int64_t(32) * columns;
         at += int64_t(gridDim.x) * blockDim.x) {
        const int row = at / columns, col = at % columns;
        padded[at] = row < rows ? input[int64_t(row) * row_stride + int64_t(col) * column_stride]
                                : __float2half_rn(0.0f);
    }
}

template <bool SourceF32, bool OutF32>
__global__ void dense_copy32(const void* value, void* out, int rows, int columns, int64_t stride) {
    for (int64_t at = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; at < int64_t(rows) * columns;
         at += int64_t(gridDim.x) * blockDim.x) {
        const int64_t dest = (at / columns) * stride + at % columns;
        float result;
        if constexpr (SourceF32) result = static_cast<const float*>(value)[at];
        else result = __half2float(static_cast<const __half*>(value)[at]);
        if constexpr (OutF32) static_cast<float*>(out)[dest] = result;
        else static_cast<__half*>(out)[dest] = __float2half_rn(result);
    }
}
} // namespace

void gguf_dense32(const at::Tensor& x, const at::Tensor& weight, at::Tensor output, bool force_f32) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.dim() == 2
                && x.size(0) <= 32 && x.size(1) > 0 && x.size(1) <= std::numeric_limits<int>::max()
                && x.stride(0) >= 0 && x.stride(1) >= 0,
                "dense32 input must be at most32 FP16 CUDA rows with nonnegative strides");
    const auto device = x.device();
    const auto* prop = at::cuda::getDeviceProperties(x.get_device());
    TORCH_CHECK(prop->major == 7 && prop->minor == 0, "dense32 provider requires SM70");
    operand(weight, at::kHalf, device, "dense32 weight");
    TORCH_CHECK(weight.dim() == 2 && weight.size(1) == x.size(1) && weight.size(0) > 0
                && weight.size(0) <= std::numeric_limits<int>::max(), "dense32 weight geometry disagrees");
    TORCH_CHECK(output.device() == device && output.dim() == 2 && output.size(0) == x.size(0)
                && output.size(1) == weight.size(0) && output.stride(1) == 1 && output.stride(0) >= output.size(1)
                && (output.scalar_type() == at::kHalf || output.scalar_type() == at::kFloat),
                "dense32 output must be FP16/FP32 CUDA rows with contiguous columns and disjoint row strides");
    // Unlike a GEMV rewrite, retaining this exact M32 ATen call preserves its
    // cuBLAS dispatch and output-dtype arithmetic. Only staging changes.
    c10::cuda::CUDAGuard guard(device);
    const int rows = int(x.size(0)), columns = int(x.size(1)), outputs = int(weight.size(0));
    if (!rows) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    const bool prepared_input = rows == 32 && x.is_contiguous()
        && reinterpret_cast<uintptr_t>(x.data_ptr()) % 512 == 0;
    auto padded = prepared_input ? x : at::empty({32, columns}, x.options());
    if (!prepared_input) {
        dense_pad32<<<std::min<int64_t>((int64_t(32) * columns + 255) / 256, 65535), 256, 0, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(padded.data_ptr<at::Half>()), rows, columns, x.stride(0), x.stride(1));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    const bool f32 = force_f32 || output.scalar_type() == at::kFloat;
    if (rows == 32 && output.is_contiguous() && reinterpret_cast<uintptr_t>(output.data_ptr()) % 512 == 0
            && (output.scalar_type() == at::kFloat || !force_f32)
            && !tensorfold::overlaps(output, padded) && !tensorfold::overlaps(output, weight)) {
        if (f32) at::mm_out(output, padded, weight.t(), at::kFloat);
        else at::mm_out(output, padded, weight.t());
        return;
    }
    auto value = f32 ? at::mm(padded, weight.t(), at::kFloat) : at::mm(padded, weight.t());
    const int blocks = int(std::min<int64_t>((int64_t(rows) * outputs + 255) / 256, 65535));
    if (output.scalar_type() == at::kFloat)
        dense_copy32<true, true><<<blocks, 256, 0, stream>>>(value.data_ptr(), output.data_ptr(),
            rows, outputs, output.stride(0));
    else if (f32)
        dense_copy32<true, false><<<blocks, 256, 0, stream>>>(value.data_ptr(), output.data_ptr(),
            rows, outputs, output.stride(0));
    else
        dense_copy32<false, false><<<blocks, 256, 0, stream>>>(value.data_ptr(), output.data_ptr(),
            rows, outputs, output.stride(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {
constexpr int MAX_CONCAT_PARTS = 8;
struct ConcatSources {
    const void* data[MAX_CONCAT_PARTS];
    int starts[MAX_CONCAT_PARTS + 1];
    int count;
};

template <bool F32>
// Every field is read-only for the grid; indexed arrays need no per-thread copy.
__global__ void concat_copy32(const __grid_constant__ ConcatSources sources,
                              void* out, int rows, int columns, int64_t stride) {
    for (int64_t at = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; at < int64_t(rows) * columns;
         at += int64_t(gridDim.x) * blockDim.x) {
        const int row = at / columns, col = at % columns;
        int part = 0;
        for (int next = 1; next < sources.count; ++next) {
            if (col < sources.starts[next]) break;
            part = next;
        }
        const int width = sources.starts[part + 1] - sources.starts[part];
        const int64_t src = int64_t(row) * width + col - sources.starts[part];
        const int64_t dst = int64_t(row) * stride + col;
        if constexpr (F32) static_cast<float*>(out)[dst] = static_cast<const float*>(sources.data[part])[src];
        else static_cast<__half*>(out)[dst] = static_cast<const __half*>(sources.data[part])[src];
    }
}
} // namespace

void gguf_concat32(const at::Tensor& x, const std::vector<at::Tensor>& weights, at::Tensor output) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.dim() == 2 && x.size(0) <= 32
                && x.size(1) > 0 && x.size(1) <= std::numeric_limits<int>::max()
                && x.stride(0) >= 0 && x.stride(1) >= 0,
                "concat32 input must be at most32 FP16 CUDA rows with nonnegative strides");
    const auto device = x.device();
    const auto* prop = at::cuda::getDeviceProperties(x.get_device());
    TORCH_CHECK(prop->major == 7 && prop->minor == 0, "concat32 provider requires SM70");
    TORCH_CHECK(!weights.empty() && weights.size() <= MAX_CONCAT_PARTS,
                "concat32 requires one through8 original dense parts");
    ConcatSources sources{};
    sources.count = int(weights.size());
    int64_t total = 0;
    for (size_t part = 0; part < weights.size(); ++part) {
        const auto& weight = weights[part];
        operand(weight, at::kHalf, device, "concat32 weight");
        TORCH_CHECK(weight.dim() == 2 && weight.size(1) == x.size(1) && weight.size(0) > 0
                    && weight.size(0) <= std::numeric_limits<int>::max() - total,
                    "concat32 original weight dimensions disagree");
        sources.starts[part] = int(total);
        total += weight.size(0);
    }
    sources.starts[weights.size()] = int(total);
    TORCH_CHECK(output.device() == device && output.dim() == 2 && output.size(0) == x.size(0)
                && output.size(1) == total && output.stride(1) == 1 && output.stride(0) >= total
                && (output.scalar_type() == at::kHalf || output.scalar_type() == at::kFloat),
                "concat32 output must be FP16/FP32 rows with contiguous columns and disjoint row strides");
    c10::cuda::CUDAGuard guard(device);
    const int rows = int(x.size(0)), columns = int(x.size(1));
    if (!rows) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    const bool prepared_input = rows == 32 && x.is_contiguous()
        && reinterpret_cast<uintptr_t>(x.data_ptr()) % 512 == 0;
    auto padded = prepared_input ? x : at::empty({32, columns}, x.options());
    if (!prepared_input) {
        dense_pad32<<<std::min<int64_t>((int64_t(32) * columns + 255) / 256, 65535), 256, 0, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(padded.data_ptr<at::Half>()), rows, columns, x.stride(0), x.stride(1));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    const bool f32 = output.scalar_type() == at::kFloat;
    const bool hc_f32 = f32 && columns == 10240 && weights.size() == 2
        && weights[0].size(0) == 320 && weights[1].size(0) == 4;
    const bool qkv_half = !f32 && columns == 2560 && weights.size() == 5
        && weights[0].size(0) == 1536 && weights[1].size(0) == 256 && weights[2].size(0) == 256
        && weights[3].size(0) == 512 && weights[4].size(0) == 128;
    bool direct = rows == 32 && output.is_contiguous()
        && reinterpret_cast<uintptr_t>(output.data_ptr()) % 512 == 0 && (hc_f32 || qkv_half);
    if (direct) {
        direct = !tensorfold::overlaps(output, padded);
        for (const auto& weight : weights) direct = direct && !tensorfold::overlaps(output, weight);
    }
    if (direct) {
        for (size_t part = 0; part < weights.size(); ++part) {
            auto target = output.narrow(1, sources.starts[part], weights[part].size(0));
            if (f32) at::mm_out(target, padded, weights[part].t(), at::kFloat);
            else at::mm_out(target, padded, weights[part].t());
        }
        return;
    }
    std::vector<at::Tensor> values;
    values.reserve(weights.size());
    for (size_t part = 0; part < weights.size(); ++part) {
        // Keep the same N, K, M32, output dtype and contiguous result as the
        // reference part call. Only its repeated input staging is shared.
        values.push_back(f32 ? at::mm(padded, weights[part].t(), at::kFloat) : at::mm(padded, weights[part].t()));
        sources.data[part] = values.back().data_ptr();
    }
    const int blocks = int(std::min<int64_t>((int64_t(rows) * total + 255) / 256, 65535));
    if (f32)
        concat_copy32<true><<<blocks, 256, 0, stream>>>(sources, output.data_ptr(), rows, int(total), output.stride(0));
    else
        concat_copy32<false><<<blocks, 256, 0, stream>>>(sources, output.data_ptr(), rows, int(total), output.stride(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {
constexpr int WMMA_M = 32, WMMA_N = 64, WMMA_K = 64, WMMA_THREADS = 128;
constexpr int WMMA_SHARED = (WMMA_M + WMMA_N) * WMMA_K * sizeof(__half);

template <bool OutF32>
__global__ __launch_bounds__(WMMA_THREADS, 8)
void dense_wmma32_kernel(const __half* x, const __half* weight, float* partials, void* out,
                           int rows, int outputs, int columns, int part_columns, int parts,
                           int64_t x_row_stride, int64_t x_column_stride, int64_t out_stride) {
    namespace wmma = nvcuda::wmma;
    extern __shared__ __align__(32) unsigned char memory[];
    auto* a = reinterpret_cast<__half*>(memory);
    auto* b = a + WMMA_M * WMMA_K;
    auto* c = reinterpret_cast<float*>(memory); // Reuse staging after every MMA has completed.
    const int warp = threadIdx.x / 32, nbase = blockIdx.x * WMMA_N, part = blockIdx.y;
    const int begin = part * part_columns, end = begin + min(part_columns, columns - begin);
    const bool live_warp = nbase + warp * 16 < outputs;
    const bool vector_x = x_column_stride == 1 && x_row_stride % 8 == 0
                           && (reinterpret_cast<uintptr_t>(x) & 15) == 0;
    const bool vector_w = (reinterpret_cast<uintptr_t>(weight) & 15) == 0;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> low, high;
    wmma::fill_fragment(low, 0.0f);
    wmma::fill_fragment(high, 0.0f);
    for (unsigned chunk = begin; chunk < unsigned(end); chunk += WMMA_K) {
        if (vector_x) {
            for (int at = threadIdx.x; at < WMMA_M * WMMA_K / 8; at += WMMA_THREADS) {
                const int r = at / (WMMA_K / 8);
                const unsigned k = chunk + at % (WMMA_K / 8) * 8;
                reinterpret_cast<uint4*>(a)[at] = r < rows && k < end
                    ? *reinterpret_cast<const uint4*>(x + int64_t(r) * x_row_stride + k)
                    : make_uint4(0, 0, 0, 0);
            }
        } else {
            for (int at = threadIdx.x; at < WMMA_M * WMMA_K; at += WMMA_THREADS) {
                const int r = at / WMMA_K;
                const unsigned k = chunk + at % WMMA_K;
                a[at] = r < rows && k < end ? x[int64_t(r) * x_row_stride + int64_t(k) * x_column_stride]
                                          : __float2half_rn(0.0f);
            }
        }
        if (vector_w) {
            for (int at = threadIdx.x; at < WMMA_N * WMMA_K / 8; at += WMMA_THREADS) {
                const int n = nbase + at / (WMMA_K / 8);
                const unsigned k = chunk + at % (WMMA_K / 8) * 8;
                reinterpret_cast<uint4*>(b)[at] = n < outputs && k < end
                    ? *reinterpret_cast<const uint4*>(weight + int64_t(n) * columns + k)
                    : make_uint4(0, 0, 0, 0);
            }
        } else {
            for (int at = threadIdx.x; at < WMMA_N * WMMA_K; at += WMMA_THREADS) {
                const int n = nbase + at / WMMA_K;
                const unsigned k = chunk + at % WMMA_K;
                b[at] = n < outputs && k < end ? weight[int64_t(n) * columns + k] : __float2half_rn(0.0f);
            }
        }
        __syncthreads();
        if (live_warp) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> alow, ahigh;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bb;
            for (int k = 0; k < WMMA_K && chunk + k < end; k += 16) {
                wmma::load_matrix_sync(alow, a + k, WMMA_K);
                wmma::load_matrix_sync(ahigh, a + 16 * WMMA_K + k, WMMA_K);
                wmma::load_matrix_sync(bb, b + warp * 16 * WMMA_K + k, WMMA_K);
                wmma::mma_sync(low, alow, bb, low);
                wmma::mma_sync(high, ahigh, bb, high);
            }
        }
        __syncthreads();
    }
    if (live_warp) {
        wmma::store_matrix_sync(c + warp * 16, low, WMMA_N, wmma::mem_row_major);
        wmma::store_matrix_sync(c + 16 * WMMA_N + warp * 16, high, WMMA_N, wmma::mem_row_major);
    }
    __syncthreads();
    for (int at = threadIdx.x; at < WMMA_M * WMMA_N; at += WMMA_THREADS) {
        const int r = at / WMMA_N, n = nbase + at % WMMA_N;
        if (r < rows && n < outputs) {
            const float value = c[at];
            if (parts > 1) partials[(int64_t(part) * rows + r) * outputs + n] = value;
            else {
                const int64_t dest = int64_t(r) * out_stride + n;
                if constexpr (OutF32) static_cast<float*>(out)[dest] = value;
                else static_cast<__half*>(out)[dest] = __float2half_rn(value);
            }
        }
    }
}

template <bool OutF32>
__global__ void dense_wmma_reduce(const float* partials, void* out, int rows, int outputs,
                                  int parts, int64_t out_stride) {
    const int64_t count = int64_t(rows) * outputs;
    for (int64_t at = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; at < count;
         at += int64_t(gridDim.x) * blockDim.x) {
        float value = partials[at];
        for (int part = 1; part < parts; ++part) value += partials[int64_t(part) * count + at];
        const int64_t dest = (at / outputs) * out_stride + at % outputs;
        if constexpr (OutF32) static_cast<float*>(out)[dest] = value;
        else static_cast<__half*>(out)[dest] = __float2half_rn(value);
    }
}
} // namespace

void gguf_dense_wmma(const at::Tensor& x, const at::Tensor& weight, at::Tensor output) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.dim() == 2
                && x.size(0) <= WMMA_M && x.size(1) > 0 && x.size(1) % 16 == 0
                && x.size(1) <= std::numeric_limits<int>::max()
                && x.stride(0) >= 0 && x.stride(1) >= 0,
                "dense WMMA input requires at most32 FP16 CUDA rows and a16-aligned input width");
    const auto device = x.device();
    const auto* prop = at::cuda::getDeviceProperties(x.get_device());
    TORCH_CHECK(prop->major == 7 && prop->minor == 0, "dense WMMA provider requires SM70");
    operand(weight, at::kHalf, device, "dense WMMA weight");
    TORCH_CHECK(weight.dim() == 2 && weight.size(1) == x.size(1)
                && weight.size(0) > 0 && weight.size(0) <= 16384,
                "dense WMMA targets non-head projection weights with at most16384outputs");
    TORCH_CHECK(output.device() == device && output.dim() == 2 && output.size(0) == x.size(0)
                && output.size(1) == weight.size(0) && output.stride(1) == 1 && output.stride(0) >= output.size(1)
                && (output.scalar_type() == at::kHalf || output.scalar_type() == at::kFloat),
                "dense WMMA output must be FP16/FP32 rows with contiguous columns and disjoint row strides");
    c10::cuda::CUDAGuard guard(device);
    const int rows = int(x.size(0)), outputs = int(weight.size(0)), columns = int(x.size(1));
    if (!rows) return;
    const int ntiles = (outputs + WMMA_N - 1) / WMMA_N;
    // Partition depends only on matrix geometry and the selected GPU. R and
    // capture bucket never change the FP32 accumulation or reduction order.
    const int desired = std::min(128, std::max(1, (prop->multiProcessorCount * 8 + ntiles - 1) / ntiles));
    const int part_columns = int((int64_t(columns) + desired * 16 - 1) / (desired * 16)) * 16;
    const int parts = 1 + (columns - 1) / part_columns;
    at::Tensor partials;
    float* scratch = nullptr;
    if (parts > 1) {
        partials = at::empty({parts, rows, outputs}, x.options().dtype(at::kFloat));
        scratch = partials.data_ptr<float>();
    }
    const dim3 grid(ntiles, parts);
    auto stream = at::cuda::getCurrentCUDAStream();
    const auto* input = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
    const auto* w = reinterpret_cast<const __half*>(weight.data_ptr<at::Half>());
    if (output.scalar_type() == at::kFloat)
        dense_wmma32_kernel<true><<<grid, WMMA_THREADS, WMMA_SHARED, stream>>>(input, w, scratch, output.data_ptr(),
            rows, outputs, columns, part_columns, parts, x.stride(0), x.stride(1), output.stride(0));
    else
        dense_wmma32_kernel<false><<<grid, WMMA_THREADS, WMMA_SHARED, stream>>>(input, w, scratch, output.data_ptr(),
            rows, outputs, columns, part_columns, parts, x.stride(0), x.stride(1), output.stride(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (parts > 1) {
        const int blocks = int(std::min<int64_t>((int64_t(rows) * outputs + 255) / 256, 65535));
        if (output.scalar_type() == at::kFloat)
            dense_wmma_reduce<true><<<blocks, 256, 0, stream>>>(scratch, output.data_ptr(), rows, outputs, parts,
                output.stride(0));
        else
            dense_wmma_reduce<false><<<blocks, 256, 0, stream>>>(scratch, output.data_ptr(), rows, outputs, parts,
                output.stride(0));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

// Match the pinned original cuBLAS GEMVX6 K80 reduction: four contiguous
// twenty-element FMA chains, then (p0+p2)+(p1+p3), independent of batch rows.
namespace {
template <typename Index, bool F32, int Kind = -1>
__global__ void expert_down80_exact(const __half* activation, const Index* picks, PackedView down,
                                  const __half* shared, int dims, int experts, void* out) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int part = lane % 4;
    const unsigned output = blockIdx.x * EXPERT_WARPS * 8 + warp * 8 + lane / 4;
    const int pair = blockIdx.y;
    const auto expert = picks[pair];
    const unsigned mask = __ballot_sync(0xffffffff, output < dims);
    if (output >= dims) return;
    float sum = 0.0f;
    if (expert >= 0 && expert <= experts) {
        const __half* input = activation + int64_t(pair) * 80;
        const int64_t row = int64_t(expert) * dims + output;
        #pragma unroll
        for (int i = 0; i < 20; ++i) {
            const int k = part * 20 + i;
            const float w = expert == experts ? __half2float(shared[int64_t(output) * 80 + k])
                                             : coefficient<Kind>(down, row, k);
            sum = fmaf(__half2float(input[k]), w, sum);
        }
        sum += __shfl_down_sync(mask, sum, 2, 4);
        sum += __shfl_down_sync(mask, sum, 1, 4);
        // Preserve the oracle alpha=1, beta=0 output epilogue, including zero.
        sum = fmaf(1.0f, sum, 0.0f) + 0.0f;
    } else {
        sum = __int_as_float(0x7fffffff);
    }
    if (part == 0) {
        const int64_t at = int64_t(pair) * dims + output;
        if constexpr (F32) static_cast<float*>(out)[at] = sum;
        else static_cast<__half*>(out)[at] = __float2half_rn(sum);
    }
}

// Qualified separately against the pinned original B8 K160 CUTLASS BMM.
// One valid query column and fifteen zero columns, ten sequential K16 MMAs.
template <typename Index, bool F32, int Kind = -1>
__global__ void expert_down160_exact(const __half* activation, const Index* picks, PackedView down,
                                   const __half* shared, int dims, int experts, void* output) {
    namespace wmma = nvcuda::wmma;
    __shared__ __align__(32) __half aa[256], bb[256];
    __shared__ __align__(32) float cc[256];
    const int first = blockIdx.x*16, pair = blockIdx.y;
    const auto expert = picks[pair];
    const bool valid = expert >= 0 && expert <= experts;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accum;
    wmma::fill_fragment(accum, 0.0f);
    #pragma unroll
    for (int base = 0; base < 160; base += 16) {
        for (int i = threadIdx.x; i < 256; i += 32) {
            const int row = i/16, k = base+i%16;
            float w = 0.0f;
            if (valid) {
                w = expert == experts ? __half2float(shared[int64_t(first+row)*160+k])
                                      : coefficient<Kind>(down, int64_t(expert)*dims+first+row, k);
            }
            aa[i] = __float2half_rn(w);
            bb[i] = valid && row == 0 ? activation[int64_t(pair)*160+k] : __float2half_rn(0.0f);
        }
        __syncwarp();
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
        wmma::load_matrix_sync(a, aa, 16);
        wmma::load_matrix_sync(b, bb, 16);
        wmma::mma_sync(accum, a, b, accum);
        __syncwarp();
    }
    wmma::store_matrix_sync(cc, accum, 16, wmma::mem_row_major);
    __syncwarp();
    if (threadIdx.x < 16) {
        float value = valid ? fmaf(1.0f, cc[threadIdx.x*16], 0.0f)+0.0f : __int_as_float(0x7fffffff);
        const int64_t at = int64_t(pair)*dims+first+threadIdx.x;
        if constexpr (F32) static_cast<float*>(output)[at] = value;
        else static_cast<__half*>(output)[at] = __float2half_rn(value);
    }
}

void run_exact_down(const at::Tensor& activation, const at::Tensor& picks, PackedView view,
                const at::Tensor& shared, at::Tensor& result, int dims, int experts) {
    dim3 grid(1 + (dims - 1) / (EXPERT_WARPS * 8), picks.numel());
    auto stream = at::cuda::getCurrentCUDAStream(activation.get_device());
    auto a = reinterpret_cast<const __half*>(activation.data_ptr());
    auto s = reinterpret_cast<const __half*>(shared.data_ptr());
    if (activation.size(1) == 160) {
        dim3 tensor_grid(dims/16, picks.numel());
        #define LAUNCH_DOWN160(INDEX, FLOAT, KIND) \
            expert_down160_exact<INDEX, FLOAT, KIND><<<tensor_grid, 32, 0, stream>>>( \
                a, picks.data_ptr<INDEX>(), view, s, dims, experts, result.data_ptr())
        if (picks.scalar_type() == at::kLong) {
            if (result.scalar_type() == at::kFloat) {
                if (view.kind == 7) LAUNCH_DOWN160(int64_t, true, 7);
                else LAUNCH_DOWN160(int64_t, true, -1);
            } else {
                if (view.kind == 7) LAUNCH_DOWN160(int64_t, false, 7);
                else LAUNCH_DOWN160(int64_t, false, -1);
            }
        } else {
            if (result.scalar_type() == at::kFloat) {
                if (view.kind == 7) LAUNCH_DOWN160(int32_t, true, 7);
                else LAUNCH_DOWN160(int32_t, true, -1);
            } else {
                if (view.kind == 7) LAUNCH_DOWN160(int32_t, false, 7);
                else LAUNCH_DOWN160(int32_t, false, -1);
            }
        }
        #undef LAUNCH_DOWN160
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }
    #define LAUNCH_DOWN80(INDEX, FLOAT, KIND) \
        expert_down80_exact<INDEX, FLOAT, KIND><<<grid, EXPERT_WARPS * 32, 0, stream>>>( \
            a, picks.data_ptr<INDEX>(), view, s, dims, experts, result.data_ptr())
    if (picks.scalar_type() == at::kLong) {
        if (result.scalar_type() == at::kFloat) {
            if (view.kind == 7) LAUNCH_DOWN80(int64_t, true, 7);
            else LAUNCH_DOWN80(int64_t, true, -1);
        } else {
            if (view.kind == 7) LAUNCH_DOWN80(int64_t, false, 7);
            else LAUNCH_DOWN80(int64_t, false, -1);
        }
    } else {
        if (result.scalar_type() == at::kFloat) {
            if (view.kind == 7) LAUNCH_DOWN80(int32_t, true, 7);
            else LAUNCH_DOWN80(int32_t, true, -1);
        } else {
            if (view.kind == 7) LAUNCH_DOWN80(int32_t, false, 7);
            else LAUNCH_DOWN80(int32_t, false, -1);
        }
    }
    #undef LAUNCH_DOWN80
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}

void gguf_experts_exact(const at::Tensor& x, const at::Tensor& picks,
                  const at::Tensor& gate, int64_t gate_kind, int64_t gate_offset,
                  const at::Tensor& up, int64_t up_kind, int64_t up_offset,
                  const at::Tensor& down, int64_t down_kind, int64_t down_offset,
                  const at::Tensor& shared_gate, const at::Tensor& shared_up,
                  const at::Tensor& shared_down, at::Tensor output) {
    TORCH_CHECK(shared_gate.dim() == 2 && (shared_gate.size(0) == 80 || shared_gate.size(0) == 160) && shared_gate.size(1) == 2560,
                "recovered packed expert down projection requires the qualified D2560/K80 or K160 geometry");
    gguf_experts_impl(x, picks, gate, gate_kind, gate_offset, up, up_kind, up_offset,
                      down, down_kind, down_offset, shared_gate, shared_up, shared_down,
                      output, false, true, true, true);
}

// The owned-bank candidate reuses the packed operator's arithmetic. Mapping
// never publishes the local shared sentinel: all non-owned IDs become -1.
namespace {
template <typename Index>
__global__ void owned_routes(const Index* global, int64_t* local, int64_t count,
                             int first, int experts) {
    for (int64_t at = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; at < count;
         at += int64_t(gridDim.x) * blockDim.x) {
        const auto id = global[at];
        local[at] = id >= first && id < int64_t(first) + experts ? int64_t(id) - first : -1;
    }
}
template <typename Index, bool F32>
__global__ void publish_owned(const Index* global, const int64_t* local, const float* shared,
                              void* output, int64_t count, int dims, int slots, int global_experts) {
    for (int64_t at = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; at < count;
         at += int64_t(gridDim.x) * blockDim.x) {
        const int64_t pair = at / dims;
        const auto id = global[pair];
        float value;
        if (id == global_experts) value = shared[(pair / slots) * dims + at % dims];
        else if (id < 0 || id > global_experts) value = __int_as_float(0x7fffffff);
        else if (local[pair] < 0) value = 0.0f;
        else continue; // Owned rows already have the reused down kernel's storage rounding.
        if constexpr (F32) static_cast<float*>(output)[at] = value;
        else static_cast<__half*>(output)[at] = __float2half_rn(value);
    }
}

// Private supplied-GU activation ABI reserves shared slot10 as immutable row
// status (+0 valid, any other representation poisoned). Old shared/down/publish
// arithmetic is unchanged; publish the failure even for all-foreign rows.
template <bool F32>
__global__ void publish_activation_status(const __half* activation, void* output, int64_t count) {
    for (int64_t at=int64_t(blockIdx.x)*blockDim.x+threadIdx.x; at<count;
         at+=int64_t(gridDim.x)*blockDim.x) {
        const int64_t row=at/(11*2560);
        if (__half_as_ushort(activation[(row*11+10)*160]) != 0) {
            const float value=__int_as_float(0x7fffffff);
            if constexpr(F32) static_cast<float*>(output)[at]=value;
            else static_cast<__half*>(output)[at]=__float2half_rn(value);
        }
    }
}

// GPU-only bounded counting/grouping. Atomic arrival order chooses columns,
// never slot order or any arithmetic reduction; every pair scatters back alone.
struct OwnedGrouping {
    int *counts, *offsets, *ordinal, *pairs, *tiles, *total;
    int max_tiles;
};

template <typename Index>
__global__ void group_owned_routes(const Index* global, int64_t* local, int count,
                                    int first, int experts, OwnedGrouping group) {
    for (int pair = blockIdx.x * blockDim.x + threadIdx.x; pair < count; pair += gridDim.x * blockDim.x) {
        const auto id = global[pair];
        const int expert = id >= first && id < int64_t(first) + experts ? int(id - first) : -1;
        local[pair] = expert;
        if (expert >= 0) group.ordinal[pair] = atomicAdd(group.counts + expert, 1);
    }
}

// One CTA of GROUP_PREFIX_THREADS (>= experts) threads: block-wide exclusive scans of the counts and
// of their 16-pair tile counts write the same offsets/tiles/total as an expert-ordered serial walk.
constexpr int GROUP_PREFIX_THREADS = 512;
__global__ __launch_bounds__(GROUP_PREFIX_THREADS) void group_owned_prefix(int experts, OwnedGrouping group) {
    __shared__ int warp_counts[GROUP_PREFIX_THREADS / 32], warp_tiles[GROUP_PREFIX_THREADS / 32];
    const int expert = threadIdx.x, lane = expert % 32, warp = expert / 32;
    const int count = expert < experts ? group.counts[expert] : 0, tiles = (count + 15) / 16;
    int counts = count, tile_sum = tiles;  // inclusive warp scans
    for (int delta = 1; delta < 32; delta *= 2) {
        const int c = __shfl_up_sync(0xffffffff, counts, delta), t = __shfl_up_sync(0xffffffff, tile_sum, delta);
        if (lane >= delta) { counts += c; tile_sum += t; }
    }
    if (lane == 31) { warp_counts[warp] = counts; warp_tiles[warp] = tile_sum; }
    __syncthreads();
    for (int w = 0; w < warp; ++w) { counts += warp_counts[w]; tile_sum += warp_tiles[w]; }
    if (expert >= experts) return;
    const int offset = counts - count, first = tile_sum - tiles;
    group.offsets[expert] = offset;
    for (int i = 0; i < tiles; ++i) {
        group.tiles[3 * (first + i)] = expert;
        group.tiles[3 * (first + i) + 1] = offset + 16 * i;
        group.tiles[3 * (first + i) + 2] = min(16, count - 16 * i);
    }
    if (expert == experts - 1) {
        group.offsets[experts] = counts;
        *group.total = tile_sum;
    }
}

__global__ void group_owned_scatter(const int64_t* local, int count, OwnedGrouping group) {
    for (int pair = blockIdx.x * blockDim.x + threadIdx.x; pair < count; pair += gridDim.x * blockDim.x) {
        const int expert = int(local[pair]);
        if (expert >= 0) group.pairs[group.offsets[expert] + group.ordinal[pair]] = pair;
    }
}

// Decode-sized grouping (pairs <= GROUP_PREFIX_THREADS) in one CTA: the counts reset, routes, prefix,
// scatter and (optionally) the non-owned activation initialization of the separate kernels above, with
// the same outputs (ordinals still come from atomic arrival order, which never reaches arithmetic).
template <typename Index>
__global__ __launch_bounds__(GROUP_PREFIX_THREADS) void group_owned_decode(
        const Index* global, int64_t* local, int count, int first, int experts, OwnedGrouping group,
        __half* unowned_activation) {
    __shared__ int warp_counts[GROUP_PREFIX_THREADS / 32], warp_tiles[GROUP_PREFIX_THREADS / 32];
    const int t = threadIdx.x, lane = t % 32, warp = t / 32;
    if (t < experts) group.counts[t] = 0;
    __syncthreads();
    int expert = -1;
    if (t < count) {
        const auto id = global[t];
        expert = id >= first && id < int64_t(first) + experts ? int(id - first) : -1;
        local[t] = expert;
        if (expert >= 0) group.ordinal[t] = atomicAdd(group.counts + expert, 1);
    }
    __syncthreads();
    const int n = t < experts ? group.counts[t] : 0, tiles = (n + 15) / 16;
    int counts = n, tile_sum = tiles;
    for (int delta = 1; delta < 32; delta *= 2) {
        const int c = __shfl_up_sync(0xffffffff, counts, delta), s = __shfl_up_sync(0xffffffff, tile_sum, delta);
        if (lane >= delta) { counts += c; tile_sum += s; }
    }
    if (lane == 31) { warp_counts[warp] = counts; warp_tiles[warp] = tile_sum; }
    __syncthreads();
    for (int w = 0; w < warp; ++w) { counts += warp_counts[w]; tile_sum += warp_tiles[w]; }
    if (t < experts) {
        const int offset = counts - n, base = tile_sum - tiles;
        group.offsets[t] = offset;
        for (int i = 0; i < tiles; ++i) {
            group.tiles[3 * (base + i)] = t;
            group.tiles[3 * (base + i) + 1] = offset + 16 * i;
            group.tiles[3 * (base + i) + 2] = min(16, n - 16 * i);
        }
        if (t == experts - 1) {
            group.offsets[experts] = counts;
            *group.total = tile_sum;
        }
    }
    __syncthreads();
    if (expert >= 0) group.pairs[group.offsets[expert] + group.ordinal[t]] = t;
    if (unowned_activation) {
        __syncthreads();   // local[] of every pair is published above
        for (int at = t; at < count * 160; at += GROUP_PREFIX_THREADS)
            if (local[at / 160] < 0)
                unowned_activation[at] = __float2half_rn(__int_as_float(0x7fffffff));
    }
}

// The incumbent writes half NaN for every non-owned local slot. Preserve that
// scratch contract without reading any bank, including on changing graph routes.
__global__ void initialize_unowned_gate_up(const int64_t* local, __half* activation, int pairs) {
    for (int at = blockIdx.x * blockDim.x + threadIdx.x; at < pairs * 160; at += gridDim.x * blockDim.x)
        if (local[at / 160] < 0)
            activation[at] = __float2half_rn(__int_as_float(0x7fffffff));
}

// Prefill-sized experts: eight consecutive coefficients of one quant block are decoded per
// thread with the exact value() expressions (then the coefficient() FP16 boundary), once per
// <=64-pair group of an expert, and multiplied on tensor cores with FP32 accumulators.
// Coefficients are FP16-rounded as in coefficient(); only the reduction order is the hardware's.
constexpr int PREFILL_PAIRS = 64, PREFILL_THREADS = 256, PREFILL_LD = 72;

__device__ __forceinline__ uint32_t pack_half2(float low, float high) {
    return uint32_t(__half_as_ushort(__float2half_rn(low)))
        | (uint32_t(__half_as_ushort(__float2half_rn(high))) << 16);
}

__device__ __forceinline__ uint4 pack8(const float (&v)[8]) {
    return make_uint4(pack_half2(v[0], v[1]), pack_half2(v[2], v[3]),
                      pack_half2(v[4], v[5]), pack_half2(v[6], v[7]));
}

// Packs value(p, Kind, i + j), j < 8, where i % 8 == 0 and all eight lie in block p.
template <int Kind>
__device__ __forceinline__ uint4 decode8_block(const uint8_t* p, int i) {
    static_assert(Kind == 8 || Kind == 18 || Kind == 20 || Kind == 21 || Kind == 22 || Kind == 23 || Kind == 42,
                  "specialized eight-coefficient decode");
    float v[8];
    if constexpr (Kind == 8) {
        const float d = half_value(p);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = d * float(int8_t(p[2 + i + j]));
    } else if constexpr (Kind == 20) {
        const float d = half_value(p);
        const int shift = 4 * (i / 16);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = d * float(tf_kvalues_iq4nl[(p[2 + i % 16 + j] >> shift) & 15]);
    } else if constexpr (Kind == 42) {
        const float d = half_value(p);
        const uint32_t bits = uint32_t(p[2 + i / 4]) | (uint32_t(p[3 + i / 4]) << 8);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = float(int((bits >> (2 * j)) & 3) - 1) * d;
    } else if constexpr (Kind == 23) {
        const int group = i / 32, element = i % 32;
        const int scale = int(((p[4 + group / 2] >> (4 * (group % 2))) & 15)
                             | (((u16(p + 2) >> (2 * group)) & 3) << 4)) - 32;
        const float d = half_value(p) * float(scale);
        const int shift = 4 * (element / 16);
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            v[j] = d * float(tf_kvalues_iq4nl[(p[8 + group * 16 + element % 16 + j] >> shift) & 15]);
    } else if constexpr (Kind == 22) {
        const int group = i / 32, sub = (i % 32) / 8;
        const int code = p[2 + group * 4 + sub] | ((uint32_t(p[66 + group]) << (8 - 2 * sub)) & 0x300);
        const uint64_t grid = tf_iq2s_grid[code];
        const int scale = (p[74 + group] >> (4 * (sub / 2))) & 15;
        const float d = (half_value(p) * (0.5f + float(scale))) * 0.25f;
        const int signs = p[34 + group * 4 + sub];
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            v[j] = (d * float(int((grid >> (8 * j)) & 255))) * ((signs & (1 << j)) ? -1.0f : 1.0f);
    } else if constexpr (Kind == 18) {
        const int group = i / 32, sub = (i % 32) / 8;
        const uint32_t auxiliary = u32(p + 66 + group * 4);
        const int signs = tf_ksigns_iq2xs[(auxiliary >> (7 * sub)) & 127];
        const uint32_t grid0 = tf_iq3xxs_grid[p[2 + group * 8 + sub * 2]];
        const uint32_t grid1 = tf_iq3xxs_grid[p[3 + group * 8 + sub * 2]];
        const float d = (half_value(p) * (0.5f + float(auxiliary >> 28))) * 0.5f;
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            v[j] = (d * float(((j < 4 ? grid0 : grid1) >> (8 * (j % 4))) & 255))
                * ((signs & (1 << j)) ? -1.0f : 1.0f);
    } else if constexpr (Kind == 21) {
        const int group = i / 32, sub = (i % 32) / 8;
        const uint32_t high = p[66 + group];
        const uint32_t grid0 = tf_iq3s_grid[p[2 + group * 8 + sub * 2] | ((high << (8 - 2 * sub)) & 256)];
        const uint32_t grid1 = tf_iq3s_grid[p[3 + group * 8 + sub * 2] | ((high << (7 - 2 * sub)) & 256)];
        const int scale = (p[106 + group / 2] >> (4 * (group % 2))) & 15;
        const float d = half_value(p) * float(1 + 2 * scale);
        const int signs = p[74 + group * 4 + sub];
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            v[j] = (d * float(((j < 4 ? grid0 : grid1) >> (8 * (j % 4))) & 255))
                * ((signs & (1 << j)) ? -1.0f : 1.0f);
    }
    return pack8(v);
}

template <int Kind, int Block, int Bytes>
__device__ __forceinline__ uint4 decode8_kind(const uint8_t* row, int col) {
    return decode8_block<Kind>(row + int64_t(col / Block) * Bytes, col % Block);
}

// Remaining kinds and unaligned views: out of line, keeping the specialized loops compact.
__device__ __noinline__ uint4 decode8_generic(PackedView bank, int64_t row, int column) {
    float v[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = coefficient(bank, row, column + j);
    return pack8(v);
}

// Coefficients (row, column .. column + 7) of a bank as eight FP16 values, bitwise equal to
// coefficient(bank, row, column + j). The kind switch is uniform across the CTA.
__device__ __forceinline__ uint4 decode8(const PackedView& bank, int64_t row, int column) {
    const int col = column + bank.offset;
    const uint8_t* r = bank.data + row * bank.stride;
    if (col % 8 == 0) {
        switch (bank.kind) {
            case 8: return decode8_kind<8, 32, 34>(r, col);
            case 18: return decode8_kind<18, 256, 98>(r, col);
            case 20: return decode8_kind<20, 32, 18>(r, col);
            case 21: return decode8_kind<21, 256, 110>(r, col);
            case 22: return decode8_kind<22, 256, 82>(r, col);
            case 23: return decode8_kind<23, 256, 136>(r, col);
            case 42: return decode8_kind<42, 64, 18>(r, col);
        }
    }
    return decode8_generic(bank, row, column);
}

// decode8 with the kind resolved at compile time (Kind < 0: the runtime switch). Same expressions,
// so bitwise equal; prefill kernels instantiate one body per admitted kind instead of all seven.
template <int Kind>
__device__ __forceinline__ uint4 decode8_fixed(const PackedView& bank, int64_t row, int column) {
    if constexpr (Kind < 0) {
        return decode8(bank, row, column);
    } else {
        constexpr int Block = Kind == 8 || Kind == 20 ? 32 : Kind == 42 ? 64 : 256;
        constexpr int Bytes = Kind == 8 ? 34 : Kind == 18 ? 98 : Kind == 20 ? 18 : Kind == 21 ? 110
                            : Kind == 22 ? 82 : Kind == 23 ? 136 : 18;
        const int col = column + bank.offset;
        if (col % 8 == 0) return decode8_kind<Kind, Block, Bytes>(bank.data + row * bank.stride, col);
        return decode8_generic(bank, row, column);
    }
}

// Formats without the integer decode below keep the reference scalar/WMMA grouped kernels.
// Each row retains the incumbent's lane-strided FMA sequence and shuffle tree.
// Only coefficient decoding is shared: no arithmetic reduction spans queries.
template <int Rows>
__global__ void expert_gate_up_grouped(const __half* x, PackedView gate, PackedView up,
                                     int slots, OwnedGrouping group, __half* activation) {
    const int tile = blockIdx.y;
    if (tile >= *group.total) return;
    const int expert = group.tiles[3 * tile], begin = group.tiles[3 * tile + 1];
    const int count = group.tiles[3 * tile + 2];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int output = blockIdx.x * EXPERT_WARPS + warp;
    const int64_t source = int64_t(expert) * 160 + output;
    for (int start = 0; start < count; start += Rows) {
        int pairs[Rows];
        float g[Rows] = {}, u[Rows] = {};
        #pragma unroll
        for (int row = 0; row < Rows; ++row)
            if (start + row < count) pairs[row] = group.pairs[begin + start + row];
        for (int k = lane; k < 2560; k += 32) {
            const float gv = coefficient(gate, source, k);
            const float uv = coefficient(up, source, k);
            #pragma unroll
            for (int row = 0; row < Rows; ++row) {
                if (start + row < count) {
                    const float xv = __half2float(x[int64_t(pairs[row] / slots) * 2560 + k]);
                    g[row] = fmaf(xv, gv, g[row]);
                    u[row] = fmaf(xv, uv, u[row]);
                }
            }
        }
        #pragma unroll
        for (int row = 0; row < Rows; ++row) {
            if (start + row < count) {
                for (int offset = 16; offset; offset /= 2) {
                    g[row] += __shfl_down_sync(0xffffffff, g[row], offset);
                    u[row] += __shfl_down_sync(0xffffffff, u[row], offset);
                }
                if (lane == 0) {
                    const float result = (g[row] / (1.0f + expf(-g[row]))) * u[row];
                    activation[int64_t(pairs[row]) * 160 + output] = __float2half_rn(result);
                }
            }
        }
    }
}

template <bool F32, int Kind = -1>
__global__ void expert_down160_grouped(const __half* activation, PackedView down,
                                      int dims, OwnedGrouping group, void* output) {
    namespace wmma = nvcuda::wmma;
    const int tile = blockIdx.y;
    if (tile >= *group.total) return; // CTA-uniform, before any warp participation.
    const int expert = group.tiles[3 * tile], begin = group.tiles[3 * tile + 1];
    const int count = group.tiles[3 * tile + 2], first = blockIdx.x * 16;
    __shared__ __align__(32) __half aa[256], bb[256];
    __shared__ __align__(32) float cc[256];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accum;
    wmma::fill_fragment(accum, 0.0f);
    #pragma unroll
    for (int base = 0; base < 160; base += 16) {
        for (int i = threadIdx.x; i < 256; i += 32) {
            const int row = i / 16, k = i % 16;
            aa[i] = __float2half_rn(coefficient<Kind>(down, int64_t(expert) * dims + first + row, base + k));
            // B is column-major: each independent original pair owns one column.
            bb[i] = row < count ? activation[int64_t(group.pairs[begin + row]) * 160 + base + k]
                                : __float2half_rn(0.0f);
        }
        __syncwarp();
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
        wmma::load_matrix_sync(a, aa, 16);
        wmma::load_matrix_sync(b, bb, 16);
        wmma::mma_sync(accum, a, b, accum);
        __syncwarp();
    }
    wmma::store_matrix_sync(cc, accum, 16, wmma::mem_row_major);
    __syncwarp();
    for (int i = threadIdx.x; i < 256; i += 32) {
        const int row = i / 16, col = i % 16;
        if (col < count) {
            const float value = fmaf(1.0f, cc[i], 0.0f) + 0.0f;
            const int64_t at = int64_t(group.pairs[begin + col]) * dims + first + row;
            if constexpr (F32) static_cast<float*>(output)[at] = value;
            else static_cast<__half*>(output)[at] = __float2half_rn(value);
        }
    }
}

// Decode-size (<=32 rows) grouped experts: one 16-pair tile of one expert per CTA row (blockIdx.y).
// Integer path (decode-size only; Q2_0 down excepted, see down_exact): every eight coefficients of the supported formats are exactly
// d * q[j], with d the format's FP32 scale expression from value() and q[j] a signed 8-bit integer.
// Inputs are quantized per 32-element block to int16 (x ~= s * xq, s = amax / 32767, |error| <=
// amax / 65534) and split into a signed high and an unsigned low byte, so each eight-coefficient
// partial sum S = sum q[j] * xq[j] is formed exactly with dp4a and contributes (d * s) * float(S) to an
// FP32 accumulator. Versus the reference (FP16-rounded d*q times FP16 x, FP32 sums) this removes the
// coefficient rounding and adds the much smaller input quantization; non-finite inputs stay non-finite.
constexpr int DECODE_PAIRS = 16, GU_WARPS = 8, GU_SLICE = 4, GU_RAW = 2720;
// Grouped owned calls of up to this many rows take the decode gate/up and down kernels (16-pair tiles,
// int16-quantized activations); larger ones take the tensor-core prefill tiles. Whole owned-expert call,
// V100, IQ3_S gate/up + IQ4_NL down, skewed routing: 33/40/48/64 rows 568/609/698/774 us (prefill) vs
// 359/428/439/547 us (decode), max error vs FP64 4.5e-4 vs 2.4e-4 (the decode kernels' own level at
// 17-32 rows, 2.2-2.8e-4).
constexpr int DECODE_MAX_ROWS = 64;
constexpr int DOWN_WARPS = 8;

__host__ __device__ constexpr bool int_kind(int kind) {
    return kind == 8 || kind == 18 || kind == 20 || kind == 21 || kind == 22 || kind == 23 || kind == 42;
}

__device__ __forceinline__ int dp4a_ss(int a, int b, int c) {
    int d;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}
__device__ __forceinline__ int dp4a_su(int a, uint32_t b, int c) {
    int d;
    asm("dp4a.s32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// Four unsigned grid bytes with byte n negated where bit n of `signs` is set. Exact because no
// IQ2_S/IQ3_XXS/IQ3_S grid byte is zero, so (b ^ 0xff) + 1 never carries into the next byte.
__device__ __forceinline__ int signed4(uint32_t grid, uint32_t signs) {
    const uint32_t ones = ((signs & 15u) * 0x00204081u) & 0x01010101u;  // bit n -> byte n
    return int((grid ^ (ones * 255u)) + ones);
}

__device__ __forceinline__ uint32_t pack4(int a, int b, int c, int d) {
    return __byte_perm(__byte_perm(a, b, 0x0040), __byte_perm(c, d, 0x0040), 0x5410);
}

// Shared-memory bytes are at least halfword aligned wherever u16s are read below.
__device__ __forceinline__ uint32_t u16s(const uint8_t* p) { return *reinterpret_cast<const uint16_t*>(p); }

// q0 (coefficients i .. i+3) and q1 (i+4 .. i+7) of block p as int8x4 words, i % 8 == 0; returns d.
// `kvs` is the IQ4 codebook (int8) in shared memory.
template <int Kind>
__device__ __forceinline__ float int8_block(const uint8_t* p, int i, int& q0, int& q1, const int8_t* kvs) {
    static_assert(int_kind(Kind), "integer eight-coefficient decode");
    if constexpr (Kind == 8) {  // block bytes 2.. are halfword aligned
        const uint8_t* c = p + 2 + i;
        q0 = int(u16s(c) | (u16s(c + 2) << 16));
        q1 = int(u16s(c + 4) | (u16s(c + 6) << 16));
        return half_value(p);
    } else if constexpr (Kind == 20 || Kind == 23) {
        const uint8_t* c;
        int shift;
        float d;
        if constexpr (Kind == 20) {
            c = p + 2 + i % 16; shift = 4 * (i / 16); d = half_value(p);
        } else {
            const int group = i / 32, element = i % 32;
            const int scale = int(((p[4 + group / 2] >> (4 * (group % 2))) & 15)
                                 | (((u16s(p + 2) >> (2 * group)) & 3) << 4)) - 32;
            c = p + 8 + group * 16 + element % 16; shift = 4 * (element / 16); d = half_value(p) * float(scale);
        }
        const uint32_t w0 = ((u16s(c) | (u16s(c + 2) << 16)) >> shift) & 0x0F0F0F0Fu;
        const uint32_t w1 = ((u16s(c + 4) | (u16s(c + 6) << 16)) >> shift) & 0x0F0F0F0Fu;
        q0 = int(pack4(kvs[w0 & 15], kvs[(w0 >> 8) & 15], kvs[(w0 >> 16) & 15], kvs[w0 >> 24]));
        q1 = int(pack4(kvs[w1 & 15], kvs[(w1 >> 8) & 15], kvs[(w1 >> 16) & 15], kvs[w1 >> 24]));
        return d;
    } else if constexpr (Kind == 42) {
        const uint32_t bits = u16s(p + 2 + i / 4);  // i / 4 is even
        // Spread the 2-bit codes to bytes; code + 0x7f stays inside its byte, and ^ 0x80 maps it to code - 1.
        const uint32_t b0 = (bits & 3) | ((bits & 0xC) << 6) | ((bits & 0x30) << 12) | ((bits & 0xC0) << 18);
        const uint32_t b1 = ((bits >> 8) & 3) | ((bits >> 2) & 0x300) | ((bits << 4) & 0x30000) | ((bits << 10) & 0x3000000);
        q0 = int((b0 + 0x7F7F7F7Fu) ^ 0x80808080u);
        q1 = int((b1 + 0x7F7F7F7Fu) ^ 0x80808080u);
        return half_value(p);
    } else {
        const int group = i / 32, sub = (i % 32) / 8;
        uint32_t lo, hi, signs;
        float d;
        if constexpr (Kind == 22) {
            const int code = p[2 + group * 4 + sub] | ((uint32_t(p[66 + group]) << (8 - 2 * sub)) & 0x300);
            const uint64_t grid = tf_iq2s_grid[code];
            lo = uint32_t(grid); hi = uint32_t(grid >> 32);
            const int scale = (p[74 + group] >> (4 * (sub / 2))) & 15;
            d = (half_value(p) * (0.5f + float(scale))) * 0.25f;
            signs = p[34 + group * 4 + sub];
        } else if constexpr (Kind == 18) {
            const uint32_t auxiliary = u16s(p + 66 + group * 4) | (u16s(p + 68 + group * 4) << 16);
            signs = tf_ksigns_iq2xs[(auxiliary >> (7 * sub)) & 127];
            lo = tf_iq3xxs_grid[p[2 + group * 8 + sub * 2]];
            hi = tf_iq3xxs_grid[p[3 + group * 8 + sub * 2]];
            d = (half_value(p) * (0.5f + float(auxiliary >> 28))) * 0.5f;
        } else {  // 21
            const uint32_t high = p[66 + group];
            lo = tf_iq3s_grid[p[2 + group * 8 + sub * 2] | ((high << (8 - 2 * sub)) & 256)];
            hi = tf_iq3s_grid[p[3 + group * 8 + sub * 2] | ((high << (7 - 2 * sub)) & 256)];
            const int scale = (p[106 + group / 2] >> (4 * (group % 2))) & 15;
            d = half_value(p) * float(1 + 2 * scale);
            signs = p[74 + group * 4 + sub];
        }
        q0 = signed4(lo, signs);
        q1 = signed4(hi, signs >> 4);
        return d;
    }
}

template <int Kind, int Block, int Bytes>
__device__ __forceinline__ float int8_kind(const uint8_t* raw, int col, int& q0, int& q1, const int8_t* kvs) {
    return int8_block<Kind>(raw + (col / Block) * Bytes, col % Block, q0, q1, kvs);
}

// Quantizes eight FP16 inputs (one of the four chunks of a 32-block held by four consecutive lanes
// of a full warp) to {hi 0-3, hi 4-7, lo 0-3, lo 4-7}; returns the block scale s.
__device__ __forceinline__ float quantize8(const uint4& h, uint4& packed) {
    const uint32_t w[4] = {h.x, h.y, h.z, h.w};
    float v[8];
    uint32_t amax = 0;  // |x| bit patterns order like their values; NaN sorts above infinity
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&w[j]));
        v[2 * j] = f.x; v[2 * j + 1] = f.y;
        amax = max(amax, max(__float_as_uint(fabsf(f.x)), __float_as_uint(fabsf(f.y))));
    }
    amax = max(amax, __shfl_xor_sync(0xffffffff, amax, 1));
    amax = max(amax, __shfl_xor_sync(0xffffffff, amax, 2));
    const float a = __uint_as_float(amax);
    const float inverse = a > 0.0f && a <= 65504.0f ? 32767.0f / a : 0.0f;
    int e[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j)  // round to nearest even: |v * inverse| <= 32767 < 2^22
        e[j] = __float_as_int(fmaf(v[j], inverse, 12582912.0f)) - 0x4B400000;
    packed.x = __byte_perm(__byte_perm(e[0], e[1], 0x0051), __byte_perm(e[2], e[3], 0x0051), 0x5410);
    packed.y = __byte_perm(__byte_perm(e[4], e[5], 0x0051), __byte_perm(e[6], e[7], 0x0051), 0x5410);
    packed.z = __byte_perm(__byte_perm(e[0], e[1], 0x0040), __byte_perm(e[2], e[3], 0x0040), 0x5410);
    packed.w = __byte_perm(__byte_perm(e[4], e[5], 0x0040), __byte_perm(e[6], e[7], 0x0040), 0x5410);
    return a / 32767.0f;  // NaN/Inf blocks keep a non-finite scale
}

// Exact sum q[j] * xq[j] over eight coefficients.
__device__ __forceinline__ int dot8(int q0, int q1, const uint4& x) {
    const int high = dp4a_ss(q1, int(x.y), dp4a_ss(q0, int(x.x), 0));
    return dp4a_su(q1, x.w, dp4a_su(q0, x.z, high * 256));
}

// Gate/up formats with compiled integer pairs (the SC117 expert banks).
__host__ __device__ constexpr bool gate_up_int_kind(int kind) {
    return kind == 18 || kind == 21 || kind == 22 || kind == 23;
}

// Bytes of one gate/up row staged in shared memory, or 0 when the generic path is required.
int staged_row_bytes(const PackedView& v) {
    if (!gate_up_int_kind(v.kind) || v.offset % v.block || 2560 % v.block || v.stride % 4
        || reinterpret_cast<uintptr_t>(v.data) % 4) return 0;
    const int64_t first = int64_t(v.offset / v.block) * v.bytes, bytes = int64_t(2560 / v.block) * v.bytes;
    return bytes % 4 || first % 4 || bytes > GU_RAW / 2 ? 0 : int(bytes);
}

struct GateUpShared {
    uint8_t* raw;       // this warp's gate row, then its up row
    const uint4* xq;    // [slice row][320] quantized inputs
    const float* xs;    // [slice row][80] block scales
    const int8_t* kvs;
};

// One output's gate and up rows against the slice's n pairs. Lane l owns coefficients 8l .. 8l + 7 of
// each 256-wide round. Both formats are compile-time, so rounds are straight-line code.
template <int GK, int GB, int GBytes, int UK, int UB, int UBytes>
__device__ __forceinline__ void gate_up_int(const GateUpShared& s, int gate_raw, int n, int lane,
                                            float (&g)[GU_SLICE], float (&u)[GU_SLICE]) {
    const uint8_t* graw = s.raw;
    const uint8_t* uraw = s.raw + gate_raw;
    #pragma unroll 2
    for (int chunk = lane; chunk < 320; chunk += 32) {
        int g0, g1, u0, u1;
        const float dg = int8_kind<GK, GB, GBytes>(graw, chunk * 8, g0, g1, s.kvs);
        const float du = int8_kind<UK, UB, UBytes>(uraw, chunk * 8, u0, u1, s.kvs);
        #pragma unroll
        for (int r = 0; r < GU_SLICE; ++r) {
            if (r < n) {
                const uint4 x = s.xq[r * 320 + chunk];
                const float scale = s.xs[r * 80 + chunk / 4];
                g[r] = fmaf(dg * scale, float(dot8(g0, g1, x)), g[r]);
                u[r] = fmaf(du * scale, float(dot8(u0, u1, x)), u[r]);
            }
        }
    }
}

// Eight FP16 values; `vector` when the base is 16-byte aligned.
__device__ __forceinline__ uint4 load8(const __half* src, bool vector) {
    if (vector) return *reinterpret_cast<const uint4*>(src);
    const auto* s = reinterpret_cast<const unsigned short*>(src);
    return make_uint4(s[0] | (uint32_t(s[1]) << 16), s[2] | (uint32_t(s[3]) << 16),
                      s[4] | (uint32_t(s[5]) << 16), s[6] | (uint32_t(s[7]) << 16));
}

#define TF_INT_KINDS(X) X(8, 32, 34) X(18, 256, 98) X(20, 32, 18) X(21, 256, 110) X(22, 256, 82) \
    X(23, 256, 136) X(42, 64, 18)
#define TF_GATE_UP_KINDS(X) X(18, 256, 98) X(21, 256, 110) X(22, 256, 82) X(23, 256, 136)

template <int GK, int GB, int GBytes>
__device__ __forceinline__ void gate_up_dispatch_up(const GateUpShared& s, int up_kind, int gate_raw, int n,
                                                    int lane, float (&g)[GU_SLICE], float (&u)[GU_SLICE]) {
    switch (up_kind) {
#define TF_UP_CASE(K, B, BYTES) case K: gate_up_int<GK, GB, GBytes, K, B, BYTES>(s, gate_raw, n, lane, g, u); return;
        TF_GATE_UP_KINDS(TF_UP_CASE)
#undef TF_UP_CASE
    }
}

// CTA = GU_WARPS intermediate outputs of one tile, one per warp; each warp stages its raw gate and up
// rows once. Pairs are processed GU_SLICE at a time; their inputs are quantized into shared memory.
// Lanes combine by shuffle tree; the SiLU product and FP16 store are the reference's.
__global__ __launch_bounds__(GU_WARPS * 32)
void expert_gate_up_decode(const __half* x, PackedView gate, PackedView up, int slots, OwnedGrouping group,
                           __half* activation, int gate_raw, int up_raw, int slice) {
    extern __shared__ __align__(16) uint8_t decode_smem[];
    __shared__ int8_t kvs[16];
    __shared__ int pairs[DECODE_PAIRS];
    const int tile = blockIdx.y;
    if (tile >= *group.total) return;  // CTA-uniform, before any barrier.
    const int expert = group.tiles[3 * tile], begin = group.tiles[3 * tile + 1];
    const int count = group.tiles[3 * tile + 2];
    const int thread = threadIdx.x, lane = thread % 32, warp = thread / 32;
    const int output = blockIdx.x * GU_WARPS + warp;
    const int64_t source = int64_t(expert) * 160 + output;
    const bool vector = (reinterpret_cast<uintptr_t>(x) & 15) == 0;
    const int row_bytes = (gate_raw + up_raw + 15) / 16 * 16;
    GateUpShared s;
    s.xq = reinterpret_cast<const uint4*>(decode_smem);
    s.xs = reinterpret_cast<const float*>(decode_smem + slice * 320 * 16);
    s.raw = decode_smem + slice * (320 * 16 + 80 * 4) + warp * row_bytes;
    s.kvs = kvs;
    if (thread < 16) kvs[thread] = tf_kvalues_iq4nl[thread];
    if (thread < count) pairs[thread] = group.pairs[begin + thread];
    {  // every load of both rows in flight before the first store
        constexpr int WORDS = GU_RAW / 4 / 32 + 1;
        const auto* gs = reinterpret_cast<const uint32_t*>(
            gate.data + source * gate.stride + (gate.offset / gate.block) * gate.bytes);
        const auto* us = reinterpret_cast<const uint32_t*>(
            up.data + source * up.stride + (up.offset / up.block) * up.bytes);
        uint32_t w[WORDS];
        #pragma unroll
        for (int j = 0; j < WORDS; ++j) {
            const int i = lane + 32 * j;
            if (i < gate_raw / 4) w[j] = __ldg(gs + i);
            else if (i < (gate_raw + up_raw) / 4) w[j] = __ldg(us + i - gate_raw / 4);
        }
        #pragma unroll
        for (int j = 0; j < WORDS; ++j)
            if (lane + 32 * j < (gate_raw + up_raw) / 4) reinterpret_cast<uint32_t*>(s.raw)[lane + 32 * j] = w[j];
    }
    for (int start = 0; start < count; start += slice) {
        const int n = min(slice, count - start);
        __syncthreads();  // pairs/kvs/raw published; the previous slice's readers are done
        for (int item = thread; item < n * 320; item += GU_WARPS * 32) {  // full warps: n * 320 % 32 == 0
            const int r = item / 320, chunk = item % 320;
            const uint4 h = load8(x + int64_t(pairs[start + r] / slots) * 2560 + chunk * 8, vector);
            uint4 packed;
            const float scale = quantize8(h, packed);
            reinterpret_cast<uint4*>(decode_smem)[item] = packed;
            if (chunk % 4 == 0) reinterpret_cast<float*>(decode_smem + slice * 320 * 16)[r * 80 + chunk / 4] = scale;
        }
        __syncthreads();
        float g[GU_SLICE] = {}, u[GU_SLICE] = {};
        switch (gate.kind) {  // the launcher admits only compiled integer pairs
#define TF_GATE_CASE(K, B, BYTES) case K: gate_up_dispatch_up<K, B, BYTES>(s, up.kind, gate_raw, n, lane, g, u); break;
            TF_GATE_UP_KINDS(TF_GATE_CASE)
#undef TF_GATE_CASE
        }
        #pragma unroll
        for (int r = 0; r < GU_SLICE; ++r) {
            if (r < n) {
                for (int offset = 16; offset; offset /= 2) {
                    g[r] += __shfl_down_sync(0xffffffff, g[r], offset);
                    u[r] += __shfl_down_sync(0xffffffff, u[r], offset);
                }
                if (lane == 0) {
                    const float result = (g[r] / (1.0f + expf(-g[r]))) * u[r];
                    activation[int64_t(pairs[start + r]) * 160 + output] = __float2half_rn(result);
                }
            }
        }
    }
}

void launch_gate_up_decode(const __half* x, int rows, const PackedView& gate, const PackedView& up, int slots,
                           const OwnedGrouping& group, __half* activation, cudaStream_t stream) {
    const int gate_raw = staged_row_bytes(gate), up_raw = staged_row_bytes(up);
    if (!gate_raw || !up_raw) {  // formats/views outside the integer path: reference scalar kernel
        expert_gate_up_grouped<4><<<dim3(160 / EXPERT_WARPS, group.max_tiles), EXPERT_WARPS * 32, 0, stream>>>(
            x, gate, up, slots, group, activation);
        return;
    }
    const int slice = std::min(GU_SLICE, rows);  // a tile holds at most `rows` distinct-token pairs
    const int bytes = slice * (320 * 16 + 80 * 4) + GU_WARPS * ((gate_raw + up_raw + 15) / 16 * 16);
    expert_gate_up_decode<<<dim3(160 / GU_WARPS, group.max_tiles), GU_WARPS * 32, bytes, stream>>>(
        x, gate, up, slots, group, activation, gate_raw, up_raw, slice);
}

// Down: Split consecutive lanes share one hidden output row (32 / Split rows per warp); lane part p
// takes the eight-coefficient chunks p, p + Split, ... of K = 160 for every pair of the tile
// (activations quantized once per CTA, as above), and the parts combine by shuffle. Split 4 trades
// per-CTA overhead for parallelism at small row counts; Split 1 serves larger batches. Each warp stages its
// rows' raw bytes (a word-aligned superset of the 160-column span) with every load in flight first.
// Rank-sliced ETP4 banks have 2 mod 4 row strides (5 IQ4_NL or 3 Q2_0 blocks of 18 bytes), so a row's
// span may start 2 bytes into its first staged word; every block then still starts halfword aligned,
// which is all int8_block requires. The extra 2 bytes are staged for those banks.
int down_row_bytes(const PackedView& v) {  // padded to an odd word count: conflict-free row strides
    if (!int_kind(v.kind) || v.offset % 8 || v.stride % 2 || reinterpret_cast<uintptr_t>(v.data) % 4) return 0;
    const int first = v.offset / v.block * v.bytes / 4 * 4, end = ((v.offset + 159) / v.block + 1) * v.bytes;
    const int words = (end - first + (v.stride % 4) + 3) / 4;
    return 4 * (words | 1);
}

// Chunks part, part + 4, ... of K = 160 (one Split-4 lane part), accumulated in order into acc.
template <int Kind, int Block, int Bytes>
__device__ __forceinline__ void down_int(const uint8_t* raw, int skip, int offset, int part, const uint4* xq,
                                         const float* xs, int n, const int8_t* kvs, float (&acc)[DECODE_PAIRS]) {
    #pragma unroll
    for (int chunk = part; chunk < 20; chunk += 4) {
        const int col = offset + chunk * 8;
        int q0, q1;
        const float d = int8_block<Kind>(raw + (col / Block) * Bytes - skip, col % Block, q0, q1, kvs);
        #pragma unroll
        for (int p = 0; p < DECODE_PAIRS; ++p)
            if (p < n) acc[p] = fmaf(d * xs[p * 5 + chunk / 4], float(dot8(q0, q1, xq[p * 20 + chunk])), acc[p]);
    }
}

// Q2_0 down (Exact): d * q with q in {-1, 0, 1, 2} is exact in FP16, so the reference path is already
// FP32-exact up to its sums (~1e-7 vs FP64) and the int16 input quantization above (~3e-5) would
// worsen it. These coefficients are formed exactly in FP32 and multiply the unquantized FP16
// activations with FP32 FMAs; other kinds keep the faster integer path, which removes their larger
// FP16 coefficient rounding.
template <int Kind, int Block, int Bytes>
__device__ __forceinline__ void down_exact(const uint8_t* raw, int skip, int offset, int part, const float4* xf,
                                           int n, const int8_t* kvs, float (&acc)[DECODE_PAIRS]) {
    #pragma unroll
    for (int chunk = part; chunk < 20; chunk += 4) {
        const int col = offset + chunk * 8;
        int q0, q1;
        const float d = int8_block<Kind>(raw + (col / Block) * Bytes - skip, col % Block, q0, q1, kvs);
        // Byte j biased by 128 becomes the low byte of 2^23 + (q + 128); subtracting 2^23 + 128 is
        // exact (I2F runs at a quarter of the FADD rate), and so is the product with d.
        const uint32_t u0 = uint32_t(q0) ^ 0x80808080u, u1 = uint32_t(q1) ^ 0x80808080u;
        float c[8];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            c[j] = (__uint_as_float(__byte_perm(u0, 0x4B000000u, 0x7540 + j)) - 8388736.0f) * d;
            c[4 + j] = (__uint_as_float(__byte_perm(u1, 0x4B000000u, 0x7540 + j)) - 8388736.0f) * d;
        }
        #pragma unroll
        for (int p = 0; p < DECODE_PAIRS; ++p) {
            if (p < n) {
                const float4 a = xf[p * 40 + chunk * 2], b = xf[p * 40 + chunk * 2 + 1];
                float v = acc[p];
                v = fmaf(c[0], a.x, v); v = fmaf(c[1], a.y, v); v = fmaf(c[2], a.z, v); v = fmaf(c[3], a.w, v);
                v = fmaf(c[4], b.x, v); v = fmaf(c[5], b.y, v); v = fmaf(c[6], b.z, v); v = fmaf(c[7], b.w, v);
                acc[p] = v;
            }
        }
    }
}

template <bool F32, int Split, bool Exact>
__global__ __launch_bounds__(DOWN_WARPS * 32)
void expert_down160_decode(const __half* activation, PackedView down, int dims, OwnedGrouping group,
                           void* output, int row_bytes) {
    extern __shared__ __align__(16) uint8_t down_smem[];
    // Exact: FP32 activations [pair][160]; otherwise int16 hi/lo chunks [pair][20] and block scales.
    __shared__ __align__(16) uint8_t xsm[Exact ? DECODE_PAIRS * 640 : DECODE_PAIRS * (20 * 16 + 5 * 4)];
    float4* xf = reinterpret_cast<float4*>(xsm);
    uint4* xq = reinterpret_cast<uint4*>(xsm);
    float* xs = reinterpret_cast<float*>(xsm + DECODE_PAIRS * 20 * 16);
    __shared__ int8_t kvs[16];
    __shared__ int pairs[DECODE_PAIRS];
    const int tile = blockIdx.y;
    if (tile >= *group.total) return;  // CTA-uniform, before any barrier.
    const int expert = group.tiles[3 * tile], begin = group.tiles[3 * tile + 1];
    const int count = group.tiles[3 * tile + 2];
    const int thread = threadIdx.x, lane = thread % 32, warp = thread / 32;
    constexpr int WARP_ROWS = 32 / Split;
    const int first = (blockIdx.x * DOWN_WARPS + warp) * WARP_ROWS;
    const int64_t row0 = int64_t(expert) * dims + first;
    const int skip = down.offset / down.block * down.bytes / 4 * 4;  // superset start within each row
    const int row = lane / Split, part = lane % Split;
    uint8_t* raw = down_smem + warp * WARP_ROWS * row_bytes;
    if (thread < 16) kvs[thread] = tf_kvalues_iq4nl[thread];
    if (thread < count) pairs[thread] = group.pairs[begin + thread];
    {
        const int words = row_bytes / 4, total = WARP_ROWS * words;
        for (int base = lane; base < total; base += 32 * 8) {
            uint32_t w[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int i = base + 32 * j;
                // Word-aligned global words from the row's span start rounded down. A leading word may
                // begin in the previous row (start % 4 != 0 implies a previous row exists); a word
                // reaching past the row end is loaded only up to it (stride is even), never beyond.
                const int r = i / words, word = i % words;
                const int64_t at = (((row0 + r) * down.stride + skip) & ~int64_t(3)) + 4 * word;
                const int64_t end = (row0 + r + 1) * down.stride;
                if (i < total && at + 4 <= end) w[j] = __ldg(reinterpret_cast<const uint32_t*>(down.data + at));
                else if (i < total && at < end) w[j] = __ldg(reinterpret_cast<const uint16_t*>(down.data + at));
                else w[j] = 0;
            }
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                if (base + 32 * j < total) reinterpret_cast<uint32_t*>(raw)[base + 32 * j] = w[j];
        }
    }
    __syncthreads();
    const bool vector = (reinterpret_cast<uintptr_t>(activation) & 15) == 0;
    if constexpr (Exact) {
        for (int item = thread; item < count * 20; item += DOWN_WARPS * 32) {
            const int p = item / 20, chunk = item % 20;
            const uint4 h = load8(activation + int64_t(pairs[p]) * 160 + chunk * 8, vector);
            const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&h.x));
            const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&h.y));
            const float2 f2 = __half22float2(*reinterpret_cast<const __half2*>(&h.z));
            const float2 f3 = __half22float2(*reinterpret_cast<const __half2*>(&h.w));
            xf[2 * item] = make_float4(f0.x, f0.y, f1.x, f1.y);
            xf[2 * item + 1] = make_float4(f2.x, f2.y, f3.x, f3.y);
        }
    } else {
        // Whole warps run quantize8 (its shuffles); a 32-block's four chunks share a warp as 20 % 4 == 0.
        for (int item = thread; item < (count * 20 + 31) / 32 * 32; item += DOWN_WARPS * 32) {
            const int p = item / 20, chunk = item % 20;
            const bool valid = item < count * 20;
            const uint4 h = valid ? load8(activation + int64_t(pairs[p]) * 160 + chunk * 8, vector)
                                  : make_uint4(0, 0, 0, 0);
            uint4 packed;
            const float scale = quantize8(h, packed);
            if (valid) {
                xq[item] = packed;
                if (chunk % 4 == 0) xs[p * 5 + chunk / 4] = scale;
            }
        }
    }
    __syncthreads();
    const uint8_t* mine = raw + row * row_bytes + int(((row0 + row) * down.stride + skip) & 3);
    auto accumulate = [&](int chunks, float (&sum)[DECODE_PAIRS]) {  // one Split-4 lane part's chunks
        if constexpr (Exact) {
            down_exact<42, 64, 18>(mine, skip, down.offset, chunks, xf, count, kvs, sum);
        } else {
            switch (down.kind) {  // the launcher admits only integer kinds with a staged span
#define TF_DOWN_CASE(K, B, BYTES) case K: down_int<K, B, BYTES>(mine, skip, down.offset, chunks, xq, xs, count, kvs, sum); break;
                TF_INT_KINDS(TF_DOWN_CASE)
#undef TF_DOWN_CASE
            }
        }
    };
    // Row-count independence: every output is ((s0 + s1) + (s2 + s3)), s_c the in-order sum of chunks
    // c, c + 4, ... Split 4 computes s_part per lane and adds them by butterfly; Split 1 computes the
    // four in turn, so a token's outputs are the same bits whichever Split its call's row count selects.
    static_assert(Split == 1 || Split == 4, "both Splits form the four-part summation tree");
    float acc[DECODE_PAIRS] = {};
    if constexpr (Split == 4) {
        accumulate(part, acc);
    } else {
        float s[DECODE_PAIRS] = {}, t[DECODE_PAIRS] = {};
        accumulate(0, acc);
        accumulate(1, s);
        #pragma unroll
        for (int p = 0; p < DECODE_PAIRS; ++p) acc[p] = acc[p] + s[p];
        #pragma unroll
        for (int p = 0; p < DECODE_PAIRS; ++p) s[p] = 0.0f;
        accumulate(2, s);
        accumulate(3, t);
        #pragma unroll
        for (int p = 0; p < DECODE_PAIRS; ++p) acc[p] = acc[p] + (s[p] + t[p]);
    }
    #pragma unroll
    for (int p = 0; p < DECODE_PAIRS; ++p) {
        if (p < count) {  // CTA uniform
            #pragma unroll
            for (int offset = 1; offset < Split; offset *= 2)  // Split 4, part 0: (s0 + s1) + (s2 + s3)
                acc[p] += __shfl_xor_sync(0xffffffff, acc[p], offset);
            if (part == 0) {
                const float value = acc[p] + 0.0f;
                const int64_t at = int64_t(pairs[p]) * dims + first + row;
                if constexpr (F32) static_cast<float*>(output)[at] = value;
                else static_cast<__half*>(output)[at] = __float2half_rn(value);
            }
        }
    }
}

void launch_down_decode(const __half* activation, int rows, const PackedView& down, int dims,
                        const OwnedGrouping& group, void* output, bool f32, cudaStream_t stream) {
    const int row_bytes = down_row_bytes(down);
    if (!row_bytes) {  // formats/views outside the integer path: reference one-warp WMMA kernel
        const dim3 grid(dims / 16, group.max_tiles);
        #define TF_GROUPED_DOWN(FLOAT, KIND) \
            expert_down160_grouped<FLOAT, KIND><<<grid, 32, 0, stream>>>(activation, down, dims, group, output)
        if (f32) { if (down.kind == 7) TF_GROUPED_DOWN(true, 7); else TF_GROUPED_DOWN(true, -1); }
        else { if (down.kind == 7) TF_GROUPED_DOWN(false, 7); else TF_GROUPED_DOWN(false, -1); }
        #undef TF_GROUPED_DOWN
        return;
    }
    // Split choice is a speed-only decision (both form the same summation tree; see the kernel):
    // measured crossover on sliced SC117 IQ4_NL/Q2_0 banks (V100), Split 4 faster through 9 rows.
    const bool narrow = rows <= 9;
    const dim3 grid(dims / (DOWN_WARPS * (narrow ? 8 : 32)), group.max_tiles);
    const int bytes = DOWN_WARPS * (narrow ? 8 : 32) * row_bytes;
    // Wide spans (Q8_0 rank slices: 172-byte rows) exceed the 48 KiB default shared memory.
    #define TF_DOWN_LAUNCH(F, S, E) do { \
        if (bytes > 32 * 1024) C10_CUDA_CHECK(cudaFuncSetAttribute( \
            expert_down160_decode<F, S, E>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)); \
        expert_down160_decode<F, S, E><<<grid, DOWN_WARPS * 32, bytes, stream>>>( \
            activation, down, dims, group, output, row_bytes); } while (0)
    #define TF_DOWN_SPLIT(F, E) do { if (narrow) TF_DOWN_LAUNCH(F, 4, E); else TF_DOWN_LAUNCH(F, 1, E); } while (0)
    const bool exact = down.kind == 42;
    if (f32) { if (exact) TF_DOWN_SPLIT(true, true); else TF_DOWN_SPLIT(true, false); }
    else { if (exact) TF_DOWN_SPLIT(false, true); else TF_DOWN_SPLIT(false, false); }
    #undef TF_DOWN_SPLIT
    #undef TF_DOWN_LAUNCH
}

// publish_owned for prompt-sized calls: one thread per 8 consecutive outputs of one pair (dims % 8
// == 0, 16-byte aligned shared/output; the launcher checks), int32 indices (pairs * dims < 2^31).
// Same values and storage rounding per element; owned pairs are skipped whole.
template <typename Index, bool F32>
__global__ void publish_owned8(const Index* global, const int64_t* local, const float* shared,
                               void* output, int pairs, int dims, int slots, int global_experts) {
    const int chunks = dims / 8;
    for (int item = blockIdx.x * blockDim.x + threadIdx.x; item < pairs * chunks; item += gridDim.x * blockDim.x) {
        const int pair = item / chunks, column = (item % chunks) * 8;
        const auto id = global[pair];
        float v[8];
        if (id == global_experts) {
            const float4* s = reinterpret_cast<const float4*>(shared + (pair / slots) * dims + column);
            const float4 a = s[0], b = s[1];
            v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
        } else {
            float value;
            if (id < 0 || id > global_experts) value = __int_as_float(0x7fffffff);
            else if (local[pair] < 0) value = 0.0f;
            else continue;  // owned: the down kernel's storage
            #pragma unroll
            for (int j = 0; j < 8; ++j) v[j] = value;
        }
        const int at = pair * dims + column;
        if constexpr (F32) {
            float4* o = reinterpret_cast<float4*>(static_cast<float*>(output) + at);
            o[0] = make_float4(v[0], v[1], v[2], v[3]);
            o[1] = make_float4(v[4], v[5], v[6], v[7]);
        } else {
            *reinterpret_cast<uint4*>(static_cast<__half*>(output) + at) = pack8(v);
        }
    }
}

// Grouping tiles are consecutive 16-pair slices of each expert. Only the tile opening each
// 64-pair group leads (CTA-uniform return), so every pair is computed by exactly one CTA.
__device__ __forceinline__ bool prefill_group(const OwnedGrouping& group, int tile,
                                              int& expert, int& begin, int& count) {
    if (tile >= *group.total) return false;
    expert = group.tiles[3 * tile];
    begin = group.tiles[3 * tile + 1];
    const int at = begin - group.offsets[expert];
    if (at % PREFILL_PAIRS) return false;
    count = min(PREFILL_PAIRS, group.counts[expert] - at);
    return true;
}

// Packed bytes per 256-coefficient block of the kinds whose gate/up rows the prefill kernel stages raw.
template <int Kind>
__host__ __device__ constexpr int staged_block_bytes() {
    return Kind == 18 ? 98 : Kind == 21 ? 110 : Kind == 22 ? 82 : Kind == 23 ? 136 : 0;
}

// CTA = (32 intermediate outputs, <=64 pairs of one expert); K steps of 64, double-buffered:
// step s+1's activations are requested before step s's MMAs and its coefficients decoded
// after them, so one barrier per step separates producers from consumers.
// Staged kinds (word-aligned rows, offset 0; the launcher checks): the CTA's 64 packed rows of
// one 256-column block are copied to shared memory with word loads issued a block (4 steps)
// ahead, and coefficients are decoded from there by the same decode8_block expressions, so the
// staged FP16 tiles and the MMA sequence are bitwise those of the direct path.
template <int Kind>
__global__ __launch_bounds__(PREFILL_THREADS, 2)
void expert_gate_up_prefill(const __half* x, PackedView gate, PackedView up,
                            int slots, OwnedGrouping group, __half* activation) {
    namespace wmma = nvcuda::wmma;
    constexpr int M = 32, N = PREFILL_PAIRS, K = 64, LD = PREFILL_LD, STAGE = (2 * M + N) * LD;
    constexpr int LOADS = N * (K / 8) / PREFILL_THREADS;
    int expert, begin, count;
    if (!prefill_group(group, blockIdx.y, expert, begin, count)) return;
    const int first = blockIdx.x * M;
    __shared__ __align__(32) __half smem[2 * STAGE];  // per stage: gate A | up A | B; reused for C
    __shared__ int rows[N], pairs[N];
    constexpr int BYTES = staged_block_bytes<Kind>(), STAGED = BYTES > 0;
    constexpr int WORDS = (BYTES + 6) / 4;  // a block's bytes from its word-aligned start (start % 4 <= 2)
    constexpr int RAW = 2 * M * WORDS, FETCH = (RAW + PREFILL_THREADS - 1) / PREFILL_THREADS;
    __shared__ uint32_t raw[STAGED ? RAW : 1];  // [gate rows | up rows][WORDS] of one 256-column block
    const int thread = threadIdx.x, warp = thread / 32;
    if (thread < N) {
        const int pair = thread < count ? group.pairs[begin + thread] : -1;
        pairs[thread] = pair;
        rows[thread] = pair >= 0 ? (pair / slots) * 2560 : -1;
    }
    __syncthreads();
    const bool vector = (reinterpret_cast<uintptr_t>(x) & 15) == 0;
    // Decode item: one (row, 8-column chunk) of gate and of up per thread per step.
    const int drow = thread / 8, dchunk = thread % 8;
    const int64_t source = int64_t(expert) * 160 + first + drow;
    int brow[LOADS];
    #pragma unroll
    for (int j = 0; j < LOADS; ++j) brow[j] = rows[(thread + j * PREFILL_THREADS) / 8];
    const int bpart = thread % 8;
    auto load_b = [&](int base, uint4 (&b)[LOADS]) {
        #pragma unroll
        for (int j = 0; j < LOADS; ++j) {
            b[j] = make_uint4(0, 0, 0, 0);
            if (brow[j] >= 0) {
                const __half* src = x + brow[j] + base + bpart * 8;
                if (vector) b[j] = *reinterpret_cast<const uint4*>(src);
                else {
                    const auto* s = reinterpret_cast<const unsigned short*>(src);
                    b[j] = make_uint4(s[0] | (uint32_t(s[1]) << 16), s[2] | (uint32_t(s[3]) << 16),
                                      s[4] | (uint32_t(s[5]) << 16), s[6] | (uint32_t(s[7]) << 16));
                }
            }
        }
    };
    auto store = [&](__half* stage, const uint4& gv, const uint4& uv, const uint4 (&b)[LOADS]) {
        *reinterpret_cast<uint4*>(stage + drow * LD + dchunk * 8) = gv;
        *reinterpret_cast<uint4*>(stage + (M + drow) * LD + dchunk * 8) = uv;
        #pragma unroll
        for (int j = 0; j < LOADS; ++j) {
            const int item = thread + j * PREFILL_THREADS;
            *reinterpret_cast<uint4*>(stage + (2 * M + item / 8) * LD + bpart * 8) = b[j];
        }
    };
    // Word i of the raw block: packed row i / WORDS (gate rows, then up rows), loaded from the
    // block's word-aligned start; words wholly past the block end are not read. Rows are word
    // aligned, so a block ending a row ends on a word boundary and no load leaves its row.
    // Both banks are word aligned (launcher), so offsets within the CTA's 32 rows decide alignment.
    const uint8_t* gbase = gate.data + (int64_t(expert) * 160 + first) * gate.stride;
    const uint8_t* ubase = up.data + (int64_t(expert) * 160 + first) * up.stride;
    const int gstride = int(gate.stride), ustride = int(up.stride);  // one packed row: < 2^31
    uint32_t fetched[FETCH];
    auto fetch = [&](int block) {
        #pragma unroll
        for (int j = 0; j < FETCH; ++j) {
            const int i = thread + j * PREFILL_THREADS, r = i / WORDS, word = i % WORDS;
            const int start = (r % M) * (r < M ? gstride : ustride) + block * BYTES;
            const int at = (start & ~3) + 4 * word;
            fetched[j] = i < RAW && at < start + BYTES
                ? __ldg(reinterpret_cast<const uint32_t*>((r < M ? gbase : ubase) + at)) : 0u;
        }
    };
    auto publish = [&] {
        #pragma unroll
        for (int j = 0; j < FETCH; ++j)
            if (thread + j * PREFILL_THREADS < RAW) raw[thread + j * PREFILL_THREADS] = fetched[j];
    };
    auto decode = [&](const PackedView& bank, int r, int column) {
        if constexpr (STAGED) {
            const int start = (r % M) * int(bank.stride) + (column / 256) * BYTES;  // base rows word aligned
            const uint8_t* p = reinterpret_cast<const uint8_t*>(raw + r * WORDS) + (start & 3);
            return decode8_block<Kind>(p, column % 256);
        } else {
            return decode8_fixed<Kind>(bank, source, column);
        }
    };
    const int mtile = warp % 2, ntile = warp / 2;
    const bool active = ntile * 16 < count;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> gacc, uacc;
    wmma::fill_fragment(gacc, 0.0f);
    wmma::fill_fragment(uacc, 0.0f);
    if constexpr (STAGED) {
        fetch(0);
        publish();
        __syncthreads();
        fetch(1);
    }
    {
        uint4 b[LOADS];
        load_b(0, b);
        store(smem, decode(gate, drow, dchunk * 8), decode(up, M + drow, dchunk * 8), b);
    }
    __syncthreads();
    for (int base = 0, stage = 0; base < 2560; base += K, stage ^= 1) {
        const bool next = base + K < 2560;
        uint4 b[LOADS];
        if (next) load_b(base + K, b);
        if (active) {
            const __half* ga = smem + stage * STAGE;
            const __half* ua = ga + M * LD;
            const __half* bx = ga + 2 * M * LD;
            #pragma unroll
            for (int k = 0; k < K; k += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bf;
                // B is column-major: column n is pair n's input row.
                wmma::load_matrix_sync(bf, bx + ntile * 16 * LD + k, LD);
                wmma::load_matrix_sync(a, ga + mtile * 16 * LD + k, LD);
                wmma::mma_sync(gacc, a, bf, gacc);
                wmma::load_matrix_sync(a, ua + mtile * 16 * LD + k, LD);
                wmma::mma_sync(uacc, a, bf, uacc);
            }
        }
        if constexpr (STAGED) {
            // Step base + K opens block (base + K) / 256: every decode of the previous block
            // finished before the last barrier, so the buffer is free (CTA-uniform barrier).
            if (next && (base + K) % 256 == 0) {
                publish();
                __syncthreads();
                if (base + K + 256 < 2560) fetch((base + K) / 256 + 1);
            }
        }
        if (next)
            store(smem + (stage ^ 1) * STAGE, decode(gate, drow, base + K + dchunk * 8),
                  decode(up, M + drow, base + K + dchunk * 8), b);
        __syncthreads();
    }
    // C is staged column-major (pair-major) so each pair's outputs are contiguous.
    float* gc = reinterpret_cast<float*>(smem);
    float* uc = gc + M * N;
    if (active) {
        wmma::store_matrix_sync(gc + ntile * 16 * M + mtile * 16, gacc, M, wmma::mem_col_major);
        wmma::store_matrix_sync(uc + ntile * 16 * M + mtile * 16, uacc, M, wmma::mem_col_major);
    }
    __syncthreads();
    for (int i = thread; i < M * N; i += PREFILL_THREADS) {
        const int pair = i / M;
        if (pair < count) {
            const float g = gc[i], u = uc[i];
            activation[int64_t(pairs[pair]) * 160 + first + i % M] = __float2half_rn((g / (1.0f + expf(-g))) * u);
        }
    }
}

// Prefill-sized down projection: CTA = (256 hidden outputs, <=64 pairs of one expert), whole
// K=160 resident. Activations are loaded once per CTA; each 64-row slice of coefficients is
// decoded once into shared memory. Output storage rounding and -0 canonicalization match the
// decode-size kernel.
template <bool F32, int Kind>
__global__ __launch_bounds__(PREFILL_THREADS)
void expert_down160_prefill(const __half* activation, PackedView down,
                            int dims, OwnedGrouping group, void* output) {
    namespace wmma = nvcuda::wmma;
    constexpr int M = 64, N = PREFILL_PAIRS, K = 160, LD = K + 8, SPAN = 256;
    int expert, begin, count;
    if (!prefill_group(group, blockIdx.y, expert, begin, count)) return;
    __shared__ __align__(32) __half bx[N * LD];
    __shared__ __align__(32) __half aa[M * LD];  // reused as column-major FP32 C [N][M]
    __shared__ int pairs[N];
    const int thread = threadIdx.x, warp = thread / 32;
    if (thread < N) pairs[thread] = thread < count ? group.pairs[begin + thread] : -1;
    __syncthreads();
    for (int item = thread; item < N * (K / 8); item += PREFILL_THREADS) {
        const int pair = item / (K / 8), part = item % (K / 8);
        uint4 value = make_uint4(0, 0, 0, 0);
        if (pairs[pair] >= 0) value = *reinterpret_cast<const uint4*>(activation + int64_t(pairs[pair]) * 160 + part * 8);
        *reinterpret_cast<uint4*>(bx + pair * LD + part * 8) = value;
    }
    const int ntile = warp / 2, mpair = warp % 2;  // each warp owns m-tiles 2*mpair, 2*mpair+1 of one n-tile
    const bool active = ntile * 16 < count;
    float* cc = reinterpret_cast<float*>(aa);
    for (int first = blockIdx.x * SPAN; first < blockIdx.x * SPAN + SPAN; first += M) {
        for (int item = thread; item < M * (K / 8); item += PREFILL_THREADS) {
            const int row = item / (K / 8), part = item % (K / 8);
            *reinterpret_cast<uint4*>(aa + row * LD + part * 8) =
                decode8_fixed<Kind>(down, int64_t(expert) * dims + first + row, part * 8);
        }
        __syncthreads();
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c0, c1;
        if (active) {
            wmma::fill_fragment(c0, 0.0f);
            wmma::fill_fragment(c1, 0.0f);
            #pragma unroll
            for (int k = 0; k < K; k += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
                // B is column-major: each independent original pair owns one column.
                wmma::load_matrix_sync(b, bx + ntile * 16 * LD + k, LD);
                wmma::load_matrix_sync(a, aa + (2 * mpair) * 16 * LD + k, LD);
                wmma::mma_sync(c0, a, b, c0);
                wmma::load_matrix_sync(a, aa + (2 * mpair + 1) * 16 * LD + k, LD);
                wmma::mma_sync(c1, a, b, c1);
            }
        }
        __syncthreads();
        if (active) {
            wmma::store_matrix_sync(cc + ntile * 16 * M + (2 * mpair) * 16, c0, M, wmma::mem_col_major);
            wmma::store_matrix_sync(cc + ntile * 16 * M + (2 * mpair + 1) * 16, c1, M, wmma::mem_col_major);
        }
        __syncthreads();
        for (int i = thread; i < M * N; i += PREFILL_THREADS) {
            const int pair = i / M;
            if (pair < count) {
                const float value = fmaf(1.0f, cc[i], 0.0f) + 0.0f;
                const int64_t at = int64_t(pairs[pair]) * dims + first + i % M;
                if constexpr (F32) static_cast<float*>(output)[at] = value;
                else static_cast<__half*>(output)[at] = __float2half_rn(value);
            }
        }
        __syncthreads();
    }
}
}
// Standalone owned GU (selector rows 2/4 alike) uses the same route/GU kernels as the full
// owned path. Its activation ABI agrees with the supplied-GU consumer: slot10's
// first Half is immutable +0 for valid rows and NaN for malformed routes.
template<typename Index>
__global__ void initialize_owned_gu_slots(const Index* ids,__half* activation) {
 const int row=blockIdx.x;
 for(int i=threadIdx.x;i<11*160;i+=blockDim.x) activation[row*11*160+i]=__float2half_rn(0.f);
 __syncthreads();
 if(!threadIdx.x) {
  bool valid=ids[row*11+10]==512;
  for(int slot=0;slot<10;++slot) {
   const auto id=ids[row*11+slot];
   valid=valid&&id>=0&&id<512;
   for(int before=0;before<slot;++before) valid=valid&&ids[row*11+before]!=id;
  }
  if(!valid) activation[(row*11+10)*160]=__float2half_rn(__int_as_float(0x7fffffff));
 }
}
void gguf_experts_owned_impl(const at::Tensor& x, const at::Tensor& picks,
    const at::Tensor& gate, int64_t gate_kind, int64_t gate_offset,
    const at::Tensor& up, int64_t up_kind, int64_t up_offset,
    const at::Tensor& down, int64_t down_kind, int64_t down_offset,
    int64_t width, int64_t first, int64_t global_experts, const at::Tensor& shared,
    at::Tensor local, at::Tensor activation, at::Tensor output, at::Tensor grouping, int64_t gate_up_rows = 0,
    bool supplied_activation = false) {
    TORCH_CHECK(gate_up_rows == 0 || gate_up_rows == 2 || gate_up_rows == 4,
                "private grouped gate/up requires exactly 2 or 4 rows");
    TORCH_CHECK(gate_up_rows == 0 || grouping.defined(), "private grouped gate/up requires grouping scratch");
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.dim() == 2 && x.is_contiguous()
                && x.size(1) > 0 && x.size(1) <= std::numeric_limits<int>::max(),
                "owned experts require contiguous FP16 CUDA [rows,hidden] inputs");
    const auto device = x.device();
    c10::cuda::CUDAGuard guard(device);
    const auto* prop = at::cuda::getDeviceProperties(x.get_device());
    TORCH_CHECK(prop->major == 7 && prop->minor == 0, "owned expert candidate requires SM70");
    TORCH_CHECK(width == 160 && global_experts > 0 && global_experts <= std::numeric_limits<int>::max()
                && first >= 0 && first < global_experts, "owned candidate requires valid ETP4 width160 placement");
    TORCH_CHECK(picks.scalar_type() == at::kInt || picks.scalar_type() == at::kLong,
                "owned picks must be int32 or int64");
    operand(picks, picks.scalar_type(), device, "owned picks");
    TORCH_CHECK(picks.dim() == 2 && picks.size(0) == x.size(0) && picks.size(1) > 0
                && picks.numel() <= 65535, "owned slot geometry exceeds its bounded launch");
    const int dims = int(x.size(1));
    auto g = packed_view(gate, gate_kind, gate_offset, dims, device);
    auto u = packed_view(up, up_kind, up_offset, dims, device);
    auto d = packed_view(down, down_kind, down_offset, int(width), device);
    TORCH_CHECK(gate.size(0) % width == 0 && up.size(0) == gate.size(0)
                && down.size(0) % dims == 0 && down.size(0) / dims == gate.size(0) / width
                && gate.size(0) / width <= global_experts - first,
                "owned bank rows exceed the global expert inventory");
    const int experts = int(gate.size(0) / width);
    operand(shared, at::kFloat, device, "shared TP8 results");
    operand(local, at::kLong, device, "owned route scratch");
    operand(activation, at::kHalf, device, "owned activation scratch");
    TORCH_CHECK(output.scalar_type() == at::kHalf || output.scalar_type() == at::kFloat,
                "owned output must be FP16 or FP32");
    operand(output, output.scalar_type(), device, "owned output");
    TORCH_CHECK(shared.sizes() == x.sizes() && local.sizes() == picks.sizes()
                && activation.sizes() == at::IntArrayRef({picks.numel(), width})
                && output.sizes() == at::IntArrayRef({x.size(0), picks.size(1), x.size(1)}),
                "owned output and scratch require exact admitted shapes");
    tensorfold::check_disjoint(output, {&x, &picks, &gate, &up, &down, &shared, &local, &activation}, "owned output overlap");
    tensorfold::check_disjoint(local, {&x, &picks, &gate, &up, &down, &shared, &activation}, "owned route scratch overlap");
    tensorfold::check_disjoint(activation, {&x, &picks, &gate, &up, &down, &shared}, "owned activation scratch overlap");
    const int pairs = int(picks.numel());
    OwnedGrouping group{};
    if (grouping.defined()) {
        TORCH_CHECK(dims == 2560 && global_experts == 512,
                    "grouped owned down160 requires the admitted SC117 hidden/global expert geometry");
        operand(grouping, at::kInt, device, "owned grouping scratch");
        const int tiles = int(std::min<int64_t>(pairs, (pairs + 15) / 16 + std::max(0, experts - 1)));
        const int64_t words = 2LL * pairs + 2LL * experts + 3LL * tiles + 2;
        TORCH_CHECK(grouping.dim() == 1 && grouping.numel() == words,
                    "owned grouping scratch requires its exact bounded layout");
        tensorfold::check_disjoint(grouping, {&x, &picks, &gate, &up, &down, &shared, &local, &activation, &output},
                                   "owned grouping scratch overlap");
        TORCH_CHECK(experts <= GROUP_PREFIX_THREADS, "grouped owned experts exceed one prefix CTA");
        int* ptr = grouping.data_ptr<int>();
        group.counts = ptr; ptr += experts;
        group.offsets = ptr; ptr += experts + 1;
        group.ordinal = ptr; ptr += pairs;
        group.pairs = ptr; ptr += pairs;
        group.tiles = ptr; ptr += 3 * tiles;
        group.total = ptr;
        group.max_tiles = tiles;
    }
    if (supplied_activation) {
        TORCH_CHECK(grouping.defined() && !gate_up_rows && dims==2560 && global_experts==512
                    && experts==256 && (first==0 || first==256) && picks.size(1)==11
                    && x.size(0)>=1 && x.size(0)<=32,
                    "supplied GU requires grouped SC117 C1..32/TP8/ETP4 E256/top10+shared");
        const at::Tensor* operands[] = {&x,&picks,&gate,&up,&down,&shared,&local,&activation,&output,&grouping};
        for (const auto* tensor : operands)
            TORCH_CHECK(!tensor->requires_grad() && !tensor->is_neg() && !tensor->is_conj(),
                        "supplied GU requires plain non-autograd tensors");
        for (const auto* tensor : {&local,&output,&grouping})
            TORCH_CHECK(!tensor->is_inference() || c10::InferenceMode::is_enabled(),
                        "inference output mutation outside inference mode");
    }
    if (!picks.numel()) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    const int blocks = int((picks.numel() + 127) / 128);
    // Decode rows with the grouped gate/up: one grouping CTA also initializes the non-owned slots.
    const bool fused_grouping = grouping.defined() && gate_up_rows && !supplied_activation
                                && x.size(0) <= DECODE_MAX_ROWS && pairs <= GROUP_PREFIX_THREADS;
    if (fused_grouping) {
        auto act = reinterpret_cast<__half*>(activation.data_ptr());
        if (picks.scalar_type() == at::kLong)
            group_owned_decode<<<1, GROUP_PREFIX_THREADS, 0, stream>>>(picks.data_ptr<int64_t>(),
                local.data_ptr<int64_t>(), pairs, int(first), experts, group, act);
        else group_owned_decode<<<1, GROUP_PREFIX_THREADS, 0, stream>>>(picks.data_ptr<int32_t>(),
                local.data_ptr<int64_t>(), pairs, int(first), experts, group, act);
    } else if (grouping.defined()) {
        C10_CUDA_CHECK(cudaMemsetAsync(group.counts, 0, experts * sizeof(int), stream));
        if (picks.scalar_type() == at::kLong)
            group_owned_routes<<<blocks, 128, 0, stream>>>(picks.data_ptr<int64_t>(), local.data_ptr<int64_t>(),
                                                         pairs, int(first), experts, group);
        else group_owned_routes<<<blocks, 128, 0, stream>>>(picks.data_ptr<int32_t>(), local.data_ptr<int64_t>(),
                                                          pairs, int(first), experts, group);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        group_owned_prefix<<<1, GROUP_PREFIX_THREADS, 0, stream>>>(experts, group);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        group_owned_scatter<<<blocks, 128, 0, stream>>>(local.data_ptr<int64_t>(), pairs, group);
    } else if (picks.scalar_type() == at::kLong)
        owned_routes<<<blocks, 128, 0, stream>>>(picks.data_ptr<int64_t>(), local.data_ptr<int64_t>(), picks.numel(), int(first), experts);
    else owned_routes<<<blocks, 128, 0, stream>>>(picks.data_ptr<int32_t>(), local.data_ptr<int64_t>(), picks.numel(), int(first), experts);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    // No mapped ID equals experts, so shared pointers are never dereferenced.
    // Existing FP16 input storage supplies a valid pointer without a dummy bank.
    if (!supplied_activation) {
    if (gate_up_rows) {
        auto act = reinterpret_cast<__half*>(activation.data_ptr());
        if (!fused_grouping) {
            const int init_blocks = std::min((pairs * 160 + 127) / 128, 65535);
            initialize_unowned_gate_up<<<init_blocks, 128, 0, stream>>>(local.data_ptr<int64_t>(), act, pairs);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        auto input = reinterpret_cast<const __half*>(x.data_ptr());
        if (x.size(0) > DECODE_MAX_ROWS)  // prefill: tensor-core tiles amortize coefficient decoding over the tile's pairs
        {
            const dim3 grid(5, group.max_tiles);
            // Raw staging needs word-aligned packed rows read from their first byte.
            const auto aligned = [](const PackedView& v) {
                return v.offset == 0 && v.stride % 4 == 0 && (reinterpret_cast<uintptr_t>(v.data) & 3) == 0;
            };
            const int kind = g.kind == u.kind && aligned(g) && aligned(u) ? g.kind : -1;
            #define TF_GU_PREFILL(K) expert_gate_up_prefill<K><<<grid, PREFILL_THREADS, 0, stream>>>( \
                input, g, u, int(picks.size(1)), group, act)
            switch (kind) {  // SC117 expert banks: IQ3_XXS, IQ3_S, IQ2_S, IQ4_XS gate/up pairs
                case 18: TF_GU_PREFILL(18); break;
                case 21: TF_GU_PREFILL(21); break;
                case 22: TF_GU_PREFILL(22); break;
                case 23: TF_GU_PREFILL(23); break;
                default: TF_GU_PREFILL(-1);
            }
            #undef TF_GU_PREFILL
        }
        else  // the 2- and 4-row selectors share one decode-size kernel (per-row results are slice independent)
            launch_gate_up_decode(input, int(x.size(0)), g, u, int(picks.size(1)), group, act, stream);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        run_experts<int64_t>(x, local, g, u, d, x, x, x, output, activation,
                             dims, int(width), int(picks.size(1)), experts, true, true, dims == 2560, grouping.defined());
    }
    }
    if (grouping.defined()) {
        auto act = reinterpret_cast<const __half*>(activation.data_ptr());
        const bool prefill = x.size(0) > DECODE_MAX_ROWS && (reinterpret_cast<uintptr_t>(act) & 15) == 0;
        #define TF_DOWN_PREFILL(F, K) expert_down160_prefill<F, K><<<dim3(dims / 256, group.max_tiles), \
            PREFILL_THREADS, 0, stream>>>(act, d, dims, group, output.data_ptr())
        #define TF_DOWN_PREFILL_KIND(F) do { switch (d.kind) {  /* SC117 down banks: IQ4_NL, Q2_0 */ \
            case 20: TF_DOWN_PREFILL(F, 20); break; \
            case 42: TF_DOWN_PREFILL(F, 42); break; \
            default: TF_DOWN_PREFILL(F, -1); } } while (0)
        if (prefill && output.scalar_type() == at::kFloat) TF_DOWN_PREFILL_KIND(true);
        else if (prefill) TF_DOWN_PREFILL_KIND(false);
        #undef TF_DOWN_PREFILL_KIND
        #undef TF_DOWN_PREFILL
        else
            launch_down_decode(act, int(x.size(0)), d, dims, group, output.data_ptr(), output.scalar_type() == at::kFloat, stream);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    const int64_t count = output.numel();
    const int publish_blocks = int(std::min<int64_t>((count + 127) / 128, 65535));
    #define PUBLISH_OWNED(INDEX, FLOAT) \
        publish_owned<INDEX, FLOAT><<<publish_blocks, 128, 0, stream>>>( \
            picks.data_ptr<INDEX>(), local.data_ptr<int64_t>(), shared.data_ptr<float>(), \
            output.data_ptr(), count, dims, int(picks.size(1)), int(global_experts))
    #define PUBLISH_OWNED8(INDEX, FLOAT) \
        publish_owned8<INDEX, FLOAT><<<std::min((pairs * (dims / 8) + 127) / 128, 65535), 128, 0, stream>>>( \
            picks.data_ptr<INDEX>(), local.data_ptr<int64_t>(), shared.data_ptr<float>(), \
            output.data_ptr(), pairs, dims, int(picks.size(1)), int(global_experts))
    // Prompt-sized calls: 8 outputs per thread (bitwise the same publication).
    const bool wide = x.size(0) > DECODE_MAX_ROWS && dims % 8 == 0 && count <= std::numeric_limits<int>::max()
                      && (reinterpret_cast<uintptr_t>(output.data_ptr()) & 15) == 0
                      && (reinterpret_cast<uintptr_t>(shared.data_ptr()) & 15) == 0;
    const bool f32_output = output.scalar_type() == at::kFloat;
    if (wide && picks.scalar_type() == at::kLong) {
        if (f32_output) PUBLISH_OWNED8(int64_t, true); else PUBLISH_OWNED8(int64_t, false);
    } else if (wide) {
        if (f32_output) PUBLISH_OWNED8(int32_t, true); else PUBLISH_OWNED8(int32_t, false);
    } else if (picks.scalar_type() == at::kLong) {
        if (f32_output) PUBLISH_OWNED(int64_t, true);
        else PUBLISH_OWNED(int64_t, false);
    } else {
        if (f32_output) PUBLISH_OWNED(int32_t, true);
        else PUBLISH_OWNED(int32_t, false);
    }
    #undef PUBLISH_OWNED8
    #undef PUBLISH_OWNED
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (supplied_activation) {
        auto input=reinterpret_cast<const __half*>(activation.data_ptr());
        if (output.scalar_type()==at::kFloat)
            publish_activation_status<true><<<publish_blocks,128,0,stream>>>(input,output.data_ptr(),count);
        else publish_activation_status<false><<<publish_blocks,128,0,stream>>>(input,output.data_ptr(),count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

void gguf_experts_owned(const at::Tensor& x, const at::Tensor& picks,
    const at::Tensor& gate, int64_t gate_kind, int64_t gate_offset,
    const at::Tensor& up, int64_t up_kind, int64_t up_offset,
    const at::Tensor& down, int64_t down_kind, int64_t down_offset,
    int64_t width, int64_t first, int64_t global_experts, const at::Tensor& shared,
    at::Tensor local, at::Tensor activation, at::Tensor output) {
    gguf_experts_owned_impl(x, picks, gate, gate_kind, gate_offset, up, up_kind, up_offset,
        down, down_kind, down_offset, width, first, global_experts, shared, local, activation, output, at::Tensor());
}

void gguf_experts_owned_grouped(const at::Tensor& x, const at::Tensor& picks,
    const at::Tensor& gate, int64_t gate_kind, int64_t gate_offset,
    const at::Tensor& up, int64_t up_kind, int64_t up_offset,
    const at::Tensor& down, int64_t down_kind, int64_t down_offset,
    int64_t width, int64_t first, int64_t global_experts, const at::Tensor& shared,
    at::Tensor local, at::Tensor activation, at::Tensor output, at::Tensor grouping) {
    gguf_experts_owned_impl(x, picks, gate, gate_kind, gate_offset, up, up_kind, up_offset,
        down, down_kind, down_offset, width, first, global_experts, shared, local, activation, output, grouping);
}

// Private explicit candidate entrypoint; ordinary owned/grouped APIs are intact.
void gguf_experts_owned_grouped_gate_up(const at::Tensor& x, const at::Tensor& picks,
    const at::Tensor& gate, int64_t gate_kind, int64_t gate_offset,
    const at::Tensor& up, int64_t up_kind, int64_t up_offset,
    const at::Tensor& down, int64_t down_kind, int64_t down_offset,
    int64_t width, int64_t first, int64_t global_experts, const at::Tensor& shared,
    at::Tensor local, at::Tensor activation, at::Tensor output, at::Tensor grouping, int64_t rows) {
    TORCH_CHECK(rows == 2 || rows == 4, "private grouped gate/up requires exactly 2 or 4 rows");
    gguf_experts_owned_impl(x, picks, gate, gate_kind, gate_offset, up, up_kind, up_offset,
        down, down_kind, down_offset, width, first, global_experts, shared, local, activation, output, grouping, rows);
}

// Explicit bridge: only routed GU is supplied. Exact current grouped down,
// independently computed shared result and original publication remain owners.
void gguf_experts_owned_from_activation(const at::Tensor& x, const at::Tensor& picks,
    const at::Tensor& gate, int64_t gate_kind, int64_t gate_offset,
    const at::Tensor& up, int64_t up_kind, int64_t up_offset,
    const at::Tensor& down, int64_t down_kind, int64_t down_offset,
    int64_t width, int64_t first, int64_t global_experts, const at::Tensor& shared,
    at::Tensor local, at::Tensor activation, at::Tensor output, at::Tensor grouping) {
    gguf_experts_owned_impl(x,picks,gate,gate_kind,gate_offset,up,up_kind,up_offset,
        down,down_kind,down_offset,width,first,global_experts,shared,local,activation,output,grouping,0,true);
}

void gguf_experts_owned_gate_up(const at::Tensor& x,const at::Tensor& picks,
 const at::Tensor& gate,int64_t gate_kind,int64_t gate_offset,
 const at::Tensor& up,int64_t up_kind,int64_t up_offset,int64_t first,
 at::Tensor local,at::Tensor activation,at::Tensor grouping) {
 TORCH_CHECK(x.is_cuda()&&x.scalar_type()==at::kHalf&&x.is_contiguous()&&x.dim()==2
  &&x.size(0)>=1&&x.size(0)<=32&&x.size(1)==2560,"owned GU requires Half CUDA R1..32/K2560");
 const auto device=x.device();c10::cuda::CUDAGuard guard(device);
 const auto* prop=at::cuda::getDeviceProperties(x.get_device());
 TORCH_CHECK(prop->major==7&&prop->minor==0,"owned GU requires SM70");
 TORCH_CHECK(first==0||first==256,"owned GU requires E256 first0|256");
 TORCH_CHECK(picks.scalar_type()==at::kInt||picks.scalar_type()==at::kLong,"integer picks required");
 operand(picks,picks.scalar_type(),device,"owned GU picks");
 operand(local,at::kLong,device,"owned GU route scratch");
 operand(activation,at::kHalf,device,"owned GU activation");
 operand(grouping,at::kInt,device,"owned GU grouping scratch");
 TORCH_CHECK(picks.sizes()==at::IntArrayRef({x.size(0),11})&&local.sizes()==picks.sizes()
  &&activation.sizes()==at::IntArrayRef({picks.numel(),160}),"owned GU exact scratch geometry required");
 const auto g=packed_view(gate,gate_kind,gate_offset,2560,device);
 const auto u=packed_view(up,up_kind,up_offset,2560,device);
 TORCH_CHECK(gate.size(0)==256*160&&up.size(0)==256*160,"owned GU E256/width160 banks required");
 const int pairs=picks.numel(),experts=256;
 const int tiles=std::min(pairs,(pairs+15)/16+experts-1);
 const int words=2*pairs+2*experts+3*tiles+2;
 TORCH_CHECK(grouping.dim()==1&&grouping.numel()==words,"owned GU grouping extent differs");
 const at::Tensor* operands[]={&x,&picks,&gate,&up,&local,&activation,&grouping};
 for(const auto* tensor:operands)
  TORCH_CHECK(!tensor->requires_grad()&&!tensor->is_neg()&&!tensor->is_conj(),"owned GU plain non-autograd operands required");
 const at::Tensor* writes[]={&local,&activation,&grouping};
 for(int i=0;i<3;++i) {
  TORCH_CHECK(!writes[i]->is_inference()||c10::InferenceMode::is_enabled(),"inference output outside InferenceMode");
  tensorfold::check_disjoint(*writes[i],{&x,&picks,&gate,&up},"owned GU output/input overlap");
  for(int j=0;j<i;++j) TORCH_CHECK(!tensorfold::overlaps(*writes[i],*writes[j]),"owned GU output overlap");
 }
 int* ptr=grouping.data_ptr<int>();OwnedGrouping group{};
 group.counts=ptr;ptr+=experts;group.offsets=ptr;ptr+=experts+1;
 group.ordinal=ptr;ptr+=pairs;group.pairs=ptr;ptr+=pairs;
 group.tiles=ptr;ptr+=3*tiles;group.total=ptr;group.max_tiles=tiles;
 const auto stream=at::cuda::getCurrentCUDAStream();const int blocks=(pairs+127)/128;
 C10_CUDA_CHECK(cudaMemsetAsync(group.counts,0,experts*sizeof(int),stream));
 if(picks.scalar_type()==at::kLong) {
  initialize_owned_gu_slots<<<x.size(0),128,0,stream>>>(picks.data_ptr<int64_t>(),reinterpret_cast<__half*>(activation.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  group_owned_routes<<<blocks,128,0,stream>>>(picks.data_ptr<int64_t>(),local.data_ptr<int64_t>(),pairs,int(first),experts,group);
 } else {
  initialize_owned_gu_slots<<<x.size(0),128,0,stream>>>(picks.data_ptr<int32_t>(),reinterpret_cast<__half*>(activation.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  group_owned_routes<<<blocks,128,0,stream>>>(picks.data_ptr<int32_t>(),local.data_ptr<int64_t>(),pairs,int(first),experts,group);
 }
 C10_CUDA_KERNEL_LAUNCH_CHECK();
 group_owned_prefix<<<1,GROUP_PREFIX_THREADS,0,stream>>>(experts,group);C10_CUDA_KERNEL_LAUNCH_CHECK();
 group_owned_scatter<<<blocks,128,0,stream>>>(local.data_ptr<int64_t>(),pairs,group);C10_CUDA_KERNEL_LAUNCH_CHECK();
 launch_gate_up_decode(reinterpret_cast<const __half*>(x.data_ptr()),int(x.size(0)),g,u,11,group,reinterpret_cast<__half*>(activation.data_ptr()),stream);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void gguf_prepare_owned_gate_up() {
 cudaFuncAttributes attributes{};
 #define RESIDENT(FUNCTION) C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes,FUNCTION))
 RESIDENT(initialize_owned_gu_slots<int32_t>);RESIDENT(initialize_owned_gu_slots<int64_t>);
 RESIDENT(group_owned_routes<int32_t>);RESIDENT(group_owned_routes<int64_t>);
 RESIDENT(group_owned_prefix);RESIDENT(group_owned_scatter);RESIDENT(expert_gate_up_decode);RESIDENT(expert_gate_up_grouped<4>);
 RESIDENT(expert_gate_up_prefill<18>);RESIDENT(expert_gate_up_prefill<21>);RESIDENT(expert_gate_up_prefill<22>);
 RESIDENT(expert_gate_up_prefill<23>);RESIDENT(expert_gate_up_prefill<-1>);
 #undef RESIDENT
}
void gguf_prepare_owned_from_activation() {
    cudaFuncAttributes attributes{};
#define RESIDENT(...) C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes,__VA_ARGS__))
    RESIDENT(group_owned_routes<int32_t>); RESIDENT(group_owned_routes<int64_t>);
    RESIDENT(group_owned_prefix); RESIDENT(group_owned_scatter);
    RESIDENT((expert_down160_decode<true,1,false>)); RESIDENT((expert_down160_decode<true,4,false>));
    RESIDENT((expert_down160_decode<false,1,false>)); RESIDENT((expert_down160_decode<false,4,false>));
    RESIDENT((expert_down160_decode<true,1,true>)); RESIDENT((expert_down160_decode<true,4,true>));
    RESIDENT((expert_down160_decode<false,1,true>)); RESIDENT((expert_down160_decode<false,4,true>));
    RESIDENT((expert_down160_grouped<true,7>)); RESIDENT((expert_down160_grouped<true,-1>));
    RESIDENT((expert_down160_grouped<false,7>)); RESIDENT((expert_down160_grouped<false,-1>));
    RESIDENT(expert_gate_up_prefill<18>); RESIDENT(expert_gate_up_prefill<21>); RESIDENT(expert_gate_up_prefill<22>);
    RESIDENT(expert_gate_up_prefill<23>); RESIDENT(expert_gate_up_prefill<-1>);
    RESIDENT((expert_down160_prefill<true,20>)); RESIDENT((expert_down160_prefill<true,42>));
    RESIDENT((expert_down160_prefill<true,-1>)); RESIDENT((expert_down160_prefill<false,20>));
    RESIDENT((expert_down160_prefill<false,42>)); RESIDENT((expert_down160_prefill<false,-1>));
    RESIDENT(publish_owned<int32_t,true>); RESIDENT(publish_owned<int32_t,false>);
    RESIDENT(publish_owned<int64_t,true>); RESIDENT(publish_owned<int64_t,false>);
    RESIDENT((publish_owned8<int32_t,true>)); RESIDENT((publish_owned8<int32_t,false>));
    RESIDENT((publish_owned8<int64_t,true>)); RESIDENT((publish_owned8<int64_t,false>));
    RESIDENT(publish_activation_status<true>); RESIDENT(publish_activation_status<false>);
#undef RESIDENT
}
