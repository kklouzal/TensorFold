// SPDX-License-Identifier: Apache-2.0
// Wire layout derived from owned onecat_moe_packing.py, itself adapted from
// 1CatAI/1Cat-vLLM 1b57260a834d40faa32bfc6ac18f050414a8f042.
// See onecat_LICENSE.txt/onecat_NOTICE.txt. No coefficient requantization.
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cstdint>
#include "gguf_decode_tables.cuh"

namespace {
struct Raw { const uint8_t* data; int64_t expert_stride, row_stride, offset; };
__device__ __forceinline__ uint32_t u16(const uint8_t* p) { return uint32_t(p[0]) | (uint32_t(p[1]) << 8); }
__device__ __forceinline__ uint32_t u32(const uint8_t* p) {
    return u16(p) | (u16(p + 2) << 16);  // Original GGUF rows need not be aligned.
}
__device__ __forceinline__ uint32_t split_signs(uint32_t value) {
    uint32_t out = 0;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        out |= ((value >> (2*i)) & 1u) << i;
        out |= ((value >> (2*i + 1)) & 1u) << (16+i);
    }
    return out;
}
__device__ __forceinline__ uint32_t rounded_shift(uint32_t value, int shift) {
    if (shift <= 0) return value << -shift;
    const uint32_t kept = value >> shift, remainder = value & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1);
    return kept + (remainder > halfway || (remainder == halfway && (kept & 1u)));
}
// Exact finite Half*integer -> Half RN-even. Integer arithmetic prevents FTZ,
// double rounding, or compiler contraction. Exceptional wire values follow the
// pinned x86 NumPy producer: quiet/preserve Half NaN payload; inf*0 -> -qNaN.
__device__ __forceinline__ uint16_t half_scale(uint16_t h, int multiplier) {
    const int exponent = (h >> 10) & 31, fraction = h & 1023;
    const uint16_t sign = (h & 0x8000u) ^ (multiplier < 0 ? 0x8000u : 0u);
    if (exponent == 31) {
        if (fraction) return h | 0x0200u;
        if (!multiplier) return 0xfe00u;
        return sign | 0x7c00u;
    }
    if (!multiplier || (!exponent && !fraction)) return sign;
    const uint32_t magnitude = uint32_t(exponent ? fraction + 1024 : fraction)
        * uint32_t(multiplier < 0 ? -multiplier : multiplier);
    const int power = exponent ? exponent - 25 : -24;
    int top = 31 - __clz(magnitude) + power;
    if (top < -14) return sign | uint16_t(magnitude << (power + 24));
    uint32_t mantissa = rounded_shift(magnitude, (31 - __clz(magnitude)) - 10);
    if (mantissa == 2048) { mantissa = 1024; ++top; }
    if (top > 15) return sign | 0x7c00u;
    return sign | uint16_t((top + 15) << 10) | uint16_t(mantissa - 1024);
}
__device__ __forceinline__ int wire_row(int lane) {
    return ((lane >> 2) & 3) * 8 + (lane & 3) + (lane & 16 ? 4 : 0);
}
template<class I> __device__ bool valid_row(const I* picks, int row) {
    #pragma unroll
    for (int j = 0; j < 11; ++j) {
        const int64_t id = picks[row*11+j];
        if (j == 10 ? id != 512 : (id < 0 || id >= 512)) return false;
        #pragma unroll
        for (int k = 0; k < j; ++k) if (int64_t(picks[row*11+k]) == id) return false;
    }
    return true;
}
template<int Kind> __device__ __forceinline__ uint32_t code_word(const uint8_t* b, int half, int component, int word) {
    if constexpr (Kind == 22) {
        return u32(b + (component ? 34 : 2) + (half*4+word)*4);
    } else if constexpr (Kind == 23) {
        uint32_t out = 0;
        const uint8_t* values = b + 8 + (half*4+component)*16;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int at = word*8+j;
            out |= uint32_t(at < 16 ? values[at] & 15 : values[at-16] >> 4) << (4*j);
        }
        return out;
    } else {
        if (component < 2) {
            const int step = half*4 + component*2 + word/2;
            return u32(b + 2 + step*8 + (word%2)*4);
        }
        uint32_t signs;
        if constexpr (Kind == 21) signs = u32(b + 74 + (half*4+word)*4);
        else {
            const uint32_t aux = u32(b + 66 + (half*4+word)*4);
            signs = uint32_t(tf_ksigns_iq2xs[aux & 127u])
                | (uint32_t(tf_ksigns_iq2xs[(aux >> 7) & 127u]) << 8)
                | (uint32_t(tf_ksigns_iq2xs[(aux >> 14) & 127u]) << 16)
                | (uint32_t(tf_ksigns_iq2xs[(aux >> 21) & 127u]) << 24);
        }
        return split_signs(signs);
    }
}
template<int Kind> __device__ __forceinline__ uint32_t meta_word(const uint8_t* b, int half, int word) {
    if constexpr (Kind == 22) {
        if (word == 0) return u32(b + 66 + half*4);
        if (word == 1) return u16(b);
        if (word == 2) return u32(b + 74 + half*4);
        return 0;
    } else {
        if constexpr (Kind == 21) if (word == 0) return u32(b + 66 + half*4);
        uint32_t scales = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int step = half*4+j;
            const uint32_t nibble = Kind == 21 ? ((b[106+step/2] >> (4*(step%2))) & 15u)
                                               : (u32(b + 66 + step*4) >> 28);
            scales |= nibble << (4*j);
        }
        return u16(b) | (scales << 16);
    }
}
template<int Kind, class I, bool Union = false> __global__ void pack(Raw gate, Raw up, const I* picks, int rows, int first,
        uint8_t* gc, uint8_t* gs, uint8_t* uc, uint8_t* us) {
    __shared__ unsigned active;
    const int expert = blockIdx.x, tid = threadIdx.x;
    if (tid < 32) {
        bool hit = false;
        if constexpr (Union) {
            // Presence across the complete bounded prefill, not only its first
            // 32 rows. Every lane participates in the final warp ballot.
            for (int row = tid; row < rows; row += 32) {
                if (valid_row(picks, row)) {
                    #pragma unroll
                    for (int j = 0; j < 10; ++j) hit = hit || int64_t(picks[row*11+j]) == first + expert;
                }
            }
        } else {
            if (tid < rows && valid_row(picks, tid)) {
                #pragma unroll
                for (int j = 0; j < 10; ++j) hit = hit || int64_t(picks[tid*11+j]) == first + expert;
            }
        }
        const unsigned mask = __ballot_sync(0xffffffffu, hit);
        if (!tid) active = mask;
    }
    __syncthreads();
    if (!active) return;  // One immutable CTA/expert; reused picks cannot duplicate writes.
    constexpr int block_bytes = Kind == 18 ? 98 : Kind == 21 ? 110 : Kind == 22 ? 82 : 136;
    constexpr int components = Kind == 22 ? 2 : Kind == 23 ? 4 : 3;
    constexpr int code_bytes = 5*20*components*32*4*4;
    constexpr int scale_bytes = Kind == 22 ? 51200 : Kind == 18 ? 12800 : 25600;
    #pragma unroll
    for (int side = 0; side < 2; ++side) {
        const Raw input = side ? up : gate;
        uint8_t* codes = (side ? uc : gc) + int64_t(expert)*code_bytes;
        uint8_t* scales = (side ? us : gs) + int64_t(expert)*scale_bytes;
        for (int at = tid; at < code_bytes/4; at += blockDim.x) {
            const int word = at%4, lane = (at/4)%32, component = (at/(4*32))%components;
            const int group = (at/(4*32*components))%20, tile = at/(4*32*components*20);
            const uint8_t* b = input.data + int64_t(expert)*input.expert_stride
                + int64_t(tile*32+wire_row(lane))*input.row_stride + input.offset + (group/2)*block_bytes;
            reinterpret_cast<uint32_t*>(codes)[at] = code_word<Kind>(b, group%2, component, word);
        }
        if constexpr (Kind == 23) {
            for (int at = tid; at < scale_bytes/2; at += blockDim.x) {
                const int member = at%4, lane = (at/4)%32, group = (at/(4*32))%20, tile = at/(4*32*20);
                const uint8_t* b = input.data + int64_t(expert)*input.expert_stride
                    + int64_t(tile*32+wire_row(lane))*input.row_stride + input.offset + (group/2)*block_bytes;
                const int s = group%2*4+member;
                const int scale = ((b[4+s/2] >> (4*(s%2))) & 15) | (((u16(b+2) >> (2*s)) & 3) << 4);
                reinterpret_cast<uint16_t*>(scales)[at] = half_scale(uint16_t(u16(b)), scale-32);
            }
        } else {
            constexpr int words = Kind == 22 ? 4 : Kind == 21 ? 2 : 1;
            for (int at = tid; at < scale_bytes/4; at += blockDim.x) {
                const int word = at%words, lane = (at/words)%32, group = (at/(words*32))%20, tile = at/(words*32*20);
                const uint8_t* b = input.data + int64_t(expert)*input.expert_stride
                    + int64_t(tile*32+wire_row(lane))*input.row_stride + input.offset + (group/2)*block_bytes;
                reinterpret_cast<uint32_t*>(scales)[at] = meta_word<Kind>(b, group%2, word);
            }
        }
    }
}
} // namespace

void onecat_gpu_packing_prepare() {
#define P(K,I) { cudaFuncAttributes a{}; C10_CUDA_CHECK(cudaFuncGetAttributes(&a, pack<K,I>)); }
    P(18,int) P(18,int64_t) P(21,int) P(21,int64_t) P(22,int) P(22,int64_t) P(23,int) P(23,int64_t)
#define U(K,I) { cudaFuncAttributes a{}; C10_CUDA_CHECK(cudaFuncGetAttributes(&a, pack<K,I,true>)); }
    U(18,int) U(18,int64_t) U(21,int) U(21,int64_t) U(22,int) U(22,int64_t) U(23,int) U(23,int64_t)
#undef U
#undef P
}
void onecat_gpu_packing_launch(const at::Tensor& gate, const at::Tensor& up, const at::Tensor& picks,
        at::Tensor gc, at::Tensor gs, at::Tensor uc, at::Tensor us, int kind, int first,
        int64_t gate_offset, int64_t up_offset) {
    auto raw = [](const at::Tensor& t, int64_t offset) {
        return Raw{t.data_ptr<uint8_t>(), t.dim()==3 ? t.stride(0) : 160*t.stride(0),
                   t.dim()==3 ? t.stride(1) : t.stride(0), offset};
    };
    const Raw g = raw(gate,gate_offset), u = raw(up,up_offset);
    const auto stream = at::cuda::getCurrentCUDAStream();
#define RUN(K,I) pack<K,I><<<256,256,0,stream>>>(g,u,picks.data_ptr<I>(),picks.size(0),first,gc.data_ptr<uint8_t>(),gs.data_ptr<uint8_t>(),uc.data_ptr<uint8_t>(),us.data_ptr<uint8_t>())
#define F(K) if(kind==K) {if(picks.scalar_type()==at::kLong) {RUN(K,int64_t);} else {RUN(K,int);}}
    F(18) F(21) F(22) F(23)
#undef F
#undef RUN
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void onecat_gpu_union_packing_launch(const at::Tensor& gate, const at::Tensor& up, const at::Tensor& picks,
        at::Tensor gc, at::Tensor gs, at::Tensor uc, at::Tensor us, int kind, int first,
        int64_t gate_offset, int64_t up_offset) {
    auto raw = [](const at::Tensor& tensor, int64_t offset) {
        return Raw{tensor.data_ptr<uint8_t>(), tensor.dim()==3 ? tensor.stride(0) : 160*tensor.stride(0),
                   tensor.dim()==3 ? tensor.stride(1) : tensor.stride(0), offset};
    };
    const Raw g=raw(gate,gate_offset), u=raw(up,up_offset);
    const auto stream=at::cuda::getCurrentCUDAStream();
#define UNION(K,I) pack<K,I,true><<<256,256,0,stream>>>(g,u,picks.data_ptr<I>(),picks.size(0),first,gc.data_ptr<uint8_t>(),gs.data_ptr<uint8_t>(),uc.data_ptr<uint8_t>(),us.data_ptr<uint8_t>())
#define FORMAT(K) if(kind==K) {if(picks.scalar_type()==at::kLong) {UNION(K,int64_t);} else {UNION(K,int);}}
    FORMAT(18) FORMAT(21) FORMAT(22) FORMAT(23)
#undef FORMAT
#undef UNION
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
