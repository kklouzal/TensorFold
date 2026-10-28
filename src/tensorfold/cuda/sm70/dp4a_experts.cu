// Standalone private Q8-activation profile; NOT the exact FP16 coefficient ABI.
// GGML format/sign/codebook facts adapted from llama.cpp b3fa69955ea81c3b60e77e3c1e2224c4a826ffb9,
// ggml/src/ggml-cuda/vecdotq.cuh (MIT, Copyright2023-2026 ggml authors).
// Changes: FP32 activation scale, exact odd-scale numerators (no integer truncation),
// IQ4_NL crop-aligned zero padding. Keep LICENSES/GGML-MIT.txt and notices.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "gguf_decode_tables.cuh"

namespace {
constexpr int NONFINITE_INPUT=1, NONFINITE_WEIGHT=2, NONFINITE_HIDDEN=4, INVALID_PICK=8;
struct Q32 { float d; signed char q[32]; };
static_assert(sizeof(Q32)==36, "private Q8_32 ABI");
__device__ float hw(const unsigned char* p) { return __half2float(__ushort_as_half(unsigned(p[0]) | unsigned(p[1])<<8)); }
__device__ unsigned u32(const unsigned char* p) { return unsigned(p[0]) | unsigned(p[1])<<8 | unsigned(p[2])<<16 | unsigned(p[3])<<24; }
__device__ int pack4(int a,int b,int c,int d) { return int((unsigned(a)&255u)|((unsigned(b)&255u)<<8)|((unsigned(c)&255u)<<16)|((unsigned(d)&255u)<<24)); }

// One group, exact signed integer arithmetic, independently oracle-checked.
__device__ int numerator(const unsigned char* block,int kind,int group,const Q32& a) {
    int sums[2]={0,0};
    for(int start=0;start<32;start+=4) {
        int w[4];
        for(int j=0;j<4;++j) {
            const int i=start+j, sub=i/8, e=i%8;
            if(kind==18) {
                const unsigned aux=u32(block+66+group*4);
                const unsigned signs=tf_ksigns_iq2xs[(aux>>(7*sub))&127];
                const unsigned grid=tf_iq3xxs_grid[block[2+group*8+sub*2+e/4]];
                const int magnitude=(grid>>(8*(e%4)))&255;
                w[j]=(signs&(1u<<e))?-magnitude:magnitude;
            } else if(kind==22) {
                const int code=block[2+group*4+sub] | ((unsigned(block[66+group])<<(8-2*sub))&0x300);
                const unsigned long long grid=tf_iq2s_grid[code];
                const int magnitude=(grid>>(8*e))&255;
                w[j]=(block[34+group*4+sub]&(1u<<e))?-magnitude:magnitude;
            } else {
                const int code=(block[2+i%16]>>(4*(i/16)))&15;
                constexpr int values[16]={TF_GGUF_IQ4NL_VALUES};
                w[j]=values[code];
            }
        }
        const int q=pack4(a.q[start],a.q[start+1],a.q[start+2],a.q[start+3]);
        // Same signed DP4A primitive as ggml_cuda_dp4a on __CUDA_ARCH__>=610.
        sums[start/16]=__dp4a(pack4(w[0],w[1],w[2],w[3]),q,sums[start/16]);
    }
    if(kind==18) return (2*int(u32(block+66+group*4)>>28)+1)*(sums[0]+sums[1]);
    if(kind==22) return (2*int(block[74+group]&15)+1)*sums[0]+(2*int(block[74+group]>>4)+1)*sums[1];
    return sums[0]+sums[1];
}

__global__ void quantize(const __half* x,Q32* out,int columns,int groups,int prefix,int* status,int failure_bit) {
    const int row=blockIdx.x, group=blockIdx.y, lane=threadIdx.x;
    const int index=group*32+lane-prefix;
    const float value=index>=0 && index<columns?__half2float(x[size_t(row)*columns+index]):0.0f;
    const unsigned bad=__ballot_sync(0xffffffffu,!isfinite(value));
    float maximum=isfinite(value)?fabsf(value):0.0f;
    for(int offset=16;offset;offset>>=1) maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffffu,maximum,offset));
    Q32& target=out[size_t(row)*groups+group];
    if(bad) {
        target.q[lane]=0;
        if(lane==0) { target.d=0.0f; atomicOr(status,failure_bit); }
        return; // No float-to-integer conversion of a nonfinite operand.
    }
    const float delta=maximum==0.0f?0.0f:__fdiv_rn(maximum,127.0f);
    const int q=maximum==0.0f?0:int(roundf(__fdiv_rn(value,delta)));
    target.q[lane]=static_cast<signed char>(q); // Proved [-127,127] for finite FP16.
    if(lane==0) target.d=delta;
}

__global__ void project(const Q32* input,const int* picks,const unsigned char* weight,
        int kind,int bytes,int stride,int groups,int outputs,int slots,bool down,
        void* output,bool half_output,int* status) {
    const int lane=threadIdx.x&31, warp=threadIdx.x>>5;
    const int column=blockIdx.x*4+warp, pair=blockIdx.y;
    if(column>=outputs) return;
    const int expert=picks[pair];
    if(expert<0 || expert>512) { if(lane==0) atomicOr(status,INVALID_PICK); }
    float total=0.0f;
    if(expert>=0 && expert<512) {
        const size_t row=size_t(expert)*outputs+column;
        const Q32* values=input+size_t(down?pair:pair/slots)*groups;
        for(int group=lane;group<groups;group+=32) {
            const unsigned char* block=weight+row*stride+size_t(kind==20?group:group/8)*bytes;
            const float scale=hw(block);
            if(!isfinite(scale)) { atomicOr(status,NONFINITE_WEIGHT); continue; }
            const int sum=numerator(block,kind,kind==20?0:group%8,values[group]);
            const float factor=kind==18?.25f:kind==22?.125f:1.0f;
            const float scaled=__fmul_rn(__int2float_rn(sum),factor);
            const float term=__fmul_rn(__fmul_rn(scale,values[group].d),scaled);
            total=__fadd_rn(total,term);
        }
        for(int offset=16;offset;offset>>=1) total=__fadd_rn(total,__shfl_down_sync(0xffffffffu,total,offset));
    }
    if(lane==0) {
        // Shared512 is an exact-path placeholder; the existing bank replaces it.
        const float value=expert==512?0.0f:expert<0 || expert>512?__int_as_float(0x7fffffff):total;
        const size_t at=size_t(pair)*outputs+column;
        if(half_output) static_cast<__half*>(output)[at]=__float2half_rn(value);
        else static_cast<float*>(output)[at]=value;
    }
}

__global__ void activate(const float* gate,const float* up,__half* out,int count) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) out[i]=__float2half_rn(__fmul_rn(__fdiv_rn(gate[i],__fadd_rn(1.0f,expf(-gate[i]))),up[i]));
}
__global__ void mask(void* output,int count,bool half,int* status) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count && *status) {
        if(half) static_cast<__half*>(output)[i]=__ushort_as_half(0x7fff);
        else static_cast<float*>(output)[i]=__int_as_float(0x7fffffff);
    }
}
__global__ void debug_groups(const Q32* input,const unsigned char* weights,int kind,int bytes,int stride,
        int groups,int* integers,float* values,int* status) {
    const int group=blockIdx.x*blockDim.x+threadIdx.x;
    if(group>=groups) return;
    const unsigned char* block=weights+size_t(kind==20?group:group/8)*bytes;
    const float scale=hw(block);
    if(!isfinite(scale)) atomicOr(status,NONFINITE_WEIGHT);
    const int sum=numerator(block,kind,kind==20?0:group%8,input[group]);
    integers[group]=sum;
    const float factor=kind==18?.25f:kind==22?.125f:1.0f;
    values[group]=__fmul_rn(__fmul_rn(scale,input[group].d),__fmul_rn(__int2float_rn(sum),factor));
}
}

void tf_dp4a_quantize(const at::Tensor& x,at::Tensor q,int prefix,at::Tensor status,int bit) {
    quantize<<<dim3(x.size(0),q.size(1)),32,0,at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(x.data_ptr()),reinterpret_cast<Q32*>(q.data_ptr()),x.size(1),q.size(1),prefix,status.data_ptr<int>(),bit);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void tf_dp4a_project(const at::Tensor& q,const at::Tensor& picks,const at::Tensor& weights,int kind,bool down,at::Tensor out,at::Tensor status) {
    const int outputs=out.size(2), slots=picks.size(1), bytes=kind==18?98:kind==22?82:18;
    project<<<dim3((outputs+3)/4,picks.numel()),128,0,at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const Q32*>(q.data_ptr()),picks.data_ptr<int>(),weights.data_ptr<unsigned char>(),kind,bytes,weights.stride(0),q.size(1),outputs,slots,down,out.data_ptr(),out.scalar_type()==at::kHalf,status.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void tf_dp4a_activate(const at::Tensor& gate,const at::Tensor& up,at::Tensor out) {
    activate<<<(out.numel()+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(gate.data_ptr<float>(),up.data_ptr<float>(),reinterpret_cast<__half*>(out.data_ptr()),out.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void tf_dp4a_mask(at::Tensor out,at::Tensor status) {
    mask<<<(out.numel()+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(out.data_ptr(),out.numel(),out.scalar_type()==at::kHalf,status.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void tf_dp4a_debug(const at::Tensor& q,const at::Tensor& weights,int kind,at::Tensor integers,at::Tensor values,at::Tensor status) {
    debug_groups<<<(q.size(1)+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const Q32*>(q.data_ptr()),weights.data_ptr<unsigned char>(),kind,kind==18?98:kind==22?82:18,weights.stride(0),q.size(1),integers.data_ptr<int>(),values.data_ptr<float>(),status.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
