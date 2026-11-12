// SPDX-License-Identifier: Apache-2.0
// Arithmetic derives from TensorFold's incumbent gdn.cu. Scheduling was
// motivated by 1Cat-vLLM75e51a5b gdn_verify_sm70.cu (value8/64-thread cfg1).
// No 1Cat implementation is copied: original TF coefficient/cast/update trees
// are retained, including decay-before-dot and --fmad=false forward/replay.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <optional>

namespace {
constexpr int NK=2, NV=6, D=128, C=1280, PW=2060;
__device__ __forceinline__ float half_round(float x) { return __half2float(__float2half_rn(x)); }
__device__ __forceinline__ float warp_sum(float x) {
    for (int offset=16; offset; offset>>=1) x += __shfl_xor_sync(0xffffffffu, x, offset);
    return x;
}
__device__ __forceinline__ float sigmoid(float x) { return 1.0f/(1.0f+expf(-x)); }
__device__ __forceinline__ float softplus(float x) { return x>20.0f ? x : log1pf(expf(x)); }
__device__ __forceinline__ float convolve(const __half* p, const __half* history, const __half* weight, int row, int channel) {
    float acc=0.0f;
#pragma unroll
    for (int tap=0; tap<4; ++tap) {
        const int at=row+tap;
        const float x=at<3 ? __half2float(history[at*C+channel]) : __half2float(p[(size_t)(at-3)*PW+channel]);
        acc=acc+__half2float(weight[channel*4+tap])*x;
    }
    return half_round(acc/(1.0f+expf(-acc)));
}

template<bool Negative>
__global__ __launch_bounds__(128) void front(const __half* p, const __half* history, const __half* weight,
        const float* parameter, const float* bias, float* q, float* k, __half* v, float* g, float* beta) {
    const int hk=blockIdx.x, row=blockIdx.y, t=threadIdx.x, warp=t>>5, lane=t&31;
    __shared__ float qs[D], ks[D];
    qs[t]=convolve(p,history,weight,row,hk*D+t);
    ks[t]=convolve(p,history,weight,row,NK*D+hk*D+t);
    __syncthreads();
    if (warp<2) {
        float* values=warp==0 ? qs : ks;
        float x[4], ss=0.0f;
#pragma unroll
        for (int i=0;i<4;++i) { x[i]=values[lane*4+i]; ss=ss+x[i]*x[i]; }
        ss=warp_sum(ss);
        float inv=1.0f/sqrtf(ss+1e-6f);
        if (warp==0) inv=inv*(1.0f/sqrtf((float)D));
        float* dest=warp==0 ? q : k;
#pragma unroll
        for (int i=0;i<4;++i) dest[((size_t)row*NK+hk)*D+lane*4+i]=x[i]*inv;
    }
#pragma unroll
    for (int repeat=0;repeat<3;++repeat) {
        const int hv=hk*3+repeat;
        v[((size_t)row*NV+hv)*D+t]=__float2half_rn(convolve(p,history,weight,row,2*NK*D+hv*D+t));
        if (t==0) {
            const float b=__half2float(p[(size_t)row*PW+C+NV*D+hv]);
            const float a=__half2float(p[(size_t)row*PW+C+NV*D+NV+hv]);
            const float rate=Negative ? parameter[hv] : -expf(parameter[hv]);
            g[row*NV+hv]=expf(rate*softplus(a+bias[hv]));
            beta[row*NV+hv]=half_round(sigmoid(b));
        }
    }
}

// EXACT incumbent update order, reused by forward and prefix replay.
__device__ __forceinline__ void step(float (&s)[4][4], const float (&key)[4],
        const float (&value)[4], float g, float beta) {
#pragma unroll
    for (int j=0;j<4;++j) {
        float dot=0.0f;
#pragma unroll
        for (int i=0;i<4;++i) { s[j][i]=s[j][i]*g; dot=dot+s[j][i]*key[i]; }
        dot=warp_sum(dot);
        const float delta=(value[j]-dot)*beta;
#pragma unroll
        for (int i=0;i<4;++i) s[j][i]=s[j][i]+key[i]*delta;
    }
}

template<int Preload, bool Replay>
__global__ __launch_bounds__(64) void core(const float* initial, const float* q, const float* k,
        const __half* v, const float* g, const float* beta, int rows, const int* keep,
        float* final, __half* raw) {
    const int hv=blockIdx.y, hk=hv/3, warp=threadIdx.x>>5, lane=threadIdx.x&31;
    const int first=blockIdx.x*8+warp*4;
    constexpr int Count=Preload ? Preload : 1;
    float pq[Count][4], pk[Count][4], pv[Count][4], pg[Count], pb[Count];
    if constexpr (Preload>0) {
#pragma unroll
        for (int r=0;r<Preload;++r) {
            pg[r]=r<rows ? g[r*NV+hv] : 0.0f; pb[r]=r<rows ? beta[r*NV+hv] : 0.0f;
#pragma unroll
            for (int i=0;i<4;++i) {
                pk[r][i]=r<rows ? k[((size_t)r*NK+hk)*D+lane*4+i] : 0.0f;
                pv[r][i]=r<rows ? __half2float(v[((size_t)r*NV+hv)*D+first+i]) : 0.0f;
                if constexpr (!Replay) pq[r][i]=r<rows ? q[((size_t)r*NK+hk)*D+lane*4+i] : 0.0f;
            }
        }
    }
    float s[4][4];
#pragma unroll
    for (int j=0;j<4;++j)
#pragma unroll
        for (int i=0;i<4;++i) s[j][i]=initial[((size_t)hv*D+first+j)*D+lane*4+i];
    const int live=Replay ? rows : (keep ? max(0,min(rows,*keep)) : rows);
    if (!live) {
#pragma unroll
        for (int j=0;j<4;++j)
#pragma unroll
            for (int i=0;i<4;++i) final[((size_t)hv*D+first+j)*D+lane*4+i]=s[j][i];
    }
#pragma unroll
    for (int r=0;r<(Preload ? Preload : rows);++r) {
        if constexpr (Preload>0) { if (r>=rows) break; }
        float key[4], value[4], query[4];
        float decay, b;
        if constexpr (Preload>0) {
            decay=pg[r]; b=pb[r];
#pragma unroll
            for (int i=0;i<4;++i) { key[i]=pk[r][i]; value[i]=pv[r][i]; if constexpr (!Replay) query[i]=pq[r][i]; }
        } else {
            decay=g[r*NV+hv]; b=beta[r*NV+hv];
#pragma unroll
            for (int i=0;i<4;++i) {
                key[i]=k[((size_t)r*NK+hk)*D+lane*4+i];
                value[i]=__half2float(v[((size_t)r*NV+hv)*D+first+i]);
                if constexpr (!Replay) query[i]=q[((size_t)r*NK+hk)*D+lane*4+i];
            }
        }
        step(s,key,value,decay,b);
        if constexpr (!Replay) {
#pragma unroll
            for (int j=0;j<4;++j) {
                float out=0.0f;
#pragma unroll
                for (int i=0;i<4;++i) out=out+s[j][i]*query[i];
                out=warp_sum(out);
                if (lane==0) raw[((size_t)r*NV+hv)*D+first+j]=__float2half_rn(half_round(out));
            }
        }
        if (r+1==live) {
#pragma unroll
            for (int j=0;j<4;++j)
#pragma unroll
                for (int i=0;i<4;++i) final[((size_t)hv*D+first+j)*D+lane*4+i]=s[j][i];
        }
    }
}

__global__ __launch_bounds__(128) void back(const __half* p, const __half* norm, float eps, __half* out, float* sums) {
    const int hv=blockIdx.x, row=blockIdx.y, t=threadIdx.x, warp=t>>5, lane=t&31;
    __shared__ float y[D], inv;
    y[t]=__half2float(out[((size_t)row*NV+hv)*D+t]);
    __syncthreads();
    if (warp==0) {
        float ss=0.0f;
#pragma unroll
        for (int i=0;i<4;++i) { const float value=y[lane*4+i]; ss=ss+value*value; }
        ss=warp_sum(ss);
        if (lane==0) inv=1.0f/sqrtf(ss/(float)D+eps);
    }
    __syncthreads();
    const float normalized=half_round(half_round(y[t]*inv)*__half2float(norm[t]));
    const float z=__half2float(p[(size_t)row*PW+C+hv*D+t]);
    const float value=half_round(normalized*sigmoid(z));
    out[((size_t)row*NV+hv)*D+t]=__float2half_rn(value);
    const float total=warp_sum(value);
    if (lane==0) sums[(size_t)row*(NV*4)+hv*4+warp]=total;
}

template<bool Replay>
void launch_core(const float* initial, const float* q, const float* k, const __half* v,
        const float* g, const float* b, int rows, const int* keep, float* final, __half* out, cudaStream_t stream) {
    const dim3 grid(16,NV);
#define RUN(N) core<N,Replay><<<grid,64,0,stream>>>(initial,q,k,v,g,b,rows,keep,final,out)
    if (rows>0 && rows<=1) { RUN(1); }
    else if (rows>0 && rows<=2) { RUN(2); }
    else if (rows>0 && rows<=4) { RUN(4); }
    else if (rows>0 && rows<=5) { RUN(5); }
    else if (rows>0 && rows<=8) { RUN(8); }
    else { RUN(0); }
#undef RUN
}
}

void value8_chain_cuda(const at::Tensor& p, const at::Tensor& history, const at::Tensor& weight,
        const at::Tensor& initial, const at::Tensor& parameter, const at::Tensor& bias, const at::Tensor& norm,
        float eps, int rows, at::Tensor& out, at::Tensor& sums, at::Tensor& final, at::Tensor& k,
        at::Tensor& v, at::Tensor& g, at::Tensor& b, at::Tensor& q, const std::optional<at::Tensor>& keep, bool negative) {
    auto stream=at::cuda::getCurrentCUDAStream();
    if (rows) {
#define FRONT(N) front<N><<<dim3(NK,rows),128,0,stream>>>(reinterpret_cast<const __half*>(p.data_ptr()),reinterpret_cast<const __half*>(history.data_ptr()),reinterpret_cast<const __half*>(weight.data_ptr()),parameter.data_ptr<float>(),bias.data_ptr<float>(),q.data_ptr<float>(),k.data_ptr<float>(),reinterpret_cast<__half*>(v.data_ptr()),g.data_ptr<float>(),b.data_ptr<float>())
        if (negative) { FRONT(true); } else { FRONT(false); }
#undef FRONT
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    launch_core<false>(initial.data_ptr<float>(),q.data_ptr<float>(),k.data_ptr<float>(),reinterpret_cast<const __half*>(v.data_ptr()),
        g.data_ptr<float>(),b.data_ptr<float>(),rows,keep ? keep->data_ptr<int>() : nullptr,final.data_ptr<float>(),reinterpret_cast<__half*>(out.data_ptr()),stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (rows) {
        back<<<dim3(NV,rows),128,0,stream>>>(reinterpret_cast<const __half*>(p.data_ptr()),reinterpret_cast<const __half*>(norm.data_ptr()),eps,reinterpret_cast<__half*>(out.data_ptr()),sums.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}
void value8_replay_cuda(const at::Tensor& initial,const at::Tensor& k,const at::Tensor& v,const at::Tensor& g,
        const at::Tensor& b,int rows,at::Tensor& final) {
    launch_core<true>(initial.data_ptr<float>(),nullptr,k.data_ptr<float>(),reinterpret_cast<const __half*>(v.data_ptr()),
        g.data_ptr<float>(),b.data_ptr<float>(),rows,nullptr,final.data_ptr<float>(),nullptr,at::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void value8_prepare_cuda() {
    cudaFuncAttributes attributes;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes,front<true>));
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes,front<false>));
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes,back));
#define ATTRIBUTE(N,R) C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes,core<N,R>))
    ATTRIBUTE(0,false); ATTRIBUTE(1,false); ATTRIBUTE(2,false); ATTRIBUTE(4,false); ATTRIBUTE(5,false); ATTRIBUTE(8,false);
    ATTRIBUTE(0,true); ATTRIBUTE(1,true); ATTRIBUTE(2,true); ATTRIBUTE(4,true); ATTRIBUTE(5,true); ATTRIBUTE(8,true);
#undef ATTRIBUTE
}
