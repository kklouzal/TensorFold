// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from 1CatAI/1Cat-vLLM 1b57260, gguf_moe_gu_sm70.cu.
// TensorFold modifications: E256 global remap, complete bounded panels, no Q8_1,
// explicit original/upstream activation epilogues, initialized foreign/shared slots.
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <vector>
#include <cstdint>
#include "../kernels/tensor_contracts.cuh"
#define DMV_FULLK 0
namespace dmvns {
#include "onecat_dmv13_core.cuh"
}
namespace owned_onecat {
using namespace dmvns;
constexpr int TN=2, MAXT=8, MAXR=256, MAXC=32;
struct EArgs {
 const uint4 *gc,*gs,*uc,*us;
 long long cstride,sstride;
 const half* x;
 int ldx;
 const void* ids;
 int routes,top_k,n,K,first;
 bool wide;
 half* hout;
 const uint4* tab;
};
__device__ bool valid_row(const void* ids,bool wide,int row) {
 int64_t seen[11];
 for(int j=0;j<11;++j) {
  const int at=row*11+j;
  const int64_t id=wide?static_cast<const int64_t*>(ids)[at]:static_cast<const int*>(ids)[at];
  if(j==10 ? id!=512 : (id<0 || id>=512)) return false;
  for(int k=0;k<j;++k) if(seen[k]==id) return false;
  seen[j]=id;
 }
 return true;
}
__global__ void initialize(const void* ids,bool wide,half* out) {
 __shared__ bool valid;
 if(threadIdx.x==0) valid=valid_row(ids,wide,blockIdx.x);
 __syncthreads();
 const half value=valid?__float2half_rn(0.f):__float2half_rn(__int_as_float(0x7fffffff));
 for(int i=threadIdx.x;i<11*160;i+=blockDim.x) out[blockIdx.x*11*160+i]=value;
}
template <int FMT, int KW, bool Upstream>
__global__ void __launch_bounds__(32 * KW * TN) moe_gu(EArgs a) {
  extern __shared__ uint4 smem[];
  __shared__ int sid[MAXR], s_expert, s_cnt, s_route[MAXC], s_tok[MAXC];
  const int t = threadIdx.x, warp = t / 32, lane = t % 32;
  // Distinct experts in ascending id order: slot u = u-th set bit of a 512-bit
  // presence mask.
  __shared__ unsigned s_mask[16];
  __shared__ bool valid[8];
  // initialize precedes every GU panel on the same stream. Slot10 is
  // immutable here: global sharedID512 never enters an E256 presence mask.
  // Its first Half is a paid row-status cell (+0 valid / NaN poisoned).
  if(t<a.routes/a.top_k) valid[t]=__half_as_ushort(a.hout[(t*a.top_k+10)*a.n])==0;
  if (t < 16) s_mask[t] = 0u;
  if (t == 0) s_expert = -1;
  __syncthreads();
  for (int r = t; r < a.routes; r += blockDim.x) {
    const int64_t global = a.wide ? static_cast<const int64_t*>(a.ids)[r] : static_cast<const int*>(a.ids)[r];
    const int e = valid[r/a.top_k] && global >= a.first && global < a.first + 256 ? global - a.first : -1;
    sid[r] = e;
    if (e >= 0) atomicOr(&s_mask[e >> 5], 1u << (e & 31));
  }
  __syncthreads();
  if (t < 32) {
    const unsigned w = t < 16 ? s_mask[t] : 0u;
    int c = __popc(w), pre = c;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, pre, o);
      if (t >= o) pre += v;
    }
    const int before = pre - c, slot = blockIdx.y;
    if (t < 16 && slot >= before && slot < pre) {
      unsigned m = w;
      for (int q = 0; q < slot - before; ++q) m &= m - 1u;
      s_expert = t * 32 + (__ffs(m) - 1);
    }
  }
  __syncthreads();
  const int e = s_expert;
  if (e < 0) return;
  if (t < 32) {
    int cnt = 0;
    for (int base = 0; base < a.routes; base += 32) {
      const int r = base + t;
      const bool hit = r < a.routes && sid[r] == e;
      const unsigned bal = __ballot_sync(0xffffffffu, hit);
      if (hit) {
        const int j = cnt + __popc(bal & ((1u << t) - 1u));
        if (j < MAXC) {
          s_route[j] = r;
          s_tok[j] = r / a.top_k;
        }
      }
      cnt += __popc(bal);
    }
    if (t == 0) s_cnt = cnt < MAXC ? cnt : MAXC;
  }
  __syncthreads();
  const int cnt = s_cnt;
  const int tin = warp % TN, kslot = warp / TN;
  Seg sg{};
  sg.codes = (tin ? a.uc : a.gc) + e * a.cstride;
  sg.scale = (tin ? a.us : a.gs) + e * a.sstride;
  sg.high = nullptr;
  sg.n = a.n;
  sg.fmt = FMT;
  const int G = a.K / 128;
  uint4* xs = smem;
  half2* lut =
      reinterpret_cast<half2*>(smem + KW * 256);  // up to TAB_VECS_IQ2S uint4
  // Tokens that picked this expert are the mma M dimension, eight at a time.
  for (int c0 = 0; c0 < cnt; c0 += MAXT) {
    const int cc = cnt - c0 < MAXT ? cnt - c0 : MAXT;
    float acc[8] = {};
    body6<FMT, KW, TN>(sg, blockIdx.x, true, kslot, tin, 0, G, 4 * G, G, a.x,
                       a.ldx, cc, xs, lut, acc, a.tab, s_tok + c0);
    __syncthreads();
    float* red = reinterpret_cast<float*>(smem);
#pragma unroll
    for (int i = 0; i < 8; ++i) red[warp * 256 + lane * 8 + i] = acc[i];
    __syncthreads();
    for (int v = t; v < 256;
         v += blockDim.x) {  // 256 outputs (8 tokens x 32 rows) per tile
      float g = 0.f, u = 0.f;
#pragma unroll
      for (int k = 0; k < KW; ++k) {
        g += red[(k * TN + 0) * 256 + v];
        u += red[(k * TN + 1) * 256 + v];
      }
      const int lv = v >> 3, i = v & 7;
      const int token = (i & 2) | ((lv & 16) ? 4 : 0) | (lv & 1);
      const int col = (i & 1) | (((lv >> 1) & 1) << 1) | ((i >> 2) << 2);
      const int lrow = ((lv >> 2) & 3) * 8 + col;
      const int row = blockIdx.x * 32 + lrow;
      if (token < cc && row < a.n) {
        half value;
        if constexpr (Upstream) {
          const float g16 = __half2float(__float2half_rn(g));
          const float u16 = __half2float(__float2half_rn(u));
          const half silu = __float2half_rn(g16 / (1.f + expf(-g16)));
          value = __hmul(silu, __float2half_rn(u16));
        } else {
          value = __float2half_rn((g / (1.f + expf(-g))) * u);
        }
        a.hout[static_cast<int64_t>(s_route[c0 + token]) * a.n + row] = value;
      }
    }
    __syncthreads();
  }
}


} // namespace owned_onecat
void onecat_owned_gu(const at::Tensor& x,const at::Tensor& ids,
 const at::Tensor& gc,const at::Tensor& gs,const at::Tensor& uc,const at::Tensor& us,
 const at::Tensor& tab,at::Tensor out,int64_t first,int64_t fmt,bool upstream) {
 TORCH_CHECK(x.is_cuda(), "CUDA required");
 c10::cuda::CUDAGuard guard(x.device());
 const auto* prop=at::cuda::getDeviceProperties(x.get_device());
 TORCH_CHECK(prop->major==7 && prop->minor==0,"SM70 required");
 auto operand=[&](const at::Tensor& t,at::ScalarType type) {
  TORCH_CHECK(t.device()==x.device() && t.scalar_type()==type && t.is_contiguous(),"owned contiguous operand required");
  TORCH_CHECK(!t.requires_grad() && !t.is_neg() && !t.is_conj(),"plain non-autograd operand required");
 };
 operand(x,at::kHalf);
 TORCH_CHECK(x.dim()==2 && x.size(0)>=1 && x.size(0)<=32 && x.size(1)==2560,"SC117 C1..32 required");
 TORCH_CHECK(ids.scalar_type()==at::kInt || ids.scalar_type()==at::kLong,"integer route IDs required");
 operand(ids,ids.scalar_type());
 TORCH_CHECK(ids.dim()==2 && ids.size(0)==x.size(0) && ids.size(1)==11,"top10 plus independent shared slot required");
 TORCH_CHECK(first==0 || first==256,"ETP4 group first must be 0 or256");
 TORCH_CHECK(fmt==3 || fmt==5 || fmt==6 || fmt==7,"explicit LUT4/IQ3S/IQ3XXS/IQ2S planes required");
 const int cb=fmt==7?102400:fmt==3?204800:153600;
 const int sb=fmt==7?51200:fmt==6?12800:25600;
 for(const auto* t:{&gc,&uc}) {operand(*t,at::kByte);TORCH_CHECK(t->sizes()==at::IntArrayRef({256,cb}),"codes plane extent differs");}
 for(const auto* t:{&gs,&us}) {operand(*t,at::kByte);TORCH_CHECK(t->sizes()==at::IntArrayRef({256,sb}),"scale plane extent differs");}
 operand(tab,at::kByte);
 TORCH_CHECK(tab.dim()==1 && tab.numel()==(fmt==7?8192:fmt==3?0:3072),"IQ lookup table extent differs");
 operand(out,at::kHalf);
 TORCH_CHECK(out.sizes()==at::IntArrayRef({x.size(0)*11,160}),"slot activation extent differs");
 for(const auto* t:{&gc,&gs,&uc,&us,&tab}) TORCH_CHECK(t->numel()==0 || reinterpret_cast<uintptr_t>(t->data_ptr())%16==0,"uint4 alignment required");
 TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr())%16==0,"input uint4 alignment required");
 tensorfold::check_disjoint(out,{&x,&ids,&gc,&gs,&uc,&us,&tab},"activation overlaps read owner");
 const auto stream=at::cuda::getCurrentCUDAStream();
 owned_onecat::initialize<<<x.size(0),128,0,stream>>>(ids.data_ptr(),ids.scalar_type()==at::kLong,reinterpret_cast<half*>(out.data_ptr()));
 C10_CUDA_KERNEL_LAUNCH_CHECK();
 // Each panel has <=8 tokens and88 routes. Never truncate total C24/C32 work.
 for(int base=0;base<x.size(0);base+=8) {
  const int rows=std::min<int>(8,x.size(0)-base);
  owned_onecat::EArgs a{};
  a.gc=reinterpret_cast<const uint4*>(gc.data_ptr());a.gs=reinterpret_cast<const uint4*>(gs.data_ptr());
  a.uc=reinterpret_cast<const uint4*>(uc.data_ptr());a.us=reinterpret_cast<const uint4*>(us.data_ptr());
  a.cstride=cb/16;a.sstride=sb/16;a.x=reinterpret_cast<const half*>(x.data_ptr())+base*2560;
  a.ldx=2560;a.ids=static_cast<const char*>(ids.data_ptr())+base*11*ids.element_size();
  a.routes=rows*11;a.top_k=11;a.n=160;a.K=2560;a.first=first;a.wide=ids.scalar_type()==at::kLong;
  a.hout=reinterpret_cast<half*>(out.data_ptr())+base*11*160;
  a.tab=tab.numel()?reinterpret_cast<const uint4*>(tab.data_ptr()):nullptr;
  constexpr size_t sm=4*256*16+dmvns::TAB_VECS_IQ2S*16;
  const dim3 grid(5,a.routes);
#define RUN(F) if(fmt==F) { if(upstream) owned_onecat::moe_gu<F,4,true><<<grid,256,sm,stream>>>(a); else owned_onecat::moe_gu<F,4,false><<<grid,256,sm,stream>>>(a); }
  RUN(3) RUN(5) RUN(6) RUN(7)
#undef RUN
  C10_CUDA_KERNEL_LAUNCH_CHECK();
 }
}

void onecat_prepare() {
#define ATTR(F,U) {cudaFuncAttributes attr{}; C10_CUDA_CHECK(cudaFuncGetAttributes(&attr,owned_onecat::moe_gu<F,4,U>));}
 ATTR(3,true) ATTR(3,false) ATTR(5,true) ATTR(5,false)
 ATTR(6,true) ATTR(6,false) ATTR(7,true) ATTR(7,false)
#undef ATTR
 cudaFuncAttributes attr{};
 C10_CUDA_CHECK(cudaFuncGetAttributes(&attr,owned_onecat::initialize));
}
