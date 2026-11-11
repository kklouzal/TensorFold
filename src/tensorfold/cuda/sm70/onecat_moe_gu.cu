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
#include <limits>
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

namespace owned_onecat {

// Separate rawIQ2 experiment. The incumbent packed kernel/body stay literal
// controls. Only these integer Ld loads differ; all Half/MMA math is shared.
struct RawSeg { const uint8_t* bytes; int64_t expert_stride, row_stride, offset; };
struct RawArgs : EArgs { RawSeg raw_gate, raw_up; };
__device__ __forceinline__ uint32_t raw16(const uint8_t* b) {
  return uint32_t(b[0]) | (uint32_t(b[1]) << 8);
}
__device__ __forceinline__ uint32_t raw32(const uint8_t* b) {
  return raw16(b) | (raw16(b + 2) << 16);
}
__device__ __forceinline__ void load_raw_iq2(Ld<IQ2S>& L, const RawSeg& source,
                                            int tile, int group, int lane) {
  const int row = tile*32 + ((lane >> 2) & 3)*8 + (lane & 3) + (lane & 16 ? 4 : 0);
  const uint8_t* b = source.bytes + int64_t(row)*source.row_stride + source.offset + (group/2)*82;
  const int half = group & 1;
  const uint8_t* lo = b + 2 + half*16;
  const uint8_t* signs = b + 34 + half*16;
  L.c[0][0] = make_uint4(raw32(lo), raw32(lo+4), raw32(lo+8), raw32(lo+12));
  L.c[1][0] = make_uint4(raw32(signs), raw32(signs+4), raw32(signs+8), raw32(signs+12));
  L.sc = make_uint4(raw32(b+66+half*4), raw16(b), raw32(b+74+half*4), 0u);
}
__global__ void raw_iq2_words(RawSeg source, int expert, int tile, int group, uint32_t* out) {
  source.bytes += int64_t(expert)*source.expert_stride;
  Ld<IQ2S> value;
  load_raw_iq2(value, source, tile, group, threadIdx.x);
  const uint4 data[3] = {value.c[0][0], value.c[1][0], value.sc};
  #pragma unroll
  for(int i=0;i<3;++i) {
    out[threadIdx.x*12+i*4] = data[i].x;
    out[threadIdx.x*12+i*4+1] = data[i].y;
    out[threadIdx.x*12+i*4+2] = data[i].z;
    out[threadIdx.x*12+i*4+3] = data[i].w;
  }
}

template <int KW, int TN>
__device__ __forceinline__ void body6_raw_iq2(
    const RawSeg& sg, int t, bool on, int kslot, int pid, int g0, int g1, int S,
    int G, const half* __restrict__ x, int ldx, int M, uint4* xs, half2* lut,
    float (&acc)[8], const uint4* __restrict__ tab, const int* rmap = nullptr) {
  constexpr int FMT = IQ2S;
  constexpr int NT = 32 * KW * TN, PARTS = 4 / TN;
  const int lane = threadIdx.x % 32;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  Ld<FMT> A;
  int g = g0 + kslot;
  uint4* slot = xs + kslot * 256;
  uint4 X[PARTS];
  auto xload = [&](int gg) {
#pragma unroll
    for (int jj = 0; jj < PARTS; ++jj) {
      const int idx = lane + 32 * (pid + TN * jj), c = idx >> 3, row = idx & 7;
      X[jj] = make_uint4(0, 0, 0, 0);
      if (row < M)
        X[jj] = __ldg(reinterpret_cast<const uint4*>(
            x + (rmap ? rmap[row] : row) * ldx + gg * 128 + c * 8));
    }
  };
  auto xstore = [&](int b) {
#pragma unroll
    for (int jj = 0; jj < PARTS; ++jj)
      slot[b * 128 + lane + 32 * (pid + TN * jj)] = X[jj];
  };
  auto xsync = [&]() {
    if constexpr (TN > 1)
      asm volatile("bar.sync %0, %1;" ::"r"(kslot + 1), "r"(TN * 32)
                   : "memory");
    else
      __syncwarp();
  };
  if (g < g1) {
    if (on) load_raw_iq2(A, sg, t, g, lane);
    xload(g);
  }
  static_assert(DMV_IQ4_PRMT,
                "pair mode needs the shared-memory-free IQ4 decode");
  if (tab != nullptr) {
    uint4* t4 = reinterpret_cast<uint4*>(lut);
    for (int i = threadIdx.x; i < tab_vecs<FMT>(); i += NT)
      t4[i] = __ldg(tab + i);
  }
  __syncthreads();
  if (g >= g1) return;
  xstore(0);
  xsync();
  int cur = 0;
  while (true) {
    Ld<FMT> B;
    const int gn = g + KW;
    const bool more = gn < g1;
    if (more) {
      if (on) load_raw_iq2(B, sg, t, gn, lane);
      xload(gn);
    }
    if (on) {
#pragma unroll
      for (int st = 0; st < 4; ++st) {
        const uint4* xp = slot + cur * 128 + (st * 4) * 8 + r;
        uint4 xa[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) xa[j] = xp[j * 8];
        uint32_t hw[16];
        decode<FMT>(A, st, hw, lut);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          mma(acc, xa[j].x, xa[j].y, hw[4 * j], hw[4 * j + 1]);
          mma(acc, xa[j].z, xa[j].w, hw[4 * j + 2], hw[4 * j + 3]);
        }
      }
    }
    if (!more) break;
    cur ^= 1;
    xstore(cur);
    xsync();
    A = B;
    g = gn;
  }
}
template <int KW, bool Upstream>
__global__ void __launch_bounds__(32 * KW * TN) moe_gu_raw_iq2(RawArgs a) {
  constexpr int FMT = IQ2S;
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
  RawSeg sg = tin ? a.raw_up : a.raw_gate;
  sg.bytes += int64_t(e) * sg.expert_stride;
  const int G = a.K / 128;
  uint4* xs = smem;
  half2* lut =
      reinterpret_cast<half2*>(smem + KW * 256);  // up to TAB_VECS_IQ2S uint4
  // Tokens that picked this expert are the mma M dimension, eight at a time.
  for (int c0 = 0; c0 < cnt; c0 += MAXT) {
    const int cc = cnt - c0 < MAXT ? cnt - c0 : MAXT;
    float acc[8] = {};
    body6_raw_iq2<KW, TN>(sg, blockIdx.x, true, kslot, tin, 0, G, 4 * G, G, a.x,
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

namespace {
struct RawIQ2Info { owned_onecat::RawSeg source; uintptr_t begin, end; };
RawIQ2Info raw_iq2_info(const at::Tensor& value, const at::Tensor& owner, int64_t offset) {
  TORCH_CHECK(value.is_cuda() && value.device()==owner.device() && value.scalar_type()==at::kByte
      && !value.is_neg() && !value.is_conj(), "raw IQ2 requires positive same-device UInt8 representation");
  TORCH_CHECK((value.dim()==2 && value.size(0)==40960)
      || (value.dim()==3 && value.size(0)==256 && value.size(1)==160), "raw IQ2 localE256/N160 geometry");
  const int64_t width=value.size(-1), rs=value.stride(-2);
  constexpr int64_t maximum=std::numeric_limits<int64_t>::max();
  TORCH_CHECK(width>=820 && width%82==0 && value.stride(-1)==1 && rs>=width
      && rs<=(maximum-width)/159 && offset>=0 && offset%256==0,
      "raw IQ2 complete blocks/positive byte rows/whole256 offset required");
  const int64_t row_span=159*rs+width;
  const int64_t es=value.dim()==3?value.stride(0):0;
  TORCH_CHECK(es>=0 && (value.dim()==2 || es==0 || es>=row_span), "raw IQ2 expert spans overlap");
  const int64_t stride=value.dim()==2 ? rs : es;
  const int64_t count=value.dim()==2 ? 40959 : 255;
  const int64_t tail=value.dim()==2 ? width : row_span;
  TORCH_CHECK(stride<=(maximum-tail)/count && offset/256<=(width-820)/82,"raw IQ2 extent/offset overflows");
  const uint64_t extent=uint64_t(count)*stride+tail;
  const uintptr_t begin=reinterpret_cast<uintptr_t>(value.data_ptr());
  TORCH_CHECK(extent<=std::numeric_limits<uintptr_t>::max()-begin,"raw IQ2 pointer overflow");
  return {{value.data_ptr<uint8_t>(),value.dim()==2?160*rs:es,rs,offset/256*82},begin,begin+extent};
}
void raw_iq2_no_alias(const at::Tensor& output,const RawIQ2Info& raw) {
  const uintptr_t begin=reinterpret_cast<uintptr_t>(output.data_ptr());
  const uint64_t bytes=uint64_t(output.numel())*output.element_size();
  TORCH_CHECK(bytes<=std::numeric_limits<uintptr_t>::max()-begin,"raw IQ2 output pointer overflow");
  TORCH_CHECK(begin+bytes<=raw.begin || raw.end<=begin,"raw IQ2 output aliases immutable original bank span");
}
}
void onecat_owned_gu_raw_iq2(const at::Tensor& x,const at::Tensor& ids,
    const at::Tensor& gate,const at::Tensor& up,const at::Tensor& table,at::Tensor out,
    int64_t first,bool upstream,int64_t gate_offset,int64_t up_offset) {
  TORCH_CHECK(x.is_cuda(),"CUDA required");
  c10::cuda::CUDAGuard guard(x.device());
  const auto* properties=at::cuda::getDeviceProperties(x.get_device());
  TORCH_CHECK(properties->major==7 && properties->minor==0,"SM70 required");
  auto operand=[&](const at::Tensor& t,at::ScalarType dtype) {
    TORCH_CHECK(t.is_cuda() && t.device()==x.device() && t.scalar_type()==dtype && t.is_contiguous()
        && !t.requires_grad() && !t.is_neg() && !t.is_conj(),"raw GU plain contiguous operand representation");
  };
  operand(x,at::kHalf);
  TORCH_CHECK(x.dim()==2 && x.size(0)>=1 && x.size(0)<=32 && x.size(1)==2560,"raw GU C1..32/K2560");
  TORCH_CHECK(ids.scalar_type()==at::kInt || ids.scalar_type()==at::kLong,"raw GU integer picks");
  operand(ids,ids.scalar_type());
  TORCH_CHECK(ids.dim()==2 && ids.size(0)==x.size(0) && ids.size(1)==11,"raw GU top10 plus shared");
  TORCH_CHECK(first==0 || first==256,"raw GU owned group first0/256");
  operand(table,at::kByte);operand(out,at::kHalf);
  TORCH_CHECK(table.dim()==1 && table.numel()==8192,"raw IQ2 exact original lookup table");
  TORCH_CHECK(out.sizes()==at::IntArrayRef({x.size(0)*11,160}),"raw GU slot activation extent");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr())%16==0
      && reinterpret_cast<uintptr_t>(table.data_ptr())%16==0,"raw GU input/table uint4 alignment");
  const RawIQ2Info g=raw_iq2_info(gate,x,gate_offset),u=raw_iq2_info(up,x,up_offset);
  tensorfold::check_disjoint(out,{&x,&ids,&table},"raw GU activation overlaps read input");
  raw_iq2_no_alias(out,g);raw_iq2_no_alias(out,u);
  const auto stream=at::cuda::getCurrentCUDAStream();
  owned_onecat::initialize<<<x.size(0),128,0,stream>>>(ids.data_ptr(),ids.scalar_type()==at::kLong,
      reinterpret_cast<half*>(out.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  for(int base=0;base<x.size(0);base+=8) {
    const int rows=std::min<int>(8,x.size(0)-base);
    owned_onecat::RawArgs a{};
    a.raw_gate=g.source;a.raw_up=u.source;
    a.x=reinterpret_cast<const half*>(x.data_ptr())+base*2560;
    a.ldx=2560;a.ids=static_cast<const char*>(ids.data_ptr())+base*11*ids.element_size();
    a.routes=rows*11;a.top_k=11;a.n=160;a.K=2560;a.first=first;a.wide=ids.scalar_type()==at::kLong;
    a.hout=reinterpret_cast<half*>(out.data_ptr())+base*11*160;
    a.tab=reinterpret_cast<const uint4*>(table.data_ptr());
    constexpr size_t sm=4*256*16+dmvns::TAB_VECS_IQ2S*16;
    const dim3 grid(5,a.routes);
    if(upstream) owned_onecat::moe_gu_raw_iq2<4,true><<<grid,256,sm,stream>>>(a);
    else owned_onecat::moe_gu_raw_iq2<4,false><<<grid,256,sm,stream>>>(a);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}
void onecat_raw_iq2_words(const at::Tensor& raw,at::Tensor out,int64_t expert,int64_t tile,int64_t group,int64_t offset) {
  TORCH_CHECK(out.is_cuda() && out.scalar_type()==at::kInt && out.is_contiguous()
      && out.sizes()==at::IntArrayRef({32,12}) && !out.is_neg() && !out.is_conj(),"raw IQ2 diagnostic words I32[32,12]");
  TORCH_CHECK(expert>=0 && expert<256 && tile>=0 && tile<5 && group>=0 && group<20,"raw IQ2 diagnostic coordinate");
  c10::cuda::CUDAGuard guard(out.device());
  const auto* properties=at::cuda::getDeviceProperties(out.get_device());
  TORCH_CHECK(properties->major==7 && properties->minor==0,"raw IQ2 diagnostic SM70");
  const RawIQ2Info info=raw_iq2_info(raw,out,offset);raw_iq2_no_alias(out,info);
  owned_onecat::raw_iq2_words<<<1,32,0,at::cuda::getCurrentCUDAStream()>>>(info.source,expert,tile,group,
      reinterpret_cast<uint32_t*>(out.data_ptr<int32_t>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void onecat_prepare_raw_iq2() {
  cudaFuncAttributes attributes{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes,owned_onecat::moe_gu_raw_iq2<4,true>));
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes,owned_onecat::moe_gu_raw_iq2<4,false>));
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes,owned_onecat::raw_iq2_words));
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes,owned_onecat::initialize));
}
