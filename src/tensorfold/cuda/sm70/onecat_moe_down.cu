// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from 1CatAI/1Cat-vLLM 1b57260 gguf_dp4a_sm70.cu/down_unroute,
// gguf_q8_1.cuh and gguf_dp4a.cuh. Integer formulas follow ggml (MIT).
// Changes: original IQ4_NL bytes (no transcode planes), E256/global remap,
// top10+excludedshared512, explicit paid Q8 scratch/poison status, panels<=8,
// fixed separate RN32 arithmetic and two explicitly different fold contracts.
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/core/InferenceMode.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>
#include <type_traits>
#include "../kernels/tensor_contracts.cuh"
#include "gguf_decode_tables.cuh"

namespace {
constexpr int D=2560,K=160,SLOTS=11,TOP=10,GROUPS=5;
struct Q81 { half2 ds; int8_t qs[32]; };
static_assert(sizeof(Q81)==36 && alignof(Q81)==4,"pinned Q8_1 ABI");
__device__ int64_t pick(const void* ids,bool wide,int at) {
 return wide?static_cast<const int64_t*>(ids)[at]:static_cast<const int32_t*>(ids)[at];
}
__device__ bool valid(const void* ids,bool wide,int row) {
 int64_t seen[SLOTS];
 for(int s=0;s<SLOTS;++s) {
  const int64_t id=pick(ids,wide,row*SLOTS+s);
  if(s==TOP ? id!=512 : id<0 || id>=512) return false;
  for(int j=0;j<s;++j) if(seen[j]==id) return false;
  seen[s]=id;
 }
 return true;
}
template<bool Original>
__global__ void initialize(const void* ids,bool wide,const half* activation,half* slots,int* status) {
 __shared__ bool ok;
 if(threadIdx.x==0) {
  const bool ids_ok=valid(ids,wide,blockIdx.x);
  const bool producer_ok=__half_as_ushort(activation[(blockIdx.x*SLOTS+TOP)*K])==0;
  status[blockIdx.x]=(ids_ok?0:1)|(producer_ok?0:4);ok=ids_ok&&producer_ok;
 }
 if constexpr(!Original) {
  __syncthreads();
  const half value=ok?__float2half_rn(0.f):__float2half_rn(__int_as_float(0x7fffffff));
  for(int i=threadIdx.x;i<SLOTS*D;i+=blockDim.x) slots[blockIdx.x*SLOTS*D+i]=value;
 }
}
__global__ void quantize(const half* h,const void* ids,bool wide,Q81* q,int* status,int rows,int first) {
 const int warp=threadIdx.x/32,lane=threadIdx.x%32;
 const int at=blockIdx.x*4+warp;
 if(at>=rows*SLOTS*GROUPS) return; // warp-uniform; all32 shuffle participants remain.
 const int route=at/GROUPS,group=at%GROUPS,row=route/SLOTS,slot=route%SLOTS;
 int active=0;
 if(!lane) {
  const int64_t id=pick(ids,wide,route);
  active=atomicAdd(status+row,0)==0 && slot<TOP && id>=first && id<first+256;
 }
 active=__shfl_sync(0xffffffffu,active,0);
 float value=active?__half2float(h[route*K+group*32+lane]):0.f;
 const bool bad=__any_sync(0xffffffffu,!isfinite(value));
 if(bad) {if(!lane) atomicOr(status+row,2);value=0.f;}
 float maximum=fabsf(value),sum=value;
 #pragma unroll
 for(int offset=16;offset;offset>>=1) {
  maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffffu,maximum,offset));
  sum=__fadd_rn(sum,__shfl_xor_sync(0xffffffffu,sum,offset));
 }
 const float d=__fdiv_rn(maximum,127.f);
 const float rounded=maximum==0.f?0.f:roundf(__fdiv_rn(value,d)); // ties away, NOT RNE.
 const bool invalid=__any_sync(0xffffffffu,!isfinite(rounded)||rounded<-127.f||rounded>127.f);
 if(invalid && !lane) atomicOr(status+row,2);
 q[at].qs[lane]=invalid?0:static_cast<int8_t>(static_cast<int>(rounded));
 if(!lane) q[at].ds=__floats2half2_rn(invalid?0.f:d,invalid?0.f:sum);
}
__device__ int iq_value(int code) {
 constexpr int table[16]={TF_GGUF_IQ4NL_VALUES};
 return table[code];
}
__device__ float raw_dot(const uint8_t* row,int group,const Q81& q,int column_offset) {
 const uint8_t* p=row+(column_offset/32+group)*18;
 int sum=0;
 #pragma unroll
 for(int packet=0;packet<8;++packet) {
  uint32_t word=0;
  #pragma unroll
  for(int j=0;j<4;++j) {
   const int k=4*packet+j;
   const int code=(p[2+k%16]>>(4*(k/16)))&15;
   word|=uint32_t(static_cast<uint8_t>(iq_value(code)))<<(8*j);
  }
  const int activation=reinterpret_cast<const int*>(q.qs)[packet];
  sum=__dp4a(static_cast<int>(word),activation,sum); // exact signed Int32; <=32*127².
 }
 const half scale=__ushort_as_half(uint16_t(p[0])|(uint16_t(p[1])<<8));
 return __fmul_rn(float(sum),__fmul_rn(__half2float(scale),__low2float(q.ds)));
}

template<bool Original>
__global__ void down_kernel(const uint8_t* raw,int row_bytes,const Q81* input,const void* ids,bool wide,
 const float* weights,half* slots,const int* status,void* output,int first,int column_offset,int row_start) {
 extern __shared__ __align__(16) uint32_t shared_words[];
 __shared__ float partial[4][32];
 const int lane=threadIdx.x%32,warp=threadIdx.x/32,row=row_start+blockIdx.y,col=blockIdx.x*32+lane;
 if(status[row]) {
  if(!warp) {
   const half nan=__float2half_rn(__int_as_float(0x7fffffff));
   if constexpr(!Original) for(int s=0;s<SLOTS;++s) slots[(row*SLOTS+s)*D+col]=nan;
   if constexpr(Original) static_cast<half*>(output)[row*D+col]=nan;
  }
  return; // CTA-uniform, before any block barrier.
 }
 const auto* source=reinterpret_cast<const uint32_t*>(input+row*SLOTS*GROUPS);
 for(int i=threadIdx.x;i<TOP*GROUPS*9;i+=blockDim.x) shared_words[i]=source[i];
 __syncthreads();
 const auto* q=reinterpret_cast<const Q81*>(shared_words);
 float total=0.f;
 for(int s=warp;s<TOP;s+=4) {
  const int64_t global=pick(ids,wide,row*SLOTS+s);
  if(global<first || global>=first+256) continue;
  const auto* coefficients=raw+(int64_t(global-first)*D+col)*row_bytes;
  float dot=0.f;
  #pragma unroll
  for(int g=0;g<GROUPS;++g) dot=__fadd_rn(dot,raw_dot(coefficients,g,q[s*GROUPS+g],column_offset));
  const half rounded=__float2half_rn(dot);
  if constexpr(!Original) slots[(row*SLOTS+s)*D+col]=rounded;
  if constexpr(Original) total=__fadd_rn(total,__fmul_rn(__half2float(rounded),weights[row*SLOTS+s]));
 }
 if constexpr(Original) {
  partial[warp][lane]=total;
  __syncthreads();
  if(!warp) {
   float sum=0.f;
   #pragma unroll
   for(int w=0;w<4;++w) sum=__fadd_rn(sum,partial[w][lane]);
   static_cast<half*>(output)[row*D+col]=__float2half_rn(sum);
  }
 }
}
// Integrated ABI publishes unweighted slots for the existing rank transport
// and ordered outer fold. The caller-selected slot type owns rounding; no
// weighted result or extra Half boundary is introduced for FP32 consumers.
template<typename Output>
__device__ __forceinline__ Output slot_value(float value) {
 if constexpr(std::is_same<Output,half>::value) return __float2half_rn(value);
 else return value;
}
template<typename Output>
__global__ void initialize_slots(const void* ids, bool wide, const half* activation,
                                Output* slots, int* status) {
 if(threadIdx.x==0) {
  const bool ids_ok=valid(ids,wide,blockIdx.x);
  const bool producer_ok=__half_as_ushort(activation[(blockIdx.x*SLOTS+TOP)*K])==0;
  status[blockIdx.x]=(ids_ok?0:1)|(producer_ok?0:4);
 }
 for(int i=threadIdx.x;i<SLOTS*D;i+=blockDim.x) slots[blockIdx.x*SLOTS*D+i]=slot_value<Output>(0.f);
}
template<typename Output>
__global__ void down_slots(const uint8_t* raw,int row_bytes,const Q81* input,
                          const void* ids,bool wide,const float* shared,Output* slots,
                          const int* status,int first,int column_offset,int row_start) {
 extern __shared__ __align__(16) uint32_t shared_words[];
 const int lane=threadIdx.x%32,warp=threadIdx.x/32,row=row_start+blockIdx.y,col=blockIdx.x*32+lane;
 if(status[row]) {
  if(!warp) for(int slot=0;slot<SLOTS;++slot)
   slots[(row*SLOTS+slot)*D+col]=slot_value<Output>(__int_as_float(0x7fffffff));
  return;
 }
 const auto* source=reinterpret_cast<const uint32_t*>(input+row*SLOTS*GROUPS);
 for(int i=threadIdx.x;i<TOP*GROUPS*9;i+=blockDim.x) shared_words[i]=source[i];
 __syncthreads();
 const auto* q=reinterpret_cast<const Q81*>(shared_words);
 for(int slot=warp;slot<TOP;slot+=4) {
  const int64_t global=pick(ids,wide,row*SLOTS+slot);
  if(global<first || global>=first+256) continue;
  const auto* coefficients=raw+(int64_t(global-first)*D+col)*row_bytes;
  float dot=0.f;
  #pragma unroll
  for(int group=0;group<GROUPS;++group)
   dot=__fadd_rn(dot,raw_dot(coefficients,group,q[slot*GROUPS+group],column_offset));
  slots[(row*SLOTS+slot)*D+col]=slot_value<Output>(dot);
 }
 if(!warp) slots[(row*SLOTS+TOP)*D+col]=slot_value<Output>(shared[row*D+col]);
}
__global__ void slot_fold(const half* slots,const float* weights,const int* status,float* out) {
 const int row=blockIdx.y,col=blockIdx.x*256+threadIdx.x;
 if(status[row]) {out[row*D+col]=__int_as_float(0x7fffffff);return;}
 float sum=0.f;
 #pragma unroll
 for(int s=0;s<TOP;++s) sum=__fadd_rn(sum,__fmul_rn(__half2float(slots[(row*SLOTS+s)*D+col]),weights[row*SLOTS+s]));
 out[row*D+col]=sum;
}
}

void onecat_down(const at::Tensor& activation,const at::Tensor& ids,const at::Tensor& weights,
 const at::Tensor& raw,at::Tensor q8,at::Tensor slots,at::Tensor status,at::Tensor out,
 int64_t kind,int64_t first,int64_t column_offset,bool original) {
 TORCH_CHECK(kind==20,"IQ4_NL only; Q2_0 remains on the incumbent");
 TORCH_CHECK(activation.is_cuda(),"CUDA activation required");
 const auto device=activation.device();c10::cuda::CUDAGuard guard(device);
 const auto* props=at::cuda::getDeviceProperties(device.index());
 TORCH_CHECK(props->major==7&&props->minor==0,"SM70 down component required");
 auto operand=[&](const at::Tensor& t,at::ScalarType type) {
  TORCH_CHECK(t.device()==device&&t.scalar_type()==type&&t.is_contiguous()
   &&!t.requires_grad()&&!t.is_neg()&&!t.is_conj(),"contiguous same-device non-grad operand required");
 };
 TORCH_CHECK(ids.scalar_type()==at::kInt||ids.scalar_type()==at::kLong,"integer route IDs required");
 operand(ids,ids.scalar_type());operand(activation,at::kHalf);operand(weights,at::kFloat);operand(raw,at::kByte);
 operand(q8,at::kByte);operand(slots,at::kHalf);operand(status,at::kInt);operand(out,original?at::kHalf:at::kFloat);
 TORCH_CHECK(ids.dim()==2&&ids.size(0)>=1&&ids.size(0)<=32&&ids.size(1)==SLOTS,"R1..32/top10+shared required");
 const int rows=ids.size(0);
 TORCH_CHECK((first==0||first==256)&&column_offset>=0&&column_offset<=480&&column_offset%32==0,"owned E256/block-aligned lane required");
 TORCH_CHECK(activation.dim()==2&&activation.size(0)==rows*SLOTS&&activation.size(1)==K
  &&weights.sizes()==ids.sizes(),"Half routed width160/F32 weights required");
 TORCH_CHECK(raw.dim()==2&&raw.size(0)==256*D&&raw.size(1)%18==0&&raw.size(1)>=90&&raw.size(1)<=360
  &&raw.size(1)>=(column_offset/32+GROUPS)*18,"original IQ4_NL raw bank extent differs");
 TORCH_CHECK(q8.dim()==3&&q8.size(0)==rows*SLOTS&&q8.size(1)==GROUPS&&q8.size(2)==36
  &&slots.dim()==2&&slots.size(0)==rows*SLOTS&&slots.size(1)==D
  &&status.dim()==1&&status.size(0)==rows&&out.dim()==2&&out.size(0)==rows&&out.size(1)==D,"caller-owned down scratch/output extent differs");
 TORCH_CHECK(uintptr_t(q8.data_ptr())%4==0,"Q8_1 alignment");
 const at::Tensor* writes[]={&q8,&slots,&status,&out};
 for(int i=0;i<4;++i) {
  TORCH_CHECK(!writes[i]->is_inference()||c10::InferenceMode::is_enabled(),"inference output outside InferenceMode");
  for(const auto* read:{&activation,&ids,&weights,&raw}) TORCH_CHECK(!tensorfold::overlaps(*writes[i],*read),"down read/write overlap");
  for(int j=0;j<i;++j) TORCH_CHECK(!tensorfold::overlaps(*writes[i],*writes[j]),"down output overlap");
 }
 const auto stream=at::cuda::getCurrentCUDAStream();const bool wide=ids.scalar_type()==at::kLong;
 const auto init=[&](auto flag) {
  initialize<decltype(flag)::value><<<rows,128,0,stream>>>(ids.data_ptr(),wide,reinterpret_cast<const half*>(activation.data_ptr<at::Half>()),reinterpret_cast<half*>(slots.data_ptr<at::Half>()),status.data_ptr<int>());
 };
 if(original) init(std::true_type{});else init(std::false_type{});
 C10_CUDA_KERNEL_LAUNCH_CHECK();
 quantize<<<(rows*SLOTS*GROUPS+3)/4,128,0,stream>>>(reinterpret_cast<const half*>(activation.data_ptr<at::Half>()),ids.data_ptr(),wide,
  reinterpret_cast<Q81*>(q8.data_ptr()),status.data_ptr<int>(),rows,first);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
 for(int begin=0;begin<rows;begin+=8) {
  const dim3 grid(D/32,std::min(8,rows-begin));
  const auto go=[&](auto flag) {
   constexpr bool Original=decltype(flag)::value;
   down_kernel<Original><<<grid,128,TOP*GROUPS*sizeof(Q81),stream>>>(raw.data_ptr<uint8_t>(),int(raw.size(1)),
    reinterpret_cast<const Q81*>(q8.data_ptr()),ids.data_ptr(),wide,weights.data_ptr<float>(),
    reinterpret_cast<half*>(slots.data_ptr<at::Half>()),status.data_ptr<int>(),out.data_ptr(),first,column_offset,begin);
  };
  if(original) go(std::true_type{});else go(std::false_type{});
  C10_CUDA_KERNEL_LAUNCH_CHECK();
 }
 if(!original) {
  slot_fold<<<dim3(D/256,rows),256,0,stream>>>(reinterpret_cast<const half*>(slots.data_ptr<at::Half>()),weights.data_ptr<float>(),status.data_ptr<int>(),out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
 }
}
void onecat_down_slots(const at::Tensor& activation,const at::Tensor& ids,const at::Tensor& raw,
 const at::Tensor& shared,at::Tensor q8,at::Tensor status,at::Tensor slots,
 int64_t kind,int64_t first,int64_t column_offset) {
 TORCH_CHECK(kind==20,"IQ4_NL only; Q2_0 remains on the incumbent");
 TORCH_CHECK(activation.is_cuda(),"CUDA activation required");
 const auto device=activation.device();c10::cuda::CUDAGuard guard(device);
 const auto* props=at::cuda::getDeviceProperties(device.index());
 TORCH_CHECK(props->major==7&&props->minor==0,"SM70 down component required");
 auto operand=[&](const at::Tensor& tensor,at::ScalarType dtype) {
  TORCH_CHECK(tensor.device()==device&&tensor.scalar_type()==dtype&&tensor.is_contiguous()
   &&!tensor.requires_grad()&&!tensor.is_neg()&&!tensor.is_conj(),"contiguous same-device non-grad operand required");
 };
 TORCH_CHECK(ids.scalar_type()==at::kInt||ids.scalar_type()==at::kLong,"integer route IDs required");
 TORCH_CHECK(slots.scalar_type()==at::kHalf||slots.scalar_type()==at::kFloat,"Half/F32 route slots required");
 operand(ids,ids.scalar_type());operand(activation,at::kHalf);operand(raw,at::kByte);
 operand(shared,at::kFloat);operand(q8,at::kByte);operand(status,at::kInt);operand(slots,slots.scalar_type());
 TORCH_CHECK(ids.dim()==2&&ids.size(0)>=1&&ids.size(0)<=32&&ids.size(1)==SLOTS,"R1..32/top10+shared required");
 const int rows=ids.size(0);
 TORCH_CHECK((first==0||first==256)&&column_offset>=0&&column_offset<=480&&column_offset%32==0,"owned E256/block-aligned lane required");
 TORCH_CHECK(activation.sizes()==at::IntArrayRef({rows*SLOTS,K})
  &&shared.sizes()==at::IntArrayRef({rows,D}),"Half activation/F32 shared extent differs");
 TORCH_CHECK(raw.dim()==2&&raw.size(0)==256*D&&raw.size(1)%18==0&&raw.size(1)>=90&&raw.size(1)<=360
  &&raw.size(1)>=(column_offset/32+GROUPS)*18,"original IQ4_NL raw bank extent differs");
 TORCH_CHECK(q8.sizes()==at::IntArrayRef({rows*SLOTS,GROUPS,36})&&status.sizes()==at::IntArrayRef({rows})
  &&slots.sizes()==at::IntArrayRef({rows,SLOTS,D}),"caller-owned slots scratch/output extent differs");
 TORCH_CHECK(uintptr_t(q8.data_ptr())%4==0,"Q8_1 alignment");
 const at::Tensor* writes[]={&q8,&status,&slots};
 for(int i=0;i<3;++i) {
  TORCH_CHECK(!writes[i]->is_inference()||c10::InferenceMode::is_enabled(),"inference output outside InferenceMode");
  for(const auto* read:{&activation,&ids,&raw,&shared}) TORCH_CHECK(!tensorfold::overlaps(*writes[i],*read),"down read/write overlap");
  for(int j=0;j<i;++j) TORCH_CHECK(!tensorfold::overlaps(*writes[i],*writes[j]),"down output overlap");
 }
 const auto stream=at::cuda::getCurrentCUDAStream();const bool wide=ids.scalar_type()==at::kLong;
 const auto execute=[&](auto* output) {
  using Output=std::remove_pointer_t<decltype(output)>;
  initialize_slots<Output><<<rows,128,0,stream>>>(ids.data_ptr(),wide,
   reinterpret_cast<const half*>(activation.data_ptr<at::Half>()),output,status.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  quantize<<<(rows*SLOTS*GROUPS+3)/4,128,0,stream>>>(reinterpret_cast<const half*>(activation.data_ptr<at::Half>()),ids.data_ptr(),wide,
   reinterpret_cast<Q81*>(q8.data_ptr()),status.data_ptr<int>(),rows,first);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  for(int begin=0;begin<rows;begin+=8) {
   const dim3 grid(D/32,std::min(8,rows-begin));
   down_slots<Output><<<grid,128,TOP*GROUPS*sizeof(Q81),stream>>>(raw.data_ptr<uint8_t>(),int(raw.size(1)),
    reinterpret_cast<const Q81*>(q8.data_ptr()),ids.data_ptr(),wide,shared.data_ptr<float>(),output,status.data_ptr<int>(),first,column_offset,begin);
   C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
 };
 if(slots.scalar_type()==at::kHalf) execute(reinterpret_cast<half*>(slots.data_ptr<at::Half>()));
 else execute(slots.data_ptr<float>());
}
void onecat_down_prepare() {
 cudaFuncAttributes a;
 C10_CUDA_CHECK(cudaFuncGetAttributes(&a,initialize<true>));
 C10_CUDA_CHECK(cudaFuncGetAttributes(&a,initialize<false>));C10_CUDA_CHECK(cudaFuncGetAttributes(&a,quantize));
 C10_CUDA_CHECK(cudaFuncGetAttributes(&a,down_kernel<true>));C10_CUDA_CHECK(cudaFuncGetAttributes(&a,down_kernel<false>));
 C10_CUDA_CHECK(cudaFuncGetAttributes(&a,slot_fold));
 C10_CUDA_CHECK(cudaFuncGetAttributes(&a,initialize_slots<half>));
 C10_CUDA_CHECK(cudaFuncGetAttributes(&a,initialize_slots<float>));
 C10_CUDA_CHECK(cudaFuncGetAttributes(&a,down_slots<half>));
 C10_CUDA_CHECK(cudaFuncGetAttributes(&a,down_slots<float>));
}
