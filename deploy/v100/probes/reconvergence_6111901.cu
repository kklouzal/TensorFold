// Synthetic coverage of the documented nested-divergence class. This is NOT
// NVIDIA's private reproducer and cannot establish SM70 applicability alone.
#include "cuda_probe_resources.cuh"
#include <vector>
#include <cstring>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 700
#error "compile this regression candidate as native SM70"
#endif

__device__ __noinline__ int component_value(const int* cache, int at, int component) {
  int a=cache[4*at], b=cache[4*at+1], c=cache[4*at+2], d=cache[4*at+3], result=0;
  if(component==0) result=a+b+c+d;
  else if(component==1) result=a-b+c-d;
  else if(component==2) result=a+b-c-d;
  else result=a-b-c+d;
  // Keep a real register definition/use across the divergent call boundary.
  asm volatile("" : "+r"(result));
  return result;
}

template<bool ExplicitConvergence>
__global__ void nested_pages(const int* ids, const int* offsets, const int* cache,
                             int rows, int visible_end, int* out) {
  int lane=threadIdx.x&31, value=0;
  bool selected=false;
  unsigned whole=__activemask(), in_tile=0, visible=0, owned=0;
  if constexpr(ExplicitConvergence) in_tile=__ballot_sync(whole,lane<10);
  if(lane<10) {
    int logical=ids[lane];
    bool valid=logical>=0 && logical<visible_end;
    if constexpr(ExplicitConvergence) visible=__ballot_sync(in_tile,valid);
    if(valid) {
      int offset=offsets[logical/128];
      if constexpr(ExplicitConvergence) owned=__ballot_sync(visible,offset>=0);
      if(offset>=0) {
        int physical=offset+logical%128;
        if(physical<rows) { value=component_value(cache,physical,lane%4); selected=true; }
        if constexpr(ExplicitConvergence) __syncwarp(owned);
      }
      if constexpr(ExplicitConvergence) __syncwarp(visible);
    }
    if constexpr(ExplicitConvergence) __syncwarp(in_tile);
  }
  if constexpr(ExplicitConvergence) __syncwarp(whole);
  int sum=value;
  for(int distance=16;distance;distance>>=1) sum+=__shfl_xor_sync(whole,sum,distance);
  int count=__popc(__ballot_sync(whole,selected));
  out[threadIdx.x]=value; out[128+threadIdx.x]=sum; out[256+threadIdx.x]=count;
}

int main(int argc,char** argv) {
  if(argc==2 && std::strcmp(argv[1],"--metadata-only")==0) {
    std::printf("{\"issue\":\"6111901\",\"nvcc\":\"%d.%d.%d\",\"coverage\":\"synthetic partial-page nested divergence and four component branches\",\"device_execution\":\"unrun\",\"SM70_applicability\":\"unresolved\",\"explicit_sync_mitigation\":\"unvalidated\"}\n",
                __CUDACC_VER_MAJOR__,__CUDACC_VER_MINOR__,__CUDACC_VER_BUILD__);
    return 0;
  }
  Resources resources;
  char primary[1024]{};
  int cases=0;
  const int ids[10]={-1,0,1,128,129,256,257,384,385,512};
  int map[5]{};
  std::vector<int> cache,result;
  try {
    if(argc!=2 || std::strcmp(argv[1],"--run")!=0) throw std::runtime_error("use --metadata-only or --run");
    int devices=0;cudaError_t available=cudaGetDeviceCount(&devices);
    if(available==cudaErrorNoDevice || available==cudaErrorInsufficientDriver || (available==cudaSuccess&&!devices)) {
      std::puts("{\"device_execution\":\"unrun\",\"reason\":\"V100 unavailable\"}");return 77;
    }
    check(available,"device count");cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties,0),"device properties");
    if(properties.major!=7 || properties.minor!=0) throw std::runtime_error("actual SM70 required");
    cache.resize(513*4);result.resize(384);
    constexpr int coefficients[4][4]={{1,1,1,1},{1,-1,1,-1},{1,1,-1,-1},{1,-1,-1,1}};
    int* di=static_cast<int*>(resources.allocate(sizeof(ids)));
    int* dm=static_cast<int*>(resources.allocate(5*4));
    int* dc=static_cast<int*>(resources.allocate(513*4*4));
    int* out=static_cast<int*>(resources.allocate(384*4));
    resources.create_stream();
    check(cudaMemcpyAsync(di,ids,sizeof(ids),cudaMemcpyHostToDevice,resources.stream),"input IDs");
    for(int degree:{1,2,4}) for(int rank=0;rank<4;++rank) for(int end:{0,511,512,513}) {
      int rows=0;
      for(int page=0;page<5;++page) {
        map[page]=-1;
        if(page%degree==rank) {
          map[page]=rows;
          int take=std::min(128,513-page*128);
          for(int i=0;i<take;++i) for(int c=0;c<4;++c) cache[(rows+i)*4+c]=(c*c+1)*(page*128+i)+2*c*c+1;
          rows+=take;
        }
      }
      check(cudaMemcpyAsync(dm,map,sizeof(map),cudaMemcpyHostToDevice,resources.stream),"page map");
      if(rows) check(cudaMemcpyAsync(dc,cache.data(),size_t(rows)*4*4,cudaMemcpyHostToDevice,resources.stream),"owned rows");
      int expected[32]{},sum=0,count=0;
      for(int lane=0;lane<10;++lane) if(ids[lane]>=0 && ids[lane]<end && map[ids[lane]/128]>=0) {
        for(int c=0;c<4;++c) expected[lane]+=coefficients[lane%4][c]*((c*c+1)*ids[lane]+2*c*c+1);
        sum+=expected[lane];++count;
      }
      for(int mode=0;mode<2;++mode) for(int repeat=0;repeat<32;++repeat) {
        // Reset all destinations on every execution; the largest oracle sum is
        // below one million, far from the 0x5a5a5a5a integer sentinel.
        check(cudaMemsetAsync(out,0x5a,384*4,resources.stream),"poison repeated destinations");
        if(mode) nested_pages<true><<<1,128,0,resources.stream>>>(di,dm,dc,rows,end,out);
        else nested_pages<false><<<1,128,0,resources.stream>>>(di,dm,dc,rows,end,out);
        check(cudaGetLastError(),"nested divergence launch");
        check(cudaMemcpyAsync(result.data(),out,result.size()*4,cudaMemcpyDeviceToHost,resources.stream),"outputs");
        check(cudaStreamSynchronize(resources.stream),"reconvergence completion");
        for(int thread=0;thread<128;++thread)
          if(result[thread]!=expected[thread%32] || result[128+thread]!=sum || result[256+thread]!=count)
            throw std::runtime_error("nested-divergence integer/count oracle mismatch");
        ++cases;
      }
    }
  } catch(const std::exception& error) { std::snprintf(primary,sizeof(primary),"%s",error.what()); }
  if(int outcome=probe_outcome(resources.cleanup(),primary)) return outcome;
  std::printf("{\"synthetic_cases\":%d,\"device_execution\":\"passed\",\"vendor_issue_applicability\":\"not established by this synthetic fixture alone\"}\n",cases);
  return 0;
}
