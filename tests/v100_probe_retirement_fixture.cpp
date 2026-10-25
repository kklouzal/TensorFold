// CPU fault controls for the owned retirement adapter, not a CUDA implementation.
#include "cuda_probe_resources.cuh"
#include <cstring>

static const char* mode;
static bool completed=false;
static int* host_borrow=nullptr;

extern "C" cudaError_t CUDARTAPI cudaMalloc(void** value,size_t bytes) {
  *value=std::malloc(bytes);return *value?cudaSuccess:cudaErrorMemoryAllocation;
}
extern "C" cudaError_t CUDARTAPI cudaStreamCreateWithFlags(cudaStream_t* value,unsigned) {
  *value=reinterpret_cast<cudaStream_t>(std::malloc(1));return cudaSuccess;
}
extern "C" cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t) {
  if(!host_borrow || *host_borrow!=37) std::abort();
  if(std::strcmp(mode,"fence-failure")==0) return cudaErrorLaunchFailure;
  completed=true;return cudaSuccess;
}
extern "C" cudaError_t CUDARTAPI cudaFree(void* value) {
  if(!completed || std::strcmp(mode,"graph-failure")==0) std::abort();
  std::free(value);return cudaSuccess;
}
extern "C" cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t value) {
  if(!completed) std::abort();
  std::free(value);return cudaSuccess;
}
extern "C" cudaError_t CUDARTAPI cudaGraphExecDestroy(cudaGraphExec_t value) {
  if(std::strcmp(mode,"graph-failure")==0) return cudaErrorInvalidValue;
  std::free(value);return cudaSuccess;
}
extern "C" cudaError_t CUDARTAPI cudaGraphDestroy(cudaGraph_t value) {
  if(std::strcmp(mode,"graph-failure")==0) std::abort();
  std::free(value);return cudaSuccess;
}
extern "C" const char* CUDARTAPI cudaGetErrorString(cudaError_t) {
  return "injected retirement refusal";
}

struct Borrower {
  int value=37;
  const char* marker;
  ~Borrower() {
    if(!completed) std::abort();
    FILE* file=std::fopen(marker,"w");
    if(!file || std::fputs("host retired after completion\n",file)<0 || std::fclose(file)) std::abort();
  }
};

int main(int argc,char** argv) {
  if(argc!=3) return 2;
  mode=argv[1];
  Resources resources;
  Borrower host{37,argv[2]};
  host_borrow=&host.value;
  resources.allocate(64);resources.create_stream();
  if(std::strcmp(mode,"graph-failure")==0) {
    resources.replay=reinterpret_cast<cudaGraphExec_t>(std::malloc(1));
    resources.graph=reinterpret_cast<cudaGraph_t>(std::malloc(1));
  }
  return probe_outcome(resources.cleanup(),"");
}
