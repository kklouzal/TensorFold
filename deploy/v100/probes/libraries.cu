// Source-built library/header/SM70 contract. No TensorFold performance claim.
#include "cuda_probe_resources.cuh"
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cudnn.h>
#include <nccl.h>
#include <cutlass/cutlass.h>
#include <cutlass/version.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <vector>
#include <cstring>

static_assert(CUBLAS_VER_MAJOR == 12 && CUBLAS_VER_MINOR == 9 && CUBLAS_VER_PATCH == 2 && CUBLAS_VER_BUILD == 10);
static_assert(CUDNN_VERSION == 91002 && NCCL_VERSION_CODE == 23203);
static_assert(CUTLASS_MAJOR == 4 && CUTLASS_MINOR == 8 && CUTLASS_PATCH == 1);

static void blas(cublasStatus_t status, const char* action) {
  if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(action) + ": cuBLAS status " + std::to_string(status));
}
using Gemm = cutlass::gemm::device::Gemm<
  cutlass::half_t, cutlass::layout::ColumnMajor,
  cutlass::half_t, cutlass::layout::ColumnMajor,
  float, cutlass::layout::ColumnMajor, float,
  cutlass::arch::OpClassTensorOp, cutlass::arch::Sm70>;

int main(int argc, char** argv) {
  int nccl_version=0, major=0, minor=0, patch=0;
  if (ncclGetVersion(&nccl_version) != ncclSuccess || nccl_version != NCCL_VERSION_CODE ||
      cudnnGetVersion() != CUDNN_VERSION ||
      cublasGetProperty(MAJOR_VERSION, &major) != CUBLAS_STATUS_SUCCESS ||
      cublasGetProperty(MINOR_VERSION, &minor) != CUBLAS_STATUS_SUCCESS ||
      cublasGetProperty(PATCH_LEVEL, &patch) != CUBLAS_STATUS_SUCCESS ||
      major != 12 || minor != 9 || patch != 2) {
    std::fputs("loaded library identities do not match selected source-build headers\n", stderr); return 1;
  }
  if (argc == 2 && std::strcmp(argv[1], "--metadata-only") == 0) {
    std::printf("{\"cublas\":\"%d.%d.%d.%d\",\"cublasLt_version\":%zu,\"cudnn\":%zu,\"nccl\":%d,\"cutlass\":\"4.8.1\",\"device_execution\":\"unrun\"}\n",
      major,minor,patch,CUBLAS_VER_BUILD,cublasLtGetVersion(),cudnnGetVersion(),nccl_version);
    return 0;
  }
  Resources resources;
  cublasHandle_t handle=nullptr;
  cublasLtHandle_t lt=nullptr;
  cublasLtMatmulDesc_t operation=nullptr;
  cublasLtMatrixLayout_t al=nullptr, bl=nullptr, cl=nullptr;
  cublasLtMatmulPreference_t preference=nullptr;
  char primary[1024]{};
  int algorithm_id=-1;
  bool clean=true;
  std::vector<cutlass::half_t> a,b;
  std::vector<float> result,expected,zero;
  try {
    if (argc != 2 || std::strcmp(argv[1], "--run") != 0) throw std::runtime_error("use --metadata-only or --run");
    int count=0; cudaError_t available=cudaGetDeviceCount(&count);
    if (available == cudaErrorNoDevice || available == cudaErrorInsufficientDriver || (available == cudaSuccess && !count)) {
      std::puts("{\"device_execution\":\"unrun\",\"reason\":\"V100 device unavailable\"}"); return 77;
    }
    check(available,"device count");
    cudaDeviceProp properties{}; check(cudaGetDeviceProperties(&properties,0),"device properties");
    if (properties.major!=7 || properties.minor!=0) throw std::runtime_error("actual SM70 device required");
    constexpr int m=32,n=32,k=32;
    constexpr size_t workspace_bytes=32u<<20; // Existing GGUF bounded BLAS workspace comparison region.
    a.resize(m*k);b.resize(k*n);result.resize(m*n);expected.resize(m*n);zero.assign(m*n,0.0f);
    for(int col=0;col<k;++col) for(int row=0;row<m;++row) a[row+m*col]=float((row+2*col)%3-1);
    for(int col=0;col<n;++col) for(int row=0;row<k;++row) b[row+k*col]=float((2*row+col)%5-2);
    for(int col=0;col<n;++col) for(int row=0;row<m;++row) {
      int total=0; for(int inner=0;inner<k;++inner) total+=int(float(a[row+m*inner]))*int(float(b[inner+k*col]));
      expected[row+m*col]=float(total);
    }
    auto da=static_cast<cutlass::half_t*>(resources.allocate(a.size()*sizeof(a[0])));
    auto db=static_cast<cutlass::half_t*>(resources.allocate(b.size()*sizeof(b[0])));
    auto dc=static_cast<float*>(resources.allocate(zero.size()*sizeof(float)));
    std::array<float*,3> outputs{};
    for(auto& pointer:outputs) pointer=static_cast<float*>(resources.allocate(zero.size()*sizeof(float)));
    void* workspace=resources.allocate(workspace_bytes);
    resources.create_stream();
    check(cudaMemcpyAsync(da,a.data(),a.size()*2,cudaMemcpyHostToDevice,resources.stream),"copy A");
    check(cudaMemcpyAsync(db,b.data(),b.size()*2,cudaMemcpyHostToDevice,resources.stream),"copy B");
    check(cudaMemcpyAsync(dc,zero.data(),zero.size()*4,cudaMemcpyHostToDevice,resources.stream),"initialize C");
    auto acquired=[&](cublasStatus_t status,auto& output,const char* action) {
      if(status!=CUBLAS_STATUS_SUCCESS) output=nullptr; // Failed outputs are never valid retirement handles.
      blas(status,action);
    };
    acquired(cublasCreate(&handle),handle,"create cuBLAS");
    blas(cublasSetStream(handle,resources.stream),"set stream");
    blas(cublasSetWorkspace(handle,workspace,workspace_bytes),"bind workspace after stream");
    const float alpha=1.0f,beta=0.0f;
    acquired(cublasLtCreate(&lt),lt,"create cuBLASLt");
    acquired(cublasLtMatmulDescCreate(&operation,CUBLAS_COMPUTE_32F_PEDANTIC,CUDA_R_32F),operation,"create FP32 accumulation descriptor");
    acquired(cublasLtMatrixLayoutCreate(&al,CUDA_R_16F,m,k,m),al,"A column-major layout");
    acquired(cublasLtMatrixLayoutCreate(&bl,CUDA_R_16F,k,n,k),bl,"B column-major layout");
    acquired(cublasLtMatrixLayoutCreate(&cl,CUDA_R_32F,m,n,m),cl,"C/D column-major layout");
    acquired(cublasLtMatmulPreferenceCreate(&preference),preference,"create preference");
    blas(cublasLtMatmulPreferenceSetAttribute(preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace_bytes,sizeof(workspace_bytes)),"set workspace ceiling");
    cublasLtMatmulHeuristicResult_t heuristic{}; int returned=0;
    blas(cublasLtMatmulAlgoGetHeuristic(lt,operation,al,bl,cl,cl,preference,1,&heuristic,&returned),"algorithm query");
    if(returned!=1 || heuristic.state!=CUBLAS_STATUS_SUCCESS || heuristic.workspaceSize>workspace_bytes)
      throw std::runtime_error("no eligible bounded cuBLASLt algorithm");
    size_t written=0;
    blas(cublasLtMatmulAlgoConfigGetAttribute(&heuristic.algo,CUBLASLT_ALGO_CONFIG_ID,&algorithm_id,sizeof(algorithm_id),&written),"algorithm identity");
    auto execute=[&](int mode,float* dd) {
      if(mode==0) blas(cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&alpha,da,CUDA_R_16F,m,db,CUDA_R_16F,k,
                                  &beta,dd,CUDA_R_32F,m,CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT),"cuBLAS FP16/FP32 GEMM");
      else if(mode==1) blas(cublasLtMatmul(lt,operation,&alpha,da,al,db,bl,&beta,dc,cl,dd,cl,&heuristic.algo,workspace,workspace_bytes,resources.stream),"cuBLASLt GEMM");
      else {
        Gemm gemm;
        Gemm::Arguments arguments({m,n,k},{da,m},{db,k},{dc,m},{dd,m},{alpha,beta});
        size_t needed=Gemm::get_workspace_size(arguments);
        if(needed>workspace_bytes) throw std::runtime_error("CUTLASS workspace exceeds caller budget");
        if(gemm(arguments,workspace,resources.stream)!=cutlass::Status::kSuccess) throw std::runtime_error("SM70 CUTLASS GEMM failed");
      }
    };
    auto validate=[&](float* dd) {
      check(cudaMemcpyAsync(result.data(),dd,result.size()*4,cudaMemcpyDeviceToHost,resources.stream),"result copy");
      check(cudaStreamSynchronize(resources.stream),"GEMM completion");
      if(result!=expected) throw std::runtime_error("FP16-input/FP32-output library GEMM integer oracle mismatch");
    };
    auto poison=[&](float* dd) {
      // Every expected result is finite. All-one FP32 bits are a quiet NaN.
      check(cudaMemsetAsync(dd,0xff,result.size()*4,resources.stream),"poison GEMM destination");
    };
    for(int mode=0;mode<3;++mode) { poison(outputs[mode]);execute(mode,outputs[mode]);validate(outputs[mode]); }
    for(auto pointer:outputs) poison(pointer);
    check(cudaStreamBeginCapture(resources.stream,cudaStreamCaptureModeThreadLocal),"GEMM capture begin");
    for(int mode=0;mode<3;++mode) execute(mode,outputs[mode]);
    resources.end_capture();
    resources.instantiate();
    check(cudaGraphLaunch(resources.replay,resources.stream),"GEMM graph replay");
    for(auto pointer:outputs) validate(pointer);
  } catch(const std::exception& error) { std::snprintf(primary,sizeof(primary),"%s",error.what()); }
  if(!resources.complete() || !resources.retire_graphs()) return probe_outcome(false,primary);
  auto retired=[&](cublasStatus_t status,const char* what) {
    if(status!=CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr,"cleanup %s: status%d\n",what,int(status));clean=false; }
  };
  if(preference) retired(cublasLtMatmulPreferenceDestroy(preference),"preference");
  for(auto layout:{al,bl,cl}) if(layout) retired(cublasLtMatrixLayoutDestroy(layout),"layout");
  if(operation) retired(cublasLtMatmulDescDestroy(operation),"operation");
  if(lt) retired(cublasLtDestroy(lt),"cuBLASLt handle");
  if(handle) retired(cublasDestroy(handle),"cuBLAS handle");
  if(!clean) return probe_outcome(false,primary);
  clean=resources.cleanup(true) && clean;
  if(int outcome=probe_outcome(clean,primary)) return outcome;
  std::printf("{\"device_execution\":\"passed\",\"eager_and_graph\":true,\"cublasLt_algorithm\":%d,\"workspace_ceiling_bytes\":33554432,\"comparison\":\"exact integer oracle; no performance claim\"}\n",algorithm_id);
  return 0;
}
