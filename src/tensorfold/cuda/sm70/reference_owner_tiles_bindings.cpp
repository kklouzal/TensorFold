#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <climits>
#include "../kernels/tensor_contracts.cuh"
using T = at::Tensor;
void reference_tiles_prepare_cuda();
void reference_keys_cuda(const T&,const T&,const T&,const T&,const T&,const T&,T&,T&,T&,T&,
                         int,int,int,int,int64_t,int64_t,int);
void reference_values_cuda(const T&,const T&,const T&,const T&,const T&,T&,T&,
                           int,int,int,int64_t,int64_t,int);
namespace {
void tensor(const T& value,const T& owner,at::ScalarType dtype,const char* name) {
    TORCH_CHECK(value.is_cuda() && value.device()==owner.device() && value.scalar_type()==dtype
                && value.is_contiguous(),name," needs contiguous same-device declared storage");
}
void payload(const T& data,const T& scales,int64_t bits,int64_t codec) {
    TORCH_CHECK((bits==8 && codec==0)||(bits==4 && (codec==9||codec==10)),"reference tiles require Q8 or compact Q4");
    tensor(data,data,bits==8?at::kChar:at::kByte,"packed history");
    tensor(scales,data,bits==8?at::kHalf:at::kFloat,"history metadata");
    TORCH_CHECK(data.dim()==3 && data.size(0)<=INT_MAX && data.size(1)==1 && data.size(2)==256*bits/8
        && scales.dim()==3 && scales.size(0)==data.size(0) && scales.size(1)==1
        && scales.size(2)==(bits==8?8:2),"reference tile payload shape differs from D256 cache");
    auto prop=at::cuda::getDeviceProperties(data.get_device());
    TORCH_CHECK(prop->major==7 && prop->minor==0,"reference tiles require SM70");
}
void controls(const T& data,const T& generations,T& status,T& out,int64_t slot,int64_t owner,int64_t layer) {
    tensor(generations,data,at::kLong,"history generations");
    tensor(status,data,at::kInt,"sticky status");
    tensor(out,data,at::kHalf,"reference working tile");
    TORCH_CHECK(generations.dim()==1 && generations.numel()==2 && status.dim()==1 && status.numel()==2
        && out.sizes()==at::IntArrayRef({128,256}) && slot>=0 && owner>=0 && layer>=0 && layer<=INT_MAX,
        "reference tile control/working extents differ");
}
void indices(const T& data,const T& offsets,const T& masks) {
    tensor(offsets,data,at::kLong,"prepared offsets");tensor(masks,data,at::kBool,"prepared masks");
    TORCH_CHECK(offsets.sizes()==at::IntArrayRef({128}) && masks.sizes()==at::IntArrayRef({128}),"reference offset/mask extent differs");
}
}
void keys(const T& data,const T& scales,const T& pages,const T& positions,const T& visible,
    const T& generations,T status,T offsets,T masks,T out,int64_t logical,int64_t page_tokens,
    int64_t bits,int64_t codec,int64_t slot,int64_t owner,int64_t layer) {
    payload(data,scales,bits,codec);controls(data,generations,status,out,slot,owner,layer);indices(data,offsets,masks);
    tensor(pages,data,at::kLong,"page offsets");tensor(positions,data,at::kLong,"logical positions");
    tensor(visible,data,at::kBool,"causal visibility");
    TORCH_CHECK(logical>0 && logical<=INT_MAX && page_tokens>0 && page_tokens<=INT_MAX
        && pages.dim()==1 && pages.numel()==(logical+page_tokens-1)/page_tokens
        && positions.dim()==1 && positions.numel()<=128 && visible.sizes()==positions.sizes(),
        "reference tile logical domain differs");
    tensorfold::check_disjoint(out,{&data,&scales,&pages,&positions,&visible,&generations,&status,&offsets,&masks},"reference Half tile overlap");
    tensorfold::check_disjoint(offsets,{&data,&scales,&pages,&positions,&visible,&generations,&status,&masks},"reference offsets overlap");
    tensorfold::check_disjoint(masks,{&data,&scales,&pages,&positions,&visible,&generations,&status},"reference mask overlap");
    tensorfold::check_disjoint(status,{&data,&scales,&pages,&positions,&visible,&generations},"reference status overlap");
    c10::cuda::CUDAGuard guard(data.device());
    reference_keys_cuda(data,scales,pages,positions,visible,generations,status,offsets,masks,out,
        logical,page_tokens,bits,codec,slot,owner,layer);
}
void values(const T& data,const T& scales,const T& offsets,const T& masks,const T& generations,
    T status,T out,int64_t count,int64_t bits,int64_t codec,int64_t slot,int64_t owner,int64_t layer) {
    payload(data,scales,bits,codec);controls(data,generations,status,out,slot,owner,layer);indices(data,offsets,masks);
    TORCH_CHECK(count>=0 && count<=128,"reference value tile exceeds its prepared rows");
    tensorfold::check_disjoint(out,{&data,&scales,&offsets,&masks,&generations,&status},"reference value tile overlap");
    tensorfold::check_disjoint(status,{&data,&scales,&offsets,&masks,&generations},"reference status overlap");
    c10::cuda::CUDAGuard guard(data.device());
    reference_values_cuda(data,scales,offsets,masks,generations,status,out,count,bits,codec,slot,owner,layer);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,module) {
    module.def("prepare", &reference_tiles_prepare_cuda, "Load reference preparation CUDA functions before budgeting");
    module.def("keys",&keys,"Fused reference page indices/masks/packed K decoding");
    module.def("values",&values,"Reuse prepared indices/masks for original packed V decoding");
}
