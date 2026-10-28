// Private standalone Q8_32/FP32-scale expert prototype; no model registration.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "../kernels/tensor_contracts.cuh"

void tf_dp4a_quantize(const at::Tensor&,at::Tensor,int,at::Tensor,int);
void tf_dp4a_project(const at::Tensor&,const at::Tensor&,const at::Tensor&,int,bool,at::Tensor,at::Tensor);
void tf_dp4a_activate(const at::Tensor&,const at::Tensor&,at::Tensor);
void tf_dp4a_mask(at::Tensor,at::Tensor);
void tf_dp4a_debug(const at::Tensor&,const at::Tensor&,int,at::Tensor,at::Tensor,at::Tensor);
namespace {
void check(const at::Tensor& t,at::ScalarType type,const at::Device& device,const char* label) {
    TORCH_CHECK(t.is_cuda() && t.device()==device && t.scalar_type()==type && t.is_contiguous(),label,": contiguous same-device CUDA operand required");
}
void status_check(const at::Tensor& status,const at::Device& device) {
    check(status,at::kInt,device,"status");
    TORCH_CHECK(status.numel()==1 && reinterpret_cast<uintptr_t>(status.data_ptr())%4==0,"status requires one aligned latched int32 word");
    const auto* p=at::cuda::getDeviceProperties(status.get_device());
    TORCH_CHECK(p->major==7 && p->minor==0,"private DP4A profile requires SM70");
}
void qcheck(const at::Tensor& q,const at::Device& device) {
    check(q,at::kByte,device,"Q8_32");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr())%4==0,"Q8_32 FP32 scale requires4-byte alignment");
    TORCH_CHECK(q.dim()==3 && q.size(0)>0 && q.size(0)<=352 && q.size(2)==36 && (q.size(1)==3 || q.size(1)==80),"private36-byte Q8_32 shape");
}
}
void quantize(const at::Tensor& x,at::Tensor q,int64_t prefix,at::Tensor status) {
    const auto device=x.device();check(x,at::kHalf,device,"activation");qcheck(q,device);status_check(status,device);
    TORCH_CHECK(x.dim()==2 && x.size(0)==q.size(0) && ((x.size(1)==2560 && x.size(0)<=32 && prefix==0 && q.size(1)==80)
        || (x.size(1)==80 && (prefix==0 || prefix==16) && q.size(1)==3)),"activation/crop geometry");
    tensorfold::check_disjoint(q,{&x,&status},"Q8 output must not alias activation/status");
    tensorfold::check_disjoint(status,{&x},"status must not alias activation");
    c10::cuda::CUDAGuard guard(device);tf_dp4a_quantize(x,q,prefix,status,x.size(1)==2560?1:4);
}
void project(const at::Tensor& q,const at::Tensor& picks,const at::Tensor& weight,int64_t kind,bool down,at::Tensor out,at::Tensor status) {
    const auto device=q.device();qcheck(q,device);check(picks,at::kInt,device,"picks");check(weight,at::kByte,device,"packed weights");status_check(status,device);
    TORCH_CHECK(picks.dim()==2 && picks.size(0)>0 && picks.size(0)<=32 && picks.size(1)>0 && picks.size(1)<=11,"picks must be int32[R<=32,S<=11]");
    TORCH_CHECK(out.scalar_type()==at::kFloat || out.scalar_type()==at::kHalf,"output requires FP32/FP16");check(out,out.scalar_type(),device,"output");
    TORCH_CHECK(out.dim()==3 && out.size(0)==picks.size(0) && out.size(1)==picks.size(1),"projection output row/slot shape");
    if(down) { TORCH_CHECK(kind==20 && q.size(1)==3 && q.size(0)==picks.numel() && out.size(2)==2560
        && weight.sizes()==at::IntArrayRef({512*2560,54}),"down requires retainedIQ4_NL80/crop0or16"); }
    else { TORCH_CHECK((kind==18 || kind==22) && q.size(1)==80 && q.size(0)==picks.size(0)
        && out.size(2)==80 && out.scalar_type()==at::kFloat
        && weight.sizes()==at::IntArrayRef({512*80,kind==18?980:820}),"gate/up require retainedIQ3_XXS orIQ2_S K2560"); }
    tensorfold::check_disjoint(out,{&q,&picks,&weight,&status},"projection outputs must be disjoint");
    tensorfold::check_disjoint(status,{&q,&picks,&weight},"status must not alias source");
    c10::cuda::CUDAGuard guard(device);tf_dp4a_project(q,picks,weight,kind,down,out,status);
}
void activate(const at::Tensor& gate,const at::Tensor& up,at::Tensor h) {
    const auto device=gate.device();check(gate,at::kFloat,device,"gate");check(up,at::kFloat,device,"up");check(h,at::kHalf,device,"hidden");
    TORCH_CHECK(gate.sizes()==up.sizes() && h.sizes()==gate.sizes() && h.dim()==3 && h.size(0)>0 && h.size(0)<=32 && h.size(1)>0 && h.size(1)<=11 && h.size(2)==80,"SwiGLU geometry");
    tensorfold::check_disjoint(h,{&gate,&up},"hidden must not alias gate/up");
    c10::cuda::CUDAGuard guard(device);tf_dp4a_activate(gate,up,h);
}
void mask(at::Tensor out,at::Tensor status) {
    const auto device=out.device();status_check(status,device);TORCH_CHECK(out.scalar_type()==at::kHalf || out.scalar_type()==at::kFloat,"mask output dtype");check(out,out.scalar_type(),device,"output");
    TORCH_CHECK(out.dim()==3 && out.size(0)>0 && out.size(0)<=32 && out.size(1)>0 && out.size(1)<=11 && out.size(2)==2560,"mask output geometry");
    tensorfold::check_disjoint(out,{&status},"mask/status must be disjoint");
    c10::cuda::CUDAGuard guard(device);tf_dp4a_mask(out,status);
}
void debug(const at::Tensor& q,const at::Tensor& weight,int64_t kind,at::Tensor integers,at::Tensor values,at::Tensor status) {
    const auto device=q.device();qcheck(q,device);check(weight,at::kByte,device,"debug weights");check(integers,at::kInt,device,"numerators");check(values,at::kFloat,device,"terms");status_check(status,device);
    TORCH_CHECK(q.size(0)==1 && ((kind==18 && q.size(1)==80 && weight.sizes()==at::IntArrayRef({1,980}))
        || (kind==22 && q.size(1)==80 && weight.sizes()==at::IntArrayRef({1,820}))
        || (kind==20 && q.size(1)==3 && weight.sizes()==at::IntArrayRef({1,54}))),"debug single-row format geometry");
    TORCH_CHECK(integers.sizes()==at::IntArrayRef({q.size(1)}) && values.sizes()==integers.sizes(),"debug output shapes");
    tensorfold::check_disjoint(integers,{&q,&weight,&values,&status},"debug integer alias");tensorfold::check_disjoint(values,{&q,&weight,&status},"debug float alias");
    c10::cuda::CUDAGuard guard(device);tf_dp4a_debug(q,weight,kind,integers,values,status);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("quantize",&quantize);m.def("project",&project);m.def("activate",&activate);m.def("mask",&mask);m.def("debug",&debug); }
