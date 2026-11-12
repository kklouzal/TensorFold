// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <limits>
#include <cstdint>
using T = at::Tensor;
void onecat_gpu_packing_prepare();
void onecat_gpu_packing_launch(const T&, const T&, const T&, T, T, T, T, int, int, int64_t, int64_t);
void onecat_gpu_union_packing_launch(const T&, const T&, const T&, T, T, T, T, int, int, int64_t, int64_t);
namespace {
struct Span { uintptr_t begin, end; };
Span span(const T& t) {
    constexpr uint64_t max = std::numeric_limits<int64_t>::max();
    uint64_t extent = t.element_size();
    for(int dim=0;dim<t.dim();++dim) {
        TORCH_CHECK(t.size(dim)>0 && t.stride(dim)>=0,"positive packing extents/nonnegative strides required");
        const uint64_t count=t.size(dim)-1, step=uint64_t(t.stride(dim))*t.element_size();
        TORCH_CHECK(!count || step <= (max-extent)/count,"packing span overflows");
        extent += count*step;
    }
    const uintptr_t begin=reinterpret_cast<uintptr_t>(t.data_ptr());
    TORCH_CHECK(extent <= std::numeric_limits<uintptr_t>::max()-begin,"packing pointer span overflows");
    return {begin,begin+extent};
}
void disjoint(Span a,Span b) {TORCH_CHECK(a.end<=b.begin || b.end<=a.begin,"packing scratch overlaps retained original/picks or another output");}
void operand(const T& t,const T& owner,at::ScalarType dtype) {
    TORCH_CHECK(t.is_cuda() && t.device()==owner.device() && t.scalar_type()==dtype && !t.is_neg() && !t.is_conj(),
                "packing requires same-device plain positive representation/dtype");
}
int64_t raw(const T& t,const T& owner,int bytes,int64_t offset) {
    operand(t,owner,at::kByte);
    TORCH_CHECK((t.dim()==2 && t.size(0)==40960) || (t.dim()==3 && t.size(0)==256 && t.size(1)==160),
                "local raw requires [40960,rowbytes] or [256,160,rowbytes]");
    const int64_t width=t.size(-1), rs=t.stride(-2), es=t.dim()==3?t.stride(0):0;
    TORCH_CHECK(t.stride(-1)==1 && rs>=width && width>=10*bytes && width%bytes==0 && offset>=0 && offset%256==0,
                "raw contiguous encoded columns/positive rows/complete256-value offset required");
    TORCH_CHECK(rs <= (std::numeric_limits<int64_t>::max()-width)/159,"raw row extent overflows");
    TORCH_CHECK(t.dim()==2 ? rs<=std::numeric_limits<int64_t>::max()/160 : (es==0 || es>=159*rs+width),
                "raw expert strides overlap or overflow");
    TORCH_CHECK(offset/256 <= (width-10*bytes)/bytes,"raw selected K2560 block prefix exceeds row");
    span(t);
    return offset/256*bytes;
}
template<bool Union>
void run(const T& gate,const T& up,const T& picks,T gc,T gs,T uc,T us,int64_t first,int64_t kind,int64_t go,int64_t uo) {
    TORCH_CHECK(kind==18 || kind==21 || kind==22 || kind==23,"four original GGUF18/21/22/23 formats only");
    TORCH_CHECK(first==0 || first==256,"owned first must be0 or256");
    TORCH_CHECK(picks.is_cuda() && (picks.scalar_type()==at::kInt || picks.scalar_type()==at::kLong)
        && picks.is_contiguous() && picks.dim()==2 && picks.size(1)==11 && picks.size(0)>=1
        && (Union ? picks.numel()<=65535 : picks.size(0)<=32)
        && !picks.is_neg() && !picks.is_conj(),"picks requires contiguous I32/I64 rows: activeR<=32 or unionpairs<=65535");
    const auto* props=at::cuda::getDeviceProperties(picks.get_device());
    TORCH_CHECK(props->major==7 && props->minor==0,"packing requires SM70");
    const int bb=kind==18?98:kind==21?110:kind==22?82:136;
    const int cb=kind==23?204800:kind==22?102400:153600, sb=kind==22?51200:kind==18?12800:25600;
    const int64_t g=raw(gate,picks,bb,go),u=raw(up,picks,bb,uo);
    const T* outputs[]={&gc,&gs,&uc,&us};
    Span written[4], originals[3];
    for(int i=0;i<4;++i) {
        const T& t=*outputs[i];operand(t,picks,at::kByte);
        TORCH_CHECK(t.is_contiguous() && t.dim()==2 && t.size(0)==256 && t.size(1)==(i%2?sb:cb)
            && reinterpret_cast<uintptr_t>(t.data_ptr())%16==0,"exact contiguous uint4-aligned plane views required");
        written[i]=span(t);
        // The retained ORIGINAL bank is immutable as an owned whole, including
        // row/expert gaps and unselected columns. No scratch inside that span.
        if(i==0) { originals[0]=span(gate);originals[1]=span(up);originals[2]=span(picks); }
        disjoint(written[i],originals[0]);disjoint(written[i],originals[1]);disjoint(written[i],originals[2]);
        for(int j=0;j<i;++j)disjoint(written[i],written[j]);
    }
    c10::cuda::CUDAGuard guard(picks.device());
    if constexpr (Union) onecat_gpu_union_packing_launch(gate,up,picks,gc,gs,uc,us,int(kind),int(first),g,u);
    else onecat_gpu_packing_launch(gate,up,picks,gc,gs,uc,us,int(kind),int(first),g,u);
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("prepare",&onecat_gpu_packing_prepare);m.def("pack_active",&run<false>);m.def("pack_union",&run<true>);}
