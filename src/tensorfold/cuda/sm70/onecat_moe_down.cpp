#include <torch/extension.h>
void onecat_down_prepare();
void onecat_down(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,
                at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t,int64_t,int64_t,bool);
void onecat_down_slots(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,
                       at::Tensor,at::Tensor,at::Tensor,int64_t,int64_t,int64_t);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
 m.def("prepare",&onecat_down_prepare);
 m.def("down",&onecat_down);
 m.def("slots",&onecat_down_slots);
}
