#include <torch/extension.h>
void onecat_owned_gu(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,at::Tensor,int64_t,int64_t,bool);
void onecat_prepare();
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("gate_up",&onecat_owned_gu);m.def("prepare",&onecat_prepare);}
