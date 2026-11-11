#include <torch/extension.h>
void onecat_owned_gu(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,at::Tensor,int64_t,int64_t,bool);
void onecat_prepare();
void onecat_owned_gu_raw_iq2(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,at::Tensor,int64_t,bool,int64_t,int64_t);
void onecat_prepare_raw_iq2();
void onecat_raw_iq2_words(const at::Tensor&,at::Tensor,int64_t,int64_t,int64_t,int64_t);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("gate_up",&onecat_owned_gu);m.def("prepare",&onecat_prepare);m.def("gate_up_raw_iq2",&onecat_owned_gu_raw_iq2);m.def("prepare_raw_iq2",&onecat_prepare_raw_iq2);m.def("raw_iq2_words",&onecat_raw_iq2_words);}
