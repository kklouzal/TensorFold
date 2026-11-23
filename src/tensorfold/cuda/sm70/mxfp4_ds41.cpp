#include <torch/extension.h>
void mxfp4_quantize(torch::Tensor, const torch::Tensor&);
void mxfp4_gate_up(torch::Tensor, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                   const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, bool);
void mxfp4_swiglu(torch::Tensor, torch::Tensor, const torch::Tensor&, bool);
void mxfp4_down(torch::Tensor, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, bool);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize", &mxfp4_quantize);
  m.def("gate_up", &mxfp4_gate_up);
  m.def("swiglu", &mxfp4_swiglu);
  m.def("down", &mxfp4_down);
}
