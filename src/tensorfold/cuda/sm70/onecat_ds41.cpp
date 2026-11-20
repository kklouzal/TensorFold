#include <torch/extension.h>
void ds41_quantize_q8_1(torch::Tensor, const torch::Tensor&);
void ds41_swiglu_q8(torch::Tensor, const torch::Tensor&, const torch::Tensor&);
void ds41_shared_swiglu(torch::Tensor, const torch::Tensor&, const torch::Tensor&);
void ds41_gate_up(torch::Tensor, const torch::Tensor&, const torch::Tensor&,
                  const torch::Tensor&, const torch::Tensor&, int64_t);
void ds41_gate_adaptive(torch::Tensor, const torch::Tensor&, const torch::Tensor&,
                        const torch::Tensor&, const torch::Tensor&, int64_t);
void ds41_down(torch::Tensor, const torch::Tensor&, const torch::Tensor&,
               const torch::Tensor&, int64_t, int64_t);
void ds41_dequantize(torch::Tensor, const torch::Tensor&, int64_t);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize_q8_1", &ds41_quantize_q8_1);
  m.def("swiglu_q8", &ds41_swiglu_q8);
  m.def("shared_swiglu", &ds41_shared_swiglu);
  m.def("gate_up", &ds41_gate_up);
  m.def("gate_adaptive", &ds41_gate_adaptive);
  m.def("down", &ds41_down);
  m.def("dequantize", &ds41_dequantize);
}
