// Bindings for the SM70 decode shared-expert FFN (shared_ffn.cu).
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include "../kernels/tensor_contracts.cuh"

void shared_ffn(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor, at::Tensor);

void run(const at::Tensor& x, const at::Tensor& gate, const at::Tensor& up, const at::Tensor& down,
         at::Tensor act, at::Tensor out) {
    tensorfold::check_disjoint(act, {&x, &gate, &up, &down, &out}, "shared FFN activation must not overlap operands");
    tensorfold::check_disjoint(out, {&x, &gate, &up, &down}, "shared FFN output must not overlap inputs");
    c10::cuda::CUDAGuard guard(x.device());
    shared_ffn(x, gate, up, down, act, out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "FP16 rows -> FP32 shared-expert SwiGLU FFN output (1..32 rows)");
}
