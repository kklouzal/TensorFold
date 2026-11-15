// Python bindings for the Gemma 4 SM70 kernels (gemma4.cu). Callers own every buffer and its stream order;
// shapes/dtypes are validated once by the Python owner (tensorfold.families.gemma4.sm70), not per launch.
#include <torch/extension.h>

using at::Tensor;

void w4_rows_cuda(const Tensor&, int64_t, const Tensor&, const Tensor&, const Tensor&, int64_t, int64_t, Tensor&,
                  int64_t, bool);
void w4_expert_gateup_cuda(const Tensor&, int64_t, int64_t, const Tensor&, const Tensor&, const Tensor&,
                           const Tensor&, int64_t, int64_t, Tensor&);
void w4_expert_down_cuda(const Tensor&, int64_t, int64_t, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                         const Tensor&, int64_t, int64_t, Tensor&, int64_t);
void w4_dequant_cuda(const Tensor&, const Tensor&, int64_t, int64_t, Tensor&);
void bf16_rows_cuda(const Tensor&, int64_t, const Tensor&, int64_t, int64_t, Tensor&, int64_t, double);
void i8_rows_cuda(const Tensor&, int64_t, const Tensor&, const Tensor&, int64_t, int64_t, Tensor&, int64_t, double);
void router_cuda(const Tensor&, int64_t, const Tensor&, int64_t, int64_t, Tensor&, Tensor&, int64_t, const Tensor&,
                 int64_t, int64_t, Tensor&, Tensor&, Tensor&);
void route_cuda(const Tensor&, int64_t, int64_t, int64_t, const Tensor&, int64_t, int64_t, Tensor&, Tensor&, Tensor&);
void rms_cuda(const Tensor&, int64_t, const c10::optional<Tensor>&, int64_t, double, double, Tensor&);
void attn_tail_cuda(const Tensor&, int64_t, int64_t, int64_t, Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                    double, double, Tensor&, Tensor&, Tensor&);
void ffn_tail_cuda(const Tensor&, int64_t, int64_t, int64_t, Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                   const Tensor&, double, Tensor&, const c10::optional<Tensor>&);
void embed_cuda(const Tensor&, int64_t, const Tensor&, int64_t, double, Tensor&);
void qkv_prep_cuda(const Tensor&, int64_t, int64_t, int64_t, int64_t, const Tensor&, const Tensor&, const Tensor&,
                   const Tensor&, Tensor&, Tensor&, Tensor&, const c10::optional<Tensor>&, int64_t, double);
void attn_cuda(const Tensor&, int64_t, int64_t, int64_t, const Tensor&, const Tensor&, const c10::optional<Tensor>&,
               int64_t, const Tensor&, int64_t, int64_t, double, int64_t, int64_t, Tensor&, Tensor&, Tensor&, int64_t,
               Tensor&);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.attr("profile") = "gemma4-sm70-w4a16-v1";
    m.def("w4_rows", &w4_rows_cuda);
    m.def("w4_expert_gateup", &w4_expert_gateup_cuda);
    m.def("w4_expert_down", &w4_expert_down_cuda);
    m.def("w4_dequant", &w4_dequant_cuda);
    m.def("bf16_rows", &bf16_rows_cuda);
    m.def("i8_rows", &i8_rows_cuda);
    m.def("router", &router_cuda);
    m.def("route", &route_cuda);
    m.def("rms", &rms_cuda);
    m.def("attn_tail", &attn_tail_cuda);
    m.def("ffn_tail", &ffn_tail_cuda);
    m.def("embed", &embed_cuda);
    m.def("qkv_prep", &qkv_prep_cuda);
    m.def("attn", &attn_cuda);
}
