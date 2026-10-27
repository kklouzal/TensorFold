#include <torch/extension.h>
#include <pybind11/stl.h>

at::Tensor gguf_decode(const at::Tensor&, int64_t, int64_t, int64_t, const at::Tensor&, bool, bool);
void gguf_experts(const at::Tensor&, const at::Tensor&,
                  const at::Tensor&, int64_t, int64_t, const at::Tensor&, int64_t, int64_t,
                  const at::Tensor&, int64_t, int64_t, const at::Tensor&, const at::Tensor&,
                  const at::Tensor&, at::Tensor);
void gguf_experts_splitk4(const at::Tensor&, const at::Tensor&,
                         const at::Tensor&, int64_t, int64_t, const at::Tensor&, int64_t, int64_t,
                         const at::Tensor&, int64_t, int64_t, const at::Tensor&, const at::Tensor&,
                         const at::Tensor&, at::Tensor);
void gguf_experts_blocks(const at::Tensor&, const at::Tensor&,
                         const at::Tensor&, int64_t, int64_t, const at::Tensor&, int64_t, int64_t,
                         const at::Tensor&, int64_t, int64_t, const at::Tensor&, const at::Tensor&,
                         const at::Tensor&, at::Tensor);
void gguf_experts_blocks2w(const at::Tensor&, const at::Tensor&,
                         const at::Tensor&, int64_t, int64_t, const at::Tensor&, int64_t, int64_t,
                         const at::Tensor&, int64_t, int64_t, const at::Tensor&, const at::Tensor&,
                         const at::Tensor&, at::Tensor);
void gguf_experts_exact(const at::Tensor&, const at::Tensor&,
                         const at::Tensor&, int64_t, int64_t, const at::Tensor&, int64_t, int64_t,
                         const at::Tensor&, int64_t, int64_t, const at::Tensor&, const at::Tensor&,
                         const at::Tensor&, at::Tensor);
void gguf_experts_grouped(const at::Tensor&, const at::Tensor&,
    const at::Tensor&, int64_t, int64_t, const at::Tensor&, int64_t, int64_t,
    const at::Tensor&, int64_t, int64_t, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor);
void gguf_dense32(const at::Tensor&, const at::Tensor&, at::Tensor, bool);
void gguf_dense_wmma(const at::Tensor&, const at::Tensor&, at::Tensor);
void gguf_concat32(const at::Tensor&, const std::vector<at::Tensor>&, at::Tensor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode", &gguf_decode);
    m.def("experts", &gguf_experts);
    m.def("experts_splitk4", &gguf_experts_splitk4);
    m.def("experts_blocks", &gguf_experts_blocks);
    m.def("experts_blocks2w", &gguf_experts_blocks2w);
    // Routed-only core: the owning family writes the original Dense shared
    // expert over initialized zero shared slots after this call.
    m.def("experts_exact", &gguf_experts_exact);
    m.def("experts_grouped", &gguf_experts_grouped);
    m.def("dense32", &gguf_dense32);
    m.def("dense_wmma", &gguf_dense_wmma);
    m.def("concat32", &gguf_concat32);
}
