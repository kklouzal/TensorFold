// SPDX-License-Identifier: Apache-2.0
// Private H2/HV6 component boundary. No incumbent dispatch changes.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <optional>
#include "../kernels/tensor_contracts.cuh"

void value8_chain_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, float, int, at::Tensor&, at::Tensor&,
    at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&,
    const std::optional<at::Tensor>&, bool);
void value8_replay_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, int, at::Tensor&);
void value8_prepare_cuda();

namespace {
void check(const at::Tensor& t, at::ScalarType kind, const at::Device& device) {
    TORCH_CHECK(t.is_cuda() && t.device() == device && t.scalar_type() == kind && t.is_contiguous(),
                "value8 operand must be contiguous same-device CUDA with declared dtype");
}
void state(const at::Tensor& t, const at::Device& device) {
    check(t, at::kFloat, device);
    TORCH_CHECK(t.sizes() == at::IntArrayRef({6, 128, 128}), "value8 requires actual TP8 H2/HV6 state");
    const auto* props = at::cuda::getDeviceProperties(t.get_device());
    TORCH_CHECK(props->major == 7 && props->minor == 0, "value8 requires SM70");
}
void saved(const at::Tensor& k, const at::Tensor& v, const at::Tensor& g, const at::Tensor& b,
           int64_t rows, const at::Device& device) {
    check(k, at::kFloat, device); check(v, at::kHalf, device);
    check(g, at::kFloat, device); check(b, at::kFloat, device);
    TORCH_CHECK(k.dim() == 3 && k.size(0) >= rows && k.size(1) == 2 && k.size(2) == 128, "value8 saved K");
    TORCH_CHECK(v.dim() == 3 && v.size(0) >= rows && v.size(1) == 6 && v.size(2) == 128, "value8 saved V");
    TORCH_CHECK(g.dim() == 2 && b.dim() == 2 && g.size(0) >= rows && b.size(0) >= rows
                && g.size(1) == 6 && b.size(1) == 6, "value8 saved gate/beta");
}
}

void chain(const at::Tensor& p, const at::Tensor& conv, const at::Tensor& weight,
           const at::Tensor& initial, const at::Tensor& parameter, const at::Tensor& bias,
           const at::Tensor& norm, double eps, int64_t rows, at::Tensor out, at::Tensor sums,
           at::Tensor final, at::Tensor k, at::Tensor v, at::Tensor g, at::Tensor b, at::Tensor q,
           std::optional<at::Tensor> keep, bool negative) {
    TORCH_CHECK(rows >= 0 && rows <= 65535, "value8 row count exceeds CUDA grid-y contract");
    const auto device = p.device();
    state(initial, device); state(final, device);
    for (const auto* t : std::initializer_list<const at::Tensor*>{&p, &conv, &weight, &norm, &out}) check(*t, at::kHalf, device);
    for (const auto* t : std::initializer_list<const at::Tensor*>{&parameter, &bias, &sums, &q}) check(*t, at::kFloat, device);
    TORCH_CHECK(p.dim() == 2 && p.size(0) >= rows && p.size(1) == 2060, "value8 projection layout");
    TORCH_CHECK(conv.sizes() == at::IntArrayRef({3, 1280}) && weight.sizes() == at::IntArrayRef({1280, 4}), "value8 conv layout");
    TORCH_CHECK(parameter.sizes() == at::IntArrayRef({6}) && bias.sizes() == at::IntArrayRef({6})
                && norm.sizes() == at::IntArrayRef({128}), "value8 weights layout");
    TORCH_CHECK(out.dim() == 2 && out.size(0) >= rows && out.size(1) == 768
                && sums.dim() == 2 && sums.size(0) >= rows && sums.size(1) == 24, "value8 output layout");
    TORCH_CHECK(q.dim() == 3 && q.size(0) >= rows && q.size(1) == 2 && q.size(2) == 128, "value8 funded Q workspace");
    TORCH_CHECK(std::isfinite(eps) && eps > 0 && std::isfinite(float(eps)) && float(eps) > 0, "value8 FP32 epsilon");
    saved(k, v, g, b, rows, device);
    if (keep) { check(*keep, at::kInt, device); TORCH_CHECK(keep->numel() == 1, "value8 live row control"); }
    const at::Tensor* reads[] = {&p, &conv, &weight, &initial, &parameter, &bias, &norm};
    const int64_t read_counts[] = {rows*2060, 3*1280, 1280*4, 6*128*128, 6, 6, 128};
    at::Tensor* writes[] = {&out, &sums, &final, &k, &v, &g, &b, &q};
    const int64_t counts[] = {rows*768, rows*24, 6*128*128, rows*256, rows*768, rows*6, rows*6, rows*256};
    for (int w = 0; w < 8; ++w) {
        for (int r = 0; r < 7; ++r) TORCH_CHECK(!tensorfold::overlaps(*writes[w], *reads[r], counts[w], read_counts[r]),
            "value8 write aliases active input");
        for (int other = 0; other < w; ++other) TORCH_CHECK(!tensorfold::overlaps(*writes[w], *writes[other], counts[w], counts[other]),
            "value8 writes must be disjoint");
        if (keep) TORCH_CHECK(!tensorfold::overlaps(*writes[w], *keep, counts[w], 1), "value8 write aliases row control");
    }
    c10::cuda::CUDAGuard guard(device);
    value8_chain_cuda(p, conv, weight, initial, parameter, bias, norm, float(eps), int(rows),
                     out, sums, final, k, v, g, b, q, keep, negative);
}

void replay(const at::Tensor& initial, const at::Tensor& k, const at::Tensor& v,
            const at::Tensor& g, const at::Tensor& b, int64_t rows, at::Tensor final) {
    TORCH_CHECK(rows >= 0 && rows <= 65535, "value8 replay rows");
    state(initial, initial.device()); state(final, initial.device());
    saved(k, v, g, b, rows, initial.device());
    tensorfold::check_disjoint(final, {&initial, &k, &v, &g, &b}, "value8 replay final state aliases");
    c10::cuda::CUDAGuard guard(initial.device());
    value8_replay_cuda(initial, k, v, g, b, int(rows), final);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("chain", &chain); m.def("replay", &replay); m.def("prepare", &value8_prepare_cuda);
}
