// SM70 specialization of the fork's GDN chain/replay ABI, with complete buffer validation.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <limits>
#include <optional>
#include "../kernels/tensor_contracts.cuh"

void sm70_gdn_chain_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                         const at::Tensor&, const at::Tensor&, const at::Tensor&, double, int64_t,
                         at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&,
                         const std::optional<at::Tensor>&, bool);
void sm70_gdn_replay_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                          const at::Tensor&, int64_t, at::Tensor&);

namespace {
void check(const at::Tensor& value, at::ScalarType kind, const at::Device& device, const char* name) {
    TORCH_CHECK(value.is_cuda() && value.device() == device && value.scalar_type() == kind && value.is_contiguous(),
                name, ": expected contiguous same-device CUDA tensor with declared dtype");
}

void state_check(const at::Tensor& state, const at::Device& device) {
    check(state, at::kFloat, device, "recurrent state");
    TORCH_CHECK(state.dim() == 3 && state.size(1) == 128 && state.size(2) == 128,
                "recurrent state must be FP32 [NV,128,128]");
    TORCH_CHECK(state.size(0) == 6 || state.size(0) == 12 || state.size(0) == 24 || state.size(0) == 48,
                "GDN requires 2/6, 4/12, 8/24 or 16/48 key/value heads");
    const auto* prop = at::cuda::getDeviceProperties(state.get_device());
    TORCH_CHECK(prop->major == 7 && prop->minor == 0, "the FP16 GDN provider requires SM70");
}

void saved_check(const at::Tensor& key, const at::Tensor& value, const at::Tensor& decay,
                  const at::Tensor& beta, int64_t rows, int64_t nv, const at::Device& device) {
    check(key, at::kFloat, device, "saved keys");
    check(value, at::kHalf, device, "saved values");
    check(decay, at::kFloat, device, "saved decay");
    check(beta, at::kFloat, device, "saved beta");
    TORCH_CHECK(key.dim() == 3 && key.size(0) >= rows && key.size(1) == nv / 3 && key.size(2) == 128,
                "saved key geometry");
    TORCH_CHECK(value.dim() == 3 && value.size(0) >= rows && value.size(1) == nv && value.size(2) == 128,
                "saved value geometry");
    TORCH_CHECK(decay.dim() == 2 && beta.dim() == 2 && decay.size(0) >= rows && beta.size(0) >= rows
                && decay.size(1) == nv && beta.size(1) == nv, "saved gate geometry");
}
}  // namespace

void chain(const at::Tensor& projection, const at::Tensor& conv, const at::Tensor& weight,
           const at::Tensor& initial, const at::Tensor& log, const at::Tensor& bias,
           const at::Tensor& norm, double eps, int64_t rows, at::Tensor output, at::Tensor sums,
           at::Tensor final, at::Tensor keys, at::Tensor values, at::Tensor decay, at::Tensor beta,
           std::optional<at::Tensor> keep_rows, bool negative_decay) {
    state_check(initial, projection.device());
    state_check(final, projection.device());
    TORCH_CHECK(rows >= 0 && rows <= std::numeric_limits<int>::max(), "invalid GDN row count");
    TORCH_CHECK(std::isfinite(eps) && eps > 0 && std::isfinite(float(eps)) && float(eps) > 0,
                "GDN epsilon must be positive finite FP32");
    const auto device = projection.device();
    const int64_t nv = initial.size(0), nk = nv / 3, channels = 2 * nk * 128 + nv * 128;
    for (const auto* input : std::initializer_list<const at::Tensor*>{&projection, &conv, &weight, &norm, &output})
        check(*input, at::kHalf, device, "half GDN operand");
    for (const auto* input : std::initializer_list<const at::Tensor*>{&log, &bias, &sums})
        check(*input, at::kFloat, device, "float GDN operand");
    TORCH_CHECK(projection.dim() == 2 && projection.size(0) >= rows
                && projection.size(1) == channels + nv * 128 + 2 * nv, "GDN projection geometry");
    TORCH_CHECK(conv.sizes() == at::IntArrayRef({3, channels}) && weight.sizes() == at::IntArrayRef({channels, 4}),
                "GDN convolution geometry");
    TORCH_CHECK(norm.dim() == 1 && norm.numel() == 128 && log.dim() == 1 && bias.dim() == 1
                && log.numel() == nv && bias.numel() == nv && final.sizes() == initial.sizes(), "GDN weight/state geometry");
    TORCH_CHECK(output.dim() == 2 && output.size(0) >= rows && output.size(1) == nv * 128
                && sums.dim() == 2 && sums.size(0) >= rows && sums.size(1) == nv * 4, "GDN output geometry");
    saved_check(keys, values, decay, beta, rows, nv, device);
    if (keep_rows) {
        check(*keep_rows, at::kInt, device, "live GDN rows");
        TORCH_CHECK(keep_rows->numel() == 1, "live GDN rows requires one device integer");
    }
    const at::Tensor* reads[] = {&projection, &conv, &weight, &initial, &log, &bias, &norm};
    const int64_t read_count[] = {rows * projection.size(1), conv.numel(), weight.numel(), initial.numel(),
                                  log.numel(), bias.numel(), norm.numel()};
    at::Tensor* writes[] = {&output, &sums, &final, &keys, &values, &decay, &beta};
    const int64_t write_count[] = {rows * nv * 128, rows * nv * 4, final.numel(), rows * nk * 128,
                                   rows * nv * 128, rows * nv, rows * nv};
    for (int write = 0; write < 7; ++write) {
        if (keep_rows)
            TORCH_CHECK(!tensorfold::overlaps(*writes[write], *keep_rows, write_count[write], 1),
                        "GDN outputs must be disjoint from live-row control");
        for (int read = 0; read < 7; ++read)
            TORCH_CHECK(!tensorfold::overlaps(*writes[write], *reads[read], write_count[write], read_count[read]),
                        "GDN outputs must be disjoint from active inputs");
        for (int other = 0; other < write; ++other)
            TORCH_CHECK(!tensorfold::overlaps(*writes[write], *writes[other], write_count[write], write_count[other]),
                        "GDN outputs must be mutually disjoint");
    }
    c10::cuda::CUDAGuard guard(device);
    sm70_gdn_chain_cuda(projection, conv, weight, initial, log, bias, norm, eps, rows, output, sums,
                         final, keys, values, decay, beta, keep_rows, negative_decay);
}

void replay(const at::Tensor& initial, const at::Tensor& keys, const at::Tensor& values,
            const at::Tensor& decay, const at::Tensor& beta, int64_t rows, at::Tensor final) {
    state_check(initial, initial.device());
    state_check(final, initial.device());
    TORCH_CHECK(rows >= 0 && rows <= std::numeric_limits<int>::max() && initial.sizes() == final.sizes(),
                "invalid replay rows/state geometry");
    saved_check(keys, values, decay, beta, rows, initial.size(0), initial.device());
    tensorfold::check_disjoint(final, {&initial, &keys, &values, &decay, &beta}, "replay final state must be disjoint");
    c10::cuda::CUDAGuard guard(initial.device());
    sm70_gdn_replay_cuda(initial, keys, values, decay, beta, rows, final);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("chain", &chain);
    module.def("replay", &replay);
}
