// Existing half cache/buffer ownership with fixed-position arithmetic and current CUDA stream.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <optional>
#include <cmath>
#include <algorithm>
#include <utility>
#include <limits>

using T = at::Tensor;
using O = std::optional<T>;

void sm70_attention_cuda(const T&, const T&, const T&, const T&, const T&, const T&, const T&,
                         T&, T&, T&, T&, const O&, const O&, int, int, int, int, bool, float, int, int);

namespace {
int partial_slots(int context) {
    int slots = (context + 127) / 128;
    for (const auto& threshold : {std::pair<int, int>{1, 1}, {881, 10}, {2731, 38}, {2737, 39},
                                 {2809, 40}, {2882, 41}, {5096, 49}, {5104, 50}}) {
        if (context >= threshold.first) slots = std::max(slots, threshold.second);
    }
    return slots;
}

void check(const T& tensor, const T& owner, at::ScalarType dtype, const char* name) {
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == owner.device() && tensor.scalar_type() == dtype
                && tensor.is_contiguous(), name, " needs contiguous same-device storage with declared dtype");
}
}  // namespace

void run_impl(const T& q, const T& k, const T& v, const T& position, const T& sparse, const T& counts, const T& ids,
          T scores, T probabilities, T partials, T out, O ks, O vs,
          int rows, int context, int kbits, int vbits, bool qsa, double scale, int kcodec, int vcodec) {
    check(q, q, at::kHalf, "queries");
    const auto* prop = at::cuda::getDeviceProperties(q.get_device());
    TORCH_CHECK(prop->major == 7 && prop->minor == 0, "fixed-position attention requires SM70");
    TORCH_CHECK(q.dim() == 3 && rows >= 0 && rows <= 24 && q.size(0) >= rows && q.size(1) == 3 && q.size(2) == 256, "query geometry");
    const bool kr = kbits == 4 && (kcodec == 4 || kcodec == 5);
    const bool vr = vbits == 4 && (vcodec == 4 || vcodec == 5);
    TORCH_CHECK((kr || (kcodec == 0 && (kbits == 0 || kbits == 8))) &&
                (vr || (vcodec == 0 && (vbits == 0 || vbits == 8))), "raw/int8 or CODEC4/5 Q4 key/value formats required");
    check(k, q, kr ? at::kByte : kbits ? at::kChar : at::kHalf, "keys");
    check(v, q, vr ? at::kByte : vbits ? at::kChar : at::kHalf, "values");
    TORCH_CHECK(k.dim() == 3 && v.dim() == 3 && v.size(0) == k.size(0) && v.size(1) == k.size(1)
                && k.size(1) == 1 && q.size(1) % k.size(1) == 0
                && k.size(2) == q.size(2) / (kr ? 2 : 1) && v.size(2) == q.size(2) / (vr ? 2 : 1)
                && context >= 65 && context <= 8192 && context <= k.size(0), "cache/context geometry");
    TORCH_CHECK(position.scalar_type() == at::kInt || position.scalar_type() == at::kLong, "position must be int32 or int64");
    check(position, q, position.scalar_type(), "position"); TORCH_CHECK(position.numel() == 1, "one base position required");
    for (const auto* x : {&sparse, &counts, &ids}) check(*x, q, at::kInt, "attention masks");
    TORCH_CHECK(sparse.numel() >= rows && counts.numel() >= rows && ids.dim() == 2 && ids.size(0) >= rows, "mask geometry");
    check(scores, q, at::kFloat, "score workspace"); check(probabilities, q, at::kHalf, "probability workspace");
    check(partials, q, at::kFloat, "partial workspace"); check(out, q, at::kHalf, "attention output");
    TORCH_CHECK(scores.dim() == 3 && scores.size(0) >= rows && scores.size(1) == q.size(1)
                && scores.size(2) >= context && scores.size(2) % 32 == 0 && probabilities.sizes() == scores.sizes()
                && partials.dim() == 4 && partials.size(0) >= rows && partials.size(1) == q.size(1)
                && partials.size(2) >= partial_slots(context) && partials.size(3) == q.size(2)
                && out.dim() == 3 && out.size(0) >= rows && out.size(1) == q.size(1) && out.size(2) == q.size(2), "workspace geometry");
    if (kbits) { TORCH_CHECK(ks.has_value(), "key scales missing"); check(*ks, q, kr ? at::kFloat : at::kHalf, "key scales");
        TORCH_CHECK(ks->numel() == k.size(0) * k.size(1) * (q.size(2) / (kr ? 128 : 32)), "key scale geometry"); }
    if (vbits) { TORCH_CHECK(vs.has_value(), "value scales missing"); check(*vs, q, vr ? at::kFloat : at::kHalf, "value scales");
        TORCH_CHECK(vs->numel() == v.size(0) * v.size(1) * (q.size(2) / (vr ? 128 : 32)), "value scale geometry"); }
    TORCH_CHECK(std::isfinite(scale) && scale > 0 && scale <= double(std::numeric_limits<float>::max()),
                "positive finite attention scale required");
    const float native_scale = float(scale);
    TORCH_CHECK(std::isfinite(native_scale) && native_scale > 0, "positive finite attention scale required");
    c10::cuda::CUDAGuard guard(q.device());
    sm70_attention_cuda(q, k, v, position, sparse, counts, ids, scores, probabilities, partials, out,
                         ks, vs, rows, context, kbits, vbits, qsa, native_scale, kcodec, vcodec);
}

void run(const T& q, const T& k, const T& v, const T& position, const T& sparse, const T& counts, const T& ids,
         T scores, T probabilities, T partials, T out, O ks, O vs,
         int rows, int context, int kbits, int vbits, bool qsa, double scale) {
    run_impl(q, k, v, position, sparse, counts, ids, scores, probabilities, partials, out, ks, vs,
             rows, context, kbits, vbits, qsa, scale, 0, 0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run);
    m.def("run_rotor", &run_impl);
}
