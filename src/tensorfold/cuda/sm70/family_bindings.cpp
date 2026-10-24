// Fused half family ABI: existing buffers, current stream, no host tensor readback.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <optional>
#include <algorithm>

using T = at::Tensor;
using O = std::optional<T>;

void family_writeback_cuda(const T&, T&, T&, int, int, const O&, const O&, const O&, const O&);
void family_normed_cuda(const T&, const T&, const T&, T&, T&, int, float);
void family_act_cuda(const T&, T&, T&, const O&, int, int);
void family_mix_cuda(const T&, const T&, T&, T&, int);
void family_rms_cuda(const T&, const T&, T&, T&, int, float);
void family_gate_cuda(const T&, const T&, T&, T&, int, int);
void family_ple_embed_cuda(const T&, T&, T&, int, int, int, float);
void family_ple_gate_cuda(const T&, const T&, const T&, const T&, const T&, T&, T&, const O&, int, float, bool);
void family_ple_conv_cuda(const T&, const T&, const T&, const T&, const T&, const T&, T&, T&, int, int, float);
void family_add_cuda(const T&, const T&, T&, int);
void family_partial_cuda(const T&, const T&, T&, int);
void family_shift_cuda(T&, const T&, int, int);
void family_prep_cuda(const T&, const T&, const T&, const T&, const T&, const T&, T&, T&, T&, T&, T&,
                      const O&, const O&, const O&, const O&, const O&, int, int, int, int, int, int, int, int, float, float, bool);
void family_pool_cuda(const T&, T&, const T&, const T&, const T&, const O&, int, int, float, float);
void family_qsa_dense_cuda(const T&, T&, T&, int);
void family_qsa_rows_cuda(const T&, const T&, const T&, T&, T&, T&, T&, int, int, int, int);

namespace {
void check(const T& tensor, const T& owner, at::ScalarType dtype, const char* name) {
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == owner.device() && tensor.scalar_type() == dtype
                && tensor.is_contiguous(), name, " requires contiguous same-device tensor with declared dtype");
}
void check_half(const T& tensor) {
    check(tensor, tensor, at::kHalf, "FP16 family operand");
    const auto* prop = at::cuda::getDeviceProperties(tensor.get_device());
    TORCH_CHECK(prop->major == 7 && prop->minor == 0, "fused FP16 family requires SM70");
}
void floating(const T& tensor, const T& owner) {
    TORCH_CHECK(tensor.scalar_type() == at::kHalf || tensor.scalar_type() == at::kFloat,
                "family partials require FP16 or FP32 storage");
    check(tensor, owner, tensor.scalar_type(), "family partials");
}
void epsilon(double eps) { TORCH_CHECK(std::isfinite(eps) && float(eps) > 0 && std::isfinite(float(eps)), "invalid epsilon"); }
void groups(const T& out, const T& xs) {
    check(xs, out, at::kFloat, "group sums");
    TORCH_CHECK(out.dim() == 2 && out.size(1) % 32 == 0 && xs.numel() == out.numel() / 32, "group-sum geometry");
}
void same_half(const T& tensor, const T& owner) { check(tensor, owner, at::kHalf, "half operand"); }
}  // namespace

void writeback(const T& h, T hout, T pss, int streams, int mode, O branch, O inject, O y, O wts) {
    check_half(h); same_half(hout, h); check(pss, h, at::kFloat, "PSS");
    TORCH_CHECK(h.dim() == 2 && streams > 0 && h.size(1) % (streams * 256) == 0 && hout.sizes() == h.sizes(), "HC geometry");
    const int rows = h.size(0), dims = h.size(1) / streams;
    TORCH_CHECK(pss.numel() == int64_t(rows) * streams * (dims / 256), "PSS geometry");
    TORCH_CHECK(mode >= 0 && mode <= 4, "invalid HC mode");
    if (mode) {
        TORCH_CHECK(inject.has_value(), "HC injection is required");
        same_half(*inject, h);
        TORCH_CHECK(inject->numel() == rows * streams, "injection geometry");
        if (mode == 2) {
            TORCH_CHECK(y.has_value() && wts.has_value(), "MoE writeback operands missing");
            floating(*y, h); check(*wts, h, at::kFloat, "route weights");
            TORCH_CHECK(y->dim() == 3 && y->size(0) == rows && y->size(2) == dims
                        && wts->sizes() == y->sizes().slice(0, 2), "MoE slot geometry");
        } else {
            TORCH_CHECK(branch.has_value(), "HC branch is required"); floating(*branch, h);
            TORCH_CHECK((mode == 1 && branch->sizes() == at::IntArrayRef({rows, dims}))
                        || (mode >= 3 && branch->dim() == 3 && branch->size(1) == rows && branch->size(2) == dims
                            && branch->size(0) > 0), "HC branch geometry");
        }
    }
    c10::cuda::CUDAGuard guard(h.device()); family_writeback_cuda(h, hout, pss, streams, mode, branch, inject, y, wts);
}

void normed(const T& h, const T& pss, const T& scale, T out, T xs, int streams, double eps) {
    check_half(h); same_half(out, h); check(pss, h, at::kFloat, "PSS"); check(scale, h, at::kFloat, "HC scale");
    epsilon(eps); groups(out, xs);
    TORCH_CHECK(h.dim() == 2 && streams > 0 && h.size(1) % (streams * 256) == 0
                && out.sizes() == h.sizes() && scale.numel() == h.size(1)
                && pss.numel() == h.numel() / 256, "HC norm geometry");
    c10::cuda::CUDAGuard guard(h.device()); family_normed_cuda(h, pss, scale, out, xs, streams, float(eps));
}

void act(const T& input, T out, T xs, O inject, int streams, int low) {
    check_half(out); floating(input, out); groups(out, xs);
    TORCH_CHECK(streams > 0 && low > 0 && low % 32 == 0 && out.size(1) == low, "HC activation geometry");
    const int rows = out.size(0);
    TORCH_CHECK((input.dim() == 2 && input.size(0) == rows)
                || (input.dim() == 3 && input.size(0) > 0 && input.size(1) == rows), "HC activation rows");
    TORCH_CHECK(input.size(-1) == low + (inject.has_value() ? streams : 0), "HC activation columns");
    if (inject) { same_half(*inject, out); TORCH_CHECK(inject->numel() == rows * streams, "HC inject geometry"); }
    c10::cuda::CUDAGuard guard(out.device()); family_act_cuda(input, out, xs, inject, streams, low);
}

void mix(const T& up, const T& normalized, T out, T xs, int streams) {
    check_half(up); same_half(normalized, up); same_half(out, up); groups(out, xs);
    TORCH_CHECK(up.dim() == 2 && normalized.sizes() == up.sizes() && streams > 0
                && up.size(1) == out.size(1) * streams && up.size(0) == out.size(0), "HC stream mix geometry");
    c10::cuda::CUDAGuard guard(up.device()); family_mix_cuda(up, normalized, out, xs, streams);
}

void rms(const T& x, const T& weight, T out, T xs, int group, double eps) {
    check_half(x); same_half(out, x); check(weight, x, at::kFloat, "RMS scale"); groups(out, xs); epsilon(eps);
    TORCH_CHECK(x.dim() == 2 && out.sizes() == x.sizes() && group > 0 && group % 32 == 0
                && x.size(1) % group == 0 && weight.numel() == x.size(1), "RMS geometry");
    c10::cuda::CUDAGuard guard(x.device()); family_rms_cuda(x, weight, out, xs, group, float(eps));
}

void gate(const T& value, const T& projection, T out, T xs, int heads, int dim) {
    check_half(value); same_half(projection, value); same_half(out, value); groups(out, xs);
    TORCH_CHECK(value.dim() == 3 && heads > 0 && dim % 32 == 0 && value.size(1) == heads && value.size(2) == dim
                && projection.dim() == 2 && projection.size(0) == value.size(0) && projection.size(1) >= 2 * heads * dim
                && out.sizes() == at::IntArrayRef({value.size(0), int64_t(heads) * dim}), "attention gate geometry");
    c10::cuda::CUDAGuard guard(value.device()); family_gate_cuda(value, projection, out, xs, heads, dim);
}

void ple_embed(const T& values, T out, T xs, int rows, int heads, int dim, double scale) {
    check_half(values); same_half(out, values); groups(out, xs);
    TORCH_CHECK(rows >= 0 && heads > 0 && dim > 0 && out.sizes() == at::IntArrayRef({rows, int64_t(heads) * dim})
                && values.numel() >= int64_t(rows) * heads * dim && std::isfinite(scale), "PLE embedding geometry");
    c10::cuda::CUDAGuard guard(out.device()); family_ple_embed_cuda(values, out, xs, rows, heads, dim, float(scale));
}

void ple_gate(const T& keys, const T& values, const T& h, const T& nk, const T& nq, T gated, T pss,
               int streams, double eps, O valid_rows, bool independent_rows) {
    check_half(h); for (const auto* x : std::initializer_list<const T*>{&keys, &values, &gated}) same_half(*x, h);
    check(nk, h, at::kFloat, "PLE key norm"); check(nq, h, at::kFloat, "PLE query norm");
    check(pss, h, at::kFloat, "PLE PSS"); epsilon(eps);
    if (valid_rows) { check(*valid_rows, h, at::kInt, "PLE live rows"); TORCH_CHECK(valid_rows->numel() == 1, "PLE live row geometry"); }
    TORCH_CHECK(h.dim() == 2 && streams > 0 && h.size(1) % streams == 0 && keys.sizes() == h.sizes()
                && gated.sizes() == h.sizes() && values.sizes() == at::IntArrayRef({h.size(0), h.size(1) / streams})
                && nk.numel() == h.size(1) && nq.numel() == h.size(1) && pss.numel() == h.size(0) * streams, "PLE gate geometry");
    c10::cuda::CUDAGuard guard(h.device());
    family_ple_gate_cuda(keys, values, h, nk, nq, gated, pss, valid_rows, streams, float(eps), independent_rows);
}

void ple_conv(const T& gated, const T& pss, const T& norm, const T& tail, const T& weight, const T& h,
              T out, T nrow, int streams, int dilation, double eps) {
    check_half(h); for (const auto* x : std::initializer_list<const T*>{&gated, &tail, &weight, &out, &nrow}) same_half(*x, h);
    check(pss, h, at::kFloat, "PLE PSS"); check(norm, h, at::kFloat, "PLE conv norm"); epsilon(eps);
    TORCH_CHECK(h.dim() == 2 && streams > 0 && h.size(1) % streams == 0 && dilation > 0 && weight.dim() == 2
                && weight.size(0) == h.size(1) && weight.size(1) > 0 && gated.sizes() == h.sizes()
                && out.sizes() == h.sizes() && nrow.sizes() == h.sizes() && norm.numel() == h.size(1)
                && pss.numel() == h.size(0) * streams
                && tail.sizes() == at::IntArrayRef({(weight.size(1) - 1) * dilation, h.size(1)}), "PLE conv geometry");
    c10::cuda::CUDAGuard guard(h.device()); family_ple_conv_cuda(gated, pss, norm, tail, weight, h, out, nrow,
                                                            streams, dilation, float(eps));
}

void add(const T& embedding, const T& streams_in, T out, int streams) {
    check_half(embedding); same_half(streams_in, embedding); same_half(out, embedding);
    TORCH_CHECK(embedding.dim() == 2 && streams > 0 && streams_in.sizes() == at::IntArrayRef({embedding.size(0) * streams, embedding.size(1)})
                && out.sizes() == at::IntArrayRef({embedding.size(0), embedding.size(1) * streams}), "MTP add geometry");
    c10::cuda::CUDAGuard guard(out.device()); family_add_cuda(embedding, streams_in, out, streams);
}

void partial(const T& values, const T& weights, T out, int rows) {
    floating(values, out); check(weights, out, at::kFloat, "route weights"); check(out, values, at::kFloat, "MoE partial");
    TORCH_CHECK(values.dim() == 3 && rows >= 0 && values.size(0) >= rows && weights.dim() == 2
                && weights.size(0) >= rows && weights.size(1) == values.size(1) && out.dim() == 2
                && out.size(0) >= rows && out.size(1) == values.size(2), "MoE partial geometry");
    c10::cuda::CUDAGuard guard(out.device()); family_partial_cuda(values, weights, out, rows);
}

void shift(T old, const T& rows, int keep, int channels) {
    check_half(old);
    TORCH_CHECK(rows.is_cuda() && rows.device() == old.device() && rows.scalar_type() == at::kHalf
                && rows.stride(-1) == 1, "shift input requires same-device FP16 channel-contiguous rows");
    TORCH_CHECK(old.dim() == 3 && rows.dim() == 3 && old.size(0) == rows.size(0) && old.size(2) == channels
                && rows.size(2) >= channels && old.size(1) <= 16 && keep >= 0 && keep <= rows.size(1), "conv shift geometry");
    c10::cuda::CUDAGuard guard(old.device()); family_shift_cuda(old, rows, keep, channels);
}

void prep(const T& projection, const T& position, const T& qw, const T& kw, const T& iw, const T& frequency,
          T queries, T kc, T vc, T index_queries, T index_cache, O ks, O vs, O status, O delta, O valid_rows,
          int qheads, int kvheads, int dim, int index_heads, int index_dim, int kbits, int vbits, int layer,
          double eps, double rope_scale, bool independent_rows) {
    check_half(projection); epsilon(eps);
    TORCH_CHECK(projection.dim() == 2 && qheads > 0 && kvheads > 0 && index_heads > 0
                && dim > 0 && dim <= 256 && dim % 32 == 0 && index_dim > 0 && index_dim <= 256 && index_dim % 32 == 0,
                "attention preparation dimensions");
    TORCH_CHECK(projection.size(1) == 2 * qheads * dim + 2 * kvheads * dim + (index_heads + 1) * index_dim,
                "attention preparation packed projection geometry");
    const int rows = projection.size(0);
    for (const auto* x : std::initializer_list<const T*>{&queries, &index_queries, &index_cache}) same_half(*x, projection);
    for (const auto* x : {&qw, &kw, &iw, &frequency}) check(*x, projection, at::kFloat, "attention norm/frequency");
    TORCH_CHECK(qw.numel() == dim && kw.numel() == dim && iw.numel() == index_dim && frequency.dim() == 1
                && frequency.numel() * 2 <= std::min(dim, index_dim) && frequency.numel() > 0,
                "attention norm/frequency geometry");
    TORCH_CHECK(queries.dim() == 3 && queries.size(0) >= rows && queries.size(1) == qheads && queries.size(2) == dim
                && index_queries.dim() == 3 && index_queries.size(0) >= rows && index_queries.size(1) == index_heads
                && index_queries.size(2) == index_dim && index_cache.dim() == 2 && index_cache.size(1) == index_dim,
                "query/indexer output geometry");
    TORCH_CHECK((kbits == 0 || kbits == 8) && (vbits == 0 || vbits == 8), "FP16 preparation supports raw/int8 pairs");
    check(kc, projection, kbits ? at::kChar : at::kHalf, "key payload");
    check(vc, projection, vbits ? at::kChar : at::kHalf, "value payload");
    TORCH_CHECK(kc.dim() == 3 && kc.size(1) == kvheads && kc.size(2) == dim && vc.sizes() == kc.sizes()
                && index_cache.size(0) == kc.size(0), "attention cache geometry");
    if (kbits) { TORCH_CHECK(ks.has_value(), "key scales missing"); same_half(*ks, projection);
        TORCH_CHECK(ks->numel() == kc.numel() / 32, "key scale geometry"); }
    if (vbits) { TORCH_CHECK(vs.has_value(), "value scales missing"); same_half(*vs, projection);
        TORCH_CHECK(vs->numel() == vc.numel() / 32, "value scale geometry"); }
    TORCH_CHECK(position.scalar_type() == at::kInt || position.scalar_type() == at::kLong, "position dtype");
    check(position, projection, position.scalar_type(), "attention position"); TORCH_CHECK(position.numel() == 1, "position geometry");
    for (const auto* control : {&status, &delta, &valid_rows}) if (*control) {
        check(**control, projection, at::kInt, "attention control");
        TORCH_CHECK((*control)->numel() == (control == &status ? 2 : 1), "attention control geometry");
    }
    TORCH_CHECK(std::isfinite(rope_scale) && float(rope_scale) > 0, "invalid rotary scale");
    c10::cuda::CUDAGuard guard(projection.device());
    family_prep_cuda(projection, position, qw, kw, iw, frequency, queries, kc, vc, index_queries, index_cache,
                     ks, vs, status, delta, valid_rows, qheads, kvheads, dim, index_heads, index_dim, kbits, vbits, layer,
                     float(eps), float(rope_scale), independent_rows);
}

void pool(const T& cache, T pooled, const T& position, const T& weight, const T& frequency,
          O delta, int ratio, int rows, double eps, double rope_scale) {
    check_half(cache); same_half(pooled, cache); epsilon(eps);
    check(weight, cache, at::kFloat, "index norm"); check(frequency, cache, at::kFloat, "index frequencies");
    TORCH_CHECK(cache.dim() == 2 && cache.size(1) > 0 && cache.size(1) <= 256 && cache.size(1) % 32 == 0
                && pooled.dim() == 2 && pooled.size(1) == cache.size(1) && ratio > 0 && rows >= 0
                && pooled.size(0) >= cache.size(0) / ratio && weight.numel() == cache.size(1)
                && frequency.dim() == 1 && frequency.numel() > 0 && frequency.numel() * 2 <= cache.size(1), "QSA pool geometry");
    TORCH_CHECK(position.scalar_type() == at::kInt || position.scalar_type() == at::kLong, "pool position dtype");
    check(position, cache, position.scalar_type(), "pool position"); TORCH_CHECK(position.numel() == 1, "pool position geometry");
    if (delta) { check(*delta, cache, at::kInt, "pool rotary delta"); TORCH_CHECK(delta->numel() == 1, "pool delta geometry"); }
    c10::cuda::CUDAGuard guard(cache.device());
    family_pool_cuda(cache, pooled, position, weight, frequency, delta, ratio, rows, float(eps), float(rope_scale));
}

void qsa_dense(const T& position, T sparse, T counts, int rows) {
    TORCH_CHECK(position.scalar_type() == at::kInt || position.scalar_type() == at::kLong, "QSA position dtype");
    check(position, position, position.scalar_type(), "QSA position");
    check(sparse, position, at::kInt, "QSA sparse flags"); check(counts, position, at::kInt, "QSA counts");
    TORCH_CHECK(position.numel() == 1 && rows >= 0 && sparse.numel() >= rows && counts.numel() >= rows, "QSA dense geometry");
    c10::cuda::CUDAGuard guard(position.device()); family_qsa_dense_cuda(position, sparse, counts, rows);
}

void qsa_rows(const T& query, const T& pooled, const T& position, T sparse, T counts, T ids, T scores,
              int rows, int ratio, int budget, int blocks) {
    check_half(query); same_half(pooled, query);
    TORCH_CHECK(position.scalar_type() == at::kInt || position.scalar_type() == at::kLong, "QSA position dtype");
    check(position, query, position.scalar_type(), "QSA position");
    for (const auto* output : std::initializer_list<const T*>{&sparse, &counts, &ids}) check(*output, query, at::kInt, "QSA integer output");
    check(scores, query, at::kFloat, "QSA scores");
    TORCH_CHECK(query.dim() == 3 && query.size(0) >= rows && pooled.dim() == 2 && pooled.size(1) == query.size(2)
                && query.size(2) > 0 && query.size(2) <= 256 && query.size(2) % 32 == 0 && rows >= 0
                && ratio > 0 && budget > 0 && budget % ratio == 0 && blocks > 0 && blocks <= 4096
                && blocks <= pooled.size(0) && position.numel() == 1 && sparse.numel() >= rows && counts.numel() >= rows
                && ids.dim() == 2 && ids.size(0) >= rows && ids.size(1) >= budget + ratio
                && scores.dim() == 2 && scores.size(0) >= rows && scores.size(1) >= blocks, "QSA selector geometry");
    c10::cuda::CUDAGuard guard(query.device());
    family_qsa_rows_cuda(query, pooled, position, sparse, counts, ids, scores, rows, ratio, budget, blocks);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("writeback", &writeback); m.def("normed", &normed); m.def("act", &act); m.def("mix", &mix);
    m.def("rms", &rms); m.def("gate", &gate); m.def("ple_embed", &ple_embed); m.def("ple_gate", &ple_gate);
    m.def("ple_conv", &ple_conv); m.def("add", &add); m.def("partial", &partial); m.def("shift", &shift);
    m.def("prep", &prep); m.def("pool", &pool); m.def("qsa_dense", &qsa_dense);
    m.def("qsa_rows", &qsa_rows);
}
