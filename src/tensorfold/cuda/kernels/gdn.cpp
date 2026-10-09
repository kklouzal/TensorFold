#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <limits>
#include "tensor_contracts.cuh"

void gdn_tree_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                   const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, int, int, int,
                   at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                   const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&);
void gdn_replay_cuda(const at::Tensor&, int, int, const at::Tensor&, const at::Tensor&, const at::Tensor&, int, int,
                     int, bool);
void gdn_prefill_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                      const at::Tensor&, at::Tensor&, at::Tensor&, int);

static void check_rows(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& g,
                       const at::Tensor& beta) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && g.is_cuda() && beta.is_cuda(), "CUDA tensors only");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && g.is_contiguous() &&
                beta.is_contiguous(), "contiguous tensors only");
    TORCH_CHECK(q.scalar_type() == k.scalar_type() &&
                (q.scalar_type() == at::kBFloat16 || q.scalar_type() == at::kFloat), "q and k: both bf16 or fp32");
    TORCH_CHECK(v.scalar_type() == at::kBFloat16 && g.scalar_type() == at::kFloat &&
                beta.scalar_type() == at::kFloat, "v bf16, g and beta fp32");
    TORCH_CHECK(q.dim() == 3 && k.sizes() == q.sizes() && q.size(2) == 128 && v.dim() == 3 &&
                v.size(0) == q.size(0) && g.dim() == 2 && g.size(0) == q.size(0) && g.size(1) == v.size(1) &&
                beta.sizes() == g.sizes() && q.size(1) >= 1 && v.size(1) >= 1 && v.size(2) >= 1 &&
                v.size(1) % q.size(1) == 0, "invalid GDN shapes");
    constexpr int64_t maximum = std::numeric_limits<int>::max();
    TORCH_CHECK(q.size(0) <= maximum && q.size(1) <= maximum && v.size(1) <= maximum &&
                v.size(2) <= maximum - 31, "GDN dimensions must fit signed-int kernel dimensions and value padding");
    TORCH_CHECK(k.device() == q.device() && v.device() == q.device() && g.device() == q.device() &&
                beta.device() == q.device(), "all GDN tensors must be on the same CUDA device");
    const uintptr_t alignment = q.scalar_type() == at::kFloat ? 16 : 8;
    TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr()) % alignment == 0 &&
                reinterpret_cast<uintptr_t>(k.data_ptr()) % alignment == 0, "GDN q and k require aligned four-element loads");
}

static void check_device(const at::Tensor& reference, const at::Tensor& tensor) {
    TORCH_CHECK(tensor.device() == reference.device(), "all GDN tensors must be on the same CUDA device");
}

static void check_state_alignment(const at::Tensor& tensor) {
    TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0, "GDN states require 16-byte alignment");
}

static void check_index_stride(const at::Tensor& tensor) {
    TORCH_CHECK(tensor.size(0) <= 1 || (tensor.stride(0) >= 0 && tensor.stride(0) <= std::numeric_limits<int>::max()),
                "GDN row/count strides must fit signed-int addressing");
}

static void check_grid(const at::Tensor& reference, int64_t hv, int64_t streams) {
    const auto* properties = at::cuda::getDeviceProperties(reference.get_device());
    TORCH_CHECK(hv >= 1 && hv <= properties->maxGridSize[1] && streams >= 1 &&
                streams <= properties->maxGridSize[2], "GDN heads/streams exceed the CUDA device's grid dimensions");
}

// A window's trees from one state or a table of every stream's; pending rows step each state first, in place.
at::Tensor tree(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& g,
                const at::Tensor& beta, const c10::optional<at::Tensor>& state,
                const c10::optional<at::Tensor>& table, const c10::optional<at::Tensor>& starts,
                const at::Tensor& plan, int64_t slots, int64_t max_rows,
                const c10::optional<std::vector<at::Tensor>>& pending, const c10::optional<at::Tensor>& final) {
    check_rows(q, k, v, g, beta);
    TORCH_CHECK(state.has_value() != table.has_value() && (state.has_value() ? !starts.has_value() : starts.has_value()),
                "pass one state, or a table and starts");
    TORCH_CHECK(plan.is_cuda() && plan.is_contiguous() && plan.scalar_type() == at::kInt &&
                plan.dim() == 2 && plan.size(0) == q.size(0) && plan.size(1) == 3, "plan: (W, 3) int32 on the device");
    check_device(q, plan);
    TORCH_CHECK(slots >= 0 && slots <= 32 && max_rows >= 1 && max_rows <= std::numeric_limits<int>::max() &&
                (slots == 0 || max_rows <= 1024),
                "slots in 0..32; trees of at most 1024 rows a stream (chains any length)");
    at::Tensor s, t, st;
    int streams = 1;
    if (state.has_value()) {
        s = *state;
        TORCH_CHECK(s.is_cuda() && s.is_contiguous() && s.scalar_type() == at::kFloat && s.dim() == 3 &&
                    s.size(0) == v.size(1) && s.size(1) == v.size(2) && s.size(2) == 128, "invalid state");
        check_device(q, s);
        check_state_alignment(s);
        TORCH_CHECK(max_rows >= q.size(0), "max_rows covers the window");
    } else {
        TORCH_CHECK(table.has_value() && starts.has_value(), "a state, or a table and starts");
        t = *table;
        st = *starts;
        TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == at::kLong && t.dim() == 1,
                    "table: int64 device pointers");
        TORCH_CHECK(t.numel() >= 1 && t.numel() < std::numeric_limits<int>::max(), "a nonempty pointer table is required");
        streams = static_cast<int>(t.numel());
        TORCH_CHECK(st.is_cuda() && st.is_contiguous() && st.scalar_type() == at::kInt &&
                    st.numel() == streams + 1, "starts: streams + 1 int32 offsets");
        check_device(q, t);
        check_device(q, st);
    }
    check_grid(q, v.size(1), streams);
    at::Tensor pk, pv, pg, pb, prows, pcounts;
    if (pending.has_value()) {
        const auto& p = *pending;
        TORCH_CHECK(p.size() == 6, "pending: k, v, g, beta, rows (streams, P), counts (streams,)");
        pk = p[0]; pv = p[1]; pg = p[2]; pb = p[3]; prows = p[4]; pcounts = p[5];
        check_rows(pk, pk, pv, pg, pb);
        TORCH_CHECK(pk.scalar_type() == q.scalar_type() && pk.size(1) == q.size(1) && pv.size(1) == v.size(1) &&
                    pv.size(2) == v.size(2), "pending rows match the window's heads");
        TORCH_CHECK(prows.is_cuda() && prows.scalar_type() == at::kInt && prows.dim() == 2 &&
                    prows.size(0) == streams && prows.stride(1) == 1, "pending rows: (streams, P) int32");
        TORCH_CHECK(pcounts.is_cuda() && pcounts.scalar_type() == at::kInt && pcounts.dim() == 1 &&
                    pcounts.numel() == streams, "pending counts: (streams,) int32");
        for (const auto* tensor : {&pk, &pv, &pg, &pb, &prows, &pcounts}) check_device(q, *tensor);
        check_index_stride(prows);
        check_index_stride(pcounts);
        if (s.defined())
            tensorfold::check_disjoint(s, {&q, &k, &v, &g, &beta, &plan, &pk, &pv, &pg, &pb, &prows, &pcounts},
                                      "pending state writes must not overlap row or schedule inputs");
    }
    at::Tensor fstate, ftable;
    if (final.has_value()) {
        TORCH_CHECK(slots == 0, "a final state is only defined for chain windows");
        if (state.has_value()) {
            fstate = *final;
            TORCH_CHECK(fstate.is_cuda() && fstate.is_contiguous() && fstate.scalar_type() == at::kFloat &&
                        fstate.sizes() == s.sizes(), "final: a state-shaped fp32 tensor");
            check_device(q, fstate);
            check_state_alignment(fstate);
            tensorfold::check_disjoint(fstate, {&q, &k, &v, &g, &beta, &s, &plan, &pk, &pv, &pg, &pb, &prows, &pcounts},
                                      "final state writes must not overlap inputs");
        } else {
            ftable = *final;
            TORCH_CHECK(ftable.is_cuda() && ftable.is_contiguous() && ftable.scalar_type() == at::kLong &&
                        ftable.numel() == streams, "final: one int64 device pointer a stream");
            check_device(q, ftable);
        }
    }
    c10::cuda::CUDAGuard guard(q.device());
    auto y = at::empty({q.size(0), v.size(1), v.size(2)}, v.options());
    gdn_tree_cuda(q, k, v, g, beta, s, t, st, plan, static_cast<int>(slots), streams, static_cast<int>(max_rows), y,
                  pk, pv, pg, pb, prows, pcounts, fstate, ftable);
    return y;
}

// Commit: states after the accepted rows, new or in place; the table holds k, v, g, beta a layer, then the states.
at::Tensor replay(const at::Tensor& table, int64_t layers, int64_t streams, const at::Tensor& rows,
                  const at::Tensor& counts, int64_t hk, int64_t hv, int64_t dv, bool fp32_keys, bool in_place) {
    constexpr int64_t maximum = std::numeric_limits<int>::max();
    TORCH_CHECK(layers >= 1 && layers <= maximum && streams >= 1 && streams <= maximum &&
                hk >= 1 && hk <= maximum && hv >= 1 && hv <= maximum && dv >= 1 && dv <= maximum - 31 &&
                hv % hk == 0, "positive replay dimensions, whole value/key head groups and signed-int value padding required");
    TORCH_CHECK(layers <= maximum / streams, "replay layer/stream grid exceeds signed-int dimensions");
    TORCH_CHECK(layers <= std::numeric_limits<int64_t>::max() / (4 + streams), "replay pointer cardinality overflows");
    TORCH_CHECK(table.is_cuda() && table.is_contiguous() && table.scalar_type() == at::kLong &&
                table.numel() == 4 * layers + layers * streams, "table: 4 * layers + layers * streams pointers");
    TORCH_CHECK(rows.is_cuda() && rows.scalar_type() == at::kInt && rows.dim() == 2 && rows.size(0) == streams &&
                rows.stride(1) == 1, "rows: (streams, P) int32, each stream's rows contiguous");
    TORCH_CHECK(counts.is_cuda() && counts.scalar_type() == at::kInt && counts.dim() == 1 &&
                counts.numel() == streams, "counts: (streams,) int32");
    check_device(table, rows);
    check_device(table, counts);
    check_index_stride(rows);
    check_index_stride(counts);
    check_grid(table, hv, layers * streams);
    c10::cuda::CUDAGuard guard(table.device());
    at::Tensor out;
    if (!in_place) out = at::empty({streams, layers, hv, dv, 128}, table.options().dtype(at::kFloat));
    gdn_replay_cuda(table, static_cast<int>(layers), static_cast<int>(streams), rows, counts, out,
                    static_cast<int>(hk), static_cast<int>(hv), static_cast<int>(dv), fp32_keys);
    return in_place ? at::empty({0}, table.options().dtype(at::kFloat)) : out;
}

// A prompt chunk as one chain from ``state``: outputs (W, Hv, 128) bf16, the last row's state into ``last``.
at::Tensor prefill(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& g,
                   const at::Tensor& beta, const at::Tensor& state, at::Tensor& last) {
    check_rows(q, k, v, g, beta);
    TORCH_CHECK(v.size(2) == 128, "value heads of 128");
    TORCH_CHECK(state.is_cuda() && state.is_contiguous() && state.scalar_type() == at::kFloat && state.dim() == 3 &&
                state.size(0) == v.size(1) && state.size(1) == 128 && state.size(2) == 128, "invalid state");
    TORCH_CHECK(last.is_cuda() && last.is_contiguous() && last.sizes() == state.sizes() &&
                last.scalar_type() == at::kFloat, "last: a state-shaped fp32 tensor");
    check_device(q, state);
    check_device(q, last);
    check_state_alignment(state);
    check_state_alignment(last);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(v.data_ptr()) % 16 == 0, "prefill values require 16-byte alignment");
    if (q.size(0) > 0)
        tensorfold::check_disjoint(last, {&q, &k, &v, &g, &beta, &state}, "prefill last writes must not overlap inputs");
    c10::cuda::CUDAGuard guard(q.device());
    auto y = at::empty({q.size(0), v.size(1), v.size(2)}, v.options());
    const auto* properties = at::cuda::getDeviceProperties(q.get_device());
    TORCH_CHECK(v.size(1) <= properties->maxGridSize[0], "prefill heads exceed the CUDA device's grid dimensions");
    const int sms = properties->multiProcessorCount;
    if (q.size(0) > 0) gdn_prefill_cuda(q, k, v, g, beta, state, last, y, sms);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tree", &tree);
    m.def("replay", &replay);
    m.def("prefill", &prefill);
}
