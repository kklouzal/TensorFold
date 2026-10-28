// SM70 specialization of the fork's GDN chain/replay ABI, with complete buffer validation.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <limits>
#include <optional>
#include <array>
#include <algorithm>
#include <cstdint>
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

namespace {
struct BatchSpan { uintptr_t first, last; };
BatchSpan batch_span(const at::Tensor& value) {
    uintptr_t first = reinterpret_cast<uintptr_t>(value.data_ptr());
    TORCH_CHECK(value.numel() <= std::numeric_limits<int64_t>::max() / value.element_size(), "GDN span bytes overflow");
    auto bytes = value.numel() * value.element_size();
    TORCH_CHECK(uint64_t(bytes) <= std::numeric_limits<uintptr_t>::max() - first, "GDN address span overflow");
    return {first, first + uintptr_t(bytes)};
}
}

// Only bounded stack arrays and existing Tensor/Python references are used.
// No new host/device tensor or per-State launch argument vector is allocated.
void fill_batch(at::Tensor host, at::Tensor frame, pybind11::sequence segments, int64_t layers) {
    namespace py = pybind11;
    TORCH_CHECK(host.device().is_cpu() && host.scalar_type() == at::kLong && host.is_contiguous()
        && host.sizes() == at::IntArrayRef({layers, 24, 10}) && layers > 0 && layers <= 48,
        "GDN host catalogue envelope");
    TORCH_CHECK(frame.device().is_cpu() && frame.scalar_type() == at::kLong && frame.is_contiguous()
        && frame.sizes() == at::IntArrayRef({24, 4}), "GDN frame descriptor envelope");
    TORCH_CHECK(py::len(segments) >= 2 && py::len(segments) <= 24, "GDN batch State count");
    std::array<BatchSpan, 24 * (48 * 4 + 2)> allocations;
    std::array<BatchSpan, 24 * 48> controls;
    size_t used = 0, ncontrols = 0;
    int64_t cursor = 0;
    auto* table = host.data_ptr<int64_t>();
    const auto* mapping = frame.data_ptr<int64_t>();
    for (size_t index = 0; index < py::len(segments); ++index) {
        const int64_t slot = mapping[index * 4], changed = mapping[index * 4 + 3];
        TORCH_CHECK(slot >= 0 && slot < 24 && (changed == 0 || changed == 1), "GDN catalogue slot/version");
        auto segment = py::reinterpret_borrow<py::sequence>(segments[index]);
        TORCH_CHECK(py::len(segment) == 3, "GDN segment tuple geometry");
        auto state = py::reinterpret_borrow<py::object>(segment[0]);
        const int64_t first = py::cast<int64_t>(segment[1]), last = py::cast<int64_t>(segment[2]);
        TORCH_CHECK(first == cursor && last > first && last - first <= 3 && last <= 72
            && mapping[index * 4 + 1] == first && mapping[index * 4 + 2] == last - first,
            "GDN requires ordered independent State spans1..3 within72 rows");
        cursor = last;
        auto parity = py::reinterpret_borrow<py::sequence>(state.attr("cur"));
        TORCH_CHECK(py::len(parity) == layers, "GDN parity length");
        if (changed) {
            auto conv = py::cast<at::Tensor>(state.attr("conv"));
            auto rec = py::cast<at::Tensor>(state.attr("rec"));
            const auto device = rec.device();
            check(conv, at::kHalf, device, "batch convolution");
            check(rec, at::kFloat, device, "batch recurrence");
            TORCH_CHECK(conv.sizes() == at::IntArrayRef({layers, 3, 1280})
                && rec.sizes() == at::IntArrayRef({2, layers, 6, 128, 128}), "GDN batch State geometry");
            auto scratch = py::reinterpret_borrow<py::sequence>(state.attr("scratch"));
            TORCH_CHECK(py::len(scratch) == layers, "GDN scratch length");
            for (int64_t layer = 0; layer < layers; ++layer) {
                auto saved = py::reinterpret_borrow<py::object>(scratch[layer]);
                auto key = py::cast<at::Tensor>(saved.attr("k"));
                auto value = py::cast<at::Tensor>(saved.attr("v"));
                auto decay = py::cast<at::Tensor>(saved.attr("g"));
                auto beta = py::cast<at::Tensor>(saved.attr("b"));
                saved_check(key, value, decay, beta, 3, 6, device);
                int64_t* row = table + (layer * 24 + slot) * 10;
                row[0] = reinterpret_cast<int64_t>(conv.data_ptr()) + layer * conv.stride(0) * 2;
                row[1] = reinterpret_cast<int64_t>(rec.data_ptr()) + layer * rec.stride(1) * 4;
                row[2] = row[1] + rec.stride(0) * 4;
                row[3] = reinterpret_cast<int64_t>(key.data_ptr()); row[4] = reinterpret_cast<int64_t>(value.data_ptr());
                row[5] = reinterpret_cast<int64_t>(decay.data_ptr()); row[6] = reinterpret_cast<int64_t>(beta.data_ptr());
                row[7] = 0;
                if (py::hasattr(saved, "keep_rows") && !saved.attr("keep_rows").is_none()) {
                    auto live = py::cast<at::Tensor>(saved.attr("keep_rows"));
                    check(live, at::kInt, device, "batch live rows");
                    TORCH_CHECK(live.numel() == 1, "batch live rows requires one integer");
                    row[7] = reinterpret_cast<int64_t>(live.data_ptr());
                }
            }
        }
        for (int64_t layer = 0; layer < layers; ++layer) {
            const int current = py::cast<int>(parity[layer]);
            TORCH_CHECK(current == 0 || current == 1, "GDN parity must be0 or1");
            int64_t* row = table + (layer * 24 + slot) * 10;
            const int64_t base = std::min(row[1], row[2]), bank_bytes = layers * 6 * 128 * 128 * 4;
            row[1] = base + current * bank_bytes; row[2] = base + (1 - current) * bank_bytes;
            row[8] = first; row[9] = last - first;
            if (layer == 0) {
                const uintptr_t conv = uintptr_t(row[0]);
                allocations[used++] = {conv, conv + uintptr_t(layers * 3 * 1280 * 2)};
                const uintptr_t rec = uintptr_t(base);
                allocations[used++] = {rec, rec + uintptr_t(2 * bank_bytes)};
            }
            const int64_t bytes[] = {3 * 2 * 128 * 4, 3 * 6 * 128 * 2, 3 * 6 * 4, 3 * 6 * 4};
            for (int item = 0; item < 4; ++item) {
                const uintptr_t pointer = uintptr_t(row[3 + item]);
                allocations[used++] = {pointer, pointer + uintptr_t(bytes[item])};
            }
            if (row[7]) controls[ncontrols++] = {uintptr_t(row[7]), uintptr_t(row[7]) + 4};
        }
    }
    std::sort(allocations.begin(), allocations.begin() + used,
        [](BatchSpan a, BatchSpan b) { return a.first < b.first; });
    for (size_t i = 1; i < used; ++i)
        TORCH_CHECK(allocations[i - 1].last <= allocations[i].first, "GDN States/scratch must own disjoint storage");
    for (size_t i = 0; i < ncontrols; ++i) {
        auto found = std::lower_bound(allocations.begin(), allocations.begin() + used, controls[i].last,
            [](BatchSpan a, uintptr_t end) { return a.first < end; });
        if (found != allocations.begin()) { --found;
            TORCH_CHECK(found->last <= controls[i].first, "GDN live-row control overlaps mutable State storage");
        }
    }
}

// The pointer table is opaque owned storage, populated only by GDNStateBatch.
// Pointee dtype/ranges/generations/disjointness are validated by that frame
// owner before upload; the native boundary validates the fixed launch envelope.
void sm70_gdn_prepare_batch_cuda();
void sm70_gdn_batch_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, double, at::Tensor, at::Tensor,
    const at::Tensor&, const at::Tensor&, int64_t, int64_t, bool);

void batch(const at::Tensor& projection, const at::Tensor& weight, const at::Tensor& decay,
    const at::Tensor& bias, const at::Tensor& norm, double eps, at::Tensor output,
    at::Tensor sums, const at::Tensor& table, const at::Tensor& frame, int64_t layer, int64_t states, bool negative_decay) {
    const auto device = projection.device();
    for (const auto* input : std::initializer_list<const at::Tensor*>{&projection, &weight, &norm, &output})
        check(*input, at::kHalf, device, "batch half operand");
    for (const auto* input : std::initializer_list<const at::Tensor*>{&decay, &bias, &sums})
        check(*input, at::kFloat, device, "batch float operand");
    TORCH_CHECK(table.device().is_cpu() && table.scalar_type() == at::kLong && table.is_contiguous(),
        "batch catalogue must be owned contiguous CPU int64");
    const auto* prop = at::cuda::getDeviceProperties(projection.get_device());
    TORCH_CHECK(prop->major == 7 && prop->minor == 0, "batch GDN requires SM70");
    TORCH_CHECK(projection.dim() == 2 && projection.size(1) == 2060 && projection.size(0) > 0,
        "batch GDN projection geometry");
    TORCH_CHECK(weight.sizes() == at::IntArrayRef({1280, 4}) && norm.sizes() == at::IntArrayRef({128})
        && decay.sizes() == at::IntArrayRef({6}) && bias.sizes() == at::IntArrayRef({6}), "batch GDN weight geometry");
    TORCH_CHECK(output.sizes() == at::IntArrayRef({projection.size(0), 768})
        && sums.sizes() == at::IntArrayRef({projection.size(0), 24}), "batch GDN output geometry");
    TORCH_CHECK(table.dim() == 3 && table.size(1) == 24 && table.size(2) == 10
        && layer >= 0 && layer < table.size(0) && states >= 2 && states <= 24,
        "batch GDN descriptor envelope");
    TORCH_CHECK(std::isfinite(eps) && eps > 0 && std::isfinite(float(eps)) && float(eps) > 0,
        "batch GDN epsilon must be positive finite FP32");
    tensorfold::check_disjoint(output, {&projection, &weight, &decay, &bias, &norm, &sums, &table},
        "batch output must be disjoint");
    tensorfold::check_disjoint(sums, {&projection, &weight, &decay, &bias, &norm, &table},
        "batch sums must be disjoint");
    TORCH_CHECK(frame.device().is_cpu() && frame.scalar_type() == at::kLong && frame.is_contiguous()
        && frame.sizes() == at::IntArrayRef({24, 4}), "GDN batch frame geometry");
    const auto* mapping = frame.data_ptr<int64_t>();
    const auto* metadata = table.data_ptr<int64_t>() + layer * 24 * 10;
    int64_t cursor = 0;
    for (int64_t slot = 0; slot < states; ++slot) {
        TORCH_CHECK(mapping[slot * 4] >= 0 && mapping[slot * 4] < 24, "GDN frame catalogue slot");
        const int64_t* row = metadata + mapping[slot * 4] * 10;
        TORCH_CHECK(row[8] == cursor && row[9] >= 1 && row[9] <= 3
            && row[9] <= projection.size(0) - cursor, "GDN batch frame span changed");
        cursor += row[9];
        const int64_t bytes[] = {3 * 1280 * 2, 6 * 128 * 128 * 4, 6 * 128 * 128 * 4,
            row[9] * 2 * 128 * 4, row[9] * 6 * 128 * 2, row[9] * 6 * 4, row[9] * 6 * 4};
        for (int item = 0; item < 7; ++item) {
            const uintptr_t address = uintptr_t(row[item]);
            TORCH_CHECK(address && uint64_t(bytes[item]) <= std::numeric_limits<uintptr_t>::max() - address,
                "invalid GDN batch address span");
            for (const auto* common : std::initializer_list<const at::Tensor*>{&projection, &weight, &decay, &bias, &norm, &output, &sums, &table}) {
                BatchSpan span = batch_span(*common);
                TORCH_CHECK(address + bytes[item] <= span.first || span.last <= address,
                    "GDN State/scratch must be disjoint from frame operands");
            }
        }
    }
    TORCH_CHECK(cursor == projection.size(0), "GDN batch must cover the complete projection frame");
    c10::cuda::CUDAGuard guard(device);
    sm70_gdn_batch_cuda(projection, weight, decay, bias, norm, eps, output, sums, table, frame,
        layer, states, negative_decay);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("chain", &chain);
    module.def("batch", &batch);
    module.def("fill_batch", &fill_batch);
    module.def("prepare_batch", &sm70_gdn_prepare_batch_cuda);
    module.def("replay", &replay);
}
