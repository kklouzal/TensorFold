#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <limits>

void expert_a4_quant_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, at::Tensor&, bool, bool);
void expert_a4_gemm_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                         const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&,
                         int64_t, int64_t, int64_t, bool, bool);

void expert_a4_quant_plan_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, at::Tensor&, at::Tensor&, int64_t, bool, int64_t);
void expert_a4_fused_up_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&,
    at::Tensor&, int64_t, int64_t, int64_t, int64_t, bool, bool, int64_t, int64_t);
void expert_a4_gemma_fused_up_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&,
    at::Tensor&, int64_t, int64_t, int64_t, bool, int64_t, int64_t);
void expert_a4_down_plan_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&,
    int64_t, int64_t, int64_t, int64_t, bool, int64_t);

static void tensor(const at::Tensor& value, const at::Tensor& on, at::ScalarType type, int rank,
                   const char* name) {
    TORCH_CHECK(value.is_cuda() && value.device() == on.device() && value.scalar_type() == type &&
                value.dim() == rank && value.is_contiguous(), name, ": contiguous CUDA tensor on the bank device");
}

// Positive tensor dimensions are established before this helper. Check each
// affine-extent term before multiplying: zero-storage tensors may legally carry
// arbitrary int64 strides, and their metadata must never cause host overflow.
static int64_t extend_span(int64_t span, int64_t count, int64_t stride, int64_t limit, const char* name) {
    TORCH_CHECK(count > 0 && stride >= 0 && span <= limit, name, ": invalid positive extent");
    TORCH_CHECK(count == 1 || stride <= (limit - span) / (count - 1), name, ": extent exceeds byte index range");
    return span + (count - 1) * stride;
}

// Native blob fields have a contiguous per-expert payload and an explicit
// outer expert stride. Admit each affine span before multiplication, including
// singleton/empty edge cases, rather than assuming a fully contiguous bank.
static void expert_field(const at::Tensor& value, const at::Tensor& on, at::ScalarType type, int rank,
                         const char* name) {
    TORCH_CHECK(value.is_cuda() && value.device() == on.device() && value.scalar_type() == type &&
                value.dim() == rank && value.stride(rank - 1) == 1, name, ": native expert field");
    const auto limit = std::numeric_limits<int64_t>::max() / value.element_size();
    int64_t span = value.size(rank - 1);
    TORCH_CHECK(span > 0, name, ": positive inner field dimension required");
    for (int axis = rank - 2; axis >= 0; --axis) {
        TORCH_CHECK(value.size(axis) > 0 && value.stride(axis) >= span &&
                    (axis == 0 || value.stride(axis) == span), name, ": complete per-expert field");
        span = extend_span(span, value.size(axis), value.stride(axis), limit, name);
    }
}

// Host-window plan rows are (source-row, destination-slot, cache-cell), constructed from admitted CPU picks by the
// owner (or, for a proven shared-input bank, the device identity row plan (row, 0, 0)); device plans pass their
// members, items and counts instead. Kernels guard every indirect access. At most 65535 planned rows (grid y).
static void quantize_impl(const at::Tensor& x, const at::Tensor& plan, const at::Tensor& input,
               at::Tensor codes, at::Tensor scales, bool paired, bool direct = false,
               const at::Tensor& items = {}, const at::Tensor& counts = {}, int64_t slots = 0, int64_t tile = 16) {
    const bool grouped = counts.defined();
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.stride(1) == 1 &&
                x.stride(0) >= x.size(1) && x.size(0) > 0 && x.size(0) < INT32_MAX && x.size(1) > 0 &&
                x.size(1) <= INT32_MAX && x.size(1) % 64 == 0, "x: BF16 rows in complete64-input tiles");
    extend_span(x.size(1), x.size(0), x.stride(0),
                std::numeric_limits<int64_t>::max() / x.element_size(), "nativeA4 input");
    tensor(plan, x, at::kInt, grouped ? 1 : 2, "plan");
    expert_field(input, x, at::kFloat, 2, "input calibration");
    tensor(codes, x, at::kByte, 3, "codes");
    tensor(scales, x, at::kByte, 3, "scales");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(codes.data_ptr()) % 8 == 0, "codes require8-byte alignment");
    const auto p = codes.size(1), k = x.size(1), projections = input.size(1);
    TORCH_CHECK(p > 0 && p <= INT32_MAX && plan.size(0) == p && (grouped || (p <= 65535 && plan.size(1) == 3)) &&
                input.size(0) > 0 && input.size(0) <= INT32_MAX && (projections == 1 || projections == 2),
                "plan/calibrations require at most65535 planned rows (grid y), one/two projections");
    TORCH_CHECK(codes.size(0) == projections && codes.size(1) == p && codes.size(2) == k / 2 &&
                scales.size(0) == projections && scales.size(1) == p && scales.size(2) == k / 16,
                "quantization buffers must match projection/pair/input dimensions");
    c10::cuda::CUDAGuard guard(x.device());
    TORCH_CHECK(!paired || x.size(0) == p, "pair-indexed inputs must hold exactly every planned pair");
    if (grouped) {
        tensor(items, x, at::kInt, 2, "items");
        tensor(counts, x, at::kInt, 1, "counts");
        TORCH_CHECK(!paired && slots > 0 && slots <= 1024 && items.size(0) > 0 && items.size(0) <= 65535 &&
                    items.size(1) == 3 && counts.numel() == 2 && x.size(0) * slots == p && (tile == 16 || tile == 64),
                    "device plan: row inputs, bounded 16/64-pair items, counts and complete row slots");
        expert_a4_quant_plan_cuda(x, items, counts, plan, input, codes, scales, slots, direct, tile);
    } else expert_a4_quant_cuda(x, plan, input, codes, scales, paired, direct);
}

// Host windows: codes/scales are pair-major; items (cell, first, count <= 16) group same-expert pairs of at most 128;
// decode K slices match checkpoint.matmul's shape-only policy exactly.
void gemm(const at::Tensor& codes, const at::Tensor& scales, const at::Tensor& blob,
          const at::Tensor& weight, const at::Tensor& input, const at::Tensor& items,
          const at::Tensor& plan, at::Tensor out, int64_t n, int64_t k, int64_t sk, bool down, bool direct) {
    tensor(codes, blob, at::kByte, 3, "codes");
    tensor(scales, blob, at::kByte, 3, "scales");
    expert_field(blob, blob, at::kByte, 3, "projection blob");
    expert_field(weight, blob, at::kFloat, 2, "weight calibration");
    expert_field(input, blob, at::kFloat, 2, "input calibration");
    tensor(items, blob, at::kInt, 2, "expert items");
    tensor(plan, blob, at::kInt, 2, "pair plan");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(codes.data_ptr()) % 4 == 0 &&
                reinterpret_cast<uintptr_t>(scales.data_ptr()) % 4 == 0 &&
                reinterpret_cast<uintptr_t>(blob.data_ptr()) % 8 == 0, "nativeA4 fragment load alignment");
    const auto projections = blob.size(1), p = codes.size(1), cells = blob.size(0);
    TORCH_CHECK(n > 0 && n < INT32_MAX && n % 64 == 0 && k > 0 && k < INT32_MAX && k % 64 == 0 &&
                sk > 0 && sk <= 8 && (k / 64) % sk == 0, "nativeA4 dimensions/slices must form complete tiles");
    TORCH_CHECK(p > 0 && p <= 128 && plan.size(0) == p && plan.size(1) == 3 && cells > 0 && cells <= INT32_MAX &&
                items.size(0) > 0 && items.size(0) <= p && items.size(1) == 3 &&
                projections == (down ? 1 : 2), "nativeA4 bounded plan/projection geometry: pairs=", p,
                " items=", items.size(0), " cells=", cells, " projections=", projections,
                " down=", static_cast<int>(down));
    // grid.x = ceil(items * n/8 / 4) converts to unsigned int: items <= 128 and n < 2^31 alone admit 2^33.
    TORCH_CHECK(items.size(0) * (n / 8) <= static_cast<int64_t>(INT32_MAX) * 4,
                "nativeA4 launch exceeds CUDA grid index range");
    TORCH_CHECK(blob.size(2) == (n / 16) * k * 9 && weight.size(0) == cells && weight.size(1) == projections &&
                input.sizes() == weight.sizes(), "nativeA4 blob/calibration geometry");
    TORCH_CHECK(codes.size(0) == projections && codes.size(2) == k / 2 &&
                scales.size(0) == projections && scales.size(1) == p && scales.size(2) == k / 16,
                "nativeA4 quantized pair dimensions");
    TORCH_CHECK(out.is_cuda() && out.device() == blob.device() && out.dim() == 3 && out.size(2) == n &&
                out.size(0) > 0 && out.size(0) <= INT32_MAX && out.size(1) > 0 && out.size(1) <= INT32_MAX &&
                out.stride(2) == 1 && out.stride(1) >= n && out.stride(0) >= 0 &&
                out.stride(0) / out.size(1) >= out.stride(1) &&
                (out.scalar_type() == at::kBFloat16 || (down && out.scalar_type() == at::kFloat)),
                "nativeA4 output: positive separate slots, BF16 or FP32 decode-down");
    const auto limit = std::numeric_limits<int64_t>::max() / out.element_size();
    const auto row_span = extend_span(n, out.size(1), out.stride(1), limit, "nativeA4 output slots");
    extend_span(row_span, out.size(0), out.stride(0), limit, "nativeA4 output rows");
    TORCH_CHECK(down || (out.is_contiguous() && out.size(0) == projections && out.size(1) == p),
                "gate/up output: (2,pairs,width) contiguous BF16");
    c10::cuda::CUDAGuard guard(blob.device());
    expert_a4_gemm_cuda(codes, scales, blob, weight, input, items, plan, out, n, k, sk, down, direct);
}

// Keep legacy tensor arguments required at the Python boundary. Undefined
// implementation-only plan tensors stay in C++; they must never become Python
// None defaults because the pinned Tensor caster does not accept that value.
void quantize(const at::Tensor& x, const at::Tensor& plan, const at::Tensor& input,
    at::Tensor codes, at::Tensor scales, bool paired, bool direct) {
    quantize_impl(x, plan, input, codes, scales, paired, direct);
}

// Gate/up activation codes of a device plan (per pair and projection); proven shared-input banks quantize rows with
// ``quantize`` and the identity row plan instead.
void quantize_plan(const at::Tensor& x, const at::Tensor& items, const at::Tensor& counts,
    const at::Tensor& members, const at::Tensor& input, at::Tensor codes, at::Tensor scales,
    int64_t slots, bool direct, int64_t tile) {
    quantize_impl(x, members, input, codes, scales, false, direct, items, counts, slots, tile);
}

static void plan_common(const at::Tensor& blob, const at::Tensor& items, const at::Tensor& counts,
                        const at::Tensor& members, int64_t n, int64_t k, int64_t sk, int64_t slots, int64_t tile) {
    expert_field(blob, blob, at::kByte, 3, "projection blob");
    tensor(items, blob, at::kInt, 2, "expert items");
    tensor(counts, blob, at::kInt, 1, "counts");
    tensor(members, blob, at::kInt, 1, "pair members");
    const auto p = members.size(0), cells = blob.size(0);
    TORCH_CHECK(n > 0 && n < INT32_MAX && n % 64 == 0 && k > 0 && k < INT32_MAX && k % 64 == 0 && sk > 0 && sk <= 8 &&
                (k / 64) % sk == 0, "native A4 device plan: complete tiles and shape-only slices");
    TORCH_CHECK((tile == 16 || tile == 64) && slots > 0 && slots <= 1024 && p > 0 && p <= INT32_MAX &&
                p % slots == 0 && cells > 0 && cells <= INT32_MAX && items.size(0) > 0 && items.size(0) <= p &&
                items.size(1) == 3 && counts.numel() == 2 && blob.size(2) == (n / 16) * k * 9,
                "native A4 device plan: 16/64-pair items over complete row slots and complete expert payloads");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(blob.data_ptr()) % 16 == 0 && blob.stride(0) % 16 == 0,
                "native A4 device plan: 16-byte aligned staged expert payloads");
}

// Gate/up GEMMs of a device plan, the BF16 activation/product and the down FP4 quantization in one launch.
// Codes/scales are the gate/up activation codes (shared rows when ``common``); the down codes/scales are pair-major.
// Per element: the same chains, BF16 boundaries and block quantizer as separate GEMM, activation and quantization.
static void check_fused_up(const at::Tensor& codes, const at::Tensor& scales, const at::Tensor& blob, const at::Tensor& weight,
    const at::Tensor& input, const at::Tensor& down_input, const at::Tensor& items, const at::Tensor& counts,
    const at::Tensor& members, const at::Tensor& down_codes, const at::Tensor& down_scales, int64_t n, int64_t k, int64_t sk,
    int64_t slots, bool common, int64_t tile, int64_t act) {
    plan_common(blob, items, counts, members, n, k, sk, slots, tile);
    tensor(codes, blob, at::kByte, 3, "codes");
    tensor(scales, blob, at::kByte, 3, "scales");
    tensor(down_codes, blob, at::kByte, 3, "down codes");
    tensor(down_scales, blob, at::kByte, 3, "down scales");
    expert_field(weight, blob, at::kFloat, 2, "weight calibration");
    expert_field(input, blob, at::kFloat, 2, "input calibration");
    expert_field(down_input, blob, at::kFloat, 2, "down input calibration");
    const auto p = members.size(0), cells = blob.size(0);
    TORCH_CHECK(act == 0 || act == 1, "fused native A4: SiLU (0) or tanh GELU (1)");
    TORCH_CHECK(blob.size(1) == 2 && weight.size(0) == cells && weight.size(1) == 2 && input.sizes() == weight.sizes() &&
                down_input.size(0) == cells && down_input.size(1) == 1, "fused native A4: gate/up blob and calibrations");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(codes.data_ptr()) % 4 == 0 &&
                reinterpret_cast<uintptr_t>(scales.data_ptr()) % 4 == 0, "fused native A4: fragment alignment");
    const int64_t code_count = common ? p / slots : p, code_projections = common ? 1 : 2;
    TORCH_CHECK(codes.size(0) == code_projections && codes.size(1) == code_count && codes.size(2) == k / 2 &&
                scales.size(0) == code_projections && scales.size(1) == code_count && scales.size(2) == k / 16,
                "fused native A4: activation code dimensions");
    TORCH_CHECK(down_codes.size(0) == 1 && down_codes.size(1) == p && down_codes.size(2) == n / 2 &&
                down_scales.size(0) == 1 && down_scales.size(1) == p && down_scales.size(2) == n / 16,
                "fused native A4: pair-major down codes/scales");
}

void fused_up(const at::Tensor& codes, const at::Tensor& scales, const at::Tensor& blob, const at::Tensor& weight,
    const at::Tensor& input, const at::Tensor& down_input, const at::Tensor& items, const at::Tensor& counts,
    const at::Tensor& members, at::Tensor down_codes, at::Tensor down_scales, int64_t n, int64_t k, int64_t sk,
    int64_t slots, bool common, bool direct, int64_t tile, int64_t act) {
    check_fused_up(codes, scales, blob, weight, input, down_input, items, counts, members,
                   down_codes, down_scales, n, k, sk, slots, common, tile, act);
    c10::cuda::CUDAGuard guard(blob.device());
    expert_a4_fused_up_cuda(codes, scales, blob, weight, input, down_input, items, counts, members, down_codes,
                            down_scales, n, k, sk, slots, common, direct, tile, act);
}

// Fixed semantics, bounded launch configuration. Callers retain the same
// owner-prepared, disjoint bank/plan/scratch fields required by fused_up.
// Production callers retain fused_up; this private entry selects no policy.
void gemma_fused_up(const at::Tensor& codes, const at::Tensor& scales, const at::Tensor& blob,
    const at::Tensor& weight, const at::Tensor& input, const at::Tensor& down_input, const at::Tensor& items,
    const at::Tensor& counts, const at::Tensor& members, at::Tensor down_codes, at::Tensor down_scales,
    int64_t n, int64_t k, int64_t slots, bool common, int64_t warps, int64_t stages) {
    TORCH_CHECK((warps == 2 || warps == 4) && stages >= 2 && stages <= 4,
                "private Gemma-up requires warps2/4 and stages2/3/4");
    check_fused_up(codes, scales, blob, weight, input, down_input, items, counts, members,
                   down_codes, down_scales, n, k, 1, slots, common, 16, 1);
    c10::cuda::CUDAGuard guard(blob.device());
    expert_a4_gemma_fused_up_cuda(codes, scales, blob, weight, input, down_input, items, counts, members,
                                 down_codes, down_scales, n, k, slots, common, warps, stages);
}

// Down GEMM of a device plan into every original (row, slot) output: FP32 decode or BF16 prompt slots.
void down_plan(const at::Tensor& codes, const at::Tensor& scales, const at::Tensor& blob, const at::Tensor& weight,
    const at::Tensor& input, const at::Tensor& items, const at::Tensor& counts, const at::Tensor& members,
    at::Tensor out, int64_t n, int64_t k, int64_t sk, int64_t slots, bool direct, int64_t tile) {
    plan_common(blob, items, counts, members, n, k, sk, slots, tile);
    tensor(codes, blob, at::kByte, 3, "codes");
    tensor(scales, blob, at::kByte, 3, "scales");
    expert_field(weight, blob, at::kFloat, 2, "weight calibration");
    expert_field(input, blob, at::kFloat, 2, "input calibration");
    const auto p = members.size(0), cells = blob.size(0);
    TORCH_CHECK(blob.size(1) == 1 && weight.size(0) == cells && weight.size(1) == 1 && input.sizes() == weight.sizes(),
                "native A4 down: one projection's blob and calibrations");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(codes.data_ptr()) % 4 == 0 &&
                reinterpret_cast<uintptr_t>(scales.data_ptr()) % 4 == 0 && codes.size(0) == 1 &&
                codes.size(1) == p && codes.size(2) == k / 2 && scales.size(0) == 1 && scales.size(1) == p &&
                scales.size(2) == k / 16, "native A4 down: pair-major codes/scales");
    TORCH_CHECK(out.is_cuda() && out.device() == blob.device() && out.dim() == 3 && out.size(2) == n &&
                out.size(1) == slots && out.size(0) * slots == p && out.stride(2) == 1 && out.stride(1) >= n &&
                out.stride(0) >= 0 && out.stride(0) / out.size(1) >= out.stride(1) &&
                (out.scalar_type() == at::kBFloat16 || out.scalar_type() == at::kFloat),
                "native A4 down: separate BF16/FP32 slots covering every original row slot");
    const auto limit = std::numeric_limits<int64_t>::max() / out.element_size();
    const auto row_span = extend_span(n, out.size(1), out.stride(1), limit, "nativeA4 output slots");
    extend_span(row_span, out.size(0), out.stride(0), limit, "nativeA4 output rows");
    c10::cuda::CUDAGuard guard(blob.device());
    expert_a4_down_plan_cuda(codes, scales, blob, weight, input, items, counts, members, out, n, k, sk, slots, direct,
                             tile);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize", &quantize, pybind11::arg("x"), pybind11::arg("plan"), pybind11::arg("input"),
          pybind11::arg("codes"), pybind11::arg("scales"), pybind11::arg("paired") = false, pybind11::arg("direct") = false);
    m.def("gemm", &gemm, pybind11::arg("codes"), pybind11::arg("scales"), pybind11::arg("blob"),
          pybind11::arg("weight"), pybind11::arg("input"), pybind11::arg("items"), pybind11::arg("plan"),
          pybind11::arg("out"), pybind11::arg("n"), pybind11::arg("k"), pybind11::arg("sk"), pybind11::arg("down"),
          pybind11::arg("direct") = false);
    m.def("quantize_plan", &quantize_plan, pybind11::arg("x"), pybind11::arg("items"), pybind11::arg("counts"),
          pybind11::arg("members"), pybind11::arg("input"), pybind11::arg("codes"), pybind11::arg("scales"),
          pybind11::arg("slots"), pybind11::arg("direct"), pybind11::arg("tile"));
    m.def("fused_up", &fused_up, pybind11::arg("codes"), pybind11::arg("scales"), pybind11::arg("blob"),
          pybind11::arg("weight"), pybind11::arg("input"), pybind11::arg("down_input"), pybind11::arg("items"),
          pybind11::arg("counts"), pybind11::arg("members"), pybind11::arg("down_codes"),
          pybind11::arg("down_scales"), pybind11::arg("n"), pybind11::arg("k"), pybind11::arg("sk"),
          pybind11::arg("slots"), pybind11::arg("common"), pybind11::arg("direct"), pybind11::arg("tile"),
          pybind11::arg("act"));
    m.def("down_plan", &down_plan, pybind11::arg("codes"), pybind11::arg("scales"), pybind11::arg("blob"),
          pybind11::arg("weight"), pybind11::arg("input"), pybind11::arg("items"), pybind11::arg("counts"),
          pybind11::arg("members"), pybind11::arg("out"), pybind11::arg("n"), pybind11::arg("k"), pybind11::arg("sk"),
          pybind11::arg("slots"), pybind11::arg("direct"), pybind11::arg("tile"));
    m.def("gemma_fused_up", &gemma_fused_up,
          "Private CT GeGLU tile16/SK1 investigation; owner-prepared disjoint fields as fused_up; no default policy.",
          pybind11::arg("codes"), pybind11::arg("scales"), pybind11::arg("blob"), pybind11::arg("weight"),
          pybind11::arg("input"), pybind11::arg("down_input"), pybind11::arg("items"), pybind11::arg("counts"),
          pybind11::arg("members"), pybind11::arg("down_codes"), pybind11::arg("down_scales"),
          pybind11::arg("n"), pybind11::arg("k"), pybind11::arg("slots"), pybind11::arg("common"),
          pybind11::arg("warps"), pybind11::arg("stages"));
}
