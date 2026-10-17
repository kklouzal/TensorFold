// PRIVATE generated stride variant; regenerate, do not hand-edit.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <limits>

void blob_a4_quant_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, at::Tensor&, bool);
void blob_a4_gemm_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                         const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&,
                         int64_t, int64_t, int64_t, bool);

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

// Plan rows are (source-row, destination-slot, cache-cell). They are constructed
// from admitted CPU picks by the owner; kernels also guard every indirect access.
void blob_quantize(const at::Tensor& x, const at::Tensor& plan, const at::Tensor& input,
               at::Tensor codes, at::Tensor scales, bool paired) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.stride(1) == 1 &&
                x.stride(0) >= x.size(1) && x.size(0) > 0 && x.size(0) < INT32_MAX && x.size(1) > 0 &&
                x.size(1) <= INT32_MAX && x.size(1) % 64 == 0, "x: BF16 rows in complete64-input tiles");
    extend_span(x.size(1), x.size(0), x.stride(0),
                std::numeric_limits<int64_t>::max() / x.element_size(), "nativeA4 input");
    tensor(plan, x, at::kInt, 2, "plan");
    expert_field(input, x, at::kFloat, 2, "input calibration");
    tensor(codes, x, at::kByte, 3, "codes");
    tensor(scales, x, at::kByte, 3, "scales");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(codes.data_ptr()) % 8 == 0, "codes require8-byte alignment");
    const auto p = plan.size(0), k = x.size(1), projections = input.size(1);
    TORCH_CHECK(p > 0 && p <= 128 && plan.size(1) == 3 && input.size(0) > 0 && input.size(0) <= INT32_MAX &&
                (projections == 1 || projections == 2), "plan/calibrations:128pairs maximum, one/two projections");
    TORCH_CHECK(codes.size(0) == projections && codes.size(1) == p && codes.size(2) == k / 2 &&
                scales.size(0) == projections && scales.size(1) == p && scales.size(2) == k / 16,
                "quantization buffers must match projection/pair/input dimensions");
    c10::cuda::CUDAGuard guard(x.device());
    TORCH_CHECK(!paired || x.size(0) == p, "pair-indexed inputs must hold exactly every planned pair");
    blob_a4_quant_cuda(x, plan, input, codes, scales, paired);
}

// Codes/scales are pair-major. Items (cell,first,count<=16) group same-expert
// pairs; decode K slices match checkpoint.matmul's shape-only policy exactly.
void blob_gemm(const at::Tensor& codes, const at::Tensor& scales, const at::Tensor& blob,
           const at::Tensor& weight, const at::Tensor& input, const at::Tensor& items,
           const at::Tensor& plan, at::Tensor out, int64_t n, int64_t k, int64_t sk, bool down) {
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
    const auto projections = blob.size(1), p = plan.size(0), cells = blob.size(0);
    TORCH_CHECK(n > 0 && n < INT32_MAX && n % 64 == 0 && k > 0 && k < INT32_MAX && k % 64 == 0 &&
                sk > 0 && sk <= 8 && (k / 64) % sk == 0, "nativeA4 dimensions/slices must form complete tiles");
    TORCH_CHECK(p > 0 && p <= 128 && plan.size(1) == 3 && cells > 0 && cells <= INT32_MAX &&
                items.size(0) > 0 && items.size(0) <= p && items.size(1) == 3 &&
                projections == (down ? 1 : 2), "nativeA4 bounded plan/projection geometry");
    TORCH_CHECK(items.size(0) * (n / 8) <= static_cast<int64_t>(INT32_MAX) * 4,
                "nativeA4 launch exceeds CUDA grid index range");
    TORCH_CHECK(blob.size(2) == (n / 16) * k * 9 && weight.size(0) == cells && weight.size(1) == projections &&
                input.sizes() == weight.sizes(), "nativeA4 blob/calibration geometry");
    TORCH_CHECK(codes.size(0) == projections && codes.size(1) == p && codes.size(2) == k / 2 &&
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
    blob_a4_gemm_cuda(codes, scales, blob, weight, input, items, plan, out, n, k, sk, down);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize", &blob_quantize, pybind11::arg("x"), pybind11::arg("plan"), pybind11::arg("input"),
          pybind11::arg("codes"), pybind11::arg("scales"), pybind11::arg("paired") = false);
    m.def("gemm", &blob_gemm);
}
