// Checked caller-owned buffers; CUDA values remain device-side and graph-safe.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

using T = at::Tensor;
using O = std::optional<T>;

void sm70_rq4_encode_cuda(const T&, T&, T&, const O&, const O&, const O&, int, int, int, int);
void sm70_rq4_rotate_cuda(const T&, T&, const O&, const O&, int, int, bool, int, int);
void sm70_rq4_decode_cuda(const T&, const T&, T&, const O&, const O&, int, int, bool, int, int);
void sm70_rq4_copy_cuda(const T&, T&, const O&, const O&, const O&, int);

namespace {
void check(const T& value, const T& owner, at::ScalarType dtype, const char* name, bool contiguous = true) {
    TORCH_CHECK(value.is_cuda() && value.device() == owner.device() && value.scalar_type() == dtype
                && (!contiguous || value.is_contiguous()), name, " requires same-device declared CUDA storage");
}

void floating(const T& value, const T& owner, const char* name, bool contiguous = true) {
    TORCH_CHECK(value.scalar_type() == at::kHalf || value.scalar_type() == at::kFloat,
                name, " requires FP16 or FP32 (never BF16) storage");
    check(value, owner, value.scalar_type(), name, contiguous);
}

void geometry(const T& value, int64_t rows, bool row_strided = false) {
    TORCH_CHECK(value.dim() == 3 && rows >= 0 && rows <= value.size(0)
                && rows <= std::numeric_limits<int>::max() && value.size(1) > 0
                && value.size(2) > 0 && value.size(2) % 128 == 0
                && value.size(1) <= std::numeric_limits<int>::max()
                && value.size(2) <= std::numeric_limits<int>::max(), "RotorQuant geometry requires [R,H,128*k]");
    TORCH_CHECK(value.size(1) <= std::numeric_limits<int>::max() / (value.size(2) / 128)
                && rows <= std::numeric_limits<int>::max() / (value.size(1) * (value.size(2) / 128)),
                "RotorQuant launch geometry exceeds bounded CUDA grid");
    if (row_strided)
        TORCH_CHECK(value.stride(2) == 1 && value.stride(1) == value.size(2)
                    && value.stride(0) >= value.size(1) * value.size(2), "source needs contiguous head rows");
}

void controls(const T& owner, const O& position, const O& status, const O& live,
              int codec, int error, int layer) {
    const auto* prop = at::cuda::getDeviceProperties(owner.get_device());
    TORCH_CHECK(prop->major == 7 && prop->minor == 0, "RotorQuant FP16 native codec requires SM70");
    TORCH_CHECK((codec >= 1 && codec <= 5) || codec == 9,
                "native Q4 codec requires a registered basis 1..5 or pinned compact Planar4 9");
    TORCH_CHECK(error == 1 || error == 2 || error == 4 || error == 8 || error == 16,
                "one declared KV numeric error bit required");
    TORCH_CHECK(layer >= -1, "KV error layer must be a model layer or MTP -1");
    if (status) { check(*status, owner, at::kInt, "KV status"); TORCH_CHECK(status->numel() == 2, "two-word KV status required"); }
    if (live) { check(*live, owner, at::kInt, "live row control"); TORCH_CHECK(live->numel() == 1, "one live row integer required"); }
    if (position) {
        TORCH_CHECK(position->scalar_type() == at::kInt || position->scalar_type() == at::kLong,
                    "position must be int32 or int64");
        check(*position, owner, position->scalar_type(), "cache position");
        TORCH_CHECK(position->numel() == 1, "one cache base position required");
    }
}

// Conservatively include row gaps. Outputs may not borrow projection, status,
// or control storage; all legitimate owners are independent allocations/views.
std::pair<uintptr_t, uintptr_t> span(const T& value, int64_t prefix = -1) {
    if (!value.numel() || prefix == 0) return {0, 0};
    const uintptr_t begin = reinterpret_cast<uintptr_t>(value.data_ptr());
    constexpr auto maximum = std::numeric_limits<uintptr_t>::max();
    uint64_t elements = 1;
    if (prefix >= 0) {
        TORCH_CHECK(value.is_contiguous() && prefix <= value.numel(), "invalid mutable prefix");
        elements = prefix;
    } else {
        for (int dim = 0; dim < value.dim(); ++dim) {
            TORCH_CHECK(value.stride(dim) >= 0, "negative RotorQuant strides unsupported");
            const uint64_t size = value.size(dim) - 1, stride = value.stride(dim);
            TORCH_CHECK(!size || stride <= (maximum - elements) / size, "RotorQuant storage extent overflow");
            elements += size * stride;
        }
    }
    TORCH_CHECK(elements <= (maximum - begin) / value.element_size(), "RotorQuant byte extent overflow");
    return {begin, begin + elements * value.element_size()};
}

void disjoint(const T& write, const T& read, int64_t prefix = -1) {
    const auto a = span(write, prefix), b = span(read);
    TORCH_CHECK(a.first == a.second || b.first == b.second || a.second <= b.first || b.second <= a.first,
                "RotorQuant mutable outputs/status must be disjoint from inputs and each other");
}

void mutable_controls(const T& source, const O& position, const O& status, const O& live) {
    if (!status) return;
    disjoint(*status, source);
    if (position) disjoint(*status, *position);
    if (live) disjoint(*status, *live);
}

void output_controls(const T& output, const O& position, const O& status, const O& live, int64_t prefix = -1) {
    if (position) disjoint(output, *position, prefix);
    if (status) disjoint(output, *status, prefix);
    if (live) disjoint(output, *live, prefix);
}
}  // namespace

void encode(const T& x, T codes, T scales, O position, O status, O valid_rows,
            int codec, int64_t rows, int error, int layer) {
    floating(x, x, "codec source", false);
    geometry(x, rows, true);
    check(codes, x, at::kByte, "packed Q4 codes"); check(scales, x, at::kFloat, "Q4 original RMS");
    TORCH_CHECK(codes.dim() == 3 && scales.dim() == 3 && codes.size(0) == scales.size(0)
                && codes.size(1) == x.size(1) && scales.size(1) == x.size(1)
                && codes.size(2) == x.size(2) / 2 && scales.size(2) == x.size(2) / 128,
                "packed Q4 cache geometry does not match its source");
    TORCH_CHECK(position || rows <= codes.size(0), "destination needs every requested source row");
    controls(x, position, status, valid_rows, codec, error, layer);
    mutable_controls(x, position, status, valid_rows);
    for (const auto* output : {&codes, &scales}) {
        disjoint(*output, x); output_controls(*output, position, status, valid_rows);
    }
    disjoint(codes, scales);
    c10::cuda::CUDAGuard guard(x.device());
    sm70_rq4_encode_cuda(x, codes, scales, position, status, valid_rows, codec, int(rows), error, layer);
}

void rotate(const T& x, T out, O status, O valid_rows,
            int codec, int64_t rows, bool inverse, int error, int layer) {
    floating(x, x, "rotation input", false); geometry(x, rows, true);
    floating(out, x, "rotation output");
    TORCH_CHECK(out.dim() == 3 && out.size(0) >= rows && out.size(1) == x.size(1) && out.size(2) == x.size(2),
                "rotation output geometry");
    controls(x, {}, status, valid_rows, codec, error, layer);
    const int64_t count = rows * x.size(1) * x.size(2);
    mutable_controls(x, {}, status, valid_rows);
    disjoint(out, x, count); output_controls(out, {}, status, valid_rows, count);
    c10::cuda::CUDAGuard guard(x.device());
    sm70_rq4_rotate_cuda(x, out, status, valid_rows, codec, int(rows), inverse, error, layer);
}

void decode(const T& codes, const T& scales, T out, O status, O valid_rows,
            int codec, int64_t rows, bool inverse, int error, int layer) {
    floating(out, out, "decoded output"); geometry(out, rows);
    check(codes, out, at::kByte, "packed Q4 codes"); check(scales, out, at::kFloat, "Q4 RMS");
    TORCH_CHECK(codes.dim() == 3 && scales.dim() == 3 && codes.size(0) >= rows && scales.size(0) == codes.size(0)
                && codes.size(1) == out.size(1) && scales.size(1) == out.size(1)
                && codes.size(2) == out.size(2) / 2 && scales.size(2) == out.size(2) / 128,
                "decoded Q4 geometry");
    controls(out, {}, status, valid_rows, codec, error, layer);
    mutable_controls(codes, {}, status, valid_rows);
    if (status) disjoint(*status, scales);
    const int64_t count = rows * out.size(1) * out.size(2);
    disjoint(out, codes, count); disjoint(out, scales, count); output_controls(out, {}, status, valid_rows, count);
    c10::cuda::CUDAGuard guard(out.device());
    sm70_rq4_decode_cuda(codes, scales, out, status, valid_rows, codec, int(rows), inverse, error, layer);
}

void copy_rows(const T& x, T target, O position, O status, O valid_rows, int layer) {
    check(x, x, x.scalar_type(), "row copy source", false);
    check(target, x, x.scalar_type(), "row copy target");
    TORCH_CHECK((x.dim() == 2 || x.dim() == 3) && target.dim() == x.dim()
                && x.size(0) <= std::numeric_limits<int>::max(), "row copy requires bounded rank2/rank3 tensors");
    int64_t width = 1;
    for (int dim = x.dim() - 1; dim >= 1; --dim) {
        TORCH_CHECK(x.size(dim) > 0 && x.size(dim) == target.size(dim) && x.stride(dim) == width,
                    "row copy trailing dimensions must match and be contiguous");
        TORCH_CHECK(width <= std::numeric_limits<int64_t>::max() / x.size(dim), "row copy width overflow");
        width *= x.size(dim);
    }
    TORCH_CHECK(x.stride(0) >= width && x.stride(0) <= std::numeric_limits<int64_t>::max() / x.element_size()
                && width <= std::numeric_limits<int64_t>::max() / x.element_size()
                && (position || x.size(0) <= target.size(0)), "row copy row extent");
    controls(x, position, status, valid_rows, 4, 16, layer);
    mutable_controls(x, position, status, valid_rows);
    disjoint(target, x); output_controls(target, position, status, valid_rows);
    c10::cuda::CUDAGuard guard(x.device());
    sm70_rq4_copy_cuda(x, target, position, status, valid_rows, layer);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("encode", &encode);
    m.def("rotate", &rotate);
    m.def("decode", &decode);
    m.def("copy_rows", &copy_rows);
}
