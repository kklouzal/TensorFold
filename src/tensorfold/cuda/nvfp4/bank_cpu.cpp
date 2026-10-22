// Exact native A4 byte layout. Python owns all buffers through this synchronous
// call and grants exclusive writable authority; no Tensor ABI is used here.
#include <ATen/Parallel.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>

namespace py = pybind11;
using U = std::uint64_t;

static U bounded(U left, U right) {
    const U limit = static_cast<U>(std::numeric_limits<py::ssize_t>::max());
    if (right && left > limit / right)
        throw py::value_error("native A4 byte geometry exceeds buffer size limits");
    return left * right;
}

static U positive(py::ssize_t value) {
    if (value <= 0) throw py::value_error("native A4 byte geometry must be positive");
    return static_cast<U>(value);
}

static py::buffer_info bytes(const py::array& array, bool writable) {
    if (!array.dtype().is(py::dtype::of<std::uint8_t>()) || array.ndim() != 1
        || !(array.flags() & py::array::c_style) || (writable && !array.writeable()))
        throw py::value_error("native A4 buffers must be contiguous U8 vectors with writable destinations");
    return array.request();
}

struct Span { std::uintptr_t first, last; };
static Span span(const void* data, U size) {
    const auto first = reinterpret_cast<std::uintptr_t>(data);
    if (size > std::numeric_limits<std::uintptr_t>::max() - first)
        throw py::value_error("native A4 buffer address exceeds address limits");
    return {first, first + size};
}
static bool overlap(Span a, Span b) { return a.first < b.last && b.first < a.last; }

static U extent(U offset, U stride, U rows, U row_bytes, U capacity) {
    const U tail = bounded(rows - 1, stride);
    if (offset > capacity || tail > capacity - offset || row_bytes > capacity - offset - tail)
        throw py::value_error("native A4 destination exceeds its physical buffer");
    return tail + row_bytes;
}

template <unsigned Tiles>
static void copy_tile_rows(std::uint8_t* words, std::uint8_t* scales,
                           const std::uint8_t* input, const std::uint8_t* input_scales,
                           U code_stride, U scale_stride) {
    // Two adjacent source tiles are consumed in each row before advancing its
    // stride. A separately instantiated singleton handles an odd final tile.
    for (U row = 0; row < 64; ++row) {
        for (unsigned tile = 0; tile < Tiles; ++tile) {
            auto* output = words + tile * 2048 + row * 32;
            const auto* source = input + row * code_stride + tile * 32;
            // Keep the fixed bit transfers in the row body: the prior helper
            // remained an actual per-row call under the pinned optimizer.
            // memcpy permits unaligned input without typed-pointer aliasing.
            std::memcpy(output, source, 4);
            std::memcpy(output + 4, source + 16, 4);
            std::memcpy(output + 8, source + 4, 4);
            std::memcpy(output + 12, source + 20, 4);
            std::memcpy(output + 16, source + 8, 4);
            std::memcpy(output + 20, source + 24, 4);
            std::memcpy(output + 24, source + 12, 4);
            std::memcpy(output + 28, source + 28, 4);
            std::memcpy(scales + tile * 256 + row * 4,
                        input_scales + row * scale_stride + tile * 4, 4);
        }
    }
}

static void pack(const py::array& code_array, const py::array& scale_array,
                 const py::array& up_array, const py::array& down_array, const py::array& index_array,
                 py::ssize_t up_offset_arg, py::ssize_t up_stride_arg,
                 py::ssize_t down_offset_arg, py::ssize_t down_stride_arg,
                 py::ssize_t experts_arg, py::ssize_t width_arg, py::ssize_t dims_arg) {
    auto codes = bytes(code_array, false), scales = bytes(scale_array, false);
    auto up = bytes(up_array, true), down = bytes(down_array, true);
    if (!index_array.dtype().is(py::dtype::of<std::int64_t>()) || index_array.ndim() != 1
        || !(index_array.flags() & py::array::c_style))
        throw py::value_error("native A4 local indices must be a contiguous int64 vector");
    auto indices = index_array.request();
    const U experts = positive(experts_arg), width = positive(width_arg), dims = positive(dims_arg);
    const U count = positive(indices.size);
    if (experts > 1023 || count > experts || width >= (U{1} << 31) || dims >= (U{1} << 31)
        || width % 64 || dims % 64 || up_offset_arg < 0 || down_offset_arg < 0)
        throw py::value_error("native A4 bank geometry or local count is invalid");
    const U words = bounded(width, dims) / 2, sf = words / 8, blob = words + sf;
    const U up_stride = positive(up_stride_arg), down_stride = positive(down_stride_arg);
    const U up_offset = static_cast<U>(up_offset_arg), down_offset = static_cast<U>(down_offset_arg);
    if (static_cast<U>(codes.size) != bounded(count, bounded(3, words))
        || static_cast<U>(scales.size) != bounded(count, bounded(3, sf))
        || up_stride < bounded(2, blob) || down_stride < blob)
        throw py::value_error("native A4 source lengths or destination row strides differ from geometry");
    const U up_extent = extent(up_offset, up_stride, experts, bounded(2, blob), up.size);
    const U down_extent = extent(down_offset, down_stride, experts, blob, down.size);
    auto* up_data = static_cast<std::uint8_t*>(up.ptr) + up_offset;
    auto* down_data = static_cast<std::uint8_t*>(down.ptr) + down_offset;
    const Span up_span = span(up_data, up_extent), down_span = span(down_data, down_extent);
    if ((up_span.first | down_span.first) % 4)
        throw py::value_error("native A4 destinations require four-byte alignment");
    if (overlap(up_span, down_span)) {
        const U delta = down_span.first >= up_span.first ? down_span.first - up_span.first : 0;
        if (up_stride != down_stride || delta < 2 * blob || delta > up_stride || blob > up_stride - delta)
            throw py::value_error("native A4 destination cells overlap");
    }
    const Span code_span = span(codes.ptr, codes.size), scale_span = span(scales.ptr, scales.size);
    const Span index_span = span(indices.ptr, bounded(count, sizeof(std::int64_t)));
    if (overlap(up_span, code_span) || overlap(up_span, scale_span) || overlap(up_span, index_span)
        || overlap(down_span, code_span) || overlap(down_span, scale_span) || overlap(down_span, index_span))
        throw py::value_error("native A4 destinations overlap immutable inputs");
    // Snapshot and validate every ID before release or dispatch. Workers read
    // only this retained map and raw buffers; no Python, Tensor or caller TLS.
    std::array<std::int64_t, 1023> local;
    std::memcpy(local.data(), indices.ptr, bounded(count, sizeof(std::int64_t)));
    for (U e = 0; e < count; ++e) {
        if (local[e] < 0 || static_cast<U>(local[e]) >= experts)
            throw py::value_error("native A4 local index is outside the authority");
        for (U previous = 0; previous < e; ++previous)
            if (local[e] == local[previous]) throw py::value_error("native A4 local indices must be distinct");
    }
    const auto* code_data = static_cast<const std::uint8_t*>(codes.ptr);
    const auto* sf_data = static_cast<const std::uint8_t*>(scales.ptr);
    const auto tasks = static_cast<std::int64_t>(count); // count <= 1023 above
    py::gil_scoped_release release;
    // The adopted synchronous range owner joins before return/rethrow. Setup
    // can fail; authority remains unpublished, with all buffers retained here.
    // Tasks own distinct experts' projection cells even for interleaved up/down
    // allocations. Do not turn the enclosing-span checks into noalias promises.
    at::parallel_for(0, tasks, 1, [&](std::int64_t first, std::int64_t last) {
        for (auto task = first; task < last; ++task) {
            const U e = static_cast<U>(task);
            for (U projection = 0; projection < 3; ++projection) {
                const U n = projection ? width : dims, k = projection ? dims : width;
                const U nt = n / 64, kt = k / 64;
                const auto* source = code_data + (e * 3 + projection) * words;
                const auto* source_sf = sf_data + (e * 3 + projection) * sf;
                auto* target = projection ? up_data + local[e] * up_stride + (projection - 1) * blob
                                          : down_data + local[e] * down_stride;
                for (U i = 0; i < nt; ++i) {
                    U j = 0;
                    for (; j + 1 < kt; j += 2) {
                        copy_tile_rows<2>(target + (i * kt + j) * 2048,
                                          target + words + (i * kt + j) * 256,
                                          source + i * 64 * (k / 2) + j * 32,
                                          source_sf + i * 64 * (k / 16) + j * 4,
                                          k / 2, k / 16);
                    }
                    if (j < kt) {
                        copy_tile_rows<1>(target + (i * kt + j) * 2048,
                                          target + words + (i * kt + j) * 256,
                                          source + i * 64 * (k / 2) + j * 32,
                                          source_sf + i * 64 * (k / 16) + j * 4,
                                          k / 2, k / 16);
                    }
                }
            }
        }
    });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    // Preparation-only observations. Actual compiler/backend/runtime-provider
    // admission is required before claiming that this dispatch is parallel.
    module.attr("_parallel_openmp") = static_cast<bool>(AT_PARALLEL_OPENMP);
    module.attr("_parallel_native") = static_cast<bool>(AT_PARALLEL_NATIVE);
#ifdef INTRA_OP_PARALLEL
    module.attr("_parallel_compiled") = true;
#else
    module.attr("_parallel_compiled") = false;
#endif
#ifdef _OPENMP
    module.attr("_openmp_version") = _OPENMP;
#else
    module.attr("_openmp_version") = 0;
#endif
    module.attr("_cxx_mode") = static_cast<long>(__cplusplus);
    module.def("_parallel_threads", &at::get_num_threads);
    module.def("_parallel_info", &at::get_parallel_info);
    module.def("_parallel_probe", [] {
        // An explicit SDK-only observation, never called by pack or import.
        // Scalars identify this probe's actual workers, not packing timings.
        std::array<std::int64_t, 30> ids;
        ids.fill(-1);
        {
            py::gil_scoped_release release;
            at::parallel_for(0, 30, 1, [&](std::int64_t first, std::int64_t last) {
                for (auto task = first; task < last; ++task)
                    ids[task] = at::get_thread_num();
            });
        }
        py::tuple result(ids.size());
        for (std::size_t task = 0; task < ids.size(); ++task)
            result[task] = ids[task];
        return result;
    });
    module.def("pack", &pack, py::arg("codes").noconvert(), py::arg("scales").noconvert(),
               py::arg("up").noconvert(), py::arg("down").noconvert(), py::arg("indices").noconvert(),
               py::arg("up_offset"), py::arg("up_stride"), py::arg("down_offset"), py::arg("down_stride"),
               py::arg("experts"), py::arg("width"), py::arg("dims"));
}
