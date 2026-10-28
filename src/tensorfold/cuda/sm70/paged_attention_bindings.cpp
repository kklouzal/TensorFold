// Bounded, caller-owned buffers for native SM70 unique-history attention.
#include <torch/extension.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <limits>
#include <cstdint>
#include <array>
#include <memory>
#include <torch/csrc/autograd/python_variable.h>
#include "paged_attention_descriptor.cuh"

using T = at::Tensor;

void sm70_paged_statistics_cuda(const T&, const T&, const T&, T&, int, int, const T&, int, float);
void sm70_paged_merge_cuda(const T&, const T&, T&, int, const T&);
void sm70_paged_values_cuda(const T&, const T&, const T&, const T&, T&, int, int, const T&, int, int, float);
void sm70_paged_merge_values_cuda(const T&, const T&, T&, int, const T&, int);

namespace {
constexpr int WORLD = 8, HEADS = 12, DIM = 256, WORDS = tf_sm70_paged::WORD_COUNT;
constexpr size_t STORAGE_TENSORS = 69, STORAGE_HOST_BYTES = 16384;

struct StorageRecord {
    c10::Storage owner;
    const void* data = nullptr;
    std::array<int64_t, 3> sizes{}, strides{};
    int64_t dimensions = 0, offset = 0;
    size_t bytes = 0;
    at::ScalarType dtype = at::ScalarType::Undefined;
    c10::Device device{c10::DeviceType::CPU};
};

bool storage_supported(const pybind11::tuple& tensors) {
    if (tensors.size() > STORAGE_TENSORS) return false;
    for (auto object : tensors) {
        // Preserve the original Python method/exception path for subclasses.
        if (Py_TYPE(object.ptr()) != reinterpret_cast<PyTypeObject*>(THPVariableClass)) return false;
        const auto& tensor = THPVariable_Unpack(object.ptr());
        if (tensor.layout() != at::kStrided || tensor.is_nested() || tensor.dim() > 3) return false;
    }
    return true;
}

class StorageBindings {
    std::array<StorageRecord, STORAGE_TENSORS> records_;
    size_t count_;
    uint64_t checks_ = 0;
    bool overflowed_ = false;
public:
    explicit StorageBindings(const pybind11::tuple& tensors) : count_(tensors.size()) {
        TORCH_CHECK(storage_supported(tensors), "storage bindings need at most69 plain strided tensors");
        for (size_t index = 0; index < count_; ++index) {
            const auto& tensor = THPVariable_Unpack(tensors[index].ptr());
            auto& record = records_[index];
            record.data = tensor.data_ptr();
            record.dimensions = tensor.dim();
            for (int64_t axis = 0; axis < record.dimensions; ++axis) {
                record.sizes[axis] = tensor.size(axis);
                record.strides[axis] = tensor.stride(axis);
            }
            record.dtype = tensor.scalar_type();
            record.device = tensor.device();
            record.owner = tensor.storage();  // Retain the original allocation across Tensor.set_.
            record.offset = tensor.storage_offset();
            record.bytes = record.owner.nbytes();
        }
    }

    pybind11::object matches(const pybind11::tuple& tensors) {
        if (checks_ < uint64_t(std::numeric_limits<int64_t>::max())) ++checks_;
        else overflowed_ = true;
        if (!storage_supported(tensors)) return pybind11::none();
        // Do not early-out on a mismatch and suppress a later metadata error.
        bool same = tensors.size() == count_;
        for (size_t index = 0; index < tensors.size(); ++index) {
            const auto& tensor = THPVariable_Unpack(tensors[index].ptr());
            const auto* data = tensor.data_ptr();
            const auto dimensions = tensor.dim();
            const auto sizes = tensor.sizes();
            const auto dtype = tensor.scalar_type();
            const auto device = tensor.device();
            const auto storage = tensor.storage();
            const auto offset = tensor.storage_offset();
            const auto strides = tensor.strides();
            const auto bytes = storage.nbytes();
            if (index >= count_) { same = false; continue; }
            const auto& record = records_[index];
            same &= data == record.data && dimensions == record.dimensions
                && dtype == record.dtype && device == record.device
                && storage.unsafeGetStorageImpl() == record.owner.unsafeGetStorageImpl()
                && offset == record.offset && bytes == record.bytes;
            for (int64_t axis = 0; axis < dimensions; ++axis)
                same &= axis < record.dimensions && sizes[axis] == record.sizes[axis]
                    && strides[axis] == record.strides[axis];
        }
        return pybind11::bool_(same);
    }
    uint64_t checks() const { return checks_; }
    bool overflowed() const { return overflowed_; }
};
static_assert(sizeof(StorageBindings) <= STORAGE_HOST_BYTES - 4096,
              "storage binding object exceeds its per-State host allowance");

void check(const T& value, const T& owner, at::ScalarType dtype, const char* name) {
    TORCH_CHECK(value.is_cuda() && value.device() == owner.device() && value.scalar_type() == dtype
                && value.is_contiguous(), name, " needs contiguous same-device storage with declared dtype");
}

int rows_of(const T& queries) {
    check(queries, queries, at::kHalf, "gathered queries");
    const auto* properties = at::cuda::getDeviceProperties(queries.get_device());
    TORCH_CHECK(properties->major == 7 && properties->minor == 0, "paged owner attention requires SM70");
    TORCH_CHECK(queries.dim() == 4 && queries.size(0) == WORLD && queries.size(1) <= INT32_MAX
                && queries.size(2) == 3 && queries.size(3) == DIM, "queries require [8,rows,3,256]");
    return int(queries.size(1));
}

void controls(const T& table, const T& owner, int rows, int group, int page_tokens, const T& layer) {
    check(table, owner, at::kLong, "owner row descriptors");
    TORCH_CHECK(table.dim() == 2 && table.size(0) == rows && table.size(1) == WORDS,
                "one 16-word descriptor per query row required");
    TORCH_CHECK(group == 0 || group == 4, "KV-head group starts at rank zero or four");
    TORCH_CHECK(page_tokens > 0 && page_tokens % 4 == 0, "history pages require positive complete pooling groups");
    check(layer, owner, at::kInt, "current layer");
    TORCH_CHECK(layer.dim() == 1 && layer.numel() == 1, "current layer requires a stable int32[1] owner");
}

void selections(const T& selected, const T& owner, int rows) {
    check(selected, owner, at::kInt, "canonical selections");
    TORCH_CHECK(selected.dim() == 3 && selected.size(0) == WORLD && selected.size(1) == rows
                && selected.size(2) >= 2 && selected.size(2) <= INT32_MAX,
                "canonical selections require [8,rows,id_width+2]");
}

void statistics(const T& value, const T& owner, int rows, bool gathered) {
    check(value, owner, at::kFloat, "softmax statistics");
    if (gathered) {
        TORCH_CHECK(value.dim() == 4 && value.size(0) == WORLD && value.size(1) == rows
                    && value.size(2) == HEADS && value.size(3) == 2, "gathered statistics require [8,rows,12,2]");
    } else {
        TORCH_CHECK(value.dim() == 3 && value.size(0) == rows && value.size(1) == HEADS && value.size(2) == 2,
                    "statistics require [rows,12,2]");
    }
}

void format(int bits, int codec) {
    TORCH_CHECK((bits == 8 && codec == 0) || (bits == 4 && (codec == 9 || codec == 10)),
                "paged owner attention supports ordinary Q8, pinned Iso4 codec10 or Planar4 codec9");
}

float attention_scale(double scale) {
    TORCH_CHECK(std::isfinite(scale) && scale > 0 && scale <= std::numeric_limits<float>::max(),
                "attention scale must be positive and finite FP32");
    const float value = float(scale);
    TORCH_CHECK(std::isfinite(value) && value > 0, "attention scale must be positive and finite FP32");
    return value;
}

void distinct(const T& output, const T& input) {
    // Contiguity was established once above; preserve only the cross-buffer
    // aliasing check needed while earlier input phases still own their data.
    at::assert_no_overlap(output, input);
}
}  // namespace

void local_statistics(const T& queries, const T& selected, const T& table, T output,
                      int group, int page_tokens, const T& layer, int key_bits, int key_codec, double scale) {
    const int rows = rows_of(queries);
    controls(table, queries, rows, group, page_tokens, layer);
    selections(selected, queries, rows);
    statistics(output, queries, rows, false);
    format(key_bits, key_codec);
    for (const auto* input : {&queries, &selected, &table, &layer}) distinct(output, *input);
    c10::cuda::CUDAGuard guard(queries.device());
    sm70_paged_statistics_cuda(queries, selected, table, output, group, page_tokens, layer,
                              key_bits == 4 ? key_codec : 0, attention_scale(scale));
}

void merge_statistics(const T& gathered, const T& table, T output, int group, int page_tokens, const T& layer) {
    check(gathered, gathered, at::kFloat, "gathered statistics");
    const auto* properties = at::cuda::getDeviceProperties(gathered.get_device());
    TORCH_CHECK(properties->major == 7 && properties->minor == 0, "paged owner attention requires SM70");
    TORCH_CHECK(gathered.dim() == 4 && gathered.size(1) <= INT32_MAX, "gathered statistics geometry");
    const int rows = int(gathered.size(1));
    controls(table, gathered, rows, group, page_tokens, layer);
    statistics(gathered, gathered, rows, true);
    statistics(output, gathered, rows, false);
    distinct(output, gathered);
    distinct(output, table);
    distinct(output, layer);
    c10::cuda::CUDAGuard guard(gathered.device());
    sm70_paged_merge_cuda(gathered, table, output, group, layer);
}

void normalized_partial(const T& queries, const T& selected, const T& table, const T& global, T output,
                        int group, int page_tokens, const T& layer, int key_bits, int key_codec,
                        int value_bits, int value_codec, double scale) {
    const int rows = rows_of(queries);
    controls(table, queries, rows, group, page_tokens, layer);
    selections(selected, queries, rows);
    statistics(global, queries, rows, false);
    check(output, queries, at::kFloat, "partial values");
    TORCH_CHECK(output.dim() == 3 && output.size(0) == rows && output.size(1) == HEADS && output.size(2) == DIM,
                "partial values require [rows,12,256]");
    format(key_bits, key_codec);
    format(value_bits, value_codec);
    TORCH_CHECK(key_bits == value_bits && key_codec == value_codec,
                "paged attention requires one homogeneous Q8 or compact Q4 key/value profile");
    for (const auto* input : {&queries, &selected, &table, &global, &layer}) distinct(output, *input);
    c10::cuda::CUDAGuard guard(queries.device());
    sm70_paged_values_cuda(queries, selected, table, global, output, group, page_tokens, layer,
                          key_bits == 4 ? key_codec : 0, value_bits == 4 ? value_codec : 0,
                          attention_scale(scale));
}

void merge_values(const T& gathered, const T& table, T output, int rank, int page_tokens,
                  const T& layer, int value_bits, int value_codec) {
    check(gathered, gathered, at::kFloat, "gathered partial values");
    const auto* properties = at::cuda::getDeviceProperties(gathered.get_device());
    TORCH_CHECK(properties->major == 7 && properties->minor == 0, "paged owner attention requires SM70");
    TORCH_CHECK(gathered.dim() == 4 && gathered.size(0) == WORLD && gathered.size(1) <= INT32_MAX
                && gathered.size(2) == HEADS && gathered.size(3) == DIM,
                "gathered partial values require [8,rows,12,256]");
    TORCH_CHECK(rank >= 0 && rank < WORLD, "local query rank must lie in0..7");
    const int rows = int(gathered.size(1));
    controls(table, gathered, rows, rank / 4 * 4, page_tokens, layer);
    check(output, gathered, at::kHalf, "local attention output");
    TORCH_CHECK(output.dim() == 3 && output.size(0) == rows && output.size(1) == 3 && output.size(2) == DIM,
                "local attention output requires [rows,3,256]");
    format(value_bits, value_codec);
    distinct(output, gathered);
    distinct(output, table);
    distinct(output, layer);
    c10::cuda::CUDAGuard guard(gathered.device());
    sm70_paged_merge_values_cuda(gathered, table, output, rank, layer, value_bits == 4 ? value_codec : 0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    pybind11::class_<StorageBindings>(module, "StorageBindings")
        .def("matches", &StorageBindings::matches)
        .def_property_readonly("checks", &StorageBindings::checks)
        .def_property_readonly("overflowed", &StorageBindings::overflowed);
    module.def("storage_bindings", [](const pybind11::tuple& tensors) -> pybind11::object {
        if (!storage_supported(tensors)) return pybind11::none();
        return pybind11::cast(std::make_unique<StorageBindings>(tensors));
    });
    module.attr("storage_bindings_host_bytes_per_state") = STORAGE_HOST_BYTES;
    module.attr("descriptor_abi") = tf_sm70_paged::ABI_VERSION;
    module.attr("descriptor_words") = int(tf_sm70_paged::WORD_COUNT);
    module.attr("descriptor_schema_id") = tf_sm70_paged::SCHEMA_ID;
    module.def("local_statistics", &local_statistics);
    module.def("merge_statistics", &merge_statistics);
    module.def("normalized_partial", &normalized_partial);
    module.def("merge_values", &merge_values);
}
