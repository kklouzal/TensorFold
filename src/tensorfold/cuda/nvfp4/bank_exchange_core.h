// Temporary public DLPack C exchange over strongly retained original Tensors.
#pragma once
#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif
#include <Python.h>
#include <ATen/dlpack.h>
#include <array>
#include <cstdint>
#include <exception>
#include <limits>
#include "bank_exchange_kernel.h"

#if DLPACK_MAJOR_VERSION != 1 || DLPACK_MINOR_VERSION < 3
#error "The advertised Tensor C exchange requires DLPack1.3 headers"
#endif

namespace a4_bank_exchange {
using U = std::uint64_t;
struct Failure { PyObject* kind; const char* message; };
struct PythonFailure {};
struct Prepared {
    PyObject* tensor_type;
    const DLPackExchangeAPI* api;
    PyObject* increment;
};

static void require(bool valid, const char* message) {
    if (!valid) throw Failure{PyExc_ValueError, message};
}

template <class Function>
static PyObject* translate(Function&& function) noexcept {
    try {
        return function();
    } catch (PythonFailure&) {
        return nullptr; // the adopted producer/callback owns the current Python error
    } catch (const Failure& error) {
        PyErr_SetString(error.kind, error.message);
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "native A4 exchange raised an unknown C++ exception");
    }
    return nullptr;
}

static const DLPackExchangeAPI* api_pointer(PyObject* capsule) {
    auto* api = static_cast<const DLPackExchangeAPI*>(PyCapsule_GetPointer(capsule, "dlpack_exchange_api"));
    if (!api) throw PythonFailure{};
    return api;
}

static PyObject* context(PyObject* tensor_type, PyObject* capsule, PyObject* increment, PyObject* module) {
    // Internal callback contract: update version counters only. It may raise,
    // but must not resize/rebind/mutate any exported Tensor, Storage, data,
    // shape or stride during reentry. Strong references alone do not prove it.
    require(PyType_Check(tensor_type), "native A4 exchange needs the actual Tensor type");
    require(PyCapsule_IsValid(capsule, "dlpack_exchange_api"), "native A4 exchange capsule is invalid");
    const auto* api = api_pointer(capsule);
    require(api->header.version.major == 1 && api->header.version.minor >= 3
            && api->dltensor_from_py_object_no_sync,
            "native A4 exchange needs the public DLPack1.3 non-owning callback");
    require(PyCallable_Check(increment), "native A4 exchange needs the version callback");
    require(PyModule_Check(module), "native A4 exchange needs its owning extension module");
    auto* declared = PyObject_GetAttrString(tensor_type, "__dlpack_c_exchange_api__");
    if (!declared) throw PythonFailure{};
    const bool matches = declared == capsule;
    Py_DECREF(declared);
    require(matches, "native A4 exchange capsule does not belong to its Tensor type");
    // Keep the actual loaded module alive with its prepared entry. The fourth
    // tuple field is its cold diagnostic origin; no module lookup runs in pack.
    // Standard GC sees every owner/cycle; there is no opaque C++ capture/global.
    auto* result = PyTuple_Pack(4, tensor_type, capsule, increment, module);
    if (!result) throw PythonFailure{};
    return result;
}

struct Owners {
    std::array<PyObject*, 5> values;
    explicit Owners(PyObject* const* arguments) {
        for (unsigned i = 0; i < 5; ++i) {
            values[i] = arguments[i];
            Py_INCREF(values[i]);
        }
    }
    Owners(const Owners&) = delete;
    ~Owners() {
        // The call's borrowed arguments are still live, so these decrefs cannot
        // finalize an original owner. Destruction always follows GIL restoration.
        for (auto* value : values) Py_DECREF(value);
    }
};

struct ReleaseGIL {
    PyThreadState* state;
    ReleaseGIL() : state(PyEval_SaveThread()) {}
    ReleaseGIL(const ReleaseGIL&) = delete;
    ~ReleaseGIL() { PyEval_RestoreThread(state); }
};

static U bounded(U left, U right) {
    require(!right || left <= static_cast<U>(PY_SSIZE_T_MAX) / right,
            "native A4 byte geometry exceeds buffer size limits");
    return left * right;
}

static U positive(std::int64_t value) {
    require(value > 0, "native A4 byte geometry must be positive");
    return static_cast<U>(value);
}

static U integer(PyObject* value) {
    const auto number = PyLong_AsSsize_t(value);
    if (number == -1 && PyErr_Occurred()) throw PythonFailure{};
    return positive(number);
}

struct Span { std::uintptr_t first, last; };
static Span span(const void* pointer, U size) {
    const auto first = reinterpret_cast<std::uintptr_t>(pointer);
    require(size <= std::numeric_limits<std::uintptr_t>::max() - first,
            "native A4 buffer address exceeds address limits");
    return {first, first + size};
}
static bool overlap(Span left, Span right) { return left.first < right.last && right.first < left.last; }

static void tensor(const DLTensor& value, int rank, int code, int bits, const char* message) {
    require(value.device.device_type == kDLCPU && value.ndim == rank && value.shape && value.strides
            && value.dtype.code == code && value.dtype.bits == bits && value.dtype.lanes == 1
            && value.byte_offset == 0 && value.data, message);
}

static U extent(U stride, U rows, U row_bytes) {
    const U tail = bounded(rows - 1, stride);
    require(row_bytes <= static_cast<U>(PY_SSIZE_T_MAX) - tail,
            "native A4 destination exceeds its physical buffer");
    return tail + row_bytes;
}

static Prepared tuple_context(PyObject* prepared) {
    return {PyTuple_GET_ITEM(prepared, 0), api_pointer(PyTuple_GET_ITEM(prepared, 1)), PyTuple_GET_ITEM(prepared, 2)};
}

static PyObject* pack(const Prepared& prepared, PyObject* const* arguments) {
    // Integer conversion must not invoke __index__ after borrowed descriptors
    // have been exported. Other int-like inputs retain the original provider.
    if (!PyLong_CheckExact(arguments[5]) || !PyLong_CheckExact(arguments[6]))
        throw Failure{PyExc_TypeError, "native A4 exchange requires exact integer width and dims"};
    auto* expected_type = reinterpret_cast<PyTypeObject*>(prepared.tensor_type);
    const auto* api = prepared.api;
    for (unsigned i = 0; i < 5; ++i) {
        if (Py_TYPE(arguments[i]) != expected_type)
            throw Failure{PyExc_TypeError, "native A4 exchange accepts only its prepared exact Tensor type"};
    }
    Owners owners(arguments);
    std::array<DLTensor, 5> views{};
    // Export all five first. No pointer is read on a failed producer status.
    for (unsigned i = 0; i < 5; ++i) {
        const int status = api->dltensor_from_py_object_no_sync(owners.values[i], &views[i]);
        if (status != 0 || PyErr_Occurred()) {
            if (!PyErr_Occurred())
                PyErr_SetString(PyExc_RuntimeError, "native A4 exchange failed without a Python exception");
            throw PythonFailure{};
        }
    }
    // Default increment_version changes no metadata/data. Both ORIGINAL target
    // objects are passed, including two increments when their counter is shared.
    auto* targets = PyTuple_Pack(2, arguments[2], arguments[3]);
    if (!targets) throw PythonFailure{};
    auto* result = PyObject_CallOneArg(prepared.increment, targets);
    Py_DECREF(targets);
    if (!result) throw PythonFailure{};
    Py_DECREF(result);

    // Exact builtin integer conversion cannot reenter Python. Preserve the
    // original conversion/native-validation position after the version step.
    const U width = integer(arguments[5]), dims = integer(arguments[6]);
    const auto& codes = views[0]; const auto& scales = views[1];
    const auto& up = views[2]; const auto& down = views[3]; const auto& indices = views[4];
    tensor(codes, 1, kDLUInt, 8, "native A4 buffers must be contiguous U8 vectors with writable destinations");
    tensor(scales, 1, kDLUInt, 8, "native A4 buffers must be contiguous U8 vectors with writable destinations");
    tensor(up, 3, kDLUInt, 8, "native A4 destination Tensor geometry is invalid");
    tensor(down, 2, kDLUInt, 8, "native A4 destination Tensor geometry is invalid");
    tensor(indices, 1, kDLInt, 64, "native A4 local indices must be a contiguous int64 vector");
    const U experts = positive(up.shape[0]), count = positive(indices.shape[0]);
    require(experts <= 1023 && count <= experts && width < (U{1} << 31) && dims < (U{1} << 31)
            && width % 64 == 0 && dims % 64 == 0,
            "native A4 bank geometry or local count is invalid");
    const U words = bounded(width, dims) / 2, sf = words / 8, blob = words + sf;
    const U up_stride = positive(up.strides[0]), down_stride = positive(down.strides[0]);
    require(codes.strides[0] == 1 && scales.strides[0] == 1 && indices.strides[0] == 1
            && positive(codes.shape[0]) == bounded(count, bounded(3, words))
            && positive(scales.shape[0]) == bounded(count, bounded(3, sf))
            && down.shape[0] == up.shape[0] && up.shape[1] == 2 && positive(up.shape[2]) == blob
            && positive(down.shape[1]) == blob && positive(up.strides[1]) == blob
            && up.strides[2] == 1 && down.strides[1] == 1
            && up_stride >= bounded(2, blob) && down_stride >= blob,
            "native A4 source lengths or destination row strides differ from geometry");
    auto* up_data = static_cast<std::uint8_t*>(up.data);
    auto* down_data = static_cast<std::uint8_t*>(down.data);
    // Every write is a member of the original valid Tensor's strided index set.
    // numel is not treated as physical capacity; offsets are already in data.
    const Span up_span = span(up_data, extent(up_stride, experts, bounded(2, blob)));
    const Span down_span = span(down_data, extent(down_stride, experts, blob));
    require((up_span.first | down_span.first) % 4 == 0,
            "native A4 destinations require four-byte alignment");
    if (overlap(up_span, down_span)) {
        const U delta = down_span.first >= up_span.first ? down_span.first - up_span.first : 0;
        require(up_stride == down_stride && delta >= 2 * blob && delta <= up_stride && blob <= up_stride - delta,
                "native A4 destination cells overlap");
    }
    const Span code_span = span(codes.data, positive(codes.shape[0]));
    const Span scale_span = span(scales.data, positive(scales.shape[0]));
    const Span index_span = span(indices.data, bounded(count, sizeof(std::int64_t)));
    require(!overlap(up_span, code_span) && !overlap(up_span, scale_span) && !overlap(up_span, index_span)
            && !overlap(down_span, code_span) && !overlap(down_span, scale_span) && !overlap(down_span, index_span),
            "native A4 destinations overlap immutable inputs");
    std::array<std::int64_t, 1023> local;
    std::memcpy(local.data(), indices.data, bounded(count, sizeof(std::int64_t)));
    for (U e = 0; e < count; ++e) {
        require(local[e] >= 0 && static_cast<U>(local[e]) < experts,
                "native A4 local index is outside the authority");
        for (U previous = 0; previous < e; ++previous)
            require(local[e] != local[previous], "native A4 local indices must be distinct");
    }
    {
        ReleaseGIL release;
        a4_bank_kernel::copy_bank(static_cast<const std::uint8_t*>(codes.data),
                                 static_cast<const std::uint8_t*>(scales.data), up_data, down_data, local.data(),
                                 count, width, dims, words, sf, blob, up_stride, down_stride);
    }
    Py_RETURN_NONE;
}

#ifndef TF_BANK_PYBIND_FRONTEND
static PyObject* direct_pack(PyObject* prepared, PyObject* const* arguments, Py_ssize_t count) {
    return translate([&]() -> PyObject* {
        if (count != 7) throw Failure{PyExc_TypeError, "native A4 exchange requires five Tensors, width and dims"};
        return pack(tuple_context(prepared), arguments);
    });
}

// CPython dispatch casts this erased C function pointer back to METH_FASTCALL's
// exact declared signature before invocation; the intermediate avoids GCC's
// incompatible-call warning without calling through the erased signature.
static PyMethodDef direct_method = {"pack", reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)()>(direct_pack)), METH_FASTCALL,
                                  "Synchronous CPU A4 packing over original Tensor descriptors."};

static PyObject* prepare(PyObject* module, PyObject* arguments) {
    return translate([&]() -> PyObject* {
        PyObject *type, *capsule, *increment;
        if (!PyArg_UnpackTuple(arguments, "prepare", 3, 3, &type, &capsule, &increment)) throw PythonFailure{};
        auto* prepared = context(type, capsule, increment, module);
        auto* name = PyModule_GetNameObject(module);
        if (!name) { Py_DECREF(prepared); throw PythonFailure{}; }
        auto* function = PyCFunction_NewEx(&direct_method, prepared, name);
        Py_DECREF(name);
        Py_DECREF(prepared);
        return function;
    });
}
#endif
} // namespace a4_bank_exchange
