// Investigation comparator: standard bound-method GC owns the shared context.
#define PY_SSIZE_T_CLEAN
#define TF_BANK_PYBIND_FRONTEND
#include <pybind11/pybind11.h>
#include "bank_exchange_core.h"

namespace py = pybind11;

struct Context {
    PyObject_HEAD
    PyObject* owners;
    a4_bank_exchange::Prepared prepared;
};

static int traverse_context(PyObject* object, visitproc visit, void* arg) {
    auto* self = reinterpret_cast<Context*>(object);
    Py_VISIT(Py_TYPE(object));
    Py_VISIT(self->owners);
    return 0;
}
static int clear_context(PyObject* object) {
    auto* self = reinterpret_cast<Context*>(object);
    self->prepared = {};
    Py_CLEAR(self->owners);
    return 0;
}
static void delete_context(PyObject* object) {
    PyObject_GC_UnTrack(object);
    clear_context(object);
    auto* type = Py_TYPE(object);
    type->tp_free(object);
    Py_DECREF(type);
}
static PyObject* no_context_constructor(PyTypeObject*, PyObject*, PyObject*) {
    PyErr_SetString(PyExc_TypeError, "native A4 exchange contexts are created only by prepare");
    return nullptr;
}

static PyObject* context_module(PyObject* object, void*) {
    auto* self = reinterpret_cast<Context*>(object);
    if (!self->owners) {
        PyErr_SetString(PyExc_RuntimeError, "native A4 exchange context is retired");
        return nullptr;
    }
    return Py_NewRef(PyTuple_GET_ITEM(self->owners, 3));
}

static bool module_member(PyObject* names, const char* name, PyObject* expected) {
    // GetItemString suppresses allocation/equality failures. Keep those errors
    // distinct from an ordinary missing/mismatched cold preparation member.
    auto* key = PyUnicode_FromString(name);
    if (!key) throw a4_bank_exchange::PythonFailure{};
    auto* value = PyDict_GetItemWithError(names, key);
    Py_DECREF(key); // exact Unicode key cannot invoke user cleanup
    if (!value && PyErr_Occurred()) throw a4_bank_exchange::PythonFailure{};
    return value == expected;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    // Native context is immutable/non-instantiable from Python and fully GC-
    // traversed. A raw _pack call cannot forge an unchecked tuple/ABI pointer.
    static PyGetSetDef getters[] = {
        {"module_owner", context_module, nullptr, "Strong prepared extension owner; diagnostic use only.", nullptr},
        {nullptr, nullptr, nullptr, nullptr, nullptr}
    };
    static PyType_Slot slots[] = {
        {Py_tp_dealloc, reinterpret_cast<void*>(delete_context)},
        {Py_tp_traverse, reinterpret_cast<void*>(traverse_context)},
        {Py_tp_clear, reinterpret_cast<void*>(clear_context)},
        {Py_tp_new, reinterpret_cast<void*>(no_context_constructor)},
        {Py_tp_getset, getters},
        {0, nullptr}
    };
    static PyType_Spec spec = {"tensorfold._A4ExchangeContext", sizeof(Context), 0,
                              Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC, slots};
    auto* context_type = PyType_FromSpec(&spec);
    if (!context_type) throw py::error_already_set();
    module.attr("_Context") = py::reinterpret_steal<py::object>(context_type);
    module.def("_pack", [](const py::object& prepared,
                           const py::object& codes, const py::object& scales,
                           const py::object& up, const py::object& down, const py::object& indices,
                           const py::object& width, const py::object& dims) {
        PyObject* arguments[] = {codes.ptr(), scales.ptr(), up.ptr(), down.ptr(), indices.ptr(), width.ptr(), dims.ptr()};
        auto* result = a4_bank_exchange::translate([&]() {
            a4_bank_exchange::require(Py_TYPE(prepared.ptr())->tp_dealloc == delete_context,
                                      "native A4 exchange context has the wrong type");
            auto* value = reinterpret_cast<Context*>(prepared.ptr());
            a4_bank_exchange::require(value->owners != nullptr, "native A4 exchange context is retired");
            return a4_bank_exchange::pack(value->prepared, arguments);
        });
        if (!result) throw py::error_already_set();
        return py::reinterpret_steal<py::object>(result);
    });
    module.def("prepare", [](const py::object& type, const py::object& capsule,
                             const py::object& increment, const py::object& function, const py::type& context_type,
                             const py::module_& owner) {
        auto* result = a4_bank_exchange::translate([&]() -> PyObject* {
            auto* native_type = reinterpret_cast<PyTypeObject*>(context_type.ptr());
            a4_bank_exchange::require(native_type->tp_dealloc == delete_context
                                      && native_type->tp_basicsize == static_cast<Py_ssize_t>(sizeof(Context)) && PyCallable_Check(function.ptr()),
                                      "native A4 exchange binding type/function is invalid");
            auto* names = PyModule_GetDict(owner.ptr());
            if (!names) throw a4_bank_exchange::PythonFailure{};
            a4_bank_exchange::require(module_member(names, "_pack", function.ptr())
                                      && module_member(names, "_Context", context_type.ptr()),
                                      "native A4 exchange function/context do not belong to their module");
            auto* prepared = a4_bank_exchange::context(type.ptr(), capsule.ptr(), increment.ptr(), owner.ptr());
            auto* value = reinterpret_cast<Context*>(native_type->tp_alloc(native_type, 0));
            if (!value) { Py_DECREF(prepared); throw a4_bank_exchange::PythonFailure{}; }
            value->owners = prepared;
            try {
                value->prepared = a4_bank_exchange::tuple_context(prepared);
            } catch (...) {
                Py_DECREF(value);
                throw;
            }
            auto* method = PyMethod_New(function.ptr(), reinterpret_cast<PyObject*>(value));
            Py_DECREF(value);
            return method;
        });
        if (!result) throw py::error_already_set();
        return py::reinterpret_steal<py::object>(result);
    });
}
