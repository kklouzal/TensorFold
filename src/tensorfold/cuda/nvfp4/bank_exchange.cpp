// Minimal CPython boundary; no Tensor C++ ABI, NumPy or pybind dispatcher.
#include "bank_exchange_core.h"

#define TF_JOIN_INNER(left, right) left##right
#define TF_JOIN(left, right) TF_JOIN_INNER(left, right)
#define TF_STRING_INNER(value) #value
#define TF_STRING(value) TF_STRING_INNER(value)

static PyMethodDef methods[] = {
    {"prepare", a4_bank_exchange::prepare, METH_VARARGS,
     "Bind the producer and version-only callback; callbacks must not resize, rebind or mutate exported data/metadata."},
    {nullptr, nullptr, 0, nullptr}
};
static PyModuleDef definition = {
    PyModuleDef_HEAD_INIT, TF_STRING(TORCH_EXTENSION_NAME), nullptr, 0, methods,
    nullptr, nullptr, nullptr, nullptr
};

PyMODINIT_FUNC TF_JOIN(PyInit_, TORCH_EXTENSION_NAME)() {
    return PyModule_Create(&definition);
}
