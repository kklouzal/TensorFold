/* Project-owned POSIX descriptor ownership, CPython Stable ABI >= 3.11.
 * A borrowed fileno must never be closed/reassigned externally, and users must
 * finish all I/O before close. Linux/XNU release an ordinary owned descriptor
 * before reporting close errors; a consumed numeric FD is never retried.
 */
#ifndef Py_LIMITED_API
#define Py_LIMITED_API 0x030B0000
#endif
#if Py_LIMITED_API != 0x030B0000
#error "tensorfold._fd_owner must use the CPython 3.11 Limited API"
#endif
#if !defined(__linux__) && !defined(__APPLE__)
#error "tensorfold._fd_owner targets the declared Linux and macOS POSIX runtimes"
#endif
#include <Python.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <string.h>
#include <unistd.h>
#ifndef O_CLOEXEC
#error "tensorfold._fd_owner requires atomic non-inheritable open"
#endif
#ifndef TENSORFOLD_FD_OWNER_SOURCE_SHA256
#error "tensorfold._fd_owner requires its maintained source identity from setup.py"
#endif

typedef struct {
    PyObject_HEAD
    int fd;
    int state;
} OwnedFD;

enum {
    OWNER_FRESH = 0,
    OWNER_ACQUIRING = 1,
    OWNER_USED = 2
};

/* The GIL protects the one-way state transition. The syscall can release the
 * GIL after retirement; another close/fileno cannot observe a live owner then.
 * No Python bytecode or foreign callback separates retirement from close.
 */
static int
consume_fd(OwnedFD *self)
{
    int descriptor = self->fd;
    int result, saved_errno;
    if (descriptor < 0) {
        return 0;
    }
    self->fd = -1;
    Py_BEGIN_ALLOW_THREADS
    result = close(descriptor);
    saved_errno = errno;
    Py_END_ALLOW_THREADS
    if (result < 0) {
        errno = saved_errno;
        return -1;
    }
    return 0;
}

static void
owned_dealloc(PyObject *object)
{
    OwnedFD *self = (OwnedFD *)object;
    PyObject *type = (PyObject *)Py_TYPE(object);
    PyObject *error_type, *error_value, *error_traceback;
    int saved_errno = errno;
    /* Explicit operation journals own every acquired descriptor before open.
     * Destruction is a misuse backstop only, never operation error transport.
     * Its unraisable diagnostic preserves an already pending exception.
     */
    PyErr_Fetch(&error_type, &error_value, &error_traceback);
    if (consume_fd(self) < 0) {
        PyErr_SetFromErrno(PyExc_OSError);
        PyErr_WriteUnraisable(NULL);
    }
    PyObject_Free(object);
    Py_DECREF(type);
    PyErr_Restore(error_type, error_value, error_traceback);
    errno = saved_errno;
}

static PyObject *
owned_new(PyTypeObject *type, PyObject *args, PyObject *keywords)
{
    static char *names[] = {NULL};
    OwnedFD *self;
    if (!PyArg_ParseTupleAndKeywords(args, keywords, ":OwnedFD", names)) {
        return NULL;
    }
    self = (OwnedFD *)PyType_GenericAlloc(type, 0);
    if (self == NULL) {
        return NULL;
    }
    self->fd = -1;
    self->state = OWNER_FRESH;
    return (PyObject *)self;
}

static PyObject *
owned_open(PyObject *object, PyObject *args, PyObject *keywords)
{
    static char *names[] = {"path", "flags", "mode", "dir_fd", NULL};
    PyObject *path, *converted = NULL, *bytes = NULL, *parent = Py_None;
    char *name;
    Py_ssize_t length;
    int flags, mode = 0600, directory_fd = AT_FDCWD, descriptor, saved_errno;
    OwnedFD *self = (OwnedFD *)object;
    if (self->state != OWNER_FRESH) {
        PyErr_SetString(PyExc_RuntimeError, self->state == OWNER_ACQUIRING
                        ? "descriptor acquisition is already in progress"
                        : "a successfully acquired descriptor owner cannot reopen");
        return NULL;
    }
    /* Protect the already journaled slot from callback reentry and concurrent
     * open/close while the syscall releases the GIL. All state uses the GIL.
     */
    self->state = OWNER_ACQUIRING;
    if (!PyArg_ParseTupleAndKeywords(args, keywords, "Oi|i$O:open", names, &path, &flags, &mode, &parent)) {
        goto failed;
    }
    if (parent != Py_None) {
        long value = PyLong_AsLong(parent);
        if (value == -1 && PyErr_Occurred()) {
            goto failed;
        }
        if (value < INT_MIN || value > INT_MAX) {
            PyErr_SetString(PyExc_OverflowError, "dir_fd exceeds the native descriptor integer range");
            goto failed;
        }
        directory_fd = (int)value;
    }
    if (mode < 0 || mode > 07777) {
        PyErr_SetString(PyExc_ValueError, "mode must be POSIX permission bits in 0..0o7777");
        goto failed;
    }
    if (!PyUnicode_FSConverter(path, &converted)) {
        goto failed;
    }
    if (PyBytes_AsStringAndSize(converted, &name, &length) < 0) {
        goto failed;
    }
    if (memchr(name, '\0', (size_t)length) != NULL) {
        PyErr_SetString(PyExc_ValueError, "embedded null byte in descriptor path");
        goto failed;
    }
    /* Normalize to exact bytes before acquisition, so releasing a bytes
     * subclass or its path protocol cannot run a foreign callback afterward.
     */
    bytes = PyBytes_FromStringAndSize(name, length);
    Py_CLEAR(converted);
    if (bytes == NULL) {
        goto failed;
    }
    name = PyBytes_AsString(bytes);
    if (name == NULL) {
        goto failed;
    }
    /* This valid closed owner was published by the caller before entry.
     * Reacquiring the GIL does not execute Python; publish descriptor ownership
     * under that lock before any Python operation or return can be observed.
     */
    Py_BEGIN_ALLOW_THREADS
    descriptor = openat(directory_fd, name, flags | O_CLOEXEC, (mode_t)mode);
    saved_errno = errno;
    Py_END_ALLOW_THREADS
    if (descriptor < 0) {
        errno = saved_errno;
        PyErr_SetFromErrnoWithFilenameObject(PyExc_OSError, bytes);
        goto failed;
    }
    self->fd = descriptor;
    self->state = OWNER_USED;
    Py_DECREF(bytes);
    Py_RETURN_NONE;

failed:
    Py_XDECREF(converted);
    Py_XDECREF(bytes);
    self->state = OWNER_FRESH;
    return NULL;
}

static PyObject *
owned_close(PyObject *object, PyObject *unused)
{
    (void)unused;
    if (((OwnedFD *)object)->state == OWNER_ACQUIRING) {
        PyErr_SetString(PyExc_RuntimeError, "await descriptor acquisition before closing its owner");
        return NULL;
    }
    if (consume_fd((OwnedFD *)object) < 0) {
        return PyErr_SetFromErrno(PyExc_OSError);
    }
    Py_RETURN_NONE;
}

static PyObject *
owned_fileno(PyObject *object, PyObject *unused)
{
    int descriptor = ((OwnedFD *)object)->fd;
    (void)unused;
    if (descriptor < 0) {
        PyErr_SetString(PyExc_ValueError, "the descriptor owner is closed");
        return NULL;
    }
    return PyLong_FromLong(descriptor);
}

static PyObject *
owned_closed(PyObject *object, void *closure)
{
    (void)closure;
    return PyBool_FromLong(((OwnedFD *)object)->fd < 0);
}

static PyMethodDef owned_methods[] = {
    /* The keyword flag selects CPython's documented three-argument dispatcher;
     * the intermediate generic function pointer avoids GCC's arity warning.
     */
    {"open", (PyCFunction)(void (*)(void))owned_open, METH_VARARGS | METH_KEYWORDS,
     "Acquire once into this already journaled closed slot; report acquisition failures explicitly."},
    {"close", owned_close, METH_NOARGS, "Consume ownership once; report close errors without retrying a numeric FD."},
    {"fileno", owned_fileno, METH_NOARGS, "Borrow the open FD; caller must finish all uses before owner.close()."},
    {NULL, NULL, 0, NULL}
};
static PyGetSetDef owned_properties[] = {
    {"closed", owned_closed, NULL, "True without a published descriptor; observe pending open/close completion separately.", NULL},
    {NULL, NULL, NULL, NULL, NULL}
};
static PyType_Slot owned_slots[] = {
    {Py_tp_new, owned_new},
    {Py_tp_dealloc, owned_dealloc},
    {Py_tp_methods, owned_methods},
    {Py_tp_getset, owned_properties},
    {Py_tp_doc, "OwnedFD(): a valid closed POSIX descriptor slot. Journal it before .open(path, flags, mode=0o600, *, dir_fd=None).\n"
                "A successful acquisition is one-shot; failed acquisition leaves the slot retryable.\n"
                "Explicit close is required; fileno is borrowed and cannot be externally closed.\n"
                "A supplied dir_fd is borrowed through openat completion.\n"
                "Observe open/close completion and finish borrowed I/O before retirement.\n"
                "Path/argument callbacks run before acquisition; open/close reentry during acquisition is refused."},
    {0, NULL}
};
static PyType_Spec owned_spec = {
    "tensorfold._fd_owner.OwnedFD", sizeof(OwnedFD), 0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE, owned_slots
};

static int
module_exec(PyObject *module)
{
    /* Each module owns its heap type; no mutable process-global type or FD.
     * No instance dict, subclasses, weakrefs or owned Python references can
     * create an owner cycle or intercept native retirement.
     */
    PyObject *type = PyType_FromSpec(&owned_spec);
    int result;
    if (type == NULL) {
        return -1;
    }
    result = PyModule_AddObjectRef(module, "OwnedFD", type);
    Py_DECREF(type);
    if (result < 0
        || PyModule_AddIntConstant(module, "_ownership_version", 2) < 0
        || PyModule_AddIntConstant(module, "_limited_api", Py_LIMITED_API) < 0
        || PyModule_AddStringConstant(module, "_source_sha256", TENSORFOLD_FD_OWNER_SOURCE_SHA256) < 0) {
        return -1;
    }
    return result;
}
static PyModuleDef_Slot module_slots[] = {
    {Py_mod_exec, module_exec}, {0, NULL}
};
static struct PyModuleDef module_definition = {
    PyModuleDef_HEAD_INIT, "_fd_owner",
    "Native consumed-state descriptor ownership for declared Linux/macOS CPython runtimes.",
    0, NULL, module_slots, NULL, NULL, NULL
};
PyMODINIT_FUNC
PyInit__fd_owner(void)
{
    return PyModuleDef_Init(&module_definition);
}
