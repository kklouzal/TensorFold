"""CPU byte packing through the public producer interface; no Tensor C++ ABI."""

import hashlib
from functools import lru_cache
from pathlib import Path

import torch

from tensorfold.cuda.tensor_file import _regular_stream


def _load(filename, headers=()):
    # Follow the existing native SSD reader's CPU extension facility. In
    # particular, CUDA build discovery must not run in a CPU loading worker.
    from torch.utils.cpp_extension import load

    source = Path(__file__).with_name(filename)
    if headers:
        digest = hashlib.sha256()
        for path in (source, *(source.with_name(name) for name in headers)):
            with _regular_stream(path) as stream:
                digest.update(path.name.encode() + b"\0")
                digest.update(stream.read())
        identity = digest.hexdigest()[:16]
    else:
        # Preserve the original required NumPy-provider recipe/cache identity.
        with _regular_stream(source) as stream:
            identity = hashlib.sha256(stream.read()).hexdigest()[:16]
    return load(name="tf_a4_bank_cpu_" + identity, sources=[str(source)],
                extra_cflags=["-O3", "-std=c++17", "-Wall", "-Wextra", "-Werror"], with_cuda=False, verbose=False)


@lru_cache(maxsize=1)
def _producer():
    absent = object()
    capsule = getattr(torch.Tensor, "__dlpack_c_exchange_api__", absent)
    if capsule is absent:
        return None
    # A declared malformed/version-mismatched API must fail native preparation;
    # it is not absence and never selects the older provider.
    return torch.Tensor, capsule, torch.autograd.graph.increment_version


@lru_cache(maxsize=1)
def _numpy_ext():
    return _load("bank_cpu.cpp")


@lru_cache(maxsize=1)
def _ext():
    producer = _producer()
    if producer is None:
        raise RuntimeError("Tensor does not advertise the native C-exchange API")
    module = _load("bank_exchange.cpp", ("bank_exchange_core.h", "bank_exchange_kernel.h"))
    return module.prepare(*producer)


@lru_cache(maxsize=1)
def _pybind_ext():
    """Independent front-end candidate for qualification; never a request flag."""
    producer = _producer()
    if producer is None:
        raise RuntimeError("Tensor does not advertise the native C-exchange API")
    module = _load("bank_exchange_pybind.cpp", ("bank_exchange_core.h", "bank_exchange_kernel.h"))
    return module.prepare(*producer, module._pack, module._Context, module)


def _native_module(prepared):
    """Owning extension of our prepared entry, for cold provider diagnostics.

    Only results of the two factories above are admitted. This is not a
    general callable resolver; pack performs no lookup through this interface.
    Both immutable contexts strongly retain the actual loaded module.
    """
    context = prepared.__self__
    return context[3] if type(context) is tuple else context.module_owner


def _array(tensor):
    """Borrow through a new DLPack Storage, preserving the original resize flag.

    NumPy marks its Tensor base's Storage non-resizable. DLPack import creates
    a distinct from_blob Storage whose deleter retains the original Tensor;
    NumPy therefore changes only that private alias's capacity contract. The
    ndarray owns the complete alias/managed-Tensor chain through the native call.
    """
    return torch.utils.dlpack.from_dlpack(tensor).numpy()


def pack(codes, scales, up, down, indices, width, dims):
    """Mutate exclusively owned final authority; a failed call remains unpublished.

    The caller proves CPU U8 geometry, disjoint source/destination cells and
    immutable inputs for this synchronous interval. Direct descriptors bound
    writes by the original valid strided Tensor index sets; the other provider
    retains its whole-storage exports. Native code checks geometry, overlap and
    snapshotted IDs before writing. Version counters advance before the write interval,
    including a failed interval; the model registers authority only on success.
    """
    producer = _producer()
    if (producer is not None and type(width) is int and type(dims) is int
            and all(type(tensor) is producer[0] for tensor in (codes, scales, up, down, indices))):
        _ext()(codes, scales, up, down, indices, width, dims)
        return
    # API-absent providers, Tensor subclasses and int-like geometry retain their
    # existing protocol/conversion semantics. The direct borrowed descriptors
    # cannot permit __index__ reentry after export. Select before compilation.
    native = _numpy_ext()
    up_storage = up.as_strided((up.untyped_storage().nbytes(),), (1,), storage_offset=0)
    down_storage = down.as_strided((down.untyped_storage().nbytes(),), (1,), storage_offset=0)
    arrays = (_array(codes), _array(scales), _array(up_storage), _array(down_storage), _array(indices))
    torch.autograd.graph.increment_version((up, down))
    native.pack(*arrays, up.storage_offset(), up.stride(0), down.storage_offset(), down.stride(0),
                up.shape[0], width, dims)
