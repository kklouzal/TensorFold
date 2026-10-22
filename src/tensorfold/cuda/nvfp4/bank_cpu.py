"""CPU-only native byte packing; no Tensor pointers or floating arithmetic cross the ABI."""

import hashlib
from functools import lru_cache
from pathlib import Path

import torch

from tensorfold.cuda.tensor_file import _regular_stream


@lru_cache(maxsize=1)
def _ext():
    # Follow the existing native SSD reader's CPU extension facility. In
    # particular, CUDA build discovery must not run in a CPU loading worker.
    from torch.utils.cpp_extension import load

    source = Path(__file__).with_suffix(".cpp")
    with _regular_stream(source) as stream:
        identity = hashlib.sha256(stream.read()).hexdigest()[:16]
    return load(name="tf_a4_bank_cpu_" + identity, sources=[str(source)],
                extra_cflags=["-O3", "-std=c++17", "-Wall", "-Wextra", "-Werror"], with_cuda=False, verbose=False)


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
    immutable inputs for this synchronous interval. Flat whole-storage exports
    make native buffer lengths physical bounds even for strided cache rows.
    Native code independently checks all offsets, lengths and snapshotted IDs
    before writing. Version counters are advanced before the raw write interval,
    including a failed interval; the model registers authority only on success.
    """
    native = _ext()
    up_storage = up.as_strided((up.untyped_storage().nbytes(),), (1,), storage_offset=0)
    down_storage = down.as_strided((down.untyped_storage().nbytes(),), (1,), storage_offset=0)
    arrays = (_array(codes), _array(scales), _array(up_storage), _array(down_storage), _array(indices))
    torch.autograd.graph.increment_version((up, down))
    native.pack(*arrays, up.storage_offset(), up.stride(0), down.storage_offset(), down.stride(0),
                up.shape[0], width, dims)


def _pack_part(native, payload, up, down, indices, width, dims):
    """Own all four exports through one synchronous unpublished write phase.

    No ndarray, DLPack alias, whole-storage view or buffer export escapes this
    call. In particular an SF caller can retire its raw owner on return before
    acquiring a larger code window. A failed phase remains unpublished, with
    the same pre-write original version update as the combined operation.
    """
    up_storage = up.as_strided((up.untyped_storage().nbytes(),), (1,), storage_offset=0)
    down_storage = down.as_strided((down.untyped_storage().nbytes(),), (1,), storage_offset=0)
    arrays = (_array(payload), _array(up_storage), _array(down_storage), _array(indices))
    torch.autograd.graph.increment_version((up, down))
    native(*arrays, up.storage_offset(), up.stride(0), down.storage_offset(), down.stride(0),
           up.shape[0], width, dims)


def pack_scales(scales, up, down, indices, width, dims):
    """Write only admitted SF tails; leave all code prefixes untouched."""
    _pack_part(_ext().pack_scales, scales, up, down, indices, width, dims)


def pack_words(codes, up, down, indices, width, dims):
    """Write only admitted code prefixes after the private SF phase retires."""
    _pack_part(_ext().pack_words, codes, up, down, indices, width, dims)
