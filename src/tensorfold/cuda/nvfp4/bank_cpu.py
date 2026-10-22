"""CPU-only native byte packing; no Tensor pointers or floating arithmetic cross the ABI."""

import hashlib
from functools import lru_cache
from pathlib import Path

import torch

from tensorfold.cuda.tensor_file import _regular_stream


@lru_cache(maxsize=1)
def _ext():
    # Follow the SSD reader's CPU extension facility without project CUDA
    # discovery or compilation. Upstream may inspect its own toolkit metadata.
    from torch.utils.cpp_extension import load

    source = Path(__file__).with_suffix(".cpp")
    with _regular_stream(source) as stream:
        identity = hashlib.sha256(stream.read()).hexdigest()[:16]
    # Enable the adopted header's OpenMP branch when Torch selects that backend.
    # The normal CPU loader already links Torch/C10. Do not add an OpenMP link
    # flag that could import another runtime; actual provider admission follows
    # the protected SDK gate before any parallel-support or performance claim.
    return load(name="tf_a4_bank_cpu_" + identity, sources=[str(source)],
                extra_cflags=["-O3", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-fopenmp"],
                with_cuda=False, verbose=False)


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
