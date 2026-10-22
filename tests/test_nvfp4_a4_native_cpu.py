"""Owned CPU byte-kernel integration, independent layout and buffer failures."""

import gc
import hashlib
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from test_nvfp4_a4_cache_control import CPUCache
from test_nvfp4_a4_host_control import oracle_blob

from tensorfold.cuda.nvfp4 import bank_cpu, experts_a4, host_layout


def inputs(width=64, dims=128, *, strided=False):
    count, selected = 4, [3, 1, 0]
    if strided:
        cache = CPUCache(width, dims)
        storage = torch.full((count, cache.entry_bytes), 0xa5, dtype=torch.uint8)
        expected_storage = storage.clone()
        fields, expected = cache.source_views(storage)[:2], cache.source_views(expected_storage)[:2]
    else:
        fields = tuple(torch.full(shape, 0xa5, dtype=torch.uint8)
                       for shape, _ in host_layout.authority_fields_a4(count, width, dims)[:2])
        expected = tuple(field.clone() for field in fields)
        storage, expected_storage = None, None
    generator = torch.Generator().manual_seed(8421 + width + dims)
    code, sf = [], []
    for local in selected:
        for projection, (n, k) in enumerate(((dims, width), (width, dims), (width, dims))):
            a = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=generator)
            b = torch.randint(0, 256, (n, k // 16), dtype=torch.uint8, generator=generator)
            b.reshape(-1)[:256] = torch.randperm(256, generator=generator).to(torch.uint8)
            code.append(a.reshape(-1))
            sf.append(b.reshape(-1))
            target = expected[1][local] if projection == 0 else expected[0][local, projection - 1]
            target.copy_(oracle_blob(a, b))
    return torch.cat(code), torch.cat(sf), fields, torch.tensor(selected), expected, storage, expected_storage


@pytest.mark.parametrize("width,dims", [(64, 64), (64, 128), (128, 128),
                                        (192, 64), (192, 320), (256, 384)])
@pytest.mark.parametrize("strided", [False, True])
def test_native_bytes_preserve_original_capacity_versions_padding_and_alias_lifetimes(width, dims, strided, monkeypatch):
    codes, scales, fields, indices, expected, storage, expected_storage = inputs(width, dims, strided=strided)
    tensors = (codes, scales, *fields, indices)
    snapshots = [(t.data_ptr(), t.untyped_storage().resizable(), t._version, t.clone()) for t in tensors]
    actual = bank_cpu._numpy_ext()
    aliases = []

    def pack(*args):
        # The borrowed ndarray owns the imported Tensor, whose managed deleter
        # owns the original backing even if Python collection runs in the call.
        arrays = args[:5]
        assert all(isinstance(array.base, torch.Tensor) for array in arrays)
        assert all(not array.base.untyped_storage().resizable() for array in arrays)
        aliases.extend(weakref.ref(array) for array in arrays)
        gc.collect()
        return actual.pack(*args)

    monkeypatch.setattr(bank_cpu, "_producer", lambda: None)
    monkeypatch.setattr(bank_cpu, "_numpy_ext", lambda: SimpleNamespace(pack=pack))
    experts_a4.pack_bank_a4(codes, scales, *fields, indices, width, dims)
    assert all(torch.equal(field, reference) for field, reference in zip(fields, expected, strict=True))
    if strided:
        assert torch.equal(storage, expected_storage)
    for position, (tensor, (pointer, resizable, version, before)) in enumerate(zip(tensors, snapshots, strict=True)):
        assert tensor.data_ptr() == pointer and tensor.untyped_storage().resizable() == resizable
        if position in (2, 3):
            assert tensor._version > version
        else:
            assert tensor._version == version and torch.equal(tensor, before)
    assert all(reference() is None for reference in aliases)


def test_native_pair_tail_copy_accepts_unaligned_sources_and_preserves_physical_margins():
    width, dims = 192, 320
    codes, scales, _, indices, expected, _, _ = inputs(width, dims)
    # Both projections have a paired prefix and odd tail. Raw input addresses
    # need no four-byte alignment; only destination cells have that contract.
    source_codes = np.full(codes.numel() + 1, 0x5a, dtype=np.uint8)
    source_scales = np.full(scales.numel() + 3, 0x5a, dtype=np.uint8)
    source_codes[1:] = codes.numpy()
    source_scales[3:] = scales.numpy()
    blob = width * dims * 9 // 16
    up_stride, down_stride = 2 * blob + 12, blob + 20
    up = np.full(4 * up_stride + 16, 0xa5, dtype=np.uint8)
    down = np.full(4 * down_stride + 24, 0xa5, dtype=np.uint8)
    expected_up, expected_down = up.copy(), down.copy()
    for local in range(4):
        expected_up[4 + local * up_stride:4 + local * up_stride + 2 * blob] = expected[0][local].numpy().reshape(-1)
        expected_down[8 + local * down_stride:8 + local * down_stride + blob] = expected[1][local].numpy()
    before = source_codes.copy(), source_scales.copy(), indices.clone()
    bank_cpu._numpy_ext().pack(source_codes[1:], source_scales[3:], up, down, indices.numpy(),
                         4, up_stride, 8, down_stride, 4, width, dims)
    assert np.array_equal(up, expected_up) and np.array_equal(down, expected_down)
    assert np.array_equal(source_codes, before[0]) and np.array_equal(source_scales, before[1])
    assert torch.equal(indices, before[2])


def native_arguments():
    # Plain buffers test our language boundary directly, independent of the
    # Python adapter. Four authority rows, one 64x64 physical expert chunk.
    blob = 64 * 64 * 9 // 16
    return [np.zeros(3 * 64 * 64 // 2, dtype=np.uint8), np.zeros(3 * 64 * 64 // 16, dtype=np.uint8),
            np.full(4 * 2 * blob, 0xa5, dtype=np.uint8), np.full(4 * blob, 0xa5, dtype=np.uint8),
            np.array([2], dtype=np.int64), 0, 2 * blob, 0, blob, 4, 64, 64]


@pytest.mark.parametrize("fault", ["negative", "large", "duplicate", "truncated", "readonly", "wrong_dtype",
                                 "offset", "stride", "unaligned", "input_alias", "map_alias", "dimension"])
def test_native_boundary_rejects_invalid_buffers_before_any_authority_write(fault):
    arguments = native_arguments()
    if fault == "negative":
        arguments[4][0] = -1
    elif fault == "large":
        arguments[4][0] = 4
    elif fault == "duplicate":
        arguments[0] = np.tile(arguments[0], 2)
        arguments[1] = np.tile(arguments[1], 2)
        arguments[4] = np.array([2, 2], dtype=np.int64)
    elif fault == "truncated":
        arguments[0] = arguments[0][:-1]
    elif fault == "readonly":
        arguments[2].flags.writeable = False
    elif fault == "wrong_dtype":
        arguments[2] = arguments[2].astype(np.int8)
    elif fault == "offset":
        arguments[5] = np.iinfo(np.int64).max
    elif fault == "stride":
        arguments[6] -= 1
    elif fault == "unaligned":
        arguments[2] = np.full(arguments[2].size + 1, 0xa5, dtype=np.uint8)[1:]
    elif fault == "input_alias":
        arguments[0] = arguments[2][:arguments[0].size]
    elif fault == "map_alias":
        arguments[4] = arguments[2][:8].view(np.int64)
    elif fault == "dimension":
        arguments[11] = 2**31
    before = [value.copy() for value in arguments[:5]]
    with pytest.raises((ValueError, TypeError)):
        bank_cpu._numpy_ext().pack(*arguments)
    assert all(np.array_equal(a, b) for a, b in zip(arguments[:5], before, strict=True))


def test_cpu_recipe_uses_owned_source_identity_without_cuda_discovery(monkeypatch):
    from torch.utils import cpp_extension

    recorded = {}

    def load(**kwargs):
        recorded.update(kwargs)
        return recorded

    monkeypatch.setattr(cpp_extension, "load", load)
    assert bank_cpu._numpy_ext.__wrapped__() is recorded
    source = Path(bank_cpu.__file__).with_suffix(".cpp")
    assert recorded == {"name": "tf_a4_bank_cpu_" + hashlib.sha256(source.read_bytes()).hexdigest()[:16],
                        "sources": [str(source)], "extra_cflags": ["-O3", "-std=c++17", "-Wall", "-Wextra", "-Werror"],
                        "with_cuda": False, "verbose": False}
