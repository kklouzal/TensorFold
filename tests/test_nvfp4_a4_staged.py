"""Whole-chunk staging: exact authority, bounded owners and failed publication."""

import gc
import weakref

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from nvfp4_a4_inputs import PREFIX, swiglu
from test_nvfp4_a4_bulk import FileContext, expected_authority
from test_nvfp4_a4_cache_control import CPUCache

from tensorfold.cuda.nvfp4 import host_layout
from tensorfold.cuda.nvfp4.host_experts import load_host_experts_a4


def test_nonperiodic_custom_bank_uses_admitted_partial_packed_owners_then_retires_them(tmp_path, monkeypatch):
    count, width, dims, chunk = 12, 64, 128, 5
    blob = 9 * width * dims // 16
    scales = count * 3 * width * dims // 16
    monkeypatch.setattr(host_layout, "READER_PIECE_BYTES", scales + 6 * 4096 + chunk * 3 * width * dims // 2)
    context = FileContext(count=count, width=width, dims=dims)
    generator = torch.Generator().manual_seed(220931)
    for name, tensor in context.payload.items():
        if name.endswith(("weight_packed", "weight_scale")):
            upper = 127 if name.endswith("weight_scale") else 256
            value = torch.randint(upper, tensor.shape, dtype=torch.uint8, generator=generator)
            value.view(-1)[:upper] = torch.randperm(upper, generator=generator).to(torch.uint8)
            context.payload[name] = value.view(tensor.dtype)
    context.materialize(tmp_path)
    cache, owners, requested, scatters = CPUCache(width, dims), [], [], []
    empty, index_copy = torch.empty, torch.Tensor.index_copy_

    def allocate(shape, *args, **kwargs):
        value = empty(shape, *args, **kwargs)
        if isinstance(shape, (tuple, list)) and len(shape) == 3 and tuple(shape[1:]) == (3, blob):
            owners.append(weakref.ref(value))
            requested.append(value.numel() * value.element_size())
        return value

    def scatter(target, dim, indices, source):
        if target.shape[-1] == blob:
            scatters.append((target.ndim, source.ndim, source.shape[0]))
        return index_copy(target, dim, indices, source)

    ids = [9, 2, 11, 0, 6, 1, 10, 3, 8, 4, 7, 5]
    try:
        with monkeypatch.context() as scoped:
            scoped.setattr(torch, "empty", allocate)
            scoped.setattr(torch.Tensor, "index_copy_", scatter)
            load_host_experts_a4(context, PREFIX, context.cfg, "cuda:0", cache=cache,
                                owned_ids=ids, swiglu=swiglu)
        expected_authority(context, cache.source_views(cache._layers[0].source[0]), ids)
        assert list(map(len, context.run_reads)) == [36, 15, 15, 6]
        assert requested == [5 * 3 * blob, 5 * 3 * blob, 2 * 3 * blob]
        assert max(requested) == host_layout.bulk_layout_a4(count, width, dims).packed_bytes
        assert scatters == [(3, 3, 5), (2, 2, 5), (3, 3, 5), (2, 2, 5), (3, 3, 2), (2, 2, 2)]
        gc.collect()
        assert owners and all(owner() is None for owner in owners)
        assert all(owner() is None for owner in context.run_refs) and not context.where
    finally:
        context.files.close()


@pytest.mark.parametrize("phase", ["private_copy", "final_down_scatter"])
def test_partial_staging_or_scatter_failure_never_publishes_and_preserves_primary(tmp_path, monkeypatch, phase):
    context, cache = FileContext(count=2).materialize(tmp_path), CPUCache()
    failure, allocated, copies = OSError("owned staged packing failed"), [], []
    original_allocate, copy, index_copy = cache.allocate_authority, torch.Tensor.copy_, torch.Tensor.index_copy_
    blob = 9 * context.cfg.moe_width * context.cfg.hidden // 16

    def authority(*args, **kwargs):
        value = original_allocate(*args, **kwargs)
        allocated.append(value)
        return value

    def copy_plane(target, source, *args, **kwargs):
        copies.append(target.ndim)
        if phase == "private_copy" and len(copies) == 2:
            raise failure
        return copy(target, source, *args, **kwargs)

    def scatter(target, dim, indices, source):
        if phase == "final_down_scatter" and target.ndim == 2 and target.shape[-1] == blob:
            raise failure
        return index_copy(target, dim, indices, source)

    monkeypatch.setattr(cache, "allocate_authority", authority)
    try:
        with monkeypatch.context() as scoped:
            scoped.setattr(torch.Tensor, "copy_", copy_plane)
            scoped.setattr(torch.Tensor, "index_copy_", scatter)
            with pytest.raises(OSError) as caught:
                load_host_experts_a4(context, PREFIX, context.cfg, "cuda:0", cache=cache, swiglu=swiglu)
        assert caught.value is failure and not cache._layers and not context.used
        fields = cache.source_views(allocated[0])
        assert bool((fields[1] == 0xa5).all())
        if phase == "private_copy":
            assert bool((fields[0] == 0xa5).all())
        else:
            assert not bool((fields[0] == 0xa5).all())  # gate/up copied before down failed
        del caught
        failure.__traceback__ = None
        gc.collect()
        assert all(owner() is None for owner in context.run_refs)
    finally:
        context.files.close()
