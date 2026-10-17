"""Private single-copy candidate versus corrected six-field native results."""

import gc
import weakref

import pytest
import torch
from nvfp4_a4_inputs import PREFIX, Context, ct_quant_reference, swiglu

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
    pytest.skip("native single-blob A4 candidate requires SM12", allow_module_level=True)

from tensorfold.cuda.expert_pin import ExpertPinPolicy
from tensorfold.cuda.nvfp4 import blob_cache_a4
from tensorfold.cuda.nvfp4.blob_cache_a4 import BlobExpertCache
from tensorfold.cuda.nvfp4.blob_grouped_a4 import _ext
from tensorfold.cuda.nvfp4.blob_host_a4 import load_blob_host_experts_a4
from tensorfold.cuda.nvfp4.blob_layout_a4 import blob_geometry_a4
from tensorfold.cuda.nvfp4.grouped_a4 import run as six_field_run
from tensorfold.cuda.nvfp4.host_experts import load_host_experts_a4


@pytest.mark.parametrize("prefill,capacity,pin", [(False, 1, False), (False, 4, False),
                                               (True, 1, False), (True, 4, False), (False, 4, True)])
def test_single_blob_native_eviction_views_copy_count_raw_oracle_and_retirement(prefill, capacity, pin, monkeypatch):
    geometry = blob_geometry_a4(512, 2048)
    entry = geometry.entry_bytes
    policy = (ExpertPinPolicy(4 * entry, mode="on", allocation_bytes=(4 * entry,)) if pin else None)
    cache = BlobExpertCache(entry, entry, "cuda:0", width=512, dims=2048, pin_policy=policy)
    try:
        context = Context(count=4, width=512, dims=2048)
        bank = load_blob_host_experts_a4(context, PREFIX, context.cfg, "cuda:0", cache=cache, swiglu=swiglu)
        assert len(cache._layers[0].source) == len(cache._layers[0].views) == 1
        authority_ref = weakref.ref(cache._layers[0].source[0])
        old_pool, old_views = cache._pool, cache._native_views
        original_views = blob_cache_a4.native_views

        def fail_views(*args):
            raise RuntimeError("private replacement view failure")

        with monkeypatch.context() as patch:
            patch.setattr(blob_cache_a4, "native_views", fail_views)
            with pytest.raises(RuntimeError, match="replacement view failure"):
                cache.configure_before_use(entry * capacity)
        assert cache._pool is old_pool and cache._native_views is old_views and cache.capacity == 1
        cache.configure_before_use(entry * capacity)
        assert cache.capacity == capacity
        assert all(t.stride(0) * t.element_size() == entry for t in cache._native_views)
        assert cache._native_down.data_ptr() == cache._native_views[1].data_ptr()
        del old_pool, old_views
        # One staged or direct H2D copy per expert miss, including calibrations.
        uploads, original_copy = [], torch.Tensor.copy_

        def copied(target, source, *args, **kwargs):
            if target.device.type == "cuda" and source.device.type == "cpu":
                uploads.append(target.numel() * target.element_size())
            return original_copy(target, source, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "copy_", copied)
            with cache.lease(0, [0]):
                pass
        assert uploads == [entry]
        assert cache._layers[0].pinned == (pin,)
        # Preparing another view after the first borrow is forbidden by the
        # candidate's own contract, independently of numerical comparison.
        with monkeypatch.context() as patch:
            patch.setattr(blob_cache_a4, "native_views", lambda *a: pytest.fail("late native view allocation"))
            source = Context(count=4, width=512, dims=2048)
            resident = load_host_experts_a4(source, PREFIX, source.cfg, "cuda:0", swiglu=swiglu)
            rows, slots = 33, 5
            x = (torch.randn(rows, 2048, generator=torch.Generator().manual_seed(1441)) * .5).to("cuda", torch.bfloat16)
            picks = torch.full((rows, slots + 1), -7, dtype=torch.int32, device="cuda")[:, :slots]
            picks[:, :4] = (torch.arange(rows, dtype=torch.int32, device="cuda")[:, None] +
                            torch.arange(4, dtype=torch.int32, device="cuda")) % 4
            expected = torch.empty((rows, slots, 2048), dtype=torch.bfloat16 if prefill else torch.float32, device="cuda")
            six_field_run(resident, x, picks, expected, prefill=prefill)
            stride = slots * 2049 + 1
            storage = torch.full((rows * stride + 1,), 17., dtype=expected.dtype, device="cuda")
            output = storage.as_strided(expected.shape, (stride, 2049, 1), 1)
            for _ in range(2):
                bank.run(x, picks, output, prefill=prefill)
                assert torch.equal(output, expected)
            mask = torch.zeros(storage.shape, dtype=torch.bool, device="cuda")
            mask.as_strided(output.shape, output.stride(), 1).fill_(True)
            assert torch.equal(storage[~mask], torch.full_like(storage[~mask], 17.))
        assert blob_cache_a4.native_views is original_views
        boundary = torch.full((1, 2048), .5, dtype=torch.bfloat16, device="cuda")
        with cache.lease(0, [0]) as (fields, mapping):
            plan = torch.tensor([[0, 0, mapping[0]]], dtype=torch.int32, device="cuda")
            codes = torch.empty((2, 1, 1024), dtype=torch.uint8, device="cuda")
            scales = torch.empty((2, 1, 128), dtype=torch.uint8, device="cuda")
            _ext().quantize(boundary, plan, fields[4], codes, scales)
        raw = float(context.payload[PREFIX + ".experts.0.gate_proj.input_global_scale"][0])
        want_codes, want_scales = ct_quant_reference(boundary.cpu(), raw)
        assert torch.equal(codes[0].cpu(), want_codes)
        assert torch.equal(scales[0].cpu(), want_scales)
        assert torch.equal(scales[0], torch.full_like(scales[0], 0x0a))
        assert cache.copied_bytes == cache.misses * entry
        if capacity == 1:
            assert cache.evictions > 0
        del fields
        cache.close()
        gc.collect()
        assert cache._pool is None and cache._native_views == () and cache._native_down is None
        assert authority_ref() is None
        assert sum(t.untyped_storage().nbytes() for t in bank.scalars) == 24 * bank.count
    finally:
        cache.close()


def test_blob_native_scalar_field_boundary_refuses_empty_extreme_stride_before_launch():
    maximum = torch.iinfo(torch.int64).max
    plan = torch.tensor([[0, 0, 0]], dtype=torch.int32, device="cuda")
    x = torch.ones((1, 64), dtype=torch.bfloat16, device="cuda")
    codes = torch.empty((2, 1, 32), dtype=torch.uint8, device="cuda")
    scales = torch.empty((2, 1, 4), dtype=torch.uint8, device="cuda")
    for shape in ((0, 2), (1, 0)):
        invalid = torch.empty_strided(shape, (maximum, 1), dtype=torch.float32, device="cuda")
        with pytest.raises(RuntimeError, match="positive inner|complete per-expert"):
            _ext().quantize(x, plan, invalid, codes, scales)
    # Unused singleton outer stride is legal and cannot overflow indexing.
    calibration = torch.empty_strided((1, 2), (maximum, 1), dtype=torch.float32, device="cuda")
    calibration.fill_(1.)
    _ext().quantize(x, plan, calibration, codes, scales)
    want_codes, want_scales = ct_quant_reference(x.cpu(), 1.)
    assert torch.equal(codes[0].cpu(), want_codes)
    assert torch.equal(scales[0].cpu(), want_scales)
    torch.cuda.synchronize()
