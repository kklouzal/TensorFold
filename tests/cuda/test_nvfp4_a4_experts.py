"""Native A4 routed banks versus original checkpoint-native individual linears."""

import gc
import weakref

import pytest
import torch
from nvfp4_a4_inputs import PREFIX, Context, ct_quant_reference, individual_reference, swiglu

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
    pytest.skip("native activation FP4 requires SM12", allow_module_level=True)

from tensorfold.cuda.expert_cache import entry_bytes_for
from tensorfold.cuda.expert_pin import ExpertPinPolicy
from tensorfold.cuda.nvfp4 import cache_a4, checkpoint
from tensorfold.cuda.nvfp4.cache_a4 import A4ExpertCache
from tensorfold.cuda.nvfp4.grouped_a4 import _ext
from tensorfold.cuda.nvfp4.host_experts import load_host_experts_a4
from tensorfold.cuda.nvfp4.host_layout import authority_fields_a4, entry_layout_a4


@pytest.mark.parametrize("prefill,rows", [(False, 1), (False, 17), (True, 133)])
def test_native_resident_cached_eviction_strided_slots_and_permutations(prefill, rows, monkeypatch):
    original = Context()
    resident = load_host_experts_a4(original, PREFIX, original.cfg, "cuda:0", swiglu=swiglu)
    authority = Context()
    entry = entry_bytes_for(tuple(torch.empty(shape, dtype=torch.uint8 if dtype == "U8" else torch.float32)
                                  for shape, dtype in authority_fields_a4(1, original.cfg.moe_width, original.cfg.hidden)))
    cache = A4ExpertCache(entry, entry, "cuda:0", width=original.cfg.moe_width, dims=original.cfg.hidden)
    try:
        cached = load_host_experts_a4(authority, PREFIX, authority.cfg, "cuda:0", cache=cache, swiglu=swiglu)
        x = (torch.randn(rows, original.cfg.hidden, generator=torch.Generator().manual_seed(901)) * .3).to(
            "cuda", torch.bfloat16)
        # Extra shared slot makes both caller views strided. Invalid routed
        # values must never borrow a cell or read expert payload.
        picks_full = torch.full((rows, 4), -1, dtype=torch.int32, device="cuda")
        picks = picks_full[:, :3]
        picks[:, 0] = torch.arange(rows, device="cuda", dtype=torch.int32) % 3
        picks[:, 1] = (picks[:, 0] + 1) % 3
        picks[::2, 2] = original.cfg.experts
        expected = individual_reference(Context(), x, picks, prefill=prefill)
        output = torch.full((rows, 4, original.cfg.hidden), 17.,
                            dtype=torch.bfloat16 if prefill else torch.float32, device="cuda")
        out = output[:, :3]
        for bank in (resident, cached, cached):
            with monkeypatch.context() as patch:
                patch.setattr(checkpoint, "_inv", lambda value: pytest.fail("CPU input reciprocal during grouped execution"))
                bank.run(x, picks, out, prefill=prefill)
            assert torch.equal(out, expected)
            assert torch.equal(output[:, 3], torch.full_like(output[:, 3], 17.))
            assert torch.count_nonzero(out[:, 2]) == 0
        if rows > 1:
            assert cache.evictions > 0
        reverse = torch.arange(rows - 1, -1, -1, device="cuda")
        permuted = torch.empty_like(expected)
        cached.run(x[reverse], picks[reverse], permuted, prefill=prefill)
        assert torch.equal(permuted, expected[reverse])
        owned = Context()
        compact = load_host_experts_a4(owned, PREFIX, owned.cfg, "cuda:0", swiglu=swiglu, owned_ids=[2, 0])
        local = torch.tensor([[0, 1, -1]], dtype=torch.int32, device="cuda").expand(rows, -1).contiguous()
        compact_out = torch.empty_like(expected)
        compact.run(x, local, compact_out, prefill=prefill)
        global_picks = torch.tensor([[2, 0, -1]], dtype=torch.int32, device="cuda").expand(rows, -1)
        assert torch.equal(compact_out, individual_reference(Context(), x, global_picks, prefill=prefill))
        alias = torch.empty((rows, 3, original.cfg.hidden), dtype=torch.bfloat16, device="cuda")
        with pytest.raises(ValueError, match="overlap"):
            cached.run(alias[:, 0], picks, alias, prefill=True)
        assert len(cached.device_tensors()) == 1
        assert not any(name.endswith("scratch") for name in vars(cached))
    finally:
        cache.close()


@pytest.mark.parametrize("prefill,capacity,pin", [(False, 1, False), (False, 4, False),
                                               (True, 1, False), (True, 4, False), (False, 4, True)])
def test_native_a4_actual_shapes_copy_pin_resize_oracle_and_retirement(prefill, capacity, pin, monkeypatch):
    geometry = entry_layout_a4(512, 2048)
    entry = geometry.entry_bytes
    policy = (ExpertPinPolicy(4 * entry, mode="on", allocation_bytes=(4 * entry,)) if pin else None)
    cache = A4ExpertCache(entry, entry, "cuda:0", width=512, dims=2048, pin_policy=policy)
    try:
        context = Context(count=4, width=512, dims=2048)
        bank = load_host_experts_a4(context, PREFIX, context.cfg, "cuda:0", cache=cache, swiglu=swiglu)
        assert len(cache._layers[0].source) == len(cache._layers[0].views) == 1
        authority_ref = weakref.ref(cache._layers[0].source[0])
        old_pool, old_views = cache._pool, cache._native_views
        original_views = cache_a4.native_views

        def fail_views(*args):
            raise RuntimeError("private replacement view failure")

        with monkeypatch.context() as patch:
            patch.setattr(cache_a4, "native_views", fail_views)
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
            patch.setattr(cache_a4, "native_views", lambda *a: pytest.fail("late native view allocation"))
            source = Context(count=4, width=512, dims=2048)
            resident = load_host_experts_a4(source, PREFIX, source.cfg, "cuda:0", swiglu=swiglu)
            rows, slots = 33, 5
            x = (torch.randn(rows, 2048, generator=torch.Generator().manual_seed(1441)) * .5).to("cuda", torch.bfloat16)
            picks = torch.full((rows, slots + 1), -7, dtype=torch.int32, device="cuda")[:, :slots]
            picks[:, :4] = (torch.arange(rows, dtype=torch.int32, device="cuda")[:, None] +
                            torch.arange(4, dtype=torch.int32, device="cuda")) % 4
            expected = torch.empty((rows, slots, 2048), dtype=torch.bfloat16 if prefill else torch.float32, device="cuda")
            expected.copy_(individual_reference(Context(count=4, width=512, dims=2048), x, picks, prefill=prefill))
            stride = slots * 2049 + 1
            storage = torch.full((rows * stride + 1,), 17., dtype=expected.dtype, device="cuda")
            output = storage.as_strided(expected.shape, (stride, 2049, 1), 1)
            for _ in range(2):
                bank.run(x, picks, output, prefill=prefill)
                assert torch.equal(output, expected)
            mask = torch.zeros(storage.shape, dtype=torch.bool, device="cuda")
            mask.as_strided(output.shape, output.stride(), 1).fill_(True)
            assert torch.equal(storage[~mask], torch.full_like(storage[~mask], 17.))
        assert cache_a4.native_views is original_views
        # Independent activation-quantizer oracle: standard quant4's byte
        # representation is row-major, while its scale layout is K-major.
        plan = torch.tensor([[0, 0, 3], [2, 1, 1], [1, 0, 0]], dtype=torch.int32, device="cuda")
        codes = torch.empty((2, 3, 1024), dtype=torch.uint8, device="cuda")
        scales = torch.empty((2, 3, 128), dtype=torch.uint8, device="cuda")
        _ext().quantize(x, plan, resident.fields[4], codes, scales)
        for projection in range(2):
            for pair, (row, _, expert) in enumerate(plan.cpu().tolist()):
                raw_multiplier = float(context.payload[
                    f"{PREFIX}.experts.{expert}.{'gate_proj' if projection == 0 else 'up_proj'}.input_global_scale"][0])
                want_codes, want_scales = ct_quant_reference(x[row:row + 1].cpu(), raw_multiplier)
                assert torch.equal(codes[projection, pair].cpu(), want_codes[0])
                assert torch.equal(scales[projection, pair].cpu(), want_scales[0])
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
        assert not any(isinstance(value, torch.Tensor) for value in vars(bank).values())
        saved = output.clone()
        with pytest.raises(RuntimeError, match="cached native A4 bank is closed"):
            bank.run(x, picks, output, prefill=prefill)
        assert torch.equal(output, saved)
    finally:
        cache.close()



def test_grouped_native_boundary_refuses_empty_extreme_strides_before_launch():
    """Valid zero-storage Tensor metadata must not overflow a native guard.

    All ordinary arguments are initialized and minimally sized, so the malformed
    output reaches its own boundary check. Immediate refusal plus a following
    successful quantizer proves no deferred trap/context poisoning was hidden.
    """
    from tensorfold.cuda.nvfp4.grouped_a4 import _ext

    ext = _ext()
    n = k = 64
    codes = torch.zeros((1, 1, k // 2), dtype=torch.uint8, device="cuda")
    scales = torch.zeros((1, 1, k // 16), dtype=torch.uint8, device="cuda")
    blob = torch.zeros((1, 1, n * k * 9 // 16), dtype=torch.uint8, device="cuda")
    calibration = torch.ones((1, 1), dtype=torch.float32, device="cuda")
    items = torch.tensor([[0, 0, 1]], dtype=torch.int32, device="cuda")
    plan = torch.tensor([[0, 0, 0]], dtype=torch.int32, device="cuda")
    maximum = torch.iinfo(torch.int64).max
    for shape, strides in (((0, 2, n), (0, 2**62, 1)),
                           ((0, 2, n), (maximum, maximum, 1)),
                           ((2, 0, n), (maximum, maximum, 1)),
                           ((0, 0, n), (maximum, maximum, 1))):
        out = torch.empty_strided(shape, strides, dtype=torch.bfloat16, device="cuda")
        assert out.untyped_storage().nbytes() == 0
        with pytest.raises(RuntimeError, match="nativeA4 output: positive separate slots"):
            ext.gemm(codes, scales, blob, calibration, calibration, items, plan, out, n, k, 1, True)
    empty_input = torch.empty_strided((0, k), (maximum, 1), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="x: BF16 rows"):
        ext.quantize(empty_input, plan, calibration, codes, scales)
    # Singleton dimensions permit unused extreme strides; their actual extent
    # is only64 values, so checked geometry must preserve that supported case.
    out = torch.empty_strided((1, 1, n), (maximum, n, 1), dtype=torch.bfloat16, device="cuda")
    ext.gemm(codes, scales, blob, calibration, calibration, items, plan, out, n, k, 1, True)
    assert torch.count_nonzero(out) == 0
    x = torch.ones((1, k), dtype=torch.bfloat16, device="cuda")
    singleton_input = x.as_strided((1, k), (maximum, 1))
    ext.quantize(singleton_input, plan, calibration, codes, scales)
    want_codes, want_scales = checkpoint.quant4(x, 1.)
    assert torch.equal(codes[0], want_codes)
    assert torch.equal(scales[0, 0], want_scales[:, 0].reshape(-1))
    torch.cuda.synchronize()


def test_native_scalar_field_boundary_refuses_empty_extreme_stride_before_launch():
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
