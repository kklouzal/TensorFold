"""Native A4 routed banks versus original checkpoint-native individual linears."""

import pytest
import torch
from nvfp4_a4_inputs import PREFIX, Context, swiglu

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
    pytest.skip("native activation FP4 requires SM12", allow_module_level=True)

from tensorfold.cuda.expert_cache import HostExpertCache, entry_bytes_for
from tensorfold.cuda.nvfp4 import checkpoint
from tensorfold.cuda.nvfp4.host_experts import load_host_experts_a4
from tensorfold.cuda.nvfp4.host_layout import authority_fields_a4
from tensorfold.cuda.nvfp4.linear import Fp4Linear


def linears(ctx, expert):
    result = []
    for projection in ("gate_proj", "up_proj", "down_proj"):
        base = f"{PREFIX}.experts.{expert}.{projection}"
        fields = ctx.payload
        result.append(Fp4Linear.from_checkpoint(fields[base + ".weight_packed"].cuda(),
            fields[base + ".weight_scale"].cuda(), 1. / float(fields[base + ".weight_global_scale"][0]),
            act=1. / float(fields[base + ".input_global_scale"][0])))
    return result


def individual_reference(ctx, x, picks, *, prefill):
    """Original CPU checkpoint fields packed by the separate CUDA implementation.

    Each row/slot executes alone, with its original expert's three independent
    globals and two separate BF16 operations before down quantization.
    """
    outputs = torch.zeros((*picks.shape, ctx.cfg.hidden), dtype=torch.bfloat16 if prefill else torch.float32,
                          device=x.device)
    native = checkpoint.prompt if prefill else checkpoint.matmul
    lins = [linears(ctx, e) for e in range(ctx.cfg.experts)]
    for row, slots in enumerate(picks.cpu().tolist()):
        for slot, expert in enumerate(slots):
            if not 0 <= expert < ctx.cfg.experts:
                continue
            gate, up, down = lins[expert]
            g = native(checkpoint.A4, x[row:row + 1], gate)
            u = native(checkpoint.A4, x[row:row + 1], up)
            silu = torch.nn.functional.silu(g)
            hidden = silu * u
            outputs[row, slot] = native(checkpoint.A4, hidden, down, f32=not prefill)[0]
    return outputs


@pytest.mark.parametrize("prefill,rows", [(False, 1), (False, 17), (True, 133)])
def test_native_resident_cached_eviction_strided_slots_and_permutations(prefill, rows):
    original = Context()
    resident = load_host_experts_a4(original, PREFIX, original.cfg, "cuda:0", swiglu=swiglu)
    authority = Context()
    entry = entry_bytes_for(tuple(torch.empty(shape, dtype=torch.uint8 if dtype == "U8" else torch.float32)
                                  for shape, dtype in authority_fields_a4(1, original.cfg.moe_width, original.cfg.hidden)))
    cache = HostExpertCache(entry, entry, "cuda:0")
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


@pytest.mark.parametrize("prefill,capacity", [(False, 1), (False, 4), (True, 1), (True, 4)])
def test_grouped_native_a4_candidate_matches_admitted_baseline_with_actual_projection_shapes(prefill, capacity):
    from tensorfold.cuda.nvfp4.grouped_a4 import _ext, run

    context = Context(count=4, width=512, dims=2048)
    resident = load_host_experts_a4(context, PREFIX, context.cfg, "cuda:0", swiglu=swiglu)
    entry = entry_bytes_for(tuple(torch.empty(shape, dtype=torch.uint8 if dtype == "U8" else torch.float32)
                                 for shape, dtype in authority_fields_a4(1, 512, 2048)))
    cache = HostExpertCache(entry * capacity, entry, "cuda:0")
    try:
        fresh = Context(count=4, width=512, dims=2048)
        cached = load_host_experts_a4(fresh, PREFIX, fresh.cfg, "cuda:0", cache=cache, swiglu=swiglu)
        rows, slots = 33, 5
        x = (torch.randn(rows, 2048, generator=torch.Generator().manual_seed(991)) * .7).to("cuda", torch.bfloat16)
        full = torch.full((rows, slots + 1), -7, dtype=torch.int32, device="cuda")
        picks = full[:, :slots]
        picks[:, :4] = (torch.arange(rows, dtype=torch.int32, device="cuda")[:, None] +
                        torch.arange(4, dtype=torch.int32, device="cuda")) % 4
        out = torch.zeros((rows, slots + 1, 2048), dtype=torch.bfloat16 if prefill else torch.float32, device="cuda")
        expected = torch.empty_like(out[:, :slots])
        resident.run(x, picks, expected, prefill=prefill)
        for bank in (resident, cached, cached):
            run(bank, x, picks, out[:, :slots], prefill=prefill)
            assert torch.equal(out[:, :slots], expected)
            assert torch.count_nonzero(out[:, slots:]) == 0
        # Odd row/slot strides and a one-element storage offset are legal
        # public views; BF16 stores cannot assume bfloat162 alignment here.
        slot_stride = 2049
        row_stride = slots * slot_stride + 1
        storage = torch.full((rows * row_stride + 1,), 17., dtype=out.dtype, device="cuda")
        odd = storage.as_strided((rows, slots, 2048), (row_stride, slot_stride, 1), 1)
        run(cached, x, picks, odd, prefill=prefill)
        assert torch.equal(odd, expected)
        mask = torch.zeros(storage.shape, dtype=torch.bool, device="cuda")
        mask.as_strided(odd.shape, odd.stride(), 1).fill_(True)
        assert torch.equal(storage[~mask], torch.full_like(storage[~mask], 17.))
        # Independent activation-quantizer oracle: standard quant4's byte
        # representation is row-major, while its scale layout is K-major.
        plan = torch.tensor([[0, 0, 3], [2, 1, 1], [1, 0, 0]], dtype=torch.int32, device="cuda")
        codes = torch.empty((2, 3, 1024), dtype=torch.uint8, device="cuda")
        scales = torch.empty((2, 3, 128), dtype=torch.uint8, device="cuda")
        _ext().quantize(x, plan, resident.fields[4], codes, scales)
        for projection in range(2):
            for pair, (row, _, expert) in enumerate(plan.cpu().tolist()):
                original_act = 1. / float(context.payload[
                    f"{PREFIX}.experts.{expert}.{'gate_proj' if projection == 0 else 'up_proj'}.input_global_scale"][0])
                want_codes, want_scales = checkpoint.quant4(x[row:row + 1], original_act)
                assert torch.equal(codes[projection, pair], want_codes[0])
                assert torch.equal(scales[projection, pair], want_scales[:, 0].reshape(-1))
        assert cache.evictions > 0 if capacity < 4 else cache.evictions == 0
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
    ext.quantize(x, plan, calibration, codes, scales)
    want_codes, want_scales = checkpoint.quant4(x, 1.)
    assert torch.equal(codes[0], want_codes)
    assert torch.equal(scales[0, 0], want_scales[:, 0].reshape(-1))
    torch.cuda.synchronize()
