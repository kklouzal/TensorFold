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
