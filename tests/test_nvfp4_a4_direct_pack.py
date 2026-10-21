"""CPU final-authority packing: independent bytes, ownership and bank boundaries."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from nvfp4_a4_inputs import PREFIX, Context, swiglu
from test_nvfp4_a4_host_control import Cache, oracle_blob

from tensorfold.cuda.nvfp4 import experts_a4
from tensorfold.cuda.nvfp4.experts_a4 import pack_projection_a4
from tensorfold.cuda.nvfp4.host_experts import load_host_experts_a4


def payload(n, k, *, strided=False):
    gen = torch.Generator().manual_seed(1701 + n * 31 + k)
    codes = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=gen)
    blocks = torch.randint(0, 256, (n, k // 16), dtype=torch.uint8, generator=gen)
    codes.reshape(-1)[:256] = torch.randperm(256, generator=gen).to(torch.uint8)
    blocks.reshape(-1)[:256] = torch.randperm(256, generator=gen).to(torch.uint8)
    if strided:
        word_storage = torch.empty((n, k), dtype=torch.uint8)
        scale_storage = torch.empty((n, k // 8), dtype=torch.uint8)
        word_storage[:, ::2], scale_storage[:, ::2] = codes, blocks
        codes, blocks = word_storage[:, ::2], scale_storage[:, ::2]
    return codes, blocks.view(torch.float8_e4m3fn)


@pytest.mark.parametrize("n,k", [(64, 64), (128, 192), (640, 2560), (2560, 640)])
@pytest.mark.parametrize("strided", [False, True])
def test_cpu_default_and_final_destination_keep_every_original_byte(n, k, strided):
    codes, blocks = payload(n, k, strided=strided)
    before_codes, before_blocks = codes.clone(), blocks.view(torch.uint8).clone()
    expected = oracle_blob(codes, blocks)
    allocated = pack_projection_a4(codes, blocks)
    backing = torch.full((expected.numel() + 8,), 0xa5, dtype=torch.uint8)
    out = backing[4:-4]
    returned = pack_projection_a4(codes, blocks, out=out)
    assert returned is out and torch.equal(out, expected) and torch.equal(allocated, expected)
    assert torch.equal(backing[:4], torch.full((4,), 0xa5, dtype=torch.uint8))
    assert torch.equal(backing[-4:], torch.full((4,), 0xa5, dtype=torch.uint8))
    assert torch.equal(codes, before_codes) and torch.equal(blocks.view(torch.uint8), before_blocks)
    out.zero_()
    assert torch.equal(allocated, expected)  # default output owns separate storage


@pytest.mark.parametrize("damage", ["dtype", "shape", "length", "stride", "alignment", "device", "object"])
def test_bad_destination_fails_without_mutating_storage_or_sources(damage):
    codes, blocks = payload(64, 64)
    size = 9 * 64 * 64 // 16
    backing = torch.full((size * 2 + 4,), 0xa5, dtype=torch.uint8)
    before = backing.clone()
    out = backing[:size]
    if damage == "dtype": out = out.view(torch.int16)
    elif damage == "shape": out = out.view(1, -1)
    elif damage == "length": out = out[:-1]
    elif damage == "stride": out = backing[:size * 2:2]
    elif damage == "alignment": out = backing[1:size + 1]
    elif damage == "device": out = torch.empty(size, dtype=torch.uint8, device="meta")
    else: out = object()
    original_codes, original_blocks = codes.clone(), blocks.view(torch.uint8).clone()
    with pytest.raises(ValueError, match="CPU destination"):
        pack_projection_a4(codes, blocks, out=out)
    assert torch.equal(backing, before)
    assert torch.equal(codes, original_codes) and torch.equal(blocks.view(torch.uint8), original_blocks)


@pytest.mark.parametrize("source", ["codes", "scales"])
def test_destination_overlap_is_rejected_before_writing(source):
    codes, blocks = payload(64, 64)
    out = torch.full((9 * 64 * 64 // 16,), 0xa5, dtype=torch.uint8)
    if source == "codes":
        codes = out[:64 * 32].view(64, 32)
    else:
        blocks = out[:64 * 4].view(64, 4).view(torch.float8_e4m3fn)
    before = out.clone()
    with pytest.raises(ValueError, match="must not overlap"):
        pack_projection_a4(codes, blocks, out=out)
    assert torch.equal(out, before)


def test_invalid_input_geometry_stops_before_writing_valid_destination():
    codes = torch.zeros((64, 31), dtype=torch.uint8)
    blocks = torch.zeros((64, 4), dtype=torch.uint8).view(torch.float8_e4m3fn)
    out = torch.full((2304,), 0xa5, dtype=torch.uint8)
    with pytest.raises(ValueError, match="complete64-wide"):
        pack_projection_a4(codes, blocks, out=out)
    assert bool((out == 0xa5).all())


def test_scalar_batch_contains_only_owned_fields_with_exact_cap_and_no_retention():
    ctx, cache = Context(), Cache()
    bank = load_host_experts_a4(ctx, PREFIX, ctx.cfg, "cuda:0", cache=cache, owned_ids=[2, 0], swiglu=swiglu)
    names, cap = ctx.batches[0]
    assert len(ctx.batches) == 1 and len(names) == 12 and cap == 48
    assert all(name.endswith(("weight_global_scale", "input_global_scale")) for name in names)
    assert not any(".experts.1." in name for name in ctx.reads)
    assert all(ctx.reads.index(name) < next(i for i, read in enumerate(ctx.reads) if read.endswith("weight_packed"))
               for name in names)
    assert not any(isinstance(value, dict) for value in vars(bank).values())
    fields = cache.source_views(cache._layers[0][0])
    for local, expert in enumerate((2, 0)):
        for p, projection in enumerate(("gate_proj", "up_proj", "down_proj")):
            base = f"{PREFIX}.experts.{expert}.{projection}"
            expected = oracle_blob(ctx.payload[base + ".weight_packed"], ctx.payload[base + ".weight_scale"])
            assert torch.equal(fields[0][local, p] if p < 2 else fields[1][local], expected)


def test_shared_257_cell_scalar_cap_is_6168_bytes_with_emulated_resident_transfer(monkeypatch):
    """CPU ownership/control check; actual resident CUDA execution has separate native gates."""
    ctx = Context(count=256, width=64, dims=128).alias_shared_expert(0)
    original_to = torch.Tensor.to

    def to(value, *args, **kwargs):
        device = kwargs.get("device", args[0] if args else None)
        if device is not None and torch.device(device).type == "cuda":
            return value.clone()
        return original_to(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to)
    monkeypatch.setattr(experts_a4, "ExpertsA4", lambda fields, width, dims, **kwargs:
                        SimpleNamespace(count=fields[0].shape[0], fields=fields))
    bank = load_host_experts_a4(ctx, PREFIX, ctx.cfg, "cuda:0", swiglu=swiglu, shared=True)
    assert bank.count == 257
    names, cap = ctx.batches[0]
    assert len(names) == 1542 and cap == 6168
    assert sum(".shared_expert." in name for name in names) == 6
    assert len(ctx.reads) == 257 * 12


def test_partial_direct_write_failure_never_publishes_authority(monkeypatch):
    ctx, cache = Context(), Cache()
    primary = KeyboardInterrupt("direct authority packing interrupted")

    def fail(codes, scales, *, out):
        out[0] = 0x13
        raise primary

    monkeypatch.setattr(experts_a4, "pack_projection_a4", fail)
    with pytest.raises(KeyboardInterrupt) as caught:
        load_host_experts_a4(ctx, PREFIX, ctx.cfg, "cuda:0", cache=cache, swiglu=swiglu)
    assert caught.value is primary
    assert len(cache.allocations) == 1 and not cache._layers and not ctx.used


def test_scalar_batch_failure_never_reads_code_payload_or_publishes(monkeypatch):
    ctx, cache = Context(), Cache()
    primary = OSError("batched scalar read failed")

    def fail(names, *, max_bytes):
        assert max_bytes == 72 and len(names) == 18
        raise primary

    monkeypatch.setattr(ctx, "read_f32_scalars", fail)
    with pytest.raises(OSError) as caught:
        load_host_experts_a4(ctx, PREFIX, ctx.cfg, "cuda:0", cache=cache, swiglu=swiglu)
    assert caught.value is primary and not ctx.reads and not cache._layers and not ctx.used
