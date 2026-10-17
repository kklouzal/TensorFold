"""RAM-backed NVFP4 retains stored bytes, native slots and complete-model output."""

import pytest
import torch

from tensorfold.cuda import moe as moe_mod
from tensorfold.cuda.expert_cache import HostExpertCache, entry_bytes_for
from tensorfold.cuda.nvfp4.host_experts import load_moe_host, load_nvfp4_host_experts
from tensorfold.families.qwen4_exp import ram_experts
from tensorfold.families.qwen4_exp.cuda import nvfp4_moe
from tensorfold.families.qwen4_exp.cuda.reader import _Reader
from tensorfold.families.qwen4_exp.cuda.weights import Config, load

from nvfp4_tiny import write


def _close(w):
    w.meta["ple_tables"].close(torch.cuda.synchronize)
    if "expert_cache" in w.meta:
        w.meta["expert_cache"].close()


def test_nvfp4_ram_layout_sizes_exact_payload_and_keeps_shared_resident(tmp_path):
    model = write(tmp_path, hidden=512)
    layout = ram_experts.layout(model, "auto")
    entry = 3 * (128 // 32) * (512 // 32) * 144 * 4 + 32
    assert layout.format == "nvfp4" and layout.entry_bytes == entry
    assert layout.host_bytes == 6 * (entry - 20) and layout.slots == 1
    assert sum(layout.authority_allocations) == layout.host_bytes and len(layout.authority_allocations) == 12
    assert layout.loading_device_bytes >= 128 * 128 * 512
    transform = layout.transform(lambda *_: (99, 0))
    assert transform("model.layers.0.mlp.experts.0.gate_proj.weight", {}) == (0, 0)
    assert transform("model.layers.0.mlp.shared_expert.gate_proj.weight", {}) == (99, 0)


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA NVFP4 execution")


@cuda
@pytest.mark.parametrize("mtp_format,scale", [("bf16", "tensor"), ("fp8", "tensor"),
                                             ("fp8", "row"), ("fp8", "block")])
def test_compact_owned_authority_is_resident_bytes_including_draft_scales(tmp_path, mtp_format, scale):
    model = write(tmp_path, hidden=512, mtp_experts=mtp_format, mtp_scale=scale)
    resident = load(model)
    reader = _Reader(model, "cuda")
    reads = []
    actual = reader.get_rows

    def recorded(name, lo, hi, device="cpu"):
        reads.append((name, lo, hi))
        return actual(name, lo, hi, device)

    reader.get_rows = recorded
    try:
        for name, expected in (("model.layers.0.mlp", resident.layers[0].moe.experts.routed_experts),
                               ("mtp.layers.0.mlp", resident.mtp.layer.moe.experts.routed_experts)):
            reads.clear()
            got = load_nvfp4_host_experts(reader, name, resident.cfg, "cuda", owned_ids=[1], include_shared=False)
            assert got.count == 1
            for field in ("up", "down", "up_scale", "down_scale"):
                assert torch.equal(getattr(got, field), getattr(expected, field)[1:2].cpu()), (name, field)
            assert not any(".experts.0." in key or ".shared_expert." in key for key, *_ in reads)
            assert all(lo == 1 and hi == 2 for key, lo, hi in reads if key.endswith(("gate_up_proj", "down_proj")))
    finally:
        reader.close()
        _close(resident)


@cuda
@pytest.mark.parametrize("prefill,rows", [(False, 1), (False, 17), (True, 133)])
def test_cached_item_eviction_preserves_all_native_slots_and_logical_plan(tmp_path, prefill, rows):
    model = write(tmp_path, hidden=512, experts=8)
    w = load(model)
    reader = _Reader(model, "cuda")
    layout = ram_experts.layout(model, "auto")
    cache = HostExpertCache(layout.entry_bytes, layout.entry_bytes, "cuda")
    try:
        workspace = {}
        bank = load_moe_host(reader, "model.layers.0.mlp", w.cfg, cache=cache, workspace=workspace)
        draft = load_moe_host(reader, "mtp.layers.0.mlp", w.cfg, cache=cache, workspace=workspace)
        cfg = type("MoECfg", (), {"num_experts": 8, "num_experts_per_tok": 2,
                                 "hidden_size": 512, "moe_intermediate_size": 128})
        x = (torch.randn(rows, 512, generator=torch.Generator().manual_seed(91)) * .1).to("cuda", torch.bfloat16)
        native = moe_mod.MoEBuffers(rows, cfg, "cuda", prefill=prefill)
        cached = moe_mod.MoEBuffers(rows, cfg, "cuda", prefill=prefill)
        router = w.layers[0].moe.router
        nvfp4_moe.moe(x, x, router, w.layers[0].moe.experts, native, cfg)
        expected = native.y.clone()
        items = native.plan.items[:int(native.plan.counts[0])].clone()
        for _ in range(2):
            nvfp4_moe.moe(x, x, router, bank, cached, cfg)
            assert torch.equal(cached.y, expected)
            assert torch.equal(cached.act, native.act)
            assert torch.equal(cached.pick, native.pick)
            assert torch.equal(cached.plan.items[:items.shape[0]], items)
        assert cache.evictions > 0
        # EP contract also accepts unowned slots and compact IDs. The slot's
        # original expert arithmetic remains identical after host regrouping.
        picks = native.pick.clone()
        picks[::2, 0] = -1
        out = torch.empty_like(expected)
        bank.run_picks(x, picks, out, prefill=prefill)
        mask = picks >= 0
        assert torch.equal(out[mask], expected[mask])
        assert torch.equal(out[~mask], torch.zeros_like(out[~mask]))
        with pytest.raises(ValueError, match="out-of-range"):
            bank.run_picks(x[:1], torch.full_like(picks[:1], 99), out[:1], prefill=prefill)
        alias = torch.empty_like(out)
        alias_x = alias.view(torch.bfloat16)[:, 0, :bank.dims]
        with pytest.raises(ValueError, match="overlap"):
            bank.run_picks(alias_x, picks, alias, prefill=prefill)
        pool_out = cache._pool[:out[:1].numel() * out.element_size()].view(out.dtype).view_as(out[:1])
        with pytest.raises(ValueError, match="overlap"):
            bank.run_picks(x[:1], picks[:1], pool_out, prefill=prefill)
        # Every main/MTP bank and both output precisions share one admitted
        # native plan and BF16 activation buffer, not one allocation per bank.
        flip = moe_mod.MoEBuffers(rows, cfg, "cuda", prefill=not prefill)
        nvfp4_moe.moe(x, x, w.mtp.layer.moe.router, w.mtp.layer.moe.experts, flip, cfg)
        draft_out = torch.empty_like(flip.y)
        draft.run_picks(x, flip.pick, draft_out, prefill=not prefill)
        assert torch.equal(draft_out, flip.y)
        assert bank._workspace is draft._workspace is workspace and len(workspace) == 1
        plan, act = next(iter(workspace.values()))
        storage = {t.data_ptr() for t in bank.device_tensors()[-1]}
        assert all(t.data_ptr() in storage for t in (plan.members, plan.items, plan.counts, plan.rank, plan.hist, act))
    finally:
        reader.close()
        cache.close()
        _close(w)


@cuda
@pytest.mark.parametrize("table_format", ["affine", "bf16", "nvfp4", "mxfp8"])
def test_full_cached_model_serial_draft_and_concurrent_match_resident(tmp_path, table_format):
    from tensorfold.cuda.streams import Stream
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, mtp_decode, prefill, serial_decode
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    options = {"ple_bf16": table_format == "bf16", "ple_nvfp4": table_format in ("nvfp4", "mxfp8"),
               "mxfp8": table_format == "mxfp8"}
    model = write(tmp_path, hidden=512, **options)
    layout = ram_experts.layout(model, "auto")
    a = load(model, draft_vocab=128)
    b = load(model, draft_vocab=128, ple_on_ssd=True, vram_experts=layout.entry_bytes / 2**30)
    assert isinstance(b.layers[0].moe.experts, nvfp4_moe.CachedMoE4)
    assert b.meta["expert_cache"].entry_bytes == entry_bytes_for(b.meta["expert_cache"]._layers[0].source)
    prompt = [5, 17, 99, 250, 7, 64, 30, 11, 12, 13]
    try:
        for sampling in (None, Sampling(seed=7, top_k=20, top_p=.95)):
            native = Engine(a, capacity=128, max_rows=8, prefill_rows=5, graphs=False)
            first = prefill(native, prompt, sampling)
            expected = serial_decode(native, first, 8, sampling).tokens
            engine = Engine(b, capacity=128, max_rows=8, prefill_rows=5, graphs=True)
            assert engine.graphs is None
            assert prefill(engine, prompt, sampling) == first
            assert serial_decode(engine, first, 8, sampling).tokens == expected
            assert prefill(engine, prompt, sampling) == first
            assert mtp_decode(engine, first, 8, sampling, depth=2, confidence=0.).tokens == expected

        def multi(w):
            decoder = MultiDecoder(w, slots=2, capacity=128, depth=2, confidence=0., prefill_rows=5,
                                   share=.5, stop_eos=False)
            streams = [Stream(p, 8, Sampling(seed=7 + i, top_k=20, top_p=.95), draft=True)
                       for i, p in enumerate((prompt, prompt[::-1]))]
            for stream in streams:
                decoder.admit(stream)
            for _ in range(100):
                if not decoder.live():
                    assert all(stream.error is None for stream in streams)
                    return [stream.out for stream in streams]
                decoder.finish(decoder.round())
            pytest.fail("concurrent NVFP4 requests did not finish")

        assert multi(a) == multi(b)
        assert b.meta["expert_cache"].evictions > 0
    finally:
        _close(a)
        _close(b)


@cuda
def test_public_engine_admits_and_closes_nvfp4_host_authority_with_ssd(tmp_path, monkeypatch):
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    model = write(tmp_path, hidden=512, ple_nvfp4=True)
    monkeypatch.setenv("TENSORFOLD_PREFILL_ROWS", "64")
    entry = ram_experts.layout(model, "auto").entry_bytes
    options = dict(depth=2, draft_vocab=128, max_len=128, context_explicit=True, prefetch=False, graphs=True,
                   ple_on_ssd=True)
    native = FlashNextEngine(model, **options)
    cached = None
    try:
        cached = FlashNextEngine(model, vram_experts=entry / 2**30, **options)
        assert cached.capacity_plan["vram_experts"]["format"] == "nvfp4"
        assert cached.capacity_plan["vram_experts"]["gpu_bytes"] == entry
        cache = cached.w.meta["expert_cache"]
        assert cache.host_bytes <= cached.capacity_plan["vram_experts"]["host_bytes"]
        prompt, sampling = [5, 17, 99, 250, 7, 64, 30, 11], Sampling(seed=9, top_k=20, top_p=.95)

        def generate(engine, draft):
            tokens = []
            engine.generate(prompt, 8, sampling, lambda new: tokens.extend(new), draft=draft, stop_eos=False)
            return tokens

        expected = generate(native, False)
        assert len(expected) == 8
        assert generate(cached, False) == expected
        assert generate(cached, True) == expected
    finally:
        if cached is not None:
            cached.close()
        native.close()
