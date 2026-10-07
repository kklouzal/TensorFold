"""Host-backed affine4 experts preserve the native resident CUDA execution bits."""

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda import experts as grouped, moe as moe_mod  # noqa: E402
from tensorfold.cuda.expert_cache import CachedExperts, HostExpertCache  # noqa: E402
from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import forward as fwd  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import Engine, mtp_decode, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.reader import _Reader  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.state import Buffers, State  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.weights import MoEW  # noqa: E402
from expert_cache_oracle import same_tensor_bits  # noqa: E402
from test_experts import mlx  # noqa: E402
from test_flashnext_forward import _model  # noqa: E402
from test_flashnext_tp import _checkpoint  # noqa: E402


def _host(ex: grouped.Experts) -> grouped.Experts:
    return grouped.Experts(ex.up.cpu(), ex.down.cpu(), ex.gs, ex.width, ex.dims, ex.limit)


def _cache(ex: grouped.Experts, slots: int) -> HostExpertCache:
    return HostExpertCache(slots * ex.bytes_per_expert(), ex.bytes_per_expert(), "cuda")


def _offload(w, slots: int = 2):
    cache = _cache(w.layers[0].moe.experts, slots)
    layers = [replace(layer, moe=replace(layer.moe, experts=CachedExperts(_host(layer.moe.experts), cache, i)))
              for i, layer in enumerate(w.layers)]
    mtp = None if w.mtp is None else replace(w.mtp, layer=replace(
        w.mtp.layer, moe=replace(w.mtp.layer.moe, experts=CachedExperts(_host(w.mtp.layer.moe.experts), cache,
                                                                    len(layers)))))
    return replace(w, layers=layers, mtp=mtp), cache


@pytest.mark.parametrize("rows,prefill_mode,slots", [(1, False, 1), (17, False, 2), (133, True, 3)])
def test_item_windows_match_resident_pairs_and_preserve_logical_plan(rows, prefill_mode, slots):
    """Cache unions exceed capacity, including shared and multi-item experts; all pair bits remain native."""

    experts, dims, width, top_k = 8, 256, 128, 4
    ex = grouped.make([mlx(experts + 1, width, dims, 32, 11), mlx(experts + 1, width, dims, 32, 12)],
                      mlx(experts + 1, dims, width, 32, 18), 32)
    x = (torch.randn((rows, dims), generator=torch.Generator(device="cuda").manual_seed(3), device="cuda")
         * 0.1).to(torch.bfloat16)
    x[:, :experts] = 0
    x[torch.arange(rows, device="cuda"), torch.arange(rows, device="cuda") % experts] = 10
    router = torch.zeros((experts + 1, dims), dtype=torch.bfloat16, device="cuda")
    router[:experts, :experts] = torch.eye(experts, dtype=torch.bfloat16, device="cuda")
    cfg = SimpleNamespace(num_experts=experts, num_experts_per_tok=top_k, hidden_size=dims,
                          moe_intermediate_size=width, top_k=top_k, experts=experts)
    w = SimpleNamespace(cfg=cfg, x3=None, comm=None)
    native = moe_mod.MoEBuffers(rows, cfg, "cuda", prefill=prefill_mode)
    pending = fwd.moe_block(SimpleNamespace(moe=MoEW(router, ex)), w,
                            SimpleNamespace(mixed=x, moe=native), rows)
    expected_y, expected_wts, expected_act = pending[1].clone(), pending[2].clone(), native.act.clone()
    valid = int(native.plan.counts[0])
    logical_items = native.plan.items[:valid].clone()
    assert torch.unique(native.pick).numel() > slots
    assert torch.all(native.pick[:, -1] == experts)
    cache = _cache(ex, slots)
    try:
        cached = CachedExperts(_host(ex), cache, 0)
        buf = moe_mod.MoEBuffers(rows, cfg, "cuda", prefill=prefill_mode)
        layer, buffers = SimpleNamespace(moe=MoEW(router, cached)), SimpleNamespace(mixed=x, moe=buf)
        # Reuse both the expert cache and item-upload scratch through repeated evictions.
        for _ in range(2):
            got = fwd.moe_block(layer, w, buffers, rows)
            assert got[0] == 2
            assert same_tensor_bits(got[1], expected_y)
            assert same_tensor_bits(got[2], expected_wts)
            assert same_tensor_bits(buf.act, expected_act)
            assert torch.equal(buf.pick, native.pick)
            assert torch.equal(buf.plan.items[:valid], logical_items)
            assert torch.equal(buf.plan.counts, native.plan.counts)
    finally:
        cache.close()


def test_main_prefill_serial_and_mtp_use_same_resident_token_oracle():
    """All main and MTP experts miss a two-slot cache; graph opt-in safely selects eager execution."""

    w = _model(seed=7)
    prompt = [5, 17, 99, 250, 1023, 7, 64, 300, 11, 12, 13]
    sampling = Sampling(seed=1234, top_k=20, top_p=0.95)
    native = Engine(w, capacity=128, max_rows=8, prefill_rows=5)
    first = prefill(native, prompt, sampling)
    expected = serial_decode(native, first, 12, sampling).tokens
    offloaded, cache = _offload(w)
    try:
        engine = Engine(offloaded, capacity=128, max_rows=8, prefill_rows=5, graphs=True)
        assert engine.graphs is None
        assert prefill(engine, prompt, sampling) == first
        assert serial_decode(engine, first, 12, sampling).tokens == expected
        assert prefill(engine, prompt, sampling) == first
        assert mtp_decode(engine, first, 12, sampling, depth=2, confidence=0.0).tokens == expected
    finally:
        cache.close()


def test_inference_warm_forward_reuses_expert_windows_in_ordinary_mode():
    """Lazy persistent item scratch survives the caller's inference-mode boundary."""

    w = _model(seed=23)
    tokens = [5, 17, 99, 250, 1023, 7, 64, 300]
    native_buf = Buffers(w, len(tokens), 128, moe_prefill=True)
    expected = fwd.forward(w, State(w, 128, len(tokens)), native_buf, tokens).clone()
    expected_streams = native_buf.streams[:len(tokens)].clone()
    offloaded, cache = _offload(w, slots=2)
    try:
        # The buffers predate inference mode; the first forward lazily creates
        # its item scratch inside that mode and the second reuses the same copy destinations.
        buf = Buffers(offloaded, len(tokens), 128, moe_prefill=True)
        state = State(offloaded, 128, len(tokens))
        with torch.inference_mode():
            got = fwd.forward(offloaded, state, buf, tokens)
            assert same_tensor_bits(got, expected)
            assert same_tensor_bits(buf.streams[:len(tokens)], expected_streams)
        got = fwd.forward(offloaded, state, buf, tokens)
        assert same_tensor_bits(got, expected)
        assert same_tensor_bits(buf.streams[:len(tokens)], expected_streams)
    finally:
        cache.close()


def test_mixed_prompt_decode_forward_matches_resident_logits_and_streams():
    """Two decode windows and two prompt pieces share cache windows without changing either output slice."""

    w = _model(seed=9)
    offloaded, cache = _offload(w, slots=3)
    chains, pieces = [[401, 33, 2048], [5, 6]], [[31, 32, 33, 34, 35, 36, 37], [41, 42, 43]]

    def run(model):
        db, pb = Buffers(model, 8, 128, moe_prefill=True), Buffers(model, 24, 128, prefill=True)
        d = [State(model, 128, 8) for _ in chains]
        p = [State(model, 128, 8) for _ in pieces]
        segs, psegs = fwd.stage(model, db, list(zip(d, chains))), fwd.stage(model, pb, list(zip(p, pieces)))
        ends = [a1 - 1 for _, _, a1 in psegs]
        logits, heads = fwd.compute_mixed(model, segs, db, psegs, pb, ends=ends)
        return (logits[:sum(map(len, chains))].clone(), db.streams[:sum(map(len, chains))].clone(),
                heads[:len(ends)].clone(), pb.streams[:sum(map(len, pieces))].clone())

    try:
        expected, got = run(w), run(offloaded)
        assert all(same_tensor_bits(a, b) for a, b in zip(expected, got))
    finally:
        cache.close()


def test_concurrent_prompt_fills_and_mtp_rounds_match_resident_tokens():
    """The real multi-stream scheduler uses the same hot cache across main, prompt and MTP launches."""

    w = _model(seed=13)
    offloaded, cache = _offload(w, slots=3)
    prompts = [[5, 17, 99, 250, 1023, 7, 64], [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]]

    def run(model):
        decoder = MultiDecoder(model, slots=2, capacity=128, depth=2, confidence=0.0,
                               prefill_rows=5, share=0.5, stop_eos=False)
        streams = [Stream(prompt, 8, Sampling(seed=3 + i, top_k=20, top_p=0.95), draft=True)
                   for i, prompt in enumerate(prompts)]
        for stream in streams:
            decoder.admit(stream)
        for _ in range(100):
            if not decoder.live():
                return [s.out for s in streams]
            decoder.finish(decoder.round())
        pytest.fail("tiny concurrent expert-cache run did not complete within 100 rounds")

    try:
        assert run(offloaded) == run(w)
    finally:
        cache.close()


@pytest.mark.parametrize("depth", [0, 2])
def test_actual_engine_cold_loading_stays_on_host_and_matches_resident_yarn(tmp_path, monkeypatch, depth):
    """Real cold load filters GPU read-ahead, preserves every packed expert, and composes RAM+YaRN flags."""

    _checkpoint(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["max_position_embeddings"] = 256
    config_path.write_text(json.dumps(config))
    monkeypatch.setenv("TENSORFOLD_PREFILL_ROWS", "256")
    options = dict(depth=depth, max_len=128, context_explicit=True, prefetch=False, graphs=False,
                   yarn_factor=2, draft_vocab=None)
    native = FlashNextEngine(tmp_path, **options)
    entry_bytes = native.w.layers[0].moe.experts.bytes_per_expert()
    queued, taken = [], []
    real_queue, real_get = _Reader.queue, _Reader.get

    def queue(reader, names):
        names = tuple(names)
        queued.extend(names)
        return real_queue(reader, names)

    def get(reader, name):
        taken.append(name)
        return real_get(reader, name)

    cached = None
    try:
        with monkeypatch.context() as patcher:
            patcher.setattr(_Reader, "queue", queue)
            patcher.setattr(_Reader, "get", get)
            cached = FlashNextEngine(tmp_path, vram_experts=2 * entry_bytes / 2**30, **options)
        cache = cached.w.meta["expert_cache"]
        assert queued and taken
        assert not any(".switch_mlp." in name or ".shared_expert." in name for name in queued + taken)
        assert any(".shared_expert_gate." in name for name in queued)
        assert cache.gpu_bytes == 2 * entry_bytes
        assert cache.capacity == 2
        pairs = list(zip(native.w.layers, cached.w.layers))
        if depth:
            assert native.w.mtp is not None and cached.w.mtp is not None
            pairs.append((native.w.mtp.layer, cached.w.mtp.layer))
        else:
            assert native.w.mtp is None and cached.w.mtp is None
        expected_host_bytes = 0
        for resident_layer, cached_layer in pairs:
            resident, wrapper = resident_layer.moe.experts, cached_layer.moe.experts
            assert isinstance(wrapper, CachedExperts)
            cold = cache._layers[wrapper.layer_id].source
            assert torch.equal(cold[0], resident.up.cpu())
            assert torch.equal(cold[1], resident.down.cpu())
            # The final logical expert is shared, not a skipped or recomputed projection.
            assert torch.equal(cold[0][-1], resident.up[-1].cpu())
            assert torch.equal(cold[1][-1], resident.down[-1].cpu())
            expected_host_bytes += resident.count * entry_bytes
        assert cache.host_bytes == expected_host_bytes
        assert cached.capacity_plan["vram_experts"]["host_bytes"] == expected_host_bytes
        assert cached.rope.factor == native.rope.factor == 2
        assert same_tensor_bits(cached.w.inv_freq, native.w.inv_freq)
        assert cached.e.graphs is None
        prompt, sampling = [5, 17, 99, 250, 1023, 7, 64, 300], Sampling(seed=99, top_k=20, top_p=0.95)

        def generate(engine, draft):
            tokens = []
            engine.generate(prompt, 8, sampling, lambda new: tokens.extend(new), draft=draft, stop_eos=False)
            return tokens

        expected = generate(native, False)
        assert len(expected) == 8
        assert generate(cached, False) == expected
        assert generate(native, True) == expected
        assert generate(cached, True) == expected
    finally:
        if cached is not None:
            cached.close()
            cached.w.meta["expert_cache"].close()
        native.close()
