"""Persisted affine authority checked against native packing and model replies."""
import json

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda import experts as grouped, host_experts  # noqa: E402
from tensorfold.cuda.packed_experts import PackedExpertCache  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.reader import _Reader  # noqa: E402
from test_flashnext_tp import _checkpoint, _save  # noqa: E402
from test_host_experts import inputs  # noqa: E402


@pytest.mark.parametrize("ids,shared", [([2, 0], False), ([1, 2], True)])
def test_persisted_rank_local_bank_equals_native_cuda_pack(tmp_path, ids, shared, monkeypatch):
    base, tensors = "mtp.layers.0.mlp", {}
    for p, (n, k) in zip(("gate_proj", "up_proj", "down_proj"), ((32, 64), (32, 64), (64, 32))):
        for kind, count in (("switch_mlp", 3), ("shared_expert", 1)):
            for component, tensor in zip(("weight", "scales", "biases"), inputs(count, n, k, 32, 91 + count)):
                tensors[f"{base}.{kind}.{p}.{component}"] = tensor if count == 3 else tensor[0]
    _save(tmp_path / "model.safetensors", tensors)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {name: "model.safetensors" for name in tensors}}))
    cache = PackedExpertCache(tmp_path / "packed", 1 << 20)
    reader = _Reader(tmp_path, "cpu")
    try:
        host_experts.load_host_experts(reader, base, packed_cache=cache, owned_ids=ids, include_shared=shared)
        def forbidden(*args, **kwargs):
            pytest.fail("verified cache hit unexpectedly repacked affine expert bits")
        monkeypatch.setattr(host_experts, "pack_cpu", forbidden)
        cached = host_experts.load_host_experts(reader, base, packed_cache=cache,
                                               owned_ids=ids, include_shared=shared)
    finally:
        reader.close()
    order = ids + ([3] if shared else [])
    native_inputs = []
    for projection in ("gate_proj", "up_proj", "down_proj"):
        native_inputs.append(tuple(torch.cat((tensors[f"{base}.switch_mlp.{projection}.{part}"],
                                               tensors[f"{base}.shared_expert.{projection}.{part}"][None]))
                                   [order].contiguous().cuda() for part in ("weight", "scales", "biases")))
    native = grouped.make(native_inputs[:2], native_inputs[2], 32)
    assert torch.equal(cached.up, native.up.cpu()) and torch.equal(cached.down, native.down.cpu())


def test_complete_engine_cache_hit_keeps_greedy_sampled_serial_and_mtp_tokens(tmp_path, monkeypatch):
    _checkpoint(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["max_position_embeddings"] = 256
    config_path.write_text(json.dumps(config))
    monkeypatch.setenv("TENSORFOLD_PREFILL_ROWS", "256")
    options = dict(depth=2, max_len=128, context_explicit=True, prefetch=False, graphs=False,
                   yarn_factor=2, draft_vocab=None, confidence=0.0)
    prompt = [5, 17, 99, 250, 1023, 7, 64, 300]
    samplings = (Sampling(seed=7, temperature=0.0), Sampling(seed=99, top_k=20, top_p=0.95))
    def replies(engine):
        results = []
        for sampling in samplings:
            for draft in (False, True):
                tokens = []
                engine.generate(prompt, 8, sampling, lambda new: tokens.extend(new),
                                draft=draft, stop_eos=False)
                assert len(tokens) == 8
                results.append(tokens)
        assert results[0] == results[1] and results[2] == results[3]
        return results
    resident = FlashNextEngine(tmp_path, **options)
    try:
        expected = replies(resident)
        entry = resident.w.layers[0].moe.experts.bytes_per_expert()
    finally:
        resident.close()
    cache = PackedExpertCache(tmp_path / "packed", 512 << 20)
    original = host_experts.load_host_experts
    def load(reader, name, **kw):
        kw["packed_cache"] = cache
        return original(reader, name, **kw)
    monkeypatch.setattr(host_experts, "load_host_experts", load)
    for warm in (False, True):
        if warm:
            def forbidden(*args, **kwargs):
                pytest.fail("complete engine cache hit unexpectedly repacked")
            monkeypatch.setattr(host_experts, "pack_cpu", forbidden)
        engine = FlashNextEngine(tmp_path, vram_experts=2 * entry / 2**30, **options)
        try:
            assert replies(engine) == expected
        finally:
            engine.close()
    assert len(list(cache.root.glob("*.tfc"))) == 3  # main DeltaNet, attention, MTP
