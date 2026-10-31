"""Real target/auxiliary State lifecycle and publication controls on CPU equations."""

import json
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from tensorfold.engine.exact_sampling import Sampling, choose_rows
from tensorfold.families.qwen4_exp.cuda import forward
from tensorfold.families.qwen4_exp.cuda.dspark_admission import inspect, is_dspark
from tensorfold.families.qwen4_exp.cuda.dspark_config import DSparkConfig
from tensorfold.families.qwen4_exp.cuda.dspark_runtime import DSparkRuntime, prepare
from tensorfold.families.qwen4_exp.cuda.dspark_weights import tensor_shapes
from tensorfold.families.qwen4_exp.cuda.state import Buffers, State, KVResizeError
from test_dspark_taps_forward import cpu_weights


def source(path, w):
    raw = {
        "architectures": ["DSparkDraftModel"],
        "model_type": "qwen3",
        "hidden_size": w.cfg.hidden,
        "vocab_size": w.cfg.vocab,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "intermediate_size": 32,
        "num_target_layers": w.cfg.layers,
        "block_size": 3,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000,
        "layer_types": ["full_attention"],
        "dflash_config": {
            "attention_mode": "gqa",
            "projector_type": "dspark",
            "markov_head_type": "vanilla",
            "confidence_head_with_markov": True,
            "markov_rank": 4,
            "mask_token_id": w.cfg.vocab - 1,
            "target_layer_ids": [0],
        },
    }
    c = DSparkConfig.from_dict(raw)
    gen = torch.Generator().manual_seed(2903)
    arrays = {
        name: (torch.randn(shape, generator=gen) * 0.03 + (1 if "norm" in name else 0)).bfloat16()
        for name, shape in tensor_shapes(c).items()
    }
    path.mkdir(exist_ok=True)
    (path / "config.json").write_text(json.dumps(raw))
    save_file(arrays, path / "model.safetensors")
    return {"hidden_size": w.cfg.hidden, "vocab_size": w.cfg.vocab, "num_hidden_layers": w.cfg.layers, "hc_count": 4}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a: NS(multi_processor_count=80))
    baseline = cpu_weights(monkeypatch)
    w = cpu_weights(monkeypatch)
    text = source(tmp_path / "draft", w)
    plan = inspect(tmp_path / "draft", text, tp=(0, 1), chunk_bytes=4096)
    runtime = prepare(w, plan, prefill_rows=8)
    yield w, baseline, runtime, plan
    runtime.close()


def compute(w, st, b, ids, *, cut=None):
    b.ids[: len(ids)].copy_(torch.tensor(ids, dtype=torch.int32))
    b.ple_v.fill_(0.125)
    return forward.compute(w, [(st, 0, len(ids))], b, cuts=() if cut is None else (cut,)).clone()


def bits(a, b):
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def assert_target(a, b, *, physical=True):
    assert a.pos == b.pos
    if physical:
        assert a.cur == b.cur
        bits(a.rec, b.rec)
    else:
        for index, (left, right) in enumerate(zip(a.cur, b.cur, strict=True)):
            bits(a.rec[left, index], b.rec[right, index])
    for n in ("conv", "ple_tail"):
        bits(getattr(a, n), getattr(b, n))
    for x, y in zip(a.kc, b.kc, strict=True):
        for n in ("k", "v", "ks", "vs"):
            torch.testing.assert_close(getattr(x, n)[: a.pos], getattr(y, n)[: b.pos], rtol=0, atol=0)


@pytest.mark.parametrize("prompt", [False, True])
@pytest.mark.parametrize("cache", ["fp16", "int8"])
def test_actual_compute_commit_attached_aux_leaves_target_bits_unchanged(setup, prompt, cache):
    w, baseline, runtime, plan = setup
    a, b = State(w, 16, 8, cache), State(baseline, 16, 8, cache)
    ab, bb = Buffers(w, 8, 16, prefill=prompt), Buffers(baseline, 8, 16, prefill=prompt)
    ids = [4, 7, 19]
    actual, expected = compute(w, a, ab, ids), compute(baseline, b, bb, ids)
    bits(actual, expected)
    assert a.dspark.pending is not None and a.dspark.length == 0
    keep = 3 if prompt else 2
    forward.commit(w, a, ab, 3, keep)
    forward.commit(baseline, b, bb, 3, keep)
    assert_target(a, b)
    assert a.dspark.length == a.pos == keep and a.dspark.pending is None
    assert a.cache_bytes() - b.cache_bytes() == 16 * plan.cache_row_bytes


def test_cut127_style_marker_copy_restore_keeps_projected_prefix_no_feature_history(setup):
    w, _, runtime, plan = setup
    a = State(w, 16, 8, "int8")
    b = Buffers(w, 8, 16, prefill=True)
    cut = forward.Cut(2)
    result = compute(w, a, b, [4, 7, 19], cut=cut)
    snap = forward.cut_snapshot(w, a, b, cut, 0)
    assert snap["dspark_pos"] == snap["pos"] == 2 and not any("features" in k for k in snap)
    assert a.dspark.length == 0 and a.dspark.pending.rows == 3
    forward.commit(w, a, b, 3, 3)
    clone = State(w, 16, 8, "int8")
    clone.copy_prefix(a, 2, 0)
    clone.restore(snap)
    assert clone.pos == clone.dspark.length == 2
    bits(clone.dspark.keys[:, :2], a.dspark.keys[:, :2])
    assert clone.dspark.keys.data_ptr() != a.dspark.keys.data_ptr()
    tail = compute(w, clone, b, [19])
    forward.commit(w, clone, b, 1, 1)
    bits(tail, result[-1:])
    assert_target(a, clone, physical=False)
    bits(a.dspark.keys[:, :3], clone.dspark.keys[:, :3])
    saved = a.snapshot()
    a.reset(w)
    a.restore(saved)
    assert a.pos == a.dspark.length == 3
    assert a.dspark.valid_length >= 3


def test_actual_state_clone_resize_reset_and_cache_byte_accounting(setup):
    w, _, runtime, plan = setup
    a = State(w, 16, 8, "fp16", limit=64)
    b = Buffers(w, 8, 16, prefill=True)
    compute(w, a, b, [4, 7, 19])
    forward.commit(w, a, b, 3, 3)
    child = a.clone()
    assert child.dspark is not a.dspark and child.dspark.target is child
    bits(a.dspark.keys[:, :3], child.dspark.keys[:, :3])
    before = a.cache_bytes()
    delta = a.ensure(20, step=8)
    assert a.capacity == a.dspark.capacity == 24 and delta == a.cache_bytes() - before
    assert a.owned_cache_bytes() == a.cache_bytes()
    bits(a.dspark.keys[:, :3], child.dspark.keys[:, :3])
    a.reset(w)
    assert a.pos == a.dspark.length == 0 and child.dspark.length == 3
    runtime.close_state(child)
    assert child.dspark is None


def test_aux_growth_failure_uses_existing_poison_accounting_and_recovery(setup, monkeypatch):
    w, _, runtime, plan = setup
    a = State(w, 16, 8, "fp16", limit=64)
    old = runtime.provider.resize
    with monkeypatch.context() as fault:
        fault.setattr(runtime.provider, "resize", lambda *a, **k: (_ for _ in ()).throw(MemoryError("aux resize")))
        with pytest.raises(MemoryError, match="aux resize"):
            a.ensure(20, step=8)
    assert isinstance(a._kv_error, KVResizeError)
    assert a.cache_bytes() == a.owned_cache_bytes()
    a.recover_resize(w, 16)
    assert a.capacity == a.dspark.capacity == 16 and a.pos == a.dspark.length == 0
    assert a._resize_error is a._kv_error is None
    assert runtime.provider.resize == old


def test_marker_and_foreign_copy_reject_before_target_snapshot_mutation(setup):
    w, _, runtime, plan = setup
    a = State(w, 16, 8, "fp16")
    b = Buffers(w, 8, 16, prefill=True)
    compute(w, a, b, [4, 7])
    forward.commit(w, a, b, 2, 2)
    snap = a.snapshot()
    before = a.rec.clone(), a.pos
    invalid = dict(snap, dspark_identity="foreign")
    with pytest.raises(ValueError, match="identity"):
        a.restore(invalid)
    bits(before[0], a.rec)
    assert a.pos == before[1]
    invalid = dict(snap, pos=3, dspark_pos=3)
    with pytest.raises(ValueError, match="admission"):
        a.restore(invalid)
    assert a.pos == 2


@pytest.mark.parametrize("sampling", [None, Sampling(17, temperature=0.7, top_k=20, top_p=0.9)])
def test_runtime_uses_original_position_keyed_sampler_and_owns_no_stochastic_q(setup, sampling):
    w, _, runtime, plan = setup
    st = State(w, 16, 8, "fp16")
    b = Buffers(w, 8, 16, prefill=True)
    compute(w, st, b, [4, 7])
    forward.commit(w, st, b, 2, 2)
    seen = []

    def sample(logits, positions, s, *, draft, gathered):
        assert not draft and not gathered and s is sampling
        seen.extend(positions)
        # The ordinary CUDA sampler refuses CPU storage; exercise its shared
        # keyed chooser equation here, then qualify actual e.sample natively.
        if s is None:
            return logits.argmax(-1).tolist()
        values = logits.float().numpy()
        ids = np.broadcast_to(np.arange(values.shape[-1]), values.shape)
        return choose_rows(values, ids, positions, s)

    e = NS(st=st, sample=sample)
    ids = runtime.draft(e, 42, 3, sampling, 0.0)
    assert len(ids) == 3 and seen == [3, 4, 5]
    assert all(0 <= x < w.cfg.vocab for x in ids)
    assert st.pos == st.dspark.length == 2


def test_admission_shared_slots_and_source_identity_mutation_guard(setup, tmp_path):
    w, _, runtime, p = setup
    assert is_dspark(p.directory) and not is_dspark(tmp_path / "absent")
    assert p.resident_bytes == runtime.nbytes
    one = p.capacity_bytes(64, main_rows=8, prompt_rows=8, slots=1)
    six = p.capacity_bytes(64, main_rows=8, prompt_rows=8, slots=6)
    assert one == six  # context capacity is a shared total, not six full windows
    assert p.capacity_bytes(128, main_rows=8, prompt_rows=8, slots=6) > six
    (p.directory / "model.safetensors").touch()
    with pytest.raises(ValueError, match="changed"):
        p.check_source()


def test_prepare_journals_owner_before_fallible_loading_and_close_is_retryable(tmp_path, monkeypatch):
    w = cpu_weights(monkeypatch)
    text = source(tmp_path / "draft", w)
    p = inspect(tmp_path / "draft", text, chunk_bytes=4096)
    from tensorfold.families.qwen4_exp.cuda import dspark_runtime

    def fail(*a, **k):
        assert isinstance(w.dspark, DSparkRuntime) and w.dspark.plan is p
        k["_journal"]["partial"] = torch.ones(8)
        raise MemoryError("load")

    monkeypatch.setattr(dspark_runtime, "load_shard", fail)
    with pytest.raises(MemoryError, match="load"):
        prepare(w, p, prefill_rows=8)
    assert w.dspark.nbytes == 32
    w.dspark.close()
    assert w.dspark.nbytes == 0 and w.dspark.closed
