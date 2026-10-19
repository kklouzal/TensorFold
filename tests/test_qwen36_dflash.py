"""Actual DFlash v1 adapter under CPU numerical/provider controls, not CUDA qualification."""
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

import torch.nn.functional as F
from safetensors.torch import save_file

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.qwen3_5_moe.cuda import dflash
from tensorfold.families.qwen3_5_moe.dflash_metadata import DraftConfig, checked_headers, draft_geometry, draft_weights


def config():
    return {"architectures": ["DFlashDraftModel"], "model_type": "qwen3", "hidden_size": 64,
            "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32,
            "num_hidden_layers": 2, "num_target_layers": 4, "vocab_size": 32, "block_size": 4,
            "rms_norm_eps": 1e-6, "max_position_embeddings": 128, "rope_theta": 10000,
            "dflash_config": {"target_layer_ids": [0, 2], "mask_token_id": 31}}


def norm(x, weight, eps):
    inverse = (x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True) + eps).rsqrt()
    return (x.float() * inverse).to(x.dtype) * weight


def rotate(x, positions, cfg, inv):
    frequencies = positions.float().unsqueeze(1) @ inv.float().unsqueeze(0)
    frequencies = torch.cat([frequencies, frequencies], dim=-1)
    cos = (frequencies.cos() * cfg.rope.attention_factor).to(x.dtype).unsqueeze(0)
    sin = (frequencies.sin() * cfg.rope.attention_factor).to(x.dtype).unsqueeze(0)
    left, right = x.chunk(2, dim=-1)
    return x * cos + torch.cat([-right, left], dim=-1) * sin


def attend(q, k, v, keys, values, length, window, scale, causal):
    if causal:
        raise ValueError("v1 block must be noncausal")
    result = []
    for i, (kc, vc) in enumerate(zip(keys, values)):
        low, high = i * length, (i + 1) * length
        key, value = torch.cat([kc, k[:, low:high]], 1), torch.cat([vc, v[:, low:high]], 1)
        group = q.shape[0] // k.shape[0]
        key, value = key.repeat_interleave(group, 0), value.repeat_interleave(group, 0)
        probability = (q[:, low:high].float() @ key.float().transpose(-1, -2) * scale).softmax(-1)
        out = (probability @ value.float()).to(q.dtype).transpose(0, 1).reshape(length, -1)
        result.append(out)
    return torch.cat(result)


class CpuLinear:
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, value):
        return F.linear(value, self.weight)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    raw = config()
    cfg = DraftConfig.from_dict(raw)
    generator = torch.Generator().manual_seed(1493)
    arrays = {name: ((torch.randn(shape, generator=generator) * .05).bfloat16() if len(shape) == 2
                     else torch.ones(shape, dtype=torch.bfloat16)) for name, shape in cfg.shapes().items()}
    (tmp_path / "config.json").write_text(json.dumps(raw))
    save_file(arrays, tmp_path / "model.safetensors")

    class Target:
        config = NS(hidden=cfg.hidden, layers=cfg.target_layers, vocab=cfg.vocab)
        norm = torch.ones(cfg.hidden, dtype=torch.bfloat16)
        embed = CpuLinear(torch.randn((cfg.vocab, cfg.hidden), generator=generator).bfloat16())
        head = CpuLinear(torch.randn((cfg.vocab, cfg.hidden), generator=generator).bfloat16())
        binding = None

        def bind_draft_taps(self, layers):
            if self.binding is not None and self.binding != layers:
                raise ValueError("conflicting one-time target tap binding")
            self.binding = tuple(layers)

    monkeypatch.setattr(dflash, "Plain", CpuLinear)
    monkeypatch.setattr(dflash, "embedding", lambda ids, table: F.embedding(ids.long(), table.weight))
    monkeypatch.setattr(dflash, "block_attention", attend)
    target = Target()
    return tmp_path, raw, cfg, arrays, target


def test_published_architecture_is_separate_from_mtp_and_dflash2():
    raw = config()
    for architecture in ("DFlash2DraftModel", "Qwen3_5MoeForCausalLM"):
        with pytest.raises(ValueError, match="DFlash v1"):
            DraftConfig.from_dict({**raw, "architectures": [architecture]})


@pytest.mark.parametrize("change", [{"attention_bias": True}, {"hidden_act": "gelu"}, {"sliding_window": 128},
                                   {"dtype": "float16"}, {"torch_dtype": "float32"},
                                   {"dtype": "bfloat16", "torch_dtype": "float16"},
                                   {"quantization_config": {"quant_method": "fp8"}},
                                   {"layer_types": ["full_attention"]}, {"head_dim": 30},
                                   {"dflash_config": {"target_layer_ids": [2, 0], "mask_token_id": 31}},
                                   {"dflash_config": {"target_layer_ids": [0, 0], "mask_token_id": 31}},
                                   {"dflash_config": {"target_layer_ids": [0, 2], "mask_token_id": 32}}])
def test_unsupported_or_inconsistent_declarations_refuse_before_loading(change):
    with pytest.raises(ValueError):
        DraftConfig.from_dict({**config(), **change})


@pytest.mark.parametrize("modern,outer", [(True, None), (False, 10000000), (True, 10000000)])
def test_rotary_theta_preserves_the_declared_representation(modern, outer):
    raw = config()
    raw.pop("rope_theta")
    if outer is not None:
        raw["rope_theta"] = outer
    if modern:
        raw["rope_parameters"] = {"rope_type": "default", "rope_theta": 10000000}
    assert DraftConfig.from_dict(raw).rope.theta == 10000000


def test_conflicting_theta_refuses_and_present_empty_modern_policy_does_not_fall_back():
    raw = {**config(), "rope_parameters": {"rope_theta": 10000000}}
    with pytest.raises(ValueError, match="theta declarations disagree"):
        DraftConfig.from_dict(raw)
    raw["rope_parameters"] = {}
    raw["rope_scaling"] = {"rope_type": "yarn", "factor": 8, "original_max_position_embeddings": 16}
    assert DraftConfig.from_dict(raw).rope.rope_type == "default"


@pytest.mark.parametrize("declaration", [{"partial_rotary_factor": True},
                                        {"rope_parameters": {"partial_rotary_factor": True}},
                                        {"rope_scaling": {"partial_rotary_factor": True}},
                                        {"partial_rotary_factor": True,
                                         "rope_parameters": {"partial_rotary_factor": 1}}])
def test_json_boolean_rotary_fraction_is_not_numeric_one(declaration):
    with pytest.raises(ValueError, match="full-head rotary"):
        DraftConfig.from_dict({**config(), **declaration})


def test_exact_headers_and_weight_estimator_preserve_bf16_default(fixture):
    directory, raw, cfg, arrays, target = fixture
    assert checked_headers(directory)[0] == cfg
    estimate = draft_weights(directory)
    assert estimate.resident == sum(t.numel() * t.element_size() for t in arrays.values()) + cfg.head_dim // 2 * 4
    assert estimate.staging == max(t.numel() * t.element_size() for t in arrays.values())
    draft = dflash.DFlash(directory, target, context_limit=64)
    assert draft.bits == 16 and target.binding == (0, 2)
    assert all(torch.equal(draft.linears[k].weight if v.ndim == 2 else draft.weights[k], v) for k, v in arrays.items())
    geometry = draft_geometry(raw, streams=2, kept=3, prompt_rows=8)
    per_token = 2 * cfg.layers * cfg.kv_heads * cfg.head_dim * 2
    assert geometry.bytes_at(33) - geometry.bytes_at(32) == (2 * 2 + 3 + 1) * per_token


def test_stochastic_chains_keep_request_width_when_another_context_has_no_proposal_room(fixture):
    directory, _raw, _cfg, _arrays, target = fixture
    draft = dflash.DFlash(directory, target, context_limit=64)
    draft.add_taps(torch.ones((3, 128), dtype=torch.bfloat16))
    snap = draft.snapshot()
    policy = Sampling(seed=71, temperature=.7, top_k=20, top_p=.95)
    first = draft.propose_chain(2, 4, 3, policy)
    assert len(first.tokens) == 3 and first.first_position == 4
    draft.add_taps(torch.ones((60, 128), dtype=torch.bfloat16))
    near = draft.snapshot()
    launched = draft.launch_chains([snap, near], [2, 2], 3)
    assert launched[1] is None
    again = draft.finish_chain(launched[0], 4, 3, policy)
    assert again.tokens == first.tokens
    assert all(np.array_equal(a.ids, b.ids) and np.array_equal(a.probs, b.probs)
               for a, b in zip(again.q_rows, first.q_rows))


@pytest.mark.parametrize("bits", [4, 8, 16])
@pytest.mark.parametrize("floating", [False, True])
def test_drafter_precision_copies_only_a_floating_head_and_accounts_for_its_ownership(fixture, monkeypatch,
                                                                                  bits, floating):
    from tensorfold.cuda.requant import resident_bytes

    directory, _raw, cfg, arrays, target = fixture
    copies, projections = [], []
    if not floating:
        target.head = NS(n=cfg.vocab)                          # stored quantized head: borrowed, no floating weight
    original = target.head

    def copy_head(source, kind):
        copies.append((source, kind))
        return NS(n=cfg.vocab)

    def copy_projection(value, kind):
        projections.append(kind)
        return CpuLinear(value)

    monkeypatch.setattr(dflash, "Head", copy_head)
    monkeypatch.setattr(dflash, "requantize", copy_projection)
    draft = dflash.DFlash(directory, target, context_limit=64, bits=bits)
    kind = {4: "nvfp4", 8: "fp8"}.get(bits)
    assert target.head is original                            # proposal precision never replaces the target head
    assert projections == ([kind] * sum(w.ndim == 2 for w in arrays.values()) if kind else [])
    assert copies == ([(original, kind)] if floating and kind else [])
    assert (draft.target_head is original) == (not floating or kind is None)
    with_head = draft_weights(directory, bits=bits)
    borrowed = draft_weights(directory, bits=bits, private_head=False)
    assert with_head.resident - borrowed.resident == (resident_bytes(kind, cfg.vocab, cfg.hidden) if kind else 0)


@pytest.mark.parametrize("rows,columns", [(64, 64), (128, 64), (192, 64), (128, 128)])
def test_shared_affine_packer_publishes_canonical_single_group_strides(rows, columns):
    from tensorfold.cuda.kernels import qmm

    generator = torch.Generator().manual_seed(rows + columns)
    words = torch.randint(-(2**31), 2**31 - 1, (rows, columns // 8), dtype=torch.int32, generator=generator)
    scales = torch.rand((rows, columns // 64), generator=generator).bfloat16()
    biases = torch.randn((rows, columns // 64), generator=generator).bfloat16()
    packed = qmm.pack(words, scales, biases, 64)
    padded = -(-rows // 128) * 128
    assert packed.scales.stride() == packed.biases.stride() == (padded, 1)
    assert packed.scales.shape == (columns // 64, padded)
    assert all(torch.equal(actual, expected) for actual, expected in zip(qmm.unpack(packed), (words, scales, biases)))


def test_bad_header_count_fails_without_materializing_scalar_layers(fixture):
    directory, raw, _cfg, _arrays, _target = fixture
    raw["num_hidden_layers"] = 2**31 - 1
    (directory / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="tensor count"):
        checked_headers(directory)


def test_wrong_target_and_conflicting_capture_binding_refuse(fixture):
    directory, _raw, _cfg, _arrays, target = fixture
    target.config = NS(hidden=64, layers=5, vocab=32)
    with pytest.raises(ValueError, match="must match the target"):
        dflash.DFlash(directory, target, context_limit=64)
    target.config.layers = 4
    target.binding = (1, 3)
    with pytest.raises(ValueError, match="conflicting"):
        dflash.DFlash(directory, target, context_limit=64)


def test_checkpoint_auto_map_python_is_inert(fixture):
    directory, raw, _cfg, _arrays, target = fixture
    raw["auto_map"] = {"AutoModel": "dflash.DFlashDraftModel"}
    (directory / "config.json").write_text(json.dumps(raw))
    (directory / "dflash.py").write_text("raise RuntimeError('checkpoint Python must never execute')")
    assert dflash.DFlash(directory, target, context_limit=64).tap_layers == (0, 2)


def test_native_tile_capacity_refuses_before_payload_allocation(fixture, monkeypatch):
    directory, raw, _cfg, _arrays, target = fixture
    raw.update(num_attention_heads=8192)
    cfg = DraftConfig.from_dict(raw)
    monkeypatch.setattr(dflash, "checked_headers", lambda path: (cfg, {}))

    def unexpected(*args):
        pytest.fail("unsupported Triton geometry reached checkpoint payload allocation")
    monkeypatch.setattr(dflash, "SafeTensors", unexpected)
    with pytest.raises(ValueError, match="installed Triton tensor capacity"):
        dflash.DFlash(directory, target, context_limit=64, block=128)


@pytest.mark.parametrize("close_failure", [False, True])
def test_reader_failure_preserves_primary_and_retirement_outcome(fixture, monkeypatch, close_failure):
    directory, _raw, _cfg, _arrays, target = fixture
    primary, cleanup, prior = KeyboardInterrupt("weight read failed"), OSError("close failed"), EOFError("prior")
    primary.__cause__ = prior
    events = []

    class Reader:
        def __init__(self, paths):
            events.append("open")

        def get(self, name):
            raise primary

        def close(self):
            events.append("close")
            if close_failure:
                raise cleanup

    monkeypatch.setattr(dflash, "SafeTensors", Reader)
    with pytest.raises(KeyboardInterrupt) as caught:
        dflash.DFlash(directory, target, context_limit=64)
    assert caught.value is primary and events == ["open", "close"]
    if close_failure:
        pending, found = [primary], set()
        while pending:
            error = pending.pop()
            if id(error) in found:
                continue
            found.add(id(error))
            pending.extend(e for e in (error.__cause__, error.__context__) if e is not None)
            if isinstance(error, BaseExceptionGroup):
                pending.extend(error.exceptions)
        assert id(cleanup) in found and id(prior) in found
    else:
        assert primary.__cause__ is prior


def test_context_projection_snapshot_rewind_and_stream_partition(fixture):
    directory, _raw, cfg, arrays, target = fixture
    draft = dflash.DFlash(directory, target, context_limit=64)
    taps = torch.linspace(-1, 1, 5 * 128).reshape(5, 128).bfloat16()
    draft.add_taps(taps[:3])
    saved = draft.snapshot()
    original = [t.clone() for t in saved[0]]
    draft.add_taps(taps[3:])
    whole = draft.snapshot()
    draft.restore(saved)
    assert draft.context_end == 3 and all(torch.equal(a, b) for a, b in zip(draft.kc, original))
    draft.add_taps(taps[3:])
    assert all(torch.equal(a, b) for a, b in zip(draft.kc, whole[0]))
    first = ([None] * cfg.layers, [None] * cfg.layers, 0, 0)
    contexts = draft.add_taps_streams([first, first], [taps, taps[:2]])
    assert all(torch.equal(a, b) for a, b in zip(contexts[0][0], whole[0]))
    for layer in range(cfg.layers):
        projected = norm(F.linear(taps, arrays["fc.weight"]), arrays["hidden_norm.weight"], cfg.eps)
        key = F.linear(projected, arrays[f"layers.{layer}.self_attn.k_proj.weight"])
        key = norm(key.view(5, cfg.kv_heads, cfg.head_dim), arrays[f"layers.{layer}.self_attn.k_norm.weight"], cfg.eps)
        expected = rotate(key.transpose(0, 1), torch.arange(5), cfg, draft.inv_freq)
        assert torch.equal(expected, draft.kc[layer])


def test_noncausal_block_proposals_do_not_mutate_committed_context(fixture):
    directory, _raw, cfg, _arrays, target = fixture
    draft = dflash.DFlash(directory, target, context_limit=64)
    taps = torch.randn((3, 128), generator=torch.Generator().manual_seed(3)).bfloat16()
    draft.add_taps(taps)
    before = draft.snapshot()
    launched = draft.launch_block(2, 3)
    ids, parents, scores = draft.finish_tree(launched, 3, 3)
    assert len(ids) == 3 and parents == [-1, 0, 1]
    assert all(0 <= value < cfg.vocab for value in ids) and all(a <= b for a, b in zip([0, *scores], scores))
    assert draft.context_end == 3 and all(a is b for a, b in zip(draft.kc, before[0]))
    assert draft.finish_tree(launched, 3, 3) == (ids, parents, scores)
    batch = draft.launch_blocks([before, before], [2, 2], 3)
    assert draft.finish_tree(batch[0], 3, 3) == draft.finish_tree(batch[1], 3, 3) == (ids, parents, scores)


@pytest.mark.parametrize("failed_copy", [1, 2])
def test_failed_restore_copy_leaves_previous_context_coherent_and_usable(fixture, monkeypatch, failed_copy):
    directory, _raw, _cfg, _arrays, target = fixture
    draft = dflash.DFlash(directory, target, context_limit=64)
    taps = torch.ones((5, 128), dtype=torch.bfloat16)
    draft.add_taps(taps[:3])
    saved, before = draft.snapshot(), draft._context
    expected = draft.propose_tree(2, 3, 3)
    replacement = draft.add_taps_streams([saved], [taps[3:]])[0]
    calls = 0
    failure = MemoryError("controlled context sequence copy failure")

    def copy_sequence(value):
        nonlocal calls
        calls += 1
        if calls == failed_copy:
            raise failure
        return tuple(value)

    with monkeypatch.context() as patch:
        patch.setattr(dflash, "tuple", copy_sequence, raising=False)
        with pytest.raises(MemoryError) as caught:
            draft.restore(replacement)
        assert caught.value is failure and draft._context is before
        assert draft.context_end == 3 and draft.propose_tree(2, 3, 3) == expected
    draft.restore(replacement)
    assert draft.context_end == 5 and draft.context_len == 5


def reference_logits(taps, anchor, target, cfg, arrays, inv):
    """Direct published v1 equations over complete raw context; no adapter cache/layer helper."""
    context = norm(F.linear(taps, arrays["fc.weight"]), arrays["hidden_norm.weight"], cfg.eps)
    x = F.embedding(torch.tensor([anchor] + [cfg.mask_id] * (cfg.block - 1)), target.embed.weight)
    context_pos, noise_pos = torch.arange(len(taps)), torch.arange(len(taps), len(taps) + cfg.block)
    for layer in range(cfg.layers):
        base = f"layers.{layer}."
        h = norm(x, arrays[base + "input_layernorm.weight"], cfg.eps)
        attention = base + "self_attn."
        query = F.linear(h, arrays[attention + "q_proj.weight"]).view(cfg.block, cfg.heads, cfg.head_dim)
        query = rotate(norm(query, arrays[attention + "q_norm.weight"], cfg.eps).transpose(0, 1), noise_pos, cfg, inv)
        all_keys = F.linear(torch.cat([context, h]), arrays[attention + "k_proj.weight"])
        all_keys = norm(all_keys.view(-1, cfg.kv_heads, cfg.head_dim), arrays[attention + "k_norm.weight"], cfg.eps)
        key = rotate(all_keys.transpose(0, 1), torch.cat([context_pos, noise_pos]), cfg, inv)
        value = F.linear(torch.cat([context, h]), arrays[attention + "v_proj.weight"])
        value = value.view(-1, cfg.kv_heads, cfg.head_dim).transpose(0, 1)
        group = cfg.heads // cfg.kv_heads
        key, value = key.repeat_interleave(group, 0), value.repeat_interleave(group, 0)
        probabilities = (query.float() @ key.float().transpose(-1, -2) * cfg.head_dim**-.5).softmax(-1)
        attended = (probabilities @ value.float()).bfloat16().transpose(0, 1).reshape(cfg.block, -1)
        x = x + F.linear(attended, arrays[attention + "o_proj.weight"])
        h = norm(x, arrays[base + "post_attention_layernorm.weight"], cfg.eps)
        activation = F.silu(F.linear(h, arrays[base + "mlp.gate_proj.weight"]))
        activation *= F.linear(h, arrays[base + "mlp.up_proj.weight"])
        x = x + F.linear(activation, arrays[base + "mlp.down_proj.weight"])
    h = norm(x[1:], arrays["norm.weight"], cfg.eps)
    return F.linear(h, target.head.weight).float()


def reference_block(taps, anchor, target, cfg, arrays, inv):
    logits = reference_logits(taps, anchor, target, cfg, arrays, inv)
    best, ids = logits.max(-1)
    return ids.tolist(), (torch.logsumexp(logits, -1) - best).cumsum(0).tolist()


@pytest.mark.parametrize("yarn", [False, True])
def test_complete_proposals_match_direct_published_block_equations(fixture, yarn):
    directory, raw, cfg, arrays, target = fixture
    if yarn:
        raw["rope_scaling"] = {"rope_type": "yarn", "factor": 8, "original_max_position_embeddings": 16,
                               "beta_fast": 32., "beta_slow": 1.}
        (directory / "config.json").write_text(json.dumps(raw))
        cfg = DraftConfig.from_dict(raw)
    draft = dflash.DFlash(directory, target, context_limit=64)
    taps = torch.randn((3, 128), generator=torch.Generator().manual_seed(101)).bfloat16()
    draft.add_taps(taps)
    ids, parents, scores = draft.finish_tree(draft.launch_block(2, 3), 3, 3)
    expected_ids, expected_scores = reference_block(taps, 2, target, cfg, arrays, draft.inv_freq)
    assert ids == expected_ids and scores == expected_scores and parents == [-1, 0, 1]


def test_proposals_are_the_argmax_and_penalized_streams_count_their_path(fixture):
    """Every stream proposes the drafter's argmax (sampled ones too); a penalized stream batched with a plain one
    proposes the argmax of its penalized rows, row j counting the proposals before it; a history differing from the
    launch is refused."""
    import numpy as np

    from tensorfold.engine.penalties import History, Penalties

    directory, _raw, cfg, arrays, target = fixture
    draft = dflash.DFlash(directory, target, context_limit=64)
    taps = torch.randn((3, 128), generator=torch.Generator().manual_seed(101)).bfloat16()
    draft.add_taps(taps)
    expected_ids, expected_scores = reference_block(taps, 2, target, cfg, arrays, draft.inv_freq)
    sampling = Sampling(seed=77, temperature=1.0, top_k=20, top_p=.95)
    assert draft.finish_tree(draft.launch_block(2, 3), 10, 3, sampling) == (
        expected_ids, [-1, 0, 1], pytest.approx(expected_scores))
    logits = reference_logits(taps, 2, target, cfg, arrays, draft.inv_freq).numpy()
    history = History(Penalties(presence=2.0, repetition=1.5), [expected_ids[0], 7])
    history.extend([expected_ids[1]])
    want = []
    for row in logits:
        scores = history.candidates(row[None], np.arange(cfg.vocab)[None], want)[0]
        want.append(int(np.argmax(scores)))
    before = draft.snapshot()
    batch = draft.launch_blocks([before, before], [2, 2], 3, histories=[None, history])
    assert draft.finish_tree(batch[0], 10, 3, sampling) == (expected_ids, [-1, 0, 1], pytest.approx(expected_scores))
    assert draft.finish_tree(batch[1], 10, 3, sampling, history)[0] == want != expected_ids
    with pytest.raises(ValueError):
        draft.finish_tree(batch[1], 10, 3, sampling)


class FixedChains(dflash.speculative.Chains):
    """Already validated host packets with fixed scores; use the production shared cutoff."""

    def __init__(self, ids, scores):
        self._sampled = False
        self.host, self.histories, self.scores = (ids, scores, None), [None] * len(ids), scores


def test_proposals_stop_where_the_drafters_cumulative_score_exceeds_the_cut(fixture):
    directory, _raw, _cfg, _arrays, target = fixture
    chains = FixedChains([[5, 6, 7, 8], [9, 10, 11, 12]], [[0.5, 2.9, 3.1, 4.0], [3.5, 3.6, 4.0, 9.0]])
    for stop, first, second in ((3.0, [5, 6], [9]), (2.9, [5, 6], [9]), (None, [5, 6, 7, 8], [9, 10, 11, 12]),
                                (0.0, [5], [9]), (10.0, [5, 6, 7, 8], [9, 10, 11, 12])):  # at the cut (2.9) stays
        draft = dflash.DFlash(directory, target, context_limit=64, stop_score=stop)
        for row, want in ((0, first), (1, second)):
            ids, parents, kept = draft.finish_tree((chains, row), 0, 4)
            assert ids == want and parents == list(range(-1, len(want) - 1))
            assert kept == chains.scores[row][:len(want)]
        assert draft.finish_tree((chains, 0), 0, 1)[0] == [5]               # the node budget still applies first
    assert dflash.DFlash(directory, target, context_limit=64).stop_score == dflash.STOP_SCORE
    for bad in (-1.0, float("inf"), float("nan"), True, "3"):
        with pytest.raises(ValueError, match="stop score"):
            dflash.DFlash(directory, target, context_limit=64, stop_score=bad)
