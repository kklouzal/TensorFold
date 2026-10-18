"""Actual Gemma DFlash adapter under CPU numerical controls, not CUDA qualification."""

import json
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

import torch.nn.functional as F
from safetensors.torch import save_file

from tensorfold.families.gemma4.cuda import dflash
from tensorfold.families.gemma4.dflash_metadata import DraftConfig
from test_gemma_cuda_dflash_metadata import config


def norm(x, weight, eps):
    return (x.float() * (x.float().pow(2).mean(-1, keepdim=True) + eps).rsqrt()).to(x.dtype) * weight


def rotate(x, positions, cfg):
    # Direct Transformers5.6 FP32 frequency construction; independent of the
    # adapter's stored inverse-frequency tensor and rotary helper.
    inverse = 1.0 / (cfg.rope.theta ** (torch.arange(0, cfg.head_dim, 2).float() / cfg.head_dim))
    phase = (inverse[None, :, None] @ positions[None, None, :].float()).transpose(1, 2)
    phase = torch.cat((phase, phase), -1)
    left, right = x.chunk(2, -1)
    return x * phase.cos().to(x.dtype) + torch.cat((-right, left), -1) * phase.sin().to(x.dtype)


def attention(q, k, v, keys, values, length, window, scale, causal):
    result = []
    for stream, (kc, vc) in enumerate(zip(keys, values)):
        start, stop = stream * length, (stream + 1) * length
        key = torch.cat((kc, k[:, start:stop]), 1).repeat_interleave(q.shape[0] // k.shape[0], 0)
        value = torch.cat((vc, v[:, start:stop]), 1).repeat_interleave(q.shape[0] // k.shape[0], 0)
        query_positions = kc.shape[1] + torch.arange(length)[:, None]
        key_positions = torch.arange(key.shape[1])[None]
        visible = (key_positions >= kc.shape[1]) | (query_positions - key_positions <= window)
        if causal:
            visible &= key_positions <= query_positions
        scores = q[:, start:stop].float() @ key.float().transpose(-1, -2) * scale
        probabilities = scores.masked_fill(~visible, -torch.inf).softmax(-1)
        result.append((probabilities @ value.float()).bfloat16().transpose(0, 1).reshape(length, -1))
    return torch.cat(result)


class CpuLinear:
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, value):
        return F.linear(value, self.weight)


def make_fixture(directory, monkeypatch, *, window=8, causal=None, kv_heads=2):
    raw = {**config(), "sliding_window": window, "num_key_value_heads": kv_heads}
    if causal is not None:
        raw["is_causal"] = causal
    cfg = DraftConfig.from_dict(raw)
    generator = torch.Generator().manual_seed(1493)
    arrays = {name: ((torch.randn(shape, generator=generator) * .05).bfloat16() if len(shape) == 2
                     else (torch.rand(shape, generator=generator) * .2 + .9).bfloat16())
              for name, shape in cfg.shapes().items()}
    (directory / "config.json").write_text(json.dumps(raw))
    save_file(arrays, directory / "model.safetensors")

    class Target:
        config = NS(hidden=cfg.hidden, layers=cfg.target_layers, vocab=cfg.vocab)
        device = torch.device("cpu")
        tap_layers = cfg.tap_layers
        embed = torch.randn((cfg.vocab, cfg.hidden), generator=generator).bfloat16()
        head = torch.randn((cfg.vocab, cfg.hidden), generator=generator).bfloat16()
        embedded, projected = [], []

        def raw_embed_tokens(self, ids):
            self.embedded.append(ids.clone())
            return F.embedding(ids.long(), self.embed)

        def embed_tokens(self, ids):
            pytest.fail("the official Torch companion must not receive Gemma-normalized embedding rows")

        def project_logits(self, hidden):
            self.projected.append(hidden.clone())
            return F.linear(hidden, self.head).float().div(30).tanh().mul(30)

    monkeypatch.setattr(dflash, "Plain", CpuLinear)
    monkeypatch.setattr(dflash, "block_attention", attention)
    return cfg, arrays, Target()


def reference_block(taps, anchor, target, cfg, arrays, *, offset=0, length=None):
    """Complete raw-context equations without adapter caches or layer helpers."""
    length = cfg.block if length is None else length
    context = norm(F.linear(taps, arrays["fc.weight"]), arrays["hidden_norm.weight"], cfg.eps)
    ids = torch.tensor([anchor] + [cfg.mask_id] * (length - 1))
    x = F.embedding(ids, target.embed)
    positions = offset + torch.arange(len(taps) + length)
    query_positions, key_positions = positions[-length:, None], positions[None]
    for layer, kind in enumerate(cfg.layer_types):
        base = f"layers.{layer}."
        hidden = norm(x, arrays[base + "input_layernorm.weight"], cfg.eps)
        prefix = base + "self_attn."
        q = F.linear(hidden, arrays[prefix + "q_proj.weight"]).view(length, cfg.heads, cfg.head_dim)
        q = rotate(norm(q, arrays[prefix + "q_norm.weight"], cfg.eps).transpose(0, 1), positions[-length:], cfg)
        source = torch.cat((context, hidden))
        k = F.linear(source, arrays[prefix + "k_proj.weight"]).view(-1, cfg.kv_heads, cfg.head_dim)
        k = rotate(norm(k, arrays[prefix + "k_norm.weight"], cfg.eps).transpose(0, 1), positions, cfg)
        v = F.linear(source, arrays[prefix + "v_proj.weight"]).view(-1, cfg.kv_heads, cfg.head_dim).transpose(0, 1)
        key, value = [a.repeat_interleave(cfg.heads // cfg.kv_heads, 0) for a in (k, v)]
        visible = torch.ones((length, len(positions)), dtype=torch.bool)
        if cfg.causal_layers[layer]:
            visible &= key_positions <= query_positions
        if kind == "sliding_attention":
            visible &= query_positions - key_positions < cfg.sliding_window
            if not cfg.causal_layers[layer]:
                visible &= key_positions - query_positions < cfg.sliding_window
        scores = q.float() @ key.float().transpose(-1, -2) * cfg.head_dim ** -.5
        probabilities = scores.masked_fill(~visible, -torch.inf).softmax(-1)
        attended = (probabilities @ value.float()).bfloat16().transpose(0, 1).reshape(length, -1)
        x = x + F.linear(attended, arrays[prefix + "o_proj.weight"])
        hidden = norm(x, arrays[base + "post_attention_layernorm.weight"], cfg.eps)
        activation = F.silu(F.linear(hidden, arrays[base + "mlp.gate_proj.weight"]))
        activation = activation * F.linear(hidden, arrays[base + "mlp.up_proj.weight"])
        x = x + F.linear(activation, arrays[base + "mlp.down_proj.weight"])
    hidden = norm(x[1:], arrays["norm.weight"], cfg.eps)
    logits = (F.linear(hidden, target.head).float() / 30).tanh() * 30
    best, ids = logits.max(-1)
    return ids.tolist(), (torch.logsumexp(logits, -1) - best).cumsum(0).tolist(), hidden


@pytest.mark.parametrize("window,rows", [(8, 7), (8, 8), (8, 9), (2048, 2047), (2048, 2048), (2048, 2053)])
def test_complete_proposals_match_independent_equations_across_swa_boundary(tmp_path, monkeypatch, window, rows):
    cfg, arrays, target = make_fixture(tmp_path, monkeypatch, window=window)
    draft = dflash.DFlash(tmp_path, target, context_limit=8192)
    taps = torch.randn((rows, 128), generator=torch.Generator().manual_seed(101)).bfloat16()
    draft.add_taps(taps)
    before = draft.snapshot()
    launched = draft.launch_block(2, 3)
    assert launched[0][3] is not None
    ids, parents, scores = draft.finish_tree(launched, rows, 3)
    assert launched[0][3] is None
    expected_ids, expected_scores, expected_hidden = reference_block(taps, 2, target, cfg, arrays)
    assert ids == expected_ids and scores == expected_scores and parents == [-1, 0, 1]
    assert torch.equal(target.projected[-1], expected_hidden)
    assert len(target.embedded) == len(target.projected) == 1
    assert draft.kc[0].shape[1] == min(window - 1, rows) and draft.kc[1].shape[1] == rows
    assert draft.context_end == rows and all(a is b for a, b in zip(draft.kc, before[0]))
    assert draft.finish_tree(launched, rows, 3) == (ids, parents, scores)


@pytest.mark.parametrize("causal", [True, False])
def test_explicit_causal_policy_matches_authoritative_override(tmp_path, monkeypatch, causal):
    cfg, arrays, target = make_fixture(tmp_path, monkeypatch, causal=causal)
    draft = dflash.DFlash(tmp_path, target, context_limit=8192)
    taps = torch.randn((13, 128), generator=torch.Generator().manual_seed(19)).bfloat16()
    draft.add_taps(taps)
    expected, scores, _ = reference_block(taps, 3, target, cfg, arrays)
    assert draft.finish_tree(draft.launch_block(3, 3), 13, 3) == (expected, [-1, 0, 1], scores)


def test_only_accepted_rows_commit_and_swa_eviction_rewinds_by_snapshot(tmp_path, monkeypatch):
    cfg, arrays, target = make_fixture(tmp_path, monkeypatch)
    draft = dflash.DFlash(tmp_path, target, context_limit=8192)
    taps = torch.randn((23, 128), generator=torch.Generator().manual_seed(93)).bfloat16()
    draft.add_taps(taps[:6])
    accepted = draft.snapshot()
    before = [t.clone() for t in accepted[0]]
    fork = draft.add_taps_streams([accepted], [taps[6:19]])[0]
    assert accepted[2:] == (6, 6) and all(torch.equal(a, b) for a, b in zip(accepted[0], before))
    draft.restore(fork)
    assert draft.context_end == 19 and draft.kc[0].shape[1] == 7
    draft.restore(accepted)
    # A target verifier accepted two of the examined rows. Rejected rows never
    # reached add_taps, so appending the chosen prefix reconstructs serial state.
    draft.add_taps(taps[6:8])
    expected, scores, _ = reference_block(taps[:8], 2, target, cfg, arrays)
    assert draft.finish_tree(draft.launch_block(2, 3), 8, 3) == (expected, [-1, 0, 1], scores)
    assert draft.kc[0].shape[1] == 7 and draft.kc[1].shape[1] == 8


def test_multi_stream_partition_empty_rows_and_absolute_skip_positions(tmp_path, monkeypatch):
    cfg, arrays, target = make_fixture(tmp_path, monkeypatch)
    draft = dflash.DFlash(tmp_path, target, context_limit=8192)
    taps = torch.randn((15, 128), generator=torch.Generator().manual_seed(55)).bfloat16()
    initial = draft.snapshot()
    draft.skip(17)
    skipped = draft.snapshot()
    contexts = draft.add_taps_streams([initial, initial, skipped], [taps, taps[:0], taps[:3]])
    assert contexts[1][2:] == (0, 0) and all(k is None for k in contexts[1][0])
    launched = draft.launch_blocks(contexts, [2, 999, 3], 3)
    assert launched[1] is None
    for index, rows, anchor, offset in ((0, 15, 2, 0), (2, 3, 3, 17)):
        expected, scores, _ = reference_block(taps[:rows], anchor, target, cfg, arrays, offset=offset)
        assert draft.finish_tree(launched[index], rows, 3) == (expected, [-1, 0, 1], scores)
    draft.restore(contexts[0])
    unchanged = draft.add_taps_streams([draft.snapshot()], [taps[:0]])[0]
    assert all(a is b for a, b in zip(unchanged[0], draft.kc))


@pytest.mark.parametrize("kv_heads", [1, 2])
def test_sliced_kv_caches_own_compact_retained_storage(tmp_path, monkeypatch, kv_heads):
    _cfg, _arrays, target = make_fixture(tmp_path, monkeypatch, kv_heads=kv_heads)
    draft = dflash.DFlash(tmp_path, target, context_limit=8192)
    taps = torch.ones((15, 128), dtype=torch.bfloat16)
    empty = draft.snapshot()
    contexts = draft.add_taps_streams([empty, empty], [taps[:1], taps])
    for keys, values, _length, _end in contexts:
        for tensor in [*keys, *values]:
            assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
    draft.restore(contexts[1])
    draft.add_taps(taps[:2])
    assert draft.kc[0].shape[1] == 7 and draft.kc[1].shape[1] == 17
    assert all(t.untyped_storage().nbytes() == t.numel() * t.element_size() for t in [*draft.kc, *draft.vc])


@pytest.mark.parametrize("failed_copy", [1, 2])
def test_restore_allocation_failure_keeps_one_coherent_context(tmp_path, monkeypatch, failed_copy):
    _cfg, _arrays, target = make_fixture(tmp_path, monkeypatch)
    draft = dflash.DFlash(tmp_path, target, context_limit=8192)
    taps = torch.ones((12, 128), dtype=torch.bfloat16)
    draft.add_taps(taps[:5])
    original = draft._context
    replacement = draft.add_taps_streams([draft.snapshot()], [taps[5:]])[0]
    calls = 0

    def copy_sequence(value):
        nonlocal calls
        calls += 1
        if calls == failed_copy:
            raise MemoryError("controlled immutable context publication failure")
        return tuple(value)

    with monkeypatch.context() as patch:
        patch.setattr(dflash, "tuple", copy_sequence, raising=False)
        with pytest.raises(MemoryError):
            draft.restore(replacement)
        assert draft._context is original
    draft.restore(replacement)
    assert draft.context_len == draft.context_end == 12


@pytest.mark.parametrize("close_failure", [False, True])
def test_reader_failure_preserves_primary_and_cleanup_transport(tmp_path, monkeypatch, close_failure):
    _cfg, _arrays, target = make_fixture(tmp_path, monkeypatch)
    primary, cleanup = KeyboardInterrupt("read failed"), OSError("reader close failed")
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
        dflash.DFlash(tmp_path, target, context_limit=8192)
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
        assert id(cleanup) in found


def test_successful_reads_do_not_hide_reader_retirement_failure(tmp_path, monkeypatch):
    _cfg, arrays, target = make_fixture(tmp_path, monkeypatch)
    failure = OSError("reader retirement failed after successful reads")

    class Reader:
        def __init__(self, paths):
            pass

        def get(self, name):
            return arrays[name]

        def close(self):
            raise failure

    monkeypatch.setattr(dflash, "SafeTensors", Reader)
    with pytest.raises(OSError) as caught:
        dflash.DFlash(tmp_path, target, context_limit=8192)
    assert caught.value is failure


def test_native_tile_capacity_and_precision_fail_before_payload_allocation(tmp_path, monkeypatch):
    cfg, _arrays, target = make_fixture(tmp_path, monkeypatch)
    for bits in (0, 4, 8, True):
        with pytest.raises(ValueError, match="BF16 storage"):
            dflash.DFlash(tmp_path, target, context_limit=8192, bits=bits)
    raw = {**config(), "num_attention_heads": 8192, "sliding_window": 2048}
    cfg = DraftConfig.from_dict(raw)
    monkeypatch.setattr(dflash, "checked_headers", lambda path: (cfg, {}))

    def unexpected(*args):
        pytest.fail("invalid attention geometry reached payload allocation")

    monkeypatch.setattr(dflash, "SafeTensors", unexpected)
    with pytest.raises(ValueError, match="installed Triton tensor capacity"):
        dflash.DFlash(tmp_path, target, context_limit=8192, block=128)


def test_target_mismatch_and_checkpoint_python_are_checked_without_execution(tmp_path, monkeypatch):
    _cfg, _arrays, target = make_fixture(tmp_path, monkeypatch)
    target.tap_layers = (1, 2)
    with pytest.raises(ValueError, match="trained layer IDs"):
        dflash.DFlash(tmp_path, target, context_limit=8192)
    target.tap_layers = (0, 2)
    target.config.layers = 5
    with pytest.raises(ValueError, match="match the target"):
        dflash.DFlash(tmp_path, target, context_limit=8192)
    target.config.layers = 4
    raw = config()
    raw["auto_map"] = {"AutoModel": "dflash.DFlashDraftModel"}
    (tmp_path / "config.json").write_text(json.dumps(raw))
    (tmp_path / "dflash.py").write_text("raise RuntimeError('checkpoint Python must remain inert')")
    draft = dflash.DFlash(tmp_path, target, context_limit=8192)
    assert draft.bits == 16 and all(v.weight.dtype == torch.bfloat16 for v in draft.linears.values())


def test_invalid_taps_anchors_limits_and_failed_append_preserve_context(tmp_path, monkeypatch):
    _cfg, _arrays, target = make_fixture(tmp_path, monkeypatch)
    draft = dflash.DFlash(tmp_path, target, context_limit=20)
    with pytest.raises(ValueError, match="fit the declared sliding window"):
        dflash.DFlash(tmp_path, target, context_limit=20, block=9)
    taps = torch.ones((5, 128), dtype=torch.bfloat16)
    draft.add_taps(taps)
    before = draft._context
    for invalid in (taps.float(), taps[:, :64], torch.ones((16, 128), dtype=torch.bfloat16), [1]):
        with pytest.raises(ValueError, match="accepted taps"):
            draft.add_taps(invalid)
        assert draft._context is before
    for anchor in (-1, 32, True, 2.0):
        with pytest.raises(ValueError, match="anchors"):
            draft.launch_block(anchor, 3)
    for nodes in (-1, 128, True):
        with pytest.raises(ValueError, match="proposal count"):
            draft.launch_block(2, nodes)
    assert draft.launch_block(2, 0) is None
    with pytest.raises(ValueError, match="prepared block"):
        draft.launch_block(2, 3, block=5)
    with pytest.raises(ValueError, match="admitted window"):
        draft.skip(16)
    assert draft._context is before


def test_proposals_are_the_argmax_and_penalized_streams_count_their_path(tmp_path, monkeypatch):
    """Every stream proposes the drafter's argmax (sampled ones too); a penalized stream batched with a plain one
    proposes the argmax of its penalized rows, row j counting the proposals before it; a history differing from the
    launch is refused."""
    import numpy as np

    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.engine.penalties import History, Penalties

    cfg, arrays, target = make_fixture(tmp_path, monkeypatch)
    draft = dflash.DFlash(tmp_path, target, context_limit=8192)
    taps = torch.randn((13, 128), generator=torch.Generator().manual_seed(29)).bfloat16()
    draft.add_taps(taps)
    sampling = Sampling(seed=77, temperature=1.0, top_k=20, top_p=.95)
    expected, scores, hidden = reference_block(taps, 2, target, cfg, arrays, length=4)
    assert draft.finish_tree(draft.launch_block(2, 3), 14, 3, sampling) == (expected, [-1, 0, 1], pytest.approx(scores))
    logits = ((F.linear(hidden, target.head).float() / 30).tanh() * 30).numpy()
    history = History(Penalties(presence=2.0, repetition=1.5), [expected[0], 7])
    history.extend([expected[1]])
    want = []
    for row in logits:
        want.append(int(np.argmax(history.candidates(row[None], np.arange(cfg.vocab)[None], want)[0])))
    before = draft.snapshot()
    batch = draft.launch_blocks([before, before], [2, 2], 3, histories=[None, history])
    assert draft.finish_tree(batch[0], 14, 3, sampling) == (expected, [-1, 0, 1], pytest.approx(scores))
    assert draft.finish_tree(batch[1], 14, 3, sampling, history)[0] == want != expected
    with pytest.raises(ValueError):
        draft.finish_tree(batch[1], 14, 3, sampling)
