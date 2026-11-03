"""Exact native CT attention words and real State/eager/graph entry integration.

The component target uses fixed BF16 projection lookups, not an NVFP4 model.
Actual AEON body/drafter/media/API/performance gates remain separate requirements.
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tensorfold.families.gemma4.cuda import attention
from tensorfold.families.gemma4.cuda.forward import Gemma4Target
from tensorfold.families.gemma4.cuda.graphs import Graphs
from test_gemma_persistent_global_cache import config


def frequencies():
    inverse = torch.pow(1e6, -torch.arange(0, 512, 2, device='cuda', dtype=torch.float32) / 512)
    inverse[64:] = 0
    return inverse


def same(a, b):
    assert torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.mark.parametrize('mode', ['k', 'kv'])
@pytest.mark.parametrize('start,rows', [(0, 1), (31, 5), (511, 5), (65535, 5), (131071, 1), (262143, 1)])
def test_append_words_and_consumer_match_original_raw(mode, start, rows):
    torch.manual_seed(719 + start)
    length, context = start + rows, min(start + rows + 7, 262144)
    raw = torch.randn((context, 2, 512), device='cuda', dtype=torch.bfloat16)
    inv = torch.rsqrt(raw.float().square().mean(-1) + 1e-6)
    kw = torch.randn(512, device='cuda', dtype=torch.bfloat16)
    qw = torch.randn_like(kw)
    freq = frequencies()
    pos = torch.tensor([start], device='cuda', dtype=torch.int32)
    qr = torch.randn((rows, 16 * 512), device='cuda', dtype=torch.bfloat16)
    kr = torch.randn((rows, 2 * 512), device='cuda', dtype=torch.bfloat16)
    keys = torch.full_like(raw, float('nan'))
    values = torch.full_like(raw, float('nan')) if mode == 'kv' else None
    if start:
        attention.prepare_global_cache(raw, inv, kw, freq, keys, values, rows=start)
    q = attention.prep_full(qr, kr, qw, freq, pos, raw, inv, eps=1e-6, round_normalized=False)
    attention.prepare_global_cache(raw, inv, kw, freq, keys, values, rows=rows, pos=pos)
    reference_keys, reference_values = torch.empty_like(raw), torch.empty_like(raw)
    attention._prepare_global[(-(-length // 32), 2)](
        raw, inv, kw, freq, reference_keys, reference_values, length,
        HK=2, D=512, ROT=64, UW=128, BK=32, ROUND=False, **attention.LAUNCH)
    same(keys[:length], reference_keys[:length])
    assert torch.isnan(keys[length:]).all()
    if values is not None:
        same(values[:length], reference_values[:length])
        assert torch.isnan(values[length:]).all()
    cached = dict(prepared_keys=keys) if mode == 'k' else dict(prepared=(keys, values))
    for sequential in (False, True):
        args = dict(chunks=-(-length // 512), round_normalized=False, sequential=sequential)
        expected = attention.full(q, raw, inv, kw, freq, pos, **args)
        actual = attention.full(q, raw, inv, kw, freq, pos, **args, **cached)
        same(actual, expected)


def component(mode):
    """Use actual Target entry/restore methods and Graphs around one attention layer."""
    cfg = replace(config(), layers=1, layer_types=('full_attention',), max_position=2048)
    target = object.__new__(Gemma4Target)
    target.config, target.device = cfg, torch.device('cuda:0')
    target._global_cache, target._global_cache_owner = mode, object() if mode != 'none' else None
    target.tap_layers = (0,)
    generator = torch.Generator(device='cuda').manual_seed(912)
    kw = torch.randn(512, generator=generator, device='cuda', dtype=torch.bfloat16)
    qw = torch.randn(512, generator=generator, device='cuda', dtype=torch.bfloat16)
    qr = torch.randn((cfg.vocab, cfg.heads * 512), generator=generator, device='cuda', dtype=torch.bfloat16)
    kr = torch.randn((cfg.vocab, cfg.global_kv_heads * 512), generator=generator, device='cuda', dtype=torch.bfloat16)
    embedding = torch.zeros((cfg.vocab, cfg.hidden), device='cuda', dtype=torch.bfloat16)
    embedding[:, 0] = torch.arange(cfg.vocab, device='cuda')
    target.embed_tokens = embedding.__getitem__
    target.weights = SimpleNamespace(layers=[SimpleNamespace(k_norm=kw)])
    target._inverse = [frequencies()]

    def layer(hidden, state, rows, chunks, *unused):
        ids = hidden[:, 0].to(torch.int64)
        q = attention.prep_full(qr[ids], kr[ids], qw, target._inverse[0], state.pos,
                                state.raw[0], state.inv[0], eps=cfg.eps, round_normalized=False)
        cache = {}
        if mode != 'none':
            values = None if state.derived_values is None else state.derived_values[0]
            attention.prepare_global_cache(state.raw[0], state.inv[0], kw, target._inverse[0],
                                            state.derived_keys[0], values, rows=rows, pos=state.pos)
            cache = dict(prepared_keys=state.derived_keys[0]) if mode == 'k' else dict(
                prepared=(state.derived_keys[0], values))
        out = attention.full(q, state.raw[0], state.inv[0], kw, target._inverse[0], state.pos,
                              chunks=chunks, round_normalized=False, **cache)
        hidden = out[:, 0, :cfg.hidden].contiguous()
        return hidden, out[:, 1, :cfg.vocab].float(), {0: hidden}

    target._layers = layer
    return target


@pytest.mark.parametrize('mode', ['k', 'kv'])
def test_real_entries_restore_rollback_reset_and_static_graph_replay(mode):
    baseline, cached = component('none'), component(mode)
    raw = baseline.make_state(1025, rows=17)
    state = cached.make_state(1025, rows=17)
    tokens = [(i * 7 + 3) % baseline.config.vocab for i in range(1025)]
    for at in range(0, 510, 17):
        expected = baseline.forward(tokens[at:at + 17], raw)
        actual = cached.forward(tokens[at:at + 17], state)
        same(actual.hidden, expected.hidden)
        assert torch.equal(actual.logits, expected.logits)
    raw_snap, snapshot = raw.snapshot(), state.snapshot()
    assert len(snapshot._layers) == 4  # raw authority only
    pointers = state.derived_keys.data_ptr(), None if state.derived_values is None else state.derived_values.data_ptr()
    graphs = Graphs(cached, state, max_rows=5, span=1)
    for cycle in range(2):
        for count in (5, 3, 1):
            start = raw.position
            expected = baseline.forward(tokens[start:start + count], raw)
            actual = graphs.forward(tokens[start:start + count], state, output_taps=True)
            same(actual.hidden, expected.hidden)
            assert torch.equal(actual.logits, expected.logits)
            raw.rollback(start + 1)
            state.rollback(start + 1)
        raw.restore(raw_snap)
        state.restore(snapshot if cycle == 0 else raw_snap)
        state.derived_keys.fill_(float('nan'))
        if state.derived_values is not None:
            state.derived_values.fill_(float('nan'))
        assert not state.global_cache_ready
    # Eager restore entry also regenerates; old graph addresses remain valid.
    expected = baseline.forward(tokens[510:515], raw)
    actual = cached.forward(tokens[510:515], state)
    same(actual.hidden, expected.hidden)
    assert torch.equal(actual.logits, expected.logits) and state.global_cache_ready
    assert pointers == (state.derived_keys.data_ptr(),
                        None if state.derived_values is None else state.derived_values.data_ptr())
    raw.reset()
    state.reset()
    expected = baseline.forward([2, 5], raw)
    actual = graphs.forward([2, 5], state, output_taps=True)
    same(actual.hidden, expected.hidden)
    assert torch.equal(actual.logits, expected.logits)


@pytest.mark.parametrize('mode', ['k', 'kv'])
def test_derived_state_is_not_rebound_to_another_target(mode):
    one, other = component(mode), component(mode)
    state = one.make_state(128, rows=5)
    with pytest.raises(ValueError, match='another target preparation contract'):
        other.forward([3], state)
    assert state.poisoned


@pytest.mark.parametrize('mode', ['k', 'kv'])
def test_cached_to_raw_target_handoff_rejects_before_mutation(mode):
    raw_target, owner = component('none'), component(mode)
    raw = raw_target.make_state(128, rows=5)
    state = owner.make_state(128, rows=5)
    expected = raw_target.forward([3, 9], raw)
    actual = owner.forward([3, 9], state)
    same(actual.hidden, expected.hidden)
    for staged in (False, True):
        with pytest.raises(ValueError, match='target'):
            if staged:
                raw_target.forward_staged(torch.tensor([4], device='cuda'), state, chunks=1)
            else:
                raw_target.forward([4], state)
        assert state.position == 2 and state.global_cache_ready and not state.poisoned
    expected = raw_target.forward([4], raw)
    actual = owner.forward([4], state)
    same(actual.hidden, expected.hidden)
    assert torch.equal(actual.logits, expected.logits)
