"""Independent pinned Speculators0.8.0 first-layer oracle and cache boundaries.

Oracle boundary from Apache2 Speculators commit894e71f96e0550c3b4cf3915fe6211aab09cadbb,
models/eagle3/model_definitions.py (first-layer fused-input adapter). HF owns the
attention/norm/MLP algorithms on both sides; no checkpoint remote Python runs.
"""
import copy

# Optional native runtime imports follow importorskip for fresh CPU checkouts.
# ruff: noqa: E402
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')
from transformers import DynamicCache, LlamaConfig
from transformers.masking_utils import create_sliding_window_causal_mask
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm

from tensorfold.families.gemma4.cuda.eagle3 import Eagle3Model, EagleChains, EagleState, Gemma4Eagle3Head
from tensorfold.families.gemma4.eagle3_metadata import Eagle3Config


class PrimaryFirstLayer(LlamaDecoderLayer):
    """Separately constructed native oracle with the primary source's operations."""
    def __init__(self, config):
        super().__init__(config, 0)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn.q_proj = torch.nn.Linear(2 * config.hidden_size, config.num_attention_heads * config.head_dim, bias=False)
        self.self_attn.k_proj = torch.nn.Linear(2 * config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.self_attn.v_proj = torch.nn.Linear(2 * config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)

    def forward(self, hidden_states, attention_mask, position_embeddings, cache):
        embeds, hidden = hidden_states.split(hidden_states.shape[-1] // 2, dim=-1)
        embeds = self.input_layernorm(embeds)
        hidden = self.hidden_norm(hidden)
        residual = hidden
        hidden_states, _ = self.self_attn(hidden_states=torch.cat([embeds, hidden], dim=-1),
            attention_mask=attention_mask, position_embeddings=position_embeddings, past_key_values=cache)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return residual + self.mlp(hidden_states)


def model():
    torch.manual_seed(313)
    config = LlamaConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, vocab_size=40,
        max_position_embeddings=131072, rms_norm_eps=1e-6, sliding_window=8,
        layer_types=['sliding_attention'], rope_parameters={'rope_type': 'default', 'rope_theta': 10000.})
    config._attn_implementation = 'sdpa'
    result = Eagle3Model(config, draft_vocab=10).eval().to(torch.bfloat16)
    result.d2t.copy_(torch.arange(10) * 2)
    result.t2d[torch.arange(10) * 3] = True
    return result


@pytest.mark.parametrize('position', [1, 2047, 2048, 131068])
def test_native_primary_layer_oracle_and_recurrent_output(position):
    actual = model()
    oracle = PrimaryFirstLayer(actual.config).eval().to(torch.bfloat16)
    oracle.load_state_dict(actual.layers[0].state_dict())
    cache, reference_cache = DynamicCache(config=actual.config), DynamicCache(config=actual.config)
    hidden = actual.project(torch.randn(1, 1, 48, dtype=torch.bfloat16))
    with torch.inference_mode():
        for step in range(3):
            ids, positions = torch.tensor([[step + 1]]), torch.tensor([[position + step]])
            inputs = torch.cat((actual.embed_tokens(ids), hidden), dim=-1)
            mask = create_sliding_window_causal_mask(actual.config, inputs, None, reference_cache,
                position_ids=positions, allow_is_causal_skip=False)
            expected = actual.norm(oracle(inputs, mask, actual.rotary(inputs, positions), reference_cache))
            got = actual.step(ids, hidden, cache, positions)
            assert torch.equal(got, expected)
            assert torch.equal(actual.lm_head(got), actual.lm_head(expected))
            hidden = expected


def head():
    result = Gemma4Eagle3Head.__new__(Gemma4Eagle3Head)
    result.model, result.device, result.context_limit = model(), torch.device('cpu'), 128
    result.cfg = SimpleNamespace(hidden=16, vocab=40)
    result.block = 4
    return result


def test_shifted_append_chunk_continuation_snapshot_and_proposal_no_commit():
    h = head()
    taps = torch.randn(11, 48, dtype=torch.bfloat16)
    ids = list(range(11))
    state = EagleState(h)
    h.append(state, taps[:1], token_ids=ids[:1], prompt=True)
    assert state.length == 1 and state.cache.get_seq_length() == 0
    h.append(state, taps[1:5], token_ids=ids[1:5], prompt=True)
    checkpoint = state.snapshot()
    h.append(state, taps[5:], token_ids=ids[5:], prompt=True)
    expected = copy.deepcopy(state.cache)
    assert state.length == 11 and state.cache.layers[0].keys.shape[-2] <= 8
    state.restore(checkpoint)
    h.append(state, taps[5:], token_ids=ids[5:], prompt=True)
    assert torch.equal(state.cache.layers[0].keys, expected.layers[0].keys)
    before = state.snapshot()
    a = h.propose(state, 11, 3).tokens
    b = h.propose(state, 11, 3).tokens
    assert a == b and len(a) == 3
    assert state.length == before.length
    assert torch.equal(state.last_taps, before.last_taps)
    assert torch.equal(state.cache.layers[0].keys, before.cache.layers[0].keys)
    other = EagleState(head())
    with pytest.raises(ValueError, match='another head'):
        other.restore(before)


def test_shifted_cache_matches_separate_primary_teacher_forced_pairs():
    h = head()
    taps = torch.randn(7, 48, dtype=torch.bfloat16)
    ids = list(range(7))
    state = EagleState(h)
    h.append(state, taps, token_ids=ids, prompt=True)
    expected_cache = DynamicCache(config=h.model.config)
    with torch.inference_mode():
        h.model.step(torch.tensor([ids[1:]]), h.model.project(taps[:-1].unsqueeze(0)),
                     expected_cache, torch.arange(1, 7).unsqueeze(0))
    assert torch.equal(state.cache.layers[0].keys, expected_cache.layers[0].keys)
    assert torch.equal(state.cache.layers[0].values, expected_cache.layers[0].values)


def test_delta_q_packet_and_greedy_contract():
    chains = EagleChains([3, 6, 9], None, True)
    packet = chains.sample(0, 3, SimpleNamespace(temperature=1.), 12)
    assert packet.tokens == (3, 6, 9)
    assert all(row.probability(t) == 1 for t, row in zip(packet.tokens, packet.q_rows))
    assert all(row.probability(1) == 0 for row in packet.q_rows)
    with pytest.raises(ValueError):
        chains.finish(0, 3)


def test_actual_metadata_is_cpu_only_and_taps_embedding_inclusive():
    import json
    path = Path(__file__).parents[1] / 'tests' / 'fixtures' / 'gemma_eagle3_config.json'
    cfg = Eagle3Config.from_dict(json.loads(path.read_text()))
    assert cfg.tap_layers == (1, 20, 38)
    assert cfg.parameter_bytes == 1745122304
    for change in [{'draft_vocab_size': 32001}, {'norm_before_fc': False}, {'dtype': 'float16'}]:
        raw = json.loads(path.read_text())
        raw.update(change)
        with pytest.raises(ValueError):
            Eagle3Config.from_dict(raw)
    assert cfg.state_bytes(131072) < 20 * 1024**2


def test_failed_append_poison_then_restore_and_encoded_context_boundary(monkeypatch):
    h = head()
    taps = torch.randn(3, 48, dtype=torch.bfloat16)
    state = h.make_state()
    h.append(state, taps[:1], token_ids=[1])
    checkpoint = state.snapshot()
    original = h.model.step
    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError('encoder cache write failure')
    monkeypatch.setattr(h.model, 'step', fail)
    with pytest.raises(OSError):
        h.append(state, taps[1:], token_ids=[2, 3])
    assert state.poisoned
    with pytest.raises(RuntimeError):
        h.propose(state, 2, 1)
    with pytest.raises(RuntimeError):
        state.snapshot()
    state.restore(checkpoint)
    assert not state.poisoned and state.length == 1
    monkeypatch.setattr(h.model, 'step', original)
    h.append(state, taps[1:], token_ids=[2, 3])
    h.context_limit = 3
    assert h.propose(state, 4, 1) is None
    state.reset()
    assert state.length == 0 and state.last_taps is None


def test_meta_loading_keeps_actual_checkpoint_state_names_without_initialization():
    source = model()
    with torch.device('meta'):
        loaded = Eagle3Model(source.config, draft_vocab=10)
    loaded.load_state_dict(source.state_dict(), strict=True, assign=True)
    # Public rotary owns its nonpersistent inverse frequencies; not in weights.
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
    loaded.rotary = LlamaRotaryEmbedding(source.config)
    assert all(parameter.device.type == 'cpu' for parameter in loaded.parameters())
    assert not any('rotary' in key for key in source.state_dict())


def test_restore_allocation_failure_keeps_original_state_and_poison():
    from tensorfold.families.gemma4.cuda.eagle3 import EagleSnapshot
    h = head()
    state = h.make_state()
    h.append(state, torch.randn(3, 48, dtype=torch.bfloat16), token_ids=[1, 2, 3])
    old_cache, old_taps, old_length = state.cache, state.last_taps, state.length
    state.poisoned = True
    class CannotClone:
        def clone(self):
            raise MemoryError('snapshot taps')
    snapshot = EagleSnapshot(h, 1, DynamicCache(config=h.model.config), CannotClone())
    with pytest.raises(MemoryError):
        state.restore(snapshot)
    assert state.cache is old_cache and state.last_taps is old_taps
    assert state.length == old_length and state.poisoned


def test_proposal_context_reserves_anchor_row_and_returns_admissible_prefix():
    h = head()
    state = h.make_state()
    h.append(state, torch.randn(3, 48, dtype=torch.bfloat16), token_ids=[1, 2, 3])
    h.context_limit = 5
    assert len(h.propose(state, 4, 3).tokens) == 1
    h.context_limit = 4
    assert h.propose(state, 4, 3) is None


def test_checkpoint_headers_reject_before_tensor_loading(monkeypatch):
    import json
    from tensorfold.cuda import capacity
    from tensorfold.families.gemma4 import eagle3_metadata as metadata
    raw = json.loads((Path(__file__).parent / 'fixtures/gemma_eagle3_config.json').read_text())
    cfg = Eagle3Config.from_dict(raw)
    found = {key: dict(shape=list(shape), dtype='I64' if key == 'd2t' else 'BOOL' if key == 't2d' else 'BF16')
             for key, shape in cfg.shapes().items()}
    monkeypatch.setattr(metadata.Eagle3Config, 'read', classmethod(lambda cls, directory: cfg))
    monkeypatch.setattr(capacity, 'headers', lambda directory: found)
    assert metadata.checked_headers('unused')[0] is cfg
    found['lm_head.weight']['dtype'] = 'F16'
    with pytest.raises(ValueError, match='dtype'):
        metadata.checked_headers('unused')
    del found['lm_head.weight']
    with pytest.raises(ValueError, match='inventory'):
        metadata.checked_headers('unused')


@pytest.mark.parametrize('failure', ['mapping', 'upload'])
def test_constructor_boundary_failure_precedes_upload_or_preserves_native_error(monkeypatch, failure):
    import safetensors.torch
    from tensorfold.families.gemma4.cuda import eagle3
    weights = model().state_dict()
    cfg = SimpleNamespace(hidden=16, vocab=40, width=32, heads=2, kv_heads=1, head_dim=8,
        max_position=131072, window=8, eps=1e-6, theta=10000., draft_vocab=10,
        block=4, parameter_bytes=1, state_bytes=lambda context: 1, memory_bytes=lambda: (1, 2))
    monkeypatch.setattr(eagle3, 'checked_headers', lambda directory: (cfg, None))
    monkeypatch.setattr(eagle3, 'checkpoint_path', lambda *args: 'not-loaded')
    monkeypatch.setattr(safetensors.torch, 'load_file', lambda *args, **kwargs: weights)
    target = SimpleNamespace(device=torch.device('cpu'), config=SimpleNamespace(hidden=16, vocab=40))
    calls = []
    native_failure = OSError('device upload')
    def upload(self, device):
        calls.append(device)
        raise native_failure
    monkeypatch.setattr(eagle3.Eagle3Model, 'to', upload)
    if failure == 'mapping':
        weights['d2t'] = weights['d2t'].clone()
        weights['d2t'][0] = 1
        with pytest.raises(ValueError, match='offset mapping'):
            Gemma4Eagle3Head('unused', target, 128)
        assert not calls
    else:
        with pytest.raises(OSError) as caught:
            Gemma4Eagle3Head('unused', target, 128)
        assert caught.value is native_failure and calls == [torch.device('cpu')]


def test_constructor_failure_drops_owned_mapping_and_module_aliases_without_stripping_frames(monkeypatch):
    import weakref
    import safetensors.torch
    from tensorfold.families.gemma4.cuda import eagle3
    source = model()
    cfg = SimpleNamespace(hidden=16, vocab=40, width=32, heads=2, kv_heads=1, head_dim=8,
        max_position=131072, window=8, eps=1e-6, theta=10000., draft_vocab=10,
        block=4, parameter_bytes=1, state_bytes=lambda context: 1, memory_bytes=lambda: (1, 2))
    class OwnedWeights(dict):
        pass
    references = []
    def load(*args, **kwargs):
        weights = OwnedWeights(source.state_dict())
        references.append(weakref.ref(weights))
        return weights
    def upload(self, device):
        raise OSError('partial native upload')
    monkeypatch.setattr(eagle3, 'checked_headers', lambda directory: (cfg, None))
    monkeypatch.setattr(eagle3, 'checkpoint_path', lambda *args: 'unused')
    monkeypatch.setattr(safetensors.torch, 'load_file', load)
    monkeypatch.setattr(eagle3.Eagle3Model, 'to', upload)
    target = SimpleNamespace(device=torch.device('cpu'), config=SimpleNamespace(hidden=16, vocab=40))
    with pytest.raises(OSError) as caught:
        Gemma4Eagle3Head('unused', target, 128)
    assert references[0]() is None
    traceback = caught.value.__traceback__
    owned, native = [], []
    while traceback is not None:
        frame = traceback.tb_frame
        if frame.f_code.co_name == '__init__' and frame.f_globals.get('__name__') == eagle3.__name__:
            owned.append(frame)
        if frame.f_code.co_name == 'upload':
            native.append(frame)
        traceback = traceback.tb_next
    assert len(owned) == 1 and len(native) == 1
    assert owned[0].f_locals['weights'] is None and owned[0].f_locals['model'] is None
    assert owned[0].f_locals['self'].model is None
    assert native[0].f_locals['self'] is not None  # native traceback deliberately preserved


@pytest.mark.parametrize('committed_pairs', [2047, 2048, 2049])
def test_filled_swa2048_against_full_prefix_absolute_mask_and_snapshot(committed_pairs):
    """Independent absolute-mask oracle, not an empty cache at a large RoPE index.

    EAGLE pairs start at position1. The reference retains ALL KV and masks them
    by actual absolute positions; the production cache physically evicts history.
    Only valid chronological KV are compared, avoiding an invented numerical
    tolerance between SDPA reductions with different masked tensor dimensions.
    """
    from transformers.masking_utils import _preprocess_mask_arguments
    actual = model()
    actual.config.sliding_window = 2048
    oracle = PrimaryFirstLayer(actual.config).eval().to(torch.bfloat16)
    oracle.load_state_dict(actual.layers[0].state_dict())
    sliding = DynamicCache(config=actual.config)
    full = DynamicCache()  # independent reference deliberately never evicts KV
    ids = torch.arange(committed_pairs + 1).remainder(40).unsqueeze(0)
    torch.manual_seed(192)
    taps = torch.randn(1, committed_pairs + 1, 48, dtype=torch.bfloat16)
    with torch.inference_mode():
        for begin in range(0, committed_pairs, 128):
            end = min(begin + 128, committed_pairs)
            positions = torch.arange(begin + 1, end + 1).unsqueeze(0)
            hidden = actual.project(taps[:, begin:end])
            inputs = torch.cat((actual.embed_tokens(ids[:, begin:end]), hidden), dim=-1)
            absolute_query = positions[0, :, None]
            absolute_keys = torch.arange(1, end + 1)[None, :]
            mask = ((absolute_keys <= absolute_query) & (absolute_keys > absolute_query - 2048))[None, None]
            oracle(inputs, mask, actual.rotary(inputs, positions), full)
            actual.step(ids[:, begin:end], hidden, sliding, positions)
        assert int(sliding.get_seq_length()) == committed_pairs
        count = min(committed_pairs, 2047)  # HF keeps window-1 between queries
        assert sliding.layers[0].keys.shape[-2] == count
        assert torch.equal(sliding.layers[0].keys, full.layers[0].keys[:, :, -count:])
        assert torch.equal(sliding.layers[0].values, full.layers[0].values[:, :, -count:])
        snapshot = copy.deepcopy(sliding)
        position = committed_pairs + 1
        positions = torch.tensor([[position]])
        hidden = actual.project(taps[:, committed_pairs:])
        inputs = torch.cat((actual.embed_tokens(ids[:, committed_pairs:]), hidden), dim=-1)
        arguments = _preprocess_mask_arguments(actual.config, inputs, None, sliding, positions, 0)
        physical_kv_offset = int(arguments[-1])
        mask = create_sliding_window_causal_mask(actual.config, inputs, None, sliding,
            position_ids=positions, allow_is_causal_skip=False)
        allowed = mask[0, 0, 0].nonzero().flatten() if mask.dtype == torch.bool else (mask[0, 0, 0] == 0).nonzero().flatten()
        absolute_allowed = allowed + physical_kv_offset + 1
        assert torch.equal(absolute_allowed, torch.arange(max(1, position - 2047), position + 1))
        first = actual.step(ids[:, committed_pairs:], hidden, sliding, positions)
        restored = copy.deepcopy(snapshot)
        replay = actual.step(ids[:, committed_pairs:], hidden, restored, positions)
        assert torch.equal(first, replay)
        # Extend full reference and compare retained K/V after the actual query.
        keys = torch.arange(1, position + 1)[None, :]
        absolute_mask = ((keys <= position) & (keys > position - 2048))[None, None]
        oracle(inputs, absolute_mask, actual.rotary(inputs, positions), full)
        assert torch.equal(sliding.layers[0].keys, full.layers[0].keys[:, :, -2047:])
        assert torch.equal(sliding.layers[0].values, full.layers[0].values[:, :, -2047:])
        assert torch.equal(snapshot.layers[0].keys, restored.layers[0].keys) is False
