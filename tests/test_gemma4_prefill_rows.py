"""Private chunk-size boundary, prefix reuse and admission contracts."""
import json
import threading
import time
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from tensorfold.families.gemma4.cuda import e2b_engine  # noqa: E402 - optional runtime admitted above
from tensorfold.families.gemma4.cuda.e2b_metadata import E2BConfig  # noqa: E402 - optional runtime admitted above
from test_gemma4_e2b_metadata import e2b_config  # noqa: E402 - optional runtime admitted above
from test_gemma4_e4b import e4b_config  # noqa: E402 - optional runtime admitted above


@pytest.mark.parametrize("rows", [True, None, 256, 4096, 1024.0])
def test_bad_chunk_size_rejected_before_config_or_gpu(tmp_path, monkeypatch, rows):
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: pytest.fail("invalid rows reached GPU"))
    with pytest.raises(ValueError, match="prefill rows"):
        e2b_engine.E2BEngine(tmp_path, prefill_rows=rows)
    with pytest.raises(ValueError, match="prefill rows"):
        e2b_engine.prefill_boundaries([1], None, rows)


@pytest.mark.parametrize("rows", [1024, 2048])
def test_e2b_nondefault_rows_rejected_before_gpu(tmp_path, monkeypatch, rows):
    (tmp_path/"config.json").write_text(json.dumps(e2b_config()))
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: pytest.fail("E2B candidate reached GPU"))
    with pytest.raises(ValueError, match="E4B"):
        e2b_engine.E2BEngine(tmp_path, prefill_rows=rows)


@pytest.mark.parametrize("rows", [512, 1024, 2048])
def test_e4b_constructor_forwards_only_nondefault_admission_option(tmp_path, monkeypatch, rows):
    import tensorfold.cuda.chat_template as template
    import tensorfold.token_metadata as tokens
    (tmp_path/"config.json").write_text(json.dumps(e4b_config()))
    seen = []
    stop = RuntimeError("stop before target allocation")
    monkeypatch.setattr(template, "prepare_chat_template", lambda *_, **__: object())
    monkeypatch.setattr(tokens, "load_tokenizer", lambda _: SimpleNamespace(token_to_id=lambda _: 99))
    monkeypatch.setattr(tokens, "resolve_eos", lambda *_, **__: [1])
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(e2b_engine, "admit", lambda *_, **kw: seen.append(kw) or {"context_window": 8192})

    def target(*_, **kw):
        raise stop

    monkeypatch.setattr(e2b_engine, "E2BTarget", target)
    with pytest.raises(RuntimeError) as caught:
        e2b_engine.E2BEngine(tmp_path, prefill_rows=rows)
    assert caught.value is stop
    assert seen == ([{}] if rows == 512 else [{"prefill_rows": rows}])


class State:
    def __init__(self):
        self.tokens = ()

    @property
    def position(self):
        return len(self.tokens)

    def reset(self):
        self.tokens = ()

    def snapshot(self):
        return self.tokens

    def restore(self, tokens):
        self.tokens = tokens

    def rollback(self, position):
        self.tokens = self.tokens[:position]


def actor(rows):
    engine = object.__new__(e2b_engine.E2BEngine)
    engine.prefill_rows, engine._turn = rows, 99
    engine._prefix, engine._prefix_lock = None, threading.Lock()
    engine._state, engine._draft, engine.head = State(), None, None
    engine._check_open = lambda: None
    engine.vision = None
    calls = []

    def forward(tokens, state, **options):
        calls.append((state.position, tuple(tokens), options["output_logits"]))
        state.tokens += tuple(tokens)
        # Independent toy state oracle is sensitive to ordering/history loss.
        logits = torch.tensor([sum((i+1)*t for i, t in enumerate(state.tokens))])
        return SimpleNamespace(logits=logits)

    engine.target = SimpleNamespace(forward=forward)
    return engine, calls


@pytest.mark.parametrize("rows", [512, 1024, 2048])
def test_same_configuration_cold_exact_warm_extension_and_new_turn_cache_law(rows):
    engine, calls = actor(rows)
    prompt = [99]+[2]*(rows+17)+[99]+[3]*(rows+31)
    state, logits, cached = engine._prefill(prompt, None, None, time.perf_counter())
    assert cached == 0 and state.tokens == tuple(prompt)
    assert [start+len(tokens) for start, tokens, _ in calls] == list(e2b_engine.prefill_boundaries(prompt, 99, rows))
    assert max(len(tokens) for _, tokens, _ in calls) == rows
    calls.clear()
    _, replay, cached = engine._prefill(prompt, None, None, time.perf_counter())
    assert cached == len(prompt) and not calls and torch.equal(logits, replay)
    extended = prompt+[4]*19
    _, warm, cached = engine._prefill(extended, None, None, time.perf_counter())
    assert cached == rows+18  # last turn, not old final partial chunk
    cold, _ = actor(rows)
    _, reference, _ = cold._prefill(extended, None, None, time.perf_counter())
    assert engine._state.tokens == cold._state.tokens == tuple(extended)
    assert torch.equal(warm, reference)
    calls.clear()
    engine._prefill(extended+[99, 5], None, None, time.perf_counter())
    assert calls[0][0] == len(extended)  # previous final is now a stable turn boundary


@pytest.mark.parametrize("rows", [512, 1024, 2048])
def test_cancellation_publishes_no_partial_prefix_and_next_request_recovers(rows):
    from tensorfold.server.cancellation import RequestCancelled
    engine, calls = actor(rows)
    prompt = [2]*(rows*2+3)
    with pytest.raises(RequestCancelled):
        engine._prefill(prompt, None, lambda *_: True, time.perf_counter())
    assert len(calls) == 1 and engine._state.position == rows and engine._prefix is None
    engine._prefill(prompt, None, None, time.perf_counter())
    assert engine._state.tokens == tuple(prompt)


@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("media", [False, True])
def test_chunk_size_charges_recording_attention_and_activation_geometry(monkeypatch, tmp_path, static, media):
    import tensorfold.cuda.capacity as capacity
    cfg = E2BConfig.from_dict(e4b_config())
    captured = []
    monkeypatch.setattr(capacity, "headers", lambda _: {"fake": {"shape": [1], "dtype": "BF16"}})

    def admission(*args, **kwargs):
        captured.append(args[4])
        return {}

    monkeypatch.setattr(capacity, "admit", admission)
    for rows in (512, 1024, 2048):
        e2b_engine.admit(tmp_path, cfg, 8192, True, torch, None, 1, media,
                         full_static_cache=static, prefill_rows=rows)
    for rows, geometry in zip((1024, 2048), captured[1:]):
        slots = 8192
        cache_delta = cfg.cache_bytes(slots, rows)-cfg.cache_bytes(slots, 512)
        attention_delta = (rows*(slots+rows)-512*(slots+512))*cfg.heads*9
        activation_delta = (rows-512)*(cfg.hidden*40+cfg.intermediate*8)*4
        assert geometry.needed(slots)-captured[0].needed(slots) == (
            cache_delta*(1 if static else 4)+attention_delta+activation_delta)


@pytest.mark.parametrize("rows", [512, 1024, 2048])
@torch.inference_mode()
def test_public_hf_cold_and_cached_extension_logits_and_donor_kv_are_exact(rows):
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig
    from tensorfold.families.gemma4.cuda.e2b_target import E2BState
    torch.manual_seed(31)
    cfg = Gemma4TextConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=4,
                          num_attention_heads=8, num_key_value_heads=2, head_dim=4, global_head_dim=8,
                          num_kv_shared_layers=2, sliding_window=512, hidden_size_per_layer_input=4,
                          vocab_size_per_layer_input=128, vocab_size=128, max_position_embeddings=8192,
                          enable_moe_block=False, use_double_wide_mlp=False, attention_k_eq_v=False,
                          layer_types=["sliding_attention", "full_attention"]*2)
    cfg._attn_implementation = "sdpa"
    model = Gemma4ForCausalLM(cfg).to(torch.bfloat16).eval()

    def make():
        engine, _ = actor(rows)
        engine._state = E2BState(cfg, 8192)

        def forward(ids, state, **options):
            end = state.position+len(ids)
            output = model(input_ids=torch.tensor([ids]), past_key_values=state.cache,
                           position_ids=torch.arange(state.position, end)[None], use_cache=True)
            state.position = end
            return SimpleNamespace(logits=output.logits[:, -1])

        engine.target = SimpleNamespace(forward=forward)
        return engine

    warm, cold = make(), make()
    prompt = [99]+[2]*(rows+7)+[99]+[3]*13
    warm._prefill(prompt, None, None, time.perf_counter())
    extended = prompt+[4]*11
    _, actual, cached = warm._prefill(extended, None, None, time.perf_counter())
    _, expected, _ = cold._prefill(extended, None, None, time.perf_counter())
    assert cached == rows+8 and torch.equal(actual, expected)
    for a, b in zip(warm._state.cache.layers, cold._state.cache.layers):
        assert torch.equal(a.keys, b.keys) and torch.equal(a.values, b.values)
    _, replay, cached = warm._prefill(extended, None, None, time.perf_counter())
    assert cached == len(extended) and torch.equal(replay, expected)
