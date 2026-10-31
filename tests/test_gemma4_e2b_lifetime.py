"""Owned late-construction failures and packed-audio boundaries."""
from types import SimpleNamespace
import threading
import gc
import weakref

import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.gemma4.cuda import e2b_engine  # noqa: E402 - optional runtime admitted above
from tensorfold.families.gemma4.cuda.e2b_target import E2BTarget  # noqa: E402 - optional runtime admitted above
from test_gemma4_e2b_metadata import e2b_config  # noqa: E402 - optional runtime admitted above


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_late_cuda_stream_failure_closes_target_and_preserves_primary(monkeypatch, tmp_path, cleanup_fails):
    import tensorfold.checkpoint_metadata as metadata
    import tensorfold.cuda.chat_template as template
    import tensorfold.token_metadata as tokens
    closed = []
    primary = RuntimeError("stream allocation failed")
    cleanup = RuntimeError("target retirement failed")

    class Target:
        device = "cuda:0"

        def __init__(self, _path):
            pass

        def make_state(self, _context):
            return object()

        def close(self):
            closed.append(True)
            if cleanup_fails:
                raise cleanup

    def fail_stream(**_):
        raise primary

    monkeypatch.setattr(metadata, "read_config", lambda _: e2b_config())
    monkeypatch.setattr(template, "prepare_chat_template", lambda *_, **__: object())
    monkeypatch.setattr(tokens, "load_tokenizer", lambda _: SimpleNamespace(token_to_id=lambda _: 2))
    monkeypatch.setattr(tokens, "resolve_eos", lambda *_, **__: [1])
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(torch.cuda, "Stream", fail_stream)
    monkeypatch.setattr(e2b_engine, "admit", lambda *args: {"context_window": 32})
    monkeypatch.setattr(e2b_engine, "E2BTarget", Target)
    with pytest.raises(RuntimeError) as caught:
        e2b_engine.E2BEngine(tmp_path)
    assert caught.value is primary
    assert closed == [True]
    if cleanup_fails:
        assert caught.value.__cause__ is cleanup


def test_close_drops_cache_without_allocating_and_is_idempotent():
    engine = object.__new__(e2b_engine.E2BEngine)
    closed, drained = [], []
    engine._owner = None
    engine._closed_lock = threading.Lock()
    engine._run_lock = threading.Lock()
    engine._prefix_lock = threading.Lock()
    engine._stream = SimpleNamespace(synchronize=lambda: drained.append(True))
    engine._state = SimpleNamespace(reset=lambda: pytest.fail("disposal allocated a new cache"))
    engine._draft, engine.head, engine._prefix = object(), object(), object()
    engine._disposed = False
    engine.target = SimpleNamespace(close=lambda: closed.append(True))
    engine.close()
    engine.close()
    assert closed == [True] and drained == [True, True]
    assert engine._state is None and engine._draft is None and engine.head is None


def test_packed_audio_lengths_bound_independent_clip_slices():
    inputs = {"input_features": torch.zeros(1, 12, 4), "input_features_mask": torch.ones(1, 12),
              "audio_feature_lengths": torch.tensor([5, 7])}
    assert E2BTarget._audio_lengths(inputs) == (5, 7)
    inputs["audio_feature_lengths"] = torch.tensor([5, 8])
    with pytest.raises(ValueError, match="lengths"):
        E2BTarget._audio_lengths(inputs)
    del inputs["audio_feature_lengths"]
    with pytest.raises(ValueError, match="packed"):
        E2BTarget._audio_lengths(inputs)


def test_media_admission_charges_retained_and_scatter_peak(monkeypatch):
    import tensorfold.cuda.capacity as capacity
    target = object.__new__(E2BTarget)
    target.config = SimpleNamespace(hidden=16)
    target.raw_config = {
        "vision_config": {"num_attention_heads": 1, "hidden_size": 1, "intermediate_size": 1},
        "audio_config": {"subsampling_conv_channels": [1], "hidden_size": 1},
    }
    inputs = {"input_features": torch.zeros(1, 1, 4),
              "input_features_mask": torch.ones(1, 1), "audio_feature_lengths": torch.tensor([1])}
    # One embedding owner fits easily; actual PLE/scatter/feature lifetime does not.
    monkeypatch.setattr(capacity, "available_bytes", lambda _: 10000)
    with pytest.raises(ValueError, match="workspace"):
        target._require_media_workspace([2] * 100, inputs, (1,))


def test_canonical_prefill_chunks_preserve_every_turn_boundary():
    turn = 99
    prompt = [turn] + [2] * 519 + [turn] + [3] * 599
    assert e2b_engine.prefill_boundaries(prompt, turn) == (512, 520, 1032, 1120)
    extended = prompt + [4] * 100
    assert e2b_engine.prefill_boundaries(extended, turn) == (512, 520, 1032, 1220)


def test_prefix_rejects_old_final_partial_and_keeps_identical_turn_geometry():
    from tensorfold.families.gemma4.cuda.engine import _Prefix
    engine = object.__new__(e2b_engine.E2BEngine)
    engine._turn = 99
    engine._prefix_lock = threading.Lock()
    old = [99] + [2] * 519 + [99] + [3] * 99
    stable = SimpleNamespace(position=520, logits=None)
    partial = SimpleNamespace(position=len(old), logits=object())
    engine._prefix = _Prefix(tuple(old), (), (stable, partial))
    assert engine._find_prefix(old, ()) is partial
    assert engine._find_prefix(old + [4] * 100, ()) is stable
    # When a new turn begins exactly at the old end, its prior chunk is unchanged.
    assert engine._find_prefix(old + [99, 4, 5], ()) is partial


@pytest.mark.parametrize("media_id", [101, 102, 103])
def test_warm_media_prefix_text_suffix_skips_encoders(media_id):
    engine = object.__new__(e2b_engine.E2BEngine)
    engine._turn = None
    engine._draft, engine.head = None, None
    engine.vision = object()
    engine._prefix_lock = threading.Lock()
    state = SimpleNamespace(position=512, restore=lambda _: None, rollback=lambda _: None)
    engine._state = state
    engine._find_prefix = lambda *_: SimpleNamespace(position=512, state=object(), logits=None)
    engine._turn_boundary = lambda *_: 0
    engine._check_open = lambda: None
    engine._checkpoint = lambda state, logits=None: SimpleNamespace(position=state.position, logits=logits)
    logits = torch.ones(1, 4)
    seen = []

    def forward(ids, state, **kwargs):
        assert kwargs["encoded"] is None
        seen.extend(ids)
        state.position += len(ids)
        return SimpleNamespace(logits=logits)

    engine.target = SimpleNamespace(raw_config={"image_token_id": 101, "video_token_id": 102, "audio_token_id": 103},
                                    forward=forward, prepare_media=lambda *_: pytest.fail("repeat encoder"))
    prompt = [media_id] + [2] * 511 + [3, 4]
    media = SimpleNamespace(media_key=("same-owned-features",), model_inputs={"attention_mask": torch.ones(1, 514)})
    _, actual, cached = engine._prefill(prompt, media, None, 0)
    assert cached == 512 and actual is logits and seen == [3, 4]


@pytest.mark.parametrize("padding", [0, 8192])
def test_token_row_merge_exactly_matches_native_masked_scatter(padding):
    target = object.__new__(E2BTarget)
    target._closed, target.device = False, torch.device("cpu")
    target.config = SimpleNamespace(hidden=3, vocab=64, max_position=16384)
    config = SimpleNamespace(image_token_id=10, video_token_id=11, audio_token_id=12,
                             text_config=SimpleNamespace(pad_token_id=0))
    image = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.bfloat16)
    video = torch.tensor([[7, 8, 9]], dtype=torch.bfloat16)
    audio = torch.tensor([[10, 11, 12], [13, 14, 15]], dtype=torch.bfloat16)
    calls = []

    def audio_features(features, _mask, **_):
        calls.append(tuple(features.shape))
        index = len(calls) - 1
        return SimpleNamespace(pooler_output=audio[index:index+1].unsqueeze(0),
                               attention_mask=torch.ones(1, 1, dtype=torch.bool))

    def embed(ids):
        return ids.to(torch.bfloat16).unsqueeze(-1).expand(-1, -1, 3).clone()
    target.model = SimpleNamespace(config=config, get_input_embeddings=lambda: embed,
                                   model=SimpleNamespace(get_audio_features=audio_features))
    target._require_media_workspace = lambda *_: None
    target._image_features = lambda pixels, positions: image if pixels.shape[0] == 2 else video
    ids = [1] * padding + [1, 10, 11, 12, 12, 10, 1]
    inputs = {"pixel_values": torch.zeros(2, 1, 1), "image_position_ids": torch.zeros(2, 1, 2),
              "pixel_values_videos": torch.zeros(1, 1, 1, 1), "video_position_ids": torch.zeros(1, 1, 1, 2),
              "input_features": torch.zeros(1, 4, 3), "input_features_mask": torch.ones(1, 4),
              "audio_feature_lengths": torch.tensor([2, 2])}
    actual = target.prepare_media(ids, inputs)
    tokens = torch.tensor([ids])
    padded = torch.where((tokens == 10) | (tokens == 11) | (tokens == 12), 0, tokens)
    original = embed(padded)
    expected = original
    for token, feature in ((10, image), (11, video), (12, audio)):
        expected = expected.masked_scatter((tokens == token).unsqueeze(-1).expand_as(expected), feature)
    assert torch.equal(actual.text_embeddings, original)
    assert torch.equal(actual.embeddings, expected)
    assert actual.embeddings.data_ptr() != actual.text_embeddings.data_ptr()
    assert calls == [(1, 2, 3), (1, 2, 3)]


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_target_late_hook_failure_retires_owned_model_and_restores_policy(monkeypatch, tmp_path, cleanup_fails):
    import transformers
    import tensorfold.checkpoint_metadata as metadata
    from tensorfold.families.gemma4.cuda import e2b_attention, e2b_norm, e2b_policy, e2b_runtime
    primary = RuntimeError("late hook registration failed")
    cleanup = RuntimeError("policy restoration failed")
    removed, restored, models = [], [], []

    class Layer:
        def __init__(self, index):
            self.index = index

        def register_forward_hook(self, _hook):
            if self.index == 12:
                raise primary
            return SimpleNamespace(remove=lambda: removed.append(True))

    class Model:
        def __init__(self):
            self.model = SimpleNamespace(language_model=SimpleNamespace(layers=[Layer(i) for i in range(35)]))
            self.lm_head = SimpleNamespace(weight=object())
            models.append(weakref.ref(self))

        def eval(self):
            return self

        def requires_grad_(self, _enabled):
            return self

        def to(self, _device):
            return self

        def parameters(self):
            return iter([SimpleNamespace(device=torch.device("cuda:0"))])

        def get_input_embeddings(self):
            return self.lm_head

    class Policy:
        def __init__(self, _torch):
            pass

        def close(self):
            restored.append(True)
            if cleanup_fails:
                raise cleanup

    monkeypatch.setattr(metadata, "read_config", lambda _: e2b_config())
    monkeypatch.setattr(e2b_runtime, "require_runtime", lambda *_: None)
    monkeypatch.setattr(e2b_attention, "install_attention", lambda *_: None)
    monkeypatch.setattr(e2b_norm, "install_row_norms", lambda *_: None)
    monkeypatch.setattr(e2b_policy, "E2BBlasPolicy", Policy)
    monkeypatch.setattr(transformers.Gemma4ForConditionalGeneration, "from_pretrained",
                        lambda *_, **__: (Model(), {}))
    with pytest.raises(RuntimeError) as caught:
        E2BTarget(tmp_path)
    assert caught.value is primary and removed == [True] and restored == [True]
    if cleanup_fails:
        assert caught.value.__cause__ is cleanup
    gc.collect()
    assert models[0]() is None  # retained primary traceback has no redundant owned model alias


def test_target_bootstrap_guard_precedes_weights_and_policy(monkeypatch, tmp_path):
    import transformers
    import tensorfold.checkpoint_metadata as metadata
    from tensorfold.families.gemma4.cuda import e2b_policy
    monkeypatch.delenv("TF_FOUNDATION_STATE", raising=False)
    monkeypatch.setattr(metadata, "read_config", lambda _: e2b_config())
    monkeypatch.setattr(e2b_policy, "E2BBlasPolicy", lambda *_: pytest.fail("policy acquired before guard"))
    monkeypatch.setattr(transformers.Gemma4ForConditionalGeneration, "from_pretrained",
                        lambda *_, **__: pytest.fail("checkpoint loaded before guard"))
    with pytest.raises(RuntimeError, match="managed CUDA bootstrap"):
        E2BTarget(tmp_path)


def test_target_uses_triton_distribution_git_pin_not_plain_module_version(monkeypatch, tmp_path):
    import importlib.metadata
    import triton
    import tensorfold.checkpoint_metadata as metadata
    from tensorfold.families.gemma4.cuda import e2b_runtime
    observed = []
    stop = RuntimeError("guard reached before allocation")
    monkeypatch.setattr(metadata, "read_config", lambda _: e2b_config())
    monkeypatch.setattr(triton, "__version__", "3.9.0")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: e2b_runtime.TRITON_VERSION if name == "triton" else "unexpected")

    def guard(_torch, _hf, distribution_version):
        observed.append(distribution_version)
        raise stop

    monkeypatch.setattr(e2b_runtime, "require_runtime", guard)
    with pytest.raises(RuntimeError) as caught:
        E2BTarget(tmp_path)
    assert caught.value is stop
    assert observed == ["3.9.0+gitfc8d31a5"]


def test_unsupported_attention_dispatch_rejected_before_constructor_allocation(tmp_path):
    with pytest.raises(TypeError, match="attention"):
        E2BTarget(tmp_path, attention="eager")
