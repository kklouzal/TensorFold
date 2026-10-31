"""Head-specific feature dispatch without changing default DF2 append contracts."""
from types import SimpleNamespace
import pytest

torch = pytest.importorskip("torch")
from tensorfold.families.gemma4.cuda.e2b_engine import E2BEngine  # noqa: E402 - optional runtime admitted above


@pytest.mark.parametrize("kind", ["dflash2", "assistant", "eagle3"])
def test_head_append_selects_only_its_actual_features_and_committed_input_ids(kind):
    engine = object.__new__(E2BEngine)
    engine._tap_features = kind != "assistant"
    engine._head_token_ids = kind == "eagle3"
    layers = (1, 20, 38) if kind == "eagle3" else (2, 12, 22, 32)
    result = SimpleNamespace(hidden=torch.arange(12).reshape(3, 4),
                             taps={i: torch.full((3, 2), i) for i in layers})
    calls = []
    engine.head = SimpleNamespace(tap_layers=layers, append=lambda *args, **kwargs: calls.append((args, kwargs)))
    companion = object()
    engine._append_head(companion, result, [7, 8, 9], 2)
    args, kwargs = calls[0]
    assert args[0] is companion
    expected = result.hidden[:2] if kind == "assistant" else torch.cat([result.taps[i][:2] for i in layers],dim=-1)
    assert torch.equal(args[1], expected)
    assert kwargs == ({"token_ids": [7, 8], "prompt": False} if kind == "eagle3" else {"prompt": False})
    calls.clear()
    engine._append_head(companion, result, [7, 8, 9], prompt=True)
    assert calls[0][0][1].shape[0] == 3
    assert calls[0][1] == ({"token_ids": [7, 8, 9], "prompt": True} if kind == "eagle3" else {"prompt": True})


@pytest.mark.parametrize("profile,layers", [("E2B", (1,20,38)), ("E4B", (2,12,22,32)),
                                          ("E4B", [1,20,38]), ("E4B", (1,20,39))])
def test_target_tap_override_is_exact_and_cpu_validated_before_runtime(tmp_path, monkeypatch, profile, layers):
    import tensorfold.checkpoint_metadata as metadata
    from tensorfold.families.gemma4.cuda.e2b_target import E2BTarget
    from test_gemma4_e2b_metadata import e2b_config
    from test_gemma4_e4b import e4b_config
    monkeypatch.setattr(metadata, "read_config", lambda _: e2b_config() if profile == "E2B" else e4b_config())
    with pytest.raises(ValueError, match="EAGLE3 contract"):
        E2BTarget(tmp_path, tap_layers=layers)
