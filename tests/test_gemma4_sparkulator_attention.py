"""Candidate dispatch/ownership contracts; actual SM121 numerical gates are separate."""
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("triton")
from transformers import Gemma4AssistantConfig, Gemma4AssistantForCausalLM  # noqa: E402
from tensorfold.families.gemma4.cuda import sparkulator_attention as candidate  # noqa: E402
from tensorfold.families.gemma4.cuda.assistant_attention import assistant_native_reference  # noqa: E402


def model():
    raw = json.loads((Path(__file__).parent / "fixtures/gemma4_sparkulator_26b_config.json").read_text())
    with torch.device("meta"):
        return Gemma4AssistantForCausalLM(Gemma4AssistantConfig.from_dict(raw)).eval()


def test_disabled_is_native_and_enabled_borrows_only_global_parameters():
    native = model()
    old = [layer.self_attn for layer in native.model.layers]
    assert candidate.install_spark_global_attention(native) is None
    assert all(layer.self_attn is prior for layer, prior in zip(native.model.layers, old))
    keys = tuple(native.state_dict())
    policy = candidate.install_spark_global_attention(native, enabled=True)
    assert policy.prompt and policy.prompt_sdpa_gqa
    assert all(native.model.layers[i].self_attn is old[i] for i in range(3))
    assert native.model.layers[3].self_attn.q_proj is old[3].q_proj
    assert tuple(native.state_dict()) == keys
    with pytest.raises(ValueError, match="native16-query"):
        candidate.install_spark_global_attention(native, enabled=True)


def test_provider_scope_restores_after_failure_and_reference_bypasses_it(monkeypatch):
    native = model()
    policy = candidate.install_spark_global_attention(native, enabled=True)
    wrapper = native.model.layers[3].self_attn
    events = []
    failure = RuntimeError("native helper failure")

    @contextmanager
    def scope(backends):
        assert backends == [candidate.SDPBackend.EFFICIENT_ATTENTION]
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    def forward(*args, **kwargs):
        events.append("native" if policy.reference_depth else "candidate")
        raise failure

    monkeypatch.setattr(candidate, "sdpa_kernel", scope)
    monkeypatch.setattr(candidate, "_validate_region", lambda *args: None)
    monkeypatch.setattr(candidate.E2BTextAttention, "forward", forward)
    for reference in (False, True):
        try:
            if reference:
                with assistant_native_reference(native, policy):
                    wrapper(None, None, None, {})
            else:
                wrapper(None, None, None, {})
        except RuntimeError as error:
            assert error is failure
    assert events == ["enter", "candidate", "exit", "native"]
    assert policy.reference_depth == 0


def region(p=65536):
    device = SimpleNamespace(type="cuda")
    hidden = SimpleNamespace(dtype=torch.bfloat16, shape=(1, 1, 1024), device=device)
    kv = SimpleNamespace(dtype=torch.bfloat16, shape=(1, 2, p, 512), device=device,
                         stride=lambda: (1024, 512, 1024, 1))
    return hidden, {"full_attention": (kv, kv)}


@pytest.mark.parametrize("p", [1, 65536, 131072])
def test_exact_region_and_memory_inventory_without_allocating_kv(monkeypatch, p):
    monkeypatch.setattr(torch, "is_grad_enabled", lambda: False)
    hidden, shared = region(p)
    candidate._validate_region(hidden, None, shared, False)
    assert candidate.candidate_workspace_bytes(p) == (88 << 20) + 65552
    for mask, training in ((object(), False), (None, True)):
        with pytest.raises(ValueError, match="outside"):
            candidate._validate_region(hidden, mask, shared, training)


def test_wrong_stride_gradient_cpu_and_context_fail_without_fallback(monkeypatch):
    hidden, shared = region()
    monkeypatch.setattr(torch, "is_grad_enabled", lambda: True)
    with pytest.raises(ValueError, match="outside"):
        candidate._validate_region(hidden, None, shared, False)
    monkeypatch.setattr(torch, "is_grad_enabled", lambda: False)
    shared["full_attention"][0].stride = lambda: (1, 1, 512, 1)
    with pytest.raises(ValueError, match="outside"):
        candidate._validate_region(hidden, None, shared, False)
    hidden, shared = region()
    hidden.device = torch.device("cpu")
    with pytest.raises(ValueError, match="outside"):
        candidate._validate_region(hidden, None, shared, False)
    for p in (0, 131073, True):
        with pytest.raises(ValueError, match="context"):
            candidate.candidate_workspace_bytes(p)


@pytest.mark.parametrize("enabled", [False, True])
def test_head_factory_selects_loader_before_upload_and_composes_native_reference(monkeypatch, enabled):
    from tensorfold.families.gemma4.cuda import sparkulator
    from tensorfold.families.gemma4.sparkulator_metadata import SparkulatorConfig

    cfg = SparkulatorConfig.from_dict(json.loads(
        (Path(__file__).parent / "fixtures/gemma4_sparkulator_26b_config.json").read_text()))
    native = model()
    policy = candidate.install_spark_global_attention(native, enabled=enabled)
    calls = []

    def load(*args, **kwargs):
        calls.append(kwargs)
        return native, {"synthetic_factory_control": True}

    monkeypatch.setattr(sparkulator, "checked_headers", lambda root: (cfg, {}))
    monkeypatch.setattr(sparkulator, "_load", load)
    target = SimpleNamespace(config=SimpleNamespace(hidden=2816, vocab=262144), device=torch.device("cuda:0"))
    head = sparkulator.Sparkulator("unused-control", target, 131072, global_gqa=enabled)
    assert calls == ([{"global_gqa": True}] if enabled else [{}])
    assert head.global_gqa is enabled and head.attention_policy is policy
    with head.native_reference() as reference:
        assert reference is native
        if enabled:
            assert policy.reference_depth == 1
    if enabled:
        assert policy.reference_depth == 0
    head.close()
    assert head.attention_policy is None and head.model is None


def test_enabled_context_refused_before_load_native_retains_full_architecture(monkeypatch):
    from tensorfold.families.gemma4.cuda import sparkulator
    from tensorfold.families.gemma4.sparkulator_metadata import SparkulatorConfig

    cfg = SparkulatorConfig.from_dict(json.loads(
        (Path(__file__).parent / "fixtures/gemma4_sparkulator_26b_config.json").read_text()))
    monkeypatch.setattr(sparkulator, "checked_headers", lambda root: (cfg, {}))
    calls = []
    def load(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(), {}
    monkeypatch.setattr(sparkulator, "_load", load)
    target = SimpleNamespace(config=SimpleNamespace(hidden=2816, vocab=262144), device=torch.device("cuda:0"))
    with pytest.raises(ValueError, match="context"):
        sparkulator.Sparkulator("unused-control", target, 262144, global_gqa=True)
    assert calls == []
    head = sparkulator.Sparkulator("unused-control", target, 262144)
    assert calls == [{}] and head.context_limit == 262144
    head.close()


@pytest.mark.parametrize("options", [{"sampled_q": 1}, {"sampled_q": True, "block": 17}])
def test_invalid_sampled_mode_fails_before_load(monkeypatch, options):
    from tensorfold.families.gemma4.cuda import sparkulator
    from tensorfold.families.gemma4.sparkulator_metadata import SparkulatorConfig

    cfg = SparkulatorConfig.from_dict(json.loads(
        (Path(__file__).parent / "fixtures/gemma4_sparkulator_26b_config.json").read_text()))
    monkeypatch.setattr(sparkulator, "checked_headers", lambda root: (cfg, {}))
    def forbidden(*args, **kwargs):
        pytest.fail("invalid sampled mode entered model loader")
    monkeypatch.setattr(sparkulator, "_load", forbidden)
    target = SimpleNamespace(config=SimpleNamespace(hidden=2816, vocab=262144), device=torch.device("cuda:0"))
    with pytest.raises(ValueError):
        sparkulator.Sparkulator("unused-control", target, 131072, **options)


def test_unqualified_torch_git_blocks_enabled_before_load_and_keeps_native(monkeypatch):
    from tensorfold.families.gemma4.cuda import sparkulator
    from tensorfold.families.gemma4.sparkulator_metadata import SparkulatorConfig

    cfg = SparkulatorConfig.from_dict(json.loads(
        (Path(__file__).parent / "fixtures/gemma4_sparkulator_26b_config.json").read_text()))
    monkeypatch.setattr(sparkulator, "checked_headers", lambda root: (cfg, {}))
    monkeypatch.setattr(torch.version, "git_version", "unqualified-source-control")
    calls = []
    def load(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(), {}
    monkeypatch.setattr(sparkulator, "_load", load)
    target = SimpleNamespace(config=SimpleNamespace(hidden=2816, vocab=262144), device=torch.device("cuda:0"))
    with pytest.raises(ValueError, match="qualified Torch git"):
        sparkulator.Sparkulator("unused-control", target, 131072, global_gqa=True)
    assert calls == []
    head = sparkulator.Sparkulator("unused-control", target, 262144)
    assert calls == [{}]
    head.close()
