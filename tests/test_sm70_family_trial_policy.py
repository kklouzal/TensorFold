"""Qualification-only FAMILY selection never grants legacy operator evidence."""

from copy import deepcopy
from types import SimpleNamespace as NS

import pytest
import torch

from test_sm70_execution_policy import policy  # noqa: F401 - shared guarded-policy fixture
from tensorfold.cuda.sm70 import execution_policy, family_native, software_binding
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.qwen4_exp.cuda import prefix_policy
from tensorfold.families.qwen4_exp.kv_formats import get_pair


@pytest.fixture
def source(policy, monkeypatch):
    module = policy.module
    monkeypatch.setattr(torch, "__version__", module._FAMILY_RUNTIME["torch"])
    monkeypatch.setattr(torch.version, "git_version", module._FAMILY_RUNTIME["torch_git"])
    monkeypatch.setattr(torch.version, "cuda", module._FAMILY_RUNTIME["cuda"])
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_70"])
    monkeypatch.setattr(torch.backends.cuda, "preferred_blas_library", lambda: "_BlasBackend.Cublas")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction", True)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_fp16_accumulation", False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    record = {"recorded_components": dict(module._FAMILY_COMPONENTS),
              "protected": {"source-pins.json": {"sha256": "a" * 64}},
              "source_specification_sha256": "b" * 64, "software_only": True,
              "release_qualified": False, "runtime_sha256": "c" * 64}
    calls = []

    def issue(*, required):
        assert required is True
        calls.append("audit")
        return NS(record=lambda: deepcopy(record))

    monkeypatch.setattr(software_binding, "issue_source_binding", issue)
    monkeypatch.setattr(family_native, "extension", lambda: calls.append("extension"))
    monkeypatch.setenv(module.PROFILE_ENV, module.FAMILY_TRIAL)
    text = {**module._FAMILY_GEOMETRY, "dtype": "float16", "mamba_ssm_dtype": "float32",
            "output_gate_type": "sigmoid"}
    return NS(policy=policy, module=module, text=text, record=record, calls=calls)


def selectors(policy):
    return (policy.family._NATIVE_FAMILY, policy.family._NATIVE_ATTENTION,
            policy.routing._NATIVE_LOGITS, policy.padded._ENABLED,
            policy.proposal._ENABLED, policy.context._ENABLED,
            policy.weights.GGUFExperts.implementation, policy.weights.GGUFConcat.implementation,
            policy.prefill.WORKSPACE_BUDGET_BYTES)


def weights(source):
    return NS(cfg=NS(**source.module._FAMILY_LOCAL, execution_dtype="float16", quant="gguf"),
              meta={"execution_provider": "sm70", "execution_dtype": "float16", "world": 8})


def test_trial_resets_all_other_optimized_operators_and_prepares_before_selection(source, monkeypatch):
    policy = source.policy
    monkeypatch.setenv(source.module.PROFILE_ENV, "optimized")
    source.module.configure(source.text, tp=8, torch=torch)
    monkeypatch.setenv(source.module.PROFILE_ENV, source.module.FAMILY_TRIAL)
    assert source.module.configure(source.text, tp=8, torch=torch) == source.module.FAMILY_TRIAL
    assert selectors(policy) == (True, False, False, False, False, False, "reference", "reference", None)
    assert source.calls == ["audit", "extension", "audit"]
    w = weights(source)
    source.module.prepare_weights(w)
    assert source.calls == ["audit", "extension", "audit", "audit"]
    assert w.meta["sm70_profile"] == source.module.FAMILY_TRIAL
    assert w.meta["sm70_family_trial"]["qualification"] == "unqualified-candidate"
    assert w.meta["sm70_family_trial"]["software"]["release_qualified"] is False
    assert not {"capture_ready", "device_greedy_mtp_ready", "device_greedy_native_ready",
                "attention_qualification"} & w.meta.keys()


def test_unset_profile_remains_reference_on_selected_source_stack(source, monkeypatch):
    monkeypatch.delenv(source.module.PROFILE_ENV)
    assert source.module.configure({}, tp=8, torch=torch) == "reference"
    assert selectors(source.policy) == (False, False, False, False, False, False, "reference", "reference", None)
    assert source.calls == []


@pytest.mark.parametrize("key", tuple(execution_policy._FAMILY_GEOMETRY))
def test_every_trial_geometry_field_refuses_wrong_values_before_native_preparation(source, key):
    source.text[key] += 1
    before = selectors(source.policy)
    with pytest.raises(ValueError, match="geometry"):
        source.module.configure(source.text, tp=8, torch=torch)
    assert selectors(source.policy) == before and source.calls == []


@pytest.mark.parametrize("change", ["tp", "dtype", "state", "gate", "typed-geometry", "missing-geometry"])
def test_trial_semantic_and_tp_envelope_refuses_before_native_preparation(source, change):
    tp = 4 if change == "tp" else 8
    if change == "dtype":
        source.text["dtype"] = "bfloat16"
    elif change == "state":
        source.text["mamba_ssm_dtype"] = "float16"
    elif change == "gate":
        source.text["output_gate_type"] = "silu"
    elif change == "typed-geometry":
        source.text["head_dim"] = "256"
    elif change == "missing-geometry":
        del source.text["head_dim"]
    before = selectors(source.policy)
    with pytest.raises(ValueError, match="geometry"):
        source.module.configure(source.text, tp=tp, torch=torch)
    assert selectors(source.policy) == before and source.calls == []


@pytest.mark.parametrize("change", ["torch", "git", "cuda", "arch", "capability", "blas", "reduced", "accumulation", "workspace"])
def test_trial_refuses_unqualified_runtime_without_widening_legacy_attention(source, monkeypatch, change):
    if change == "torch":
        monkeypatch.setattr(torch, "__version__", "2.14.1+cu126")
    elif change in ("git", "cuda"):
        monkeypatch.setattr(torch.version, "git_version" if change == "git" else "cuda", "wrong")
    elif change == "arch":
        monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_80"])
    elif change == "capability":
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (8, 0))
    elif change == "blas":
        monkeypatch.setattr(torch.backends.cuda, "preferred_blas_library", lambda: "_BlasBackend.Cublaslt")
    elif change in ("reduced", "accumulation"):
        monkeypatch.setattr(torch.backends.cuda.matmul,
                            "allow_fp16_reduced_precision_reduction" if change == "reduced" else "allow_fp16_accumulation",
                            change == "accumulation")
    else:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    before = selectors(source.policy)
    with pytest.raises(ValueError, match="source04cf"):
        source.module.configure(source.text, tp=8, torch=torch)
    assert selectors(source.policy) == before and source.calls == []


@pytest.mark.parametrize("change", ["absent", "components", "extension", "protected"])
def test_source_or_native_preparation_failure_precedes_trial_selection(source, monkeypatch, change):
    if change == "absent":
        def absent(**kwargs):
            raise software_binding.SoftwareBindingError("required source software artifacts are absent")
        monkeypatch.setattr(software_binding, "issue_source_binding", absent)
    elif change == "components":
        source.record["recorded_components"]["cuda"] = "12.6"
    elif change == "extension":
        def failed():
            raise RuntimeError("native family build failed")
        monkeypatch.setattr(family_native, "extension", failed)
    else:
        def changed():
            source.record["protected"]["source-pins.json"]["sha256"] = "d" * 64
        monkeypatch.setattr(family_native, "extension", changed)
    before = selectors(source.policy)
    with pytest.raises((ValueError, RuntimeError)):
        source.module.configure(source.text, tp=8, torch=torch)
    assert selectors(source.policy) == before


@pytest.mark.parametrize("key", ["capture_ready", "device_greedy_mtp_ready", "device_greedy_native_ready", "attention_qualification"])
def test_trial_cannot_inherit_unqualified_readiness(source, key):
    source.module.configure(source.text, tp=8, torch=torch)
    w = weights(source)
    w.meta[key] = True
    before = deepcopy(w.meta)
    with pytest.raises(ValueError, match="cannot inherit"):
        source.module.prepare_weights(w)
    assert w.meta == before


@pytest.mark.parametrize("change", ["dtype", "quant", "shape", "provider", "world", "source"])
def test_loaded_weight_or_changed_source_failure_precedes_trial_publication(source, change):
    source.module.configure(source.text, tp=8, torch=torch)
    w = weights(source)
    if change == "dtype":
        w.cfg.execution_dtype = "bfloat16"
    elif change == "quant":
        w.cfg.quant = "nvfp4"
    elif change == "shape":
        w.cfg.heads = 6
    elif change == "provider":
        w.meta["execution_provider"] = "cuda"
    elif change == "world":
        w.meta["world"] = 4
    else:
        source.record["protected"]["source-pins.json"]["sha256"] = "d" * 64
    before = deepcopy(w.meta)
    with pytest.raises(ValueError):
        source.module.prepare_weights(w)
    assert w.meta == before


def test_source_stack_does_not_become_legacy_optimized(source, monkeypatch):
    monkeypatch.setenv(source.module.PROFILE_ENV, "optimized")
    from tensorfold.cuda.sm70 import attention_qualification
    monkeypatch.setattr(attention_qualification, "runtime_identity", lambda device: dict(source.module._FAMILY_RUNTIME))
    with pytest.raises(ValueError, match="qualified"):
        source.module.configure(source.text, tp=8, torch=torch)


@pytest.mark.parametrize("format", ["fp16", "int8", "rotorquant-signed128-4", "rotorquant-signed64-norm4"])
@pytest.mark.parametrize("sampling", [None, Sampling(17, 0), Sampling(19, .7, 40, .9), Sampling(19, float("nan"))])
def test_trial_has_distinct_worker_identity_and_no_serial_or_drafted_prefix_borrow(format, sampling):
    w = NS(meta={"execution_provider": "sm70", "sm70_profile": "source-family-trial"})
    pair = get_pair(format)
    assert prefix_policy.identity("source-family-trial") not in (None, prefix_policy.identity("reference"))
    assert not prefix_policy.reuse_allowed(w, sampling, draft=True, vision=None, position=128, kv_pair=pair)
    assert not prefix_policy.serial_reuse_allowed(w, sampling, draft=True, position=128, kv_pair=pair)


def test_trial_prefix_policy_does_not_change_other_provider_or_nonreuse_domains():
    w = NS(meta={"execution_provider": "cuda", "sm70_profile": "source-family-trial"})
    assert prefix_policy.reuse_allowed(w, None, draft=True, vision=None, position=128)
    w.meta["execution_provider"] = "sm70"
    assert not prefix_policy.reuse_allowed(w, None, draft=False, vision=None, position=128)
    assert not prefix_policy.reuse_allowed(w, None, draft=True, vision=object(), position=128)
