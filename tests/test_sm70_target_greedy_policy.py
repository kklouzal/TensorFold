"""Source target-greedy trial is prepared explicitly and grants no old readiness."""

import pytest

pytest.importorskip("torch")

from . import test_sm70_family_trial_policy as family  # noqa: E402
from tensorfold.cuda.sm70 import gguf_ops, greedy_native  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import prefix_policy  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.target_greedy import POLICY  # noqa: E402

policy = family.policy
source = family.source


def test_target_modules_precede_selection_and_only_unqualified_metadata_is_published(source, monkeypatch):
    monkeypatch.setenv(source.module.PROFILE_ENV, source.module.TARGET_GREEDY_TRIAL)
    monkeypatch.setattr(gguf_ops, "extension", lambda: source.calls.append("packed"))
    monkeypatch.setattr(greedy_native, "extension", lambda: source.calls.append("target"))
    assert source.module.configure(source.text, tp=8, torch=family.torch) == source.module.TARGET_GREEDY_TRIAL
    assert source.calls == ["audit", "extension", "packed", "target", "audit"]
    assert family.selectors(source.policy) == (True, False, False, False, False, False, "packed_exact", "native32", None)
    weights = family.weights(source)
    source.module.prepare_weights(weights)
    assert weights.meta["sm70_target_greedy_trial"] == {
        "policy": POLICY, "qualification": "unqualified-candidate", "mtp_confidence": "original-fp32-lse-numpy-fp64"}
    assert weights.meta["target_greedy_owner"] is None
    assert not {"capture_ready", "device_greedy_mtp_ready", "device_greedy_native_ready", "attention_qualification"} & weights.meta.keys()


def test_target_native_load_failure_does_not_publish_or_modify_prior_dispatch(source, monkeypatch):
    selected, before = source.module._SELECTED, family.selectors(source.policy)
    monkeypatch.setenv(source.module.PROFILE_ENV, source.module.TARGET_GREEDY_TRIAL)
    monkeypatch.setattr(gguf_ops, "extension", lambda: source.calls.append("packed"))
    primary = RuntimeError("target component build refused")
    def fail():
        raise primary
    monkeypatch.setattr(greedy_native, "extension", fail)
    with pytest.raises(RuntimeError) as caught:
        source.module.configure(source.text, tp=8, torch=family.torch)
    assert caught.value is primary
    assert source.module._SELECTED == selected and family.selectors(source.policy) == before


def test_target_profile_and_prefix_epoch_do_not_reuse_prior_authority():
    old = ("reference", "optimized", "source-family-trial", "source-family-prefix-trial", "source-packed-trial")
    actual = prefix_policy.TARGET_GREEDY_PROFILE
    assert all(prefix_policy.identity(actual) != prefix_policy.identity(profile) for profile in old)
    assert all(prefix_policy.state_epoch(actual) != prefix_policy.state_epoch(profile) for profile in old)
