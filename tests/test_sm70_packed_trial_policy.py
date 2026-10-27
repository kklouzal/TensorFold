"""Packed trial selects only the numerically screened operators before admission."""
from types import SimpleNamespace as NS

import pytest

pytest.importorskip("torch")

from . import test_sm70_family_trial_policy as family  # noqa: E402
from tensorfold.cuda.sm70 import gguf_ops  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import prefix_policy  # noqa: E402
from tensorfold.families.qwen4_exp.kv_formats import get_pair  # noqa: E402

policy = family.policy
source = family.source


def test_selected_modules_warm_before_policy_publication_and_no_other_grants(source, monkeypatch):
    monkeypatch.setenv(source.module.PROFILE_ENV, source.module.PACKED_TRIAL)
    monkeypatch.setattr(gguf_ops, "extension", lambda: source.calls.append("packed-extension"))
    assert source.module.configure(source.text, tp=8, torch=family.torch) == source.module.PACKED_TRIAL
    assert source.calls == ["audit", "extension", "packed-extension", "audit"]
    assert family.selectors(source.policy) == (
        True, False, False, False, False, False, "packed_exact", "native32", None)
    weights = family.weights(source)
    source.module.prepare_weights(weights)
    assert weights.meta["sm70_profile"] == source.module.PACKED_TRIAL
    assert weights.meta["sm70_family_trial"]["experts"] == "packed_exact"
    assert weights.meta["sm70_family_trial"]["concat"] == "native32"
    assert not any(weights.meta.get(name) for name in (
        "capture_ready", "device_greedy_mtp_ready", "device_greedy_native_ready", "attention_qualification"))


def test_packed_load_failure_preserves_the_previous_policy(source, monkeypatch):
    before = family.selectors(source.policy)
    selected = source.module._SELECTED
    monkeypatch.setenv(source.module.PROFILE_ENV, source.module.PACKED_TRIAL)
    error = RuntimeError("native load failed")

    def failed():
        raise error

    monkeypatch.setattr(gguf_ops, "extension", failed)
    with pytest.raises(RuntimeError) as caught:
        source.module.configure(source.text, tp=8, torch=family.torch)
    assert caught.value is error
    assert family.selectors(source.policy) == before
    assert source.module._SELECTED == selected


def test_packed_profile_epoch_is_distinct_from_every_prior_profile():
    profiles = ("reference", "optimized", "source-family-trial", "source-family-prefix-trial")
    actual = prefix_policy.identity("source-packed-trial")
    epoch = prefix_policy.state_epoch("source-packed-trial")
    assert actual and epoch
    assert all(actual != prefix_policy.identity(profile) for profile in profiles)
    assert all(epoch != prefix_policy.state_epoch(profile) for profile in profiles)


@pytest.mark.parametrize("pair,greedy,vision,allowed", [
    (get_pair("int8"), True, None, True),
    (get_pair("int8"), False, None, False),
    (get_pair("fp16"), True, None, False),
    (get_pair("int8", "int8", "fp16"), True, None, False),
    (get_pair("rotorquant-iso4-sm70-v1"), True, None, False),
    (get_pair("int8"), True, object(), False),
])
def test_only_same_existing_greedy_int8_text_domain_can_borrow(pair, greedy, vision, allowed):
    weights = NS(meta={"sm70_profile": "source-packed-trial", "execution_provider": "sm70"})
    sampling = Sampling(17, 0 if greedy else .7)
    assert prefix_policy.reuse_allowed(weights, sampling, draft=True, vision=vision,
                                       position=24, kv_pair=pair) is allowed
    assert prefix_policy.serial_reuse_allowed(weights, sampling, draft=True, vision=vision,
                                              position=24, kv_pair=pair) is allowed
