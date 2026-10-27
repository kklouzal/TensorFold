"""The private Dense trial adds one staging selection to the packed contract."""
import json
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
PROFILE = prefix_policy.NATIVE_DENSE_PROFILE


def test_only_native_dense_is_added_after_native_warm_and_source_audits(source, monkeypatch):
    dense = source.policy.weights.GGUFDense
    previous = dense.implementation
    monkeypatch.setenv(source.module.PROFILE_ENV, PROFILE)

    def warm():
        assert dense.implementation == previous
        source.calls.append("packed-extension")

    monkeypatch.setattr(gguf_ops, "extension", warm)
    assert source.module.configure(source.text, tp=8, torch=family.torch) == PROFILE
    assert source.calls == ["audit", "extension", "packed-extension", "audit"]
    assert family.selectors(source.policy) == (
        True, False, False, False, False, False, "packed_exact", "native32", None)
    assert dense.implementation == "native32"
    weights = family.weights(source)
    source.module.prepare_weights(weights)
    assert weights.meta["sm70_family_trial"]["dense"] == "native32"
    assert weights.meta["sm70_family_trial"]["dense_math"] == "original-fixed-m32-aten-v1"
    assert weights.meta["sm70_profile"] == PROFILE
    assert not any(weights.meta.get(name) for name in (
        "capture_ready", "device_greedy_mtp_ready", "device_greedy_native_ready", "attention_qualification"))


@pytest.mark.parametrize("profile", ["reference", "optimized", "source-family-trial",
                                     "source-family-prefix-trial", "source-packed-trial"])
def test_prior_profiles_keep_reference_dense_and_reset_a_previous_trial(source, monkeypatch, profile):
    monkeypatch.setattr(gguf_ops, "extension", lambda: None)
    monkeypatch.setenv(source.module.PROFILE_ENV, PROFILE)
    source.module.configure(source.text, tp=8, torch=family.torch)
    assert source.policy.weights.GGUFDense.implementation == "native32"
    monkeypatch.setenv(source.module.PROFILE_ENV, profile)
    source.module.configure(source.text, tp=8, torch=family.torch)
    assert source.policy.weights.GGUFDense.implementation == "reference"


@pytest.mark.parametrize("failure", ["geometry", "runtime", "native", "post-warm-audit"])
def test_refusal_never_publishes_dense_or_other_partial_selectors(source, monkeypatch, failure):
    monkeypatch.setenv(source.module.PROFILE_ENV, PROFILE)
    before = family.selectors(source.policy), source.policy.weights.GGUFDense.implementation, source.module._SELECTED

    def warm():
        if failure == "native":
            raise RuntimeError("native load failed")
        if failure == "post-warm-audit":
            source.record["protected"]["source-pins.json"]["sha256"] = "d" * 64

    monkeypatch.setattr(gguf_ops, "extension", warm)
    if failure == "geometry":
        source.text["hidden_size"] += 1
    if failure == "runtime":
        monkeypatch.setattr(family.torch.version, "git_version", "wrong")
    with pytest.raises((ValueError, RuntimeError)):
        source.module.configure(source.text, tp=8, torch=family.torch)
    assert (family.selectors(source.policy), source.policy.weights.GGUFDense.implementation,
            source.module._SELECTED) == before


def test_cache_epoch_is_distinct_and_only_existing_greedy_int8_text_domain_is_borrowed():
    profiles = ("reference", "optimized", "source-family-trial", "source-family-prefix-trial", "source-packed-trial")
    assert all(prefix_policy.identity(PROFILE) != prefix_policy.identity(old) for old in profiles)
    assert all(prefix_policy.state_epoch(PROFILE) != prefix_policy.state_epoch(old) for old in profiles)
    weights = NS(meta={"execution_provider": "sm70", "sm70_profile": PROFILE})
    for dtype, temperature, vision, draft, expected in (
            ("int8", 0, None, True, True), ("fp16", 0, None, True, False),
            ("int8", .7, None, True, False), ("int8", 0, object(), True, False),
            ("int8", 0, None, False, False)):
        kwargs = dict(draft=draft, vision=vision, position=24, kv_pair=get_pair(dtype))
        sampling = Sampling(17, temperature)
        assert prefix_policy.reuse_allowed(weights, sampling, **kwargs) is expected
        assert prefix_policy.serial_reuse_allowed(weights, sampling, **kwargs) is expected


def test_worker_handshake_seals_actual_dense_selection_and_keeps_old_profile_packet(monkeypatch):
    from .test_flashnext_stochastic_prefix import method
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
    from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFDense

    torch = family.torch
    precision = NS(fp8=lambda: False, same_on_ranks=lambda a, b: None)
    agree = method(FlashNextEngine, "_same_settings", {"json": json, "prompt_precision": precision})
    proxy = NS(int64=torch.int64, tensor=lambda data, **kw: torch.tensor(data, dtype=kw["dtype"]),
               empty=lambda shape, **kw: torch.empty(shape, dtype=kw["dtype"]), equal=torch.equal)
    packets, stale = [], []

    def gather(value, destination):
        packets.append(value.clone())
        destination.copy_(value.repeat(8))
        if stale:
            destination.view(8, -1)[1, 9] = stale[0]

    engine = NS(execution_provider="sm70", sm70_profile="source-packed-trial", depth=2, confidence=.7,
        max_len=262144, streams=26, graphs_enabled=False, prefill_rows=256, tp=8,
        rope=NS(metadata=lambda: {"rope_type": "default"}), kv_pair=NS(handshake=0), comm=NS(all_gather=gather))
    monkeypatch.setattr(GGUFDense, "implementation", "reference")
    agree(engine, proxy, None)
    old_packet = packets[-1].clone()
    monkeypatch.setattr(GGUFDense, "implementation", "native32")
    agree(engine, proxy, None)
    assert torch.equal(old_packet, packets[-1])
    engine.sm70_profile = PROFILE
    agree(engine, proxy, None)
    native_digest = int(packets[-1][9])
    monkeypatch.setattr(GGUFDense, "implementation", "reference")
    agree(engine, proxy, None)
    assert int(packets[-1][9]) != native_digest
    stale.append(native_digest)
    with pytest.raises(RuntimeError, match="different settings"):
        agree(engine, proxy, None)
