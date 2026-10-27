"""Explicit WMMA math/profile selection preserves exact defaults and cache domains."""

from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")

from . import test_sm70_family_trial_policy as family  # noqa: E402
from . import test_sm70_rotorquant_state as state_controls  # noqa: E402
from . import test_cuda_geometry as geometry_controls  # noqa: E402
from tensorfold.cuda.sm70 import gguf_ops  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import prefix_policy  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.state import State  # noqa: E402
from tensorfold.families.qwen4_exp.kv_formats import get_pair  # noqa: E402

policy = family.policy
source = family.source
allocations = geometry_controls.allocations
PROFILE = "source-wmma80-trial"


def test_wmma_profile_warms_admitted_module_and_grants_only_declared_math(source, monkeypatch):
    monkeypatch.setenv(source.module.PROFILE_ENV, PROFILE)
    monkeypatch.setattr(gguf_ops, "extension", lambda: source.calls.append("gguf-extension"))
    assert source.module.configure(source.text, tp=8, torch=torch) == PROFILE
    assert source.calls == ["audit", "extension", "gguf-extension", "audit"]
    assert family.selectors(source.policy) == (
        True, False, False, False, False, False, "packed_wmma80", "native32", None)
    loaded = family.weights(source)
    source.module.prepare_weights(loaded)
    assert loaded.meta["sm70_profile"] == PROFILE
    declaration = loaded.meta["sm70_family_trial"]
    assert declaration["qualification"] == "unqualified-candidate"
    assert declaration["numerical_policy"] == "approximate-wmma-fp16-fp32-split8-v1"
    assert declaration["experts"] == "packed_wmma80" and declaration["concat"] == "native32"
    assert not {"attention_qualification", "capture_ready", "device_greedy_mtp_ready",
                "device_greedy_native_ready"} & loaded.meta.keys()


@pytest.mark.parametrize("change", ["geometry", "tp", "runtime", "load"])
def test_wmma_failure_preserves_existing_policy_before_publication(source, monkeypatch, change):
    monkeypatch.setenv(source.module.PROFILE_ENV, PROFILE)
    before, selected = family.selectors(source.policy), source.module._SELECTED
    tp = 8
    if change == "geometry":
        source.text["head_dim"] = 128
    elif change == "tp":
        tp = 4
    elif change == "runtime":
        monkeypatch.setattr(torch.version, "git_version", "wrong")
    else:
        def failed():
            raise RuntimeError("WMMA module unavailable")
        monkeypatch.setattr(gguf_ops, "extension", failed)
    with pytest.raises((ValueError, RuntimeError)):
        source.module.configure(source.text, tp=tp, torch=torch)
    assert family.selectors(source.policy) == before and source.module._SELECTED == selected


@pytest.mark.parametrize("pair,sampling,vision,allowed", [
    (get_pair("int8"), None, None, True), (get_pair("int8"), Sampling(17, 0), None, True),
    (get_pair("int8"), Sampling(17, .7), None, False), (get_pair("int8"), Sampling("bad", 0), None, False),
    (get_pair("fp16"), Sampling(17, 0), None, False),
    (get_pair("int8", "int8", "fp16"), Sampling(17, 0), None, False),
    (get_pair("rotorquant-iso4-sm70-v1"), Sampling(17, 0), None, False),
    (get_pair("int8"), Sampling(17, 0), object(), False),
])
def test_wmma_cached_domain_is_only_valid_int8_greedy_text(pair, sampling, vision, allowed):
    weights = NS(meta={"sm70_profile": PROFILE, "execution_provider": "sm70"})
    assert prefix_policy.reuse_allowed(weights, sampling, draft=True, vision=vision,
                                       position=24, kv_pair=pair) is allowed
    assert prefix_policy.serial_reuse_allowed(weights, sampling, draft=True, vision=vision,
                                              position=24, kv_pair=pair) is allowed


def test_wmma_prefix_and_state_epoch_differ_from_every_prior_math():
    identity, epoch = prefix_policy.identity(PROFILE), prefix_policy.state_epoch(PROFILE)
    assert identity and epoch and len(epoch[1]) == 64
    for old in ("reference", "optimized", "source-family-trial", "source-family-prefix-trial", "source-packed-trial"):
        assert identity != prefix_policy.identity(old) and epoch != prefix_policy.state_epoch(old)


@pytest.mark.usefixtures("allocations")
def test_wmma_and_exact_states_refuse_cross_math_snapshot_copy_without_tensor_changes():
    w = state_controls.model(8)
    w.meta["sm70_profile"] = "source-packed-trial"
    exact = State(w, 37, 4, kv_pair=get_pair("int8"), limit=128)
    w.meta["sm70_profile"] = PROFILE
    approximate = State(w, 37, 4, kv_pair=get_pair("int8"), limit=128)
    assert prefix_policy.WMMA80_STATE in approximate.kv_identity
    assert approximate.kv_identity != exact.kv_identity
    assert approximate.cache_bytes() == exact.cache_bytes()
    before = approximate.rec.clone()
    with pytest.raises(ValueError, match="matching cache formats"):
        approximate.copy_prefix(exact, 0, 0)
    with pytest.raises(ValueError, match="matching ordered"):
        approximate.restore(exact.snapshot())
    assert torch.equal(approximate.rec, before)
