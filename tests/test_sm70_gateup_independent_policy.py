"""Independent gate/up ownership is opt-in, preadmitted and State-incompatible."""
from types import SimpleNamespace as NS

import pytest

pytest.importorskip("torch")

from . import test_sm70_family_trial_policy as family  # noqa: E402
from . import test_sm70_rotorquant_state as states  # noqa: E402
from . import test_cuda_geometry as geometry  # noqa: E402
from tensorfold.cuda.sm70 import family_native, gguf_ops, greedy_native  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import history_owner, prefix_policy  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.state import State  # noqa: E402
from tensorfold.families.qwen4_exp.kv_formats import get_pair  # noqa: E402

policy, source = family.policy, family.source
allocations = geometry.allocations


def test_attribute_preparation_failure_preserves_all_previous_selectors(source, monkeypatch):
    module = source.module
    monkeypatch.setenv(module.PROFILE_ENV, module.GATEUP_INDEPENDENT_TRIAL)
    monkeypatch.setattr(family_native, "extension", lambda: NS(prepare_value_guard=lambda: None,
                                                              prepare_owner_prelude=lambda: None))
    monkeypatch.setattr(greedy_native, "extension", lambda: None)
    failure = RuntimeError("selected independent kernel admission failed")
    def refuse():
        raise failure
    monkeypatch.setattr(gguf_ops, "extension", lambda: NS(prepare_experts_exact_independent=refuse))
    before = family.selectors(source.policy)
    with pytest.raises(RuntimeError) as caught:
        module.configure(source.text, tp=8, torch=family.torch)
    assert caught.value is failure
    assert family.selectors(source.policy) == before


def test_candidate_preserves_existing_native_prelude_factory(monkeypatch):
    calls = []
    monkeypatch.setattr(history_owner, "_native_frame_prepare", lambda *a, **k: calls.append(k) or True)
    weights = NS(meta={"sm70_profile": prefix_policy.GATEUP_INDEPENDENT_PROFILE})
    history_owner.prepare(None, None, None, weights, None, 0, 1, mtp=True)
    assert calls == [{"mtp": True}]


@pytest.mark.usefixtures("allocations")
def test_candidate_epoch_refuses_original_prelude_prefix_before_borrow(monkeypatch):
    weights = states.model(8)
    weights.meta["sm70_profile"] = prefix_policy.GATEUP_INDEPENDENT_PROFILE
    candidate = State(weights, 37, 4, kv_pair=get_pair("int8"), limit=128)
    assert prefix_policy.GATEUP_INDEPENDENT_STATE in candidate.kv_identity
    weights.meta["sm70_profile"] = prefix_policy.OWNER_TARGET_PRELUDE_PROFILE
    original = State(weights, 37, 4, kv_pair=get_pair("int8"), limit=128)
    monkeypatch.setattr(original, "kv_check", lambda: pytest.fail("old profile was borrowed"))
    for target, source in [(candidate, original), (original, candidate)]:
        with pytest.raises(ValueError, match="source/profile"):
            target.copy_prefix(source, 0, 0)
