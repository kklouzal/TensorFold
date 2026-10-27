"""Explicit current-stack graph capability never grants the legacy fast paths."""

from types import SimpleNamespace as NS

import pytest

pytest.importorskip("torch")

from . import test_sm70_family_trial_policy as family  # noqa: E402
from . import test_sm70_rotorquant_state as state_controls  # noqa: E402
from . import test_cuda_geometry as geometry_controls  # noqa: E402
from tensorfold.cuda.sm70 import gguf_ops  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import prefix_policy  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi_graphs import UNIQUE_CAPTURE_CONTRACT  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.state import State  # noqa: E402
from tensorfold.families.qwen4_exp.kv_formats import get_pair  # noqa: E402

policy, source = family.policy, family.source
allocations = geometry_controls.allocations


def request(module, **changes):
    arguments = dict(method="mtp", depth=2, kv_pair=get_pair("int8"), unique_history=True,
                     indexer_history="gpu", graphs=True, concurrent=True)
    arguments.update(changes)
    return module.unique_graph_request(module.PACKED_GRAPH_TRIAL, **arguments)


def test_only_explicit_issued_source_graph_profile_sets_capture_without_other_grants(source, monkeypatch):
    module = source.module
    monkeypatch.setenv(module.PROFILE_ENV, module.PACKED_GRAPH_TRIAL)
    monkeypatch.setattr(gguf_ops, "extension", lambda: source.calls.append("packed-extension"))
    assert module.configure(source.text, tp=8, torch=family.torch) == module.PACKED_GRAPH_TRIAL
    assert family.selectors(source.policy) == (True, False, False, False, False, False,
                                              "packed_exact", "native32", None)
    weights = family.weights(source)
    weights.meta["history_layout"] = "unique"
    weights.mtp = NS(layer=NS(linear=False))
    module.prepare_weights(weights, graph_contract=request(module))
    assert weights.meta["capture_ready"] is True
    assert weights.meta["unique_capture_contract"] == UNIQUE_CAPTURE_CONTRACT
    assert weights.meta["sm70_profile"] == prefix_policy.PACKED_GRAPH_PROFILE
    assert not any(weights.meta.get(name) for name in (
        "device_greedy_mtp_ready", "device_greedy_native_ready", "attention_qualification"))


@pytest.mark.parametrize("change", [{"depth": 3}, {"depth": 1}, {"depth": True}, {"method": "none"},
    {"method": "dspark"}, {"kv_pair": get_pair("fp16")}, {"kv_pair": get_pair("int8", "int8", "fp16")},
    {"unique_history": False}, {"indexer_history": "ram"}, {"graphs": False}, {"concurrent": False}])
def test_wrong_real_method_or_layout_cannot_issue_graph_contract(source, change):
    with pytest.raises(ValueError, match="depth2"):
        request(source.module, **change)


@pytest.mark.parametrize("profile", ["reference", "optimized", "source-family-trial",
                                    "source-family-prefix-trial", "source-packed-trial"])
def test_existing_profiles_do_not_issue_or_validate_graph_request_fields(source, profile):
    assert source.module.unique_graph_request(profile, method=None, depth=None, kv_pair=None,
        unique_history=None, indexer_history=None, graphs=None, concurrent=None) is None


@pytest.mark.parametrize("change", ["no-issuer", "wrong-issuer", "replicated", "no-mtp", "linear-mtp"])
def test_loaded_owner_refuses_an_unissued_or_different_graph_contract(source, monkeypatch, change):
    module = source.module
    monkeypatch.setenv(module.PROFILE_ENV, module.PACKED_GRAPH_TRIAL)
    monkeypatch.setattr(gguf_ops, "extension", lambda: None)
    module.configure(source.text, tp=8, torch=family.torch)
    weights = family.weights(source)
    weights.meta["history_layout"] = "replicated" if change == "replicated" else "unique"
    weights.mtp = None if change == "no-mtp" else NS(layer=NS(linear=change == "linear-mtp"))
    contract = None if change == "no-issuer" else "wrong" if change == "wrong-issuer" else request(module)
    with pytest.raises(ValueError, match="issued depth2"):
        module.prepare_weights(weights, graph_contract=contract)
    assert not weights.meta.get("capture_ready") and "unique_capture_contract" not in weights.meta


def test_prefix_epoch_is_distinct_and_keeps_the_existing_greedy_int8_domain():
    profile = prefix_policy.PACKED_GRAPH_PROFILE
    for old in ("reference", "optimized", "source-family-trial", "source-family-prefix-trial", "source-packed-trial"):
        assert prefix_policy.identity(profile) != prefix_policy.identity(old)
        assert prefix_policy.state_epoch(profile) != prefix_policy.state_epoch(old)
    weights = NS(meta={"execution_provider": "sm70", "sm70_profile": profile})
    for pair, sampling, vision, allowed in ((get_pair("int8"), None, None, True),
            (get_pair("int8"), Sampling(17, .7), None, False), (get_pair("fp16"), None, None, False),
            (get_pair("int8"), None, object(), False)):
        assert prefix_policy.reuse_allowed(weights, sampling, draft=True, vision=vision,
            position=24, kv_pair=pair) is allowed


@pytest.mark.usefixtures("allocations")
def test_graph_state_cannot_borrow_old_profile_prefix_before_mutation(monkeypatch):
    weights = state_controls.model(8)
    weights.meta["sm70_profile"] = prefix_policy.PACKED_GRAPH_PROFILE
    graph = State(weights, 37, 4, kv_pair=get_pair("int8"), limit=128)
    weights.meta["sm70_profile"] = prefix_policy.PACKED_PROFILE
    packed = State(weights, 37, 4, kv_pair=get_pair("int8"), limit=128)
    assert prefix_policy.PACKED_GRAPH_STATE in graph.kv_identity
    assert prefix_policy.PACKED_STATE in packed.kv_identity
    monkeypatch.setattr(packed, "kv_check", lambda: pytest.fail("old profile State was borrowed"))
    with pytest.raises(ValueError, match="source/profile"):
        graph.copy_prefix(packed, 0, 0)
    with pytest.raises(ValueError, match="source/profile"):
        packed.copy_prefix(graph, 0, 0)
