"""Explicit combined policy; native authority stays on canonical State owners."""
from types import SimpleNamespace as NS

import pytest

pytest.importorskip("torch")

from . import test_sm70_family_trial_policy as family  # noqa: E402
from . import test_sm70_rotorquant_state as state_controls  # noqa: E402
from . import test_cuda_geometry as geometry_controls  # noqa: E402
from tensorfold.cuda.sm70 import gguf_ops, greedy_native, family_native  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import prefix_policy  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi_graphs import UNIQUE_NATIVE_OWNER_CAPTURE_CONTRACT  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.owner_execution import KERNEL_FAMILY, UNCAPTURED, FULL_COLLECTIVE  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.state import State  # noqa: E402
from tensorfold.families.qwen4_exp.kv_formats import get_pair  # noqa: E402

policy, source = family.policy, family.source
allocations = geometry_controls.allocations


def specification():
    return dict(kernel_family=KERNEL_FAMILY, graph_family=UNCAPTURED,
                collective_family=FULL_COLLECTIVE, graph_budget_bytes=0, trial=True)


def request(module, **changes):
    args = dict(method="mtp", depth=2, kv_pair=get_pair("int8"), unique_history=True,
                indexer_history="gpu", graphs=True, concurrent=True, vision=False,
                owner_preparation=specification())
    args.update(changes)
    return module.unique_graph_request(module.OWNER_GRAPH_TRIAL, **args)


@pytest.mark.parametrize("target,guard,prelude,dense,independent", [(False, False, False, False, False), (True, False, False, False, False), (True, True, False, False, False), (True, True, True, False, False), (True, True, True, True, False), (True, True, True, False, True), (True, True, True, True, True)])
def test_new_profile_requires_issued_native_owner_and_grants_only_exact_nonattention_pieces(source, monkeypatch, target, guard, prelude, dense, independent):
    module = source.module
    selected = module.DENSE_INDEPENDENT_TRIAL if dense and independent else module.GATEUP_INDEPENDENT_TRIAL if independent else module.OWNER_TARGET_DENSE_TRIAL if dense else module.OWNER_TARGET_PRELUDE_TRIAL if prelude else module.OWNER_TARGET_GUARD_TRIAL if guard else module.OWNER_TARGET_GREEDY_TRIAL if target else module.OWNER_GRAPH_TRIAL
    monkeypatch.setenv(module.PROFILE_ENV, selected)
    if guard:
        monkeypatch.setattr(family_native, "extension", lambda: source.calls.append("extension") or
                            NS(prepare_value_guard=lambda: source.calls.append("guard-code"),
                               prepare_owner_prelude=lambda: source.calls.append("prelude-code")))
    monkeypatch.setattr(greedy_native, "extension", lambda: None)
    monkeypatch.setattr(gguf_ops, "extension", lambda: NS(prepare_dense32=lambda: source.calls.append("dense-code") or 4,
        prepare_experts_exact_independent=lambda: source.calls.append("independent-code") or [[18, 32, 32, 0, 1024, 70]]))
    assert module.configure(source.text, tp=8, torch=family.torch) == selected
    assert family.selectors(source.policy) == (True, False, False, False, False, False,
                                              "packed_exact_independent" if independent else "packed_exact", "native32", None)
    assert source.policy.weights.GGUFDense.implementation == ("native32" if dense else "reference")
    assert ("dense-code" in source.calls) is dense
    assert ("independent-code" in source.calls) is independent
    weights = family.weights(source)
    if prelude:
        weights.device = "cpu"
        monkeypatch.setattr(family.torch.cuda, "current_stream", lambda *a: NS(synchronize=lambda: None))
    weights.meta.update(history_layout="unique", owner_workspaces={role: object() for role in ("decode", "mtp", "prefill")})
    weights.mtp = NS(layer=NS(linear=False))
    contract = module.unique_graph_request(selected, method="mtp", depth=2, kv_pair=get_pair("int8"),
        unique_history=True, indexer_history="gpu", graphs=True, concurrent=True, vision=False,
        owner_preparation=specification())
    assert contract == UNIQUE_NATIVE_OWNER_CAPTURE_CONTRACT
    module.prepare_weights(weights, graph_contract=contract)
    assert ("sm70_target_greedy_trial" in weights.meta) is target
    assert weights.meta.get("sm70_native_guard_trial", False) is guard
    if guard:
        assert source.calls[:3] == ["audit", "extension", "guard-code"]
    if dense:
        assert weights.meta["sm70_family_trial"]["dense_math"] == "original-fixed-m32-aten-v1"
        assert weights.meta["sm70_family_trial"]["dense"] == "native32"
    assert weights.meta["capture_ready"] is True
    assert weights.meta["unique_capture_contract"] == contract
    assert weights.meta["sm70_family_trial"]["attention"] == "prepared-native-owner-eager-trial"
    assert not any(weights.meta.get(name) for name in (
        "device_greedy_mtp_ready", "device_greedy_native_ready", "attention_qualification"))


@pytest.mark.parametrize("change", [None, {"graph_family": "owner-attention-v1", "graph_budget_bytes": 1},
    {"collective_family": "head4-allgather-v1"}, {"trial": False}, {"graph_budget_bytes": True},
    {"kernel_family": "torch-reference"}])
def test_combination_refuses_missing_captured_head4_or_unissued_owner_preparation(source, change):
    owner = None if change is None else {**specification(), **change}
    with pytest.raises(ValueError, match="native eager full8"):
        request(source.module, owner_preparation=owner)


@pytest.mark.usefixtures("allocations")
def test_combined_prefix_epoch_cannot_borrow_old_graph_or_eager_state_before_mutation(monkeypatch):
    weights = state_controls.model(8)
    weights.meta["sm70_profile"] = prefix_policy.OWNER_GRAPH_PROFILE
    combined = State(weights, 37, 4, kv_pair=get_pair("int8"), limit=128)
    assert prefix_policy.OWNER_GRAPH_STATE in combined.kv_identity
    for old in (prefix_policy.PACKED_GRAPH_PROFILE, prefix_policy.PACKED_PROFILE):
        weights.meta["sm70_profile"] = old
        other = State(weights, 37, 4, kv_pair=get_pair("int8"), limit=128)
        assert combined.kv_identity != other.kv_identity
        monkeypatch.setattr(other, "kv_check", lambda: pytest.fail("old numerical profile was borrowed"))
        with pytest.raises(ValueError, match="source/profile"):
            combined.copy_prefix(other, 0, 0)
        with pytest.raises(ValueError, match="source/profile"):
            other.copy_prefix(combined, 0, 0)


@pytest.mark.parametrize("profile,slots,refused", [
    (prefix_policy.DENSE_INDEPENDENT_PROFILE, 23, True),
    (prefix_policy.DENSE_INDEPENDENT_PROFILE, 24, False),
    (prefix_policy.DENSE_INDEPENDENT_PROFILE, 26, False),
    (prefix_policy.GATEUP_INDEPENDENT_PROFILE, 23, True),
    (prefix_policy.GATEUP_INDEPENDENT_PROFILE, 24, False),
    (prefix_policy.OWNER_TARGET_DENSE_PROFILE, 23, True),
    (prefix_policy.OWNER_TARGET_DENSE_PROFILE, 24, False),
    (prefix_policy.OWNER_TARGET_DENSE_PROFILE, 26, False),
    (prefix_policy.OWNER_TARGET_PRELUDE_PROFILE, 23, True),
    (prefix_policy.OWNER_TARGET_PRELUDE_PROFILE, 24, False),
    (prefix_policy.OWNER_TARGET_PRELUDE_PROFILE, 26, False),
    (prefix_policy.OWNER_TARGET_GUARD_PROFILE, 23, True),
    (prefix_policy.OWNER_TARGET_GUARD_PROFILE, 24, False),
    (prefix_policy.OWNER_TARGET_GUARD_PROFILE, 26, False),
    (prefix_policy.OWNER_TARGET_GREEDY_PROFILE, 23, True),
    (prefix_policy.OWNER_TARGET_GREEDY_PROFILE, 24, False),
    (prefix_policy.OWNER_TARGET_GREEDY_PROFILE, 26, False),
    (prefix_policy.OWNER_GRAPH_PROFILE, 23, True),
    (prefix_policy.OWNER_GRAPH_PROFILE, 24, False),
    (prefix_policy.OWNER_GRAPH_PROFILE, 26, False),
    (prefix_policy.PACKED_GRAPH_PROFILE, 1, False),
    ("reference", 1, False)])
def test_actual_engine_resident_floor_refuses_before_window_publication_or_weight_work(profile, slots, refused):
    import ast
    import inspect
    import textwrap
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    # Like the existing Engine graph-intent controls, execute the real ordered
    # constructor control using an admitted-plan result, without payload/GPU work.
    method = ast.parse(textwrap.dedent(inspect.getsource(FlashNextEngine._initialize))).body[0]
    selected = next(node for node in method.body if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name) and node.test.id == "adaptive_sessions"
        and any(isinstance(child, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "streams"
            for target in child.targets) for child in node.body))
    boundary = next(index for index, child in enumerate(selected.body) if isinstance(child, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "resident_windows" for target in child.targets))
    # The next two actual statements publish the resident local/Engine values.
    control = ast.Module(body=selected.body[:boundary + 2], type_ignores=[])
    load = next(node for node in method.body if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "load")
    assert selected.lineno < load.lineno
    engine = NS(sm70_profile=profile, capacity_plan={"session_capacity": {"resident_capacity": slots}},
                streams="prior", kv_full_windows="prior")
    namespace = {"self": engine, "resident_windows": "prior"}
    code = compile(control, inspect.getfile(FlashNextEngine), "exec")
    if refused:
        with pytest.raises(ValueError, match="at least24"):
            exec(code, namespace)
        assert namespace["resident_windows"] == engine.streams == engine.kv_full_windows == "prior"
    else:
        exec(code, namespace)
        assert namespace["resident_windows"] == engine.streams == engine.kv_full_windows == slots
