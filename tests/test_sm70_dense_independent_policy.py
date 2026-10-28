"""Combined candidate preserves selected Dense preparation and row/floor contracts."""
import ast
import inspect
from types import SimpleNamespace as NS

import pytest

pytest.importorskip("torch")

from . import test_sm70_family_trial_policy as family  # noqa: E402
from tensorfold.cuda.sm70 import family_native, gguf_ops, greedy_native  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import prefix_policy  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFExperts  # noqa: E402

policy, source = family.policy, family.source


@pytest.mark.parametrize("failure_at", [None, "dense", "independent"])
def test_both_code_families_are_preadmitted_before_any_selector_publication(source, monkeypatch, failure_at):
    module = source.module
    monkeypatch.setenv(module.PROFILE_ENV, module.DENSE_INDEPENDENT_TRIAL)
    before = family.selectors(source.policy), source.policy.weights.GGUFDense.implementation, module._SELECTED
    calls = []
    failure = RuntimeError("selected code admission refused")
    monkeypatch.setattr(family_native, "extension", lambda: NS(prepare_value_guard=lambda: None,
                                                              prepare_owner_prelude=lambda: None))
    def prepare(name, value):
        assert (family.selectors(source.policy), source.policy.weights.GGUFDense.implementation, module._SELECTED) == before
        calls.append(name)
        if failure_at == name:
            raise failure
        return value
    monkeypatch.setattr(gguf_ops, "extension", lambda: NS(
        prepare_dense32=lambda: prepare("dense", 4),
        prepare_experts_exact_independent=lambda: prepare("independent", [[18, 32, 32, 0, 1024, 70]])))
    monkeypatch.setattr(greedy_native, "extension", lambda: None)
    if failure_at is not None:
        with pytest.raises(RuntimeError) as caught:
            module.configure(source.text, tp=8, torch=family.torch)
        assert caught.value is failure
        assert (family.selectors(source.policy), source.policy.weights.GGUFDense.implementation, module._SELECTED) == before
    else:
        assert module.configure(source.text, tp=8, torch=family.torch) == module.DENSE_INDEPENDENT_TRIAL
        assert calls == ["dense", "independent"]
        assert source.policy.weights.GGUFDense.implementation == "native32"
        assert source.policy.weights.GGUFExperts.implementation == "packed_exact_independent"
        assert module._FAMILY_SOFTWARE["dense_function_count"] == 4
        assert module._FAMILY_SOFTWARE["independent_gate_up_functions"] == [[18, 32, 32, 0, 1024, 70]]


def test_combined_epoch_is_distinct_from_both_component_profiles():
    combined = prefix_policy.DENSE_INDEPENDENT_PROFILE
    for old in [prefix_policy.OWNER_TARGET_DENSE_PROFILE, prefix_policy.GATEUP_INDEPENDENT_PROFILE]:
        assert prefix_policy.identity(combined) != prefix_policy.identity(old)
        assert prefix_policy.state_epoch(combined) != prefix_policy.state_epoch(old)


@pytest.mark.parametrize("rows", [33, 48, 64, 72, 96, 256])
def test_selected_expert_row_cap_preserves_reference_fallback_even_for_direct_exact(rows):
    torch = family.torch
    bank = object.__new__(GGUFExperts)
    bank.width, bank.dims, bank.implementation = 80, 2560, "packed_exact_independent"
    assert bank.packed_rows == 32
    x, picks = torch.zeros((rows, 2560), dtype=torch.float16), torch.zeros((rows, 3), dtype=torch.int32)
    expected, calls = object(), []
    bank.forward_reference = lambda *args: calls.append(args) or expected
    assert bank.forward(x, picks) is expected
    assert bank.forward_exact(x, picks, independent_gate_up=True) is expected
    assert len(calls) == 2


def test_combined_metadata_funding_preserves_all_existing_device_workspace():
    from tensorfold.cuda.sm70 import execution_policy as policy
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    tree = ast.parse(inspect.getsource(FlashNextEngine))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                and "GATEUP_INDEPENDENT_TRIAL" in ast.unparse(n.test)
                and len(n.body) == 1 and isinstance(n.body[0], ast.Assign)
                and isinstance(n.body[0].targets[0], ast.Subscript)
                and ast.unparse(n.body[0].targets[0]) == "options['host_workspace_bytes']")
    options = {"device_workspace_bytes": 123456789, "host_workspace_bytes": 1024}
    namespace = dict(self=NS(sm70_profile=policy.DENSE_INDEPENDENT_TRIAL), options=options, tp=8,
                     GATEUP_INDEPENDENT_TRIAL=policy.GATEUP_INDEPENDENT_TRIAL,
                     DENSE_INDEPENDENT_TRIAL=policy.DENSE_INDEPENDENT_TRIAL)
    exec(compile(ast.Module(body=[node], type_ignores=[]), inspect.getfile(FlashNextEngine), "exec"), namespace)
    assert options == {"device_workspace_bytes": 123456789, "host_workspace_bytes": 1024 + 8 * 8192}
