"""Private numeric selection and actual owner/resource boundaries, CPU only."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tensorfold.cuda.sm70 import router_blas as rb
from tensorfold.families.qwen4_exp.cuda import prefix_policy


def test_role_storage_is_shared_and_weight_owners_stay_retained(monkeypatch):
    calls = []
    class Plan:
        def __init__(self, weight, rows, shared):
            calls.append((weight, rows, shared))
        def run(self, value, out):
            return out
    monkeypatch.setattr(rb, "extension", lambda: NS(RouterPlan=Plan))
    weights = [object(), object(), object()]
    role = rb.RolePlans(weights, 32)
    assert role.owners == tuple(weights)
    assert calls[0][2] is None
    assert calls[1][2] is calls[2][2] is role.plans[id(weights[0])]
    sentinel = object()
    assert role.run(weights[2], None, sentinel) is sentinel
    assert role.calls == 1 and not role.overflowed
    role.calls = (1 << 63) - 1
    role.run(weights[0], None, sentinel)
    assert role.calls == (1 << 63) - 1 and role.overflowed
    with pytest.raises(ValueError, match="not admitted"):
        role.run(object(), None, sentinel)


def test_actual_role_creation_excludes_prefill_and_graph_stream_buffers(monkeypatch):
    made = []
    monkeypatch.setattr(rb, "RolePlans", lambda weights, rows: made.append((weights, rows)) or made[-1])
    target, draft = object(), object()
    w = NS(meta={"sm70_profile": rb.PROFILE}, layers=[NS(moe=NS(router=target))],
           mtp=NS(layer=NS(moe=NS(router=draft))))
    assert rb.create_role(w, 32, prefill=True, role=None) is None
    assert rb.create_role(w, 3, prefill=False, role=None, eager=False) is None
    assert not made
    assert rb.create_role(w, 32, prefill=False, role=None) == ([target], 32)
    assert rb.create_role(w, 32, prefill=False, role="mtp") == ([draft], 32)
    w.meta["sm70_profile"] = prefix_policy.OWNER_BATCH_PRELUDE_PROFILE
    assert rb.create_role(w, 32, prefill=False, role=None) is None
    assert len(made) == 2


def test_private_numeric_epoch_is_distinct_and_fixed_roles_are_funded_before_admit():
    assert prefix_policy.identity(rb.PROFILE) != prefix_policy.identity(prefix_policy.OWNER_BATCH_PRELUDE_PROFILE)
    assert prefix_policy.state_epoch(rb.PROFILE) != prefix_policy.state_epoch(prefix_policy.OWNER_BATCH_PRELUDE_PROFILE)
    assert rb.role_bytes(32, 48) == (406528, 851968)
    assert rb.role_bytes(32, 1) == (394496, 81920)
    source = Path(__file__).parents[1] / "src/tensorfold/families/qwen4_exp/cuda/engine.py"
    tree = ast.parse(source.read_text())
    initialize = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_initialize")
    block = next(n for n in ast.walk(initialize) if isinstance(n, ast.If)
                 and ast.unparse(n.test) == "self.sm70_profile == ROUTER_BLAS_TRIAL")
    options = {"device_workspace_bytes": 0, "host_workspace_bytes": 0}
    scope = dict(self=NS(sm70_profile=rb.PROFILE), ROUTER_BLAS_TRIAL=rb.PROFILE, options=options, tp=8, mtp=True)
    exec(compile(ast.Module(body=[block], type_ignores=[]), str(source), "exec"), scope)
    assert options == {"device_workspace_bytes": 801024, "host_workspace_bytes": 7471104}
    admission = next(n for n in ast.walk(initialize) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Attribute) and t.attr == "capacity_plan" for t in n.targets)
                     and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == "admit")
    assert block.end_lineno < admission.lineno
