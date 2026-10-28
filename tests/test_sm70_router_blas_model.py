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
    assert rb.role_bytes(32, 48) == (419328, 851968)
    assert rb.role_bytes(32, 1) == (395264, 81920)
    source = Path(__file__).parents[1] / "src/tensorfold/families/qwen4_exp/cuda/engine.py"
    tree = ast.parse(source.read_text())
    initialize = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_initialize")
    block = next(n for n in ast.walk(initialize) if isinstance(n, ast.If)
                 and ast.unparse(n.test) == "self.sm70_profile == ROUTER_BLAS_TRIAL")
    options = {"device_workspace_bytes": 0, "host_workspace_bytes": 0}
    scope = dict(self=NS(sm70_profile=rb.PROFILE), ROUTER_BLAS_TRIAL=rb.PROFILE, options=options, tp=8, mtp=True)
    exec(compile(ast.Module(body=[block], type_ignores=[]), str(source), "exec"), scope)
    assert options == {"device_workspace_bytes": 814592, "host_workspace_bytes": 7471104}
    admission = next(n for n in ast.walk(initialize) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Attribute) and t.attr == "capacity_plan" for t in n.targets)
                     and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == "admit")
    assert block.end_lineno < admission.lineno


@pytest.mark.parametrize("profile", [rb.PROFILE, prefix_policy.OWNER_BATCH_PRELUDE_PROFILE])
def test_actual_prefill_pieces_preserve_independent_prompt_math(profile):
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    decoder = object.__new__(MultiDecoder)
    streams = [NS(sid=index, prompt=[1, 2, 3]) for index in range(2)]
    decoder.w = NS(meta={"execution_provider": "sm70", "sm70_profile": profile})
    decoder.fills = {index: (None, None, 0) for index in range(2)}
    decoder._order = lambda: streams
    decoder._pass_rows = lambda: 32
    decoder._points = lambda *_args: []
    assert decoder._pieces() == [(streams[0], 0, 3)]


def test_actual_ple_gate_call_binds_original_real_callee_signature():
    import inspect
    import textwrap
    import torch
    from tensorfold.families.qwen4_exp.cuda import forward, glue

    body = ast.parse(textwrap.dedent(inspect.getsource(forward.ple_block)))
    call = next(n for n in ast.walk(body) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == "ple_gate")
    signature = inspect.signature(glue.ple_gate)
    tensor = torch.empty((3, 4))
    buffers = NS(ple_keys=tensor, ple_vals=tensor, h=tensor, ple_gated=tensor,
                 ple_pss=tensor, prefill=False, router_blas=object())
    bound = eval(compile(ast.Expression(call), "actual_forward_ple_call", "eval"),
                 dict(glue=NS(ple_gate=lambda *a, **kw: signature.bind(*a, **kw)),
                      b=buffers, p=NS(norm_key=tensor, norm_query=tensor),
                      c=NS(eps=1e-6, streams=4), R=3))
    assert bound.arguments["independent_rows"] is True
    assert "prepared" not in bound.arguments


def test_actual_moe_forward_reaches_real_router_prepared_boundary(monkeypatch):
    import inspect
    import torch
    from tensorfold.families.qwen4_exp.cuda import forward
    from tensorfold.cuda import moe

    calls = []
    class SelectedBoundaryReached(Exception):
        pass
    class Prepared:
        def run(self, weight, value, out):
            calls.append((weight, value, out))
            out.fill_(7)
            return out
    select_signature = inspect.signature(moe.select_rows)
    def stop_at_selected_boundary(*args, **kwargs):
        bound = select_signature.bind(*args, **kwargs)
        assert torch.equal(bound.arguments["logits"], torch.full((2, 513), 7.))
        raise SelectedBoundaryReached
    monkeypatch.setattr(moe, "select_rows", stop_at_selected_boundary)
    weight = torch.zeros((513, 2560), dtype=torch.float32)
    mixed = torch.zeros((2, 2560), dtype=torch.float16)
    logits = torch.empty((2, 513), dtype=torch.float32)
    layer = NS(moe=NS(router=weight, experts=NS(kernel="gguf")))
    buffers = NS(mixed=mixed, moe=NS(logits=logits), prefill=False, router_blas=Prepared())
    with pytest.raises(SelectedBoundaryReached):
        forward.moe_block(layer, NS(cfg=NS(top_k=10, experts=512)), buffers, 2)
    assert len(calls) == 1 and calls[0][0] is weight
    assert calls[0][1].data_ptr() == mixed.data_ptr()
    assert calls[0][2].data_ptr() == logits.data_ptr()
    assert rb.role_payload_bytes(32, 48) + rb.role_payload_bytes(32, 1) == 801024
    assert sum(rb.role_bytes(32, n)[0] for n in (48, 1)) == 814592



def test_actual_router_factory_resource_preflight_on_each_rank():
    from tools import v100_backend_probe as probe

    def role(count):
        owner = object.__new__(rb.RolePlans)
        owner.owners = tuple(object() for _ in range(count))
        owner.plans = {id(weight): object() for weight in owner.owners}
        owner.gpu_bytes, owner.host_bytes = rb.role_bytes(32, count)
        owner.gpu_payload_bytes = rb.role_payload_bytes(32, count)
        owner.overflowed = False
        return owner
    engine = NS(sm70_profile=rb.PROFILE, execution_rows=32,
        multi=NS(buf=NS(router_blas=role(48)), mbuf=NS(router_blas=role(1))),
        capacity_plan={"device_workspace_bytes_estimate": 814592, "host_workspace_bytes_estimate": 7471104})
    result = probe.check_router_blas_engine(engine)
    assert result["device_charge_per_rank"] == 814592
    assert result["raw_device_payload_per_rank"] == 801024
    engine.capacity_plan["device_workspace_bytes_estimate"] -= 1
    with pytest.raises(ValueError, match="funding"):
        probe.check_router_blas_engine(engine)
