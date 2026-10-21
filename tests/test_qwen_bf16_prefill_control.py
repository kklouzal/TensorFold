"""Host checks for BF16 prompt allocation and sum-selection contracts; no tensor runtime required."""

from __future__ import annotations
import __future__

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

BASE = Path(__file__).resolve().parents[1] / "src/tensorfold/families/qwen3_5/cuda"
OPERATIONS = ("add_rmsnorm", "gated_norm", "swiglu", "gate_mul")


def functions(filename, scope):
    path = BASE / filename
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in OPERATIONS]
    # Execute only trusted project AST fixture functions; no external code enters this host test.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec",  # noqa: S102
                 flags=__future__.annotations.compiler_flag), scope)
    return SimpleNamespace(**{name: scope[name] for name in OPERATIONS})


@pytest.fixture
def control():
    allocations, launches = [], []

    def tensor(shape, dtype="bf16", device="cuda:0"):
        return SimpleNamespace(shape=shape, dtype=dtype, device=device)

    def empty(shape, dtype, device):
        value = tensor(shape, dtype, device)
        allocations.append(value)
        return value

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((grid, args, kwargs))

            return launch

    scope = {
        "torch": SimpleNamespace(empty=empty, empty_like=lambda x: empty(x.shape, x.dtype, x.device),
                                 bfloat16="bf16", float32="f32"),
        "triton": SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length(),
                                  cdiv=lambda a, b: (a + b - 1) // b),
        **{"_" + name: Kernel() for name in OPERATIONS},
    }
    glue = functions("glue.py", scope)
    prompt = functions("prefill_bf16.py", {"glue": glue})
    return SimpleNamespace(glue=glue, prompt=prompt, allocations=allocations, launches=launches, tensor=tensor)


@pytest.mark.parametrize("name,residual", [("add_rmsnorm", False), ("add_rmsnorm", True),
                                           ("gated_norm", False), ("swiglu", False), ("gate_mul", False)])
def test_prompt_omits_sum_storage_and_keeps_bf16_outputs(control, name, residual):
    t = control.tensor
    if name == "add_rmsnorm":
        args, kwargs, expected = (t((5, 192)), t((5, 192)) if residual else None, t((192,)), 1e-6), {}, (5, 192)
    elif name == "gated_norm":
        args, kwargs, expected = (t((5, 3, 64)), t((5, 3, 64)), t((64,)), 1e-6), {}, (5, 192)
    elif name == "swiglu":
        args, kwargs, expected = (t((5, 1088)), t((5, 1088))), {}, (5, 1088)
    else:
        args, kwargs, expected = (t((5, 3, 64)), t((5, 384))), {"heads": 3, "head_dim": 64}, (5, 192)
    baseline = getattr(control.glue, name)(*args, **kwargs)
    baseline_allocations = len(control.allocations)
    assert baseline[-1].dtype == "f32" and baseline[-1].shape == (5, expected[1] // 64)
    assert control.launches[-1][2]["SUMS"] is True
    control.allocations.clear()
    output = getattr(control.prompt, name)(*args, **kwargs)
    assert len(control.allocations) == baseline_allocations - 1
    assert all(value.dtype == "bf16" and value.shape == expected for value in control.allocations)
    assert control.launches[-1][2]["SUMS"] is False
    retained = output if name == "add_rmsnorm" else (output,)
    assert all(value.shape == expected and value.dtype == "bf16" and value.device == "cuda:0" for value in retained)
    if name == "add_rmsnorm" and not residual:
        assert output[0] is args[0]
    omitted = getattr(control.glue, name)(*args, **kwargs, group_sums=False)
    assert omitted[-1] is None

