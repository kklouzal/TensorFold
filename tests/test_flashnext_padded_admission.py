"""Admission charges optional prepared projections even without native attention."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorfold.cuda.capacity import Geometry
from tensorfold.cuda.sm70 import family_ops
from tensorfold.cuda.sm70 import attention_native
from tensorfold.families.qwen4_exp.cuda import padded_linears


def geometry_wrapper(monkeypatch, streams, each, mtp, native):
    source = Path(__file__).parents[1] / "src/tensorfold/families/qwen4_exp/cuda/engine.py"
    tree = ast.parse(source.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "gguf_geometry"
                and any(isinstance(c, ast.Name) and c.id == "padded_bytes" for c in ast.walk(n)))
    monkeypatch.setattr(family_ops, "_NATIVE_FAMILY", native)
    monkeypatch.setattr(family_ops, "_NATIVE_ATTENTION", native)
    calls = []
    monkeypatch.setattr(padded_linears, "workspace_bytes_from_text",
                        lambda text, rows, world: calls.append((text, rows, world)) or 127)
    monkeypatch.setattr(attention_native, "workspace_bytes", lambda rows, heads, dim, cap: rows*cap)
    base = Geometry(lambda cap: cap*10, 123, 7)
    observed = []
    env = {"__package__": "tensorfold.families.qwen4_exp.cuda", "target_geometry":
           lambda text, count: observed.append(text) or base, "tp": 8, "streams": streams,
           "each": each, "mtp": mtp, "rows0": 2048, "head_dim": 256,
           "concurrent": streams > 1, "row_envelope": lambda count: count * each,
           "history_placement": None}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), env)
    return env["gguf_geometry"], base, calls, observed


@pytest.mark.parametrize("streams,each,mtp,native", [
    (1, 3, False, False), (1, 3, True, False), (6, 3, True, False), (6, 3, True, True),
])
def test_prepared_owners_are_charged_once_per_execution_buffer(monkeypatch, streams, each, mtp, native):
    wrapper, base, calls, observed = geometry_wrapper(monkeypatch, streams, each, mtp, native)
    text = {"num_attention_heads": 24, "num_key_value_heads": 2}
    geometry = wrapper(text)
    rows = streams*each if streams > 1 else max(8, each)
    assert calls == [(text, rows, 8)]
    assert text["num_key_value_heads"] == 2 and observed[0]["num_key_value_heads"] == 8
    assert (geometry.reserve, geometry.minimum_slots) == (base.reserve, base.minimum_slots)
    for capacity in (256, 8192):
        attention = ((1+int(mtp))*rows+256)*capacity if native else 0
        assert geometry.bytes_at(capacity) == base.bytes_at(capacity)+(1+int(mtp))*127+attention


def test_disabled_prepared_and_native_policy_retains_original_geometry(monkeypatch):
    wrapper, base, _calls, _observed = geometry_wrapper(monkeypatch, 6, 3, True, False)
    monkeypatch.setattr(padded_linears, "workspace_bytes_from_text", lambda *args: 0)
    assert wrapper({"num_attention_heads": 24, "num_key_value_heads": 2}) is base


def test_padded_policy_is_in_the_existing_rank_settings_digest():
    source = Path(__file__).parents[1] / "src/tensorfold/families/qwen4_exp/cuda/engine.py"
    tree = ast.parse(source.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_same_settings")
    assignment = next(n for n in ast.walk(method) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "native_policy" for t in n.targets)
                      and isinstance(n.value, ast.List))
    expression = ast.Expression(assignment.value)
    policy = {"family_ops": SimpleNamespace(_NATIVE_FAMILY=True, _NATIVE_ATTENTION=False),
              "padded_linears": SimpleNamespace(_ENABLED=False), "routing_native": SimpleNamespace(_NATIVE_LOGITS=False)}
    before = eval(compile(expression, str(source), "eval"), policy)
    policy["padded_linears"]._ENABLED = True
    after = eval(compile(expression, str(source), "eval"), policy)
    assert before == [True, False, False, False] and after == [True, False, True, False]
    policy["routing_native"]._NATIVE_LOGITS = True
    assert eval(compile(expression, str(source), "eval"), policy) == [True, False, True, True]
