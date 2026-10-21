"""Actual resident/offload estimator selection through real shared admission."""

import ast
import math
from functools import partial
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tensorfold.cuda import build, capacity
from tensorfold.families.qwen3_5.cuda import nvfp4_memory

ROOT = Path(__file__).parents[1]
ENGINE = ROOT / "src/tensorfold/families/qwen3_5_moe/cuda/nvfp4_engine.py"


def selection(monkeypatch, *, offload=False, available=1 << 30, headers=None):
    tree = ast.parse(ENGINE.read_bytes())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "QwenNvfp4MoeEngine")
    start = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_start")
    targets = {"resident_estimate", "(resident, host_weight_staging)", "transform", "self.capacity_plan"}
    selected = [node for node in start.body if isinstance(node, ast.Assign)
                and any(ast.unparse(left) in targets for left in node.targets)]
    assert len(selected) == 4
    source = ROOT / "src/tensorfold/families/qwen3_5/cuda/nvfp4_load.py"
    functions = [node for node in ast.parse(source.read_bytes()).body
                 if isinstance(node, ast.FunctionDef) and node.name in {"skipped", "weight_bytes"}]
    scope = {"math": math, "prompt_precision": NS(fp8=lambda: False)}
    exec(compile(ast.fix_missing_locations(ast.Module(functions, [])), str(source), "exec"), scope)  # noqa: S102 - actual owned pure metadata functions.
    fields = headers if headers is not None else {
        "model.embed_tokens.weight": {"dtype": "BF16", "shape": [4, 64]},
        "lm_head.weight": {"dtype": "BF16", "shape": [4, 64]},
        "model.norm.weight": {"dtype": "BF16", "shape": [64]},
        "model.layers.0.mlp.weight_packed": {"dtype": "U8", "shape": [64, 32]},
        "model.layers.0.mlp.weight_scale": {"dtype": "F8_E4M3", "shape": [64, 4]},
        "model.layers.0.mlp.weight_global_scale": {"dtype": "F32", "shape": [1]},
        "model.layers.0.mlp.input_global_scale": {"dtype": "F32", "shape": [1]},
    }
    model_dir, calls, admitted = Path("owned-target"), [], []
    monkeypatch.setattr(capacity, "headers", lambda directory: fields)
    monkeypatch.setattr(capacity, "config", lambda directory: {"max_position_embeddings": 64})
    monkeypatch.setattr(capacity, "floor", lambda directory: (12, 0))
    monkeypatch.setattr(build, "refuse_old_gpu", lambda _floor: None)
    monkeypatch.setattr(capacity, "unified", lambda _runtime: False)
    monkeypatch.setattr(capacity, "host_stream_bytes", lambda: None)
    monkeypatch.setattr(capacity, "host_cgroup_bytes", lambda: None)
    monkeypatch.setattr(capacity, "page_room", lambda _runtime: None)
    monkeypatch.setattr(capacity, "available_bytes", lambda _runtime: available)

    def estimate(directory, transform, *, scalar_batches):
        assert directory == model_dir and scalar_batches is True
        value = nvfp4_memory.weight_estimate(directory, transform, scalar_batches=scalar_batches)
        calls.append((directory, transform, value))
        return value

    def admit(directory, *args, **kwargs):
        assert directory == model_dir
        callback = kwargs["weight_estimator"]

        def observe(path):
            assert path == model_dir
            result = callback(path)
            admitted.append(result)
            return result

        kwargs["weight_estimator"] = observe
        return capacity.admit(directory, *args, **kwargs)

    weight_bytes = scope["weight_bytes"]
    plan = NS(transform=lambda original: lambda name, info: (0, 0) if ".mlp." in name else original(name, info),
              host_bytes=0, gpu_bytes=0, automatic=False) if offload else None
    scope.update(model_dir=model_dir, weight_estimate=estimate, plan=plan, partial=partial,
                 self=NS(rejection_workspace={"device_bytes": 0, "host_bytes": 0}), admit=admit,
                 context=16, context_explicit=True, torch=object(),
                 runtime=NS(geometry=capacity.Geometry(lambda slots: slots * 8, 1)), host_extra=0,
                 allocate=lambda value: value, draft_dir=None, drafted=None, draft_runtime=None, yarn=None)
    exec(compile(ast.fix_missing_locations(ast.Module(selected, [])), str(ENGINE), "exec"), scope)  # noqa: S102 - actual engine selection and admission call.
    return scope, calls, admitted, weight_bytes


@pytest.mark.parametrize("offload", [False, True])
def test_admission_uses_same_frozen_resident_result_or_new_offload_transform(monkeypatch, offload):
    scope, calls, admitted, weight_bytes = selection(monkeypatch, offload=offload)
    assert calls[0][1] is weight_bytes and len(admitted) == 1
    original = calls[0][2]
    if offload:
        assert len(calls) == 2 and calls[1][1] is scope["transform"]
        assert admitted[0] is calls[1][2]
        assert admitted[0][0].resident < original[0].resident
    else:
        assert len(calls) == 1 and admitted[0] is original
        assert scope["transform"] is weight_bytes
    selected, host = admitted[0]
    assert type(host) is int and host >= 0
    assert scope["self"].capacity_plan["weight_bytes_estimate"] == selected.resident
    assert scope["self"].capacity_plan["loading_bytes_estimate"] == selected.staging


def test_reuse_does_not_bypass_live_budget_refusal(monkeypatch):
    with pytest.raises(ValueError, match="cannot fit|does not fit|memory"):
        selection(monkeypatch, available=1)


def test_independent_construction_recomputes_its_own_changed_inputs(monkeypatch):
    first, _, _, _ = selection(monkeypatch)
    changed = {"model.embed_tokens.weight": {"dtype": "BF16", "shape": [8, 64]},
               "lm_head.weight": {"dtype": "BF16", "shape": [8, 64]},
               "model.norm.weight": {"dtype": "BF16", "shape": [64]}}
    second, calls, admitted, _ = selection(monkeypatch, headers=changed)
    assert len(calls) == 1 and admitted[0] is calls[0][2]
    assert first["resident_estimate"][0].resident != second["resident_estimate"][0].resident
