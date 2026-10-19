"""Actual Qwen MoE loader/control paths with opaque payloads, no numerical SDK."""

from __future__ import annotations
import __future__

import ast
import contextlib
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from test_nvfp4_loader_phase_control import fixture

from tensorfold.cuda.nvfp4.format import LinearMetadata

ROOT = Path(__file__).resolve().parents[1]
DENSE = ROOT / "src/tensorfold/families/qwen3_5/cuda/nvfp4_load.py"
SOURCE = ROOT / "src/tensorfold/families/qwen3_5_moe/cuda/nvfp4_load.py"


def definitions(path, scope):
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec",  # noqa: S102 -- Owned source AST only.
                 flags=__future__.annotations.compiler_flag), scope)
    return scope


def context_class():
    node = next(node for node in ast.parse(DENSE.read_text()).body
                if isinstance(node, ast.ClassDef) and node.name == "MLPContext")
    scope = {"__name__": __name__, "dataclass": dataclass}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(DENSE), "exec",  # noqa: S102 -- Owned source AST only.
                 flags=__future__.annotations.compiler_flag), scope)
    return scope["MLPContext"]


def test_real_backbone_mlp_callback_replaces_dense_projection_reads(monkeypatch):
    load, events, state, modules = fixture()
    load.__globals__["MLPContext"] = context_class()
    calls = []
    bank = object()

    def mlp(prefix, context, cfg):
        calls.append((prefix, context, cfg))
        return {"moe": bank}

    with monkeypatch.context() as scoped:
        for name, module in modules.items():
            scoped.setitem(sys.modules, name, module)
        with contextlib.redirect_stdout(io.StringIO()):
            result = load(Path("opaque"), mlp=mlp)
    assert len(calls) == 1 and calls[0][0] == "model.layers.0.mlp."
    assert result.layers[0].moe is bank
    assert (result.layers[0].gate, result.layers[0].up, result.layers[0].down) == (None, None, None)
    assert not any("mlp.gate_proj" in name or "mlp.up_proj" in name or "mlp.down_proj" in name for name in events)
    assert state["closes"] == 1


def test_callback_cpu_read_consumes_the_original_tensor_once_and_failure_closes(monkeypatch):
    load, events, state, modules = fixture()
    load.__globals__["MLPContext"] = context_class()
    original = object()
    reader_type = modules["tensorfold.families.qwen3_5.cuda.weights"]._Tensors

    class Reader(reader_type):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.where = {"expert.raw": None}
            self.files = SimpleNamespace(get=lambda name, device: events.append((name, device)) or original)

    modules["tensorfold.families.qwen3_5.cuda.weights"]._Tensors = Reader

    def mlp(prefix, context, cfg):
        assert context.read("expert.raw", device="cpu") is original
        context.read("expert.raw", device="cpu")

    with monkeypatch.context() as scoped:
        for name, module in modules.items():
            scoped.setitem(sys.modules, name, module)
        with pytest.raises(KeyError, match="expert.raw"):
            load(Path("opaque"), mlp=mlp)
    assert events.count(("expert.raw", "cpu")) == 1
    assert state["closes"] == 1


def qwen_api(**overrides):
    scope = {"__name__": __name__, "dataclass": dataclass, "Path": Path,
             "partial": __import__("functools").partial,
             "SUFFIXES": ("weight", "weight_packed", "weight_scale", "weight_global_scale", "input_global_scale")}
    scope.update(overrides)
    return definitions(SOURCE, scope)


def projection_info(prefix, width=64, dims=64):
    result = {prefix + "gate.weight": ("BF16", [2, dims]),
              prefix + "shared_expert_gate.weight": ("BF16", [1, dims])}
    for field, n, k in (("gate_proj", width, dims), ("up_proj", width, dims), ("down_proj", dims, width)):
        base = prefix + "shared_expert." + field
        result.update({base + ".weight_packed": ("U8", [n, k // 2]),
                       base + ".weight_scale": ("F8_E4M3", [n, k // 16]),
                       base + ".weight_global_scale": ("F32", [1]),
                       base + ".input_global_scale": ("F32", [1])})
    return result


def metadata(prefix):
    return LinearMetadata({"quant_method": "compressed-tensors", "ignore": [prefix + "gate", prefix + "shared_expert_gate"],
                           "config_groups": {"g": {"targets": ["Linear"],
                               "weights": {"num_bits": 4, "type": "float", "group_size": 16},
                               "input_activations": {"num_bits": 4, "type": "float", "group_size": 16,
                                                     "strategy": "tensor_group", "dynamic": "local"}}}})


def test_router_keeps_original_bf16_rows_and_shared_projections_use_canonical_linear():
    prefix = "model.language_model.layers.0.mlp."
    inputs, linear_calls = [], []
    router, gate = object(), object()

    class Result:
        def contiguous(self):
            return self

    def cat(values, dim):
        inputs.append((values, dim))
        return Result()

    cfg = SimpleNamespace(experts=2, hidden=64)
    context = SimpleNamespace(info=projection_info(prefix), metadata=metadata(prefix), checkpoint_math=True,
                              read=lambda name: router if name == prefix + "gate.weight" else gate,
                              linear=lambda name: linear_calls.append(name) or name)
    api = qwen_api(torch=SimpleNamespace(cat=cat))
    api["_router"](context, prefix, cfg)
    shared = api["_shared"](context, prefix, cfg, 64)
    assert inputs == [([router, gate], 0)]
    assert (shared.gate, shared.up, shared.down) == tuple(linear_calls)
    assert linear_calls == [prefix + "shared_expert." + projection for projection in ("gate_proj", "up_proj", "down_proj")]


@pytest.mark.parametrize("change", ["router_float32", "router_shape", "shared_shape", "missing_global_scale"])
def test_actual_metadata_and_geometry_are_checked_before_custom_payload_reads(change):
    prefix = "model.language_model.layers.0.mlp."
    info = projection_info(prefix)
    if change == "router_float32":
        info[prefix + "gate.weight"] = ("F32", [2, 64])
    elif change == "router_shape":
        info[prefix + "gate.weight"] = ("BF16", [3, 64])
    elif change == "shared_shape":
        info[prefix + "shared_expert.gate_proj.weight_packed"] = ("U8", [128, 32])
        info[prefix + "shared_expert.gate_proj.weight_scale"] = ("F8_E4M3", [128, 4])
    else:
        del info[prefix + "shared_expert.gate_proj.weight_global_scale"]
    calls = []
    context = SimpleNamespace(info=info, metadata=metadata(prefix), checkpoint_math=True,
                              read=lambda name: calls.append(name), linear=lambda name: calls.append(name))
    api = qwen_api()
    with pytest.raises(ValueError):
        if change.startswith("router"):
            api["_router"](context, prefix, SimpleNamespace(experts=2, hidden=64))
        else:
            api["_shared"](context, prefix, SimpleNamespace(hidden=64), 64)
    assert not calls


@pytest.mark.parametrize("fused", [False, True])
def test_qwen_loader_delegates_a_full_prefix_and_forwards_eos_without_payload_conversion(fused):
    # A resident bank of experts + 1 cells consumed the shared expert as its last cell: no separate shared load.
    bank = SimpleNamespace(count=257 if fused else 256)
    router, shared, operator = object(), object(), object()
    original_head = object()
    context, cfg = object(), SimpleNamespace(top_k=8, experts=256)
    calls, operators = [], []

    def backbone(path, device, **options):
        calls.append((path, device, options["eos_token_ids"]))
        return SimpleNamespace(head=original_head,
                               fields=options["mlp"]("model.language_model.layers.0.mlp.", context, cfg))

    api = qwen_api(read_config=lambda path: {"text_config": {"moe_intermediate_size": 512,
                                                           "shared_expert_intermediate_size": 1024}},
                   load_backbone=backbone, Routed=lambda *args, **kwargs: (args, kwargs),
                   SimpleNamespace=SimpleNamespace)

    def shared_load(*args):
        calls.append("shared")
        return shared

    def native(actual_bank, actual_shared):
        operators.append((actual_bank, actual_shared))
        return operator

    api.update(_router=lambda *args: router, _shared=shared_load, NativeMoE=native)

    def factory(actual_context, prefix, actual_cfg):
        assert (actual_context, prefix, actual_cfg) == (context, "model.language_model.layers.0.mlp", cfg)
        return bank

    result = api["load_nvfp4"](Path("opaque"), "cuda:0", expert_factory=factory, eos_token_ids=(7, 9))
    assert calls[0] == (Path("opaque"), "cuda:0", (7, 9)) and calls[1:] == ([] if fused else ["shared"])
    assert operators == [(bank, None if fused else shared)]
    assert result.fields == {"moe": ((router, bank, 8), {"run": operator})}
    assert result.head is original_head


def test_missing_bank_factory_is_refused_before_reading_checkpoint_metadata():
    api = qwen_api(read_config=lambda path: pytest.fail("factory refused after metadata read"))
    with pytest.raises(TypeError, match="expert-bank factory"):
        api["load_nvfp4"]("opaque", expert_factory=None)


@pytest.mark.parametrize("quantized", [False, True])
def test_family_weight_dispatch_keeps_affine_and_nvfp4_factories_separate(quantized, monkeypatch):
    path = SOURCE.with_name("weights.py")
    node = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "load")
    calls = []
    scope = {"__name__": __name__, "__package__": "tensorfold.families.qwen3_5_moe.cuda", "Path": Path,
             "load_dense": lambda *args, **kwargs: calls.append(("affine", args, kwargs)) or "affine"}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec",  # noqa: S102 -- Owned source AST only.
                 flags=__future__.annotations.compiler_flag), scope)
    dense = ModuleType("tensorfold.families.qwen3_5.cuda.nvfp4_load")
    dense.quantized = lambda path: quantized
    native = ModuleType("tensorfold.families.qwen3_5_moe.cuda.nvfp4_load")
    native.load_nvfp4 = lambda *args, **kwargs: calls.append(("nvfp4", args, kwargs)) or "nvfp4"
    monkeypatch.setitem(sys.modules, dense.__name__, dense)
    monkeypatch.setitem(sys.modules, native.__name__, native)
    factory = object()
    assert scope["load"]("opaque", eos_token_ids=(7, 9), expert_factory=factory) == ("nvfp4" if quantized else "affine")
    kind, args, options = calls[0]
    assert args == ("opaque",) and options["eos_token_ids"] == (7, 9)
    if quantized:
        assert kind == "nvfp4" and options["expert_factory"] is factory
    else:
        assert kind == "affine" and options["tiled"] is True and options["prefill_mlp"] == ()


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("prefill", [False, True])
def test_prepared_operator_preserves_slots_swiglu_and_decode_fp32(native, prefill, monkeypatch):
    from tensorfold import cuda
    from tensorfold.cuda import nvfp4

    events = []

    class Tensor:
        def __init__(self, name, shape=(3, 64), dtype="bf16", key=None):
            self.name, self.shape, self.dtype, self.key, self.device = name, shape, dtype, key, "cuda"

        def __getitem__(self, key):
            return Tensor(self.name, self.shape, self.dtype, key)

        def copy_(self, source):
            events.append(("copy", self.name, self.key, source.name, source.dtype))

    class Projection:
        def __init__(self, name):
            self.name, self.n, self.act = name, 64, .125 if native else None

        def __call__(self, value, out=None):
            events.append(("linear", self.name, value.name, None if out is None else out.dtype))
            return Tensor(self.name) if out is None else out

        def prefill(self, value):
            events.append(("prefill", self.name, value.name))
            return Tensor(self.name)

    def swiglu(gate, up):
        events.append(("swiglu_bf16", gate.name, up.name))
        return Tensor("activation")

    moe = ModuleType("tensorfold.cuda.moe")
    moe.swiglu_bf16 = swiglu
    moe.router = lambda x, weights, out: events.append(("router", weights))
    moe.select_rows = lambda logits, buf, topk, count: events.append(("select", topk, count))
    monkeypatch.setitem(sys.modules, moe.__name__, moe)
    monkeypatch.setattr(cuda, "moe", moe, raising=False)

    def matmul(mode, value, *, lin, out, f32):
        events.append(("native_down", mode, value.name, lin.name, out.dtype, f32))
        return out

    checkpoint = ModuleType("tensorfold.cuda.nvfp4.checkpoint")
    checkpoint.A4, checkpoint.matmul = 4, matmul
    monkeypatch.setitem(sys.modules, checkpoint.__name__, checkpoint)
    monkeypatch.setattr(nvfp4, "checkpoint", checkpoint, raising=False)

    def bank_run(x, picks, out, *, prefill):
        events.append(("bank", picks.key, out.key, out.dtype, prefill))

    bank = SimpleNamespace(capturable=False, run=bank_run)
    torch = SimpleNamespace(float32="fp32", empty=lambda shape, dtype, device: Tensor("down32", shape, dtype))
    api = qwen_api(torch=torch)
    shared = api["SharedExpert"](Projection("gate"), Projection("up"), Projection("down"))
    operator = api["NativeMoE"](bank, shared)
    buf = SimpleNamespace(logits=Tensor("logits"), pick=Tensor("picks", (3, 3)),
                          y=Tensor("y", (3, 3, 64), "bf16" if prefill else "fp32"),
                          plan=SimpleNamespace(prefill=prefill))
    assert operator(Tensor("input"), "router rows", bank, buf, 2, 256) is buf
    assert operator.capturable is False
    assert events[:3] == [("router", "router rows"), ("select", 2, 256),
                          ("bank", (slice(None, 3), slice(None, 2)),
                           (slice(None, 3), slice(None, 2)), "bf16" if prefill else "fp32", prefill)]
    assert ("swiglu_bf16", "gate", "up") in events
    if prefill:
        assert ("prefill", "down", "activation") in events
        assert not any(event[0] == "native_down" for event in events)
    elif native:
        assert ("native_down", 4, "activation", "down", "fp32", True) in events
    else:
        assert ("linear", "down", "activation", "fp32") in events
    assert events[-1] == ("copy", "y", (slice(None, 3), 2), "down" if prefill else "down32",
                           "bf16" if prefill else "fp32")
