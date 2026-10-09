"""Actual loader control/owner order with opaque values; no numerical SDK."""

from __future__ import annotations

import __future__
import ast
import contextlib
import io
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "src/tensorfold/families/qwen3_5/cuda/nvfp4_load.py"


class Value:
    dtype = object()

    def contiguous(self):
        return self

    def to(self, *args):
        return self

    def float(self):
        return self

    def __add__(self, other):
        return self

    def __neg__(self):
        return self

    def __truediv__(self, other):
        return self

    def __rpow__(self, other):
        return self


def fixture():
    events, state = [], {"closes": 0, "releases": 0}

    def release():
        events.append("release")
        state["releases"] += 1
        if state.get("release_failure") is not None:
            raise state["release_failure"]

    torch = SimpleNamespace(
        bfloat16=object(),
        float32=object(),
        float64=object(),
        arange=lambda *args, **kwargs: Value(),
        cuda=SimpleNamespace(empty_cache=release),
    )
    namespace = dict(
        torch=torch,
        Path=Path,
        checkpoint_path=lambda root, name: root / name,
        read_metadata_json=lambda path: {},
        skipped=lambda name: False,
        __name__="tensorfold.families.qwen3_5.cuda.nvfp4_load",
        __package__="tensorfold.families.qwen3_5.cuda",
        maths=lambda: ({"nvfp4": False, "fp8": False}, "opaque mode"),
        SUFFIXES=("weight",),
        Plain=lambda value: SimpleNamespace(value=value),
        prompt_precision=SimpleNamespace(fp8=lambda: False),
    )
    node = next(
        n for n in ast.parse(SOURCE.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "load_nvfp4"
    )
    exec(
        compile(
            ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec", flags=__future__.annotations.compiler_flag
        ),
        namespace,
    )
    cfg = SimpleNamespace(layers=1, is_linear=lambda index: False, rope_dims=128, rope_theta=10000)
    names = [
        "model.layers.0." + name
        for name in (
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        )
    ]
    names += ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]

    class Reader:
        def __init__(self, *args, **kwargs):
            events.append("reader")

        def pop(self, name):
            events.append(name)
            if name == state.get("failed_key"):
                raise state["read_failure"]
            return Value()

        def __iter__(self):
            return iter(state.get("left", ()))

        def close(self):
            events.append("close")
            state["closes"] += 1
            if state.get("close_failure") is not None:
                raise state["close_failure"]

    weights = ModuleType("tensorfold.families.qwen3_5.cuda.weights")
    weights.__dict__.update(
        GDN=lambda **kw: None,
        Attention=lambda **kw: SimpleNamespace(**kw),
        Layer=lambda **kw: SimpleNamespace(**kw),
        Config=SimpleNamespace(read=lambda path: cfg),
        Weights=lambda **kw: SimpleNamespace(attention_origin=object(), **kw),
        _Tensors=Reader,
    )
    capacity = ModuleType("tensorfold.cuda.capacity")
    capacity.headers = lambda path: {name: {"dtype": "BF16", "shape": [1, 1]} for name in names}
    nv = ModuleType("tensorfold.cuda.nvfp4")
    nv.format = SimpleNamespace(config_block=lambda value: {}, scheme=lambda parts: "bf16")
    linear = ModuleType("tensorfold.cuda.nvfp4.linear")
    linear.__dict__.update(Fp4Linear=object, Fp8Linear=object, Staging=lambda: None)
    modules = {m.__name__: m for m in (weights, capacity, nv, linear)}
    return namespace["load_nvfp4"], events, state, modules


class PhaseControls(unittest.TestCase):
    def invoke(self, load, modules):
        with (
            patch.dict(sys.modules, modules),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            return load(Path("opaque-owner-model"))

    def test_phase_release_precedes_late_weights_and_reader_closes_once(self):
        load, events, state, modules = fixture()
        result = self.invoke(load, modules)
        first = events.index("release")
        self.assertLess(events.index("model.layers.0.mlp.down_proj.weight"), first)
        self.assertLess(first, events.index("model.embed_tokens.weight"))
        self.assertLess(first, events.index("lm_head.weight"))
        self.assertEqual((state["closes"], state["releases"]), (1, 2))
        self.assertEqual(result.precision, "full")

    def test_failed_phase_release_stops_before_late_weights_and_closes_reader(self):
        load, events, state, modules = fixture()
        primary = RuntimeError("allocator release failed")
        state["release_failure"] = primary
        with self.assertRaises(RuntimeError) as caught:
            self.invoke(load, modules)
        self.assertIs(caught.exception, primary)
        self.assertNotIn("model.embed_tokens.weight", events)
        self.assertEqual(state["closes"], 1)

    def test_original_close_and_post_close_refusal_do_not_retry_close(self):
        for failed_close in (False, True):
            load, events, state, modules = fixture()
            primary = OSError("original close failed")
            if failed_close:
                state["close_failure"] = primary
            else:
                state["left"] = ["unused.weight"]
            with self.assertRaises(OSError if failed_close else ValueError) as caught:
                self.invoke(load, modules)
            if failed_close:
                self.assertIs(caught.exception, primary)
            self.assertEqual(state["closes"], 1)

    def test_early_failure_preserves_primary_and_failed_cleanup_annotation(self):
        load, events, state, modules = fixture()

        class Primary(KeyboardInterrupt):
            def add_note(self, message):
                raise AssertionError("foreign annotation hook")

        primary, cleanup = Primary("read interrupted"), OSError("reader close failed")
        primary.__notes__ = 1
        state.update(failed_key="model.layers.0.self_attn.q_proj.weight", read_failure=primary, close_failure=cleanup)
        with self.assertRaises(Primary) as caught:
            self.invoke(load, modules)
        self.assertIs(caught.exception, primary)
        self.assertEqual(state["closes"], 1)
        self.assertIsInstance(primary.__cause__, BaseExceptionGroup)
        self.assertIs(primary.__cause__.exceptions[0], cleanup)
        self.assertIsInstance(primary.__cause__.exceptions[1], TypeError)


if __name__ == "__main__":
    unittest.main()
