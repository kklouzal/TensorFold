"""Real family compute/commit on CPU reference operators, with a private tap injection."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

from tensorfold.families.qwen4_exp.cuda import forward as target, gdn  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.dspark_taps import TapWorkspace  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.state import Buffers, State  # noqa: E402


def cpu_weights(monkeypatch):
    # Reuse the maintained tiny architecture without importing CUDA test marks
    # or changing its math. Only fixture tensor placement is redirected.
    path = Path(__file__).parent / "cuda/test_flashnext_sm70_family.py"
    module = ast.parse(path.read_text())
    function = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "_small_weights")
    env = {"torch": torch, "np": np, "gdn": gdn}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), env)
    with monkeypatch.context() as placement:
        for name in ("randn", "ones", "zeros", "empty", "tensor", "arange"):
            original = getattr(torch, name)

            def factory(*args, _original=original, **kwargs):
                if "device" in kwargs and torch.device(kwargs["device"]).type == "cuda":
                    kwargs["device"] = "cpu"
                return _original(*args, **kwargs)

            placement.setattr(torch, name, factory)
        weights = env["_small_weights"](True)
    # Put PLE on the tapped layer so a post-PLE hook is a negative control.
    weights.layers[1].ple = weights.layers[0].ple
    weights.layers[0].ple = None
    weights.cfg.ple_layers = [1]
    return weights


@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("cache", ["fp16", "int8"])
def test_pending_safe_hook_leaves_target_logits_and_committed_state_byte_identical(monkeypatch, prefill, cache):
    w = cpu_weights(monkeypatch)
    # AttnScratch sizes its reference partitions from the device's SM count;
    # this CPU arithmetic fixture supplies the actual V100 count, not a CUDA driver.
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *args: SimpleNamespace(multi_processor_count=80))
    rows = 3
    states = [State(w, 16, 8, cache) for _ in range(2)]
    buffers = [Buffers(w, 8, 16, prefill=prefill) for _ in range(2)]
    workspace = TapWorkspace(SimpleNamespace(hidden=256, taps=(0,), capture_before=(1,)), rows, "cpu")
    original_layer, original_ple = target.layer_forward, target.ple_block
    expected, after_ple = [], []
    for enabled, state, buf in zip((False, True), states, buffers):
        buf.ids[:rows].copy_(torch.tensor([4, 7, 19], dtype=torch.int32))
        buf.ple_v.fill_(.125)
        capture = workspace.begin(rows, [(state, 0, rows)]) if enabled else None

        def layer_hook(layer, weights, segs, b, count, pending, **kwargs):
            if enabled:
                before = b.h[:count].clone()
                sums = b.pss[:count].clone()
                capture.before_layer(layer.index, b.h[:count], pending)
                torch.testing.assert_close(b.h[:count].view(torch.int16), before.view(torch.int16), rtol=0, atol=0)
                torch.testing.assert_close(b.pss[:count].view(torch.int32), sums.view(torch.int32), rtol=0, atol=0)
                if layer.index == 1:
                    assert pending is not None
                    mode, branch, weights_, inject = pending
                    materialized, pss = before.clone(), torch.empty_like(sums)
                    from tensorfold.families.qwen4_exp.cuda import glue
                    glue.hc_writeback(before, materialized, pss, 4, mode,
                                      y=branch if mode == 2 else None, wts=weights_,
                                      branch=None if mode == 2 else branch, inject=inject[:count])
                    expected.append(materialized.view(count, 4, 256).float().mean(1).half())
            return original_layer(layer, weights, segs, b, count, pending, **kwargs)

        def ple_hook(layer, weights, segs, b, count):
            original_ple(layer, weights, segs, b, count)
            if enabled and layer.index == 1:
                after_ple.append(b.h[:count].view(count, 4, 256).float().mean(1).half().clone())

        with monkeypatch.context() as hooks:
            if enabled:
                hooks.setattr(target, "layer_forward", layer_hook)
                hooks.setattr(target, "ple_block", ple_hook)
            result = target.compute(w, [(state, 0, rows)], buf).clone()
        if enabled:
            batch = capture.finish()
            torch.testing.assert_close(batch.features.view(torch.int16), expected[0].view(torch.int16), rtol=0, atol=0)
            assert not torch.equal(batch.features, after_ple[0])
            batch.release()
        target.commit(w, state, buf, rows, rows if prefill else 2)
        if not enabled:
            reference = result
        else:
            torch.testing.assert_close(result.view(torch.int32), reference.view(torch.int32), rtol=0, atol=0)
    a, b = states
    assert a.pos == b.pos and a.cur == b.cur
    for name in ("rec", "conv", "ple_tail"):
        torch.testing.assert_close(getattr(a, name), getattr(b, name), rtol=0, atol=0)
    for ka, kb in zip(a.kc, b.kc):
        for name in ("k", "v", "ks", "vs"):
            torch.testing.assert_close(getattr(ka, name)[:a.pos], getattr(kb, name)[:b.pos], rtol=0, atol=0)
    for va, vb in zip(a.ikc, b.ikc):
        torch.testing.assert_close(va[:a.pos], vb[:b.pos], rtol=0, atol=0)
    for va, vb in zip(a.pooled, b.pooled):
        torch.testing.assert_close(va[:a.pos // a.ratio], vb[:b.pos // b.ratio], rtol=0, atol=0)
