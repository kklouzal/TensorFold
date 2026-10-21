"""Unselected native head epilogue: exact arithmetic, boundaries and lifetime."""
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.families.gemma4.cuda import forward, head
from tensorfold.families.gemma4.cuda.engine import Gemma4Engine
from tensorfold.families.gemma4.cuda.state import State
from tensorfold.families.qwen3_5.cuda.weights import Plain

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="native head epilogue requires CUDA")


def target(projector):
    model = object.__new__(forward.Gemma4Target)
    model._native = True
    model.config = NS(softcap=30.)
    model.weights = NS(embed=projector)
    return model


def same_bits(actual, expected):
    assert actual.dtype == expected.dtype == torch.bfloat16 and actual.shape == expected.shape
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("reuse", [None, 2, 4, 8, 16])
def test_prepared_epilogue_pins_parameter_policy_and_native_operation(monkeypatch, reuse):
    original = Plain(torch.ones((4, 4), dtype=torch.bfloat16))
    authority = original.weight
    projector = original if reuse is None else head._RowsHead(original, reuse)
    calls, loads, regions = [], [], []

    def native(hidden, weight, rows):
        calls.append((hidden, weight, rows))
        return hidden.clone()

    monkeypatch.setattr(head, "_cap30_extension", lambda: loads.append(True) or NS(b16_linear_cap30=native))
    monkeypatch.setattr(forward, "original_bf16_region", lambda weight: regions.append(weight) or True)
    model = target(projector)
    prepared = model.prepare_epilogue_logits(projector)
    assert loads == [True] and len(regions) == 1 and regions[0] is original.weight
    monkeypatch.setattr(head, "_cap30_extension", lambda: pytest.fail("HOT native lookup"))
    monkeypatch.setattr(forward, "original_bf16_region", lambda weight: pytest.fail("HOT region discovery"))
    # Stable preparation pins the parameter even if the unused wrapper changes.
    projector.weight = original.weight.clone()
    source = torch.arange(16, dtype=torch.float32).reshape(4, 4).T
    before = source.clone()
    actual = prepared(source)
    hidden, weight, rows = calls[0]
    assert hidden.dtype == torch.bfloat16 and hidden.is_contiguous()
    assert weight is authority
    assert rows == (0 if reuse is None else reuse)
    assert torch.equal(actual.float(), source) and torch.equal(source, before)


@pytest.mark.parametrize("fault", ["foreign", "authority", "region", "cap", "native", "policy"])
def test_epilogue_rejects_unproved_contract_before_loading_native_module(monkeypatch, fault):
    original = Plain(torch.ones((4, 4), dtype=torch.bfloat16))
    projector = head._RowsHead(original, 16)
    model = target(projector)
    monkeypatch.setattr(forward, "original_bf16_region", lambda weight: True)
    monkeypatch.setattr(head, "_cap30_extension", lambda: pytest.fail("loaded rejected native module"))
    if fault == "foreign":
        projector = NS(weight=original.weight)
    elif fault == "authority":
        projector = Plain(original.weight.clone())
    elif fault == "region":
        monkeypatch.setattr(forward, "original_bf16_region", lambda weight: False)
    elif fault == "cap":
        model.config.softcap = 7.75
    elif fault == "native":
        model._native = False
    else:
        projector._multiply = lambda value, weight: value
    with pytest.raises(ValueError, match="original fresh BF16 head"):
        model.prepare_epilogue_logits(projector)


def test_tool_binds_same_epilogue_for_target_and_drafter_before_fresh_graphs(monkeypatch):
    from tools import gemma4_cuda_softcap_qualify as tool

    original = Plain(torch.ones((4, 4), dtype=torch.bfloat16))
    selected = head._RowsHead(original, 16)
    model = target(selected)
    engine = NS(w=model.weights, target=model, head=NS(), drain=lambda: None)
    monkeypatch.setattr(head, "prepare_head", lambda incumbent, reuse_rows: head._RowsHead(incumbent, reuse_rows))
    monkeypatch.setattr(forward, "original_bf16_region", lambda weight: True)
    operation = object()
    monkeypatch.setattr(head, "prepare_cap30", lambda projector: operation)

    def bind(owner, projector):
        assert projector is not selected and projector.incumbent is original
        assert owner.target.project_logits is owner.head.project_logits is operation
        assert owner.w.embed is selected  # cap preparation cannot replace the raw head
        return {"changed": True}

    monkeypatch.setattr(tool.head_bench, "_bind_head", bind)
    modules = {"tensorfold.families.gemma4.cuda.head": head, "tensorfold.families.gemma4.cuda.forward": forward}
    assert tool.bind(engine, "native_epilogue", modules, None) == {"changed": True}


@cuda
@pytest.mark.parametrize("reuse", [None, 16])
def test_native_epilogue_all_65536_weight_encodings_and_underflow_against_raw_head(reuse):
    # Only the first K element contributes. The native raw projection, not the
    # epilogue, determines exceptional NaN encodings and signed-zero dot sums.
    bits = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16)
    weight = torch.zeros((65536, 256), device="cuda", dtype=torch.bfloat16)
    weight[:, 0] = bits.view(torch.bfloat16).to("cuda")
    hidden = torch.zeros((5, 256), device="cuda", dtype=torch.bfloat16)
    hidden[:, 0] = torch.tensor([1., 2.**-133, -1., 2.**-7, -2.**-7], device="cuda", dtype=torch.bfloat16)
    original = Plain(weight)
    projector = original if reuse is None else head._RowsHead(original, reuse)
    raw = projector(hidden)
    assert (raw.view(torch.int16) == -32768).any()  # negative zero from BF16 underflow
    expected = torch.tanh(raw / 30.) * 30.
    before = raw.clone()
    actual = head.prepare_cap30(projector)(hidden)
    same_bits(actual, expected)
    same_bits(raw, before)
    assert actual.data_ptr() not in (raw.data_ptr(), hidden.data_ptr(), weight.data_ptr())


@cuda
@pytest.mark.parametrize("reuse", [None, 2, 4, 8, 16])
@pytest.mark.parametrize("rows", [1, 5, 17, 513])
def test_native_epilogue_matches_raw_head_rounding_at_reuse_and_grid_tails(reuse, rows):
    torch.manual_seed(1983)
    weight = torch.randn((131, 264), device="cuda", dtype=torch.bfloat16)
    hidden = torch.randn((rows, 264), device="cuda", dtype=torch.bfloat16)
    # An exact halfway result and its neighboring cancellation perturbations
    # make accumulator→BF16 rounding observable before the nonlinear cap.
    weight[:3].zero_()
    weight[:3, 0] = 1.
    weight[:3, 1] = 2.**-7
    weight[1, 2], weight[2, 2] = 2.**-16, -2.**-16
    hidden[0].zero_()
    hidden[0, :3] = torch.tensor([1., .5, 1.], device="cuda", dtype=torch.bfloat16)
    original = Plain(weight)
    projector = original if reuse is None else head._RowsHead(original, reuse)
    raw = projector(hidden)
    assert raw[0, 0] == 1. and raw[0, 1] > raw[0, 0] and raw[0, 2] == 1.
    actual = head.prepare_cap30(projector)(hidden)
    same_bits(actual, torch.tanh(raw / 30.) * 30.)


@cuda
@pytest.mark.parametrize("fault", ["rows", "dtype", "stride", "rank", "alignment", "k", "n", "m"])
def test_native_epilogue_boundary_rejects_invalid_inputs(fault):
    native = head._cap30_extension().b16_linear_cap30
    hidden = torch.ones((2, 8), device="cuda", dtype=torch.bfloat16)
    weight = torch.ones((5, 8), device="cuda", dtype=torch.bfloat16)
    rows = 16
    if fault == "rows":
        rows = 3
    elif fault == "dtype":
        hidden, weight = hidden.half(), weight.half()
    elif fault == "stride":
        hidden = torch.ones((8, 2), device="cuda", dtype=torch.bfloat16).T
    elif fault == "rank":
        hidden = hidden[0]
    elif fault == "alignment":
        hidden = torch.ones(17, device="cuda", dtype=torch.bfloat16)[1:].reshape(2, 8)
    elif fault == "k":
        hidden, weight = hidden[:, :7].contiguous(), weight[:, :7].contiguous()
    elif fault == "n":
        weight = weight[:0]
    else:
        hidden = hidden[:0]
    with pytest.raises(RuntimeError):
        native(hidden, weight, rows)


@cuda
def test_native_epilogue_zero_k_and_graph_prefix_snapshot(monkeypatch):
    from test_gemma_cuda_softcap import _config

    native = head._cap30_extension().b16_linear_cap30
    zero = native(torch.empty((2, 0), device="cuda", dtype=torch.bfloat16),
                  torch.empty((5, 0), device="cuda", dtype=torch.bfloat16), 16)
    same_bits(zero, torch.zeros_like(zero))
    torch.manual_seed(907)
    weight = torch.randn((131, 128), device="cuda", dtype=torch.bfloat16)
    projector = head._RowsHead(Plain(weight), 16)
    model = target(projector)
    monkeypatch.setattr(forward, "original_bf16_region", lambda weight: True)
    project = model.prepare_epilogue_logits(projector)
    hidden = torch.randn((5, 128), device="cuda", dtype=torch.bfloat16)
    project(hidden)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = project(hidden)
    state = State(_config(30.), torch.device("cuda", 0), 64, rows=5)
    state.keys.zero_()
    state.values.zero_()
    state.raw.zero_()
    state.inv.fill_(1.)
    state.advance(1)
    saved = None
    for shift in (0, 1, 7, 29):
        hidden.copy_(torch.arange(640, device="cuda", dtype=torch.float32).reshape(5, 128).roll(shift, 1).bfloat16() / 128)
        graph.replay()
        same_bits(output, torch.tanh(projector(hidden) / 30.) * 30.)
        if saved is None:
            saved = Gemma4Engine._checkpoint(NS(_draft=None), state, output)
            snapshot = saved.logits.clone()
            assert saved.logits.data_ptr() != output.data_ptr()
        same_bits(saved.logits, snapshot)
