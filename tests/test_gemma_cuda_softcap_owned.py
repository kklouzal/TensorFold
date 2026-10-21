"""Owned-head softcap prototypes: admission, exact bytes and graph/snapshot lifetime."""
from functools import partial
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.families.gemma4.cuda import forward, glue, head
from tensorfold.families.gemma4.cuda.engine import Gemma4Engine
from tensorfold.families.gemma4.cuda.state import State
from tensorfold.families.qwen3_5.cuda import b16
from tensorfold.families.qwen3_5.cuda.weights import Plain
from tools.gemma4_cuda_softcap_qualify import torch_softcap30_owned

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="native owned projection runs CUDA kernels")


def target(projector):
    result = object.__new__(forward.Gemma4Target)
    result._native = True
    result.config = NS(softcap=30.0)
    result.weights = NS(embed=projector)
    return result


@pytest.mark.parametrize("reuse", [None, 16])
def test_preparation_proves_owner_before_binding_writer_and_does_not_rediscover(monkeypatch, reuse):
    # CPU adapters test project-owned ownership/dispatch, not CUDA dot products.
    original = Plain(torch.ones((4, 4), dtype=torch.bfloat16))
    calls = []

    def project(x, weight, *args, **kwargs):
        fresh = x.clone()
        calls.append(fresh)
        return fresh

    monkeypatch.setattr(b16, "matmul", project)
    monkeypatch.setattr(head, "matmul_rows", project)
    projector = original if reuse is None else head._RowsHead(original, reuse)
    model = target(projector)
    probes = []
    monkeypatch.setattr(forward, "original_bf16_region", lambda weight: probes.append(weight) or True)
    prepared = model.prepare_owned_logits(projector, torch_softcap30_owned)
    assert len(probes) == 1 and probes[0] is original.weight
    monkeypatch.setattr(forward, "original_bf16_region", lambda weight: pytest.fail("hot region discovery"))
    x = torch.full((4, 4), 2., dtype=torch.bfloat16)
    before = x.clone()
    actual = prepared(x)
    assert actual is calls[0] and len(calls) == 1
    assert torch.equal(actual, torch.tanh(before / 30.) * 30.)
    assert torch.equal(x, before) and actual.data_ptr() != x.data_ptr()


@pytest.mark.parametrize("fault", ["foreign", "incumbent", "weight", "authority", "callback", "partial",
                                  "calibration", "region", "native"])
def test_unknown_or_changed_projector_fails_before_any_owned_write(monkeypatch, fault):
    original = Plain(torch.ones((4, 4), dtype=torch.bfloat16))

    class Borrowed(Plain):
        def __call__(self, x):
            return x

    projector = head._RowsHead(original, 16)
    model = target(projector)
    monkeypatch.setattr(forward, "original_bf16_region", lambda weight: True)
    if fault == "foreign":
        projector = Borrowed(original.weight)
    elif fault == "incumbent":
        projector.incumbent = Borrowed(original.weight)
    elif fault == "weight":
        projector.weight = original.weight.clone()
    elif fault == "authority":
        projector = Plain(original.weight.clone())
    elif fault == "callback":
        projector._multiply = lambda x, weight: x
    elif fault == "partial":
        projector._multiply = partial(head.matmul_rows, rows=8)
    elif fault == "calibration":
        model.config.softcap = 7.75
    elif fault == "region":
        monkeypatch.setattr(forward, "original_bf16_region", lambda weight: False)
    else:
        model._native = False
    calls = []
    with pytest.raises(ValueError, match="original fresh BF16 head"):
        model.prepare_owned_logits(projector, lambda value: calls.append(value))
    assert not calls


@pytest.mark.parametrize("arm", ["torch_owned", "triton_owned"])
def test_tool_binds_owned_projector_and_drafter_before_graph_preparation(monkeypatch, arm):
    from tools import gemma4_cuda_softcap_qualify as tool

    produced, writes = [], []

    def project(x, weight, *args, **kwargs):
        value = x.clone()
        produced.append(value)
        return value

    def consume(value):
        writes.append(value)
        return torch_softcap30_owned(value)

    monkeypatch.setattr(head, "matmul_rows", project)
    monkeypatch.setattr(glue, "softcap30_owned", consume)
    monkeypatch.setattr(tool, "torch_softcap30_owned", consume)
    original = Plain(torch.ones((4, 4), dtype=torch.bfloat16))
    selected = head._RowsHead(original, 16)
    model = target(selected)
    monkeypatch.setattr(forward, "original_bf16_region", lambda weight: True)
    monkeypatch.setattr(head, "prepare_head", lambda incumbent, reuse_rows: head._RowsHead(incumbent, reuse_rows))
    engine = NS(w=model.weights, target=model, head=NS(), drain=lambda: None)
    hidden = torch.full((4, 4), 2., dtype=torch.bfloat16)

    def bind_graphs(owner, projector):
        assert projector is not selected and projector.incumbent is original
        assert owner.head.project_logits is owner.target.project_logits and not writes
        owner.w.embed = projector
        result = owner.target.project_logits(hidden)
        assert result is produced[0] and writes[0] is result
        return {"changed": True}

    monkeypatch.setattr(tool.head_bench, "_bind_head", bind_graphs)
    modules = {"tensorfold.families.gemma4.cuda.head": head,
               "tensorfold.families.gemma4.cuda.forward": forward,
               "tensorfold.families.gemma4.cuda.glue": glue}
    assert tool.bind(engine, arm, modules, None) == {"changed": True}
    assert len(writes) == len(produced) == 1


def same_bits(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype == torch.bfloat16
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@cuda
@pytest.mark.parametrize("operation", [torch_softcap30_owned, glue.softcap30_owned], ids=["torch-owned", "triton-owned"])
def test_owned_operations_match_all_65536_bf16_bits_in_original_storage(operation):
    value = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).to("cuda")
    expected = torch.tanh(value / 30.) * 30.
    pointer = value.data_ptr()
    result = operation(value)
    assert result is value and result.data_ptr() == pointer
    same_bits(result, expected)  # includes subnormals, signed zeros and NaN payloads


@cuda
@pytest.mark.parametrize("rows", [1, 4, 5])
def test_native_plain_and_reused_heads_return_distinct_owned_results(rows):
    weight = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16)
    x = torch.randn((rows, 128), device="cuda", dtype=torch.bfloat16)
    plain = Plain(weight)
    for projector in (plain, head._RowsHead(plain, 16)):
        assert head.owns_logits_result(projector)
        first, second = projector(x), projector(x)
        assert first.data_ptr() != second.data_ptr() and first._base is second._base is None
        assert first.data_ptr() not in (weight.data_ptr(), x.data_ptr())
        uncapped = second.clone()
        actual = glue.softcap30_owned(first)
        same_bits(actual, torch.tanh(uncapped / 30.) * 30.)
        same_bits(second, uncapped)


@cuda
@pytest.mark.parametrize("operation", [torch_softcap30_owned, glue.softcap30_owned], ids=["torch-owned", "triton-owned"])
def test_graph_producer_rewrites_raw_head_and_engine_prefix_snapshot_owns_capped_copy(monkeypatch, operation):
    # Synthetic head geometry exercises actual b16 allocation/graph operations;
    # bypass only the full-model/SM121 region discriminator, not ownership.
    from test_gemma_cuda_softcap import _config

    torch.manual_seed(901)
    weight = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16)
    projector = head._RowsHead(Plain(weight), 16)
    model = target(projector)
    monkeypatch.setattr(forward, "original_bf16_region", lambda weight: True)
    project = model.prepare_owned_logits(projector, operation)
    hidden = torch.randn((5, 128), device="cuda", dtype=torch.bfloat16)
    project(hidden)  # compile kernels before capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = project(hidden)
    address = output.data_ptr()
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
        expected = torch.tanh(projector(hidden) / 30.) * 30.
        same_bits(output, expected)
        assert output.data_ptr() == address
        if saved is None:
            saved = Gemma4Engine._checkpoint(NS(_draft=None), state, output)
            snapshot = saved.logits.clone()
            assert saved.logits.data_ptr() != address
        same_bits(saved.logits, snapshot)
