"""Ordered-pair owner and boundary regressions; malformed input must not mutate state."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from tensorfold.families.qwen4_exp.kv_formats import get_pair
from tests.test_cuda_geometry import allocations  # noqa: F401
from tests.test_flashnext_kv_pairs import make_state, context_tensors


@pytest.mark.torch
@pytest.mark.parametrize("changed", ["w", "buf", "mbuf", "st"])
@pytest.mark.parametrize("route", ["forward", "mtp_forward"])
def test_graph_owner_must_reject_changed_static_capture_owner_before_staging(monkeypatch, changed, route):
    import torch
    from tensorfold.families.qwen4_exp.cuda import graphs

    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: None)
    state = SimpleNamespace(kv_identity=(get_pair("bf16").identity, "stored-basis-native64-v1"), version=0)
    engine = SimpleNamespace(w=object(), st=state, buf=object(), mbuf=object())
    captured = graphs.Graphs(engine)
    calls = []
    def forbidden_stage(*args, **kwargs):
        calls.append(args)
        raise AssertionError("stale graph owner reached staging")
    monkeypatch.setattr(graphs, "stage", forbidden_stage)
    monkeypatch.setattr(graphs, "mtp_stage", forbidden_stage)
    if changed == "st":
        engine.st = SimpleNamespace(kv_identity=state.kv_identity, version=0)
    else:
        setattr(engine, changed, object())
    with pytest.raises(ValueError, match="original|owner|buffer|capture"):
        if route == "forward":
            captured.forward([7])
        else:
            captured.mtp_forward([7], object())
    assert calls == []


@pytest.mark.torch
@pytest.mark.parametrize("dtype,key,value", [("bf16", None, None), ("bf16", "int8", "int8"), ("int8", "int4", None),
                                             ("bf16", "rotorquant8-norm", "int8")])
def test_legacy_state_dtype_reports_effective_symmetric_format_or_mixed_error(allocations, dtype, key, value):  # noqa: F811
    pair = get_pair(dtype, key, value)
    state = make_state(pair)
    if pair.symmetric:
        assert state.kv_dtype == pair.key_dtype
    else:
        with pytest.raises(ValueError, match="mixed"):
            _ = state.kv_dtype
    assert state.kv_key_dtype == pair.key_dtype
    assert state.kv_value_dtype == pair.value_dtype


@pytest.mark.torch
@pytest.mark.parametrize("field", ["pos", "mtp_len"])
@pytest.mark.parametrize("bad", [True, False, 1., -1, 1000, "1", None])
def test_snapshot_bad_counts_reject_before_mutating_recurrent_or_context(allocations, field, bad):  # noqa: F811
    import torch
    state = make_state(get_pair("bf16", "rotorquant8-norm", "int8"))
    state.set_pos(4)
    state.set_mtp_len(3)
    snapshot = state.snapshot()
    snapshot[field] = bad
    snapshot["rec"].fill_(7)
    snapshot["conv"].fill_(11)
    recurrent = state.rec.clone()
    convolution = state.conv.clone()
    caches = [tensor.clone() for tensor in context_tensors(state)]
    with pytest.raises(ValueError):
        state.restore(snapshot)
    assert torch.equal(state.rec, recurrent) and torch.equal(state.conv, convolution)
    assert state.pos == 4 and state.mtp_len == 3
    for original, current in zip(caches, context_tensors(state), strict=True):
        assert torch.equal(original.view(torch.uint8), current.view(torch.uint8))


@pytest.mark.torch
@pytest.mark.parametrize("bad", [(False, -2), (0., -2), (0, -2.), (0, True), [0, -2], None, (0,), (0, -2, 0)])
def test_snapshot_numeric_status_requires_exact_validated_types_before_restore(allocations, bad):  # noqa: F811
    import torch
    state = make_state(get_pair("rotorquant8-norm"))
    snapshot = state.snapshot()
    snapshot["kv_status"] = bad
    original = state.rec.clone()
    snapshot["rec"].fill_(23)
    with pytest.raises(ValueError):
        state.restore(snapshot)
    assert torch.equal(state.rec, original)


@pytest.mark.torch
@pytest.mark.parametrize("field", ["pos", "mtp_len"])
@pytest.mark.parametrize("bad", [True, False, 1., -1, "1", None])
def test_prefix_bad_counts_reject_before_any_destination_write(allocations, field, bad):  # noqa: F811
    import torch
    pair = get_pair("bf16", "rotorquant8-norm", "int4")
    source, destination = [make_state(pair) for _ in range(2)]
    source.set_pos(4)
    source.set_mtp_len(3)
    for tensor in context_tensors(source):
        tensor.view(torch.uint8).fill_(9)
    previous = [tensor.clone() for tensor in context_tensors(destination)]
    counts = {"pos": 4, "mtp_len": 3, field: bad}
    with pytest.raises(ValueError):
        destination.copy_prefix(source, **counts)
    for original, current in zip(previous, context_tensors(destination), strict=True):
        assert torch.equal(original.view(torch.uint8), current.view(torch.uint8))
