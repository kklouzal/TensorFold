"""Private grouped gate/up preparation contracts; native arithmetic is not mocked."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace as NS

import pytest

from tensorfold.cuda.sm70 import gguf_ops
from tensorfold.families.qwen4_exp.cuda import owned_experts
from tensorfold.families.qwen4_exp.expert_partition import ExpertPartition


@pytest.mark.parametrize("rows", [True, False, 0, 1, 3, 8, 2.0, "2", None])
def test_private_selector_rejects_wrong_row_domains(rows):
    with pytest.raises(ValueError, match="exactly 2 or 4"):
        gguf_ops._GroupedGateUp(rows)


@pytest.mark.parametrize("rows", [2, 4])
def test_private_selector_is_immutable(rows):
    selection = gguf_ops._GroupedGateUp(rows)
    with pytest.raises(FrozenInstanceError):
        selection.rows = 1


@pytest.mark.parametrize("rows", [None, 2, 4])
def test_wrapper_explicit_selection_borrows_existing_buffers(monkeypatch, rows):
    import torch

    selected = None if rows is None else gguf_ops._GroupedGateUp(rows)
    calls = []
    native = NS(experts_owned=lambda *a: calls.append(("pair", a)),
                experts_owned_grouped=lambda *a: calls.append(("down", a)),
                _experts_owned_grouped_gate_up=lambda *a: calls.append(("gate-up", a)))
    monkeypatch.setattr(gguf_ops, "extension", lambda: native)
    x = torch.empty((2, 2560), dtype=torch.float16)
    picks = torch.empty((2, 11), dtype=torch.int32)
    local = torch.empty_like(picks, dtype=torch.int64)
    activation = torch.empty((22, 160), dtype=torch.float16)
    out = torch.empty((2, 11, 2560), dtype=torch.float32)
    grouping = torch.empty(gguf_ops.grouped_workspace_words(22, 256), dtype=torch.int32)
    shared = torch.empty_like(x, dtype=torch.float32)
    bank = NS(data=object(), ggml_type=18, column_offset=0)
    def call(grouping, selection):
        return gguf_ops.experts_owned(x, picks, bank, bank, bank, first=0, experts=512, width=160,
            shared=shared, local=local, activation=activation, out=out, grouping=grouping, _gate_up=selection)
    assert call(grouping, selected) is out
    name, arguments = calls.pop()
    assert name == ("down" if rows is None else "gate-up")
    assert all(a is b for a, b in zip(arguments[14:19], (shared, local, activation, out, grouping)))
    assert arguments[19:] == (() if rows is None else (rows,))
    if rows is None:
        call(None, None)
        assert calls.pop()[0] == "pair"
    else:
        with pytest.raises(ValueError, match="typed selector"):
            call(None, selected)
    with pytest.raises(ValueError, match="typed selector"):
        call(grouping, 2)
    assert calls == []


@pytest.mark.parametrize("rows", [2, 4])
def test_prepared_identity_and_arena_bind_private_variant_without_extra_storage(monkeypatch, rows):
    import torch

    monkeypatch.setattr(owned_experts, "_GROUPED_DOWN", "grouped-v1")
    part = ExpertPartition(8, 0, 512, 640, 4)
    selection = gguf_ops._GroupedGateUp(rows)
    control = owned_experts.Plan(part, 2560, 11, 32, 256, True)
    candidate = owned_experts.Plan(part, 2560, 11, 32, 256, True, _gate_up=selection)
    assert candidate.identity == control.identity + ("grouped-gate-up", rows)
    expected = candidate.resources(32)
    assert expected.pop("grouped_gate_up_rows") == rows
    assert expected == control.resources(32)
    b = NS(rows=3, prefill=False, moe=NS(act=torch.empty((3, 11, 160), dtype=torch.float16),
           y=torch.empty((3, 11, 2560), dtype=torch.float32), pick=torch.empty((3, 11), dtype=torch.int64)))
    arena = candidate.arena(b)
    calls = []
    bank = NS(_owned_expert_plan=candidate, implementation=owned_experts.POLICY,
              forward_partitioned_native_candidate=lambda *a, **kw: calls.append(kw))
    arena.execute(bank, torch.empty((3, 2560), dtype=torch.float16), b.moe.pick, b.moe.y)
    assert calls[0]["_gate_up"] is selection
    assert arena.activation is b.moe.act and arena.output is b.moe.y
    assert arena.grouping.numel() == gguf_ops.grouped_workspace_words(33, 256)
    monkeypatch.setattr(owned_experts, "_GROUPED_DOWN", "inherit")
    with pytest.raises(ValueError, match="prepared grouped down"):
        owned_experts.Plan(part, 2560, 11, 32, 256, True, _gate_up=selection)


def test_family_boundary_carries_the_private_selector(monkeypatch):
    import torch
    from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFExperts

    selection = gguf_ops._GroupedGateUp(2)
    calls = []
    x = torch.zeros((1, 2560), dtype=torch.float16)
    gate = NS(matmul=lambda *a, **kw: torch.zeros((1, 80)))
    down = NS(matmul=lambda *a, **kw: torch.zeros((1, 2560)))
    bank = NS(partition=NS(etp=4, width=160, world=8, first=0, experts=512),
              shared=(gate, gate, down), gate=object(), up=object(), down=object(), width=160)
    monkeypatch.setattr(gguf_ops, "experts_owned", lambda *a, **kw: calls.append(kw))
    GGUFExperts.forward_partitioned_native_candidate(bank, x, torch.zeros((1, 11)),
        local=object(), activation=object(), out=object(), grouping=object(), _gate_up=selection)
    assert calls[0]["_gate_up"] is selection
