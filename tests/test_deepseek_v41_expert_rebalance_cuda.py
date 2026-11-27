"""Host-only movement fixtures. Native byte movement needs an independent TP8 run."""
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest


class _Bytes:
    """A CPU tensor view with fake CUDA metadata for the host control tests."""
    device = SimpleNamespace(type="cuda")
    dtype = object()

    def __init__(self, value):
        self.value = value

    @property
    def shape(self):
        return self.value.shape

    @property
    def ndim(self):
        return self.value.ndim

    def is_contiguous(self):
        return self.value.is_contiguous()

    def numel(self):
        return self.value.numel()

    def data_ptr(self):
        return self.value.data_ptr()

    def __getitem__(self, key):
        return _Bytes(self.value[key])

    def view_as(self, other):
        return _Bytes(self.value.view_as(other.value))

    def copy_(self, other):
        self.value.copy_(other.value)


def _fixture(monkeypatch, rank=0):
    import torch
    from tensorfold.families.deepseek_v4.cuda import expert_rebalance_cuda as move

    monkeypatch.setattr(move, "torch", SimpleNamespace(Tensor=_Bytes, uint8=_Bytes.dtype,
                                                     inference_mode=nullcontext))
    def bank(first, rows):
        tensor = torch.empty((2, rows, 3), dtype=torch.uint8)
        tensor[0].fill_(first)
        tensor[1].fill_(first + 1)
        return _Bytes(tensor)

    layer = SimpleNamespace(index=0, exp_gate=bank(10, 2), exp_up=bank(20, 2), exp_down=bank(30, 4),
                            gate_up_type=16, down_type=29, down_left=(rank % 2) * 2)
    weights=SimpleNamespace(rank=rank, expert_tensor_parallel_size=2, layers=[layer],
                            geometry=SimpleNamespace(layers=1, experts=8, hidden=4, moe_inter=4))
    workspace=SimpleNamespace(banks={"gu": _Bytes(torch.zeros((24,), dtype=torch.uint8))})
    comm=SimpleNamespace(rank=rank, world=8, device=_Bytes.device)
    return move, weights, workspace, comm


def test_local_movement_keeps_storage_and_does_not_reapply(monkeypatch):
    from tensorfold.families.deepseek_v4.cuda.expert_transfer import plan_expert_transfers

    move, weights, workspace, comm = _fixture(monkeypatch)
    current = (tuple(range(8)),)
    plan = plan_expert_transfers(current, ((1, 0, *range(2, 8)),), etp=2)
    movement = move.prepare_movement(weights, workspace, comm, plan, current=current)
    layer = weights.layers[0]
    banks = layer.exp_gate, layer.exp_up, layer.exp_down
    pointers = [bank.data_ptr() for bank in banks]
    expected = [bank.value.flip([0]) for bank in banks]
    assert movement.scratch_bytes == 24
    assert movement.carrier_layout == ((16, 29, 0, ((2, 3), (2, 3), (4, 3))),)
    movement.enqueue()
    assert [bank.data_ptr() for bank in banks] == pointers
    assert all(bank.value.equal(value) for bank, value in zip(banks, expected, strict=True))
    with pytest.raises(RuntimeError, match="already been submitted"):
        movement.enqueue()


@pytest.mark.parametrize("lane", (0, 1))
def test_remote_movement_pairs_identical_lanes_and_disjoint_receives(monkeypatch, lane):
    from tensorfold.families.deepseek_v4.cuda.expert_transfer import plan_expert_transfers

    move, weights, workspace, comm = _fixture(monkeypatch, rank=lane)
    current = (tuple(range(8)),)
    plan = plan_expert_transfers(current, ((4, 1, 2, 3, 0, 5, 6, 7),), etp=2)
    calls = []
    def exchange(sends, recvs, peer):
        assert peer == 4 + lane
        assert [row.value.flatten()[0].item() for row in sends] == [10, 20, 30]
        assert [row.numel() for row in recvs] == [6, 6, 12]
        base = workspace.banks["gu"].data_ptr()
        assert [row.data_ptr() for row in recvs] == [base, base + 6, base + 12]
        for row, value in zip(recvs, (60, 70, 80), strict=True):
            row.value.fill_(value)
        calls.append(peer)

    comm.exchange = exchange
    movement = move.prepare_movement(weights, workspace, comm, plan, current=current)
    movement.enqueue()
    bank = weights.layers[0].exp_down.value
    assert calls == [4 + lane] and (bank[0] == 80).all() and (bank[1] == 31).all()


def test_preflight_rejects_stale_incoherent_and_borrowed_storage(monkeypatch):
    from tensorfold.families.deepseek_v4.cuda.expert_transfer import plan_expert_transfers

    move, weights, workspace, comm = _fixture(monkeypatch)
    current = (tuple(range(8)),)
    plan = plan_expert_transfers(current, ((1, 0, *range(2, 8)),), etp=2)
    with pytest.raises(ValueError, match="active placement"):
        move.prepare_movement(weights, workspace, comm, plan, current=plan.proposed)
    with pytest.raises(ValueError, match="schedule"):
        move.prepare_movement(weights, workspace, comm, replace(plan, rounds=()), current=current)
    workspace.banks["gu"] = workspace.banks["gu"][:23]
    with pytest.raises(ValueError, match="cannot fit"):
        move.prepare_movement(weights, workspace, comm, plan, current=current)
    workspace.banks["gu"] = _Bytes(weights.layers[0].exp_down.value.flatten())
    with pytest.raises(ValueError, match="must be disjoint"):
        move.prepare_movement(weights, workspace, comm, plan, current=current)
    assert (weights.layers[0].exp_down.value[0] == 30).all()
