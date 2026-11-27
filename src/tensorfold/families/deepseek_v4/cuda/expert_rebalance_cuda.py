"""Prepare and enqueue held in-place expert carrier swaps on one CUDA stream."""
from dataclasses import dataclass

import torch

from .expert_transfer import ExpertTransferPlan, plan_expert_transfers


@dataclass(frozen=True, slots=True)
class _Move:
    banks: tuple[torch.Tensor, ...]
    slot: int
    other: int
    peer: int | None


def _byte_span(tensor, device):
    if (not isinstance(tensor, torch.Tensor) or tensor.device != device or device.type != "cuda"
            or tensor.dtype != torch.uint8 or not tensor.is_contiguous() or not tensor.numel()):
        raise ValueError("expert movement requires nonempty contiguous CUDA byte storage on the rank device")
    first = tensor.data_ptr()
    return first, first + tensor.numel()


class PreparedMovement:
    """A one-shot submission owned by the quiescent CUDA actor.

    Preparing moves does not write any tensor. Before submission, the caller
    must obtain all peers' matching readiness and retire every workspace and
    weight reader. It must hold this object and its storage until the current
    stream completes, including on failure. Routing and epoch publication
    belong to the caller and must wait for confirmed completion on all ranks.
    """

    def __init__(self, moves, scratch, comm, carrier_layout):
        self._moves = moves
        self._scratch = scratch
        self._comm = comm
        self.carrier_layout = carrier_layout
        self._submitted = False

    @property
    def scratch_bytes(self):
        return self._scratch.numel()

    def enqueue(self):
        """Submit once in plan order. No synchronization or route-map writes."""
        if self._submitted:
            raise RuntimeError("expert movement has already been submitted")
        self._submitted = True
        with torch.inference_mode():
            for move in self._moves:
                source = [bank[move.slot] for bank in move.banks]
                if move.peer is None:
                    for bank, row in zip(move.banks, source, strict=True):
                        temp = self._scratch[:row.numel()].view_as(row)
                        temp.copy_(row)
                        row.copy_(bank[move.other])
                        bank[move.other].copy_(temp)
                    continue
                received = []
                offset = 0
                for row in source:
                    end = offset + row.numel()
                    received.append(self._scratch[offset:end].view_as(row))
                    offset = end
                self._comm.exchange(source, received, move.peer)
                for row, temp in zip(source, received, strict=True):
                    row.copy_(temp)


def prepare_movement(weights, workspace, comm, plan: ExpertTransferPlan, *, current):
    """Check the immutable plan and established workspace and bank spans without device writes."""
    if not isinstance(plan, ExpertTransferPlan) or current != plan.current:
        raise ValueError("expert movement requires a plan matching the active placement")
    rank, world, etp = comm.rank, comm.world, weights.expert_tensor_parallel_size
    if world != 8 or not 0 <= rank < world or rank != weights.rank or etp != plan.etp:
        raise ValueError("expert movement rank and topology must match the actual TP8 weights")
    if plan != plan_expert_transfers(current, plan.proposed, etp=etp):
        raise ValueError("expert movement schedule does not match the proposed placement")
    geometry = weights.geometry
    if (len(plan.current) != geometry.layers or len(weights.layers) != geometry.layers
            or plan.group_slots * (world // etp) != geometry.experts):
        raise ValueError("expert movement cannot change layer or expert geometry")
    scratch = workspace.banks["gu"]
    spans = [_byte_span(scratch, comm.device)]
    if scratch.ndim != 1:
        raise ValueError("expert movement requires a flat reusable lattice bank")
    layers, layout = [], []
    for index, layer in enumerate(weights.layers):
        banks = (layer.exp_gate, layer.exp_up, layer.exp_down)
        spans.extend(_byte_span(bank, comm.device) for bank in banks)
        if (layer.index != index or any(bank.ndim != 3 or bank.shape[0] != plan.group_slots for bank in banks)
                or banks[0].shape != banks[1].shape or banks[0].shape[1] != geometry.moe_inter // etp
                or banks[2].shape[1] != geometry.hidden):
            raise ValueError("expert movement carrier shapes disagree with fixed model geometry")
        layers.append(banks)
        # TP peers compare this receipt with the same lane before allowing mutation.
        layout.append((layer.gate_up_type, layer.down_type, layer.down_left,
                       tuple(tuple(bank.shape[1:]) for bank in banks)))
    spans.sort()
    if any(end > first for (_, end), (first, _) in zip(spans, spans[1:])):
        raise ValueError("expert movement weight banks and reusable workspace must be disjoint")

    owner, lane = divmod(rank, etp)
    moves = []
    scratch_bytes = 0
    for round_ in plan.rounds:
        for swap in round_:
            a, left = divmod(swap.left, plan.group_slots)
            b, right = divmod(swap.right, plan.group_slots)
            if owner not in (a, b):
                continue
            banks = layers[swap.layer]
            peer = None if a == b else (b if owner == a else a) * etp + lane
            slot, other = (left, right) if owner == a else (right, left)
            moves.append(_Move(banks, slot, other, peer))
            scratch_bytes = max(scratch_bytes, sum(bank.numel() // plan.group_slots for bank in banks))
    if scratch_bytes > scratch.numel():
        raise ValueError("expert movement cannot fit an expert in the reusable lattice bank")
    if any(move.peer is not None for move in moves) and not callable(getattr(comm, "exchange", None)):
        raise ValueError("cross-group expert movement requires native paired exchange")
    return PreparedMovement(tuple(moves), scratch[:scratch_bytes], comm, tuple(layout))
