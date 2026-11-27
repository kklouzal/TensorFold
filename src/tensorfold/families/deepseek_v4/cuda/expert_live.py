"""Apply one expert placement at a time on the quiescent TP8 cohort."""

from dataclasses import dataclass
from itertools import chain

from .expert_rebalance_cuda import PreparedMovement, prepare_movement
from .expert_rebalance_wire import (
    PeerRefusal,
    Phase,
    Receipt,
    Refusal,
    Transaction,
    encode_receipt,
    placement_digest,
    placement_from_ids,
    validate_receipts,
)
from .expert_transfer import ExpertTransferPlan, expert_groups_changed, local_expert_layout, plan_expert_transfers


@dataclass(frozen=True, slots=True)
class _Prepared:
    transaction: Transaction
    plan: ExpertTransferPlan
    movement: PreparedMovement | None


class LiveExpertPlacement:
    def __init__(self, engine, *, epoch, placement, operation):
        self.engine = engine
        self.operation = operation
        self.epoch = epoch
        self.placement = placement
        self.digest = placement_digest(placement, etp=engine.expert_tensor_parallel_size)
        self.nonce = 0
        self.prepared = None

    def _round_trip(self, phase, transaction, ids=()):
        engine = self.engine
        local = engine.announce(self.operation, int(phase), transaction.current_epoch,
                                transaction.target_epoch, transaction.nonce, ids)
        return validate_receipts((local, *engine.pipes.acknowledgments()), phase=phase, transaction=transaction)

    def apply(self, plan, *, expected_epoch, history):
        """Leader only. The scheduler has stopped admitting work and drained all borrowers."""
        engine = self.engine
        if engine.rank != 0 or self.prepared is not None:
            raise RuntimeError("only the idle placement leader may start a transfer")
        if not history.validate_live_placement(plan, expected_epoch=expected_epoch):
            return False
        if expected_epoch != self.epoch or plan.current != self.placement:
            raise RuntimeError("live expert placement disagrees with the durable epoch")

        transaction = Transaction(plan.etp, self.epoch, self.epoch + 1, self.nonce + 1, self.digest,
                                  placement_digest(plan.proposed, etp=plan.etp))
        try:
            try:
                self._round_trip(Phase.READY, transaction, tuple(chain.from_iterable(plan.proposed)))
            except PeerRefusal:
                self._round_trip(Phase.ABORTED, transaction)
                raise
            engine.decoder.retire_placement_prefixes()
            self._round_trip(Phase.APPLIED, transaction)
            committed = history.commit_live_placement(plan, expected_epoch=expected_epoch)
            if committed != self.epoch:
                raise RuntimeError("durable expert placement disagrees with the applied epoch")
        except PeerRefusal:
            raise
        except BaseException as error:
            # A protocol error or a failed submission can leave the cohort partially mutated. Admission may
            # not resume on this process, and its live owners stay retained until the supervisor retires it.
            engine.decoder.broken = error
            raise
        return True

    def handle(self, phase_code, current_epoch, target_epoch, nonce, ids):
        """Execute one leader command on the rank's current stream and return its stamped receipt."""
        phase = Phase(phase_code)
        if phase == Phase.READY:
            return self._prepare(current_epoch, target_epoch, nonce, ids)
        pending = self.prepared
        if (ids or pending is None or (current_epoch, target_epoch, nonce) != (
                pending.transaction.current_epoch, pending.transaction.target_epoch, pending.transaction.nonce)):
            raise RuntimeError("expert placement command does not match a prepared transfer")
        if phase == Phase.ABORTED:
            self.prepared = None
        else:
            self._apply_prepared(pending)
        return encode_receipt(Receipt(phase, self.engine.rank, pending.transaction))

    def _prepare(self, current_epoch, target_epoch, nonce, ids):
        import torch

        engine = self.engine
        if self.prepared is not None or current_epoch != self.epoch or nonce <= self.nonce or engine.part is not None:
            raise RuntimeError("expert placement prepare found stale or active rank work")
        etp = engine.expert_tensor_parallel_size
        proposed = placement_from_ids(ids)
        plan = plan_expert_transfers(self.placement, proposed, etp=etp)
        transaction = Transaction(etp, current_epoch, target_epoch, nonce, self.digest,
                                  placement_digest(proposed, etp=etp))
        torch.cuda.current_stream(engine.comm.device).synchronize()
        self.nonce = nonce
        try:
            movement = prepare_movement(engine.w, engine.target.lattice, engine.comm, plan, current=self.placement)
        except ValueError:
            self.prepared = _Prepared(transaction, plan, None)
            return encode_receipt(Receipt(Phase.READY, engine.rank, transaction, Refusal.PREFLIGHT))
        self.prepared = _Prepared(transaction, plan, movement)
        return encode_receipt(Receipt(Phase.READY, engine.rank, transaction, carrier_layout=movement.carrier_layout))

    def _apply_prepared(self, pending):
        import torch

        engine = self.engine
        if pending.movement is None:
            raise RuntimeError("expert placement apply requires a ready transfer")
        engine.prefix.retire_placement_prefixes()
        # Release the previous captures only after the rank has fenced their readers in READY.
        engine.graphs = None
        pending.movement.enqueue()
        torch.cuda.current_stream(engine.comm.device).synchronize()
        engine.comm.check()
        for layer, row in zip(engine.w.layers, pending.plan.proposed, strict=True):
            layer.expert_groups_changed = expert_groups_changed(row, group_slots=pending.plan.group_slots)
            layout = local_expert_layout(row, owner=engine.w.group, group_slots=pending.plan.group_slots)
            if layout is None:
                layer.expert_to_local = None
                continue
            _, local = layout
            updated = torch.tensor(local, dtype=torch.int32, device=engine.comm.device)
            if layer.expert_to_local is None:
                layer.expert_to_local = updated
            else:
                layer.expert_to_local.copy_(updated)
        # All ranks must finish paired P2P work before entering the capture collectives.
        engine.comm.barrier()
        engine.recapture_expert_graphs()
        torch.cuda.current_stream(engine.comm.device).synchronize()
        engine.comm.check()
        self.placement = pending.plan.proposed
        self.epoch = pending.transaction.target_epoch
        self.digest = pending.transaction.target_digest
        self.prepared = None
