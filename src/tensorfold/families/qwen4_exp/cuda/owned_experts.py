"""Private, exclusive TP8/ETP4 native qualification preparation.

Default serving retains reference execution. These arenas are execution scratch,
not sequence state; target/prompt/method owners coexist and execute serially.
"""
from dataclasses import dataclass
import os

from ..expert_partition import ExpertPartition

POLICY = "owned-etp4-native-slots-v1"
_GROUPED_DOWN = os.environ.get("TENSORFOLD_PRIVATE_SM70_GROUPED_DOWN", "inherit")
if _GROUPED_DOWN not in ("inherit", "grouped-v1"):
    raise ValueError("TENSORFOLD_PRIVATE_SM70_GROUPED_DOWN requires inherit or grouped-v1")


@dataclass(frozen=True)
class Plan:
    partition: ExpertPartition
    hidden: int
    slots: int
    rows: int
    prefill_rows: int
    mtp: bool
    _gate_up: object = None

    def __post_init__(self):
        p = self.partition
        if (type(p) is not ExpertPartition or p.world != 8 or p.etp != 4 or p.width != 160
                or type(self.mtp) is not bool):
            raise ValueError("owned native preparation requires exact TP8/ETP4 placement and method binding")
        if any(type(v) is not int or v <= 0 for v in (self.hidden, self.slots, self.rows, self.prefill_rows)):
            raise ValueError("owned native preparation requires positive exact geometry")
        if max(self.rows, self.prefill_rows) * self.slots > 65535:
            raise ValueError("owned native preparation exceeds its 65535-pair envelope")
        if _GROUPED_DOWN == "grouped-v1" and (self.hidden != 2560 or p.experts != 512):
            raise ValueError("grouped owned down160 requires the admitted SC117 hidden/global expert geometry")
        if self._gate_up is not None:
            from tensorfold.cuda.sm70.gguf_ops import _GroupedGateUp

            if type(self._gate_up) is not _GroupedGateUp or _GROUPED_DOWN != "grouped-v1":
                raise ValueError("private grouped gate/up requires its typed selector and prepared grouped down")
        self.resources(self.rows)

    @property
    def identity(self):
        p = self.partition
        original = (POLICY, p.world, p.etp, p.experts, p.full_width, self.hidden,
                    self.slots, self.rows, self.prefill_rows, self.mtp)
        identity = original if _GROUPED_DOWN == "inherit" else original + (_GROUPED_DOWN,)
        return identity if self._gate_up is None else identity + ("grouped-gate-up", self._gate_up.rows)

    def resources(self, rows):
        if type(rows) is not int or not 0 < rows <= self.rows:
            raise ValueError("owned decode arena exceeds its prepared rows")
        local = 8 * self.slots * ((1 + int(self.mtp)) * rows + self.prefill_rows)
        reserve = (1 + int(self.mtp)) * self.partition.routing_workspace(
            max(rows, self.prefill_rows), self.slots)
        grouped = 0
        if _GROUPED_DOWN == "grouped-v1":
            from tensorfold.cuda.sm70.gguf_ops import grouped_workspace_words

            grouped = 4 * ((1 + int(self.mtp)) * grouped_workspace_words(rows * self.slots, self.partition.count)
                           + grouped_workspace_words(self.prefill_rows * self.slots, self.partition.count))
        if local + grouped > reserve:
            raise MemoryError("owned arenas exceed the paid reference routing reservation")
        experiment = {} if self._gate_up is None else {"grouped_gate_up_rows": self._gate_up.rows}
        return {**experiment, "policy": POLICY, "exclusive_provider": True, "local_arena_bytes": local,
                "grouped_down": _GROUPED_DOWN, "grouping_arena_bytes": grouped,
                "contained_routing_reserve_bytes": reserve, "additional_device_bytes": 0,
                "activation_storage": "existing-moe-act", "output_storage": "existing-moe-y",
                "qualification": "private-unqualified"}

    def bind(self, w):
        from .gguf_weights import GGUFExperts

        c = w.cfg
        if (w.meta.get("expert_partition") != self.partition.identity()
                or c.hidden != self.hidden or c.top_k + 1 != self.slots or c.moe_width != 160
                or c.shared_width != 80 or (w.mtp is not None) != self.mtp):
            raise ValueError("owned preparation differs from actual target/method geometry")
        banks = [layer.moe.experts for layer in w.layers]
        if w.mtp is not None:
            banks.append(w.mtp.layer.moe.experts)
        for bank in banks:
            if (type(bank) is not GGUFExperts or bank.partition != self.partition
                    or bank.implementation != "reference" or bank.dims != self.hidden
                    or any(part.n != 80 for part in bank.shared[:2]) or bank.shared[2].k != 80):
                raise ValueError("owned preparation requires matching resident target/MTP banks")
        # Validate all banks before publishing any exclusive provider binding.
        for bank in banks:
            bank.implementation = POLICY
            bank._owned_expert_plan = self
        w.meta["owned_expert_plan"] = self

    def arena(self, buffers):
        import torch

        rows = buffers.rows
        if buffers.prefill:
            if rows != self.prefill_rows:
                raise ValueError("owned prompt arena differs from its prepared rows")
        elif not 0 < rows <= self.rows:
            raise ValueError("owned decode arena differs from its prepared rows")
        b = buffers.moe
        if (b.act.shape != (rows, self.slots, 160) or b.act.dtype != torch.float16
                or not b.act.is_contiguous() or b.y.shape != (rows, self.slots, self.hidden)
                or b.y.dtype not in (torch.float16, torch.float32) or not b.y.is_contiguous()
                or b.pick.shape != (rows, self.slots) or b.pick.device != b.act.device
                or b.y.device != b.act.device):
            raise ValueError("owned arena cannot reuse the actual MoE storage")
        grouping = None
        if _GROUPED_DOWN == "grouped-v1":
            from tensorfold.cuda.sm70.gguf_ops import grouped_workspace_words

            grouping = torch.empty(grouped_workspace_words(rows * self.slots, self.partition.count),
                                   dtype=torch.int32, device=b.act.device)
        return Arena(self, torch.empty((rows, self.slots), dtype=torch.int64, device=b.act.device), b.act, b.y, grouping)

    def reconcile(self, decoder):
        arenas = [decoder.buf, decoder.pbuf] + ([decoder.mbuf] if self.mtp else [])
        rows = decoder.buf.rows
        if self.mtp and decoder.mbuf.rows != rows:
            raise ValueError("owned method and target arena extents differ")
        expected = self.resources(rows)
        owners = [b.owned_experts for b in arenas]
        if any(a is None or a.plan is not self for a in owners):
            raise ValueError("owned execution arena lacks its prepared owner")
        pointers = [a.local.data_ptr() for a in owners]
        actual = sum(a.local.numel() * a.local.element_size() for a in owners)
        grouping_bytes = sum(0 if a.grouping is None else a.grouping.numel() * a.grouping.element_size() for a in owners)
        grouping_pointers = [a.grouping.data_ptr() for a in owners if a.grouping is not None]
        if len(set(grouping_pointers)) != len(grouping_pointers) or grouping_bytes != expected["grouping_arena_bytes"]:
            raise MemoryError("materialized grouped routing arenas differ from their contained reservation")
        if len(set(pointers)) != len(pointers) or actual != expected["local_arena_bytes"]:
            raise MemoryError("materialized owned arenas differ from their contained reservation")
        return {**expected, "materialized_local_arena_bytes": actual, "materialized_grouping_arena_bytes": grouping_bytes,
                "arena_rows": [b.rows for b in arenas]}


@dataclass(frozen=True)
class Arena:
    plan: Plan
    local: object
    activation: object
    output: object
    grouping: object = None

    def execute(self, bank, x, picks, out):
        if (getattr(bank, "_owned_expert_plan", None) is not self.plan or bank.implementation != POLICY
                or out.data_ptr() != self.output.data_ptr() or x.shape[0] > self.local.shape[0]):
            raise ValueError("owned execution lacks its exclusive prepared bank/arena")
        rows = x.shape[0]
        if self.grouping is None:
            return bank.forward_partitioned_native_candidate(x, picks,
                local=self.local[:rows], activation=self.activation[:rows].view(-1, 160), out=out)
        from tensorfold.cuda.sm70.gguf_ops import grouped_workspace_words

        words = grouped_workspace_words(rows * picks.shape[1], self.plan.partition.count)
        return bank.forward_partitioned_native_candidate(x, picks,
            local=self.local[:rows], activation=self.activation[:rows].view(-1, 160), out=out,
            grouping=self.grouping[:words], _gate_up=self.plan._gate_up)
