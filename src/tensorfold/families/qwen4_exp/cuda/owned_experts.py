"""Private, exclusive TP8/ETP4 native qualification preparation.

Default serving retains reference execution. These arenas are execution scratch,
not sequence state; target/prompt/method owners coexist and execute serially.
"""
from dataclasses import dataclass

from ..expert_partition import ExpertPartition

POLICY = "owned-etp4-native-slots-v1"


@dataclass(frozen=True)
class Plan:
    partition: ExpertPartition
    hidden: int
    slots: int
    rows: int
    prefill_rows: int
    mtp: bool

    def __post_init__(self):
        p = self.partition
        if (type(p) is not ExpertPartition or p.world != 8 or p.etp != 4 or p.width != 160
                or type(self.mtp) is not bool):
            raise ValueError("owned native preparation requires exact TP8/ETP4 placement and method binding")
        if any(type(v) is not int or v <= 0 for v in (self.hidden, self.slots, self.rows, self.prefill_rows)):
            raise ValueError("owned native preparation requires positive exact geometry")
        if max(self.rows, self.prefill_rows) * self.slots > 65535:
            raise ValueError("owned native preparation exceeds its 65535-pair envelope")
        self.resources(self.rows)

    @property
    def identity(self):
        p = self.partition
        return (POLICY, p.world, p.etp, p.experts, p.full_width, self.hidden,
                self.slots, self.rows, self.prefill_rows, self.mtp)

    def resources(self, rows):
        if type(rows) is not int or not 0 < rows <= self.rows:
            raise ValueError("owned decode arena exceeds its prepared rows")
        local = 8 * self.slots * ((1 + int(self.mtp)) * rows + self.prefill_rows)
        reserve = (1 + int(self.mtp)) * self.partition.routing_workspace(
            max(rows, self.prefill_rows), self.slots)
        if local > reserve:
            raise MemoryError("owned arenas exceed the paid reference routing reservation")
        return {"policy": POLICY, "exclusive_provider": True, "local_arena_bytes": local,
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
        return Arena(self, torch.empty((rows, self.slots), dtype=torch.int64, device=b.act.device), b.act, b.y)

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
        if len(set(pointers)) != len(pointers) or actual != expected["local_arena_bytes"]:
            raise MemoryError("materialized owned arenas differ from their contained reservation")
        return {**expected, "materialized_local_arena_bytes": actual,
                "arena_rows": [b.rows for b in arenas]}


@dataclass(frozen=True)
class Arena:
    plan: Plan
    local: object
    activation: object
    output: object

    def execute(self, bank, x, picks, out):
        if (getattr(bank, "_owned_expert_plan", None) is not self.plan or bank.implementation != POLICY
                or out.data_ptr() != self.output.data_ptr() or x.shape[0] > self.local.shape[0]):
            raise ValueError("owned execution lacks its exclusive prepared bank/arena")
        rows = x.shape[0]
        return bank.forward_partitioned_native_candidate(x, picks,
            local=self.local[:rows], activation=self.activation[:rows].view(-1, 160), out=out)
