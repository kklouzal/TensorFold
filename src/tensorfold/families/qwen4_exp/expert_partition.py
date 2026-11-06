"""Immutable routed-expert placement inside one existing tensor-parallel stage."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ExpertPartition:
    world: int
    rank: int
    experts: int
    full_width: int
    etp: int | None = None

    def __post_init__(self):
        if type(self.world) is not int or self.world not in (1, 2, 4, 8):
            raise ValueError("expert placement requires an existing TP1/2/4/8 stage")
        if type(self.rank) is not int or not 0 <= self.rank < self.world:
            raise ValueError("expert placement rank is outside its TP stage")
        if self.etp is not None and (self.world != 8 or type(self.etp) is not int or self.etp not in (2, 4, 8)):
            raise ValueError("explicit expert tensor parallelism requires TP8 and ETP8/4/2")
        if self.etp is None:
            object.__setattr__(self, "etp", self.world)  # Preserve existing smaller-world reference profiles.
        if (type(self.experts) is not int or self.experts <= 0 or self.experts % self.groups
                or type(self.full_width) is not int or self.full_width <= 0 or self.full_width % self.etp):
            raise ValueError("expert count and intermediate width must divide their prepared EP/ETP map")

    @property
    def groups(self):
        return self.world // self.etp

    @property
    def group(self):
        return self.rank // self.etp

    @property
    def lane(self):
        return self.rank % self.etp

    @property
    def count(self):
        return self.experts // self.groups

    @property
    def first(self):
        return self.group * self.count

    @property
    def stop(self):
        return self.first + self.count

    @property
    def width(self):
        return self.full_width // self.etp

    @property
    def column(self):
        return self.lane * self.width

    def identity(self):
        return {"tp": self.world, "pp": 1, "etp": self.etp, "ep": self.groups,
                "rank": self.rank, "expert_first": self.first, "expert_count": self.count,
                "intermediate_first": self.column, "intermediate_width": self.width,
                "mapping": "contiguous-expert-groups-contiguous-width-lanes-v1"}


    @classmethod
    def bound(cls, identity, *, world, rank, experts, full_width):
        if type(identity) is not dict or "etp" not in identity:
            raise ValueError("complete immutable expert placement identity required")
        result = cls(world, rank, experts, full_width, identity["etp"])
        expected = result.identity()
        if identity.keys() != expected.keys() or any(type(identity[k]) is not type(v) or identity[k] != v
                                                    for k, v in expected.items()):
            raise ValueError("expert placement identity differs from actual rank/model geometry")
        return result

    def routing_workspace(self, rows, slots):
        """Six int64 index spans and four byte masks bound owned/shared remapping.

        Matrix temporaries retain the existing decoded-pair/shared GEMM bounds;
        caller output remains independently charged. Empty groups keep the cap.
        """
        if type(rows) is not int or rows <= 0 or type(slots) is not int or slots <= 0:
            raise ValueError("expert routing workspace requires positive row/slot bounds")
        return rows * slots * (6 * 8 + 4) if self.groups > 1 else 0


    def agreement(self):
        """Rank-independent startup facts, including default/explicit ETP8 parity."""
        return self.etp, self.experts, self.full_width
