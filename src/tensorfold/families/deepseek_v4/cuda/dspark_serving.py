"""Resident owner for the opt-in, single-live-request DSpark verify shape."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DraftFlight:
    stream: object
    slot: int
    anchor: int
    output: object
    done: object


def load_weights(root, rank, world, device):
    from .dspark_weights import Layout, load

    folder = Path(root)
    return load(folder / "ft", layout=Layout(rank, world, 4), device=device, config_dir=folder / "official")


def state_bytes(slots, geometry):
    from .dspark_head import RING
    from .dspark_weights import HEAD_DIM, TAPS
    from .kernels.kv_pack import ROTOR_ROW
    from .speculation import Speculation

    c2 = sum(geometry.compress_ratios[layer] == 2 for layer in geometry.kv_sources)
    return (Speculation.state_bytes(slots - 1, geometry.layers, c2, ROTOR_ROW)
            + slots * len(TAPS) * RING * HEAD_DIM * 2)


def decide(choices, anchor, proposed, room, eos):
    from tensorfold.engine.draft_acceptance import accepted_prefix

    if len(choices) != len(proposed) + 1 or room <= 0:
        raise ValueError("invalid DSpark verification result geometry")

    def reachable():
        for token, error in choices:
            if error is not None:
                raise error
            yield token

    return accepted_prefix(reachable(), [anchor, *proposed], room, eos)


class DraftOwner:
    """Exactly one unresolved C1 verify block, ordered on the serving stream.

    Each rank records the same pending slot after launch. Commit uses the
    consumed verify inputs, including the anchor but excluding the final output
    token. Cancellation before consumption commits zero. Computed draft-ring
    future positions are beyond the next anchor and within a nonaliasing
    guard region, so serial fallback or later verify may overwrite them.
    """

    def __init__(self, engine, weight, depth):
        import torch
        from .dspark_graph import SpecShape
        from .dspark_head import DSparkHead
        from .speculation import Speculation

        self.depth = depth
        self.target = engine.target
        self.pending = None
        device = self.target.device
        self.head = DSparkHead(weight, slots=engine.slots, comm=engine.coll, device=device,
                               vocab_first=weight.layout.rows(engine.vocab).start, q8_experts=False)
        self.spec = Speculation(self.target)
        self.shape = SpecShape(engine, self.head, self.spec, 1, depth, engine.context)
        self._slot = torch.full((1,), engine.slots - 1, dtype=torch.int32, device=device)
        self._count = torch.full((1,), -1, dtype=torch.int64, device=device)
        # Prime the aligned serving commit operands separately from the
        # staged row-view ABI used by the verify shape's warm-up.
        self.spec.commit(self._slot, self._count)
        self.shape.capture(torch.cuda.graph_pool_handle(), torch.cuda.Stream())

    def verify(self, slot, position, anchor):
        from .dspark_graph import VerifyEntry

        if self.pending is not None:
            raise RuntimeError("DSpark verify has an unresolved previous block")
        result = self.shape.run((VerifyEntry(slot, position, anchor),))
        self.pending = slot
        return result

    def commit(self, slot, count):
        import torch

        if self.pending != slot or not 0 <= count <= self.depth + 1:
            raise RuntimeError("DSpark commit has an invalid owner or consumed count")
        self._slot.fill_(slot)
        self._count.fill_(count)
        self.spec.commit(self._slot, self._count)
        # Observe the device's sticky history and compressor contract failures
        # before permitting another consumer or reuse of the slot.
        torch._assert_async(self.target.hash_state.err == 0, "invalid DSpark Engram verify state")
        for state in self.target.c2.values():
            torch._assert_async(state.err == 0, "invalid DSpark compressor verify state")
        self.pending = None

    def discard(self, slot):
        if self.pending == slot:
            self.commit(slot, 0)
