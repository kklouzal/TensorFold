"""Expert ownership and bounded native-kernel plans; no floating-point cross-rank reduction."""
from __future__ import annotations

from dataclasses import dataclass

MAX_ROWS = 128
MAX_SLOTS = 1024  # Native grouped plan EMAX=1024 includes the shared expert.


def expert_tensor(name):
    return any(part in name for part in (".switch_mlp.", ".shared_expert.", ".experts."))


@dataclass(frozen=True)
class Ownership:
    counts: tuple[int, ...]
    rank: int

    def __post_init__(self):
        if (not 1 <= len(self.counts) <= 4 or type(self.rank) is not int
                or not 0 <= self.rank < len(self.counts)
                or any(type(n) is not int or n <= 0 for n in self.counts)
                or sum(self.counts) > 1023):
            raise ValueError("expert shards require 1..4 positive counts, at most 1023 experts, and a valid rank")

    @property
    def begin(self):
        return sum(self.counts[:self.rank])

    @property
    def end(self):
        return self.begin + self.counts[self.rank]

    @property
    def ids(self):
        return range(self.begin, self.end)

    def local(self, picks):
        """A disjoint owner per routed ID; rank zero also owns the shared expert (global E)."""
        import torch
        if bool(((picks < 0) | (picks > sum(self.counts))).any()):
            raise ValueError("expert wire contains an invalid global expert ID")
        valid = (picks >= self.begin) & (picks < self.end)
        result = torch.where(valid, picks - self.begin, -1)
        if self.rank == 0:
            result = torch.where(picks == sum(self.counts), self.counts[0], result)
        return result


def ownership(ranks, shards, rank, experts):
    if type(ranks) is not int or not 1 <= ranks <= 4:
        raise ValueError("--expert-ranks must be 1 through 4")
    if ranks == 1 and shards is None:
        counts = (experts,)
    else:
        try:
            if isinstance(shards, str):
                # Native EMAX bounds each positive count to four digits and
                # world<=4 bounds the entire comma-separated wire grammar.
                if len(shards) > 19:
                    raise ValueError("expert count grammar exceeds native plan bounds")
                counts = tuple(int(part) for part in shards.split(","))
            else:
                if type(shards) not in (tuple, list) or len(shards) > 4:
                    raise ValueError("expert counts require a bounded list")
                counts = tuple(shards)
        except (ValueError, TypeError):
            raise ValueError("--expert-shards requires comma-separated expert counts") from None
    owner = Ownership(counts, rank)
    if len(counts) != ranks or sum(counts) != experts:
        raise ValueError("--expert-shards must name each rank and cover every routed expert exactly")
    return owner


def plan_owned(picks, experts, *, prefill=False, tile=None):
    """Build an exact bounded native plan, excluding -1 before native histogram indexing.

    Members retain their original row/slot positions and expert IDs are local.
    CPU routing is explicit here: this initial EP path does not claim device-only
    routing or graph capture. No unowned expert is evaluated.
    """
    import torch

    from . import experts as grouped

    if (picks.dtype != torch.int32 or picks.ndim != 2 or not picks.is_contiguous()
            or not 1 <= picks.shape[0] <= MAX_ROWS or not 1 <= picks.shape[1] <= MAX_SLOTS
            or type(experts) is not int or not 1 <= experts <= 1024):
        raise ValueError("expert work exceeds the typed row/native-expert plan envelope")
    values = picks.cpu().reshape(-1).tolist()
    plan = grouped.Plan(*picks.shape, experts, picks.device, prefill=prefill)
    plan.tile = (grouped.PREFILL_TILE if prefill else grouped.TILE) if tile is None else tile
    if plan.tile not in (grouped.TILE, grouped.PREFILL_TILE):
        raise ValueError("unsupported native expert tile")
    flat, items = owned_items(values, experts, plan.tile)
    if items:
        plan.items[:len(items)].copy_(torch.tensor(items, dtype=torch.int32, device=picks.device))
        plan.members[:len(flat)].copy_(torch.tensor(flat, dtype=torch.int32, device=picks.device))
    plan.counts.copy_(torch.tensor([len(items), len({item[0] for item in items})], dtype=torch.int32,
                                  device=picks.device))
    return plan


def owned_items(values, experts, tile):
    """Stable native items from a bounded local-ID list; -1 is excluded before native address calculation."""
    if (type(values) is not list or len(values) > MAX_ROWS * MAX_SLOTS
            or type(experts) is not int or not 1 <= experts <= 1024 or tile not in (16, 64)):
        raise ValueError("expert member list exceeds native plan bounds")
    members = [[] for _ in range(experts)]
    for pair, expert in enumerate(values):
        if type(expert) is not int or expert < -1 or expert >= experts:
            raise ValueError("expert work contains an unowned local expert ID")
        if expert != -1:
            members[expert].append(pair)
    flat, items = [], []
    for expert, pairs in enumerate(members):
        for start in range(0, len(pairs), tile):
            part = pairs[start:start + tile]
            items.append((expert, len(flat), len(part)))
            flat.extend(part)
    return flat, items


class AffineBank:
    """Local compact authority/cache; output slots follow the caller's original pair order."""
    def __init__(self, cached):
        self.cached = cached
        self.count, self.width, self.dims = cached.count, cached.width, cached.dims

    def device_tensors(self):
        return self.cached.device_tensors()

    def run_picks(self, x, picks, out, *, prefill):
        import torch

        from tensorfold.families.qwen4_exp.cuda.forward import _ExpertWindow

        from . import experts as grouped

        rows, slots = picks.shape
        plan = plan_owned(picks, self.count, prefill=prefill,
                          tile=grouped.window_tile(x.device) if prefill else grouped.TILE)
        scratch = _ExpertWindow(plan)
        items = scratch.read(plan)
        act = torch.empty((rows * slots, self.width), dtype=torch.bfloat16, device=x.device)
        out.zero_()
        start = 0
        while start < len(items):
            end, ids = start, []
            while end < len(items):
                expert = int(items[end, 0])
                if expert not in ids:
                    if len(ids) == self.cached.capacity:
                        break
                    ids.append(expert)
                end += 1
            with self.cached.lease(ids) as (hot, mapping):
                window = scratch.stage(items[start:end], mapping, self.cached.capacity)
                grouped.gate_up(x, hot, window, act, rows)
                grouped.down(act, hot, window, out.view(rows * slots, self.dims), rows)
            start = end
        # Caller retains this borrowed scratch until current-stream completion,
        # including pinned plan-upload sources and their recorded events.
        return plan, scratch, act
