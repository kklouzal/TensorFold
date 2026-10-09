"""Gated delta rule for trees and replays: each runs the serial step, so a row's bits never depend on its window."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from operator import index
from typing import Sequence

import torch


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_gdn_v2", sources=[str(here / "gdn.cpp"), str(here / "gdn.cu"),
                                                   str(here / "gdn_prefill.cu")],
                extra_cuda_cflags=["-O3", "--fmad=false"], verbose=False)


def _order_slots(parents: Sequence[int], order: list[int]) -> tuple[list[int], int]:
    """Plan entries (node, source, destination) for one visiting order, and the slots it needs."""

    pos = {node: i for i, node in enumerate(order)}
    children: dict[int, list[int]] = {}
    for node, parent in enumerate(parents):
        if parent >= 0:
            children.setdefault(parent, []).append(node)
    slot_of: dict[int, int] = {}
    free: list[int] = []
    used = 0
    entries: list[int] = []
    for i, node in enumerate(order):
        parent = parents[node]
        if parent < 0:
            source = -1
        elif i > 0 and order[i - 1] == parent:
            source = -2
        else:
            source = slot_of[parent]
        if parent >= 0 and parent in slot_of and pos[max(children[parent], key=pos.__getitem__)] == i:
            free.append(slot_of.pop(parent))        # its last child reads it now; the slot may be reused
        dest = -1
        if any(pos[c] != i + 1 for c in children.get(node, ())):
            if free:
                free.sort()
                dest = free.pop(0)
            else:
                dest, used = used, used + 1
            slot_of[node] = dest
        entries += [node, source, dest]
    return entries, used


def schedule(parents: Sequence[int]) -> tuple[list[int], int]:
    """(node, source, dest) entries and slot count for one tree, depth-first or level order, whichever needs fewer."""

    try:
        parents = [index(p) if not isinstance(p, bool) else None for p in parents]
    except TypeError as exc:
        raise ValueError("tree parents must be integers") from exc
    if any(p is None for p in parents):
        raise ValueError("tree parents must be integers")
    if not parents or parents[0] != -1 or any(not -1 <= p < i or (p == -1) != (i == 0) for i, p in enumerate(parents)):
        raise ValueError("a tree needs one root at row zero and parents before their children")
    children: dict[int, list[int]] = {}
    depth = [0] * len(parents)
    for node, parent in enumerate(parents[1:], start=1):
        children.setdefault(parent, []).append(node)
        depth[node] = depth[parent] + 1
    dfs, stack = [], [0]
    while stack:
        node = stack.pop()
        dfs.append(node)
        stack.extend(reversed(children.get(node, ())))
    level = sorted(range(len(parents)), key=lambda n: (depth[n], n))
    best = min((_order_slots(parents, o) for o in (dfs, level)), key=lambda e: e[1])
    if best[1] > 32:
        raise ValueError("this tree needs more than 32 live states")
    return best


@dataclass
class Plan:
    """A window's GDN schedule on the device: (W, 3) entries in stream order, stream row ranges, slot count."""

    entries: torch.Tensor
    starts: torch.Tensor
    slots: int
    max_rows: int


def plan_host(streams: Sequence[Sequence[int]]) -> tuple[list[int], list[int], int, int]:
    """Host half of ``plan`` for callers packing their own copy: entries in window rows, starts, slots and max rows."""

    if not streams or any(not parents for parents in streams):
        raise ValueError("a plan needs at least one stream and one row per stream")
    if sum(len(parents) for parents in streams) > 2**31 - 1:
        raise ValueError("plan row offsets must fit int32")
    entries, starts, slots = [], [0], 0
    for parents in streams:
        flat, need = schedule(parents)
        base = starts[-1]
        entries += [x + base if j % 3 == 0 else x for j, x in enumerate(flat)]
        starts.append(base + len(parents))
        slots = max(slots, need)
    maximum = max(b - a for a, b in zip(starts, starts[1:]))
    if slots and maximum > 1024:
        raise ValueError("a tree plan takes at most 1024 rows per stream")
    return entries, starts, slots, maximum


class _PointerValues(list[int]):
    """Pointer values retain their tensor owners until ``to_device`` transfers that ownership."""

    def __init__(self, values, owners):
        super().__init__(values)
        self.owners = tuple(owners)


def to_device(values: Sequence[int], dtype: torch.dtype, device) -> torch.Tensor:
    """Pinned copy on the current stream; pointer lists keep owners and register each consuming launch's stream.

    A consumer on another stream must wait for this copy and all pointee producers.
    Copying or mutating the resulting pointer tensor requires the raw ABI ownership contract.
    """

    if isinstance(values, _PointerValues) and dtype != torch.int64:
        raise ValueError("pointer tables require int64 storage")
    if isinstance(values, _PointerValues) and (len(values) != len(values.owners)
            or any(value != tensor.data_ptr() for value, tensor in zip(values, values.owners))):
        raise ValueError("pointer values and retained owners must remain unchanged before upload")
    out = torch.tensor(values, dtype=dtype).pin_memory().to(device, non_blocking=True)
    if isinstance(values, _PointerValues):
        if any(t.device != out.device for t in values.owners):
            raise ValueError("pointer table and pointees must be on the same CUDA device")
        out._tensorfold_gdn_owners = values.owners
    return out


def _record_pointers(table: torch.Tensor | None) -> tuple[torch.Tensor, ...]:
    owners = getattr(table, "_tensorfold_gdn_owners", ())
    if owners:
        stream = torch.cuda.current_stream(table.device)
        for tensor in owners:
            tensor.record_stream(stream)
    return owners


def _overlaps(write: torch.Tensor, read: torch.Tensor) -> bool:
    if not write.numel() or not read.numel():
        return False
    start, stop = write.data_ptr(), write.data_ptr() + write.numel() * write.element_size()
    begin = read.data_ptr()
    if read.is_contiguous():
        return begin < stop and begin + read.numel() * read.element_size() > start
    rows, width = read.shape[0], (1 if read.dim() == 1 else read.shape[1]) * read.element_size()
    stride = read.stride(0) * read.element_size()
    row = (start - begin - width) // stride + 1 if stride > 0 and start >= begin + width else 0
    return row < rows and begin + row * stride < stop and begin + row * stride + width > start


def _check_state_writes(writes: Sequence[torch.Tensor], reads: Sequence[torch.Tensor]) -> None:
    ranges = sorted((t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()) for t in writes)
    if any(end > following for (_, end), (following, _) in zip(ranges, ranges[1:])):
        raise ValueError("GDN state writes must be disjoint")
    if any(_overlaps(write, read) for write in writes for read in reads):
        raise ValueError("GDN state writes must not overlap row, schedule or read-only state inputs")


def plan(streams: Sequence[Sequence[int]], device) -> Plan:
    entries, starts, slots, rows = plan_host(streams)
    dev = to_device(entries + starts, torch.int32, device)
    w = starts[-1]
    return Plan(dev[:3 * w].view(w, 3), dev[3 * w:], slots, rows)


def tree(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, p: Plan,
         state: torch.Tensor | None = None, table: torch.Tensor | None = None,
         pending: Sequence[torch.Tensor] | None = None, final: torch.Tensor | None = None) -> torch.Tensor:
    """Outputs (W, Hv, Dv) bf16; pending writes require exclusive, disjoint committed states.

    Plans come from ``plan`` or an equivalent validated host schedule. Raw pointer
    tables and device row metadata follow ``docs/cuda-kernel-contracts.md``.
    """

    if (state is None) == (table is None):
        raise ValueError("pass one stream's state or a table of every stream's")
    if not isinstance(v, torch.Tensor) or v.dim() != 3:
        raise ValueError("GDN values must have (W, Hv, Dv) shape")
    owners = _record_pointers(table)
    finals = _record_pointers(final) if table is not None else ()
    if owners or finals:
        shape = (v.shape[1], v.shape[2], 128)
        if any(t.dtype != torch.float32 or t.shape != shape or t.device != q.device for t in (*owners, *finals)):
            raise ValueError("tree pointer tables contain same-device fp32 (Hv, Dv, 128) states")
        if owners and len(owners) != table.numel() or finals and len(finals) != table.numel():
            raise ValueError("one state pointer per stream")
        inputs = [q, k, v, g, beta, p.entries, p.starts, table]
        if pending is not None:
            _check_state_writes(owners, inputs + list(pending))
        if finals:
            _check_state_writes(finals, inputs + list(owners) + ([] if pending is None else list(pending)))
    return _ext().tree(q, k, v, g, beta, state, table, None if table is None else p.starts, p.entries, p.slots,
                       p.max_rows, None if pending is None else list(pending), final)


def pointers(tensors: Sequence[torch.Tensor]) -> list[int]:
    """Contiguous same-device tensor addresses; ``to_device`` preserves owners through consuming launches.

    A plain list/copy of these addresses instead requires callers to retain and
    record pointees until the consuming stream completes, and order producers.
    """

    tensors = tuple(tensors)
    if any(not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous() for t in tensors):
        raise ValueError("pointer tables take contiguous CUDA tensors")
    if tensors and any(t.device != tensors[0].device for t in tensors):
        raise ValueError("pointer table pointees must be on the same CUDA device")
    return _PointerValues([t.data_ptr() for t in tensors], tensors)


def pointer_tables(groups: Sequence[Sequence[torch.Tensor]], device) -> list[torch.Tensor]:
    """One packed copy for equal-sized pointer groups; each returned view retains its own pointees."""

    if not groups:
        return []
    width = len(groups[0])
    if width < 1 or any(len(group) != width for group in groups):
        raise ValueError("pointer table groups must have the same positive width")
    packed = to_device(pointers([tensor for group in groups for tensor in group]), torch.int64, device)
    out = list(packed.view(len(groups), width).unbind(0))
    for table, group in zip(out, groups):
        table._tensorfold_gdn_owners = tuple(group)
    return out


def replay_table(k: Sequence[torch.Tensor], v: Sequence[torch.Tensor], g: Sequence[torch.Tensor],
                 beta: Sequence[torch.Tensor], states: Sequence[Sequence[torch.Tensor]], *,
                 in_place: bool = False) -> list[int]:
    """Host pointers for ``replay``: k, v, g, beta of each layer, then states[stream][layer]."""

    layers = len(k)
    if layers < 1 or not states or not (len(v) == len(g) == len(beta) == layers) or any(len(s) != layers for s in states):
        raise ValueError("one k, v, g, beta per layer and one state per stream and layer")
    owners = [t for layer in range(layers) for t in (k[layer], v[layer], g[layer], beta[layer])]
    owners += [t for stream in states for t in stream]
    values = pointers(owners)
    shape = states[0][0].shape
    if (len(shape) != 3 or shape[0] < 1 or shape[1] < 1 or shape[2] != 128
            or any(t.shape != shape or t.dtype != torch.float32 or t.data_ptr() % 16 for s in states for t in s)):
        raise ValueError("every state is one fp32 (Hv, Dv, 128) tensor")
    key_shape, value_shape, key_dtype = k[0].shape, v[0].shape, k[0].dtype
    if (len(key_shape) != 3 or key_shape[1] < 1 or key_shape[2] != 128 or len(value_shape) != 3
            or value_shape != (key_shape[0], shape[0], shape[1]) or shape[0] % key_shape[1]
            or key_dtype not in (torch.bfloat16, torch.float32)):
        raise ValueError("replay rows need matching (W, Hk, 128) keys and (W, Hv, Dv) bf16 values, Hv divisible by Hk")
    for layer in range(layers):
        if (k[layer].shape != key_shape or k[layer].dtype != key_dtype
                or k[layer].data_ptr() % (16 if key_dtype == torch.float32 else 8)
                or v[layer].shape != value_shape or v[layer].dtype != torch.bfloat16
                or g[layer].shape != value_shape[:2] or beta[layer].shape != value_shape[:2]
                or g[layer].dtype != torch.float32 or beta[layer].dtype != torch.float32):
            raise ValueError("all replay layers must share row geometry, key dtype and aligned keys; gates are fp32")
    if in_place:
        _check_state_writes(owners[4 * layers:], owners[:4 * layers])
    return values


def replay(table: torch.Tensor, layers: int, streams: int, rows: torch.Tensor, counts: torch.Tensor,
           k0: torch.Tensor, v0: torch.Tensor, *, in_place: bool = False) -> torch.Tensor | None:
    """Replay trusted bounded device paths; in-place states are exclusive and disjoint.

    Use ``replay_table(..., in_place=True)`` for checked in-place tables; pointer
    owners are recorded on this stream, whose waits must already cover producers.
    """

    if (not isinstance(k0, torch.Tensor) or not isinstance(v0, torch.Tensor) or not k0.is_cuda or not v0.is_cuda
            or k0.dim() != 3 or v0.dim() != 3 or k0.dtype not in (torch.bfloat16, torch.float32)
            or v0.dtype != torch.bfloat16 or k0.shape[2] != 128 or k0.shape[0] != v0.shape[0]
            or k0.device != table.device or v0.device != table.device):
        raise ValueError("replay geometry references must be same-device CUDA keys and bf16 values")
    owners = _record_pointers(table)
    if owners and len(owners) != layers * (4 + streams):
        raise ValueError("replay table needs four row tensors per layer and one state per stream and layer")
    if in_place and owners:
        _check_state_writes(owners[4 * layers:], list(owners[:4 * layers]) + [table, rows, counts])
    out = _ext().replay(table, layers, streams, rows, counts, k0.shape[1], v0.shape[1], v0.shape[2],
                        k0.dtype == torch.float32, in_place)
    return None if in_place else out


def chain(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, state: torch.Tensor,
          final: torch.Tensor) -> torch.Tensor:
    """Prefill chain from ``state`` (read only) to ``final``: chunk-invariant bits, not the verify kernel's."""

    return _ext().prefill(q, k, v, g, beta, state, final)
