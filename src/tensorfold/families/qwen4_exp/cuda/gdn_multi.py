"""A concurrent round's DeltaNet one launch a step; each tree first folds in last round's kept rows. Bit-equal."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from tensorfold.cuda.kernels import gdn as shared

from . import gdn_io
from .gdn import DK, DV


class Scratch:
    """Two rounds' replay inputs a DeltaNet layer (this round's rows, last round's pending ones), and the queries."""

    def __init__(self, w, rows: int) -> None:
        c, dev = w.cfg, w.device
        lin = sum(1 for layer in w.layers if layer.linear)
        self.q = torch.empty((rows, c.nk, DK), dtype=torch.float32, device=dev)
        self.k = torch.empty((2, lin, rows, c.nk, DK), dtype=torch.float32, device=dev)
        self.v = torch.empty((2, lin, rows, c.nv, DV), dtype=torch.bfloat16, device=dev)
        self.g = torch.empty((2, lin, rows, c.nv), dtype=torch.float32, device=dev)
        self.beta = torch.empty((2, lin, rows, c.nv), dtype=torch.float32, device=dev)
        self.lin, self.parity = lin, 0


class RoundTables:
    """A decoder's DeltaNet round tables at fixed device addresses, so a captured graph reads each round's: rows'
    streams and conv taps, chain entries and starts, pending rows (``width`` a stream: their row stride) and counts,
    and each layer's conv and state pointers (``slots`` a layer). ``Tables`` writes them through a pinned mirror,
    one copy each on the current stream before the forward; the mirror is rewritten only after that copy completes.
    Pointees are the decoder's States: a resize or drop frees their storage through the same stream's allocator."""

    def __init__(self, w, scratch: Scratch, slots: int, width: int) -> None:
        dev, pin = w.device, torch.cuda.is_available()
        self.slots, self.width, self.rows, self.lin = slots, width, slots * width, scratch.lin
        rows = self.rows
        self.at = {"sid": 0, "win": rows, "entries": 5 * rows, "starts": 8 * rows, "held": 8 * rows + slots + 1,
                   "counts": 8 * rows + slots + 1 + slots * width}
        size = self.at["counts"] + slots
        self.i32 = torch.zeros((size,), dtype=torch.int32, device=dev)
        self.h32 = torch.zeros((size,), dtype=torch.int32, pin_memory=pin)
        self.i64 = torch.zeros((2 * self.lin * slots,), dtype=torch.int64, device=dev)
        self.h64 = torch.zeros((2 * self.lin * slots,), dtype=torch.int64, pin_memory=pin)
        self.copied = torch.cuda.Event() if pin else None
        self.queued = False

    def write(self, segs: Sequence, pending: Sequence[Sequence[int]]) -> None:
        """This round's tables into the mirror, then one copy each to the device."""

        n, R, at, slots = len(segs), segs[-1][2], self.at, self.slots
        if n > slots or R > self.rows or any(len(rows_) > self.width for rows_ in pending):
            raise ValueError(f"a round of {n} streams and {R} rows exceeds the static tables' {slots} x {self.width}")
        if self.queued:
            self.copied.synchronize()                    # the last copy out of the mirror is done
        h = self.h32.numpy()
        sid, win = h[:R], h[at["win"]:at["win"] + 4 * R].reshape(R, 4)
        entries = h[at["entries"]:at["entries"] + 3 * R].reshape(R, 3)
        held = h[at["held"]:at["held"] + n * self.width].reshape(n, self.width)
        taps = np.arange(4)[None, :]
        entries[:, 2] = -1                               # chains: no slots (gdn.plan_host's entries for chains)
        entries[:, 1] = -2
        held[:] = 0
        for s, ((st, a0, a1), rows_) in enumerate(zip(segs, pending)):
            j = np.arange(a1 - a0)[:, None] + taps
            win[a0:a1] = np.where(j < 3, j, a0 + j)            # < 3: a conv state row, else window row tap - 3
            sid[a0:a1] = s
            entries[a0:a1, 0] = np.arange(a0, a1)
            entries[a0, 1] = -1
            held[s, :len(rows_)] = rows_
        h[at["starts"]:at["starts"] + n + 1] = [a0 for _, a0, _ in segs] + [R]
        h[at["counts"]:at["counts"] + n] = [len(rows_) for rows_ in pending]
        p = self.h64.numpy().reshape(2, self.lin, slots)
        layers = np.arange(self.lin, dtype=np.int64)
        for s, (st, _, _) in enumerate(segs):            # st.conv[li] and st.rec[cur[li], li] by their strides
            conv, rec = st.conv, st.rec
            p[0, :, s] = conv.data_ptr() + layers * (conv.stride(0) * conv.element_size())
            p[1, :, s] = (rec.data_ptr() + np.asarray(st.cur, dtype=np.int64) * (rec.stride(0) * rec.element_size())
                          + layers * (rec.stride(1) * rec.element_size()))
        self.i32.copy_(self.h32, non_blocking=True)
        self.i64.copy_(self.h64, non_blocking=True)
        if self.copied is not None:
            self.copied.record()
            self.queued = True


class Tables:
    """A round's device tables: rows' streams and conv taps, layers' conv and state pointers, chains, pending rows.

    ``static`` (a decoder's ``RoundTables``; benchmark switch "graphs"): written there, viewed at addresses fixed for
    a round's stream and row counts, and the trees always fold the pending rows (a zero count is the documented
    no-op), so a captured forward reads every round's. Else (A/B baseline) a fresh upload a round."""

    def __init__(self, w, scratch: Scratch, segs: Sequence, pending: Sequence[Sequence[int]],
                 static: RoundTables | None = None) -> None:
        self.scratch, self.segs, self.cur = scratch, list(segs), scratch.parity
        if static is not None:
            n, R, at, t = len(segs), segs[-1][2], static.at, static
            static.write(segs, pending)
            self.sid, self.win = t.i32[:R], t.i32[at["win"]:at["win"] + 4 * R].view(R, 4)
            most = max(a1 - a0 for _, a0, a1 in segs)
            self.plan = shared.Plan(t.i32[at["entries"]:at["entries"] + 3 * R].view(R, 3),
                                    t.i32[at["starts"]:at["starts"] + n + 1], 0, most)
            self.held = t.i32[at["held"]:at["held"] + n * t.width].view(n, t.width)
            self.held_counts = t.i32[at["counts"]:at["counts"] + n]
            self.folds = True
            ptrs = t.i64.view(2, t.lin, t.slots)[:, :, :n]
            self.conv, self.state = ptrs[0], ptrs[1]
            return
        n, rows = len(segs), segs[-1][2]                 # A/B baseline: remove after selection (to the end)
        lin = scratch.lin
        taps = np.arange(4)[None, :]
        win = np.empty((rows, 4), dtype=np.int32)
        sid = np.empty((rows,), dtype=np.int32)
        for s, (_, a0, a1) in enumerate(segs):
            j = np.arange(a1 - a0)[:, None] + taps
            win[a0:a1] = np.where(j < 3, j, a0 + j)            # < 3: a conv state row, else window row tap - 3
            sid[a0:a1] = s
        entries, starts, slots, most = shared.plan_host([list(range(-1, a1 - a0 - 1)) for _, a0, a1 in segs])
        width = max([len(rows_) for rows_ in pending] + [1])
        held = np.zeros((n, width), dtype=np.int32)             # each stream's last-round kept rows, not yet folded
        for s, rows_ in enumerate(pending):
            held[s, :len(rows_)] = rows_
        counts = np.asarray([len(rows_) for rows_ in pending], dtype=np.int32)
        ints = np.concatenate([sid, win.ravel(), np.asarray(entries + starts, dtype=np.int32), held.ravel(), counts])
        ptrs = np.empty((2, lin, n), dtype=np.int64)
        for s, (st, _, _) in enumerate(segs):
            for li in range(lin):
                ptrs[0, li, s] = st.conv[li].data_ptr()
                ptrs[1, li, s] = st.rec[st.cur[li], li].data_ptr()
        dev = w.device
        i32 = shared.to_device(ints.tolist(), torch.int32, dev)
        i64 = shared.to_device(ptrs.ravel().tolist(), torch.int64, dev)
        self.sid, self.win = i32[:rows], i32[rows:5 * rows].view(rows, 4)
        at = 8 * rows + n + 1
        self.plan = shared.Plan(i32[5 * rows:8 * rows].view(rows, 3), i32[8 * rows:at], slots, most)
        self.held, self.held_counts = i32[at:at + n * width].view(n, width), i32[at + n * width:]
        self.folds = bool(counts.any())
        self.conv, self.state = i64[:lin * n].view(lin, n), i64[lin * n:].view(lin, n)


def block(g, li: int, t: Tables, p: torch.Tensor, eps: float, out: torch.Tensor, xs: torch.Tensor, nk: int) -> None:
    """Layer ``li``'s DeltaNet over the round's rows: projections ``p`` [R, W] -> ``out`` [R, NV*DV] and group sums."""

    sc, rows, cur = t.scratch, p.shape[0], t.cur
    q, k, v, gt, beta = (sc.q[:rows], sc.k[cur, li, :rows], sc.v[cur, li, :rows], sc.g[cur, li, :rows],
                         sc.beta[cur, li, :rows])
    gdn_io.front(p, t.conv[li], t.sid, t.win, g.conv, g.a_log, g.dt_bias, nk, out=(q, k, v, gt, beta))
    last = 1 - cur
    pending = ([sc.k[last, li], sc.v[last, li], sc.g[last, li], sc.beta[last, li], t.held, t.held_counts]
               if t.folds else None)
    y = shared.tree(q, k, v, gt, beta, t.plan, table=t.state[li], pending=pending)
    gdn_io.back(y, p, g.norm, eps, out, xs)


def keep(t: Tables, kept: Sequence[int]) -> list[list[int]]:
    """This round's kept window rows a stream, for the next round's trees to fold in (which writes the other set)."""

    t.scratch.parity = 1 - t.cur
    return [list(range(a0, a0 + n)) for (_, a0, _), n in zip(t.segs, kept)]
