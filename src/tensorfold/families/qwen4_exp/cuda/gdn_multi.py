"""A concurrent round's DeltaNet one launch a step; each tree first folds in last round's kept rows. Bit-equal."""

from __future__ import annotations

from typing import Sequence
import weakref

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
        self.v = torch.empty((2, lin, rows, c.nv, DV), dtype=getattr(torch, getattr(c, "execution_dtype", "bfloat16")), device=dev)
        self.g = torch.empty((2, lin, rows, c.nv), dtype=torch.float32, device=dev)
        self.beta = torch.empty((2, lin, rows, c.nv), dtype=torch.float32, device=dev)
        self.lin, self.parity = lin, 0


class RoundTables:
    """A decoder's DeltaNet round tables at fixed device addresses, so a captured graph reads each round's: rows'
    streams and conv taps, chain entries and starts, pending rows (``width`` a stream: their row stride) and counts,
    and each layer's conv and state pointers (``slots`` a layer). ``Tables`` writes them through a pinned mirror,
    one copy each on the current stream before the forward; the mirror is rewritten only after that copy completes.
    Pointees are the decoder's States: a resize or drop frees their storage through the same stream's allocator."""

    def __init__(self, w, scratch: Scratch, slots: int, width: int, *, native_half=False) -> None:
        dev, pin = w.device, torch.cuda.is_available()
        self.slots, self.width, self.rows, self.lin = slots, width, slots * width, scratch.lin
        self.native_half, self.native_states, self.native_current = native_half, (), None
        self.pointer_fields = 8 if native_half else 2
        rows = self.rows
        self.at = {"sid": 0, "win": rows, "entries": 5 * rows, "starts": 8 * rows, "held": 8 * rows + slots + 1,
                   "counts": 8 * rows + slots + 1 + slots * width}
        size = self.at["counts"] + slots
        self.i32 = torch.zeros((size,), dtype=torch.int32, device=dev)
        self.h32 = torch.zeros((size,), dtype=torch.int32, pin_memory=pin)
        self.i64 = torch.zeros((self.pointer_fields * self.lin * slots,), dtype=torch.int64, device=dev)
        self.h64 = torch.zeros((self.pointer_fields * self.lin * slots,), dtype=torch.int64, pin_memory=pin)
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


    def bind_native(self, states, b):
        """Cold registration of immutable recurrent/scratch storage, before warmup.

        State.reset/resize/restore keep these allocations. Clones/replacements
        are different objects and retain the old scalar path; current parity is
        selected afresh at each frame. Strong tensor owners outlive queued work.
        """
        if not self.native_half:
            return
        if len(states) != self.slots or b.h.dtype != torch.float16:
            raise ValueError("native GDN registration requires its complete Half State envelope")
        records, spans = [], []
        def tensor(value, dtype, shape, writable):
            if (value.device != b.h.device or value.dtype != dtype or tuple(value.shape) != tuple(shape)
                    or not value.is_contiguous() or value.requires_grad or value.is_neg() or value.is_conj()):
                raise ValueError("native GDN pointee representation differs from State construction")
            begin = value.data_ptr()
            spans.append((begin, begin + value.numel()*value.element_size(), writable))
            return value
        nv = b.gout.shape[1] // 128
        nk, channels = nv // 3, (2 * (nv // 3) + nv) * 128
        if nv not in (6, 12, 24, 48):
            raise ValueError("native GDN registration head geometry")
        tensor(b.proj, torch.float16, (self.lin, b.rows, channels + nv*128 + 2*nv), False)
        tensor(b.gout, torch.float16, (b.rows, nv*128), True)
        tensor(b.gxs, torch.float32, (b.rows, nv*4), True)
        for state in states:
            conv = tensor(state.conv, torch.float16, (self.lin, 3, channels), False)
            rec = tensor(state.rec, torch.float32, (2, self.lin, nv, 128, 128), True)
            if len(state.scratch) != self.lin:
                raise ValueError("native GDN scratch layer count")
            saved, pointers = [], []
            for li, scratch in enumerate(state.scratch):
                count = scratch.k.shape[0]
                if count < self.width:
                    raise ValueError("native GDN saved window is smaller than the declared frame")
                k = tensor(scratch.k, torch.float32, (count, nk, 128), True)
                v = tensor(scratch.v, torch.float16, (count, nv, 128), True)
                g = tensor(scratch.g, torch.float32, (count, nv), True)
                beta = tensor(scratch.b, torch.float32, (count, nv), True)
                keep = getattr(scratch, "keep_rows", None)
                if keep is not None:
                    tensor(keep, torch.int32, (1,), False)
                saved.append((scratch, k, v, g, beta, keep))
                pointers.append((conv.data_ptr() + li*conv.stride(0)*conv.element_size(),
                    rec.data_ptr() + li*rec.stride(1)*rec.element_size(),
                    rec.stride(0)*rec.element_size(), k.data_ptr(), v.data_ptr(), g.data_ptr(), beta.data_ptr(),
                    0 if keep is None else keep.data_ptr()))
            records.append((state, conv, rec, state.scratch, tuple(saved), tuple(pointers)))
        # Inputs may share readonly storage; no mutable recurrent/save/output
        # byte may alias another operand. This work is cold, not per layer.
        end_any = end_write = 0
        for begin, end, writable in sorted(spans):
            if begin < end_write or writable and begin < end_any:
                raise ValueError("native GDN registered storage overlaps a mutable operand")
            end_any = max(end_any, end)
            if writable:
                end_write = max(end_write, end)
        self.native_states, self.native_buffers = tuple(records), b

    def native_frame(self, segs, b, *, allow_single=False):
        """Stage before submission; only the private exact owner opts C1 in."""
        if type(allow_single) is not bool:
            raise ValueError('native GDN single-State admission must be explicit boolean')
        if (not self.native_half or b is not self.native_buffers
                or len(segs) <= (0 if allow_single else 1) or b.prefill or b.split is not None
                or b.owner_graph_family != "uncaptured"):
            return None
        selected = []
        for state, begin, end in segs:
            record = next((row for row in self.native_states if row[0] is state), None)
            if (record is None or state.conv is not record[1] or state.rec is not record[2]
                    or state.scratch is not record[3]):
                return None
            selected.append(record)
        return NativeTables(self, segs, b, selected)


class NativeTables:
    """One owned Half frame; eight pointer planes use the existing DMA journal.

    Preparation is outside capture on the actor's ordered current stream.
    Kernel submission itself has no host wait and may be captured separately.
    Registered State storage is immutable; numeric contents/generation/parity
    remain live. The static owner and State pool retain every pointee.
    """
    def __init__(self, static, segs, b, records):
        n, rows = len(segs), segs[-1][2]
        if (n > static.slots or rows > b.rows or len({id(state) for state, _, _ in segs}) != n
                or segs[0][1] != 0 or any(end <= begin or end-begin > static.width
                    or i and begin != segs[i-1][2] for i, (_, begin, end) in enumerate(segs))):
            raise ValueError("native GDN requires disjoint contiguous bounded State windows")
        static.native_current = None  # Failed staging must never leave a stale frame submit-capable.
        if static.queued:
            static.copied.synchronize()  # Previous pinned-mirror DMA, not a device-wide fence.
        h = static.h64.numpy().reshape(8, static.lin, static.slots)
        for slot, ((state, _, _), record) in enumerate(zip(segs, records)):
            for li, (conv, rec, stride, k, v, g, beta, keep) in enumerate(record[5]):
                parity = state.cur[li]
                if type(parity) is not int or parity not in (0, 1):
                    raise ValueError("native GDN current parity must be zero or one")
                h[:, li, slot] = (conv, rec+parity*stride, rec+(1-parity)*stride, k, v, g, beta, keep)
        at = static.at["starts"]
        starts = static.h32.numpy()[at:at+n+1]
        starts[:] = [begin for _, begin, _ in segs] + [rows]
        self.static, self.buffers, self.segs, self.rows = static, b, tuple(segs), rows
        self.parities = tuple(tuple(state.cur) for state, _, _ in segs)
        self.records = tuple(records)
        self.stream = torch.cuda.current_stream(b.h.device)
        self.pointers = static.i64.view(8, static.lin, static.slots)[:, :, :n]
        self.starts = static.i32[at:at+n+1]
        static.i64.copy_(static.h64, non_blocking=True)
        self.starts.copy_(static.h32[at:at+n+1], non_blocking=True)
        if static.copied is not None:
            static.copied.record()
            static.queued = True
        static.native_current = weakref.ref(self)

    def validate_current(self, segs, b):
        """Validate one prepared exact replay frame without publishing it again."""
        if (b is not self.buffers or self.static.native_current is None
                or self.static.native_current() is not self
                or torch.cuda.current_stream(b.h.device) != self.stream
                or len(segs) != len(self.segs)
                or any(state is not original or begin != first or end != last
                    or state.conv is not record[1] or state.rec is not record[2] or state.scratch is not record[3]
                    or tuple(state.cur) != parity
                    for (state, begin, end), (original, first, last), parity, record
                    in zip(segs, self.segs, self.parities, self.records))):
            raise ValueError('native GDN replay lacks its current prepared State/parity frame')

    def operands(self, layer, p, output, sums):
        b = self.buffers
        if (self.static.native_current is None or self.static.native_current() is not self
                or torch.cuda.current_stream(b.h.device) != self.stream or not 0 <= layer < self.static.lin
                or p.data_ptr() != b.proj.data_ptr()+layer*b.proj.stride(0)*b.proj.element_size()
                or p.shape != (self.rows, b.proj.shape[2]) or output.data_ptr() != b.gout.data_ptr()
                or sums.data_ptr() != b.gxs.data_ptr()):
            raise ValueError("native GDN submission is not the current owned layer/frame")
        return self.pointers[:, layer, :], self.starts


class Tables:
    """A round's device tables, viewed in the decoder's ``RoundTables`` (``static``, written now) at addresses fixed
    for a round's stream and row counts: rows' streams and conv taps, layers' conv and state pointers, chains, pending
    rows. The trees always fold the pending rows (a zero count is the documented no-op), so a captured forward reads
    every round's."""

    def __init__(self, scratch: Scratch, segs: Sequence, pending: Sequence[Sequence[int]], static: RoundTables) -> None:
        self.scratch, self.segs, self.cur = scratch, list(segs), scratch.parity
        n, R, at, t = len(segs), segs[-1][2], static.at, static
        static.write(segs, pending)
        self.sid, self.win = t.i32[:R], t.i32[at["win"]:at["win"] + 4 * R].view(R, 4)
        most = max(a1 - a0 for _, a0, a1 in segs)
        self.plan = shared.Plan(t.i32[at["entries"]:at["entries"] + 3 * R].view(R, 3),
                                t.i32[at["starts"]:at["starts"] + n + 1], 0, most)
        self.held = t.i32[at["held"]:at["held"] + n * t.width].view(n, t.width)
        self.held_counts = t.i32[at["counts"]:at["counts"] + n]
        ptrs = t.i64.view(2, t.lin, t.slots)[:, :, :n]
        self.conv, self.state = ptrs[0], ptrs[1]


def block(g, li: int, t: Tables, p: torch.Tensor, eps: float, out: torch.Tensor, xs: torch.Tensor, nk: int) -> None:
    """Layer ``li``'s DeltaNet over the round's rows: projections ``p`` [R, W] -> ``out`` [R, NV*DV] and group sums."""

    sc, rows, cur = t.scratch, p.shape[0], t.cur
    q, k, v, gt, beta = (sc.q[:rows], sc.k[cur, li, :rows], sc.v[cur, li, :rows], sc.g[cur, li, :rows],
                         sc.beta[cur, li, :rows])
    gdn_io.front(p, t.conv[li], t.sid, t.win, g.conv, g.a_log, g.dt_bias, nk, out=(q, k, v, gt, beta))
    last = 1 - cur
    pending = [sc.k[last, li], sc.v[last, li], sc.g[last, li], sc.beta[last, li], t.held, t.held_counts]
    y = shared.tree(q, k, v, gt, beta, t.plan, table=t.state[li], pending=pending)
    gdn_io.back(y, p, g.norm, eps, out, xs)


def keep(t: Tables, kept: Sequence[int]) -> list[list[int]]:
    """This round's kept window rows a stream, for the next round's trees to fold in (which writes the other set)."""

    t.scratch.parity = 1 - t.cur
    return [list(range(a0, a0 + n)) for (_, a0, _), n in zip(t.segs, kept)]
