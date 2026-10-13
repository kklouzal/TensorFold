"""CUDA graphs for Flash Next's concurrent decode rounds: a pure decode round's verify forward and each MTP step.

A forward is captured at startup in pieces, one key per (streams, rows) and, for the verify, the DeltaNet scratch
parity: everything that changes within a key reaches the kernels through the decoder's static tables
(``gdn_multi.RoundTables``, ``attn_multi.StepTables``) and its static buffers, written before each replay. What a
step's own host data shapes runs eagerly between the pieces, on the current step's data: each attention layer's block
selection (its grid follows the longest context), each stream's n-gram conv tail (a State pointer) and the late
n-gram reads (``forward.stage_late``). Replays run exactly the kernels and launch parameters of the static-table eager
path for the same key, so their bits are its bits.

Pieces of every key share one private memory pool (``torch.cuda.graph_pool_handle``): a round replays one key's
pieces in order and no piece's pool memory outlives it as an output (outputs are the static buffers). Keys are
captured in order of expected use (fewer streams first) while their measured memory stays within the budget
(``TENSORFOLD_MULTI_GRAPH_MIB``, counted in the startup memory plan); the rest run eagerly. A key's memory is the
drop of free memory across its capture alone (its pool blocks and driver graph memory), measured after its eager
warm-up's own allocator blocks are in place; the stream caches' room loses the larger of the kept keys' sum and the
whole capture's settled drop. Rounds with a prompt pass, image rows or the lookup decoder, and steps whose key was
not captured, run the same eager path.
"""

from __future__ import annotations

import gc
import time
from functools import partial

import torch

from tensorfold.cleanup import raise_failures
from tensorfold.cuda import pdl
# re-exported for the decoder (multi.py): the captured rounds' budget, geometry's variable and parser (one source)
from tensorfold.cuda.geometry import MULTI_GRAPH_ENV as ENV, multi_graph_bytes as budget_bytes  # noqa: F401

from . import attn_multi, gdn_multi
from .forward import _kv_begin, _kv_check, compute, ple_conv_streams, stage, stage_late
from .mtp import mtp_compute, mtp_stage

MIB = 1 << 20


def free_bytes() -> int:
    """The memory graphs take from: a unified GPU's MemAvailable (the host's), else the card's free memory. Uncapped
    by TENSORFOLD_CUDA_MEMORY_LIMIT_GB, so a key's driver graph memory shows beside its private pool's blocks."""

    from tensorfold.cuda.capacity import page_room

    room = page_room(torch)
    return int(room) if room is not None else int(torch.cuda.mem_get_info()[0])


def capturable(w) -> str | None:
    """Why this model's rounds cannot be captured (None: they can): experts that read their plan on the host."""

    if w.comm is not None:
        return "tensor-parallel ranks"
    if any(getattr(layer.moe.experts, "capturable", True) is False for layer in w.layers):
        return "this MoE reads its plan's item list on the host, which a capture rejects"
    if w.mtp is not None and getattr(w.mtp.layer.moe.experts, "capturable", True) is False:
        return "the MTP head's MoE reads its plan's item list on the host"
    return None


class Pieces:
    """One key's captured pieces, the inserts between them (``(kind, layer)``) and its output view."""

    __slots__ = ("graphs", "inserts", "out")

    def __init__(self) -> None:
        self.graphs: list[torch.cuda.CUDAGraph] = []
        self.inserts: list[tuple] = []
        self.out: torch.Tensor | None = None


class RoundGraphs:
    """A ``MultiDecoder``'s captured verify forwards and MTP steps (``capture`` at its warm-up, before any request)."""

    def __init__(self, dec, budget: int) -> None:
        self.dec, self.budget = dec, int(budget)
        self.pool = torch.cuda.graph_pool_handle()
        self.side = torch.cuda.Stream()
        self.verify: dict[tuple[bool, int, int, int], Pieces] = {}    # (PDL, streams, rows, DeltaNet parity)
        self.mtp: dict[tuple[bool, int, int], Pieces] = {}             # (PDL, streams, rows)
        self.bytes = 0                                   # the kept keys' measured memory (each >= 0): within budget
        self.settled_bytes = 0                           # the whole capture's drop of free memory, settled (raw)
        self.room_bytes = 0                              # what the stream caches' room loses: the larger of the two
        self.capture_s = 0.0
        self.measured: list[dict] = []                   # per key: raw bytes (< 0: outside noise), pieces, seconds
        self.stopped: str | None = None                  # why capture stopped before the last key (the budget)

    # -- replay ------------------------------------------------------------------------------------------------
    def verify_forward(self, segs) -> torch.Tensor | None:
        """The round's verify logits from its captured key (its tables written), or None: this round runs eagerly."""

        b = self.dec.buf
        if b.rotor_lookup or b.attn_step.vision:
            return None
        pieces = self.verify.get((pdl.ON, len(segs), segs[-1][2], self.dec.gdn.parity))
        if pieces is None:
            return None
        self._replay(pieces, segs, b)
        return pieces.out

    def mtp_step(self, segs) -> torch.Tensor | None:
        b = self.dec.mbuf
        if b.rotor_lookup or b.attn_step.vision:
            return None
        pieces = self.mtp.get((pdl.ON, len(segs), segs[-1][2]))
        if pieces is None:
            return None
        self._replay(pieces, segs, b)
        return pieces.out

    def _replay(self, pieces: Pieces, segs, b) -> None:
        w = self.dec.w
        _kv_begin(segs)                                  # the frame's codec status: checked by the caller
        for graph, insert in zip(pieces.graphs, pieces.inserts + [None]):
            graph.replay()
            if insert is None:
                continue
            kind, layer = insert
            if kind == "select":
                attn_multi.select(layer, w, b, b.attn_step)
            elif kind == "ple_conv":
                ple_conv_streams(layer, w, segs, b)
            elif b.ple_late is not None:                 # "ple_stage": the late n-gram reads (stage(late=True))
                stage_late(w, b)

    # -- capture -----------------------------------------------------------------------------------------------
    def _pieces(self, run, b) -> Pieces:
        """``run()`` captured in pieces, one ending at each ``b.split`` insert. Only a begun capture is ended, once;
        a failure inside ``run`` stays the raised error, with a failed end of its open capture attached."""

        pieces = Pieces()
        open_graph: list[torch.cuda.CUDAGraph] = []      # the piece being captured (begun, not yet ended)

        def begin() -> None:
            graph = torch.cuda.CUDAGraph()
            graph.capture_begin(pool=self.pool, capture_error_mode="thread_local")
            open_graph.append(graph)
            pieces.graphs.append(graph)

        def end() -> None:
            open_graph.pop().capture_end()               # popped first: a failed end is never retried

        def split(insert) -> None:
            end()
            pieces.inserts.append(insert)
            begin()

        b.split = split
        try:
            with torch.cuda.stream(self.side):
                begin()
                try:
                    pieces.out = run()
                except BaseException as primary:
                    errors = []
                    if open_graph:
                        try:
                            end()
                        except BaseException as error:
                            errors.append(error)
                    raise_failures(primary, errors)
                end()
        finally:
            b.split = None
        return pieces

    def capture(self) -> None:
        """Capture every key in order (streams 1, 2, ...: verify at both parities, then the MTP step) on the free
        slots while the kept keys' measured memory stays within the budget; a key past it is dropped and capture
        stops. A key is measured across its capture alone (``free_bytes`` after its eager warm-up's allocator blocks,
        then after its capture): its private pool blocks and driver graph memory, plus any outside change in the
        same interval, recorded raw (a negative figure is outside noise and counts as 0 against the budget). The
        slots are reset afterwards; the settled drop of the whole capture (after ``empty_cache``) is recorded, and
        the stream caches' room loses the larger of it and the keys' sum (``room_bytes``).

        Keys are captured with the decoder's PDL policy (``dec.pdl``, which its rounds launch with). A failure stays
        the raised error; a failed restore of the slots afterwards is attached to it."""

        dec = self.dec
        states = list(dec.free)
        parity, launch = dec.gdn.parity, pdl.ON
        started = time.perf_counter()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        before = free_bytes()
        enabled = gc.isenabled()
        gc.disable()                                     # collecting a graph's exec invalidates an active capture
        primary = None
        try:
            self._keys(states)
        except BaseException as error:
            primary = error
        finally:
            if enabled:
                gc.enable()
            pdl.ON = launch
            dec.gdn.parity = parity
            dec.buf.gdn_tables = dec.buf.attn_step = None
            if dec.mbuf is not None:
                dec.mbuf.attn_step = None
        try:
            torch.cuda.synchronize()
            for st in states:                            # the warm-up forwards wrote their caches and states
                st.reset(dec.w)
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()                     # a dropped key's exec and the warm-ups' blocks back
            self.settled_bytes = before - free_bytes()
        except BaseException as error:
            if primary is None:
                raise
            raise_failures(primary, [error])
        if primary is not None:
            raise_failures(primary, [])
        self.room_bytes = max(self.bytes, self.settled_bytes)
        self.capture_s = time.perf_counter() - started

    def _keys(self, states) -> None:
        dec = self.dec
        slots, width, launch = len(states), dec.depth + 1, dec.pdl
        pdl.ON = launch
        for n in range(1, slots + 1):
            for rows in range(n, n * width + 1):
                keys = [("verify", rows, p) for p in (0, 1)] + ([("mtp", rows, 0)] if dec.mbuf is not None else [])
                for kind, R, p in keys:
                    t0 = time.perf_counter()
                    run, b = self._warm(kind, states[:n], R, p)
                    # the warm-up's allocator blocks and module loads are taken as it is enqueued: no wait
                    free = free_bytes()
                    pieces = self._pieces(run, b)
                    torch.cuda.synchronize()
                    took = free - free_bytes()
                    if pieces.out is None:
                        raise RuntimeError(f"a captured {kind} step returned no logits")
                    if self.bytes + max(0, took) > self.budget:
                        del pieces                       # this key's graphs go; the earlier keys stay
                        self.stopped = (f"{(self.bytes + took) / MIB:.0f} MiB with the next key exceeds the "
                                        f"{self.budget // MIB} MiB budget")
                        return
                    if kind == "verify":
                        self.verify[(launch, n, R, p)] = pieces
                    else:
                        self.mtp[(launch, n, R)] = pieces
                    self.measured.append({"key": [kind, launch, n, R, p], "pieces": len(pieces.graphs),
                                          "bytes": took, "s": round(time.perf_counter() - t0, 4)})
                    self.bytes += max(0, took)

    def _warm(self, kind: str, states, R: int, parity: int):
        """One key's eager warm-up forward (compiles and configures its kernels) on its tables; returns the forward
        to capture and its buffers."""

        dec = self.dec
        w, n = dec.w, len(states)
        counts = [R // n + (s < R % n) for s in range(n)]
        if kind == "verify":
            b = dec.buf
            segs = stage(w, b, [(st, [0] * c) for st, c in zip(states, counts)])
            dec.gdn.parity = parity
            b.gdn_tables = gdn_multi.Tables(dec.gdn, segs, [[] for _ in states], static=dec.round_tables)
            b.attn_step = attn_multi.Step(w, segs, mtp=False, tables=dec.step_tables)
            run = partial(compute, w, segs, b, check=False)
        else:
            b = dec.mbuf
            at, windows, zeros = 0, [], torch.zeros_like(dec.buf.streams[:R])      # defined input streams
            for st, c in zip(states, counts):
                windows.append((st, [0] * c, zeros[at:at + c]))
                at += c
            segs = mtp_stage(w, b, windows)
            b.attn_step = attn_multi.Step(w, segs, mtp=True, tables=dec.mtp_tables)
            run = partial(mtp_compute, w, segs, b, check=False)
        b.rotor_lookup = False
        run()
        _kv_check(segs)
        return run, b

    def record(self) -> dict:
        """The startup record (capacity plan): captured keys, measured bytes, capture time, per-key measures."""

        return {"verify_keys": len(self.verify), "mtp_keys": len(self.mtp), "bytes": self.bytes,
                "settled_bytes": self.settled_bytes, "room_bytes": self.room_bytes,
                "budget_bytes": self.budget, "capture_s": round(self.capture_s, 3), "stopped": self.stopped,
                "pieces": sum(len(p.graphs) for p in (*self.verify.values(), *self.mtp.values())),
                "keys": self.measured}
