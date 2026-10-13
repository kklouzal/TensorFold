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
(``TENSORFOLD_MULTI_GRAPH_MIB``, counted in the startup memory plan and taken from the stream caches' room); the
rest run eagerly. Rounds with a prompt pass, image rows or the lookup decoder, and steps whose key was not
captured, run the same eager path.
"""

from __future__ import annotations

import gc
import time

import torch

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
    """One key's captured pieces, the inserts between them (``(kind, layer, mtp)``) and its output view."""

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
        self.verify: dict[tuple[int, int, int], Pieces] = {}
        self.mtp: dict[tuple[int, int], Pieces] = {}
        self.bytes = 0
        self.capture_s = 0.0
        self.measured: list[dict] = []                   # per key: bytes, pieces, capture seconds (startup record)
        self.stopped: str | None = None                  # why capture stopped before the last key (the budget)

    # -- replay ------------------------------------------------------------------------------------------------
    def verify_forward(self, segs) -> torch.Tensor | None:
        """The round's verify logits from its captured key (its tables written), or None: this round runs eagerly."""

        b = self.dec.buf
        step = b.attn_step
        if b.rotor_lookup or step.vision or not step.fixed:
            return None
        pieces = self.verify.get((len(segs), segs[-1][2], self.dec.gdn.parity))
        if pieces is None:
            return None
        self._replay(pieces, segs, b)
        return pieces.out

    def mtp_step(self, segs) -> torch.Tensor | None:
        b = self.dec.mbuf
        step = b.attn_step
        if b.rotor_lookup or step.vision or not step.fixed:
            return None
        pieces = self.mtp.get((len(segs), segs[-1][2]))
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
            kind, layer, mtp = insert
            if kind == "select":
                attn_multi.select(layer, w, b, b.attn_step, mtp)
            elif kind == "ple_conv":
                ple_conv_streams(layer, w, segs, b)
            elif b.ple_late is not None:                 # "ple_stage": the late n-gram reads (stage(late=True))
                stage_late(w, b)

    # -- capture -----------------------------------------------------------------------------------------------
    def _pieces(self, run, b) -> Pieces:
        """``run()`` captured in pieces, one ending at each ``b.split`` insert."""

        pieces = Pieces()

        def begin() -> None:
            graph = torch.cuda.CUDAGraph()
            pieces.graphs.append(graph)
            graph.capture_begin(pool=self.pool, capture_error_mode="thread_local")

        def split(insert) -> None:
            pieces.graphs[-1].capture_end()
            pieces.inserts.append(insert)
            begin()

        b.split = split
        try:
            with torch.cuda.stream(self.side):
                begin()
                try:
                    pieces.out = run()
                finally:
                    pieces.graphs[-1].capture_end()
        finally:
            b.split = None
        return pieces

    def capture(self) -> None:
        """Capture every key in order (streams 1, 2, ...: verify at both parities, then the MTP step) on the free
        slots while the memory they take (``free_bytes``: the private pool's blocks and the driver's graph memory)
        stays within the budget; a key past it is dropped and capture stops. The slots are reset afterwards."""

        dec = self.dec
        w, slots, width = dec.w, len(dec.free), dec.depth + 1
        states = list(dec.free)
        parity = dec.gdn.parity
        started = time.perf_counter()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        before = free_bytes()
        enabled = gc.isenabled()
        gc.disable()                                     # collecting a graph's exec invalidates an active capture
        try:
            for n in range(1, slots + 1):
                for rows in range(n, n * width + 1):
                    keys = [("verify", rows, p) for p in (0, 1)] + ([("mtp", rows, 0)] if dec.mbuf is not None else [])
                    for kind, R, p in keys:
                        t0 = time.perf_counter()
                        pieces = self._key(kind, states[:n], R, p)
                        torch.cuda.synchronize()
                        used = before - free_bytes()
                        if used > self.budget:
                            del pieces                   # this key's graphs go; the earlier keys stay
                            self.stopped = f"{used / MIB:.0f} MiB with the next key exceeds the {self.budget // MIB} MiB budget"
                            return
                        (self.verify if kind == "verify" else self.mtp)[(n, R, p) if kind == "verify" else (n, R)] = \
                            pieces
                        self.measured.append({"key": [kind, n, R, p], "pieces": len(pieces.graphs),
                                              "bytes": used - self.bytes, "s": round(time.perf_counter() - t0, 4)})
                        self.bytes = used
        finally:
            if enabled:
                gc.enable()
            dec.gdn.parity = parity
            dec.buf.gdn_tables = dec.buf.attn_step = None
            if dec.mbuf is not None:
                dec.mbuf.attn_step = None
            torch.cuda.synchronize()
            for st in states:                            # the warm-up forwards wrote their caches and states
                st.reset(w)
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()                     # a dropped key's pool blocks back before the final measure
            self.bytes = max(0, before - free_bytes())
            self.capture_s = time.perf_counter() - started

    def _key(self, kind: str, states, R: int, parity: int) -> Pieces:
        """One key: an eager warm-up forward (compiles and configures its kernels), then its capture."""

        dec = self.dec
        w, n = dec.w, len(states)
        counts = [R // n + (s < R % n) for s in range(n)]
        if kind == "verify":
            b = dec.buf
            segs = stage(w, b, [(st, [0] * c) for st, c in zip(states, counts)])
            dec.gdn.parity = parity
            b.gdn_tables = gdn_multi.Tables(w, dec.gdn, segs, [[] for _ in states], static=dec.round_tables)
            b.attn_step = attn_multi.Step(w, segs, mtp=False, tables=dec.step_tables)
            run = lambda: compute(w, segs, b, check=False)                   # noqa: E731
        else:
            b = dec.mbuf
            at, windows, zeros = 0, [], torch.zeros_like(dec.buf.streams[:R])      # defined input streams
            for st, c in zip(states, counts):
                windows.append((st, [0] * c, zeros[at:at + c]))
                at += c
            segs = mtp_stage(w, b, windows)
            b.attn_step = attn_multi.Step(w, segs, mtp=True, tables=dec.mtp_tables)
            run = lambda: mtp_compute(w, segs, b, check=False)               # noqa: E731
        b.rotor_lookup = False
        run()
        _kv_check(segs)
        pieces = self._pieces(run, b)
        if pieces.out is None:
            raise RuntimeError(f"a captured {kind} step returned no logits")
        return pieces

    def record(self) -> dict:
        """The startup record (capacity plan): captured keys, measured bytes, capture time, per-key measures."""

        return {"verify_keys": len(self.verify), "mtp_keys": len(self.mtp), "bytes": self.bytes,
                "budget_bytes": self.budget, "capture_s": round(self.capture_s, 3), "stopped": self.stopped,
                "pieces": sum(len(p.graphs) for p in (*self.verify.values(), *self.mtp.values())),
                "keys": self.measured}
